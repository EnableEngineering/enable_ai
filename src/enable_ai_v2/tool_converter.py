"""
Convert OpenAPI spec to LLM tool definitions.

Transforms OpenAPI endpoint definitions into tool schemas for
OpenAI and Anthropic native function calling.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .types import Endpoint


def load_openapi_spec(spec_path: str | Path) -> dict:
    """Load OpenAPI spec from file (JSON or YAML)."""
    path = Path(spec_path)
    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML required for YAML specs: pip install pyyaml")
    else:
        return json.loads(content)


def extract_endpoints(spec: dict) -> list[Endpoint]:
    """Extract all endpoints from OpenAPI spec."""
    endpoints = []
    paths = spec.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue

            endpoint = Endpoint(
                path=path,
                method=method.upper(),
                operation_id=details.get("operationId", ""),
                summary=details.get("summary", ""),
                description=details.get("description", ""),
                parameters=_resolve_parameters(details.get("parameters", []), spec),
                request_body=_resolve_request_body(details.get("requestBody"), spec),
                responses=details.get("responses", {}),
            )
            endpoints.append(endpoint)

    return endpoints


def _resolve_ref(ref: str, spec: dict) -> dict:
    """Resolve a $ref pointer to its definition."""
    if not ref.startswith("#/"):
        return {}

    parts = ref[2:].split("/")
    result = spec
    for part in parts:
        result = result.get(part, {})
    return result


def _resolve_parameters(params: list, spec: dict) -> list[dict]:
    """Resolve parameter references and return full definitions."""
    resolved = []
    for param in params:
        if "$ref" in param:
            param = _resolve_ref(param["$ref"], spec)
        resolved.append(param)
    return resolved


def _resolve_request_body(body: Optional[dict], spec: dict) -> Optional[dict]:
    """Resolve request body schema references."""
    if not body:
        return None

    if "$ref" in body:
        body = _resolve_ref(body["$ref"], spec)

    content = body.get("content", {})
    json_content = content.get("application/json", {})
    schema = json_content.get("schema", {})

    if "$ref" in schema:
        schema = _resolve_ref(schema["$ref"], spec)

    return {
        "required": body.get("required", False),
        "schema": _resolve_schema(schema, spec),
    }


def _resolve_schema(schema: dict, spec: dict, depth: int = 0) -> dict:
    """Recursively resolve schema references."""
    if depth > 10:  # Prevent infinite recursion
        return schema

    if "$ref" in schema:
        schema = _resolve_ref(schema["$ref"], spec)

    result = dict(schema)

    # Resolve nested properties
    if "properties" in result:
        result["properties"] = {
            k: _resolve_schema(v, spec, depth + 1)
            for k, v in result["properties"].items()
        }

    # Resolve array items
    if "items" in result:
        result["items"] = _resolve_schema(result["items"], spec, depth + 1)

    # Resolve allOf, anyOf, oneOf
    for key in ("allOf", "anyOf", "oneOf"):
        if key in result:
            result[key] = [
                _resolve_schema(s, spec, depth + 1) for s in result[key]
            ]

    return result


def endpoint_to_tool(endpoint: Endpoint, spec: dict) -> dict:
    """Convert a single endpoint to an LLM tool definition."""
    properties = {}
    required = []

    # Add path parameters
    for param in endpoint.parameters:
        if param.get("in") == "path":
            prop = _param_to_property(param)
            properties[param["name"]] = prop
            required.append(param["name"])

    # Add query parameters
    for param in endpoint.parameters:
        if param.get("in") == "query":
            prop = _param_to_property(param)
            properties[param["name"]] = prop
            if param.get("required"):
                required.append(param["name"])

    # Add request body properties
    if endpoint.request_body and endpoint.request_body.get("schema"):
        body_schema = endpoint.request_body["schema"]
        if body_schema.get("type") == "object":
            for name, prop in body_schema.get("properties", {}).items():
                properties[name] = _clean_property(prop)
            if endpoint.request_body.get("required"):
                required.extend(body_schema.get("required", []))

    # Build description
    description = endpoint.summary or endpoint.description or f"{endpoint.method} {endpoint.path}"
    if endpoint.description and endpoint.summary:
        description = f"{endpoint.summary}. {endpoint.description}"

    # Truncate overly long descriptions
    if len(description) > 1000:
        description = description[:997] + "..."

    tool = {
        "name": endpoint.tool_name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    return tool


def _param_to_property(param: dict) -> dict:
    """Convert OpenAPI parameter to JSON Schema property."""
    schema = param.get("schema", {})
    prop = _clean_property(schema)

    # Add description from parameter
    if param.get("description"):
        prop["description"] = param["description"]

    # Add enum values if present
    if "enum" in schema:
        prop["enum"] = schema["enum"]

    return prop


def _clean_property(prop: dict) -> dict:
    """Clean a property for LLM tool schema."""
    result = {}

    # Map type
    prop_type = prop.get("type", "string")
    result["type"] = prop_type

    # Add description
    if prop.get("description"):
        result["description"] = prop["description"]

    # Handle arrays
    if prop_type == "array" and "items" in prop:
        result["items"] = _clean_property(prop["items"])

    # Handle objects
    if prop_type == "object" and "properties" in prop:
        result["properties"] = {
            k: _clean_property(v) for k, v in prop["properties"].items()
        }
        if "required" in prop:
            result["required"] = prop["required"]

    # Copy enum
    if "enum" in prop:
        result["enum"] = prop["enum"]

    # Copy format for validation hints
    if "format" in prop:
        result["format"] = prop["format"]

    return result


def convert_spec_to_tools(spec: dict | str | Path) -> tuple[list[dict], list[Endpoint]]:
    """
    Convert full OpenAPI spec to LLM tools.

    Args:
        spec: OpenAPI spec dict, or path to spec file

    Returns:
        Tuple of (tools list for LLM, endpoints list for execution)
    """
    if isinstance(spec, (str, Path)):
        spec = load_openapi_spec(spec)

    endpoints = extract_endpoints(spec)
    tools = [endpoint_to_tool(ep, spec) for ep in endpoints]

    return tools, endpoints


def get_endpoint_by_tool_name(
    tool_name: str,
    endpoints: list[Endpoint]
) -> Optional[Endpoint]:
    """Find endpoint matching a tool name."""
    for endpoint in endpoints:
        if endpoint.tool_name == tool_name:
            return endpoint
    return None
