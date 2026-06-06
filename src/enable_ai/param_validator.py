"""
Rule-based validation and repair of parsed queries against schema.

No extra LLM calls — uses SchemaIntrospector and resource_hints already in the schema.
"""

from typing import Any, Dict, List, Optional, Tuple

from .schema_introspector import SchemaIntrospector
from .user_context_resolver import resolve_user_context_in_parsed
from .hint_utils import query_implies_user_scope
from .utils import setup_logger

logger = setup_logger("enable_ai.param_validator")


def _resolve_resource(parsed: Dict[str, Any], schema: Dict[str, Any]) -> str:
    """Map parsed resource to a schema resource name via hints synonyms."""
    resource = (parsed.get("resource") or "").lower()
    resources = (schema or {}).get("resources") or {}
    hints = (schema or {}).get("resource_hints") or {}

    if resource in resources:
        return resource

    parsed_norm = resource.replace("-", "_").replace(" ", "_")
    for name in resources:
        if name.lower().replace("-", "_") == parsed_norm:
            return name

    for hint_name, hint_data in hints.items():
        if not isinstance(hint_data, dict):
            continue
        if hint_name.lower().replace("-", "_") == parsed_norm:
            return hint_name
        syns = hint_data.get("__resource_synonyms__") or []
        if not isinstance(syns, list):
            syns = [syns]
        for s in syns:
            if str(s).lower().replace("-", "_").replace(" ", "_") == parsed_norm:
                return hint_name

    return parsed.get("resource", "")


def validate_parsed(
    parsed: Dict[str, Any],
    schema: Dict[str, Any],
    query: str = "",
    user_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    """
    Validate and lightly repair parsed query.

    Returns:
        (repaired_parsed, warnings, clarification_message or None)
    """
    if not isinstance(parsed, dict):
        return parsed, [], None

    result = dict(parsed)
    warnings: List[str] = []
    clarification: Optional[str] = None
    schema = schema or {}

    # Resolve resource name
    if result.get("resource"):
        resolved = _resolve_resource(result, schema)
        if resolved and resolved != result.get("resource"):
            result["resource"] = resolved

    # Coerce limit
    if result.get("limit") is not None:
        try:
            result["limit"] = int(result["limit"])
        except (TypeError, ValueError):
            warnings.append(f"Invalid limit {result['limit']!r}; removed")
            result.pop("limit", None)

    # Resolve __current_user_id__ before schema validation (placeholders are not real values)
    hints = schema.get("resource_hints") or {}
    if user_context:
        result = resolve_user_context_in_parsed(
            result, user_context, query, resource_hints=hints,
        )

    # Validate filter values via introspector
    resource = result.get("resource", "")
    if resource and result.get("filters"):
        intro = SchemaIntrospector(schema)
        repaired_filters = {}
        for field, fval in result["filters"].items():
            value = fval.get("value") if isinstance(fval, dict) else fval
            vr = intro.validate_filter_value(resource, field, value)
            if vr.corrected_value is not None and vr.corrected_value != value:
                if isinstance(fval, dict):
                    repaired_filters[field] = {**fval, "value": vr.corrected_value}
                else:
                    repaired_filters[field] = vr.corrected_value
            else:
                repaired_filters[field] = fval
            warnings.extend(vr.warnings or [])
        result["filters"] = repaired_filters

    # Pronouns without user context → ask for clarification
    if not user_context and query_implies_user_scope(query):
        if result.get("question_type") != "needs_clarification":
            clarification = (
                "This query refers to you ('me' / 'my') but no user context was provided. "
                "Pass user_context with user_id, or rephrase with a specific identifier."
            )
            result["question_type"] = "needs_clarification"

    if result.get("question_type") == "needs_clarification" and not clarification:
        clarification = (
            "I need more information to complete this query. "
            "Please be more specific about the resource, filters, or user context."
        )

    return result, warnings, clarification
