"""
Validation for tool calls before execution.

Three-stage validation:
1. Schema validation - required params, types
2. Semantic validation - valid values, ranges
3. Clarification - ambiguous inputs
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .types import ConfidenceScore, Endpoint, ToolCall, ValidationResult
from .config import IntentPhrases, DEFAULT_INTENT_PHRASES
from .tool_filter import (
    extract_business_codes,
    numeric_suffix_from_code,
    has_business_code,
    is_read_only_query,
    is_mutation_tool,
    EMBEDDED_LOOKUP_BAD_MARKERS,
    TRANSITION_TOOL_MARKERS,
)
from .query_intent import mentions_service_orders, query_wants_embedded_fields, is_scheduling_status_query


class Validator:
    """Validate tool calls before execution."""

    def __init__(
        self,
        endpoints: list[Endpoint],
        intent_phrases: IntentPhrases | None = None,
    ):
        self._endpoints = {ep.tool_name: ep for ep in endpoints}
        self._intent_phrases = intent_phrases or DEFAULT_INTENT_PHRASES

    def validate(
        self,
        tool_call: ToolCall,
        query: str,
        prior_tool_calls: Optional[list[ToolCall]] = None,
    ) -> ValidationResult:
        """
        Validate a tool call.

        Returns ValidationResult with errors, warnings, and clarification needs.
        """
        endpoint = self._endpoints.get(tool_call.name)
        if not endpoint:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown tool: {tool_call.name}"],
            )

        errors = []
        warnings = []

        # Stage 1: Schema validation
        schema_errors = self._validate_schema(tool_call, endpoint)
        errors.extend(schema_errors)

        # Stage 2: Semantic validation
        semantic_errors, semantic_warnings = self._validate_semantic(
            tool_call, endpoint
        )
        errors.extend(semantic_errors)
        warnings.extend(semantic_warnings)

        # Stage 3: Clarification check
        needs_clarification, clarification_question = self._check_clarification(
            tool_call, endpoint, query
        )

        # Business code used as path id (SO-169 → id=169)
        code_clarify, code_question = self._check_business_code_as_path_id(
            tool_call, endpoint, query
        )
        if code_clarify:
            needs_clarification = True
            clarification_question = code_question
            warnings.append(
                "Business reference code should use list/search, not path id"
            )

        # Wrong resource family (statuses/activities for SO queries)
        wrong_resource, wrong_msg = self._check_wrong_resource_family(tool_call, query)
        if wrong_resource:
            needs_clarification = True
            clarification_question = wrong_msg
            warnings.append("Wrong resource family for query")

        # Invalid scoping filters (technician=0, null)
        scope_errors = self._check_scoping_filters(tool_call)
        if scope_errors:
            errors.extend(scope_errors)
            needs_clarification = True
            if not clarification_question:
                clarification_question = scope_errors[0]

        # Mutation tools on read-only queries (Q53 partial_update)
        mutation_block, mutation_msg = self._check_mutation_on_read_query(tool_call, query)
        if mutation_block:
            needs_clarification = True
            clarification_question = mutation_msg
            warnings.append("Mutation tool on read query")

        # Wrong embedded lookup (Q01 by_service_id)
        bad_lookup, lookup_msg = self._check_wrong_embedded_lookup(tool_call, query)
        if bad_lookup:
            needs_clarification = True
            clarification_question = lookup_msg
            warnings.append("Wrong embedded lookup endpoint")

        transition_block, transition_msg = self._check_transition_tool_for_scheduling(
            tool_call, query
        )
        if transition_block:
            needs_clarification = True
            clarification_question = transition_msg
            warnings.append("Transition tool on scheduling query")

        if prior_tool_calls:
            dup, dup_msg = self._check_duplicate_list_call(tool_call, prior_tool_calls)
            if dup:
                needs_clarification = True
                if not clarification_question:
                    clarification_question = dup_msg
                warnings.append("Duplicate list call")

        # Calculate confidence
        confidence = self._calculate_confidence(
            tool_call, endpoint, errors, warnings
        )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
        )

    def normalize_arguments(
        self,
        tool_call: ToolCall,
        status_synonyms: Optional[dict[str, str]] = None,
    ) -> ToolCall:
        """Map natural-language / fuzzy values to OpenAPI enum values before validation."""
        endpoint = self._endpoints.get(tool_call.name)
        if not endpoint:
            return tool_call

        synonyms = status_synonyms or {}
        new_args = dict(tool_call.arguments)

        for param in endpoint.parameters:
            name = param["name"]
            if name not in new_args:
                continue
            schema = param.get("schema", {})
            if "enum" not in schema:
                continue
            resolved = self._resolve_enum_value(
                new_args[name], schema["enum"], synonyms
            )
            if resolved is not None:
                new_args[name] = resolved

        return ToolCall(id=tool_call.id, name=tool_call.name, arguments=new_args)

    def _resolve_enum_value(
        self,
        value: Any,
        enum_values: list,
        status_synonyms: dict[str, str],
    ) -> Any:
        """Resolve user/LLM value to a valid enum entry when possible."""
        if value in enum_values:
            return value

        str_val = str(value).strip()
        lower = str_val.lower()

        if lower in status_synonyms:
            mapped = status_synonyms[lower]
            if mapped in enum_values:
                return mapped

        for candidate in enum_values:
            cand_str = str(candidate)
            if cand_str.lower() == lower:
                return candidate

        normalized_input = re.sub(r"[\s_\-]+", " ", lower)
        for candidate in enum_values:
            cand_norm = re.sub(r"[\s_\-]+", " ", str(candidate).lower())
            if normalized_input == cand_norm:
                return candidate
            if normalized_input in cand_norm or cand_norm in normalized_input:
                return candidate

        return value

    def _validate_schema(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
    ) -> list[str]:
        """Validate required parameters and types."""
        errors = []

        # Get required path parameters
        path_params = set(re.findall(r"\{(\w+)\}", endpoint.path))

        # Check required path params
        for param in path_params:
            if param not in tool_call.arguments:
                errors.append(f"Missing required path parameter: {param}")

        # Check required query/body params from endpoint definition
        for param in endpoint.parameters:
            if param.get("required") and param["name"] not in tool_call.arguments:
                errors.append(f"Missing required parameter: {param['name']}")

        # Basic type checking
        for param in endpoint.parameters:
            name = param["name"]
            if name in tool_call.arguments:
                value = tool_call.arguments[name]
                schema = param.get("schema", {})

                type_error = self._check_type(name, value, schema)
                if type_error:
                    errors.append(type_error)

        return errors

    def _check_type(
        self,
        name: str,
        value: Any,
        schema: dict,
    ) -> Optional[str]:
        """Check value matches schema type."""
        expected_type = schema.get("type", "string")

        if expected_type == "integer":
            if not isinstance(value, int):
                try:
                    int(value)
                except (ValueError, TypeError):
                    return f"{name} must be an integer, got: {value}"

        elif expected_type == "number":
            if not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    return f"{name} must be a number, got: {value}"

        elif expected_type == "boolean":
            if not isinstance(value, bool):
                if str(value).lower() not in ("true", "false", "1", "0"):
                    return f"{name} must be a boolean, got: {value}"

        elif expected_type == "array":
            if not isinstance(value, list):
                return f"{name} must be an array, got: {type(value).__name__}"

        return None

    def _validate_semantic(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
    ) -> tuple[list[str], list[str]]:
        """Validate semantic correctness of values."""
        errors = []
        warnings = []

        for param in endpoint.parameters:
            name = param["name"]
            if name not in tool_call.arguments:
                continue

            value = tool_call.arguments[name]
            schema = param.get("schema", {})

            # Check enum values
            if "enum" in schema:
                if value not in schema["enum"]:
                    errors.append(
                        f"{name} must be one of: {schema['enum']}, got: {value}"
                    )

            # Check numeric ranges
            if "minimum" in schema:
                try:
                    if float(value) < schema["minimum"]:
                        errors.append(
                            f"{name} must be >= {schema['minimum']}, got: {value}"
                        )
                except (ValueError, TypeError):
                    pass

            if "maximum" in schema:
                try:
                    if float(value) > schema["maximum"]:
                        errors.append(
                            f"{name} must be <= {schema['maximum']}, got: {value}"
                        )
                except (ValueError, TypeError):
                    pass

            # Check string patterns
            if "pattern" in schema and isinstance(value, str):
                if not re.match(schema["pattern"], value):
                    warnings.append(
                        f"{name} doesn't match expected pattern: {schema['pattern']}"
                    )

            # Check date formats
            if schema.get("format") == "date":
                if not self._is_valid_date(value):
                    warnings.append(
                        f"{name} doesn't look like a valid date: {value}"
                    )

        return errors, warnings

    def _is_valid_date(self, value: str) -> bool:
        """Check if value looks like a date."""
        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",  # ISO format
            r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # US format
            r"^\d{1,2}-\d{1,2}-\d{2,4}$",  # Alternative
        ]
        return any(re.match(p, str(value)) for p in date_patterns)

    def _check_clarification(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
        query: str,
    ) -> tuple[bool, Optional[str]]:
        """Check if clarification is needed."""
        # Check for ambiguous values
        for param in endpoint.parameters:
            name = param["name"]
            if name not in tool_call.arguments:
                continue

            value = tool_call.arguments[name]
            schema = param.get("schema", {})

            # If there's an enum and value doesn't exactly match
            if "enum" in schema and isinstance(value, str):
                matches = [e for e in schema["enum"] if value.lower() in e.lower()]
                if len(matches) > 1:
                    return True, (
                        f"Did you mean {matches[0]} or {matches[1]}? "
                        f"Please clarify which {name} you want."
                    )

        return False, None

    def _check_business_code_as_path_id(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
        query: str,
    ) -> tuple[bool, Optional[str]]:
        """Detect SO-### style codes incorrectly passed as path id."""
        if "id" not in tool_call.arguments:
            return False, None

        path_has_id = "id" in endpoint.path or "{id}" in endpoint.path
        if not path_has_id:
            return False, None

        id_val = str(tool_call.arguments["id"])
        for code in extract_business_codes(query):
            suffix = numeric_suffix_from_code(code)
            if suffix and id_val == suffix:
                return True, (
                    f"'{code}' is a business reference code, not the database id. "
                    f"Use a list/search tool with search={code} instead of id={id_val}."
                )
        return False, None

    def _check_wrong_resource_family(
        self,
        tool_call: ToolCall,
        query: str,
    ) -> tuple[bool, Optional[str]]:
        """Reject statuses/activities endpoints for service-order detail queries."""
        name = tool_call.name.lower()
        q = query.lower()

        workflow_markers = ("statuses", "activities", "available_transitions", "create_retrieve")
        if not any(m in name for m in workflow_markers):
            return False, None

        so_query = (
            mentions_service_orders(query, self._intent_phrases)
            or has_business_code(query)
            or any(w in q for w in ("technician", "assigned", "scheduled", "address", "inspection", "so-"))
        )
        if not so_query:
            return False, None

        return True, (
            "Use the service-orders list or detail tool (with search=SO-###), "
            "not statuses, activities, or workflow endpoints."
        )

    def _check_scoping_filters(self, tool_call: ToolCall) -> list[str]:
        """Reject null/zero user-scoping filters — server handles permissions."""
        errors = []
        scoping_fields = {
            "technician", "company", "created_by", "customer", "service_order__technician"
        }
        for field in scoping_fields:
            if field not in tool_call.arguments:
                continue
            val = tool_call.arguments[field]
            if val in (0, "0", None, "null", "None"):
                errors.append(
                    f"Do not filter by {field}={val!r}. The server scopes data by user permissions."
                )
        return errors

    def _check_mutation_on_read_query(
        self,
        tool_call: ToolCall,
        query: str,
    ) -> tuple[bool, Optional[str]]:
        """Reject update/create tools when user asked to show/list/count."""
        if not is_read_only_query(query):
            return False, None
        if not is_mutation_tool(tool_call.name):
            return False, None
        return True, (
            f"'{tool_call.name}' modifies data. Use a list or retrieve tool for this read query."
        )

    def _check_wrong_embedded_lookup(
        self,
        tool_call: ToolCall,
        query: str,
    ) -> tuple[bool, Optional[str]]:
        """Reject by_service_id-style lookups for observations/embedded field queries."""
        if not query_wants_embedded_fields(query, phrases=self._intent_phrases):
            return False, None
        name = tool_call.name.lower()
        if any(marker in name for marker in EMBEDDED_LOOKUP_BAD_MARKERS):
            return True, (
                "Use the retrieve/{id}/ tool with the report id from the prior list result, "
                "not a by_service_id lookup endpoint."
            )
        return False, None

    def _check_transition_tool_for_scheduling(
        self,
        tool_call: ToolCall,
        query: str,
    ) -> tuple[bool, Optional[str]]:
        """Reject transition-impact tools for 'scheduled yet?' queries (Q03)."""
        if not is_scheduling_status_query(query, self._intent_phrases):
            return False, None
        name = tool_call.name.lower()
        if any(m in name for m in TRANSITION_TOOL_MARKERS):
            return True, (
                "Use service_orders_list to check whether the order is scheduled, "
                "not transition or impact workflow endpoints."
            )
        return False, None

    def _check_duplicate_list_call(
        self,
        tool_call: ToolCall,
        prior_tool_calls: list[ToolCall],
    ) -> tuple[bool, Optional[str]]:
        """Block identical repeated list calls on referential follow-ups."""
        if "list" not in tool_call.name.lower():
            return False, None
        for prior in prior_tool_calls:
            if prior.name == tool_call.name and prior.arguments == tool_call.arguments:
                return True, (
                    f"Do not repeat {tool_call.name} with the same empty filters. "
                    "Use id__in or search from prior list results."
                )
        return False, None

    def _calculate_confidence(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
        errors: list[str],
        warnings: list[str],
    ) -> float:
        """Calculate confidence score for the tool call."""
        if errors:
            return 0.0

        # Start at 1.0
        confidence = 1.0

        # Deduct for warnings
        confidence -= len(warnings) * 0.1

        # Check parameter coverage
        required_params = {
            p["name"] for p in endpoint.parameters
            if p.get("required")
        }
        provided_params = set(tool_call.arguments.keys())
        path_params = set(re.findall(r"\{(\w+)\}", endpoint.path))
        required_params.update(path_params)

        if required_params:
            coverage = len(provided_params & required_params) / len(required_params)
            confidence *= coverage

        return max(0.0, min(1.0, confidence))

    def calculate_full_confidence(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
        validation_result: ValidationResult,
        cache_hit: bool = False,
    ) -> ConfidenceScore:
        """Calculate full multi-factor confidence score."""
        # Tool match confidence
        tool_match = 1.0 if endpoint else 0.0

        # Parameter coverage
        if endpoint:
            required_params = {
                p["name"] for p in endpoint.parameters
                if p.get("required")
            }
            path_params = set(re.findall(r"\{(\w+)\}", endpoint.path))
            required_params.update(path_params)

            if required_params:
                provided = set(tool_call.arguments.keys())
                param_coverage = len(provided & required_params) / len(required_params)
            else:
                param_coverage = 1.0
        else:
            param_coverage = 0.0

        # Schema validity
        schema_valid = 1.0 if validation_result.valid else 0.0
        schema_valid -= len(validation_result.warnings) * 0.1
        schema_valid = max(0.0, schema_valid)

        # Cache boost
        cache_boost = 0.1 if cache_hit else 0.0

        # Overall (weighted average)
        overall = (
            tool_match * 0.3 +
            param_coverage * 0.3 +
            schema_valid * 0.4 +
            cache_boost
        )
        overall = min(1.0, overall)

        return ConfidenceScore(
            tool_match=tool_match,
            param_coverage=param_coverage,
            schema_valid=schema_valid,
            cache_boost=cache_boost,
            overall=overall,
        )
