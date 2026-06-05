"""
Helpers for passing execution context (sort, limit, filters) through the pipeline.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from . import constants

# Fields that must flow from parsed query into each execution step
EXECUTION_CONTEXT_FIELDS = (
    "sort",
    "limit",
    "display_mode",
    "question_type",
    "original_input",
)


def merge_execution_context(step: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge sort/limit/display fields from parsed query into a step definition."""
    if not parsed:
        return step
    merged = dict(step)
    for field in EXECUTION_CONTEXT_FIELDS:
        if field in parsed and field not in merged:
            merged[field] = parsed[field]
    return merged


def enrich_step_from_parsed(step: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a complete step dict from planner output + parsed query context."""
    base = {
        "step_id": step.get("step_id", 1),
        "intent": step.get("intent", parsed.get("intent", "read") if parsed else "read"),
        "resource": step.get("resource", parsed.get("resource", "") if parsed else ""),
        "entities": step.get("entities", {}),
        "filters": step.get("filters", {}),
        "depends_on": step.get("depends_on", []),
        "description": step.get("description", ""),
    }
    for optional in ("extract", "fetch_all_pages", "type", "url"):
        if optional in step:
            base[optional] = step[optional]
    return merge_execution_context(base, parsed)


def format_sort_param(sort: Any) -> Optional[str]:
    """
    Convert parsed sort to an API ordering string.

    Supports:
        {"field": "created_at", "order": "desc"} -> "-created_at"
        "created_at" -> "created_at"
        "-created_at" -> "-created_at"
    """
    if sort is None:
        return None
    if isinstance(sort, str):
        return sort.strip() or None
    if isinstance(sort, dict):
        field = sort.get("field") or sort.get("name")
        if not field:
            return None
        order = (sort.get("order") or sort.get("direction") or "asc").lower()
        if order in ("desc", "descending", "-1"):
            return f"-{field}"
        return str(field)
    return str(sort)


def get_endpoint_query_param_names(endpoint_data: Dict[str, Any]) -> Set[str]:
    """Collect query parameter names supported by an endpoint."""
    names: Set[str] = set()
    params_def = endpoint_data.get("parameters", {})
    query_params = params_def.get("query", []) if isinstance(params_def, dict) else []

    for param in query_params:
        if isinstance(param, dict):
            name = param.get("name", "")
        else:
            name = str(param)
        if name:
            names.add(name)
            names.add(name.split("__")[0])
    return names


def filter_matches_endpoint(field: str, available_params: Set[str]) -> bool:
    """Return True if a filter field can be sent as a server-side query param."""
    base_field = field.split("__")[0]
    if field in available_params or base_field in available_params:
        return True
    return any(p.startswith(base_field + "__") or p == base_field for p in available_params)


def dedupe_fk_lookup_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop base FK fields when a Django lookup variant is present.

    e.g. keep status__name=New, remove status=New (FK expects integer id).
    """
    if not filters:
        return filters

    bases_with_lookup = {
        field.split("__", 1)[0]
        for field in filters
        if "__" in field
    }
    if not bases_with_lookup:
        return filters

    return {
        field: value
        for field, value in filters.items()
        if field not in bases_with_lookup or "__" in field
    }


def split_filters_for_endpoint(
    filters: Dict[str, Any],
    endpoint_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Split filters into server-side (API params) and client-side (post-filter).

    Returns:
        (server_filters, client_filters, warnings)
    """
    if not filters:
        return {}, {}, []

    available = get_endpoint_query_param_names(endpoint_data)
    server_filters: Dict[str, Any] = {}
    client_filters: Dict[str, Any] = {}
    warnings: List[str] = []

    for field, value in filters.items():
        if available and filter_matches_endpoint(field, available):
            server_filters[field] = value
        elif not available:
            # No param metadata — attempt server-side (legacy behaviour)
            server_filters[field] = value
        else:
            client_filters[field] = value
            warnings.append(
                f"Filter '{field}' is not supported by this endpoint; "
                f"will apply client-side after fetching results."
            )

    return server_filters, client_filters, warnings


def apply_limit_to_params(
    params: Dict[str, Any],
    parsed: Dict[str, Any],
    endpoint_data: Dict[str, Any],
    default_page_size: int,
) -> Dict[str, Any]:
    """Apply user limit or default page_size to query params when appropriate."""
    params = dict(params)
    available = get_endpoint_query_param_names(endpoint_data)

    limit = parsed.get("limit")
    if limit is not None:
        try:
            n = min(int(limit), constants.PAGE_SIZE_CAP)
            for limit_param in constants.LIMIT_PARAM_NAMES:
                if not available or limit_param in available:
                    params[limit_param] = n
                    break
        except (TypeError, ValueError):
            pass
    elif (
        parsed.get("question_type") != "count"
        and parsed.get("display_mode") != "full"
        and parsed.get("intent") == "read"
    ):
        for limit_param in constants.LIMIT_PARAM_NAMES:
            if limit_param not in params and (not available or limit_param in available):
                params[limit_param] = default_page_size
                break

    return params
