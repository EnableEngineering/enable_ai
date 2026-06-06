"""
Shared helpers for contextual follow-up suggestions and query templates.

Used by workflow (API response fields) and response_formatter (summary text).
"""

from typing import Any, Dict, List, Optional

from . import constants


def _resource_label(parsed: Optional[Dict[str, Any]]) -> str:
    resource = (parsed or {}).get("resource") or "items"
    return str(resource).replace("_", " ")


def generate_suggestions(
    pagination_info: Dict[str, Any],
    parsed: Optional[Dict[str, Any]] = None,
    data: Any = None,
) -> List[str]:
    """Generate contextual suggested_actions based on result count and pagination."""
    suggestions: List[str] = []
    total = pagination_info.get("total_count", 0)
    has_more = pagination_info.get("has_more", False)

    if total == 0:
        suggestions.append(constants.SUGGESTION_TRY_BROADER)
    elif total == 1:
        suggestions.append(constants.SUGGESTION_SHOW_DETAILS)
    elif total <= 5:
        suggestions.append(constants.SUGGESTION_SHOW_DETAILS_ITEM)
    elif total <= constants.TABLE_ROW_SAMPLE:
        suggestions.append(constants.SUGGESTION_FILTER_SPECIFIC)
    elif total > constants.TABLE_ROW_SAMPLE:
        suggestions.append(constants.SUGGESTION_NARROW_FILTERS)
        if has_more:
            suggestions.append(constants.SUGGESTION_SHOW_MORE)

    return suggestions[: constants.SUGGESTIONS_MAX]


def generate_follow_up_queries(
    pagination_info: Dict[str, Any],
    parsed: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Return structured follow-up query templates for the frontend.

    Each entry has:
        - label: short display text
        - query: natural-language query the user can send
    """
    queries: List[Dict[str, str]] = []
    resource = _resource_label(parsed)
    total = pagination_info.get("total_count", 0)
    shown = pagination_info.get("actual_count", 0)
    has_more = pagination_info.get("has_more", False)
    remaining = max(total - shown, 0)

    if total == 0:
        queries.append({
            "label": "Try a broader search",
            "query": f"show all {resource}",
        })
        return queries[: constants.SUGGESTIONS_MAX]

    if has_more or (total > shown and remaining > 0):
        next_n = min(remaining, 20) if remaining > 0 else 20
        queries.append({
            "label": f"Show next {next_n}",
            "query": f"show me next {next_n}",
        })

    if has_more or total > shown:
        queries.append({
            "label": f"Show all {resource}",
            "query": f"show all {resource}",
        })

    if total == 1:
        queries.append({
            "label": "Show details",
            "query": "show details",
        })
    elif 2 <= total <= 5:
        queries.append({
            "label": "Show item details",
            "query": "show details on the first one",
        })
    elif total > constants.TABLE_ROW_SAMPLE:
        queries.append({
            "label": "Narrow results",
            "query": f"show {resource} with ",
        })

    return queries[: constants.SUGGESTIONS_MAX]


def enrich_response_with_follow_ups(
    response: Dict[str, Any],
    pagination_info: Optional[Dict[str, Any]] = None,
    parsed: Optional[Dict[str, Any]] = None,
    data: Any = None,
) -> Dict[str, Any]:
    """Add suggested_actions and follow_up_queries to a response dict."""
    pagination_info = pagination_info or {}
    if "suggested_actions" not in response:
        response["suggested_actions"] = generate_suggestions(pagination_info, parsed, data)
    if "follow_up_queries" not in response:
        response["follow_up_queries"] = generate_follow_up_queries(pagination_info, parsed)
    return response


def _extract_resource_names(schema: Optional[Dict[str, Any]]) -> List[str]:
    """Extract resource names from schema for dynamic suggestions."""
    if not schema:
        return []
    resources: List[str] = []
    # Try resource_hints keys first (most reliable)
    hints = schema.get("resource_hints") or {}
    for name in hints.keys():
        if not name.startswith("__"):
            resources.append(str(name).replace("-", " ").replace("_", " "))
    # Fallback to paths if no hints
    if not resources:
        paths = schema.get("paths") or {}
        for path in paths.keys():
            parts = str(path).strip("/").split("/")
            if parts and parts[0] and not parts[0].startswith("{"):
                resources.append(parts[0].replace("-", " ").replace("_", " "))
    # Dedupe and limit
    seen: set = set()
    unique: List[str] = []
    for r in resources:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique[:10]


def schema_based_error_suggestions(schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Generate error suggestions dynamically from schema resources.

    No hardcoded domain examples — if schema unavailable, return generic message only.
    """
    suggestions = ["Try rephrasing your question"]
    resources = _extract_resource_names(schema)
    if resources:
        sample = ", ".join(f"'{r}'" for r in resources[:3])
        suggestions.append(f"Ask about a resource like {sample}")
    return suggestions[: constants.SUGGESTIONS_MAX]


def schema_based_follow_up_queries(schema: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    Generate follow-up query templates dynamically from schema resources.

    No hardcoded domain examples — returns empty list if no schema available.
    """
    resources = _extract_resource_names(schema)
    if not resources:
        return []
    queries: List[Dict[str, str]] = []
    for resource in resources[: constants.SUGGESTIONS_MAX]:
        queries.append({
            "label": f"List {resource}",
            "query": f"list {resource}",
        })
    return queries


def default_error_suggestions(schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generic suggestions for error / missing-context responses (schema-aware)."""
    return schema_based_error_suggestions(schema)


def default_error_follow_up_queries(schema: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Generic follow-up query templates when there is no result context (schema-aware)."""
    return schema_based_follow_up_queries(schema)
