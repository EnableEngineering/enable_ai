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


def default_error_suggestions() -> List[str]:
    """Generic suggestions for error / missing-context responses."""
    return [
        "Try rephrasing your question",
        "Ask about a specific resource like 'service orders' or 'users'",
        "Use filters like 'show new service orders' or 'list active users'",
    ]


def default_error_follow_up_queries() -> List[Dict[str, str]]:
    """Generic follow-up query templates when there is no result context."""
    return [
        {"label": "List service orders", "query": "list service orders"},
        {"label": "Show all users", "query": "show all users"},
        {"label": "List documents", "query": "list all documents"},
    ][: constants.SUGGESTIONS_MAX]
