"""
Normalize and enforce tool call arguments before validation/execution.
"""

from __future__ import annotations

from typing import Any, Optional

from .query_intent import QueryIntent
from .types import ToolCall


def flatten_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested ``filters`` / ``filters.filters`` shapes into top-level query params.

    LLMs often emit:
        {"filters": {"filters": {"status__name": "Completed"}}}
    APIs expect:
        {"status__name": "Completed"}
    """
    result = dict(arguments)
    nested = result.pop("filters", None)

    while isinstance(nested, dict):
        inner = nested.pop("filters", None)
        for key, value in nested.items():
            if key not in result:
                result[key] = value
        nested = inner if isinstance(inner, dict) else None

    return result


def enforce_count_tool_args(
    tool_calls: list[ToolCall],
    intent: QueryIntent,
    *,
    retain_page_size: int = 20,
    aggregate_page_size: Optional[int] = 100,
) -> list[ToolCall]:
    """
    For COUNT intent: never use page_size=1; keep a capped page for follow-up drill-down.

    For AGGREGATE intent: same page_size=1 guard, but with a larger page so
    grouping/ranking (e.g. "which technician has the most service orders") has
    enough rows to work with instead of silently aggregating a single row.
    """
    if intent not in (QueryIntent.COUNT, QueryIntent.AGGREGATE):
        return tool_calls

    page_size = aggregate_page_size if intent == QueryIntent.AGGREGATE else retain_page_size

    updated: list[ToolCall] = []
    for tc in tool_calls:
        if "list" not in tc.name.lower():
            updated.append(tc)
            continue

        args = flatten_tool_arguments(dict(tc.arguments))
        for key in ("page_size", "limit", "page_size_limit"):
            val = args.get(key)
            if val is not None:
                try:
                    if int(val) <= 1:
                        args.pop(key, None)
                except (TypeError, ValueError):
                    pass

        if page_size and page_size > 0:
            args["page_size"] = page_size

        updated.append(ToolCall(id=tc.id, name=tc.name, arguments=args))
    return updated


def extract_list_rows(data: Any, *, max_rows: int = 20) -> list[dict]:
    """Pull list rows from a paginated or plain list API payload."""
    if isinstance(data, dict) and "results" in data:
        rows = data.get("results")
        if isinstance(rows, list):
            return [r for r in rows[:max_rows] if isinstance(r, dict)]
    if isinstance(data, list):
        return [r for r in data[:max_rows] if isinstance(r, dict)]
    if isinstance(data, dict) and "id" in data:
        return [data]
    return []


def build_list_cache(
    tool_calls: list[ToolCall],
    results: list[dict],
    *,
    max_rows: int = 20,
) -> list[dict]:
    """Structured rows from the latest list response(s) for follow-up turns."""
    cache: list[dict] = []
    for tc, result in zip(tool_calls, results):
        if "error" in result:
            continue
        rows = extract_list_rows(result.get("data"), max_rows=max_rows)
        if rows:
            cache.extend(rows)
    return cache[:max_rows]
