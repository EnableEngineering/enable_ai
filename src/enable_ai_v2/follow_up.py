"""
Referential follow-up detection and deterministic tool calls.

Handles "details of those", "first 5 of above", etc. using prior list_cache rows.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .types import Endpoint, ToolCall
from .tool_filter import _tool_resource_segment


REFERENTIAL_PHRASES = (
    "those",
    "above",
    "prior results",
    "previous results",
    "from that list",
    "of above",
    "of those",
    "the ones above",
    "that list",
    "same ones",
)

SLICE_PATTERN = re.compile(
    r"\b(?:first|top|show)\s+(\d+)\b",
    re.IGNORECASE,
)


def is_referential_follow_up(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in REFERENTIAL_PHRASES)


def extract_slice_count(query: str, *, default: int = 5) -> int:
    match = SLICE_PATTERN.search(query)
    if match:
        return max(1, int(match.group(1)))
    if "first" in query.lower() or "top" in query.lower():
        return default
    return default


def extract_list_cache_from_history(
    history: Optional[list[dict]],
) -> list[dict]:
    """
    Pull list_cache from conversation history entries.

    Parent (KQSPL) can attach ``list_cache`` on assistant message dicts.
    """
    if not history:
        return []
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        cache = msg.get("list_cache")
        if isinstance(cache, list) and cache:
            return [r for r in cache if isinstance(r, dict)]
    return []


def _find_list_tool(
    endpoints: list[Endpoint],
    *,
    resource_hint: str = "service_orders",
) -> Optional[str]:
    for ep in endpoints:
        name = ep.tool_name.lower()
        if "list" not in name:
            continue
        segment = _tool_resource_segment(ep.tool_name)
        if resource_hint.replace("-", "_") in segment:
            return ep.tool_name
    return None


def _row_label(row: dict) -> Optional[str]:
    for key in ("display_name", "name", "so_sequence", "invoice_number", "req_id"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return None


def build_referential_tool_calls(
    query: str,
    list_cache: list[dict],
    endpoints: list[Endpoint],
) -> Optional[list[ToolCall]]:
    """
    Build list/retrieve calls using ids or search codes from prior list_cache.

    Prefer a single list call with id__in when ids are available.
    """
    if not list_cache:
        return None

    n = extract_slice_count(query)
    rows = list_cache[:n]
    ids = [r["id"] for r in rows if r.get("id") is not None]
    if not ids:
        return None

    list_tool = _find_list_tool(endpoints)
    if not list_tool:
        return None

    args: dict[str, Any] = {}
    if len(ids) == 1:
        args["search"] = _row_label(rows[0]) or str(ids[0])
        args["page_size"] = 1
    else:
        args["id__in"] = ids if len(ids) <= 20 else ids[:20]
        args["page_size"] = len(ids)

    return [ToolCall(id="ref-list", name=list_tool, arguments=args)]


def referential_prompt_hint(list_cache: list[dict]) -> str:
    """Inject prior row ids/codes into continuation context."""
    if not list_cache:
        return ""
    labels = []
    for row in list_cache[:10]:
        label = _row_label(row) or f"id={row.get('id')}"
        labels.append(label)
    return (
        "Prior list results (use these ids/search codes — do not repeat blank list calls): "
        + ", ".join(labels)
    )
