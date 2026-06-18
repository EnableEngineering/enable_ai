"""
Deterministic multi-step tool chaining (flash reports, scoped lists).
"""

from __future__ import annotations

from typing import Optional

from .query_intent import first_result_row, query_wants_flash_report
from .config import IntentPhrases, DEFAULT_INTENT_PHRASES
from .tool_filter import _tool_resource_segment
from .types import Endpoint, ToolCall

COMPOUND_RESOURCE_MAP: tuple[tuple[str, str], ...] = (
    ("service orders", "service_order"),
    ("service order", "service_order"),
    ("invoices", "invoice"),
    ("invoice", "invoice"),
    ("flash report", "flash"),
    ("flash reports", "flash"),
    ("equipment", "equipment"),
    ("reports", "report"),
    ("report", "report"),
)


def _segment_already_called(segment: str, called_segments: set[str]) -> bool:
    base = segment.rstrip("s")
    for called in called_segments:
        normalized = called.replace("-", "_")
        if segment in normalized or base in normalized:
            return True
    return False


def query_needs_scoped_second_list(query: str) -> bool:
    return query_wants_flash_report(query)


def find_tool_by_segment(
    endpoints: list[Endpoint],
    *,
    segment_contains: str,
    action: str = "list",
) -> Optional[str]:
    for ep in endpoints:
        name = ep.tool_name.lower()
        if action not in name:
            continue
        segment = _tool_resource_segment(ep.tool_name)
        if segment_contains.replace("-", "_") in segment:
            return ep.tool_name
    return None


def build_scoped_flash_report_list(
    query: str,
    prior_tool_name: str,
    prior_result: dict,
    endpoints: list[Endpoint],
    phrases: IntentPhrases | None = None,
) -> Optional[ToolCall]:
    """
    After latest SO list, scope flash-reports list to that service order (Q07).
    """
    if not query_wants_flash_report(query, phrases):
        return None
    if "list" not in prior_tool_name.lower():
        return None
    prior_segment = _tool_resource_segment(prior_tool_name)
    # Handle nested paths like service_orders_service_orders
    if not ("service_order" in prior_segment or "service-order" in prior_segment):
        return None
    if "error" in prior_result:
        return None

    row = first_result_row(prior_result.get("data"))
    if not row:
        return None

    flash_tool = find_tool_by_segment(endpoints, segment_contains="flash", action="list")
    if not flash_tool:
        flash_tool = find_tool_by_segment(endpoints, segment_contains="flash_report", action="list")
    if not flash_tool:
        return None

    args: dict = {"page_size": 5}
    if row.get("id") is not None:
        args["service_order"] = row["id"]
    label = row.get("display_name") or row.get("name") or row.get("so_sequence")
    if label and str(label).strip():
        text = str(label).strip()
        args["search"] = text
        if text.upper().startswith("SO"):
            args.setdefault("service_order__search", text)

    return ToolCall(id="auto-flash-list", name=flash_tool, arguments=args)


def build_workload_service_orders_list(
    query: str,
    prior_tool_calls: list[ToolCall],
    endpoints: list[Endpoint],
    phrases: IntentPhrases | None = None,
) -> Optional[ToolCall]:
    """
    After users/technicians list, fetch open service orders for availability synthesis (Q13).
    """
    from .query_intent import is_technician_availability_query

    if not is_technician_availability_query(query, phrases):
        return None
    if not any(
        "user" in tc.name.lower() or "technician" in tc.name.lower()
        for tc in prior_tool_calls
    ):
        return None
    if any("service_order" in tc.name.lower() for tc in prior_tool_calls):
        return None

    so_tool = find_tool_by_segment(endpoints, segment_contains="service_order", action="list")
    if not so_tool:
        return None

    return ToolCall(
        id="auto-so-workload",
        name=so_tool,
        arguments={"page_size": 100},
    )


def build_compound_second_list(
    query: str,
    prior_tool_calls: list[ToolCall],
    endpoints: list[Endpoint],
    phrases: IntentPhrases | None = None,
) -> Optional[ToolCall]:
    """After first list call, auto-fetch the other resource in a compound query (Q19)."""
    from .query_intent import is_compound_query, needs_compound_follow_up

    if not is_compound_query(query, phrases) or not needs_compound_follow_up(query, prior_tool_calls, phrases):
        return None

    called = {_tool_resource_segment(tc.name) for tc in prior_tool_calls}
    q = query.lower()
    p = phrases or DEFAULT_INTENT_PHRASES
    for phrase, segment in COMPOUND_RESOURCE_MAP:
        if phrase not in q:
            continue
        if _segment_already_called(segment, called):
            continue
        tool = find_tool_by_segment(endpoints, segment_contains=segment, action="list")
        if tool:
            return ToolCall(
                id=f"auto-{segment}-compound",
                name=tool,
                arguments={"page_size": 20},
            )
    return None


def build_technician_directory_list(
    query: str,
    prior_tool_calls: list[ToolCall],
    endpoints: list[Endpoint],
    phrases: IntentPhrases | None = None,
) -> Optional[ToolCall]:
    """When availability query fetched SOs first, pull the technician directory next (Q13)."""
    from .query_intent import is_technician_availability_query

    if not is_technician_availability_query(query, phrases):
        return None
    if any(
        "user" in tc.name.lower() or "technician" in tc.name.lower()
        for tc in prior_tool_calls
    ):
        return None
    if not any("service_order" in tc.name.lower() for tc in prior_tool_calls):
        return None

    for segment in ("user", "technician"):
        tool = find_tool_by_segment(endpoints, segment_contains=segment, action="list")
        if tool:
            return ToolCall(
                id=f"auto-{segment}-directory",
                name=tool,
                arguments={"page_size": 100},
            )
    return None


def build_retrieve_for_embedded(
    query: str,
    prior_tool_name: str,
    prior_result: dict,
    endpoints: list[Endpoint],
    phrases: IntentPhrases | None = None,
) -> Optional[ToolCall]:
    """
    After a list call for embedded fields, auto-retrieve the first row.

    Handles "observations for latest", "flash reports for SO", etc.
    """
    from .query_intent import (
        query_wants_embedded_fields,
        has_embedded_content,
        is_limited_list_args,
    )

    # Only applies to list tool calls
    if "list" not in prior_tool_name.lower():
        return None

    # Check if query wants embedded content
    p = phrases or DEFAULT_INTENT_PHRASES
    if not query_wants_embedded_fields(query, phrases=p):
        return None

    # Skip if error
    if "error" in prior_result:
        return None

    # Extract first row
    row = first_result_row(prior_result.get("data"))
    if not row or "id" not in row:
        return None

    # Already has embedded content? No need to retrieve
    if has_embedded_content(prior_result.get("data"), p.embedded_field_names):
        return None

    # Find retrieve tool for same resource
    prior_segment = _tool_resource_segment(prior_tool_name)
    retrieve_tool = find_tool_by_segment(
        endpoints, segment_contains=prior_segment, action="retrieve"
    )
    if not retrieve_tool:
        return None

    return ToolCall(
        id="auto-retrieve-embedded",
        name=retrieve_tool,
        arguments={"id": row["id"]},
    )
