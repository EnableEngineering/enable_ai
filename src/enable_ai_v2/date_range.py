"""
Extract temporal date-range filters from natural language queries.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional


COMPLETION_WORDS = ("completed", "finished", "done", "closed", "resolved")
DATE_FIELD_CREATED = ("created", "submitted", "opened", "new")
DATE_FIELD_UPDATED = ("updated", "modified", "changed")


def extract_date_range(
    query: str,
    reference: Optional[datetime] = None,
) -> dict[str, str]:
    """
    Map temporal phrases to ISO date filter params.

    Returns keys like created_at__gte / created_at__lte (or updated_at__* when
    the query implies completion/update time).
    """
    ref = reference or datetime.now()
    today = ref.date()
    q = query.lower()

    if not _has_temporal_phrase(q):
        return {}

    start, end = _resolve_range(q, today)
    if start is None and end is None:
        return {}

    field = _date_field_prefix(query)
    result: dict[str, str] = {}
    if start is not None:
        result[f"{field}__gte"] = start.isoformat()
    if end is not None:
        result[f"{field}__lte"] = end.isoformat()
    return result


def date_range_prompt_hint(query: str) -> str:
    """Short system-prompt hint when temporal filters are detected."""
    filters = extract_date_range(query)
    if not filters:
        return ""
    pairs = ", ".join(f"{k}={v}" for k, v in filters.items())
    return (
        f"Temporal phrase detected — include list filters: {pairs}. "
        "Use updated_at filters when the query refers to completion; "
        "created_at for submission/open dates."
    )


def _has_temporal_phrase(q: str) -> bool:
    patterns = (
        r"\bthis month\b",
        r"\bthis week\b",
        r"\blast week\b",
        r"\blast month\b",
        r"\bthis year\b",
        r"\btoday\b",
        r"\byesterday\b",
        r"\bpast (\d+) days?\b",
        r"\bnext (\d+) days?\b",
        r"\bdue (?:in )?next\b",
        r">\s*(\d+)\s*days?\b",
    )
    return any(re.search(p, q) for p in patterns)


def _date_field_prefix(query: str) -> str:
    q = query.lower()
    # Due date for invoice/payment queries
    if "due" in q or "overdue" in q or "payment" in q:
        return "due_date"
    if any(w in q for w in COMPLETION_WORDS):
        return "updated_at"
    if any(w in q for w in DATE_FIELD_UPDATED) and not any(w in q for w in DATE_FIELD_CREATED):
        return "updated_at"
    return "created_at"


def _resolve_range(q: str, today: date) -> tuple[Optional[date], Optional[date]]:
    if "today" in q:
        return today, today

    if "yesterday" in q:
        y = today - timedelta(days=1)
        return y, y

    if "this month" in q:
        return today.replace(day=1), today

    if "last month" in q:
        # First day of last month to last day of last month
        first_of_this = today.replace(day=1)
        last_of_prev = first_of_this - timedelta(days=1)
        first_of_prev = last_of_prev.replace(day=1)
        return first_of_prev, last_of_prev

    if "this week" in q:
        start = today - timedelta(days=today.weekday())
        return start, today

    if "last week" in q:
        end = today - timedelta(days=today.weekday() + 1)
        start = end - timedelta(days=6)
        return start, end

    if "this year" in q:
        return today.replace(month=1, day=1), today

    # "past N days" - looking backward
    match = re.search(r"\bpast (\d+) days?\b", q)
    if match:
        days = int(match.group(1))
        start = today - timedelta(days=max(days - 1, 0))
        return start, today

    # "next N days" - looking forward (for due dates)
    match = re.search(r"\bnext (\d+) days?\b", q)
    if match:
        days = int(match.group(1))
        end = today + timedelta(days=days)
        return today, end

    # "> N days" - items older than N days (InProgress > 7 days)
    match = re.search(r">\s*(\d+)\s*days?\b", q)
    if match:
        days = int(match.group(1))
        cutoff = today - timedelta(days=days)
        # Return as range ending at cutoff (older than N days)
        return None, cutoff

    return None, None


def extract_duration_filter(
    query: str,
    reference: Optional[datetime] = None,
) -> dict[str, str]:
    """
    Map 'more than N days' / 'older than N days' to updated_at__lte cutoff.

    Used for InProgress age queries (Q23).
    """
    ref = reference or datetime.now()
    today = ref.date()
    q = query.lower()

    patterns = (
        r"(?:more than|over|older than|at least|for more than)\s+(\d+)\s+days?",
        r"(\d+)\+\s*days?",
        r"(\d+)\s+days?\s+(?:old|in\s*progress|inprogress)",
    )
    days = None
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            days = int(match.group(1))
            break
    if days is None:
        return {}

    cutoff = today - timedelta(days=days)
    return {"updated_at__lte": cutoff.isoformat()}


def has_duration_phrase(query: str) -> bool:
    q = query.lower()
    return bool(extract_duration_filter(query)) or any(
        p in q for p in ("more than", "older than", " days old", "days in progress")
    )


def inject_temporal_filters(tool_calls: list, query: str) -> list:
    """Merge date-range and duration filters into list tool arguments."""
    from .types import ToolCall

    temporal = extract_date_range(query)
    temporal.update(extract_duration_filter(query))
    if not temporal:
        return tool_calls

    updated: list = []
    for tc in tool_calls:
        if "list" not in tc.name.lower():
            updated.append(tc)
            continue
        args = dict(tc.arguments)
        for key, value in temporal.items():
            if key not in args:
                args[key] = value
        updated.append(ToolCall(id=tc.id, name=tc.name, arguments=args))
    return updated


def inject_date_filters(
    tool_calls: list,
    query: str,
) -> list:
    """Backward-compatible alias — includes duration filters."""
    return inject_temporal_filters(tool_calls, query)
