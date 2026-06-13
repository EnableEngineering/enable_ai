"""
Map relative date phrases to filter ranges (created_at, due_date, etc.).
"""

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .utils import setup_logger

logger = setup_logger("enable_ai.temporal_filters")

_RELATIVE_PHRASES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bthis\s+week\b", re.I), "this_week"),
    (re.compile(r"\blast\s+week\b", re.I), "last_week"),
    (re.compile(r"\bthis\s+month\b", re.I), "this_month"),
    (re.compile(r"\blast\s+month\b", re.I), "last_month"),
    (re.compile(r"\btoday\b", re.I), "today"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
]


def _start_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _start_of_month(d: date) -> date:
    return d.replace(day=1)


def _end_of_month(d: date) -> date:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return d.replace(day=last_day)


def resolve_relative_range(phrase_key: str, today: Optional[date] = None) -> Optional[Tuple[str, str]]:
    """Return (gte_iso, lte_iso) date strings for a relative phrase key."""
    today = today or date.today()

    if phrase_key == "today":
        s = today.isoformat()
        return s, s
    if phrase_key == "yesterday":
        y = today - timedelta(days=1)
        s = y.isoformat()
        return s, s
    if phrase_key == "this_week":
        start = _start_of_week(today)
        return start.isoformat(), today.isoformat()
    if phrase_key == "last_week":
        end = _start_of_week(today) - timedelta(days=1)
        start = _start_of_week(end)
        return start.isoformat(), end.isoformat()
    if phrase_key == "this_month":
        start = _start_of_month(today)
        return start.isoformat(), today.isoformat()
    if phrase_key == "last_month":
        first_this = _start_of_month(today)
        end = first_this - timedelta(days=1)
        start = _start_of_month(end)
        return start.isoformat(), end.isoformat()
    return None


def detect_temporal_phrase(query: str) -> Optional[str]:
    for pattern, key in _RELATIVE_PHRASES:
        if pattern.search(query or ""):
            return key
    return None


def get_temporal_fields(resource: str, resource_hints: Dict[str, Any]) -> List[str]:
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return ["created_at"]
    fields = hints.get("__temporal_fields__") or hints.get("__date_filter_fields__")
    if isinstance(fields, (list, tuple)) and fields:
        return [str(f) for f in fields]
    return ["created_at"]


def apply_temporal_filters(
    parsed: Dict[str, Any],
    query: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Inject gte/lte date filters when query mentions this week/month etc."""
    if not isinstance(parsed, dict) or not query:
        return parsed

    phrase_key = detect_temporal_phrase(query)
    if not phrase_key:
        return parsed

    date_range = resolve_relative_range(phrase_key)
    if not date_range:
        return parsed

    gte_val, lte_val = date_range
    resource = parsed.get("resource") or ""
    hints = (schema or {}).get("resource_hints") or {}
    fields = get_temporal_fields(resource, hints)

    result = dict(parsed)
    filters = dict(result.get("filters") or {})
    applied = False
    for field in fields:
        gte_key = f"{field}__gte"
        lte_key = f"{field}__lte"
        if gte_key not in filters and lte_key not in filters and field not in filters:
            filters[gte_key] = {"operator": "gte", "value": gte_val}
            filters[lte_key] = {"operator": "lte", "value": lte_val}
            applied = True
            break

    if applied:
        result["filters"] = filters
        logger.info(
            "Temporal filter (%s): %s gte=%s lte=%s",
            phrase_key, fields[0], gte_val, lte_val,
        )
    return result
