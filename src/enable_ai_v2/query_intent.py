"""
Query intent classification for tool selection and formatting.

Runs before LLM tool selection to distinguish count vs detail vs multi-step queries.

Domain phrase lists live in Config.intent_phrases (parent/KQSPL); pip supplies defaults
via DEFAULT_INTENT_PHRASES when the parent does not override.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

from .config import DEFAULT_INTENT_PHRASES, IntentPhrases, resolve_intent_phrases
from .tool_filter import extract_business_codes, has_business_code, is_count_query

if TYPE_CHECKING:
    from .config import Config

# Backward-compatible re-exports (defaults; prefer resolve_intent_phrases(config))
DEFAULT_EMBEDDED_FIELDS = DEFAULT_INTENT_PHRASES.embedded_field_names
MULTI_STEP_KEYWORDS = DEFAULT_INTENT_PHRASES.multi_step_keywords
SINGLE_DETAIL_KEYWORDS = DEFAULT_INTENT_PHRASES.single_detail_keywords
SERVICE_ORDER_KEYWORDS = DEFAULT_INTENT_PHRASES.service_order_keywords
AGGREGATE_KEYWORDS = DEFAULT_INTENT_PHRASES.aggregate_keywords
COMPOUND_RESOURCE_PHRASES = DEFAULT_INTENT_PHRASES.compound_resource_phrases


class QueryIntent(str, Enum):
    COUNT = "count"
    SINGLE_DETAIL = "single_detail"
    LIST = "list"
    MULTI_STEP = "multi_step"
    AGGREGATE = "aggregate"


def _phrases(phrases: Optional[IntentPhrases] = None) -> IntentPhrases:
    return phrases or DEFAULT_INTENT_PHRASES


def _compound_segment(phrase: str, phrases: IntentPhrases) -> str:
    if phrase in phrases.compound_resource_segments:
        return phrases.compound_resource_segments[phrase]
    if phrase.startswith("service"):
        return "service_order"
    if phrase in ("invoice", "invoices"):
        return "invoice"
    if phrase.startswith("flash"):
        return "flash"
    return phrase.split()[0].replace("-", "_")


def is_compound_query(query: str, phrases: Optional[IntentPhrases] = None) -> bool:
    """True when query asks about two+ resources joined by 'and' (Q19)."""
    p = _phrases(phrases)
    q = query.lower()
    if " and " not in q:
        return False
    hits = sum(1 for phrase in p.compound_resource_phrases if phrase in q)
    return hits >= 2


def is_technician_availability_query(
    query: str,
    phrases: Optional[IntentPhrases] = None,
) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _phrases(phrases).availability_phrases)


def compound_continuation_hint(
    query: str,
    tool_calls: list,
    phrases: Optional[IntentPhrases] = None,
) -> str:
    """Tell the LLM which resource still needs a list call (Q19)."""
    p = _phrases(phrases)
    if not is_compound_query(query, p) or not needs_compound_follow_up(query, tool_calls, p):
        return ""
    from .tool_filter import _tool_resource_segment

    called = {_tool_resource_segment(tc.name) for tc in tool_calls}
    missing: list[str] = []
    q = query.lower()
    for phrase in p.compound_resource_phrases:
        if phrase not in q:
            continue
        segment = _compound_segment(phrase, p)
        if not any(segment.rstrip("s") in c for c in called):
            missing.append(phrase)
    if not missing:
        return ""
    return (
        "Compound query — call a separate list tool for each resource mentioned "
        f"(still needed: {', '.join(dict.fromkeys(missing))}). "
        "Do not answer from only one endpoint."
    )


def needs_compound_follow_up(
    query: str,
    tool_calls: list,
    phrases: Optional[IntentPhrases] = None,
) -> bool:
    if not is_compound_query(query, phrases):
        return False
    from .tool_filter import _tool_resource_segment

    segments: set[str] = set()
    for tc in tool_calls:
        seg = _tool_resource_segment(tc.name).replace("-", "_")
        base = seg.rstrip("s") if seg.endswith("s") and len(seg) > 1 else seg
        segments.add(base)
    return len(segments) < 2


def classify_query_intent(
    query: str,
    phrases: Optional[IntentPhrases] = None,
) -> QueryIntent:
    """Classify user query intent before tool selection."""
    p = _phrases(phrases)
    q = query.lower()

    if is_count_query(query):
        return QueryIntent.COUNT

    if any(kw in q for kw in p.multi_step_keywords):
        return QueryIntent.MULTI_STEP

    if is_compound_query(query, p):
        return QueryIntent.AGGREGATE

    if any(kw in q for kw in p.aggregate_keywords):
        return QueryIntent.AGGREGATE

    if is_technician_availability_query(query, p):
        return QueryIntent.AGGREGATE

    if has_business_code(query) or extract_business_codes(query):
        return QueryIntent.SINGLE_DETAIL

    if any(kw in q for kw in p.single_detail_keywords):
        return QueryIntent.SINGLE_DETAIL

    return QueryIntent.LIST


def intent_prompt_hint(intent: QueryIntent) -> str:
    """Short hint appended to system prompt for the detected intent."""
    hints = {
        QueryIntent.MULTI_STEP: (
            "Detected intent: MULTI_STEP. Step 1: list with page_size=1 (or search). "
            "Step 2: scoped list/retrieve — pass service_order or search=SO-### from step 1 "
            "for flash reports; use *_retrieve with id for observations. "
            "NOT by_service_id endpoints."
        ),
        QueryIntent.COUNT: (
            "Detected intent: COUNT. Use list tools without page_size=1; read total from paginated count. "
            "Apply date range filters (created_at__gte/lte or updated_at__gte/lte) when the query mentions "
            "this month/week/year."
        ),
        QueryIntent.SINGLE_DETAIL: (
            "Detected intent: SINGLE_DETAIL. Use list with page_size=1 or search filter; "
            "format the single row, not the total count."
        ),
        QueryIntent.AGGREGATE: (
            "Detected intent: AGGREGATE. Call tools for each resource mentioned; "
            "for compound queries with 'and', use one list tool per resource before summarizing."
        ),
        QueryIntent.LIST: (
            "Detected intent: LIST. Return matching rows; use filters from the query."
        ),
    }
    return hints.get(intent, "")


def mentions_service_orders(
    query: str,
    phrases: Optional[IntentPhrases] = None,
) -> bool:
    q = query.lower()
    return any(kw in q for kw in _phrases(phrases).service_order_keywords) or has_business_code(query)


def is_limited_list_args(arguments: dict) -> bool:
    """True when tool args request a single row (page_size=1 or limit=1)."""
    for key in ("page_size", "limit", "page_size_limit"):
        if key not in arguments:
            continue
        try:
            if int(arguments[key]) == 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def first_result_row(data: object) -> dict | None:
    """Extract first row from API payload."""
    if isinstance(data, dict) and "results" in data:
        results = data.get("results")
        if isinstance(results, list) and results:
            return results[0] if isinstance(results[0], dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict) and "id" in data:
        return data
    return None


def has_embedded_content(
    data: object,
    fields: tuple[str, ...] | None = None,
) -> bool:
    """True if payload already contains requested nested/embedded fields."""
    field_names = fields or DEFAULT_EMBEDDED_FIELDS
    row = first_result_row(data)
    if not row:
        return False
    for field in field_names:
        val = row.get(field)
        if val is None:
            continue
        if isinstance(val, list) and len(val) > 0:
            return True
        if isinstance(val, dict) and val:
            return True
        if isinstance(val, str) and val.strip():
            return True
    return False


def query_wants_embedded_fields(
    query: str,
    fields: tuple[str, ...] | None = None,
    phrases: Optional[IntentPhrases] = None,
) -> bool:
    p = _phrases(phrases)
    field_names = fields or p.embedded_field_names
    q = query.lower()
    if query_wants_flash_report(query, p):
        return True
    return any(f.replace("_", " ") in q or f in q for f in field_names)


def query_wants_flash_report(
    query: str,
    phrases: Optional[IntentPhrases] = None,
) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _phrases(phrases).flash_report_phrases)


def is_scheduling_status_query(
    query: str,
    phrases: Optional[IntentPhrases] = None,
) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _phrases(phrases).scheduling_phrases)


def needs_follow_up_round(
    query: str,
    intent: QueryIntent,
    tool_calls: list,
    results: list[dict],
    *,
    embedded_fields: tuple[str, ...] | None = None,
    phrases: Optional[IntentPhrases] = None,
) -> bool:
    """
    True when another LLM tool-selection round is needed after executing tools.
    """
    p = _phrases(phrases)
    fields = embedded_fields or p.embedded_field_names

    if not tool_calls or not results:
        return False

    last_tc = tool_calls[-1]
    last_result = results[-1]
    if "error" in last_result:
        return False

    if needs_compound_follow_up(query, tool_calls, p):
        return True

    if is_technician_availability_query(query, p):
        has_users = any(
            "user" in tc.name.lower() or "technician" in tc.name.lower()
            for tc in tool_calls
        )
        has_orders = any("service_order" in tc.name.lower() for tc in tool_calls)
        if has_users and not has_orders:
            return True

    if intent != QueryIntent.MULTI_STEP and not query_wants_embedded_fields(query, fields, p):
        return False

    data = last_result.get("data")

    # Flash report chain: SO list done → need scoped flash-reports list
    if query_wants_flash_report(query, p):
        if any("flash" in tc.name.lower() for tc in tool_calls):
            return False
        last_segment = last_tc.name.lower()
        if "service_order" in last_segment and "list" in last_segment:
            return True

    if has_embedded_content(data, fields):
        return False

    if query_wants_embedded_fields(query, fields, p):
        return True

    if intent == QueryIntent.MULTI_STEP:
        row = first_result_row(data)
        if row and "list" in last_tc.name.lower():
            return True

    return False


def embedded_continuation_hint(
    query: str,
    list_tool_name: str,
    row: dict,
) -> str:
    """Explicit follow-up instruction for list → retrieve embedded-field chains."""
    from .tool_filter import _tool_resource_segment

    segment = _tool_resource_segment(list_tool_name)
    retrieve_name = f"{segment}_retrieve"
    row_id = row.get("id")
    if row_id is None:
        return ""
    return (
        f"Next step: call {retrieve_name} with id={row_id} to read nested content. "
        "Do NOT use by_service_id or service_order lookup endpoints."
    )


def resolve_phrases_from_config(config: Optional["Config"] = None) -> IntentPhrases:
    """Public alias used by Orchestrator and parent modules."""
    return resolve_intent_phrases(config)
