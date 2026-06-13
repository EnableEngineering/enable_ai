"""
LLM-based follow-up query detection and context anchoring.

No hardcoded phrase lists — an LLM classifies whether the current query
continues or refines the previous conversation turn.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Set

from . import constants
from .hint_utils import (
    expand_aggregate_follow_up,
    get_endpoint_role,
    get_related_list_resource,
    get_user_scoped_fields,
    resolve_summary_field_from_query,
    should_force_standalone_for_resource_switch,
)
from .response_projector import (
    apply_chat_window,
    build_chat_summary,
    get_chat_window_size,
    get_list_display_fields,
)
from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP

logger = setup_logger("enable_ai.follow_up_detection")

# Cache classifications within a process to avoid duplicate LLM calls
_classification_cache: Dict[str, Dict[str, Any]] = {}


def clear_classification_cache() -> None:
    """Clear in-process classification cache (for tests)."""
    _classification_cache.clear()

FOLLOW_UP_TYPES = (
    "standalone",
    "reset",
    "next_page",
    "first_n",
    "last_n",
    "reference",
    "refinement",
)

# Types that start a fresh query — never inherit previous filters
NO_MERGE_TYPES = frozenset({"standalone", "reset"})


def extract_result_items_from_data(
    data: Any,
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    """Extract slim item records from API response data for session context."""
    if not data:
        return []

    if isinstance(data, dict) and "results" in data:
        rows = data.get("results") or []
    elif isinstance(data, dict) and "items" in data:
        rows = data.get("items") or []
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and "id" in data:
        rows = [data]
    else:
        return []

    items: List[Dict[str, Any]] = []
    for row in rows[:max_items]:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        slim: Dict[str, Any] = {"id": row.get("id")}
        for key in ("name", "code", "title", "number", "reference"):
            if row.get(key) is not None:
                slim[key] = row.get(key)
        company = row.get("company")
        if isinstance(company, dict):
            if company.get("name"):
                slim["company_name"] = company.get("name")
            if company.get("id"):
                slim["company_id"] = company.get("id")
        elif row.get("company_name"):
            slim["company_name"] = row.get("company_name")
        items.append(slim)
    return items


def build_session_metadata(
    parsed: Dict[str, Any],
    response: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build assistant message metadata for follow-up / pronoun resolution."""
    pagination = response.get("pagination") or {}
    data = response.get("data")
    items = extract_result_items_from_data(data)
    count = pagination.get("total_count")
    if count is None and isinstance(data, dict):
        count = data.get("count") or data.get("total_count")
    if count is None and items:
        count = len(items)

    primary_item = None
    if len(items) == 1:
        primary_item = items[0]
    elif count == 1 and items:
        primary_item = items[0]

    resource = parsed.get("resource")
    hints = (schema or {}).get("resource_hints") or {}
    list_cache = (projection or {}).get("list_cache") or []
    chat_offset = (projection or {}).get("chat_offset", 0)
    chat_window_size = (projection or {}).get("chat_window_size") or get_chat_window_size(
        resource or "", hints,
    )
    list_display_fields = (projection or {}).get("list_display_fields") or get_list_display_fields(
        resource or "", hints,
    )
    has_more_in_chat = (projection or {}).get("has_more_in_chat", False)
    if list_cache and not projection:
        has_more_in_chat = (chat_offset + chat_window_size) < len(list_cache)
    total_known = count or len(list_cache)
    if not has_more_in_chat and len(list_cache) < total_known:
        has_more_in_chat = True
    if not has_more_in_chat and pagination.get("next_url"):
        has_more_in_chat = True

    return {
        "resource": resource,
        "intent": parsed.get("intent"),
        "question_type": parsed.get("question_type"),
        "display_mode": parsed.get("display_mode"),
        "filters": parsed.get("filters", {}),
        "next_url": pagination.get("next_url"),
        "count": count,
        "has_more": pagination.get("has_more", False),
        "result_items": items,
        "primary_item": primary_item,
        "list_cache": list_cache,
        "chat_offset": chat_offset,
        "chat_window_size": chat_window_size,
        "list_display_fields": list_display_fields,
        "has_more_in_chat": has_more_in_chat,
        "total_cached": len(list_cache) if list_cache else (projection or {}).get("total_cached"),
        "multiple_resources": parsed.get("multiple_resources"),
        "related_list_resource": get_related_list_resource(resource or "", hints),
        "endpoint_role": get_endpoint_role(resource or "", hints),
    }


LIST_CACHE_FOLLOW_UP_TYPES = ("reference", "first_n", "last_n", "next_page")


def should_pivot_count_to_list(
    last_metadata: Dict[str, Any],
    follow_up_type: str,
) -> bool:
    """True when a list-style follow-up follows a count turn."""
    if follow_up_type not in LIST_CACHE_FOLLOW_UP_TYPES:
        return False
    return last_metadata.get("question_type") == "count"


def build_count_list_pivot_parsed(
    last_metadata: Dict[str, Any],
    query: str,
    follow_up_type: str,
    requested_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a list query preserving filters from a prior count turn."""
    resource = last_metadata.get("resource") or ""
    limit = requested_limit
    if limit is None and follow_up_type == "first_n":
        limit = 5
    return {
        "intent": "read",
        "resource": resource,
        "question_type": "list",
        "display_mode": "summary",
        "limit": limit,
        "filters": dict(last_metadata.get("filters") or {}),
        "original_input": query,
        "merge_with_previous": False,
    }


def should_pivot_summary_to_list(
    last_metadata: Dict[str, Any],
    follow_up_type: str,
) -> bool:
    """True when a list-style follow-up follows a summary turn with no list_cache."""
    if follow_up_type not in LIST_CACHE_FOLLOW_UP_TYPES:
        return False
    if last_metadata.get("list_cache"):
        return False
    return last_metadata.get("question_type") == "summary"


def build_list_pivot_parsed(
    last_metadata: Dict[str, Any],
    resource_hints: Dict[str, Any],
    query: str,
    follow_up_type: str,
    requested_limit: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build a list query on __related_list_resource__ after a summary turn."""
    resource = last_metadata.get("resource") or ""
    related = get_related_list_resource(resource, resource_hints)
    if not related:
        return None

    limit = requested_limit
    q_lower = (query or "").lower()
    if limit is None and follow_up_type == "first_n":
        limit = 5
    if limit is None and follow_up_type in ("reference", "refinement", "next_page"):
        if "first" in q_lower:
            limit = 1
        elif "second" in q_lower:
            limit = 2

    return {
        "intent": "read",
        "resource": related,
        "question_type": "list",
        "display_mode": "summary",
        "limit": limit,
        "filters": dict(last_metadata.get("filters") or {}),
        "original_input": query,
        "merge_with_previous": False,
    }


def summary_list_follow_up_refusal_message(
    last_metadata: Dict[str, Any],
    resource_hints: Dict[str, Any],
) -> str:
    """User-facing message when list follow-up cannot pivot from summary."""
    resource = last_metadata.get("resource") or "that metric"
    related = get_related_list_resource(resource, resource_hints)
    if related:
        label = related.replace("-", " ")
        return (
            f"The previous answer was a summary metric for {resource.replace('-', ' ')}, "
            f"not a browsable list. Try 'list {label}' or 'show overdue {label}' instead."
        )
    return (
        "The previous answer was a summary metric with no item list to browse. "
        "Please ask for a list explicitly, such as 'list invoices'."
    )


def extract_last_result_metadata(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Read resource/filters/items from the most recent assistant message metadata."""
    if not conversation_history:
        return {}

    for msg in reversed(conversation_history):
        if msg.get("role") != "assistant":
            continue
        metadata = msg.get("metadata") or {}
        if metadata and (metadata.get("resource") or metadata.get("next_url")):
            return {
                "resource": metadata.get("resource"),
                "intent": metadata.get("intent"),
                "filters": metadata.get("filters", {}),
                "next_url": metadata.get("next_url"),
                "count": metadata.get("count"),
                "has_more": metadata.get("has_more", False),
                "question_type": metadata.get("question_type"),
                "display_mode": metadata.get("display_mode"),
                "result_items": metadata.get("result_items") or [],
                "primary_item": metadata.get("primary_item"),
                "list_cache": metadata.get("list_cache") or [],
                "chat_offset": metadata.get("chat_offset", 0),
                "chat_window_size": metadata.get("chat_window_size"),
                "list_display_fields": metadata.get("list_display_fields") or [],
                "has_more_in_chat": metadata.get("has_more_in_chat", False),
                "total_cached": metadata.get("total_cached"),
                "multiple_resources": metadata.get("multiple_resources"),
                "related_list_resource": metadata.get("related_list_resource"),
                "endpoint_role": metadata.get("endpoint_role"),
            }
    return {}


def try_advance_chat_window(
    last_metadata: Dict[str, Any],
    step_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Advance chat_offset within list_cache (no API call).

    Returns display payload when more rows are available in cache, else None.
    """
    list_cache = last_metadata.get("list_cache") or []
    if not list_cache:
        return None

    current_offset = int(last_metadata.get("chat_offset") or 0)
    window_size = int(
        step_size
        or last_metadata.get("chat_window_size")
        or constants.CHAT_WINDOW_SIZE
    )
    resource = last_metadata.get("resource") or "items"
    fields = last_metadata.get("list_display_fields") or []
    new_offset = current_offset + window_size

    if new_offset >= len(list_cache):
        return None

    window, offset, has_more_in_chat = apply_chat_window(
        list_cache, new_offset, window_size,
    )
    summary = build_chat_summary(
        window,
        fields,
        resource=resource,
        offset=offset,
        window_size=window_size,
        total_cached=len(list_cache),
        total_count=last_metadata.get("count") or len(list_cache),
        has_more_in_chat=has_more_in_chat,
    )
    return {
        "summary": summary,
        "window_items": window,
        "chat_offset": offset,
        "chat_window_size": window_size,
        "has_more_in_chat": has_more_in_chat,
        "list_cache": list_cache,
        "list_display_fields": fields,
        "resource": resource,
        "filters": last_metadata.get("filters", {}),
        "count": last_metadata.get("count"),
    }


def has_prior_context(conversation_history: Optional[List[Dict[str, Any]]]) -> bool:
    return bool(extract_last_result_metadata(conversation_history or []).get("resource"))


def _cache_key(query: str, conversation_history: List[Dict[str, Any]]) -> str:
    tail = conversation_history[-6:] if conversation_history else []
    blob = json.dumps({"q": query.strip(), "h": tail}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def _clean_history_for_prompt(conversation_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Strip internal markers; keep recent turns for the classifier."""
    cleaned = []
    for msg in conversation_history[-6:]:
        content = msg.get("content", "")
        if "\n[Context:" in content:
            content = content.split("\n[Context:")[0].strip()
        cleaned.append({"role": msg.get("role", "user"), "content": content})
    return cleaned


def classify_follow_up(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    resource_hints: Optional[Dict[str, Any]] = None,
    schema_resources: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Use LLM to classify whether query is a follow-up and how to route it.

    Returns:
        {
            "is_follow_up": bool,
            "follow_up_type": standalone|next_page|first_n|last_n|reference|refinement,
            "merge_with_previous": bool,
            "keep_previous_resource": bool,
            "question_type_override": str|None,
            "display_mode_override": str|None,
        }
    """
    default = {
        "is_follow_up": False,
        "follow_up_type": "standalone",
        "merge_with_previous": False,
        "keep_previous_resource": False,
        "question_type_override": None,
        "display_mode_override": None,
        "referent": None,
    }

    if not query or not query.strip():
        return default

    history = conversation_history or []
    key = _cache_key(query, history)
    if key in _classification_cache:
        return _classification_cache[key]

    prior_meta = extract_last_result_metadata(history)
    recent = _clean_history_for_prompt(history)

    prompt = f"""Classify the CURRENT user query in the context of this conversation.

PREVIOUS RESULT METADATA (from last assistant turn):
{json.dumps(prior_meta, indent=2) if prior_meta else "None — no prior structured context"}

PREVIOUS RESULT ITEMS (use for "it"/"this"/"that" pronouns):
{json.dumps(prior_meta.get("result_items") or prior_meta.get("primary_item") or [], indent=2)}

RECENT CONVERSATION:
{json.dumps(recent, indent=2) if recent else "None — first message in session"}

CURRENT QUERY:
{query.strip()}

Decide:
1. Is the current query a FOLLOW-UP that continues or refines the previous turn?
   - Follow-ups refer back to prior results, paginate them, or ask for more detail about them
   - Standalone/reset queries start fresh — do NOT inherit previous filters
   - IMPORTANT: A query on the SAME resource but with DIFFERENT/NEW filters is STANDALONE, not a follow-up
     Example: After "new service orders assigned to me" → "low priority service orders" is STANDALONE (different filter scope)
   - Follow-ups must reference the PREVIOUS RESULTS specifically ("those", "them", "which of them", "more", "next")
2. If follow-up, what type?
   - reset: user wants a fresh unfiltered list on a resource — clears prior filters
   - next_page: user wants more/paginated results from the prior list
   - first_n / last_n: user wants a specific slice (first N, last N)
   - reference: user wants to see/enumerate items from the prior scoped result set
   - refinement: user asks a detail or subset question about the prior result(s) without changing topic/resource
   - standalone: new independent query — includes same resource with DIFFERENT filters
3. merge_with_previous=true ONLY for next_page, first_n, last_n, reference, refinement — NEVER for reset or standalone
4. CRITICAL — Standalone detection:
   - Query mentions different filter values than previous (e.g., "low priority" vs prior "new status") → STANDALONE
   - Query uses generic quantifiers ("all", "any", "which") without pronouns referencing prior results → STANDALONE
   - Query introduces new filter criteria not present in prior context → STANDALONE
5. Use prior question_type from PREVIOUS RESULT METADATA:
   - After question_type=count: if the user now wants to see/name/subset those items, set question_type_override="list"
   - After question_type=list: pagination/subset/detail queries stay follow-ups when scope is unchanged
6. If the query refers to a specific prior item via pronoun/deixis, set referent from PREVIOUS RESULT ITEMS (id, resource)
7. Decide follow-up vs standalone from meaning, not resource name — same resource + different filters = standalone
8. RESOURCE SWITCH → STANDALONE:
   - Query explicitly names a different resource than previous (e.g. prior users → "service orders") → STANDALONE
   - Prior aggregate count (multiple_resources) and query names ONE child type (e.g. "how many detailed reports?") → STANDALONE
   - Vague deixis only ("those", "them") without naming a resource → may stay follow-up

Return JSON only:
{{
  "is_follow_up": boolean,
  "follow_up_type": "standalone" | "reset" | "next_page" | "first_n" | "last_n" | "reference" | "refinement",
  "merge_with_previous": boolean,
  "keep_previous_resource": boolean,
  "question_type_override": "count" | "list" | "details" | null,
  "display_mode_override": "summary" | "detailed" | "full" | null,
  "referent": null | {{"resource": "resource_name", "id": number, "id_field": "id", "label": "optional display id"}}
}}"""

    try:
        client = get_openai_client()
        result = client.parse_json_response(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify conversational follow-ups for an API query assistant. "
                        "Return only valid JSON. No hardcoded assumptions about domain entities."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=DETERMINISTIC_TEMP,
        )
        if not isinstance(result, dict):
            result = default
        else:
            result = {**default, **result}
            if result.get("follow_up_type") not in FOLLOW_UP_TYPES:
                result["follow_up_type"] = "standalone"
            if result.get("follow_up_type") in NO_MERGE_TYPES:
                result["is_follow_up"] = False
                result["merge_with_previous"] = False
                result["keep_previous_resource"] = False
            referent = result.get("referent")
            if (
                isinstance(referent, dict)
                and referent.get("id") is not None
            ):
                result["is_follow_up"] = True
                result["merge_with_previous"] = False
                if result.get("follow_up_type") in NO_MERGE_TYPES:
                    result["follow_up_type"] = "refinement"
                if not result.get("question_type_override"):
                    result["question_type_override"] = "details"
                if not result.get("display_mode_override"):
                    result["display_mode_override"] = "detailed"
    except Exception as exc:
        logger.warning("Follow-up LLM classification failed: %s — treating as standalone", exc)
        result = default

    if prior_meta and resource_hints is not None:
        resources = schema_resources or set()
        if should_force_standalone_for_resource_switch(
            query.strip(), prior_meta, resource_hints, resources,
        ):
            logger.info(
                "Forcing standalone: query names different resource scope than prior turn",
            )
            result = dict(default)

    logger.info(
        "Follow-up classification: is_follow_up=%s type=%s merge=%s keep_resource=%s referent=%s",
        result.get("is_follow_up"),
        result.get("follow_up_type"),
        result.get("merge_with_previous"),
        result.get("keep_previous_resource"),
        (result.get("referent") or {}).get("id"),
    )

    _classification_cache[key] = result
    return result


def is_follow_up_query(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Return True when the LLM classifies the query as a follow-up."""
    return classify_follow_up(query, conversation_history).get("is_follow_up", False)


def get_follow_up_type(query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Return the LLM-classified follow-up type."""
    return classify_follow_up(query, conversation_history).get("follow_up_type", "standalone")


def _filter_values_equal(a: Any, b: Any) -> bool:
    """Compare filter values whether raw or {operator, value} dicts."""
    val_a = a.get("value") if isinstance(a, dict) and "value" in a else a
    val_b = b.get("value") if isinstance(b, dict) and "value" in b else b
    return val_a == val_b


def strip_inherited_session_filters(
    parsed: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]],
    user_context: Optional[Dict[str, Any]] = None,
    resource_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Remove filters copied from the previous turn (reset / fresh list queries)."""
    result = dict(parsed)
    meta = extract_last_result_metadata(conversation_history or [])
    prev_filters = meta.get("filters") or {}
    filters = dict(result.get("filters") or {})

    for field, prev_val in prev_filters.items():
        if field in filters and _filter_values_equal(filters[field], prev_val):
            del filters[field]

    # Remove user-scoped filters that match current user (schema-driven, not hardcoded)
    user_id = (user_context or {}).get("user_id")
    if user_id is not None and resource_hints:
        resource = result.get("resource") or meta.get("resource") or ""
        user_scoped_fields = get_user_scoped_fields(
            resource, resource_hints, user_context,
        )
        for field in user_scoped_fields:
            if field in filters:
                field_val = filters[field]
                val = field_val.get("value") if isinstance(field_val, dict) else field_val
                if val == user_id:
                    del filters[field]

    result["filters"] = filters
    result["merge_with_previous"] = False
    return result


def apply_referent_context(
    parsed: Dict[str, Any],
    classification: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply LLM-resolved item referent (it/this/that) as an id filter + detail fetch."""
    if not isinstance(parsed, dict) or not classification:
        return parsed

    referent = classification.get("referent")
    if not referent or not isinstance(referent, dict) or referent.get("id") is None:
        return parsed

    result = dict(parsed)
    id_field = referent.get("id_field") or "id"
    result["filters"] = dict(result.get("filters") or {})
    result["filters"][id_field] = {
        "operator": "equals",
        "value": referent["id"],
    }
    if referent.get("resource"):
        result["resource"] = referent["resource"]
    result["question_type"] = classification.get("question_type_override") or "details"
    result["display_mode"] = classification.get("display_mode_override") or "detailed"
    result["limit"] = 1
    result["merge_with_previous"] = False
    result["_referent"] = referent
    logger.info(
        "Applied referent context: resource=%s %s=%s",
        result.get("resource"),
        id_field,
        referent["id"],
    )
    return result


def should_merge_previous_filters(
    parsed: Dict[str, Any],
    classification: Optional[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, Any]]],
) -> bool:
    """
    Return True only when the follow-up classifier explicitly allows merging.

    Reset/standalone queries (e.g. "show all service orders") must not inherit filters.
    """
    if not conversation_history or not has_prior_context(conversation_history):
        return False

    clf = classification or {}
    if clf.get("follow_up_type") in NO_MERGE_TYPES:
        return False
    if clf.get("merge_with_previous") is False:
        return False
    if clf.get("is_follow_up") is False and not parsed.get("merge_with_previous"):
        return False

    return bool(
        clf.get("merge_with_previous") is True
        or (
            parsed.get("merge_with_previous") is True
            and clf.get("is_follow_up") is True
        )
    )


def apply_summary_metric_follow_up(
    parsed: Dict[str, Any],
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    resource_hints: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    When the prior turn was a summary dashboard, route metric follow-ups to one field.
    """
    if not isinstance(parsed, dict) or not query:
        return parsed

    meta = extract_last_result_metadata(conversation_history or [])
    prev_resource = meta.get("resource")
    if not prev_resource or meta.get("question_type") != "summary":
        return parsed

    if parsed.get("question_type") == "list":
        return parsed

    role = get_endpoint_role(prev_resource, resource_hints)
    if role not in ("summary", "dashboard", "metrics"):
        return parsed

    schema_resource = (schema or {}).get("resources", {}).get(prev_resource)
    field = resolve_summary_field_from_query(
        query, prev_resource, resource_hints, parsed, schema_resource,
    )
    if not field:
        return parsed

    result = dict(parsed)
    result["resource"] = prev_resource
    result["question_type"] = "summary"
    result["summary_field"] = field
    result["merge_with_previous"] = False
    return result


def apply_follow_up_context(
    parsed: Dict[str, Any],
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    is_follow_up: bool = False,
    classification: Optional[Dict[str, Any]] = None,
    user_context: Optional[Dict[str, Any]] = None,
    resource_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Anchor parsed query to previous resource/filters when LLM says to keep context.
    """
    if not isinstance(parsed, dict):
        return parsed

    clf = classification or classify_follow_up(query, conversation_history)

    # Item pronoun referent (it/this/that) — fetch specific prior record
    if clf.get("referent") and isinstance(clf.get("referent"), dict):
        return apply_referent_context(parsed, clf)

    # Reset/standalone: fresh query — strip inherited session filters
    if clf.get("follow_up_type") in NO_MERGE_TYPES:
        return strip_inherited_session_filters(
            parsed, conversation_history, user_context, resource_hints
        )

    if not clf.get("keep_previous_resource") and not clf.get("merge_with_previous"):
        return parsed

    if not clf.get("is_follow_up") and not parsed.get("merge_with_previous"):
        return parsed

    meta = extract_last_result_metadata(conversation_history or [])
    prev_resource = meta.get("resource")
    if not prev_resource:
        return parsed

    result = dict(parsed)
    result["resource"] = prev_resource
    result["merge_with_previous"] = True

    prev_filters = meta.get("filters") or {}
    current_filters = result.get("filters") or {}
    merged = dict(prev_filters)
    for key, value in current_filters.items():
        merged[key] = value
    result["filters"] = merged

    q_override = clf.get("question_type_override")
    d_override = clf.get("display_mode_override")
    if q_override:
        result["question_type"] = q_override
    if d_override:
        result["display_mode"] = d_override
    elif clf.get("follow_up_type") == "refinement" and not q_override:
        result["question_type"] = result.get("question_type") or "details"
        result["display_mode"] = result.get("display_mode") or "detailed"

    if meta.get("count") == 1 and not result.get("limit"):
        result["limit"] = 1

    result = expand_aggregate_follow_up(result, meta, clf, resource_hints or {})

    if should_pivot_summary_to_list(meta, clf.get("follow_up_type") or ""):
        related = get_related_list_resource(prev_resource, resource_hints or {})
        if related:
            result["resource"] = related
            result["question_type"] = "list"
            result["merge_with_previous"] = False
            result.pop("summary_field", None)

    return result
