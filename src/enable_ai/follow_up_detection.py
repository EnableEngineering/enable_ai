"""
LLM-based follow-up query detection and context anchoring.

No hardcoded phrase lists — an LLM classifies whether the current query
continues or refines the previous conversation turn.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP

logger = setup_logger("enable_ai.follow_up_detection")

# Cache classifications within a process to avoid duplicate LLM calls
_classification_cache: Dict[str, Dict[str, Any]] = {}


def clear_classification_cache() -> None:
    """Clear in-process classification cache (for tests)."""
    _classification_cache.clear()

FOLLOW_UP_TYPES = (
    "standalone",
    "next_page",
    "first_n",
    "last_n",
    "reference",
    "refinement",
)


def extract_last_result_metadata(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Read resource/filters from the most recent assistant message metadata."""
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
            }
    return {}


def has_prior_context(conversation_history: Optional[List[Dict[str, Any]]]) -> bool:
    return bool(extract_last_result_metadata(conversation_history or {}).get("resource"))


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

RECENT CONVERSATION:
{json.dumps(recent, indent=2) if recent else "None — first message in session"}

CURRENT QUERY:
{query.strip()}

Decide:
1. Is the current query a FOLLOW-UP that continues or refines the previous turn?
   - Follow-ups refer back to prior results, paginate them, or ask for more detail about them
   - Standalone queries start a new topic or restate a full request independently
2. If follow-up, what type?
   - next_page: user wants more/paginated results from the prior list
   - first_n / last_n: user wants a specific slice (first N, last N)
   - reference: user refers to prior items ("those", "them") wanting to see them
   - refinement: user asks a detail question about the prior result(s) without changing topic
     (e.g. after a count, asking which company/customer/technician those items belong to)
3. Should previous filters be merged? Should the same resource be kept?
4. If refinement after a count, override question_type to "details" and display_mode to "detailed"

Return JSON only:
{{
  "is_follow_up": boolean,
  "follow_up_type": "standalone" | "next_page" | "first_n" | "last_n" | "reference" | "refinement",
  "merge_with_previous": boolean,
  "keep_previous_resource": boolean,
  "question_type_override": "count" | "list" | "details" | null,
  "display_mode_override": "summary" | "detailed" | "full" | null
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
            if result["follow_up_type"] == "standalone":
                result["is_follow_up"] = False
    except Exception as exc:
        logger.warning("Follow-up LLM classification failed: %s — treating as standalone", exc)
        result = default

    logger.info(
        "Follow-up classification: is_follow_up=%s type=%s merge=%s keep_resource=%s",
        result.get("is_follow_up"),
        result.get("follow_up_type"),
        result.get("merge_with_previous"),
        result.get("keep_previous_resource"),
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


def apply_follow_up_context(
    parsed: Dict[str, Any],
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    is_follow_up: bool = False,
    classification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Anchor parsed query to previous resource/filters when LLM says to keep context.
    """
    if not is_follow_up or not isinstance(parsed, dict):
        return parsed

    clf = classification or classify_follow_up(query, conversation_history)
    if not clf.get("keep_previous_resource") and not clf.get("merge_with_previous"):
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
    elif clf.get("follow_up_type") == "refinement":
        result["question_type"] = result.get("question_type") or "details"
        result["display_mode"] = result.get("display_mode") or "detailed"

    if meta.get("count") == 1 and not result.get("limit"):
        result["limit"] = 1

    return result
