"""
Detect conversational follow-up queries that continue a previous turn.

Follow-ups include pagination ("show me more"), references ("which are those"),
and refinements ("and assigned to which company?").
"""

import re
from typing import Any, Dict, List, Optional

# Pagination / reference patterns (work even without parsing prior metadata)
PAGINATION_PATTERNS = (
    "next page",
    "previous page",
    "show me more",
    "show more",
    "continue",
)

REFERENCE_PATTERNS = (
    "which are those",
    "what are those",
    "which ones",
    "show them",
    "list them",
    "the same",
    "of them",
    "of those",
    "from those",
)

REFINEMENT_PATTERNS = (
    "which company",
    "what company",
    "which customer",
    "what customer",
    "assigned to",
    "belongs to",
    "who is",
    "who are",
    "what is the",
    "what are the",
    "tell me more",
    "more details",
    "more about",
    "details for",
    "details of",
    "what about the",
    "for that one",
    "for that",
    "about it",
    "about that",
    "the company",
    "the customer",
    "the technician",
    "the status",
)

CONTINUATION_STARTERS = (
    "and ",
    "also ",
    "what about ",
    "how about ",
    "ok and ",
    "okay and ",
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


def is_follow_up_query(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Return True when the query continues or refines a previous turn.

    Pagination/reference patterns are detected without prior context.
    Refinement/continuation patterns require conversation history.
    """
    if not query or not query.strip():
        return False

    query_lower = query.lower().strip()
    history = conversation_history or []
    has_context = has_prior_context(history)

    # Pagination
    if any(p in query_lower for p in PAGINATION_PATTERNS):
        return True
    for token in ("next", "more", "previous", "first"):
        if token in query_lower:
            return True
    if re.search(r"\b(next|first|last|show me)\s+\d+\b", query_lower):
        return True

    # Explicit references to prior results
    if any(p in query_lower for p in REFERENCE_PATTERNS):
        return True

    if not has_context:
        return False

    # Conversational continuations need prior context
    if any(query_lower.startswith(s) for s in CONTINUATION_STARTERS):
        return True

    if any(p in query_lower for p in REFINEMENT_PATTERNS):
        return True

    # Short questions after a prior turn are usually follow-ups
    if "?" in query_lower and len(query_lower.split()) <= 10:
        return True

    return False


def get_follow_up_type(query: str) -> str:
    """
    Classify follow-up intent for routing.

    Returns: next_page | first_n | last_n | reference | refinement | unknown
    """
    query_lower = (query or "").lower().strip()

    if any(p in query_lower for p in ("next page", "next", "more", "continue", "show more")):
        return "next_page"

    if re.search(r"\b(first|show me first|the first)\s*\d*\b", query_lower):
        return "first_n"

    if re.search(r"\b(last|show me last|the last)\s*\d*\b", query_lower):
        return "last_n"

    if any(p in query_lower for p in ("those", "them", "these", "the same")):
        return "reference"

    if (
        any(query_lower.startswith(s) for s in CONTINUATION_STARTERS)
        or any(p in query_lower for p in REFINEMENT_PATTERNS)
    ):
        return "refinement"

    return "unknown"


def apply_follow_up_context(
    parsed: Dict[str, Any],
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    is_follow_up: bool = False,
) -> Dict[str, Any]:
    """
    When a follow-up is detected, anchor parsing to the previous resource/filters.

    Prevents refinements like "and which company?" from switching to a different
    resource (e.g. listing all companies).
    """
    if not is_follow_up or not isinstance(parsed, dict):
        return parsed

    meta = extract_last_result_metadata(conversation_history or [])
    prev_resource = meta.get("resource")
    if not prev_resource:
        return parsed

    follow_type = get_follow_up_type(query)
    if follow_type not in ("refinement", "reference", "unknown"):
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

    if follow_type == "refinement":
        result["question_type"] = "details"
        result["display_mode"] = "detailed"
        if meta.get("count") == 1 and not result.get("limit"):
            result["limit"] = 1

    return result
