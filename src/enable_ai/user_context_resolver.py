"""
Resolve user-context placeholders in parsed queries (filters, entities).

When callers pass user_context with user_id, placeholders like __current_user_id__
are replaced with the real ID instead of triggering FK lookup API calls.
"""

from typing import Any, Dict, Optional, Set

from .utils import setup_logger

logger = setup_logger("enable_ai.user_context_resolver")

PRONOUN_MARKERS = ("assigned to me", " my ", " me ", "mine", " my,", "for me")

USER_ID_PLACEHOLDERS: Set[str] = {
    "__current_user_id__",
    "{{current_user_id}}",
    "{current_user_id}",
    "{user_id}",
    "me",
    "my",
    "mine",
}

COMPANY_ID_PLACEHOLDERS: Set[str] = {
    "__current_company_id__",
    "{{current_company_id}}",
    "{current_company_id}",
    "{company_id}",
    "my company",
}

USER_ID_FIELDS: Set[str] = {
    "technician",
    "assigned_to",
    "created_by",
    "updated_by",
    "owner",
    "assignee",
    "user",
    "user_id",
}

COMPANY_ID_FIELDS: Set[str] = {
    "company",
    "company_id",
    "customer",
    "customer_id",
}


def _norm_placeholder(value: Any) -> str:
    return str(value).strip().lower()


def is_user_id_placeholder(value: Any) -> bool:
    if value is None:
        return False
    return _norm_placeholder(value) in USER_ID_PLACEHOLDERS


def is_company_id_placeholder(value: Any) -> bool:
    if value is None:
        return False
    return _norm_placeholder(value) in COMPANY_ID_PLACEHOLDERS


def _resolve_value(
    field: str,
    value: Any,
    user_context: Dict[str, Any],
) -> Any:
    field_lower = field.lower().replace("-", "_")
    base = field_lower.split("__")[0]

    if (field_lower in USER_ID_FIELDS or base in USER_ID_FIELDS) and is_user_id_placeholder(value):
        uid = user_context.get("user_id")
        if uid is not None:
            return uid

    if (field_lower in COMPANY_ID_FIELDS or base in COMPANY_ID_FIELDS) and is_company_id_placeholder(value):
        cid = user_context.get("company_id")
        if cid is not None:
            return cid

    return value


def _resolve_filter_entry(
    field: str,
    fval: Any,
    user_context: Dict[str, Any],
) -> Any:
    if isinstance(fval, dict):
        value = fval.get("value")
        resolved = _resolve_value(field, value, user_context)
        if resolved != value:
            return {**fval, "value": resolved}
        return fval
    resolved = _resolve_value(field, fval, user_context)
    return resolved if resolved != fval else fval


def _query_needs_user_filter(query: str) -> bool:
    q_lower = (query or "").lower()
    return any(m in q_lower for m in PRONOUN_MARKERS)


def resolve_user_context_in_parsed(
    parsed: Dict[str, Any],
    user_context: Optional[Dict[str, Any]],
    query: str = "",
    follow_up_classification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Replace user/company placeholders with IDs from user_context.

    Also injects technician=user_id when the query uses pronouns but no user filter exists.
    Skips injection on reset/standalone queries and item-referent follow-ups.
    """
    if not isinstance(parsed, dict) or not user_context:
        return parsed

    user_id = user_context.get("user_id")
    if user_id is None:
        return parsed

    clf = follow_up_classification or {}
    if clf.get("follow_up_type") in ("standalone", "reset"):
        skip_injection = True
    elif clf.get("referent"):
        skip_injection = True
    elif parsed.get("_referent"):
        skip_injection = True
    else:
        skip_injection = False

    result = dict(parsed)
    filters = dict(result.get("filters") or {})
    entities = dict(result.get("entities") or {})

    for field, fval in list(filters.items()):
        new_val = _resolve_filter_entry(field, fval, user_context)
        if new_val != fval:
            filters[field] = new_val
            resolved = new_val.get("value") if isinstance(new_val, dict) else new_val
            logger.info(
                "Resolved user-context placeholder: %s=%r (user_id=%s)",
                field, resolved, user_id,
            )

    for field, value in list(entities.items()):
        resolved = _resolve_value(field, value, user_context)
        if resolved != value:
            entities[field] = resolved
            logger.info(
                "Resolved user-context entity placeholder: %s=%r (user_id=%s)",
                field, resolved, user_id,
            )

    # Inject technician filter for pronoun queries when not already set
    if not skip_injection and _query_needs_user_filter(query):
        user_field = "technician"
        if user_field not in filters and user_field not in entities:
            filters[user_field] = {"operator": "equals", "value": user_id}
            logger.info(
                "Injected %s=%s from user_context for pronoun query",
                user_field, user_id,
            )

    result["filters"] = filters
    result["entities"] = entities
    return result
