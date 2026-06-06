"""
Resolve user-context placeholders in parsed queries (filters, entities).

When callers pass user_context with user_id, placeholders like __current_user_id__
are replaced with the real ID instead of triggering FK lookup API calls.
"""

from typing import Any, Dict, Optional, Set

from .hint_utils import get_user_scoped_fields, query_implies_user_scope
from .utils import setup_logger

logger = setup_logger("enable_ai.user_context_resolver")

# Legacy export for param_validator; prefer query_implies_user_scope()
PRONOUN_MARKERS = ("assigned to me", " my ", "for me", "mine")

USER_ID_PLACEHOLDERS: Set[str] = {
    "__current_user_id__",
    "current_user_id",
    "{{current_user_id}}",
    "{current_user_id}",
    "{user_id}",
    "me",
    "my",
    "mine",
}

COMPANY_ID_PLACEHOLDERS: Set[str] = {
    "__current_company_id__",
    "current_company_id",
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


def normalize_user_context(user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize caller user_context to canonical user_id / company_id keys.

    Accepts common aliases (id, userId, nested user.id) from host applications.
    """
    if not user_context:
        return {}

    normalized = dict(user_context)
    if normalized.get("user_id") is None:
        for key in ("id", "userId", "user_id", "pk"):
            if normalized.get(key) is not None:
                normalized["user_id"] = normalized[key]
                break
        if normalized.get("user_id") is None:
            nested = normalized.get("user")
            if isinstance(nested, dict):
                for key in ("id", "user_id", "pk"):
                    if nested.get(key) is not None:
                        normalized["user_id"] = nested[key]
                        break

    if normalized.get("company_id") is None:
        for key in ("companyId", "company_id"):
            if normalized.get(key) is not None:
                normalized["company_id"] = normalized[key]
                break
        if normalized.get("company_id") is None:
            nested = normalized.get("company")
            if isinstance(nested, dict) and nested.get("id") is not None:
                normalized["company_id"] = nested["id"]
            elif isinstance(nested, (int, str)) and str(nested).isdigit():
                normalized["company_id"] = int(nested)

    return normalized


def _resolve_value(
    field: str,
    value: Any,
    user_context: Dict[str, Any],
) -> Any:
    # Resolve by placeholder token first (any field)
    if is_user_id_placeholder(value):
        uid = user_context.get("user_id")
        if uid is not None:
            return uid

    if is_company_id_placeholder(value):
        cid = user_context.get("company_id")
        if cid is not None:
            return cid

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


def _filter_display_value(val: Any) -> Any:
    if isinstance(val, dict):
        return val.get("value", val)
    return val


def _sync_entities_to_filters(filters: Dict[str, Any], entities: Dict[str, Any]) -> Dict[str, Any]:
    """Keep filters in sync with resolved entity values for display and execution."""
    synced = dict(filters)
    for field, entity_val in entities.items():
        if field not in synced:
            continue
        fval = synced[field]
        if isinstance(fval, dict):
            synced[field] = {**fval, "value": entity_val}
        else:
            synced[field] = entity_val
    return synced


def filters_for_display(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return filters with resolved values for user-facing messages.

    Prefers resolved entities over placeholder values still in filters.
    """
    if not isinstance(parsed, dict):
        return {}
    filters = dict(parsed.get("filters") or {})
    entities = parsed.get("entities") or {}
    for field, entity_val in entities.items():
        if field in filters:
            fval = filters[field]
            display = _filter_display_value(fval)
            if is_user_id_placeholder(display) or is_company_id_placeholder(display):
                if isinstance(fval, dict):
                    filters[field] = {**fval, "value": entity_val}
                else:
                    filters[field] = entity_val
        elif entity_val is not None:
            filters[field] = entity_val
    return filters


def resolve_params_dict(
    params: Optional[Dict[str, Any]],
    user_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve placeholders in flat API query params (last-mile before HTTP call)."""
    if not params or not user_context:
        return dict(params or {})

    ctx = normalize_user_context(user_context)
    if ctx.get("user_id") is None and ctx.get("company_id") is None:
        return dict(params)

    resolved: Dict[str, Any] = {}
    for field, value in params.items():
        if isinstance(value, dict) and "value" in value:
            resolved[field] = _resolve_filter_entry(field, value, ctx)
        else:
            resolved[field] = _resolve_value(field, value, ctx)
    return resolved


def resolve_user_context_placeholders(
    parsed: Dict[str, Any],
    user_context: Optional[Dict[str, Any]],
    query: str = "",
    follow_up_classification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Public alias for resolving placeholders in parsed query dicts."""
    return resolve_user_context_in_parsed(
        parsed,
        user_context,
        query=query,
        follow_up_classification=follow_up_classification,
    )


def resolve_user_context_in_parsed(
    parsed: Dict[str, Any],
    user_context: Optional[Dict[str, Any]],
    query: str = "",
    follow_up_classification: Optional[Dict[str, Any]] = None,
    resource_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Replace user/company placeholders with IDs from user_context.

    Injects user-scoped filters only when the query implies user ownership
    (not imperative "show me") and the resource defines __user_scoped_fields__.
    Skips injection on reset/standalone queries and item-referent follow-ups.
    """
    if not isinstance(parsed, dict) or not user_context:
        return parsed

    ctx = normalize_user_context(user_context)
    user_id = ctx.get("user_id")
    if user_id is None and ctx.get("company_id") is None:
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
        new_val = _resolve_filter_entry(field, fval, ctx)
        if new_val != fval:
            filters[field] = new_val
            resolved = new_val.get("value") if isinstance(new_val, dict) else new_val
            logger.info(
                "Resolved user-context placeholder: %s=%r (user_id=%s)",
                field, resolved, user_id,
            )

    for field, value in list(entities.items()):
        resolved = _resolve_value(field, value, ctx)
        if resolved != value:
            entities[field] = resolved
            logger.info(
                "Resolved user-context entity placeholder: %s=%r (user_id=%s)",
                field, resolved, user_id,
            )

    # Inject user-scoped filters only for resources that declare them
    resource = result.get("resource", "")
    scoped_fields = get_user_scoped_fields(resource, resource_hints or {})
    if (
        not skip_injection
        and user_id is not None
        and scoped_fields
        and query_implies_user_scope(query)
    ):
        for user_field in scoped_fields:
            if user_field not in filters and user_field not in entities:
                filters[user_field] = {"operator": "equals", "value": user_id}
                entities[user_field] = user_id
                logger.info(
                    "Injected %s=%s from user_context for user-scoped query (resource=%s)",
                    user_field, user_id, resource,
                )

    filters = _sync_entities_to_filters(filters, entities)
    result["filters"] = filters
    result["entities"] = entities
    return result
