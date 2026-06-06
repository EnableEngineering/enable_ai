"""
Schema-driven helpers for resource_hints (user scope, aggregates, multi-resource).
"""

import re
from typing import Any, Dict, List, Optional, Set

# Imperative phrases where "me" is not a user-scope pronoun
_IMPERATIVE_ME = re.compile(
    r"\b(?:show|give|tell|let|help|bring)\s+me\b",
    re.IGNORECASE,
)

_BREADTH_PATTERN = re.compile(r"\b(?:all|every|any|entire)\b", re.IGNORECASE)

_USER_SCOPE_PHRASES = (
    "assigned to me",
    "belonging to me",
    "created by me",
    "owned by me",
    "reported by me",
    "submitted by me",
    "for me",
    " to me",
    " mine",
    "mine ",
)


def get_user_scoped_fields(resource: str, resource_hints: Dict[str, Any]) -> List[str]:
    """Return field names from resource_hints.__user_scoped_fields__ for a resource."""
    if not resource or not resource_hints:
        return []
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return []
    fields = hints.get("__user_scoped_fields__") or []
    return list(fields) if isinstance(fields, (list, tuple)) else []


def query_implies_breadth(query: str) -> bool:
    """True when the query asks for an unqualified broad set (all, every, any)."""
    return bool(_BREADTH_PATTERN.search(query or ""))


def _has_explicit_user_scope_markers(query: str) -> bool:
    q = (query or "").lower()
    if not q.strip():
        return False
    for phrase in _USER_SCOPE_PHRASES:
        if phrase in q:
            return True
    if re.search(r"\bmy\s+\w", q):
        return True
    stripped = _IMPERATIVE_ME.sub("", q)
    return bool(re.search(r"(?<!\w)me(?!\w)", stripped))


def query_implies_user_scope(query: str) -> bool:
    """
    True when the query asks for user-owned/assigned data.

    Excludes imperative phrases like "show me", "give me", "tell me".
    Breadth tokens (all/every/any) without explicit user markers are not user-scoped.
    """
    if not (query or "").strip():
        return False
    if query_implies_breadth(query) and not _has_explicit_user_scope_markers(query):
        return False
    return _has_explicit_user_scope_markers(query)


def get_embedded_fields(resource: str, resource_hints: Dict[str, Any]) -> List[str]:
    """Return nested field names from resource_hints.__embedded_fields__."""
    if not resource or not resource_hints:
        return []
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return []
    fields = hints.get("__embedded_fields__") or []
    return list(fields) if isinstance(fields, (list, tuple)) else []


def get_extra_query_params(resource: str, resource_hints: Dict[str, Any]) -> Set[str]:
    """Params supported by the API but missing from OpenAPI (declared in hints)."""
    if not resource or not resource_hints:
        return set()
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return set()
    extra = hints.get("__extra_query_params__") or []
    return {str(p) for p in extra} if isinstance(extra, (list, tuple)) else set()


def get_client_side_filter_fields(resource: str, resource_hints: Dict[str, Any]) -> Set[str]:
    """Filters intentionally applied client-side (no warning)."""
    if not resource or not resource_hints:
        return set()
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return set()
    fields = hints.get("__client_side_filters__") or []
    return {str(f) for f in fields} if isinstance(fields, (list, tuple)) else set()


def strip_user_scoped_filters_on_breadth(
    parsed: Dict[str, Any],
    query: str,
    resource_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """Remove user-scoped filters when the user asked for an unqualified broad list."""
    if not isinstance(parsed, dict):
        return parsed
    if query_implies_user_scope(query):
        return parsed

    resource = parsed.get("resource", "")
    scoped = get_user_scoped_fields(resource, resource_hints or {})
    if not scoped:
        return parsed

    result = dict(parsed)
    filters = dict(result.get("filters") or {})
    entities = dict(result.get("entities") or {})
    for field in scoped:
        filters.pop(field, None)
        entities.pop(field, None)
    result["filters"] = filters
    result["entities"] = entities
    return result


def _normalize_term(term: str) -> str:
    return (term or "").lower().replace("-", " ").strip()


def _term_in_query(term: str, query_lower: str) -> bool:
    norm = _normalize_term(term)
    if not norm:
        return False
    if " " in norm:
        return norm in query_lower
    return bool(re.search(rf"(?<!\w){re.escape(norm)}(?:s|es)?(?!\w)", query_lower))


def get_aggregate_resources(hints: Dict[str, Any]) -> Optional[List[str]]:
    """
    Read child resource list from __aggregate_resources__ or legacy __aggregate__.
    """
    if not isinstance(hints, dict):
        return None
    for key in ("__aggregate_resources__", "__aggregate__"):
        agg = hints.get(key)
        if isinstance(agg, (list, tuple)) and len(agg) >= 2:
            return list(agg)
    return None


def is_virtual_aggregate_resource(resource: str, resource_hints: Dict[str, Any]) -> bool:
    """True when a hint key defines aggregate children (virtual parent resource)."""
    if not resource or not resource_hints:
        return False
    hints = resource_hints.get(resource) or {}
    return get_aggregate_resources(hints) is not None


def find_aggregate_resources_for_query(
    query: str,
    resource_hints: Dict[str, Any],
) -> Optional[List[str]]:
    """
    Match virtual hint entries with __aggregate_resources__ and __resource_synonyms__.
    """
    if not query or not resource_hints:
        return None
    q = query.lower()
    best: Optional[List[str]] = None
    best_len = 0

    for name, hints in resource_hints.items():
        if not isinstance(hints, dict):
            continue
        agg = get_aggregate_resources(hints)
        if not agg:
            continue
        terms = [_normalize_term(name)] + [
            _normalize_term(s) for s in (hints.get("__resource_synonyms__") or [])
        ]
        for term in terms:
            if term and _term_in_query(term, q) and len(term) > best_len:
                best = list(agg)
                best_len = len(term)
    return best


def find_mentioned_resources(
    query: str,
    resource_hints: Dict[str, Any],
    schema_resources: Set[str],
) -> List[str]:
    """Find schema resources whose name or synonym appears in the query."""
    if not query or not schema_resources:
        return []
    q = query.lower()
    found: List[str] = []
    for name in sorted(schema_resources):
        rh = resource_hints.get(name, {}) if isinstance(resource_hints, dict) else {}
        if not isinstance(rh, dict):
            rh = {}
        terms = [name, name.replace("-", " ")]
        syns = rh.get("__resource_synonyms__") or []
        if isinstance(syns, list):
            terms.extend(str(s) for s in syns)
        elif syns:
            terms.append(str(syns))
        if any(_term_in_query(t, q) for t in terms if t):
            found.append(name)
    return found


def expand_query_resources(
    parsed: Dict[str, Any],
    query: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Set multiple_resources from schema hints when the query implies several resources.
    """
    if not isinstance(parsed, dict):
        return parsed

    result = dict(parsed)
    hints = (schema or {}).get("resource_hints") or {}
    resources = set((schema or {}).get("resources") or {})

    existing = result.get("multiple_resources")
    if isinstance(existing, list) and len(existing) > 1:
        return result

    agg = find_aggregate_resources_for_query(query, hints)
    if agg:
        result["multiple_resources"] = agg
        if not result.get("resource"):
            result["resource"] = agg[0]
        return result

    q_lower = (query or "").lower()
    if " and " not in q_lower and " & " not in q_lower:
        return result

    mentioned = find_mentioned_resources(query, hints, resources)
    if len(mentioned) > 1:
        result["multiple_resources"] = mentioned
        if not result.get("resource"):
            result["resource"] = mentioned[0]

    return result


def _is_aggregate_list_follow_up(
    parsed: Dict[str, Any],
    meta: Dict[str, Any],
    classification: Optional[Dict[str, Any]],
) -> bool:
    """True when a follow-up should list all children of a virtual aggregate resource."""
    clf = classification or {}
    prev_qt = meta.get("question_type")
    new_qt = clf.get("question_type_override") or parsed.get("question_type")

    if clf.get("follow_up_type") == "reference":
        return new_qt in (None, "list") or clf.get("question_type_override") == "list"

    if prev_qt == "count" and new_qt == "list":
        return True

    if prev_qt == "count" and clf.get("is_follow_up") and clf.get("follow_up_type") in (
        "reference", "refinement", "first_n",
    ):
        if clf.get("question_type_override") != "details":
            return True

    return bool(meta.get("multiple_resources") and new_qt == "list")


def expand_aggregate_follow_up(
    parsed: Dict[str, Any],
    meta: Dict[str, Any],
    classification: Optional[Dict[str, Any]],
    resource_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Expand vague deixis follow-ups ("show me those") after aggregate count queries.

    Uses previous_resource __aggregate_resources__ or session multiple_resources.
    """
    if not isinstance(parsed, dict):
        return parsed

    result = dict(parsed)
    if not _is_aggregate_list_follow_up(result, meta, classification):
        return result

    prev_multi = meta.get("multiple_resources")
    if isinstance(prev_multi, list) and len(prev_multi) > 1:
        result["multiple_resources"] = list(prev_multi)
        result["resource"] = prev_multi[0]
        return result

    prev_resource = meta.get("resource") or result.get("resource")
    if not prev_resource:
        return result

    hints = (resource_hints or {}).get(prev_resource) or {}
    agg = get_aggregate_resources(hints)
    if agg:
        result["multiple_resources"] = agg
        result["resource"] = agg[0]

    return result
