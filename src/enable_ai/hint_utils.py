"""
Schema-driven helpers for resource_hints (user scope, aggregates, multi-resource).

This module provides:
- HintAccessor class: Unified interface for accessing resource hints
- Module functions: For backward compatibility and multi-resource operations
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# HintAccessor Class - Unified hint access interface
# =============================================================================

class HintAccessor:
    """
    Unified interface for accessing hints for a specific resource.

    Usage:
        accessor = HintAccessor("service-orders", resource_hints)
        fields = accessor.embedded_fields
        role = accessor.endpoint_role

    This class replaces repetitive patterns like:
        hints = (resource_hints or {}).get(resource) or {}
        if not isinstance(hints, dict): return []
        return hints.get("__embedded_fields__") or []
    """

    def __init__(
        self,
        resource: str,
        resource_hints: Dict[str, Any],
        schema_resource: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ):
        self.resource = resource or ""
        self.resource_hints = resource_hints or {}
        self.schema_resource = schema_resource
        self.user_context = user_context or {}
        self._hints = self._load_hints()

    def _load_hints(self) -> Dict[str, Any]:
        """Load hints for this resource."""
        if not self.resource or not self.resource_hints:
            return {}
        hints = self.resource_hints.get(self.resource)
        return hints if isinstance(hints, dict) else {}

    def _get_list(self, key: str) -> List[str]:
        """Get a list value from hints."""
        val = self._hints.get(key) or []
        return list(val) if isinstance(val, (list, tuple)) else []

    def _get_set(self, key: str) -> Set[str]:
        """Get a set value from hints."""
        val = self._hints.get(key) or []
        return {str(v) for v in val} if isinstance(val, (list, tuple)) else set()

    def _get_dict(self, key: str) -> Dict[str, Any]:
        """Get a dict value from hints."""
        val = self._hints.get(key) or {}
        return dict(val) if isinstance(val, dict) else {}

    def _get_str(self, key: str) -> Optional[str]:
        """Get a string value from hints."""
        val = self._hints.get(key)
        return str(val) if val else None

    def _get_int(self, key: str, cap: Optional[int] = None) -> Optional[int]:
        """Get an int value from hints with optional cap."""
        val = self._hints.get(key)
        if val is None:
            return None
        try:
            result = int(val)
            return min(result, cap) if cap else result
        except (TypeError, ValueError):
            return None

    # =========================================================================
    # Properties - Simple accessors
    # =========================================================================

    @property
    def embedded_fields(self) -> List[str]:
        """Nested field names from __embedded_fields__."""
        return self._get_list("__embedded_fields__")

    @property
    def extra_query_params(self) -> Set[str]:
        """Params supported by API but missing from OpenAPI."""
        return self._get_set("__extra_query_params__")

    @property
    def extra_response_fields(self) -> Set[str]:
        """Response fields present at runtime but absent from OpenAPI."""
        return self._get_set("__extra_response_fields__")

    @property
    def client_side_filter_fields(self) -> Set[str]:
        """Filters intentionally applied client-side."""
        fields = set(self.extra_response_fields)
        declared = self._hints.get("__client_side_filters__") or []
        if isinstance(declared, (list, tuple)):
            fields.update(str(f) for f in declared)
        return fields

    @property
    def response_count_fields(self) -> List[str]:
        """Fields to read numeric totals from dashboard responses."""
        fields = self._hints.get("__response_count_fields__") or []
        return [str(f) for f in fields] if isinstance(fields, (list, tuple)) else []

    @property
    def response_summary_fields(self) -> Dict[str, str]:
        """Map response field names to display labels for dashboard payloads."""
        fields = self._hints.get("__response_summary_fields__") or {}
        if isinstance(fields, dict) and fields:
            return {str(k): str(v) for k, v in fields.items()}

        if self.endpoint_role in ("summary", "dashboard", "metrics"):
            discovered: Dict[str, str] = {}
            res_data = self.schema_resource or {}
            for item in res_data.get("fields") or []:
                if isinstance(item, dict) and item.get("name"):
                    name = str(item["name"])
                    label = item.get("label") or item.get("title") or name.replace("_", " ").title()
                    discovered[name] = str(label)
                elif isinstance(item, str):
                    discovered[item] = item.replace("_", " ").title()
            if discovered:
                return discovered
        return {}

    @property
    def response_summary_field_synonyms(self) -> Dict[str, str]:
        """Map query phrases to summary response field names."""
        syns = self._hints.get("__response_summary_field_synonyms__") or {}
        return {str(k).lower(): str(v) for k, v in syns.items()} if isinstance(syns, dict) else {}

    @property
    def endpoint_role(self) -> Optional[str]:
        """Hint-declared endpoint role: list, summary, settings, etc."""
        role = self._hints.get("__endpoint_role__")
        return str(role).lower() if role else None

    @property
    def related_list_resource(self) -> Optional[str]:
        """List resource to pivot to after a summary turn."""
        return self._get_str("__related_list_resource__")

    @property
    def related_summary_resource(self) -> Optional[str]:
        """Summary resource for amount-metric queries on a list sibling."""
        return self._get_str("__related_summary_resource__")

    @property
    def user_scoped_fields(self) -> List[str]:
        """User-scoped filter fields for this resource."""
        role = self.user_context.get("role") or ""
        by_role = self._hints.get("__user_scoped_fields_by_role__") or {}
        if isinstance(by_role, dict) and by_role:
            if role:
                role_l = str(role).lower()
                for key, fields in by_role.items():
                    if str(key).lower() == role_l:
                        return list(fields) if isinstance(fields, (list, tuple)) else []
                return []
        fields = self._hints.get("__user_scoped_fields__") or []
        return list(fields) if isinstance(fields, (list, tuple)) else []

    @property
    def count_page_size(self) -> Optional[int]:
        """Page size for count queries (capped)."""
        from . import constants
        return self._get_int("__count_page_size__", cap=constants.PAGE_SIZE_CAP)

    @property
    def count_default_filters(self) -> Dict[str, Any]:
        """Default filters to apply for count queries."""
        return self._get_dict("__count_default_filters__")

    @property
    def aggregate_resources(self) -> Optional[List[str]]:
        """Child resource list from __aggregate_resources__ or __aggregate__."""
        for key in ("__aggregate_resources__", "__aggregate__"):
            agg = self._hints.get(key)
            if isinstance(agg, (list, tuple)) and len(agg) >= 2:
                return list(agg)
        return None

    @property
    def is_virtual_aggregate(self) -> bool:
        """True when this resource defines aggregate children."""
        return self.aggregate_resources is not None

    @property
    def resource_synonyms(self) -> List[str]:
        """Synonym terms for this resource."""
        syns = self._hints.get("__resource_synonyms__") or []
        if isinstance(syns, list):
            return [str(s) for s in syns]
        return [str(syns)] if syns else []


# =============================================================================
# Static Query Analysis Functions (no resource context needed)
# =============================================================================

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


def get_user_scoped_fields(
    resource: str,
    resource_hints: Dict[str, Any],
    user_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Return user-scoped filter fields for a resource.

    Uses __user_scoped_fields_by_role__[role] when user_context.role is set,
    otherwise __user_scoped_fields__.
    """
    if not resource or not resource_hints:
        return []
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return []

    role = (user_context or {}).get("role") or ""
    by_role = hints.get("__user_scoped_fields_by_role__") or {}
    if isinstance(by_role, dict) and by_role:
        if role:
            role_l = str(role).lower()
            for key, fields in by_role.items():
                if str(key).lower() == role_l:
                    return list(fields) if isinstance(fields, (list, tuple)) else []
            return []
        # by_role configured but no role in context — legacy fallback below

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


def get_extra_response_fields(resource: str, resource_hints: Dict[str, Any]) -> Set[str]:
    """Response fields present at runtime but absent from OpenAPI (client-side filter/count)."""
    if not resource or not resource_hints:
        return set()
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return set()
    fields = hints.get("__extra_response_fields__") or []
    return {str(f) for f in fields} if isinstance(fields, (list, tuple)) else set()


def get_client_side_filter_fields(resource: str, resource_hints: Dict[str, Any]) -> Set[str]:
    """Filters intentionally applied client-side (no warning)."""
    fields = set(get_extra_response_fields(resource, resource_hints))
    if not resource or not resource_hints:
        return fields
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return fields
    declared = hints.get("__client_side_filters__") or []
    if isinstance(declared, (list, tuple)):
        fields.update(str(f) for f in declared)
    return fields


def get_response_count_fields(resource: str, resource_hints: Dict[str, Any]) -> List[str]:
    """Fields to read numeric totals from singleton/dashboard responses."""
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return []
    fields = hints.get("__response_count_fields__") or []
    return [str(f) for f in fields] if isinstance(fields, (list, tuple)) else []


def get_response_summary_fields(
    resource: str,
    resource_hints: Dict[str, Any],
    schema_resource: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Map response field names to display labels for summary/dashboard payloads."""
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        hints = {}
    fields = hints.get("__response_summary_fields__") or {}
    if isinstance(fields, dict) and fields:
        return {str(k): str(v) for k, v in fields.items()}

    if get_endpoint_role(resource, resource_hints) in ("summary", "dashboard", "metrics"):
        discovered: Dict[str, str] = {}
        res_data = schema_resource or {}
        for item in res_data.get("fields") or []:
            if isinstance(item, dict) and item.get("name"):
                name = str(item["name"])
                label = item.get("label") or item.get("title") or name.replace("_", " ").title()
                discovered[name] = str(label)
            elif isinstance(item, str):
                discovered[item] = item.replace("_", " ").title()
        if discovered:
            return discovered
    return {}


def get_response_summary_field_synonyms(
    resource: str,
    resource_hints: Dict[str, Any],
) -> Dict[str, str]:
    """Map query phrases to summary response field names."""
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return {}
    syns = hints.get("__response_summary_field_synonyms__") or {}
    if isinstance(syns, dict):
        return {str(k).lower(): str(v) for k, v in syns.items()}
    return {}


def get_endpoint_role(resource: str, resource_hints: Dict[str, Any]) -> Optional[str]:
    """Hint-declared endpoint role: list, summary, settings, etc."""
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return None
    role = hints.get("__endpoint_role__")
    return str(role).lower() if role else None


def get_related_list_resource(
    resource: str,
    resource_hints: Dict[str, Any],
) -> Optional[str]:
    """List resource to pivot to when user follows up on a summary/dashboard turn."""
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return None
    related = hints.get("__related_list_resource__")
    return str(related) if related else None


def get_related_summary_resource(
    resource: str,
    resource_hints: Dict[str, Any],
) -> Optional[str]:
    """Summary/dashboard resource for amount-metric queries on a list sibling."""
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return None
    related = hints.get("__related_summary_resource__")
    return str(related) if related else None


# Built-in phrase → candidate response field names (first match wins per resource)
_BUILTIN_SUMMARY_PHRASE_FIELDS: Dict[str, List[str]] = {
    "pending amount": ["total_outstanding", "pending_amount", "outstanding_amount"],
    "pending": ["total_outstanding", "pending_amount"],
    "outstanding amount": ["total_outstanding"],
    "the outstanding": ["total_outstanding"],
    "outstanding": ["total_outstanding"],
    "collected this month": ["collected_this_month", "amount_collected_this_month"],
    "collected": ["collected_this_month", "total_collected"],
    "overdue amount": ["total_overdue"],
    "overdue": ["total_overdue"],
    "invoiced amount": ["total_invoiced", "total_invoiced_amount"],
    "total invoiced": ["total_invoiced"],
    "total invoiced amount": ["total_invoiced"],
}


def _available_summary_field_names(
    resource: str,
    resource_hints: Dict[str, Any],
    schema_resource: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    return set(
        get_response_summary_fields(resource, resource_hints, schema_resource).keys()
    ) | set(get_response_count_fields(resource, resource_hints))


def _field_for_builtin_phrase(
    phrase: str,
    available: Set[str],
) -> Optional[str]:
    for candidate in _BUILTIN_SUMMARY_PHRASE_FIELDS.get(phrase.lower(), []):
        if candidate in available:
            return candidate
    return None


def _matching_term_lengths(query: str, resource_name: str, hints: Dict[str, Any]) -> List[int]:
    """Return lengths of all resource terms that match the query."""
    if not query:
        return []
    q = query.lower()
    terms = [resource_name, resource_name.replace("-", " ")]
    syns = hints.get("__resource_synonyms__") or []
    if isinstance(syns, list):
        terms.extend(str(s) for s in syns)
    elif syns:
        terms.append(str(syns))
    return [
        len(_normalize_term(t))
        for t in terms
        if t and _term_in_query(t, q)
    ]


def find_child_resource_over_aggregate(
    query: str,
    resource_hints: Dict[str, Any],
    schema_resources: Set[str],
) -> Optional[str]:
    """
    Prefer a specific aggregate child when the query names it explicitly.

    E.g. "how many detailed reports?" → details-reports, not virtual reports aggregate.
    """
    if not query or not resource_hints:
        return None

    child_to_parent: Dict[str, str] = {}
    for name, hints in resource_hints.items():
        if not isinstance(hints, dict):
            continue
        agg = get_aggregate_resources(hints)
        if agg:
            for child in agg:
                child_to_parent[child] = name

    best_child: Optional[str] = None
    best_child_len = 0
    for child in schema_resources:
        if child not in child_to_parent:
            continue
        child_hints = resource_hints.get(child) or {}
        if not isinstance(child_hints, dict):
            child_hints = {}
        lengths = _matching_term_lengths(query, child, child_hints)
        if lengths and max(lengths) > best_child_len:
            best_child = child
            best_child_len = max(lengths)

    if not best_child:
        return None

    parent = child_to_parent[best_child]
    parent_hints = resource_hints.get(parent) or {}
    if not isinstance(parent_hints, dict):
        parent_hints = {}
    parent_lengths = _matching_term_lengths(query, parent, parent_hints)
    parent_best = max(parent_lengths) if parent_lengths else 0

    if best_child_len >= parent_best:
        return best_child
    return None


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
        lengths = _matching_term_lengths(query, name, hints)
        if not lengths:
            continue
        term_len = max(lengths)
        if term_len <= best_len:
            continue
        # Skip aggregate when a child matches with equal or greater specificity
        child_lengths = []
        for child in agg:
            child_hints = resource_hints.get(child) or {}
            if isinstance(child_hints, dict):
                child_lengths.extend(_matching_term_lengths(query, child, child_hints))
        if child_lengths and max(child_lengths) >= term_len:
            continue
        best = list(agg)
        best_len = term_len
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

    specific_child = find_child_resource_over_aggregate(query, hints, resources)
    if specific_child:
        result["resource"] = specific_child
        result.pop("multiple_resources", None)
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


def find_explicit_resources_in_query(
    query: str,
    resource_hints: Dict[str, Any],
    schema_resources: Set[str],
) -> List[str]:
    """Resources explicitly named in the query (schema + virtual hint keys)."""
    if not query:
        return []
    found: Set[str] = set(find_mentioned_resources(query, resource_hints, schema_resources))
    q = query.lower()
    for name, hints in (resource_hints or {}).items():
        if not isinstance(hints, dict):
            continue
        terms = [name, name.replace("-", " ")]
        syns = hints.get("__resource_synonyms__") or []
        if isinstance(syns, list):
            terms.extend(str(s) for s in syns)
        elif syns:
            terms.append(str(syns))
        if any(_term_in_query(t, q) for t in terms if t):
            found.add(name)
    return sorted(found)


def find_best_explicit_resource(
    query: str,
    resource_hints: Dict[str, Any],
    schema_resources: Set[str],
    prior_resource: Optional[str] = None,
) -> Optional[str]:
    """
    Pick the single best-matching resource using longest synonym/term length.

    Prefers prior_resource on ties for session continuity.
    """
    mentioned = find_explicit_resources_in_query(query, resource_hints, schema_resources)
    if not mentioned:
        return None
    if len(mentioned) == 1:
        return mentioned[0]

    scored: List[tuple] = []
    for name in mentioned:
        rh = resource_hints.get(name) or {}
        if not isinstance(rh, dict):
            rh = {}
        lengths = _matching_term_lengths(query, name, rh)
        scored.append((name, max(lengths) if lengths else 0))

    if not scored:
        return mentioned[0]

    top_len = max(s for _, s in scored)
    top = [n for n, s in scored if s == top_len]
    if len(top) == 1:
        return top[0]
    if prior_resource and prior_resource in top:
        return prior_resource
    return top[0]


def should_force_standalone_for_resource_switch(
    query: str,
    prior_meta: Dict[str, Any],
    resource_hints: Dict[str, Any],
    schema_resources: Set[str],
) -> bool:
    """
    Force standalone when the query explicitly names a different resource scope.

    Covers: users → service-orders, aggregate reports → detailed reports only.
    """
    if not prior_meta.get("resource"):
        return False

    mentioned = find_explicit_resources_in_query(query, resource_hints, schema_resources)
    if not mentioned:
        return False

    prev = prior_meta.get("resource")
    prev_multi = prior_meta.get("multiple_resources") or []

    if len(mentioned) == 1:
        target = mentioned[0]
        if len(prev_multi) > 1 and target in prev_multi:
            return True
        if target != prev and target not in prev_multi:
            return True

    if len(mentioned) > 1 and set(mentioned) != set(prev_multi):
        return True

    return False


def apply_count_default_filters(
    parsed: Dict[str, Any],
    query: str,
    resource_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply __count_default_filters__ when question_type=count and query
    does not explicitly mention the excluded values (e.g. archived).
    """
    if not isinstance(parsed, dict) or parsed.get("question_type") != "count":
        return parsed

    resource = parsed.get("resource", "")
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return parsed

    defaults = hints.get("__count_default_filters__") or {}
    if not defaults:
        return parsed

    q_lower = (query or "").lower()
    result = dict(parsed)
    filters = dict(result.get("filters") or {})

    for field, fval in defaults.items():
        if field in filters:
            continue
        excluded = fval.get("value") if isinstance(fval, dict) else fval
        if excluded is not None and str(excluded).lower() in q_lower:
            continue
        field_hints = hints.get(field) or {}
        skip = False
        if isinstance(field_hints, dict):
            for phrase in (field_hints.get("synonyms") or {}).keys():
                if phrase and str(phrase).lower() in q_lower:
                    skip = True
                    break
            if not skip:
                for val in field_hints.get("values") or []:
                    if val is not None and str(val).lower() in q_lower:
                        skip = True
                        break
        if skip:
            continue
        filters[field] = fval if isinstance(fval, dict) else {
            "operator": "equals", "value": fval,
        }

    result["filters"] = filters
    return result


def get_count_page_size(resource: str, resource_hints: Dict[str, Any]) -> Optional[int]:
    """Read __count_page_size__ from resource hints (capped at PAGE_SIZE_CAP)."""
    from . import constants

    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return None
    raw = hints.get("__count_page_size__")
    if raw is None:
        return None
    try:
        return min(int(raw), constants.PAGE_SIZE_CAP)
    except (TypeError, ValueError):
        return None


def should_fetch_all_pages_for_count(
    resource: str,
    resource_hints: Dict[str, Any],
    has_client_side_filters: bool,
) -> bool:
    """
    Whether to paginate through all API pages before client-side count filtering.

    Defaults to True when client-side filters are in play unless hints opt out.
    """
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return has_client_side_filters
    if "__count_fetch_all_when_client_filters__" in hints:
        return bool(hints["__count_fetch_all_when_client_filters__"])
    return has_client_side_filters


def build_count_filter_description(
    filters: Dict[str, Any],
    resource: str = "",
    resource_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """Build human-readable filter context for count summaries."""
    if not filters:
        return ""

    hints = (resource_hints or {}).get(resource) or {}
    parts: List[str] = []

    for field, fval in filters.items():
        if field.startswith("_"):
            continue
        val = fval.get("value") if isinstance(fval, dict) else fval
        op = fval.get("operator", "equals") if isinstance(fval, dict) else "equals"

        if field == "name" and val is not None:
            parts.append(f"named {val}")
            continue
        if field == "status" and val is not None:
            parts.append(str(val))
            continue
        if field == "is_available":
            if val is True:
                parts.append("available")
            elif val is False:
                parts.append("unavailable")
            continue

        if val is None:
            continue

        display_val = str(val)
        field_hints = hints.get(field) if isinstance(hints, dict) else None
        if isinstance(field_hints, dict):
            syns = field_hints.get("synonyms") or {}
            for phrase, maps_to in syns.items():
                if maps_to == val or str(maps_to).lower() == str(val).lower():
                    display_val = str(phrase)
                    break

        if op in ("not_equals", "ne"):
            parts.append(f"not {display_val}")
        else:
            parts.append(display_val)

    return " ".join(parts)


def _phrase_matches_query(phrase: str, query_lower: str) -> bool:
    """Match phrase in query, allowing a single missing character on single-token phrases."""
    norm = phrase.lower().strip()
    if not norm:
        return False
    if norm in query_lower:
        return True
    if _term_in_query(norm, query_lower):
        return True
    if " " not in norm and len(norm) >= 5:
        for i in range(len(norm)):
            variant = norm[:i] + norm[i + 1:]
            if variant in query_lower:
                return True
    return False


def resolve_summary_field_from_query(
    query: str,
    resource: str,
    resource_hints: Dict[str, Any],
    parsed: Optional[Dict[str, Any]] = None,
    schema_resource: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Pick one dashboard metric field based on query phrasing or parsed summary_field."""
    if isinstance(parsed, dict) and parsed.get("summary_field"):
        return str(parsed["summary_field"])

    q = (query or "").lower()
    if not q:
        return None

    summary_fields = get_response_summary_fields(
        resource, resource_hints, schema_resource,
    )
    available = _available_summary_field_names(resource, resource_hints, schema_resource)
    synonyms = get_response_summary_field_synonyms(resource, resource_hints)

    best_field: Optional[str] = None
    best_len = 0
    for phrase, field in synonyms.items():
        if field not in available and field not in summary_fields:
            continue
        if _phrase_matches_query(phrase, q) and len(phrase) > best_len:
            best_field = field
            best_len = len(phrase)

    for phrase, candidates in _BUILTIN_SUMMARY_PHRASE_FIELDS.items():
        if not _phrase_matches_query(phrase, q):
            continue
        field = _field_for_builtin_phrase(phrase, available)
        if field and len(phrase) > best_len:
            best_field = field
            best_len = len(phrase)

    for field, label in summary_fields.items():
        candidates = [
            field.lower(),
            field.replace("_", " ").lower(),
            label.lower(),
        ]
        for cand in candidates:
            if cand and _phrase_matches_query(cand, q) and len(cand) > best_len:
                best_field = field
                best_len = len(cand)

    return best_field


def _query_implies_list_count(query: str) -> bool:
    return bool(re.search(
        r"\b(?:how many|count of|number of)\b", query or "", re.IGNORECASE,
    ))


def _query_implies_amount_metric(query: str) -> bool:
    return bool(re.search(
        r"\b(?:how much|total amount|invoiced amount|pending amount|outstanding|"
        r"collected|overdue amount|what(?:'s| is) the (?:total|pending|outstanding|"
        r"collected|overdue|invoiced))\b",
        query or "",
        re.IGNORECASE,
    ))


def align_parsed_resource_with_query(
    parsed: Dict[str, Any],
    query: str,
    resource_hints: Dict[str, Any],
    schema_resources: Set[str],
    prior_resource: Optional[str] = None,
) -> Dict[str, Any]:
    """Override parsed resource when the query explicitly names a different one."""
    if not isinstance(parsed, dict) or not query:
        return parsed

    target = find_best_explicit_resource(
        query, resource_hints, schema_resources, prior_resource=prior_resource,
    )
    if not target or parsed.get("resource") == target:
        return parsed

    result = dict(parsed)
    result["resource"] = target
    multi = result.get("multiple_resources")
    if multi and target not in multi:
        result.pop("multiple_resources", None)
    result["merge_with_previous"] = False
    return result


def should_lock_parsed_resource(
    parsed: Dict[str, Any],
    resource_hints: Dict[str, Any],
) -> bool:
    """True when API matcher must not override parsed resource via synonym heuristics."""
    if not isinstance(parsed, dict):
        return False
    resource = parsed.get("resource")
    if not resource:
        return False
    qt = (parsed.get("question_type") or "").lower()
    if qt in ("summary", "aggregate_metric"):
        return True
    if parsed.get("summary_field"):
        return True
    role = get_endpoint_role(resource, resource_hints)
    return role in ("summary", "dashboard", "metrics")


def resolve_match_resource(
    parsed: Dict[str, Any],
    query: str,
    resource_hints: Dict[str, Any],
) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Resolve resource for API matching.

    Returns (resource_name, locked, updated_parsed).
    """
    if not isinstance(parsed, dict):
        return "", False, parsed or {}

    result = dict(parsed)
    resource = (result.get("resource") or "").strip()
    q = query or result.get("original_input") or ""

    if (
        resource
        and _query_implies_amount_metric(q)
        and not _query_implies_list_count(q)
    ):
        related = get_related_summary_resource(resource, resource_hints)
        if related:
            resource = related
            result["resource"] = related
            result["question_type"] = "summary"
            field = resolve_summary_field_from_query(q, related, resource_hints, result)
            if field:
                result["summary_field"] = field

    locked = should_lock_parsed_resource(result, resource_hints)
    return resource, locked, result


def build_summary_list_mismatch_message(
    resource: str,
    resource_hints: Dict[str, Any],
    parsed: Optional[Dict[str, Any]] = None,
) -> str:
    """User-facing message when a summary turn receives paginated list data."""
    label = (resource or "that resource").replace("-", " ")
    role = get_endpoint_role(resource, resource_hints)
    if role in ("summary", "dashboard", "metrics"):
        return (
            f"I expected dashboard metrics from {label}, but the API returned a paginated list. "
            "Please verify the summary endpoint path in your schema configuration."
        )
    related = get_related_summary_resource(resource, resource_hints)
    if related:
        return (
            f"That question needs dashboard metrics from {related.replace('-', ' ')}, "
            f"not a list of {label}. Try asking for the AR summary instead."
        )
    field = (parsed or {}).get("summary_field")
    if field:
        return (
            f"I couldn't find {field.replace('_', ' ')} in the list response. "
            "This metric may only be available on the dashboard summary endpoint."
        )
    return (
        f"I expected a dashboard metric for {label}, but received a list response instead."
    )


def apply_resource_question_defaults(
    parsed: Dict[str, Any],
    query: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Adjust question_type for summary/dashboard resources and metric queries."""
    if not isinstance(parsed, dict):
        return parsed

    resource = parsed.get("resource") or ""
    hints = (schema or {}).get("resource_hints") or {}
    role = get_endpoint_role(resource, hints)

    if _query_implies_amount_metric(query) and not _query_implies_list_count(query):
        related_summary = get_related_summary_resource(resource, hints)
        if related_summary:
            result = dict(parsed)
            result["resource"] = related_summary
            result["question_type"] = "summary"
            field = resolve_summary_field_from_query(query, related_summary, hints, result)
            if field:
                result["summary_field"] = field
            return result

    if role not in ("summary", "dashboard", "metrics"):
        return parsed

    qt = (parsed.get("question_type") or "").lower()
    list_signals = re.search(
        r"\b(?:how many|list|show all|show me all|count of)\b", query or "", re.IGNORECASE,
    )
    if qt == "count" and not list_signals:
        result = dict(parsed)
        result["question_type"] = "summary"
        field = resolve_summary_field_from_query(query, resource, hints, result)
        if field:
            result["summary_field"] = field
        return result
    if not qt or qt == "read":
        field = resolve_summary_field_from_query(query, resource, hints, parsed)
        if field:
            result = dict(parsed)
            result["question_type"] = "summary"
            result["summary_field"] = field
            return result
    return parsed


def is_summary_response(
    data: Any,
    resource: str,
    resource_hints: Dict[str, Any],
    schema_resource: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when data is a singleton dashboard/metrics dict, not a list page."""
    if not isinstance(data, dict) or "results" in data:
        return False
    summary_fields = get_response_summary_fields(
        resource, resource_hints, schema_resource,
    )
    count_fields = get_response_count_fields(resource, resource_hints)
    for field in list(summary_fields.keys()) + count_fields:
        if _get_nested_summary_value(data, field) is not None:
            return True
    return False


def _format_summary_value(
    val: Any,
    field: str = "",
    resource: str = "",
    resource_hints: Optional[Dict[str, Any]] = None,
) -> str:
    from .display_formatting import format_display_value
    return format_display_value(val, field, resource, resource_hints)


def _get_nested_summary_value(data: Dict[str, Any], field_path: str) -> Any:
    """Read a summary value supporting dotted paths (e.g. days_1_30_bucket.amount)."""
    if field_path in data:
        return data[field_path]
    parts = field_path.split(".")
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def build_summary_response_text(
    data: Dict[str, Any],
    resource: str,
    resource_hints: Dict[str, Any],
    query: str = "",
    parsed: Optional[Dict[str, Any]] = None,
    schema_resource: Optional[Dict[str, Any]] = None,
    follow_up_only: bool = False,
) -> str:
    """Format dashboard/summary API payloads — one metric when query names it."""
    summary_fields = get_response_summary_fields(
        resource, resource_hints, schema_resource,
    )

    target = resolve_summary_field_from_query(
        query, resource, resource_hints, parsed, schema_resource,
    )
    if target:
        val = _get_nested_summary_value(data, target)
        if val is not None:
            label = summary_fields.get(target, target.replace("_", " ").title())
            formatted = _format_summary_value(val, target, resource, resource_hints)
            return f"{label} is {formatted}."

    if follow_up_only:
        return (
            "I couldn't tell which metric you meant from the dashboard. "
            "Try asking for a specific value, such as 'what is the outstanding amount?'"
        )

    # Standalone turn with no field match: emit all configured fields (legacy fallback)
    parts: List[str] = []
    for field, label in summary_fields.items():
        val = _get_nested_summary_value(data, field)
        if val is None:
            continue
        formatted = _format_summary_value(val, field, resource, resource_hints)
        parts.append(f"{label} is {formatted}")

    if parts:
        return ". ".join(parts) + "."

    count_fields = get_response_count_fields(resource, resource_hints)
    for field in count_fields:
        val = _get_nested_summary_value(data, field)
        if val is not None:
            label = summary_fields.get(field, field.replace("_", " ").title())
            formatted = _format_summary_value(val, field, resource, resource_hints)
            return f"{label} is {formatted}."

    return ""
