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
    if role and isinstance(by_role, dict):
        if role in by_role:
            fields = by_role[role]
            return list(fields) if isinstance(fields, (list, tuple)) else []
        role_l = str(role).lower()
        for key, fields in by_role.items():
            if str(key).lower() == role_l:
                return list(fields) if isinstance(fields, (list, tuple)) else []

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
    synonyms = get_response_summary_field_synonyms(resource, resource_hints)

    best_field: Optional[str] = None
    best_len = 0
    for phrase, field in synonyms.items():
        if _phrase_matches_query(phrase, q) and len(phrase) > best_len:
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


def align_parsed_resource_with_query(
    parsed: Dict[str, Any],
    query: str,
    resource_hints: Dict[str, Any],
    schema_resources: Set[str],
) -> Dict[str, Any]:
    """Override parsed resource when the query explicitly names a different one."""
    if not isinstance(parsed, dict) or not query:
        return parsed

    mentioned = find_explicit_resources_in_query(query, resource_hints, schema_resources)
    if len(mentioned) != 1:
        return parsed

    target = mentioned[0]
    if parsed.get("resource") == target:
        return parsed

    result = dict(parsed)
    result["resource"] = target
    multi = result.get("multiple_resources")
    if multi and target not in multi:
        result.pop("multiple_resources", None)
    result["merge_with_previous"] = False
    return result


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
    if role not in ("summary", "dashboard", "metrics"):
        return parsed

    qt = (parsed.get("question_type") or "").lower()
    q = (query or "").lower()
    list_signals = re.search(
        r"\b(?:how many|list|show all|show me all|count of)\b", q, re.IGNORECASE,
    )
    if qt == "count" and not list_signals:
        result = dict(parsed)
        result["question_type"] = "summary"
        return result
    if not qt or qt == "read":
        if resolve_summary_field_from_query(query, resource, hints, parsed):
            result = dict(parsed)
            result["question_type"] = "summary"
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
    keys = set(data.keys())
    if summary_fields and keys.intersection(summary_fields.keys()):
        return True
    if count_fields and keys.intersection(count_fields):
        return True
    return False


def _format_summary_value(val: Any) -> Any:
    if isinstance(val, float) and val == int(val):
        return int(val)
    return val


def build_summary_response_text(
    data: Dict[str, Any],
    resource: str,
    resource_hints: Dict[str, Any],
    query: str = "",
    parsed: Optional[Dict[str, Any]] = None,
    schema_resource: Optional[Dict[str, Any]] = None,
) -> str:
    """Format dashboard/summary API payloads — one metric when query names it."""
    summary_fields = get_response_summary_fields(
        resource, resource_hints, schema_resource,
    )

    target = resolve_summary_field_from_query(
        query, resource, resource_hints, parsed, schema_resource,
    )
    if target and target in data and data[target] is not None:
        label = summary_fields.get(target, target.replace("_", " ").title())
        val = _format_summary_value(data[target])
        return f"{label} is {val}."

    # No query match: emit all configured fields (legacy fallback)
    parts: List[str] = []
    for field, label in summary_fields.items():
        if field not in data or data[field] is None:
            continue
        val = _format_summary_value(data[field])
        parts.append(f"{label} is {val}")

    if parts:
        return ". ".join(parts) + "."

    count_fields = get_response_count_fields(resource, resource_hints)
    for field in count_fields:
        if field in data and data[field] is not None:
            label = field.replace("_", " ")
            return f"{label.title()} is {data[field]}."

    return ""
