"""
Unified filter module for Enable AI.

Combines:
- Semantic filter injection (from hints/synonyms)
- Temporal filter injection (date ranges)
- Client-side post-filtering
- Parameter validation
"""

import calendar
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .hint_utils import apply_count_default_filters, query_implies_user_scope
from .query_utils import strip_quotes_for_matching
from .schema_introspector import SchemaIntrospector
from .user_context_resolver import resolve_user_context_in_parsed
from .utils import setup_logger

logger = setup_logger("enable_ai.filters")


# =============================================================================
# Section 1: Semantic Filter Injection
# =============================================================================

def _normalize_resource_name(name: str) -> str:
    return (name or "").lower().replace("-", "_").replace(" ", "_")


def _get_resource_hints(
    resource_hints: Dict[str, Any],
    resource: str,
) -> Dict[str, Any]:
    """Look up hints for a resource, trying exact and normalized names."""
    if not resource_hints or not resource:
        return {}
    if resource in resource_hints and isinstance(resource_hints[resource], dict):
        return resource_hints[resource]
    target = _normalize_resource_name(resource)
    for name, hints in resource_hints.items():
        if isinstance(hints, dict) and _normalize_resource_name(name) == target:
            return hints
    return {}


def _filter_value_from_synonym(mapped_value: Any, phrase: str) -> Any:
    """
    Resolve the filter value to inject from a synonym mapping.

    Simple mappings use the value as-is (e.g. "low stock" -> "low").
    List mappings use operator "in" (e.g. "urgent or high" -> ["High", "Urgent"]).
    Complex API mappings (field__op=value) use the phrase when it is a
    single token; api_matcher._validate_filter_values expands them later.
    """
    if isinstance(mapped_value, (list, tuple)):
        return list(mapped_value)
    if not isinstance(mapped_value, str):
        return mapped_value
    if "=" in mapped_value and "__" in mapped_value.split("=", 1)[0]:
        parts = phrase.strip().split()
        return parts[0] if parts else mapped_value
    return mapped_value


def _phrase_in_query(phrase: str, text: str) -> bool:
    """Match a synonym phrase in query text (word-boundary for single tokens)."""
    if not phrase or not text:
        return False
    if " " in phrase:
        return phrase in text
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?:s|es)?(?!\w)", text))


def _synonym_phrases_for_field(field_hints: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Return (phrase_lower, mapped_value) pairs, longest phrases first."""
    synonyms = field_hints.get("synonyms") or {}
    if not isinstance(synonyms, dict):
        return []
    pairs = [(str(phrase).lower().strip(), value) for phrase, value in synonyms.items() if phrase]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def inject_semantic_filters(
    filters: Optional[Dict[str, Any]],
    resource: str,
    query: str,
    resource_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add filters when the query text matches a synonym phrase from resource_hints.

    Idempotent — skips fields already present in filters.
    """
    filters = dict(filters or {})
    text = (query or "").lower()
    rh = _get_resource_hints(resource_hints or {}, resource)
    if not rh:
        return filters

    for field_name, field_hints in rh.items():
        if field_name.startswith("__") or not isinstance(field_hints, dict):
            continue
        if field_name in filters:
            continue
        base = field_name.split("__")[0]
        if base in filters:
            continue

        matched = False
        for phrase, mapped_value in _synonym_phrases_for_field(field_hints):
            if _phrase_in_query(phrase, text):
                value = _filter_value_from_synonym(mapped_value, phrase)
                if isinstance(value, (list, tuple)):
                    filters[field_name] = {"operator": "in", "value": list(value)}
                else:
                    filters[field_name] = {"operator": "equals", "value": value}
                logger.info(
                    "Semantic filter from hint: %s=%r (phrase=%r, resource=%s)",
                    field_name, value, phrase, resource,
                )
                matched = True
                break

        if not matched:
            for value in field_hints.get("values") or []:
                val_str = str(value).lower()
                if _phrase_in_query(val_str, text):
                    filters[field_name] = {"operator": "equals", "value": value}
                    logger.info(
                        "Semantic filter from hint value: %s=%r (resource=%s)",
                        field_name, value, resource,
                    )
                    break

        if not matched and field_name in ("report_type", "type"):
            for token, canonical in (("ut", "UT"), ("pt", "PT")):
                if re.search(rf"\b{token}\b", text):
                    filters[field_name] = {"operator": "equals", "value": canonical}
                    logger.info(
                        "Semantic filter report_type=%s from token %s",
                        canonical, token,
                    )
                    break

    return filters


def apply_semantic_filters(
    parsed: Dict[str, Any],
    query: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply semantic filter injection to a parsed query dict."""
    if not isinstance(parsed, dict):
        return parsed
    result = dict(parsed)
    hints = (schema or {}).get("resource_hints") or {}
    result["filters"] = inject_semantic_filters(
        result.get("filters"),
        result.get("resource", ""),
        strip_quotes_for_matching(query),
        hints,
    )
    result = apply_count_default_filters(result, query, hints)
    return result


# =============================================================================
# Section 2: Temporal Filter Injection
# =============================================================================

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


# =============================================================================
# Section 3: Client-Side Post-Filtering
# =============================================================================

def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip().lower()


def _get_item_value(item: Any, field: str) -> Any:
    """Resolve a field value from an item, including simple nested paths."""
    if not isinstance(item, dict):
        return item

    if field in item:
        return item[field]

    if "__" in field:
        parts = field.split("__")
        current: Any = item
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return item.get(field) or item.get("_".join(parts))
        return current

    return item.get(field)


def _compare(actual: Any, expected: Any, operator: str) -> bool:
    op = (operator or "equals").lower()

    if op in ("equals", "eq", "exact"):
        if isinstance(actual, dict) and "name" in actual:
            return _normalize_value(actual.get("name")) == _normalize_value(expected)
        if isinstance(expected, bool):
            return bool(actual) == expected
        return _normalize_value(actual) == _normalize_value(expected)

    if op in ("not_equals", "ne"):
        return not _compare(actual, expected, "equals")

    if op in ("contains", "icontains"):
        exp_norm = _normalize_value(expected)
        act_norm = _normalize_value(actual)
        if re.match(r"^[a-z]{2,4}-[\w-]+$", exp_norm, re.I):
            return act_norm == exp_norm
        return exp_norm in act_norm

    if op in ("starts_with", "istartswith"):
        return _normalize_value(actual).startswith(_normalize_value(expected))

    if op in ("ends_with", "iendswith"):
        return _normalize_value(actual).endswith(_normalize_value(expected))

    if op in ("in",):
        if isinstance(expected, (list, tuple, set)):
            return _normalize_value(actual) in {_normalize_value(v) for v in expected}
        return _normalize_value(actual) == _normalize_value(expected)

    if op in ("gt", "gte", "lt", "lte"):
        try:
            a = float(actual)
            e = float(expected)
            if op == "gt":
                return a > e
            if op == "gte":
                return a >= e
            if op == "lt":
                return a < e
            if op == "lte":
                return a <= e
        except (TypeError, ValueError):
            return False

    return _normalize_value(actual) == _normalize_value(expected)


def item_matches_filters(item: Any, filters: Dict[str, Any]) -> bool:
    """Return True if item satisfies all client-side filters (AND logic)."""
    for field, filter_val in filters.items():
        actual = _get_item_value(item, field)
        if isinstance(filter_val, dict):
            expected = filter_val.get("value")
            operator = filter_val.get("operator", "equals")
        else:
            expected = filter_val
            operator = "equals"

        if not _compare(actual, expected, operator):
            return False
    return True


def _filter_list(items: List[Any], filters: Dict[str, Any]) -> List[Any]:
    if not filters:
        return items
    return [item for item in items if item_matches_filters(item, filters)]


def apply_client_side_filters(
    data: Any,
    client_filters: Dict[str, Any],
) -> Tuple[Any, int]:
    """
    Apply client-side filters to API response data.

    Returns:
        (filtered_data, removed_count)
    """
    if not client_filters or data is None:
        return data, 0

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        original = data["results"]
        filtered = _filter_list(original, client_filters)
        removed = len(original) - len(filtered)
        result = {**data, "results": filtered}
        if "count" in result and isinstance(result["count"], int):
            if data.get("next"):
                result["_client_filter_partial"] = True
                result["_filtered_page_count"] = len(filtered)
            else:
                result["count"] = len(filtered)
        logger.info(
            "Client-side filter: %d -> %d items (removed %d)",
            len(original), len(filtered), removed,
        )
        return result, removed

    if isinstance(data, list):
        filtered = _filter_list(data, client_filters)
        removed = len(data) - len(filtered)
        logger.info(
            "Client-side filter: %d -> %d items (removed %d)",
            len(data), len(filtered), removed,
        )
        return filtered, removed

    if isinstance(data, dict):
        if item_matches_filters(data, client_filters):
            return data, 0
        return None, 1

    return data, 0


# =============================================================================
# Section 4: Parameter Validation
# =============================================================================

def _resolve_resource(parsed: Dict[str, Any], schema: Dict[str, Any]) -> str:
    """Map parsed resource to a schema resource name via hints synonyms."""
    resource = (parsed.get("resource") or "").lower()
    resources = (schema or {}).get("resources") or {}
    hints = (schema or {}).get("resource_hints") or {}

    if resource in resources:
        return resource

    parsed_norm = resource.replace("-", "_").replace(" ", "_")
    for name in resources:
        if name.lower().replace("-", "_") == parsed_norm:
            return name

    for hint_name, hint_data in hints.items():
        if not isinstance(hint_data, dict):
            continue
        if hint_name.lower().replace("-", "_") == parsed_norm:
            return hint_name
        syns = hint_data.get("__resource_synonyms__") or []
        if not isinstance(syns, list):
            syns = [syns]
        for s in syns:
            if str(s).lower().replace("-", "_").replace(" ", "_") == parsed_norm:
                return hint_name

    return parsed.get("resource", "")


def validate_parsed(
    parsed: Dict[str, Any],
    schema: Dict[str, Any],
    query: str = "",
    user_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    """
    Validate and lightly repair parsed query.

    Returns:
        (repaired_parsed, warnings, clarification_message or None)
    """
    if not isinstance(parsed, dict):
        return parsed, [], None

    result = dict(parsed)
    warnings: List[str] = []
    clarification: Optional[str] = None
    schema = schema or {}

    if result.get("resource"):
        resolved = _resolve_resource(result, schema)
        if resolved and resolved != result.get("resource"):
            result["resource"] = resolved

    if result.get("limit") is not None:
        try:
            result["limit"] = int(result["limit"])
        except (TypeError, ValueError):
            warnings.append(f"Invalid limit {result['limit']!r}; removed")
            result.pop("limit", None)

    hints = schema.get("resource_hints") or {}
    if user_context:
        result = resolve_user_context_in_parsed(
            result, user_context, query, resource_hints=hints,
        )

    resource = result.get("resource", "")
    if resource and result.get("filters"):
        intro = SchemaIntrospector(schema)
        repaired_filters = {}
        for field, fval in result["filters"].items():
            value = fval.get("value") if isinstance(fval, dict) else fval
            vr = intro.validate_filter_value(resource, field, value)
            if vr.corrected_value is not None and vr.corrected_value != value:
                if isinstance(fval, dict):
                    repaired_filters[field] = {**fval, "value": vr.corrected_value}
                else:
                    repaired_filters[field] = vr.corrected_value
            else:
                repaired_filters[field] = fval
            warnings.extend(vr.warnings or [])
        result["filters"] = repaired_filters

    if not user_context and query_implies_user_scope(query):
        if result.get("question_type") != "needs_clarification":
            clarification = (
                "This query refers to you ('me' / 'my') but no user context was provided. "
                "Pass user_context with user_id, or rephrase with a specific identifier."
            )
            result["question_type"] = "needs_clarification"

    if result.get("question_type") == "needs_clarification" and not clarification:
        clarification = (
            "I need more information to complete this query. "
            "Please be more specific about the resource, filters, or user context."
        )

    return result, warnings, clarification
