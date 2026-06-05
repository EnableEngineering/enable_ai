"""
Client-side filtering when the API does not support a requested filter param.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import setup_logger

logger = setup_logger("enable_ai.post_filter")


def _normalize(value: Any) -> str:
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

    # Django-style status__name -> item["status"]["name"] or item["status_name"]
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
            return _normalize(actual.get("name")) == _normalize(expected)
        if isinstance(expected, bool):
            return bool(actual) == expected
        return _normalize(actual) == _normalize(expected)

    if op in ("not_equals", "ne"):
        return not _compare(actual, expected, "equals")

    if op in ("contains", "icontains"):
        return _normalize(expected) in _normalize(actual)

    if op in ("starts_with", "istartswith"):
        return _normalize(actual).startswith(_normalize(expected))

    if op in ("ends_with", "iendswith"):
        return _normalize(actual).endswith(_normalize(expected))

    if op in ("in",):
        if isinstance(expected, (list, tuple, set)):
            return _normalize(actual) in {_normalize(v) for v in expected}
        return _normalize(actual) == _normalize(expected)

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

    return _normalize(actual) == _normalize(expected)


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
