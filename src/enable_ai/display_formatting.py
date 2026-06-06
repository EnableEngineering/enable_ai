"""
Human-friendly formatting for summary values and list/table cells.
"""

from typing import Any, Dict, List, Optional, Set

_CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
}


def get_currency_code(resource: str, resource_hints: Dict[str, Any]) -> str:
    hints = (resource_hints or {}).get(resource) or {}
    if isinstance(hints, dict):
        code = hints.get("__currency_code__")
        if code:
            return str(code).upper()
    return "USD"


def get_currency_fields(resource: str, resource_hints: Dict[str, Any]) -> Set[str]:
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return set()
    fields = hints.get("__currency_fields__") or []
    return {str(f) for f in fields} if isinstance(fields, (list, tuple)) else set()


def get_list_display_labels(resource: str, resource_hints: Dict[str, Any]) -> Dict[str, str]:
    hints = (resource_hints or {}).get(resource) or {}
    if not isinstance(hints, dict):
        return {}
    labels = hints.get("__list_display_labels__") or {}
    if isinstance(labels, dict):
        return {str(k): str(v) for k, v in labels.items()}
    return {}


def _looks_like_currency_field(field: str) -> bool:
    name = (field or "").lower()
    return any(
        token in name
        for token in ("amount", "total", "overdue", "outstanding", "collected", "balance", "price")
    )


def _coerce_numeric(val: Any) -> Optional[float]:
    """Parse int/float or numeric strings (e.g. DRF decimal fields as \"51364.00\")."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        stripped = val.strip().replace(",", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def format_field_label(field: str, resource_hints: Dict[str, Any], resource: str = "") -> str:
    labels = get_list_display_labels(resource, resource_hints)
    if field in labels:
        return labels[field]
    if "." in field:
        return field.split(".")[-1].replace("_", " ").title()
    return field.replace("__", ".").replace("_", " ").title()


def format_display_value(
    val: Any,
    field: str = "",
    resource: str = "",
    resource_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a value for user-facing summary/table text."""
    hints = resource_hints or {}
    if val is None:
        return ""

    if isinstance(val, bool):
        if "active" in (field or "").lower():
            return "Active" if val else "Inactive"
        return "Yes" if val else "No"

    if isinstance(val, dict):
        for key in ("name", "label", "title"):
            if val.get(key) is not None:
                return str(val[key])
        if "amount" in val and len(val) <= 3:
            return format_display_value(val["amount"], f"{field}.amount", resource, hints)
        return str(val)

    res_hints = (hints or {}).get(resource) or {}
    has_currency_config = bool(res_hints.get("__currency_fields__")) or bool(
        res_hints.get("__currency_code__"),
    )
    currency_fields = get_currency_fields(resource, hints)
    is_money = field in currency_fields or (
        has_currency_config and _looks_like_currency_field(field)
    )
    num = _coerce_numeric(val)
    if is_money and num is not None:
        code = get_currency_code(resource, hints)
        sym = _CURRENCY_SYMBOLS.get(code, f"{code} ")
        if num == int(num):
            return f"{sym}{int(num):,}"
        return f"{sym}{num:,.2f}"

    if num is not None and isinstance(val, str):
        if num == int(num):
            return str(int(num))
        return str(num)

    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)
