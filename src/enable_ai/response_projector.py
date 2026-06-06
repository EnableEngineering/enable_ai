"""
Config-driven list projection and chat-window display.

Uses resource_hints.__list_display_fields__ and __chat_window_size__ to slim
formatter input and paginate list results in-session without API next_url.
"""

from typing import Any, Dict, List, Optional, Tuple

from . import constants
from .utils import setup_logger

logger = setup_logger("enable_ai.response_projector")


def get_list_display_fields(resource: str, resource_hints: Dict[str, Any]) -> List[str]:
    """Fields to show per row from resource_hints.__list_display_fields__."""
    if not resource or not resource_hints:
        return []
    hints = resource_hints.get(resource) or {}
    if not isinstance(hints, dict):
        return []
    fields = hints.get("__list_display_fields__") or []
    return [str(f) for f in fields] if isinstance(fields, (list, tuple)) else []


def get_chat_window_size(resource: str, resource_hints: Dict[str, Any]) -> int:
    """Rows per chat window from hints or ENABLE_AI_CHAT_WINDOW_SIZE."""
    if resource and resource_hints:
        hints = resource_hints.get(resource) or {}
        if isinstance(hints, dict):
            size = hints.get("__chat_window_size__")
            if size is not None:
                try:
                    return max(1, int(size))
                except (TypeError, ValueError):
                    pass
    return constants.CHAT_WINDOW_SIZE


def extract_raw_items(data: Any) -> List[Dict[str, Any]]:
    """Pull list rows from API response shapes."""
    if not data:
        return []
    if isinstance(data, dict):
        for key in ("results", "items"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        if "id" in data:
            return [data]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [r for r in data["results"] if isinstance(r, dict)]
    return []


def _resolve_field_value(item: Dict[str, Any], field: str) -> Any:
    """Resolve dotted paths and simple nested dicts (e.g. role.name)."""
    if field in item:
        val = item[field]
        if isinstance(val, dict):
            for k in ("name", "label", "title", "id"):
                if val.get(k) is not None:
                    return val[k]
        return val
    if "." in field:
        current = item
        for part in field.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        if isinstance(current, dict):
            return current.get("name") or current.get("id") or current
        return current
    base = field.split("__")[0]
    if base != field and base in item:
        val = item[base]
        if isinstance(val, dict):
            return val.get("name") or val.get("id") or val
    return None


def project_row(item: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """Slim one row to configured display fields."""
    if not fields:
        return dict(item)
    projected: Dict[str, Any] = {}
    for field in fields:
        val = _resolve_field_value(item, field)
        if val is not None:
            projected[field] = val
    if not projected and item.get("id") is not None:
        projected["id"] = item["id"]
    return projected


def project_items(
    items: List[Dict[str, Any]],
    resource: str,
    resource_hints: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Project all items; returns (projected_rows, field_names)."""
    fields = get_list_display_fields(resource, resource_hints)
    if not fields:
        return items, []
    return [project_row(item, fields) for item in items], fields


def apply_chat_window(
    items: List[Dict[str, Any]],
    offset: int,
    window_size: int,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """
    Slice items for the current chat window.

    Returns (window_items, new_offset, has_more_in_chat).
    """
    offset = max(0, int(offset or 0))
    window_size = max(1, int(window_size or constants.CHAT_WINDOW_SIZE))
    window = items[offset: offset + window_size]
    has_more = (offset + window_size) < len(items)
    return window, offset, has_more


def format_projected_table(
    rows: List[Dict[str, Any]],
    fields: List[str],
    *,
    resource: str = "items",
    resource_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a deterministic markdown table from projected rows (no LLM)."""
    from .display_formatting import format_display_value, format_field_label

    if not rows:
        return f"No {resource} found."
    if not fields:
        fields = list(rows[0].keys())[: constants.TABLE_FIELDS_MAX]
    hints = resource_hints or {}
    header_labels = [format_field_label(f, hints, resource) for f in fields]
    lines = [
        "| " + " | ".join(header_labels) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        cells = []
        for field in fields:
            val = row.get(field, "")
            if val is None:
                val = ""
            else:
                val = format_display_value(val, field, resource, hints)
            cells.append(str(val)[: constants.TABLE_FIELD_PREVIEW])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_chat_summary(
    window_rows: List[Dict[str, Any]],
    fields: List[str],
    *,
    resource: str,
    offset: int,
    window_size: int,
    total_cached: int,
    total_count: Optional[int] = None,
    has_more_in_chat: bool = False,
    resource_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """Human-readable summary for a chat window."""
    shown = len(window_rows)
    total = total_count if total_count is not None else total_cached
    start = offset + 1 if shown else 0
    end = offset + shown
    table = format_projected_table(
        window_rows, fields, resource=resource, resource_hints=resource_hints,
    )
    header = f"Showing {start}–{end} of {total} {resource.replace('-', ' ')}"
    if has_more_in_chat:
        header += " (say \"show me next\" or \"next 10\" for more)"
    elif total_cached < total:
        header += " (more available from the API)"
    return f"{header}\n\n{table}"


class ResponseProjector:
    """Apply projection + chat windows using schema resource_hints."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema or {}
        self.resource_hints = self.schema.get("resource_hints") or {}

    def prepare_list_display(
        self,
        data: Any,
        resource: str,
        *,
        display_mode: str = "summary",
        chat_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Build projected list cache and optional chat window for display.

        display_mode=full → all projected rows, no window.
        summary/detailed → chat window slice.
        """
        raw_items = extract_raw_items(data)
        projected, fields = project_items(raw_items, resource, self.resource_hints)
        window_size = get_chat_window_size(resource, self.resource_hints)
        total_count = None
        if isinstance(data, dict):
            for key in ("count", "total_count", "total"):
                if data.get(key) is not None:
                    try:
                        total_count = int(data[key])
                        break
                    except (TypeError, ValueError):
                        pass

        if display_mode == "full":
            window = projected
            offset = 0
            has_more_in_chat = False
        else:
            window, offset, has_more_in_chat = apply_chat_window(
                projected, chat_offset, window_size,
            )

        return {
            "raw_items": raw_items,
            "list_cache": projected,
            "window_items": window,
            "list_display_fields": fields,
            "chat_offset": offset,
            "chat_window_size": window_size,
            "has_more_in_chat": has_more_in_chat,
            "total_cached": len(projected),
            "total_count": total_count if total_count is not None else len(projected),
        }
