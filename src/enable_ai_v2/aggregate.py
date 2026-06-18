"""
Aggregate query formatting — count/group without LLM guessing.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import IntentPhrases
from .tool_args import extract_list_rows
from .tool_filter import _tool_resource_segment


TECHNICIAN_FIELDS = (
    "technician_name",
    "technician",
    "assigned_technician",
    "assigned_to",
    "technician_id",
)


def _technician_label(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, dict):
        for key in ("display_name", "name", "full_name", "username"):
            if value.get(key):
                return str(value[key])
        return str(value.get("id", "Unknown"))
    return str(value)


def _user_label(user: dict) -> str:
    if user.get("first_name") or user.get("last_name"):
        parts = [str(user.get("first_name") or "").strip(), str(user.get("last_name") or "").strip()]
        combined = " ".join(p for p in parts if p)
        if combined:
            return combined
    for key in ("full_name", "display_name", "username", "email", "name"):
        if user.get(key):
            return str(user[key])
    return str(user.get("id", "Unknown"))


def _collect_assigned_technicians(service_orders: list[dict]) -> tuple[set[Any], set[str]]:
    assigned_ids: set[Any] = set()
    assigned_names: set[str] = set()
    for item in service_orders:
        if not isinstance(item, dict):
            continue
        for field in TECHNICIAN_FIELDS:
            if field not in item or item[field] is None:
                continue
            val = item[field]
            if isinstance(val, dict):
                if val.get("id") is not None:
                    assigned_ids.add(val["id"])
                assigned_names.add(_technician_label(val).lower())
            else:
                assigned_names.add(str(val).lower())
                try:
                    assigned_ids.add(int(val))
                except (TypeError, ValueError):
                    assigned_ids.add(val)
            break
    return assigned_ids, assigned_names


def format_technician_availability(
    users: list[dict],
    service_orders: list[dict],
    query: str,
    phrases: IntentPhrases | None = None,
) -> Optional[str]:
    """Synthesize 'technicians with no jobs' from directory + open SO list (Q13)."""
    from .query_intent import is_technician_availability_query

    if not is_technician_availability_query(query, phrases):
        return None
    if not users:
        return None

    assigned_ids, assigned_names = _collect_assigned_technicians(service_orders)
    free: list[str] = []
    for user in users:
        if not isinstance(user, dict):
            continue
        uid = user.get("id")
        label = _user_label(user)
        if uid in assigned_ids:
            continue
        if label.lower() in assigned_names:
            continue
        free.append(label)

    if free:
        lines = ["Technicians with no assigned service orders:"]
        for name in free[:15]:
            lines.append(f"  • {name}")
        if len(free) > 15:
            lines.append(f"  ... and {len(free) - 15} more")
        return "\n".join(lines)

    if service_orders:
        return "All listed technicians appear to have at least one assigned service order."

    names = [_user_label(u) for u in users[:10] if isinstance(u, dict)]
    if not names:
        return None
    lines = ["Technicians in directory:"]
    lines.extend(f"  • {name}" for name in names)
    lines.append("\nNo open service orders were returned to compare workload.")
    return "\n".join(lines)


def aggregate_service_orders_by_technician(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        tech = None
        for field in TECHNICIAN_FIELDS:
            if field in item and item[field] is not None:
                tech = _technician_label(item[field])
                break
        if tech is None:
            tech = "Unassigned"
        counts[tech] = counts.get(tech, 0) + 1
    return counts


def format_technician_ranking(items: list[dict], *, query: str = "") -> Optional[str]:
    """Format 'who has the most service orders' style answers from real counts."""
    counts = aggregate_service_orders_by_technician(items)
    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_name, top_count = ranked[0]
    lines = [f"{top_name} has the most service orders ({top_count}).", "", "Breakdown:"]
    for name, count in ranked[:8]:
        noun = "service order" if count == 1 else "service orders"
        lines.append(f"  • {name}: {count} {noun}")
    if len(ranked) > 8:
        lines.append(f"  ... and {len(ranked) - 8} more technicians")
    return "\n".join(lines)


def format_compound_response(
    results: list[dict],
    query: str,
    tool_calls: Optional[list] = None,
    phrases: IntentPhrases | None = None,
) -> Optional[str]:
    """Summarize each resource section for compound queries (Q19)."""
    from .query_intent import is_compound_query

    if not is_compound_query(query, phrases) or not tool_calls or len(tool_calls) < 2:
        return None

    sections: list[str] = []
    for tc, result in zip(tool_calls, results):
        if "error" in result:
            sections.append(f"{getattr(tc, 'name', 'tool')}: error")
            continue
        rows = extract_list_rows(result.get("data"), max_rows=500)
        title = str(getattr(tc, "name", "results")).replace("_", " ").title()
        if not rows:
            sections.append(f"{title}: no results")
            continue
        if len(rows) == 1 and isinstance(rows[0], dict):
            label = (
                rows[0].get("display_name")
                or rows[0].get("name")
                or rows[0].get("invoice_number")
                or rows[0].get("so_sequence")
                or f"id {rows[0].get('id', '?')}"
            )
            sections.append(f"{title}:\n  • {label}")
        else:
            sections.append(f"{title}: {len(rows)} results")

    return "\n\n".join(sections) if len(sections) >= 2 else None


def format_aggregate_response(
    results: list[dict],
    query: str,
    tool_calls: Optional[list] = None,
    phrases: IntentPhrases | None = None,
) -> Optional[str]:
    """Best-effort aggregate formatting from API list payloads."""
    users: list[dict] = []
    orders: list[dict] = []
    all_items: list[dict] = []

    if tool_calls and len(tool_calls) == len(results):
        for tc, result in zip(tool_calls, results):
            if "error" in result:
                continue
            rows = extract_list_rows(result.get("data"), max_rows=500)
            segment = _tool_resource_segment(getattr(tc, "name", ""))
            if "user" in segment or "technician" in segment:
                users.extend(r for r in rows if isinstance(r, dict))
            elif "service_order" in segment:
                orders.extend(r for r in rows if isinstance(r, dict))
            all_items.extend(r for r in rows if isinstance(r, dict))
    else:
        for result in results:
            if "error" in result:
                continue
            all_items.extend(extract_list_rows(result.get("data"), max_rows=500))

    if users:
        avail = format_technician_availability(users, orders, query, phrases=phrases)
        if avail:
            return avail

    compound = format_compound_response(results, query, tool_calls=tool_calls, phrases=phrases)
    if compound:
        return compound

    if not all_items:
        return None

    q = query.lower()
    if any(w in q for w in ("technician", "most", "each", "per technician")):
        return format_technician_ranking(all_items, query=query)

    return None
