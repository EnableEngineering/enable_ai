"""
Format API responses as chat-friendly messages.

Supports multiple format types:
- auto: Let the system choose best format
- concise: Brief summary
- detailed: Full breakdown
- table: Markdown table
- grouped: Grouped by category
- chart: JSON ready for visualization
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from .config import Config, resolve_intent_phrases
from .types import ToolCall
from .tool_filter import is_count_query
from .query_intent import (
    QueryIntent,
    classify_query_intent,
    is_limited_list_args,
    is_technician_availability_query,
    DEFAULT_EMBEDDED_FIELDS,
)
from .date_range import has_duration_phrase, extract_duration_filter

FormatType = Literal["auto", "concise", "detailed", "table", "grouped", "chart"]

_TOOL_ACTIONS = frozenset({
    "list", "retrieve", "create", "update", "destroy", "partial", "get", "read", "search",
})

# Per-resource display label priority (first non-empty wins)
_RESOURCE_LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    "service-orders": (
        "display_name", "name", "so_sequence", "req_id", "title", "description", "id",
    ),
    "service_orders": (
        "display_name", "name", "so_sequence", "req_id", "title", "description", "id",
    ),
    "invoices": (
        "invoice_number", "number", "display_name", "name", "company_name", "id",
    ),
    "invoicing": (
        "invoice_number", "number", "display_name", "name", "company_name", "id",
    ),
    "users": (
        "full_name", "display_name", "first_name", "last_name", "username", "email", "name", "id",
    ),
    "technicians": (
        "full_name", "display_name", "first_name", "last_name", "username", "email", "name", "id",
    ),
}

_STATUS_FIELDS = (
    "status", "status_name", "state", "status__name", "service_order_status",
)

_BANNED_SUGGESTIONS = frozenset({
    "show more details about this",
    "filter these results by...",
    "show more details about this.",
})

_DEFAULT_LABEL_FIELDS = (
    "display_name", "name", "title", "label", "so_sequence", "invoice_number",
    "req_id", "number", "description", "id",
)


class ResponseFormatter:
    """Format API results as conversational responses."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._intent_phrases = resolve_intent_phrases(self.config)

        # Configurable field lists
        self.name_fields = ["name", "title", "label", "description", "id"]
        self.group_fields = ["type", "category", "status", "group", "kind"]
        self.priority_fields = ["name", "title", "id", "type", "status", "amount", "date"]

    def format_error(self, error: str, query: str) -> str:
        """Format an error as a chat message."""
        return f"I couldn't complete that request. {error}"

    def format_clarification(
        self,
        question: str,
        options: Optional[list[str]] = None,
    ) -> str:
        """Format a clarification question."""
        msg = question
        if options:
            msg += "\n\nOptions:\n"
            for i, opt in enumerate(options, 1):
                msg += f"  {i}. {opt}\n"
        return msg

    def format_validation_errors(self, errors: list[str]) -> str:
        """Format validation errors as a chat message."""
        if len(errors) == 1:
            return f"I need more information: {errors[0]}"

        msg = "I need some clarification:\n"
        for error in errors:
            msg += f"  • {error}\n"
        return msg

    def format_no_tools(self, query: str) -> str:
        """Format response when no tools were called."""
        return (
            "I'm not sure how to help with that. Could you rephrase your request? "
            "Try asking about specific data or actions available in this system."
        )

    def format_api_error(
        self,
        tool_name: str,
        error: str,
        status_code: Optional[int] = None,
        hint: Optional[str] = None,
        role: Optional[str] = None,
    ) -> str:
        """Format an API error."""
        if status_code == 401:
            msg = "I couldn't access that data. You may need to log in again."
        elif status_code == 403:
            msg = "You don't have permission to access that data."
            if not hint:
                hint = self._permission_hint_for_role(role)
        elif status_code == 404:
            msg = "I couldn't find what you're looking for. It may not exist."
        elif status_code and status_code >= 500:
            msg = "The server encountered an error. Please try again later."
        else:
            msg = f"Something went wrong while getting that data: {error}"

        if hint:
            msg = f"{msg} {hint}"
        return msg

    @staticmethod
    def _permission_hint_for_role(role: Optional[str]) -> Optional[str]:
        if not role:
            return "Try asking about data scoped to your role."
        hints = {
            "customer": "Try asking about your own service orders or invoices.",
            "technician": "Try asking about service orders assigned to you.",
            "accountant": "Try asking about invoices or billing records you can access.",
            "admin": "This endpoint may be restricted even for admins.",
        }
        return hints.get(role.lower(), "Try asking about data available to your role.")

    def format_success_simple(
        self,
        data: Any,
        tool_name: str,
        query: str = "",
        tool_args: Optional[dict] = None,
        intent: Optional[QueryIntent] = None,
    ) -> str:
        """Format a simple success response (for caching, no LLM needed)."""
        if data is None:
            return "Done."

        intent = intent or classify_query_intent(query, self._intent_phrases)
        tool_args = tool_args or {}

        ar_msg = self._format_ar_dashboard(data, tool_name, query)
        if ar_msg:
            return ar_msg
        if self._is_ar_context(tool_name, query):
            rows, _ = self._unwrap_list_data(data)
            if rows and isinstance(rows[0], dict):
                return self._format_result_item(rows[0], query=query, tool_name=tool_name)
            # Don't return AR-specific message for simple list queries
            # Let it fall through to standard list/count handling

        if is_technician_availability_query(query, self._intent_phrases) and self._is_user_or_technician_tool(tool_name):
            rows, _ = self._unwrap_list_data(data)
            from .aggregate import format_technician_availability

            avail = format_technician_availability(rows, [], query, phrases=self._intent_phrases)
            if avail:
                return avail

        limited = is_limited_list_args(tool_args)
        use_count = intent == QueryIntent.COUNT or (
            is_count_query(query) and not limited
        )
        if intent == QueryIntent.COUNT or is_count_query(query):
            use_count = True

        # Unwrap DRF paginated response
        if isinstance(data, dict) and "results" in data and "count" in data:
            results = data["results"]
            count = data.get("count", len(results) if isinstance(results, list) else 0)
            if isinstance(results, list):
                if len(results) == 0 or count == 0:
                    return "No results found."
                if use_count:
                    label = self._resource_label(tool_name)
                    msg = f"You have {count} {self._count_noun(label, count)}."
                    # Add sample rows for context (only when we have 2+ results)
                    # Skip if page_size=1 was used (efficiency fetch, not real sample)
                    if count > 1 and len(results) > 1:
                        sample_size = getattr(self.config, "count_sample_size", 0)
                        sample_items = results[:sample_size] if sample_size > 0 else results
                        sample = self._format_sample_rows(sample_items, tool_name)
                        if sample:
                            msg = f"{msg}\n\n{sample}"
                    return self._append_duration_note(msg, query, tool_args, count)
                if len(results) == 1:
                    item_msg = self._format_result_item(results[0], query=query, tool_name=tool_name)
                    return self._append_duration_note(item_msg, query, tool_args, count)
                list_msg = self._format_list(results, total_count=count, tool_name=tool_name)
                return self._append_duration_note(list_msg, query, tool_args, count)

        if isinstance(data, list):
            if len(data) == 0:
                return "No results found."
            if use_count and not limited:
                label = self._resource_label(tool_name)
                return f"You have {len(data)} {self._count_noun(label, len(data))}."
            if len(data) == 1:
                return self._format_result_item(data[0], query=query, tool_name=tool_name)
            return self._format_list(data, tool_name=tool_name)

        if isinstance(data, dict):
            if use_count and not limited:
                label = self._resource_label(tool_name)
                return f"You have 1 {self._count_noun(label, 1)}."
            return self._format_result_item(data, query=query, tool_name=tool_name)

        return str(data)

    def _append_duration_note(
        self,
        message: str,
        query: str,
        tool_args: dict,
        result_count: int,
    ) -> str:
        """Note when duration was requested but filters may not have applied (Q23)."""
        if not has_duration_phrase(query):
            return message
        duration = extract_duration_filter(query)
        if duration and not any(k in tool_args for k in duration):
            if result_count > 5:
                return (
                    f"{message}\n\n"
                    "Note: this API may not support age/duration filters; "
                    "results may include all matching status rows, not only those older than the requested period."
                )
        return message

    def _is_ar_context(self, tool_name: str, query: str) -> bool:
        name = tool_name.lower()
        q = query.lower()
        if "ar_dashboard" in name or "ar-dashboard" in name.replace("_", "-"):
            return True
        if "dashboard" in name and "ar" in name:
            return True
        if "invoicing" in name and "ar" in q:
            return True
        if "receivable" in name or "accounts_receivable" in name:
            return True
        ar_keywords = getattr(self._intent_phrases, "ar_context_keywords", (
            "outstanding", "ar dashboard", "accounts receivable", "highest outstanding", "overdue"
        ))
        return any(w in q for w in ar_keywords)

    @staticmethod
    def _is_user_or_technician_tool(tool_name: str) -> bool:
        name = tool_name.lower()
        if "user" in name or "technician" in name:
            return True
        parts = name.replace("-", "_").split("_")
        resource = [p for p in parts if p not in _TOOL_ACTIONS]
        joined = "_".join(resource)
        return joined in ("users", "user", "technicians", "technician")

    def _format_ar_dashboard(
        self,
        data: Any,
        tool_name: str,
        query: str,
    ) -> Optional[str]:
        """Format AR dashboard / outstanding balance responses (Q21/Q56)."""
        if not self._is_ar_context(tool_name, query):
            return None

        rows, _ = self._unwrap_list_data(data)
        if not rows:
            return None

        best_company = None
        best_amount = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            company, amount = self._extract_ar_fields(row)
            if amount is not None and (best_amount is None or float(amount) > float(best_amount)):
                best_amount = amount
                best_company = company or best_company
            elif company and not best_company:
                best_company = company

        q = query.lower()
        wants_top = any(w in q for w in ("highest", "most", "top", "which company", "outstanding"))

        if best_company and best_amount is not None:
            return f"The highest outstanding balance is {best_company} at {best_amount}."
        if best_company and wants_top:
            return f"Top outstanding account: {best_company}."
        if len(rows) == 1 and isinstance(rows[0], dict):
            company, amount = self._extract_ar_fields(rows[0])
            if company or amount is not None:
                parts = []
                if company:
                    parts.append(f"Company: {company}")
                if amount is not None:
                    parts.append(f"Outstanding: {amount}")
                return "\n".join(parts)

        return None

    def _extract_ar_fields(self, row: dict) -> tuple[Optional[str], Any]:
        """Extract company name and amount from AR data using configurable fields."""
        company = None
        amount = None

        # Get field names from config (no hardcoded fallbacks)
        company_fields = getattr(self._intent_phrases, "ar_company_fields", ())
        amount_fields = getattr(self._intent_phrases, "ar_amount_fields", ())

        def extract_name(value: Any) -> Optional[str]:
            """Extract display name from nested object or string."""
            if isinstance(value, dict):
                return value.get("name") or value.get("display_name")
            if isinstance(value, str) and value and not value.isdigit():
                return value
            return None

        # Check configured company fields
        for field in company_fields:
            val = row.get(field)
            if val is not None:
                company = extract_name(val)
                if company:
                    break

        # Check configured amount fields
        for field in amount_fields:
            val = row.get(field)
            if val is not None:
                if isinstance(val, (int, float)):
                    amount = val
                    break
                elif isinstance(val, str):
                    try:
                        amount = float(val.replace(",", ""))
                        break
                    except ValueError:
                        pass

        return company, amount

    @staticmethod
    def _extract_status_value(item: dict) -> Optional[str]:
        for field in _STATUS_FIELDS:
            if field not in item:
                continue
            val = item[field]
            if val is None or val == "":
                continue
            if isinstance(val, dict):
                return str(val.get("name") or val.get("display_name") or val.get("label") or val)
            return str(val)
        return None

    def _format_result_item(
        self,
        item: Any,
        query: str = "",
        tool_name: str = "",
    ) -> str:
        """Format one row, including embedded/nested fields when present."""
        if not isinstance(item, dict):
            return str(item)

        embedded = self._embedded_fields_for_tool(tool_name)
        for field in embedded:
            if field not in item:
                continue
            summary = self._summarize_embedded_field(field, item[field])
            if summary:
                prefix = self._format_single_item(item, tool_name=tool_name, query=query)
                return f"{prefix}\n\n{summary}" if prefix else summary

        return self._format_single_item(item, tool_name=tool_name, query=query)

    def _embedded_fields_for_tool(self, tool_name: str) -> tuple[str, ...]:
        if self.config.embedded_response_fields:
            resource = self._resource_key_from_tool(tool_name)
            for key, fields in self.config.embedded_response_fields.items():
                if self._resource_key_from_tool(key) == resource:
                    return tuple(fields)
        return DEFAULT_EMBEDDED_FIELDS

    def _summarize_embedded_field(self, field: str, value: Any) -> str:
        """Summarize nested list/dict fields (observations, equipment, etc.)."""
        title = self._humanize_key(field)
        if isinstance(value, list):
            if not value:
                return f"{title}: none"
            lines = [f"{title}:"]
            for entry in value[: self.config.max_items_in_response]:
                if isinstance(entry, dict):
                    text = entry.get("text") or entry.get("note") or entry.get("name") or entry.get("description")
                    if text:
                        lines.append(f"  • {text}")
                    else:
                        lines.append(f"  • {self._format_single_item(entry)}")
                else:
                    lines.append(f"  • {entry}")
            remaining = len(value) - self.config.max_items_in_response
            if remaining > 0:
                lines.append(f"  ... and {remaining} more")
            return "\n".join(lines)
        if isinstance(value, dict):
            return f"{title}:\n{self._format_single_item(value)}"
        if isinstance(value, str) and value.strip():
            return f"{title}: {value}"
        return ""

    def format_multi_success(
        self,
        tool_calls: list[ToolCall],
        results: list[dict],
        query: str = "",
        intent: Optional[QueryIntent] = None,
    ) -> str:
        """Format responses from multiple API calls in one query."""
        intent = intent or classify_query_intent(query, self._intent_phrases)

        if intent == QueryIntent.AGGREGATE:
            return ""  # caller should use LLM generate_response

        sections: list[str] = []

        for tc, result in zip(tool_calls, results):
            data = result.get("data")
            if "error" in result:
                sections.append(f"{self._humanize_tool_name(tc.name)}: {result['error']}")
                continue

            section = self.format_success_simple(
                data,
                tc.name,
                query=query,
                tool_args=tc.arguments,
                intent=intent,
            )
            title = self._humanize_tool_name(tc.name)
            sections.append(f"{title}:\n{section}")

        return "\n\n".join(sections)

    def _resource_label(self, tool_name: str) -> str:
        """
        Derive a human label for count responses from tool name and config.

        Labels come from parent resource_hints or generic tool-name parsing —
        not from query stop-word lists (NL understanding stays in the LLM prompt).
        """
        tool_resource = self._resource_key_from_tool(tool_name)
        name_l = tool_name.lower()
        if "ar" in name_l and "dashboard" in name_l:
            return "AR accounts"
        if "receivable" in name_l:
            return "AR accounts"
        for hint_key in self.config.resource_hints:
            if self._resource_key_from_tool(hint_key) == tool_resource:
                return hint_key.replace("-", " ").replace("_", " ")

        parts = tool_name.lower().split("_")
        resource_parts = [p for p in parts if p not in _TOOL_ACTIONS]
        if resource_parts:
            return " ".join(resource_parts)

        return "result"

    @staticmethod
    def _count_noun(label: str, count: int) -> str:
        """Pluralize a resource label for count responses."""
        words = label.split()
        # De-duplicate stuttered labels ("service orders service orders")
        if len(words) >= 2 and len(words) % 2 == 0:
            half = len(words) // 2
            if words[:half] == words[half:]:
                words = words[:half]
        label = " ".join(words)

        if count == 1:
            if words and words[-1].endswith("s") and not words[-1].endswith("ss"):
                words[-1] = words[-1][:-1]
            return " ".join(words)
        if label.endswith("s"):
            return label
        return f"{label}s"

    @staticmethod
    def _resource_key_from_tool(name: str) -> str:
        """Normalize resource identifiers for matching (service_orders -> service-orders)."""
        parts = name.lower().replace("-", "_").split("_")
        resource_parts = [p for p in parts if p not in _TOOL_ACTIONS]
        return "-".join(resource_parts)

    def _humanize_tool_name(self, tool_name: str) -> str:
        return tool_name.replace("_", " ").title()

    def _label_fields_for_tool(self, tool_name: str) -> tuple[str, ...]:
        key = self._resource_key_from_tool(tool_name)
        if key in _RESOURCE_LABEL_FIELDS:
            return _RESOURCE_LABEL_FIELDS[key]
        normalized = key.replace("-", "_")
        for pattern, fields in _RESOURCE_LABEL_FIELDS.items():
            if pattern.replace("-", "_") == normalized:
                return fields
        return _DEFAULT_LABEL_FIELDS

    def _item_display_label(self, item: Any, tool_name: str = "") -> str:
        if not isinstance(item, dict):
            return str(item)
        if item.get("first_name") or item.get("last_name"):
            parts = [str(item.get("first_name") or "").strip(), str(item.get("last_name") or "").strip()]
            combined = " ".join(p for p in parts if p)
            if combined:
                return combined
        for field in self._label_fields_for_tool(tool_name):
            value = item.get(field)
            if value is None or value == "":
                continue
            if isinstance(value, (dict, list)):
                continue
            return str(value)
        return "Item"

    def _format_single_item(self, item: dict, tool_name: str = "", query: str = "") -> str:
        """Format a single item."""
        if not isinstance(item, dict):
            return str(item)

        name = self._item_display_label(item, tool_name)

        # Build a simple summary
        lines = []
        if name and name != "Item":
            lines.append(name)

        status_val = self._extract_status_value(item)
        q = (query or "").lower()
        wants_status = "status" in q or "service_order" in tool_name.lower()
        if status_val and wants_status:
            lines.append(f"Status: {status_val}")

        label_fields = set(self._label_fields_for_tool(tool_name))
        skip_fields = (
            {"id", "created_at", "updated_at", "name", "title", "description"}
            | label_fields
            | set(_STATUS_FIELDS)
        )
        details = []
        for key, value in item.items():
            if key in skip_fields or key.startswith("_"):
                continue
            if value is None or value == "":
                continue
            if isinstance(value, (dict, list)):
                continue
            details.append(f"{self._humanize_key(key)}: {value}")
            if len(details) >= 4:
                break

        if details:
            lines.extend(details)

        return "\n".join(lines) if lines else json.dumps(item, indent=2)

    def _format_list(
        self,
        items: list,
        total_count: Optional[int] = None,
        tool_name: str = "",
    ) -> str:
        """Format a list of items."""
        shown = len(items)
        total = total_count if total_count is not None else shown
        max_show = self.config.max_items_in_response

        lines = [f"Found {total} result{'s' if total != 1 else ''}:"]

        for item in items[:max_show]:
            if isinstance(item, dict):
                lines.append(f"  • {self._item_display_label(item, tool_name)}")
            else:
                lines.append(f"  • {item}")

        remaining = total - min(shown, max_show)
        if remaining > 0:
            lines.append(f"  ... and {remaining} more")

        return "\n".join(lines)

    def _format_sample_rows(
        self,
        items: list,
        tool_name: str = "",
    ) -> str:
        """Format sample rows for COUNT queries (show first few with details)."""
        if not items:
            return ""

        lines: list[str] = []
        # No slicing needed - caller passes already-sliced list (or all items if size=0)
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            # Build rich label: name | status | priority | company
            parts: list[str] = []

            # Primary label
            label = self._item_display_label(item, tool_name)
            parts.append(label)

            # Status if available
            for sf in _STATUS_FIELDS:
                if item.get(sf):
                    val = item[sf]
                    if isinstance(val, dict):
                        val = val.get("name") or val.get("display_name") or str(val)
                    parts.append(str(val))
                    break

            # Priority if available
            if item.get("priority"):
                prio = item["priority"]
                if isinstance(prio, dict):
                    prio = prio.get("name") or prio.get("display_name") or str(prio)
                parts.append(f"{prio} priority")

            # Company if available
            for cf in ("company", "customer", "company_name", "customer_name"):
                if item.get(cf):
                    val = item[cf]
                    if isinstance(val, dict):
                        val = val.get("name") or val.get("display_name") or str(val)
                    parts.append(str(val))
                    break

            lines.append(f"{i}. {' | '.join(parts)}")

        return "\n".join(lines) if lines else ""

    def _humanize_key(self, key: str) -> str:
        """Convert snake_case to Title Case."""
        return key.replace("_", " ").replace("-", " ").title()

    def build_suggestions(
        self,
        query: str,
        tool_calls: list[ToolCall],
        results: list[dict],
    ) -> list[str]:
        """Generate deduplicated, context-aware follow-up suggestions."""
        intent = classify_query_intent(query, self._intent_phrases)
        suggestions: list[str] = []
        seen: set[str] = set()

        def add(text: str) -> None:
            key = " ".join(text.lower().strip().rstrip(".").split())
            if not text or key in seen or key in _BANNED_SUGGESTIONS:
                return
            if key.startswith("show more details about this"):
                return
            for existing in seen:
                if key in existing or existing in key:
                    return
            seen.add(key)
            suggestions.append(text)

        for tc, result in zip(tool_calls, results):
            if "error" in result:
                continue
            data = result.get("data", {})
            items, total = self._unwrap_list_data(data)
            count = total if total is not None else len(items)
            resource = self._resource_label(tc.name)

            if count > 1 and "list" in tc.name.lower():
                add(f"Show details for these {count} {resource}")
                labels = [
                    self._item_display_label(row, tc.name)
                    for row in items[:3]
                    if isinstance(row, dict)
                ]
                labels = [l for l in labels if l and l != "Item"]
                if labels:
                    add(f"Open {labels[0]}")
                add(f"Filter these {resource} by status or company")
            elif count == 1 and items and isinstance(items[0], dict):
                label = self._item_display_label(items[0], tc.name)
                if label != "Item":
                    add(f"Show full details for {label}")

        if intent == QueryIntent.COUNT and results:
            data = results[0].get("data", {})
            items, total = self._unwrap_list_data(data)
            total = total or len(items)
            if total and total > 1:
                add("Show details of those")
                add("List the first 5 of those")

        return suggestions[:3]

    # -------------------------------------------------------------------------
    # Advanced formatting methods
    # -------------------------------------------------------------------------

    def format_response(
        self,
        data: Any,
        query: str,
        format_type: FormatType = "auto",
    ) -> dict[str, Any]:
        """
        Format API response with specified format type.

        Args:
            data: The API response data
            query: Original user query
            format_type: Format type (auto, concise, detailed, table, grouped, chart)

        Returns:
            Dict with format, summary, formatted, and raw_data
        """
        # Handle empty/null data
        items, total_count = self._unwrap_list_data(data)
        if not items and (total_count is None or total_count == 0):
            return {
                "format": "text",
                "summary": "No results found.",
                "formatted": "No results found.",
                "raw_data": data,
            }

        # Normalize paginated responses
        items, total_count = self._unwrap_list_data(data)

        # Auto-select format
        if format_type == "auto":
            format_type = self._auto_select_format(items, query)

        # Format based on type
        if format_type == "table":
            formatted = self._format_table(items)
        elif format_type == "grouped":
            formatted = self._format_grouped(items)
        elif format_type == "chart":
            formatted = self._format_chart(items)
        elif format_type == "detailed":
            formatted = self._format_detailed(items)
        else:  # concise
            formatted = self._format_concise(items)

        summary = self._generate_summary(items, query, total_count=total_count)

        return {
            "format": format_type,
            "summary": summary,
            "formatted": formatted,
            "raw_data": data,
        }

    def _unwrap_list_data(self, data: Any) -> tuple[list, Optional[int]]:
        """Unwrap list data and optional total count from API payloads."""
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            items = results if isinstance(results, list) else [results]
            total = data.get("count") if "count" in data else None
            return items, total
        if isinstance(data, list):
            return data, None
        if isinstance(data, dict):
            return [data], None
        return [data], None

    def _auto_select_format(self, items: list, query: str) -> FormatType:
        """Auto-select the best format based on data and query."""
        if not items:
            return "concise"

        # Single item -> detailed
        if len(items) == 1:
            return "detailed"

        # Check for grouping keywords in query
        query_lower = query.lower()
        if any(w in query_lower for w in ["group", "by type", "by status", "by category"]):
            return "grouped"

        # Check for chart keywords
        if any(w in query_lower for w in ["chart", "graph", "visualize", "plot"]):
            return "chart"

        # List of items -> table for comparison
        if len(items) <= 20 and all(isinstance(i, dict) for i in items):
            return "table"

        # Default to concise for large datasets
        return "concise"

    def _format_table(self, items: list) -> str:
        """Format items as a markdown table."""
        if not items or not isinstance(items[0], dict):
            return self._format_concise(items)

        # Get columns (prioritize common fields)
        all_keys = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        # Order columns by priority
        columns = []
        for field in self.priority_fields:
            if field in all_keys:
                columns.append(field)
                all_keys.discard(field)

        # Add remaining columns (up to 6 total)
        for key in sorted(all_keys):
            if len(columns) >= 6:
                break
            if not key.startswith("_"):
                columns.append(key)

        if not columns:
            return self._format_concise(items)

        # Build table
        lines = []

        # Header
        header = "| " + " | ".join(self._humanize_key(c) for c in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        lines.append(header)
        lines.append(separator)

        # Rows
        max_rows = self.config.max_items_in_response
        for item in items[:max_rows]:
            if not isinstance(item, dict):
                continue
            row_values = []
            for col in columns:
                value = item.get(col, "")
                if isinstance(value, (dict, list)):
                    value = "..."
                row_values.append(str(value)[:50])  # Truncate long values
            lines.append("| " + " | ".join(row_values) + " |")

        if len(items) > max_rows:
            lines.append(f"\n*...and {len(items) - max_rows} more rows*")

        return "\n".join(lines)

    def _format_grouped(self, items: list) -> str:
        """Format items grouped by a category field."""
        if not items:
            return "No items to group."

        # Find the best grouping field
        group_field = None
        for field in self.group_fields:
            if isinstance(items[0], dict) and field in items[0]:
                group_field = field
                break

        if not group_field:
            return self._format_concise(items)

        # Group items
        groups: dict[str, list] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get(group_field, "Other"))
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        # Format groups
        lines = [f"Results grouped by {self._humanize_key(group_field)}:\n"]

        for group_name, group_items in sorted(groups.items()):
            lines.append(f"**{group_name}** ({len(group_items)} items)")
            for item in group_items[:5]:
                name = self._get_item_name(item)
                lines.append(f"  • {name}")
            if len(group_items) > 5:
                lines.append(f"  • ...and {len(group_items) - 5} more")
            lines.append("")

        return "\n".join(lines)

    def _format_chart(self, items: list) -> str:
        """Format items as chart-ready JSON."""
        if not items:
            return json.dumps({"labels": [], "values": []})

        # Find grouping field for aggregation
        group_field = None
        for field in self.group_fields:
            if isinstance(items[0], dict) and field in items[0]:
                group_field = field
                break

        if group_field:
            # Aggregate by group
            counts: dict[str, int] = {}
            for item in items:
                if isinstance(item, dict):
                    key = str(item.get(group_field, "Other"))
                    counts[key] = counts.get(key, 0) + 1

            chart_data = {
                "type": "bar",
                "labels": list(counts.keys()),
                "values": list(counts.values()),
                "title": f"Count by {self._humanize_key(group_field)}",
            }
        else:
            # Simple count
            chart_data = {
                "type": "summary",
                "total": len(items),
            }

        return json.dumps(chart_data, indent=2)

    def _format_detailed(self, items: list) -> str:
        """Format items with full details."""
        lines = []

        for i, item in enumerate(items[:self.config.max_items_in_response]):
            if not isinstance(item, dict):
                lines.append(f"{i+1}. {item}")
                continue

            name = self._get_item_name(item)
            lines.append(f"### {name}")

            for key, value in item.items():
                if key.startswith("_"):
                    continue
                if value is None or value == "":
                    continue
                if isinstance(value, dict):
                    value = json.dumps(value)
                elif isinstance(value, list):
                    value = ", ".join(str(v) for v in value[:5])
                    if len(item[key]) > 5:
                        value += f" (+{len(item[key]) - 5} more)"

                lines.append(f"- **{self._humanize_key(key)}**: {value}")

            lines.append("")

        if len(items) > self.config.max_items_in_response:
            lines.append(f"*...and {len(items) - self.config.max_items_in_response} more items*")

        return "\n".join(lines)

    def _format_concise(self, items: list) -> str:
        """Format items as a brief summary."""
        if not items:
            return "No results."

        if len(items) == 1:
            return self._format_single_item(items[0])

        return self._format_list(items)

    def _generate_summary(
        self,
        items: list,
        query: str,
        total_count: Optional[int] = None,
    ) -> str:
        """Generate a brief summary of the results."""
        if not items and (total_count is None or total_count == 0):
            return "No results found."

        count = total_count if total_count is not None else len(items)
        if count == 1 and items:
            name = self._get_item_name(items[0])
            return f"Found: {name}"

        return f"Found {count} results."

    def _get_item_name(self, item: dict, tool_name: str = "") -> str:
        """Get the display name for an item."""
        if not isinstance(item, dict):
            return str(item)
        label = self._item_display_label(item, tool_name)
        return label if label != "Item" else "Item"
