"""
Response Formatter - Intelligent formatting of API responses

Provides context-aware formatting including:
- Concise text summaries (data-bound, generic wording)
- Markdown tables for structured data
- Chart-ready JSON for visualizations
- Grouping and categorization

Summarization is designed to be generic and accurate: it uses only the provided
data and query, includes exact counts, and avoids domain-specific or invented content.
"""

from typing import Dict, Any, List, Optional, Union, Literal
import json
from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP
from . import constants

logger = setup_logger(__name__)

FormatType = Literal["auto", "concise", "detailed", "table", "chart", "grouped"]


class ResponseFormatter:
    """
    Response formatter: chooses presentation format and produces summaries.
    Summaries are data-bound (no invented facts) and use generic wording
    unless the query or data clearly indicates a specific type.
    """
    
    def __init__(self, model: str = "gpt-4o", config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model: OpenAI model name.
            config: Optional configuration dict to customize formatting behavior.
                    Supported keys (all optional, with sensible defaults):
                        - valid_formats: list[str]
                        - group_fields: list[str]
                        - name_fields: list[str]
                        - chart_group_fields: list[str]
        """
        self.client = get_openai_client()
        self.model = model
        self.config = config or {}
        logger.info(f"ResponseFormatter initialized with model: {self.model}")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _get_valid_formats(self) -> List[str]:
        return self.config.get(
            "valid_formats",
            ["concise", "table", "grouped", "chart", "detailed"],
        )

    def _get_group_fields(self) -> List[str]:
        return self.config.get(
            "group_fields",
            ["type", "category", "status", "group", "kind"],
        )

    def _get_name_fields(self) -> List[str]:
        return self.config.get(
            "name_fields",
            ["name", "title", "description", "id"],
        )

    def _get_chart_group_fields(self) -> List[str]:
        return self.config.get(
            "chart_group_fields",
            ["type", "category", "status", "group"],
        )

    def _get_priority_fields(self) -> List[str]:
        """
        Fields to prioritize as columns in tables.
        """
        return self.config.get(
            "priority_fields",
            ["name", "title", "id", "type", "status", "amount", "quantity", "date"],
        )
    
    def format_response(
        self,
        data: Any,
        query: str,
        format_type: FormatType = "auto",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format API response intelligently based on data structure and query context.

        Args:
            data: The API response data
            query: Original user query
            format_type: Desired format ("auto" lets LLM decide)
            context: Optional context about the data (e.g., field names, data types)
                     Can include schema with formatting_hints for resource-specific formatting

        Returns:
            Dict with:
                - format: chosen format type
                - summary: concise text summary
                - formatted: formatted output (markdown, JSON, etc.)
                - raw_data: original data
        """
        logger.info(
            "Formatting response | type=%s | format_type=%s | query=%r",
            type(data).__name__,
            format_type,
            query[: constants.QUERY_PREVIEW_LENGTH],
        )

        if not data:
            return {
                "format": "text",
                "summary": constants.FORMAT_NO_DATA_RETURNED,
                "formatted": constants.FORMAT_NO_DATA_RETURNED,
                "raw_data": data
            }

        # Normalize data for formatting:
        # - If this is a paginated dict with "results", work on results list
        # - Otherwise, work on data as-is
        format_data = data
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            format_data = data["results"]
            logger.debug(
                "Detected paginated response; normalizing format_data to results list (len=%d)",
                len(format_data),
            )
        
        # For simple/small data, just provide text
        if self._is_simple_data(format_data):
            return self._format_simple(format_data, query)
        
        # Analyze data structure (optionally using context/schema hints)
        data_analysis = self._analyze_data_structure(format_data, context)
        
        # Let heuristics/LLM decide format if auto
        if format_type == "auto":
            format_type = self._determine_format(format_data, query, data_analysis, context)
        
        # Format based on chosen type
        if format_type == "table":
            return self._format_as_table(format_data, query, data_analysis)
        elif format_type == "chart":
            return self._format_for_chart(format_data, query, data_analysis)
        elif format_type == "grouped":
            return self._format_grouped(format_data, query, data_analysis)
        elif format_type == "detailed":
            return self._format_detailed(format_data, query, data_analysis)
        else:  # concise
            return self._format_concise(format_data, query, data_analysis)
    
    def _is_simple_data(self, data: Any) -> bool:
        """
        Return True only for trivial data that does not need AI formatting.
        Any list of records (dicts) or nested structure is not simple — we use
        the formatter so the user gets an understandable summary, not raw JSON.
        """
        if isinstance(data, (str, int, float, bool)) or data is None:
            return True
        if isinstance(data, list):
            if len(data) == 0:
                return True
            # List of any dicts → never simple; always format via AI
            if any(isinstance(x, dict) for x in data):
                return False
            # List of primitives only, small length → simple
            if all(isinstance(x, (str, int, float, bool, type(None))) for x in data):
                return len(data) <= constants.SIMPLE_LIST_MAX_PRIMITIVE_ITEMS
            return False
        if isinstance(data, dict):
            # Only flat key-value with primitive values is simple
            if len(data) > constants.SIMPLE_DICT_MAX_KEYS:
                return False
            for v in data.values():
                if isinstance(v, (dict, list)):
                    return False
                if not isinstance(v, (str, int, float, bool, type(None))):
                    return False
            return True
        return False
    
    def _analyze_data_structure(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze data structure to inform formatting decisions."""
        analysis = {
            "type": type(data).__name__,
            "is_list": isinstance(data, list),
            "is_dict": isinstance(data, dict),
            "count": 0,
            "has_categories": False,
            "has_numbers": False,
            "fields": [],
            "display_field": None,
        }
        
        if isinstance(data, list) and data:
            analysis["count"] = len(data)
            # Analyze first item to understand structure
            first_item = data[0]
            if isinstance(first_item, dict):
                analysis["fields"] = list(first_item.keys())
                # Check for numeric data
                analysis["has_numbers"] = any(
                    isinstance(v, (int, float)) for v in first_item.values()
                )
                # Check for categorical fields (type, category, status, etc.)
                group_fields = {f.lower() for f in self._get_group_fields()}
                analysis["has_categories"] = any(
                    isinstance(k, str) and k.lower() in group_fields
                    for k in first_item.keys()
                )
        elif isinstance(data, dict):
            analysis["fields"] = list(data.keys())
            analysis["count"] = 1
        
        # Try to infer a display field from context/schema if available
        if context:
            display_field = self._get_display_field_from_context(context)
            if display_field:
                analysis["display_field"] = display_field
        
        return analysis

    def _get_display_field_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Extract a primary display field from schema/context if available.
        
        Expected context structure (best-effort, fully generic):
            {
                "resource": "service_orders",
                "schema": {
                    "type": "api",
                    "resources": {
                        "service_orders": {
                            "display_field": "order_number",
                            "fields": [...]
                        }
                    }
                }
            }
        
        This function is intentionally generic and makes no assumptions about
        your specific domain beyond common "name"-style heuristics.
        """
        resource = context.get("resource")
        schema = context.get("schema", {}) or {}
        resources = schema.get("resources", {}) or {}
        
        if not isinstance(resources, dict) or not resource or resource not in resources:
            return None
        
        meta = resources.get(resource, {}) or {}
        display_field = meta.get("display_field")
        if isinstance(display_field, str) and display_field:
            return display_field
        
        # Fallback: infer from 'fields' if present
        fields = meta.get("fields") or []
        if not isinstance(fields, list) or not fields:
            return None
        
        def is_unit_like(fname: str) -> bool:
            lf = fname.lower()
            return any(
                key in lf
                for key in ["unit", "uom", "measurement", "measure", "qty_per", "per_unit"]
            )
        
        # Exact "name"
        for f in fields:
            if isinstance(f, str) and f.lower() == "name" and not is_unit_like(f):
                return f
        
        # Any field containing "name"
        for f in fields:
            if isinstance(f, str) and "name" in f.lower() and not is_unit_like(f):
                return f
        
        # Other common identifier-style fields
        preferred = ["title", "label", "display_name", "short_name", "code", "sku", "order_number", "id"]
        for pref in preferred:
            for f in fields:
                if isinstance(f, str) and f.lower() == pref and not is_unit_like(f):
                    return f
        
        # Fallback: first non unit-like field
        for f in fields:
            if isinstance(f, str) and not is_unit_like(f):
                return f
        
        # Last resort: first field
        return fields[0] if fields else None
    
    def _determine_format(
        self,
        data: Any,
        query: str,
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> FormatType:
        """
        Decide the best format for the data.
        
        Strategy:
        - Apply heuristics using analysis/context when they clearly apply.
        - When heuristics don't pick a format, ask the LLM to choose.
        - Raises on LLM failure or invalid format response (caller can surface to user).
        """
        context = context or {}

        # ------------------------------------------------------------------
        # Heuristic layer (no LLM)
        # ------------------------------------------------------------------
        question_type = context.get("question_type")
        display_mode = context.get("display_mode")
        is_list = analysis.get("is_list", False)
        count = analysis.get("count", 0)
        has_fields = bool(analysis.get("fields"))

        if question_type == "count":
            return "concise"

        if is_list and 1 <= count <= 5:
            if display_mode == "detailed":
                return "detailed"
            return "concise"

        if is_list and count > 5 and has_fields:
            if question_type in (None, "list"):
                return "table"

        # Heuristics didn't pick a format: ask LLM to choose (raises on failure or invalid response)
        prompt = f"""Given the data structure and user query below, choose the best presentation format.

Query: "{query}"

Data: type={analysis['type']}, count={analysis['count']} items, has_categories={analysis['has_categories']}, has_numbers={analysis['has_numbers']}. Fields: {', '.join(analysis['fields'][:constants.ANALYSIS_FIELDS_MAX])}

Sample:
{json.dumps(data[:constants.LLM_DATA_SAMPLE_SMALL] if isinstance(data, list) else data, indent=2)[:constants.LLM_DATA_PREVIEW_500]}

Choose ONE format that best fits the data and what the user asked for:
- concise: Brief text summary
- table: Markdown table (structured lists, comparisons)
- grouped: Grouped by category
- chart: Chart-ready JSON (numeric trends, statistics)
- detailed: Detailed breakdown (complex or single items)

Respond with only the format name, lowercase.
"""
        
        try:
            content = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=DETERMINISTIC_TEMP,
                max_tokens=constants.MAX_TOKENS_CONCISE,
            )
            
            format_choice = (content or "").strip().lower()
            valid_formats = self._get_valid_formats()
            
            if format_choice in valid_formats:
                logger.info(f"LLM selected format: {format_choice}")
                return format_choice
            raise ValueError(
                f"Format selection failed: LLM returned invalid format {format_choice!r} "
                f"(expected one of {valid_formats}). Response was: {content[:100]!r}"
            )
                
        except Exception as e:
            logger.error("Error determining format: %s", e)
            raise
    
    def _format_simple(self, data: Any, query: str) -> Dict[str, Any]:
        """Format trivial data as short plain text. Only used for primitives, empty list, or tiny flat dict/list of primitives."""
        if isinstance(data, (str, int, float, bool)):
            summary = str(data)
        elif data is None:
            summary = ""
        elif isinstance(data, list):
            if len(data) == 0:
                summary = constants.FORMAT_NO_DATA_RETURNED
            else:
                summary = ", ".join(str(x) for x in data)
        elif isinstance(data, dict):
            summary = json.dumps(data, indent=2)
        else:
            summary = str(data)
        return {
            "format": "text",
            "summary": summary,
            "formatted": summary,
            "raw_data": data
        }
    
    def _format_concise(self, data: Any, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a rich but still compact summary using LLM."""
        display_field = analysis.get("display_field")
        is_small_list = isinstance(data, list) and 1 <= len(data) <= 5

        hint_section = ""
        # Pass the exact count so the summary is accurate
        n = len(data) if isinstance(data, list) else 0
        if n > 0:
            hint_section += f"\nTotal count: {n} item(s). Include this exact count in your summary (e.g. 'Found {n} items: ...'). Use wording that matches the user's query.\n"
        if display_field:
            hint_section += f"\nWhen listing items, prefer the '{display_field}' field from the data as the primary identifier.\n"
        if is_small_list:
            hint_section += "\nUse a short bullet list (one bullet per item), optionally with a one-line summary first.\n"
        if isinstance(data, list) and constants.MIN_LIST_LENGTH_MEDIUM_SAMPLE <= len(data) <= constants.LLM_DATA_SAMPLE_MEDIUM:
            hint_section += "\nInclude as many concrete item names or identifiers from the data as reasonably possible, not only the count.\n"

        prompt = f"""Summarize this API response for the user in a small number of short sentences or bullets (typically 2-5).

Rules:
- Base your summary only on the data and query below. Do not add or assume information not present in the data.
- Use the exact count given for the total number of items. Keep wording generic (e.g. "items" or terms that match what the user asked for).
- Where the data contains names or identifiers, you may list representative examples; do not invent examples.

User query: "{query}"

Data: {json.dumps(data[:constants.LLM_DATA_SAMPLE_MEDIUM] if isinstance(data, list) else data, indent=2)[:constants.LLM_DATA_PREVIEW_1000]}

{hint_section}

Provide a concise, accurate summary that answers what the user asked and reflects only the data above.
"""
        
        try:
            content = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=DETERMINISTIC_TEMP,
                max_tokens=constants.MAX_TOKENS_SUMMARY,
            )
            
            summary = (content or "").strip()
            
            return {
                "format": "concise",
                "summary": summary,
                "formatted": summary,
                "raw_data": data
            }
        except Exception as e:
            logger.error("Error creating concise summary: %s", e)
            raise
    
    def _format_as_table(self, data: Any, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format data as markdown table."""
        if not isinstance(data, list) or not data:
            return self._format_concise(data, query, analysis)
        
        # Get fields from first item
        first_item = data[0]
        if not isinstance(first_item, dict):
            return self._format_concise(data, query, analysis)
        
        # Select most relevant fields (max 6)
        fields = self._select_important_fields(first_item, query)[: constants.TABLE_FIELDS_MAX]
        
        # If we have a display_field hint, ensure it is the first column
        display_field = analysis.get("display_field")
        if display_field and display_field in first_item:
            if display_field in fields:
                fields = [display_field] + [f for f in fields if f != display_field]
            else:
                fields = [display_field] + fields
                fields = fields[: constants.TABLE_FIELDS_MAX]
        
        # Build markdown table
        table_lines = []
        # Header
        table_lines.append("| " + " | ".join(fields) + " |")
        table_lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        
        # Rows (max 20 for readability)
        for item in data[: constants.TABLE_ROW_SAMPLE]:
            if not isinstance(item, dict):
                continue
            row_values = [str(item.get(field, ""))[: constants.TABLE_FIELD_PREVIEW] for field in fields]
            table_lines.append("| " + " | ".join(row_values) + " |")
        
        if len(data) > constants.TABLE_ROW_SAMPLE:
            table_lines.append(f"\n*...and {len(data) - constants.TABLE_ROW_SAMPLE} more items*")
        
        table = "\n".join(table_lines)
        summary = f"Found {len(data)} items"
        
        return {
            "format": "table",
            "summary": summary,
            "formatted": table,
            "raw_data": data
        }
    
    def _format_grouped(self, data: Any, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Group data by category field."""
        if not isinstance(data, list) or not data:
            return self._format_concise(data, query, analysis)
        
        # Find category field
        category_field = None
        for field in self._get_group_fields():
            if field in data[0]:
                category_field = field
                break
        
        if not category_field:
            # No clear category, fall back to table
            return self._format_as_table(data, query, analysis)
        
        # Group by category
        groups = {}
        for item in data:
            category = item.get(category_field, "Other")
            if category not in groups:
                groups[category] = []
            groups[category].append(item)
        
        # Format grouped output
        formatted_lines = []
        formatted_lines.append(f"## Results grouped by {category_field.title()}\n")
        
        for category, items in groups.items():
            formatted_lines.append(f"### {category}")
            formatted_lines.append(f"- Count: {len(items)}")
            
            # Show sample items
            name_fields = self._get_name_fields()
            for item in items[: constants.GROUPED_ITEMS_SAMPLE]:
                # Find a descriptive field
                desc_field = next((f for f in name_fields if f in item), None)
                if desc_field:
                    formatted_lines.append(f"  - {item[desc_field]}")
            
            if len(items) > 3:
                formatted_lines.append(f"  - ...and {len(items) - 3} more")
            formatted_lines.append("")
        
        formatted = "\n".join(formatted_lines)
        summary = f"Found {len(data)} items across {len(groups)} categories"
        
        return {
            "format": "grouped",
            "summary": summary,
            "formatted": formatted,
            "raw_data": data,
            "groups": {k: len(v) for k, v in groups.items()}
        }
    
    def _format_for_chart(self, data: Any, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for chart visualization."""
        if not isinstance(data, list) or not data:
            return self._format_concise(data, query, analysis)
        
        # Prepare chart-ready data
        chart_data = {
            "type": "bar",  # default
            "labels": [],
            "datasets": [],
            "summary": ""
        }
        
        # Group by category if present
        category_field = None
        for field in self._get_chart_group_fields():
            if field in data[0]:
                category_field = field
                break
        
        if category_field:
            # Count by category
            counts = {}
            for item in data:
                cat = item.get(category_field, "Other")
                counts[cat] = counts.get(cat, 0) + 1
            
            chart_data["labels"] = list(counts.keys())
            chart_data["datasets"] = [{
                "label": "Count",
                "data": list(counts.values())
            }]
            chart_data["summary"] = f"Distribution by {category_field}"
        else:
            # Just show count over time or by index
            n = min(constants.CHART_MAX_ITEMS, len(data))
            chart_data["labels"] = [f"Item {i+1}" for i in range(n)]
            chart_data["datasets"] = [{
                "label": "Items",
                "data": [1] * n
            }]
            chart_data["summary"] = f"{len(data)} items"
        
        return {
            "format": "chart",
            "summary": chart_data["summary"],
            "formatted": json.dumps(chart_data, indent=2),
            "raw_data": data,
            "chart_data": chart_data
        }
    
    def _format_detailed(self, data: Any, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed breakdown of data."""
        prompt = f"""Provide a detailed, well-structured breakdown of this API response that answers what the user asked.

Rules:
- Base your breakdown only on the data below. Do not add facts, numbers, or details that are not present in the data.
- Use generic wording (e.g. "items", "records") unless the query or data clearly indicates a specific type.
- Organize with key findings and important details from the data. Use markdown (headers, lists, bold) where it helps.

User query: "{query}"

Data: {json.dumps(data, indent=2)[:constants.LLM_DATA_PREVIEW_2000]}

Summarize accurately from the data above only.
"""
        
        try:
            content = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=DETERMINISTIC_TEMP,
                max_tokens=constants.MAX_TOKENS_DETAILED,
            )
            
            detailed = (content or "").strip()
            summary = f"Detailed breakdown of {analysis['count']} item(s)"
            
            return {
                "format": "detailed",
                "summary": summary,
                "formatted": detailed,
                "raw_data": data
            }
        except Exception as e:
            logger.error("Error creating detailed summary: %s", e)
            raise
    
    def _select_important_fields(self, item: Dict, query: str) -> List[str]:
        """Select most important fields for display based on query."""
        fields = list(item.keys())

        # Prioritize common important fields (configurable)
        priority_fields = self._get_priority_fields()

        # Put priority fields first
        sorted_fields = []
        for pf in priority_fields:
            if pf in fields:
                sorted_fields.append(pf)
                fields.remove(pf)

        # Add remaining fields
        sorted_fields.extend(fields)

        return sorted_fields

    def _get_formatting_hints(self, context: Optional[Dict[str, Any]], resource: str) -> Optional[Dict[str, Any]]:
        """
        Extract formatting hints from schema for a specific resource (v0.3.32).

        Args:
            context: Context dict containing schema
            resource: Resource name (e.g., 'service_orders')

        Returns:
            Formatting hints dict or None
        """
        if not context:
            return None

        schema = context.get("schema", {})
        formatting_hints = schema.get("formatting_hints", {})

        return formatting_hints.get(resource)

    def _format_with_hints(
        self,
        data: List[Dict],
        hints: Dict[str, Any],
        resource: str
    ) -> str:
        """
        Format data using schema-provided formatting hints (v0.3.32).

        This enables resource-specific formatting like tables with links,
        status indicators, etc.

        Args:
            data: List of items to format
            hints: Formatting hints from schema
            resource: Resource name

        Returns:
            Formatted markdown string
        """
        list_format = hints.get("list_format", "table")
        columns = hints.get("columns", [])
        column_labels = hints.get("column_labels", columns)
        link_patterns = hints.get("link_patterns", {})
        status_indicators = hints.get("status_indicators", {})

        if list_format == "table" and columns:
            return self._format_table_with_hints(data, columns, column_labels, link_patterns)
        elif list_format == "detailed":
            return self._format_detailed_with_hints(data, hints)
        else:
            # Fall back to default formatting
            return None

    def _format_table_with_hints(
        self,
        data: List[Dict],
        columns: List[str],
        column_labels: List[str],
        link_patterns: Dict[str, str]
    ) -> str:
        """Format data as markdown table with links using hints."""
        lines = []

        # Header
        lines.append("| " + " | ".join(column_labels) + " |")
        lines.append("| " + " | ".join(["---"] * len(column_labels)) + " |")

        # Rows
        for item in data[:20]:  # Limit to 20 rows
            row_values = []
            for col in columns:
                value = self._extract_nested_value(item, col)

                # Apply link pattern if exists for this column
                base_col = col.split(".")[0]
                if base_col in link_patterns:
                    pattern = link_patterns[base_col]
                    try:
                        link = pattern.format(**item, **{base_col: value})
                        value = f"[{value}]({link})"
                    except (KeyError, ValueError):
                        pass

                row_values.append(str(value)[:50] if value else "-")

            lines.append("| " + " | ".join(row_values) + " |")

        if len(data) > 20:
            lines.append(f"\n*...and {len(data) - 20} more items*")

        return "\n".join(lines)

    def _format_detailed_with_hints(self, data: List[Dict], hints: Dict[str, Any]) -> str:
        """Format data in detailed view with specific fields."""
        include_fields = hints.get("include_fields", [])
        observations_format = hints.get("observations_format", "default")

        lines = []
        for item in data[:5]:  # Limit to 5 items for detailed view
            for field in include_fields:
                value = self._extract_nested_value(item, field)

                if field == "observations" and observations_format == "bullet_list_with_status":
                    lines.append(f"\n**Observations:**\n")
                    if isinstance(value, list):
                        for obs in value:
                            area = obs.get("area", "Unknown")
                            result = obs.get("result", "")
                            notes = obs.get("notes", "")
                            status = "✅" if result.lower() in ["pass", "acceptable", "ok"] else "⚠️"
                            lines.append(f"{status} **{area}** - {result}")
                            if notes:
                                lines.append(f"   {notes}")
                else:
                    lines.append(f"**{field.replace('_', ' ').title()}:** {value}")

            lines.append("")  # Blank line between items

        return "\n".join(lines)

    def _extract_nested_value(self, item: Dict, path: str) -> Any:
        """Extract value from nested dict using dot notation (e.g., 'company.name')."""
        keys = path.split(".")
        value = item
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
