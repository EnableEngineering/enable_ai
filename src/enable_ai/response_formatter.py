"""
Response Formatter - Intelligent formatting of API responses

Provides context-aware formatting including:
- Concise text summaries
- Markdown tables for structured data
- Chart-ready JSON for visualizations
- Grouping and categorization
"""

from typing import Dict, Any, List, Optional, Union, Literal
import json
from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP
from . import constants

logger = setup_logger(__name__)

FormatType = Literal["auto", "concise", "detailed", "table", "chart", "grouped"]


class ResponseFormatter:
    """
    Intelligent response formatter using LLM to choose optimal format.
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
        """Check if data is simple enough to not need special formatting."""
        if isinstance(data, (str, int, float, bool)):
            return True
        if isinstance(data, dict) and len(data) <= 3:
            return True
        if isinstance(data, list) and len(data) <= 2:
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
        - First, apply lightweight heuristics using analysis/context
        - If still ambiguous, fall back to LLM-based choice
        """
        context = context or {}

        # ------------------------------------------------------------------
        # Heuristic layer (fast, no LLM)
        # ------------------------------------------------------------------
        question_type = context.get("question_type")
        display_mode = context.get("display_mode")
        is_list = analysis.get("is_list", False)
        count = analysis.get("count", 0)
        has_fields = bool(analysis.get("fields"))

        # Count-style questions are handled upstream; default to concise here
        if question_type == "count":
            return "concise"

        # Very small lists: concise/bullet-style text is usually better
        if is_list and 1 <= count <= 5:
            if display_mode == "detailed":
                return "detailed"
            # small set → concise summary (prompt will guide bullets)
            return "concise"

        # Larger structured lists: tables tend to work well
        if is_list and count > 5 and has_fields:
            if question_type in (None, "list"):
                return "table"

        # Fallback: let LLM choose
        prompt = f"""Given this data structure and user query, determine the best format:

Query: "{query}"

Data Analysis:
- Type: {analysis['type']}
- Count: {analysis['count']} items
- Has categories: {analysis['has_categories']}
- Has numbers: {analysis['has_numbers']}
- Fields: {', '.join(analysis['fields'][:constants.ANALYSIS_FIELDS_MAX])}

Sample data:
{json.dumps(data[:constants.LLM_DATA_SAMPLE_SMALL] if isinstance(data, list) else data, indent=2)[:constants.LLM_DATA_PREVIEW_500]}

Choose ONE format:
- "concise": Brief text summary (good for simple queries, confirmations)
- "table": Markdown table (good for structured data, comparisons, lists)
- "grouped": Grouped by category (good for inventory, categorized items)
- "chart": Chart-ready JSON (good for numeric trends, statistics)
- "detailed": Detailed breakdown (good for complex single items)

Respond with ONLY the format name (lowercase).
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
            else:
                logger.warning(f"Invalid format from LLM: {format_choice}, defaulting to concise")
                return "concise"
                
        except Exception as e:
            logger.error("Error determining format: %s", e)
            raise
    
    def _format_simple(self, data: Any, query: str) -> Dict[str, Any]:
        """Format simple data as plain text."""
        if isinstance(data, (str, int, float, bool)):
            summary = str(data)
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
        if display_field:
            hint_section = f"\nWhen listing items, use the '{display_field}' field as the primary item name.\n"

        if is_small_list:
            # Encourage the model to produce a short bullet list for small result sets
            hint_section += "\nFormat the answer as a short bullet list, one bullet per item, optionally preceded by a one-line summary.\n"

        # For medium-size lists, encourage including many concrete items, not just a count
        if isinstance(data, list) and constants.MIN_LIST_LENGTH_MEDIUM_SAMPLE <= len(data) <= constants.LLM_DATA_SAMPLE_MEDIUM:
            hint_section += "\nThe user prefers less aggressive summarisation: include as many concrete item names or identifiers as reasonably possible instead of only saying 'Found N items'.\n"

        prompt = f"""Summarize this API response for the user in a small number of short sentences or bullets (typically 2-5), not a single ultra-brief line.

User Query: "{query}"

Data: {json.dumps(data[:constants.LLM_DATA_SAMPLE_MEDIUM] if isinstance(data, list) else data, indent=2)[:constants.LLM_DATA_PREVIEW_1000]}

{hint_section}

Provide a helpful, concise summary focused on what the user asked for.
Include key numbers/counts and, where meaningful, list representative item names.
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
        prompt = f"""Provide a detailed, well-structured breakdown of this API response.

User Query: "{query}"

Data: {json.dumps(data, indent=2)[:constants.LLM_DATA_PREVIEW_2000]}

Create a clear, detailed summary with:
- Key findings/highlights
- Important details
- Organized in sections if appropriate
- Use markdown formatting (headers, lists, bold)
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
