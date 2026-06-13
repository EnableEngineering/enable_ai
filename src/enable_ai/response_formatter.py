"""
Response Formatter - Intelligent LLM-driven formatting of API responses

The LLM decides the best response format and generates natural, helpful responses.
Supports:
- Natural text responses (conversational, informative)
- Markdown tables (for lists/comparisons)
- Chart-ready JSON (for visualizations)
- Detailed breakdowns (for complex data)

The LLM has full control over the response - no code-based overrides.
"""

from typing import Dict, Any, List, Optional, Union, Literal, cast
import json
from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP
from .follow_up_suggestions import generate_follow_up_queries
from . import constants

logger = setup_logger(__name__)

FormatType = Literal["auto", "concise", "detailed", "table", "chart", "grouped"]


class ResponseFormatter:
    """
    LLM-driven response formatter.

    The LLM receives the user's query and API data, then decides:
    1. What format is best (text, table, chart, etc.)
    2. What information to include
    3. How to present it naturally

    No code-based heuristics override the LLM's decisions.
    """

    def __init__(self, model: str = "gpt-4o", config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model: OpenAI model name.
            config: Optional configuration dict to customize formatting behavior.
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
        Format API response using intelligent LLM-driven formatting.

        The LLM decides the best format and generates a natural, helpful response.
        No code-based heuristics override the LLM's decisions.

        Args:
            data: The API response data
            query: Original user query
            format_type: Hint for format ("auto" lets LLM fully decide)
            context: Optional context about the data

        Returns:
            Dict with:
                - format: chosen format type
                - summary: the main response text (what the user sees)
                - formatted: formatted output (may include table/chart if LLM chose)
                - raw_data: original data
        """
        logger.info(
            "Formatting response | type=%s | format_type=%s | query=%r",
            type(data).__name__,
            format_type,
            query[: constants.QUERY_PREVIEW_LENGTH],
        )

        # Extract pagination info before normalizing
        pagination = self._extract_pagination(data)

        # Check for empty results
        if self._is_empty_result(data):
            return self._format_empty_response(query, context)

        if not data:
            return {
                "format": "text",
                "summary": constants.FORMAT_NO_DATA_RETURNED,
                "formatted": constants.FORMAT_NO_DATA_RETURNED,
                "raw_data": data,
                "pagination": pagination
            }

        # Normalize data for formatting
        format_data = data
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            format_data = data["results"]
            logger.debug(
                "Detected paginated response; normalizing format_data to results list (len=%d)",
                len(format_data),
            )

        # For simple/small data, just provide text
        if self._is_simple_data(format_data):
            result = self._format_simple(format_data, query)
            result["pagination"] = pagination
            return result

        # Use intelligent LLM-driven formatting for all complex data.
        # The LLM is responsible for understanding the user's intent
        # (e.g. "give me their email ids") and choosing which fields
        # to surface, while strictly following the constraints in the
        # prompt (no invention, include all items, etc.).
        result = self._generate_intelligent_response(format_data, query, context, pagination)

        # Append pagination info to summary and attach follow-up query templates
        if pagination.get("has_more") or pagination.get("total", 0) > pagination.get("returned", 0):
            result["summary"] = self._append_pagination_info(
                result.get("summary", ""),
                pagination,
                context
            )
            pagination_info = {
                "total_count": pagination.get("total", 0),
                "actual_count": pagination.get("returned", 0),
                "has_more": pagination.get("has_more", False),
            }
            result["follow_up_queries"] = generate_follow_up_queries(pagination_info, context)

        result["pagination"] = pagination
        return result

    def _generate_intelligent_response(
        self,
        data: Any,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        pagination: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an intelligent response using a single LLM call.

        The LLM decides:
        1. What format is best (text, table, chart, bullets, etc.)
        2. What information to include based on what the user asked
        3. How to present it naturally and helpfully

        Args:
            data: The data to format
            query: User's original query
            context: Optional context (resource name, schema, etc.)
            pagination: Optional pagination info from API response

        Returns:
            Dict with format, summary, formatted, and raw_data
        """
        # BUG FIX (v0.3.36): Show ALL data to LLM for small datasets
        # Previously only showed 10 items, causing "1 of 3 users" summaries
        if isinstance(data, list):
            data_count = len(data)
            # For small datasets (<=25 items), show ALL data to LLM
            # For larger datasets, show a representative sample
            if data_count <= 25:
                sample_size = data_count  # Show ALL items
                data_sample = data
                data_preview = json.dumps(data_sample, indent=2, default=str)[:constants.LLM_DATA_PREVIEW_LARGE]
            else:
                # For large datasets, show first 25 + mention total
                sample_size = min(data_count, 25)
                data_sample = data[:sample_size]
                data_preview = json.dumps(data_sample, indent=2, default=str)[:constants.LLM_DATA_PREVIEW_LARGE]
            remaining = data_count - sample_size
        else:
            data_count = 1
            sample_size = 1
            data_preview = json.dumps(data, indent=2, default=str)[:constants.LLM_DATA_PREVIEW_LARGE]
            remaining = 0

        logger.debug(
            "Formatter LLM input: data_count=%s sample_size=%s preview_chars=%s",
            data_count,
            sample_size if isinstance(data, list) else 1,
            len(data_preview),
        )

        # Build pagination context for LLM
        pagination_info = ""
        if pagination:
            total = pagination.get("total", data_count)
            returned = pagination.get("returned", data_count)
            has_more = pagination.get("has_more", False)
            if total > 0 or has_more:
                pagination_info = f"""
PAGINATION INFO:
- Total items available: {total}
- Items shown in this response: {returned}
- Has more pages: {has_more}

IMPORTANT: Include the total count in your response. If there are more items available,
mention that the user can say "show more" or "next page" to see additional results.
"""

        # Build the prompt
        prompt = f"""You are a helpful AI assistant. The user asked a question and I retrieved data from an API.
Generate a natural, helpful response that answers the user's question.
CRITICAL CONSTRAINT: The response MUST be derived STRICTLY from the API data provided below.
You MUST NOT invent, guess, or fabricate any users, roles, names, ids, counts, or other values
that are not present in the data. If a value is missing in the data, clearly say it is missing
instead of making something up (e.g. "unnamed user (id=123)" if only an id is present).
{pagination_info}

USER'S QUESTION: "{query}"

DATA RETRIEVED ({data_count} item{"s" if data_count != 1 else ""} total):
{data_preview}
{"[... and " + str(remaining) + " more items not shown]" if remaining > 0 else ""}

CRITICAL INSTRUCTIONS:
1. **USE ONLY THE DATA PROVIDED (NO INVENTION)**:
   - Every user, role, id, status, or field you mention MUST come directly from the data shown below.
   - Do NOT create placeholder labels like "User 3" or "Role 3" unless those exact strings appear in the data.
   - If the data does not contain a value, say that it is not available instead of inventing it.

2. **USE ALL THE DATA**: You MUST include ALL {data_count} items in your response, not just a subset.
   - If there are 3 users, list ALL 3 users
   - If there are 5 orders, show ALL 5 orders
   - NEVER say "Found X items" without listing them ALL (unless there are more than 15)

3. **Understand what the user wants**: Are they asking for a count? A list? Details? Specific information?
   - If they ask for specific fields (e.g. email IDs, names, roles), extract EXACTLY those fields for EVERY item that has them.
   - If some items do not have that field, say that it is missing instead of skipping them or making up a value.

4. **Choose the best response format**:
   - **Text response**: For counts, summaries, single items, or when natural language is clearest
   - **Bullet list**: For 2-10 items where the user wants to see them (SHOW ALL)
   - **Markdown table**: For 5+ items with multiple fields worth comparing (SHOW ALL)
   - **Detailed breakdown**: For complex single items or when user asks for details

5. **Be specific and helpful (without inventing anything)**:
   - Include actual names, IDs, or identifiers from the data for EVERY item
   - If showing a list, show ALL the actual items (names, key details) - do not truncate
   - If it's a count question, give the count AND list what they are
   - Only truncate if there are more than 15 items

6. **Response format**:
   Return a JSON object with exactly these fields:
   {{
     "format": "text" | "table" | "bullets" | "detailed",
     "response": "Your complete response showing ALL items"
   }}

EXAMPLES:

User: "List all users"
Data: [{{"username": "john@example.com", "role": "Admin"}}, {{"username": "jane@example.com", "role": "User"}}, {{"username": "bob@example.com", "role": "Manager"}}]
Response: {{"format": "bullets", "response": "Here are all 3 users:\\n\\n• **john@example.com** - Admin\\n• **jane@example.com** - User\\n• **bob@example.com** - Manager"}}

User: "Show me the service orders assigned to me"
Data: [{{order_number: "SO-001", status: "New", customer: "ABC Corp"}}, {{order_number: "SO-002", status: "In Progress", customer: "XYZ Inc"}}]
Response: {{"format": "bullets", "response": "You have 2 service orders assigned to you:\\n\\n• **SO-001** - ABC Corp (New)\\n• **SO-002** - XYZ Inc (In Progress)"}}

User: "How many items are low stock?"
Data: [{{"name": "Widget A"}}, {{"name": "Widget B"}}]
Response: {{"format": "text", "response": "There are 2 items with low stock: Widget A and Widget B."}}

Now generate your response for the user's question. Return ONLY the JSON object, no markdown code blocks.
REMEMBER: Include ALL {data_count} items in your response!
"""

        try:
            content = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=DETERMINISTIC_TEMP,
                max_tokens=constants.MAX_TOKENS_DETAILED,
            )

            # Parse the LLM response
            response_text = (content or "").strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                # Extract JSON from code block
                lines = response_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or not line.startswith("```"):
                        json_lines.append(line)
                response_text = "\n".join(json_lines).strip()

            try:
                parsed = json.loads(response_text)
                chosen_format = parsed.get("format", "text")
                response = parsed.get("response", response_text)
            except json.JSONDecodeError:
                # LLM didn't return valid JSON, use the text as-is
                logger.warning("LLM returned non-JSON response, using as text")
                chosen_format = "text"
                response = response_text

            logger.info("LLM chose format: %s", chosen_format)

            # Build table deterministically — LLM often truncates markdown tables in output
            # Configurable via ENABLE_AI_LIST_FORMAT_TABLE (default 1 = always table for lists)
            if constants.LIST_FORMAT_TABLE and isinstance(data, list) and len(data) > 1:
                analysis = self._analyze_data_structure(data, context)
                table_result = self._format_as_table(data, query, analysis)
                return {
                    "format": "table",
                    "summary": response,
                    "formatted": table_result["formatted"],
                    "raw_data": data,
                }

            return {
                "format": chosen_format,
                "summary": response,
                "formatted": response,
                "raw_data": data,
            }

        except Exception as e:
            logger.error("Error in intelligent response generation: %s", e)
            # Fallback to basic formatting
            return self._fallback_format(data, query)
    
    def _fallback_format(self, data: Any, query: str) -> Dict[str, Any]:
        """
        Fallback formatting when LLM fails.
        Generates a basic but useful response.
        """
        if isinstance(data, list):
            count = len(data)
            # Try to extract names/identifiers
            items_preview = []
            name_fields = self._get_name_fields()
            for item in data[:5]:
                if isinstance(item, dict):
                    for nf in name_fields:
                        if nf in item and item[nf]:
                            items_preview.append(str(item[nf]))
                            break

            if items_preview:
                if count <= 5:
                    response = f"Found {count} items: {', '.join(items_preview)}"
                else:
                    response = f"Found {count} items including: {', '.join(items_preview)}, and {count - 5} more"
            else:
                response = f"Found {count} items"
        else:
            response = f"Retrieved data: {json.dumps(data, indent=2, default=str)[:500]}"

        return {
            "format": "text",
            "summary": response,
            "formatted": response,
            "raw_data": data
        }

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
            if context.get("list_display_fields"):
                analysis["list_display_fields"] = context["list_display_fields"]
        
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
                # Cast is safe here because we validate against the allowed set first
                typed_choice = cast(FormatType, format_choice)
                logger.info(f"LLM selected format: {typed_choice}")
                return typed_choice
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
        """Generate a rich but still compact summary using LLM.

        When the serialised data exceeds LLM_DATA_PREVIEW_1000 characters the
        dataset is split into chunks, each chunk is summarised independently,
        and then a final merge call produces one cohesive response.  Nothing
        is ever silently truncated.
        """
        display_field = analysis.get("display_field")
        is_small_list = isinstance(data, list) and 1 <= len(data) <= 5
        total_count = len(data) if isinstance(data, list) else 1
        budget = constants.LLM_DATA_PREVIEW_1000

        hint_section = ""
        if total_count > 0:
            hint_section += (
                f"\nTotal count: {total_count} item(s). Include this exact count in your"
                f" summary (e.g. 'Found {total_count} items: ...'). Use wording that"
                " matches the user's query.\n"
            )
        if display_field:
            hint_section += f"\nWhen listing items, prefer the '{display_field}' field from the data as the primary identifier.\n"
        if is_small_list:
            hint_section += "\nUse a short bullet list (one bullet per item), optionally with a one-line summary first.\n"
        if isinstance(data, list) and constants.MIN_LIST_LENGTH_MEDIUM_SAMPLE <= len(data) <= constants.LLM_DATA_SAMPLE_MEDIUM:
            hint_section += "\nInclude as many concrete item names or identifiers from the data as reasonably possible, not only the count.\n"

        # Determine whether all data fits in one call
        items = data[:constants.LLM_DATA_SAMPLE_MEDIUM] if isinstance(data, list) else data
        data_json = json.dumps(items, indent=2, default=str)

        if len(data_json) <= budget:
            # ── single call path ──────────────────────────────────────────
            prompt = (
                "Summarize this API response for the user in a small number of short"
                " sentences or bullets (typically 2-5).\n\n"
                "Rules:\n"
                "- Base your summary only on the data and query below."
                "  Do not add or assume information not present in the data.\n"
                "- Use the exact count given for the total number of items."
                "  Keep wording generic (e.g. \"items\" or terms that match what the user asked for).\n"
                "- Where the data contains names or identifiers, you may list representative"
                "  examples; do not invent examples.\n\n"
                f'User query: "{query}"\n\n'
                f"Data: {data_json}\n\n"
                f"{hint_section}\n"
                "Provide a concise, accurate summary that answers what the user asked"
                " and reflects only the data above.\n"
            )
            try:
                content = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=DETERMINISTIC_TEMP,
                    max_tokens=constants.MAX_TOKENS_SUMMARY,
                )
                summary = (content or "").strip()
            except Exception as e:
                logger.error("Error creating concise summary: %s", e)
                raise
        else:
            # ── chunked path: summarise each chunk then merge ─────────────
            all_items = data if isinstance(data, list) else [data]
            chunks = self._split_data_into_chunks(all_items, budget)
            logger.debug(
                "_format_concise: data_json=%d chars → %d chunk(s)", len(data_json), len(chunks)
            )
            partials: List[str] = []
            for i, chunk in enumerate(chunks):
                partials.append(
                    self._summarize_one_chunk(
                        chunk, query, total_count, i, len(chunks), display_field
                    )
                )
            summary = (
                self._merge_chunk_summaries(partials, query, total_count)
                if len(partials) > 1
                else partials[0]
            )

        return {
            "format": "concise",
            "summary": summary,
            "formatted": summary,
            "raw_data": data,
        }
    
    def _format_as_table(self, data: Any, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format data as markdown table."""
        if not isinstance(data, list) or not data:
            return self._format_concise(data, query, analysis)

        # Get fields from first item
        first_item = data[0]
        if not isinstance(first_item, dict):
            return self._format_concise(data, query, analysis)

        # Prefer schema-driven display fields when provided in context
        ctx_fields = (analysis.get("list_display_fields") or []) if isinstance(analysis, dict) else []
        if ctx_fields:
            fields = [f for f in ctx_fields if f in first_item or "." in f][: constants.TABLE_FIELDS_MAX]
            if not fields:
                fields = ctx_fields[: constants.TABLE_FIELDS_MAX]
        else:
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

        # Let LLM generate the summary - use _format_concise for rich summary
        concise_result = self._format_concise(data, query, analysis)
        summary = concise_result.get("summary", f"Found {len(data)} items")

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
        """Provide detailed breakdown of data.

        When the serialised data exceeds LLM_DATA_PREVIEW_2000 characters the
        dataset is split into chunks, each chunk is broken down independently,
        and the sections are merged by a final LLM call.  Nothing is truncated.
        """
        budget = constants.LLM_DATA_PREVIEW_2000
        total_count = analysis.get("count", 1)

        def _single_call(payload: Any) -> str:
            prompt = (
                "Provide a detailed, well-structured breakdown of this API response"
                " that answers what the user asked.\n\n"
                "Rules:\n"
                "- Base your breakdown only on the data below. Do not add facts,"
                "  numbers, or details that are not present in the data.\n"
                "- Use generic wording (e.g. \"items\", \"records\") unless the query"
                "  or data clearly indicates a specific type.\n"
                "- Organise with key findings and important details from the data."
                "  Use markdown (headers, lists, bold) where it helps.\n\n"
                f'User query: "{query}"\n\n'
                f"Data: {json.dumps(payload, indent=2, default=str)}\n\n"
                "Summarize accurately from the data above only.\n"
            )
            content = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=DETERMINISTIC_TEMP,
                max_tokens=constants.MAX_TOKENS_DETAILED,
            )
            return (content or "").strip()

        data_json = json.dumps(data, indent=2, default=str)

        if len(data_json) <= budget:
            # ── single call path ──────────────────────────────────────────
            try:
                detailed = _single_call(data)
            except Exception as e:
                logger.error("Error creating detailed summary: %s", e)
                raise
        else:
            # ── chunked path ──────────────────────────────────────────────
            all_items = data if isinstance(data, list) else [data]
            chunks = self._split_data_into_chunks(all_items, budget)
            logger.debug(
                "_format_detailed: data_json=%d chars → %d chunk(s)", len(data_json), len(chunks)
            )
            try:
                section_parts: List[str] = []
                for i, chunk in enumerate(chunks):
                    section_parts.append(_single_call(chunk))

                if len(section_parts) == 1:
                    detailed = section_parts[0]
                else:
                    # Merge all section breakdowns into one response
                    combined = "\n\n---\n\n".join(section_parts)
                    merge_prompt = (
                        f'The user asked: "{query}". The API returned {total_count} items.\n\n'
                        "Below are detailed breakdowns of each batch. Merge them into a single"
                        " well-structured response. Do not invent or omit any information.\n\n"
                        f"{combined}\n\n"
                        "Final merged response:"
                    )
                    content = self.client.chat_completion(
                        messages=[{"role": "user", "content": merge_prompt}],
                        model=self.model,
                        temperature=DETERMINISTIC_TEMP,
                        max_tokens=constants.MAX_TOKENS_DETAILED,
                    )
                    detailed = (content or "").strip()
            except Exception as e:
                logger.error("Error creating detailed summary (chunked): %s", e)
                raise

        return {
            "format": "detailed",
            "summary": f"Detailed breakdown of {total_count} item(s)",
            "formatted": detailed,
            "raw_data": data,
        }
    
    # ------------------------------------------------------------------
    # Chunked LLM helpers — used when a dataset is too large for one call
    # ------------------------------------------------------------------

    def _split_data_into_chunks(self, data: list, max_chars: int) -> List[list]:
        """Split *data* into sublists where each sublist serialises to ≤ max_chars."""
        chunks: List[list] = []
        current: list = []
        current_size = 0
        for item in data:
            item_json = json.dumps(item, default=str)
            item_size = len(item_json)
            if item_size >= max_chars:
                if current:
                    chunks.append(current)
                    current = []
                    current_size = 0
                chunks.append([item])
            elif current_size + item_size > max_chars and current:
                chunks.append(current)
                current = [item]
                current_size = item_size
            else:
                current.append(item)
                current_size += item_size
        if current:
            chunks.append(current)
        return chunks

    def _summarize_one_chunk(
        self,
        chunk: list,
        query: str,
        total_count: int,
        chunk_index: int,
        total_chunks: int,
        display_field: Optional[str] = None,
    ) -> str:
        """Return one-line-per-item bullet text for a single chunk."""
        n = len(chunk)
        start = chunk_index * n + 1
        display_hint = (
            f"\nPrefer the '{display_field}' field as the primary identifier per item."
            if display_field
            else ""
        )
        prompt = (
            f'The user asked: "{query}"\n\n'
            f"Total dataset: {total_count} items. "
            f"This batch covers items {start}–{start + n - 1} of {total_chunks} batch(es).{display_hint}\n\n"
            "List each item as ONE bullet line. Use only information from the data below. "
            "Do not invent or omit anything.\n\n"
            f"Data:\n{json.dumps(chunk, indent=2, default=str)}\n\n"
            "Output (one bullet per item):\n- ..."
        )
        content = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=DETERMINISTIC_TEMP,
            max_tokens=constants.MAX_TOKENS_SUMMARY,
        )
        return (content or "").strip()

    def _merge_chunk_summaries(
        self, partial_summaries: List[str], query: str, total_count: int
    ) -> str:
        """Combine per-chunk bullet lists into one cohesive final response."""
        combined = "\n".join(partial_summaries)
        prompt = (
            f'The user asked: "{query}". The API returned {total_count} items in total.\n\n'
            "The items were processed in batches. Below are the per-batch summaries.\n"
            "Merge them into a single, coherent response that:\n"
            "- Mentions the exact total count\n"
            "- Lists all items (no omissions, no invented items)\n"
            "- Uses a consistent bullet or prose style\n\n"
            f"Batch summaries:\n{combined}\n\n"
            "Final combined response:"
        )
        content = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=DETERMINISTIC_TEMP,
            max_tokens=constants.MAX_TOKENS_SUMMARY * 3,
        )
        return (content or "").strip()

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
            # Fall back to default formatting when hints don't apply
            return ""

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

    def format_embedded_field(
        self,
        data: Dict[str, Any],
        field_name: str,
        resource_hints: Optional[Dict[str, Any]] = None,
        resource: Optional[str] = None,
    ) -> Optional[str]:
        """Format a nested/embedded field from a detail API response."""
        if not isinstance(data, dict) or field_name not in data:
            return None
        hints = {}
        if resource_hints and resource:
            hints = dict(resource_hints.get(resource) or {})
        hints["include_fields"] = [field_name]
        return self._format_detailed_with_hints([data], hints)

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

    # =========================================================================
    # PAGINATION AND EMPTY RESPONSE HANDLING
    # =========================================================================

    def _extract_pagination(self, data: Any) -> Dict[str, Any]:
        """
        Extract pagination info from API response.

        Supports common pagination formats:
        - DRF: {count, next, previous, results}
        - Simple: {total, items}
        - Plain list

        Args:
            data: Raw API response

        Returns:
            Dict with total, returned, has_more, next_url
        """
        if not data:
            return {"total": 0, "returned": 0, "has_more": False}

        # DRF pagination format
        if isinstance(data, dict):
            if "results" in data and "count" in data:
                results = data.get("results", [])
                return {
                    "total": data.get("count", 0),
                    "returned": len(results) if isinstance(results, list) else 1,
                    "has_more": data.get("next") is not None,
                    "next_url": data.get("next"),
                    "previous_url": data.get("previous"),
                }

            # Simple total/items format
            if "total" in data and ("items" in data or "data" in data):
                items = data.get("items") or data.get("data", [])
                return {
                    "total": data.get("total", 0),
                    "returned": len(items) if isinstance(items, list) else 1,
                    "has_more": len(items) < data.get("total", 0) if isinstance(items, list) else False,
                }

            # Single object
            return {"total": 1, "returned": 1, "has_more": False}

        # Plain list response
        if isinstance(data, list):
            return {
                "total": len(data),
                "returned": len(data),
                "has_more": False
            }

        return {"total": 1, "returned": 1, "has_more": False}

    def _is_empty_result(self, data: Any) -> bool:
        """
        Check if the API response is empty.

        Args:
            data: API response data

        Returns:
            True if empty, False otherwise
        """
        if data is None:
            return True

        if isinstance(data, list) and len(data) == 0:
            return True

        if isinstance(data, dict):
            # DRF pagination: check results
            if "results" in data and isinstance(data["results"], list):
                return len(data["results"]) == 0

            # Check for empty items/data
            if "items" in data and isinstance(data["items"], list):
                return len(data["items"]) == 0
            if "data" in data and isinstance(data["data"], list):
                return len(data["data"]) == 0

            # Empty dict counts as empty
            if len(data) == 0:
                return True

        return False

    def _format_empty_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate helpful message when query returns no results.

        Provides:
        - Clear acknowledgment that no results found
        - List of filters that were applied
        - Suggestions for broadening the search
        - Valid filter values (if schema hints available)

        Args:
            query: Original user query
            context: Context with resource, filters, schema

        Returns:
            Formatted response dict
        """
        context = context or {}
        resource = context.get("resource", "items")
        filters = context.get("filters", {})
        if not filters and context.get("parsed"):
            from .user_context_resolver import filters_for_display
            filters = filters_for_display(context["parsed"])
        schema = context.get("schema", {})
        resource_hints = schema.get("resource_hints", {}).get(resource, {}) if schema else {}

        # Build the message
        message = constants.EMPTY_RESPONSE_MESSAGE.format(resource=resource)

        # Show filters that were applied
        if filters:
            message += f"\n\n{constants.EMPTY_RESPONSE_FILTERS_APPLIED}"
            for field, val in filters.items():
                if isinstance(val, dict):
                    value = val.get("value", val)
                else:
                    value = val
                message += f"\n  • {field}: {value}"

        message += f"\n\n💡 **Suggestions:**"
        message += f"\n  • {constants.EMPTY_RESPONSE_SUGGESTION}"

        if filters:
            message += f"\n  • Remove filters: \"Show all {resource}\""

        # Show valid filter values from resource_hints
        if resource_hints and filters:
            for field, val in filters.items():
                if isinstance(val, dict):
                    value = val.get("value", val)
                else:
                    value = val

                field_hints = resource_hints.get(field, {})
                if isinstance(field_hints, dict):
                    valid_values = field_hints.get("values", [])
                    if valid_values:
                        display_values = valid_values[:5]
                        values_str = ", ".join(str(v) for v in display_values)
                        if len(valid_values) > 5:
                            values_str += f" (+{len(valid_values) - 5} more)"
                        message += f"\n  • {constants.EMPTY_RESPONSE_VALID_VALUES_HINT.format(field=field, values=values_str)}"

        if not filters:
            message += f"\n\n💡 This {resource} collection may be empty, or you may not have access."
            message += f"\n  • Check with your administrator if you expect to see data here."

        return {
            "format": "text",
            "summary": message,
            "formatted": message,
            "raw_data": [],
            "pagination": {"total": 0, "returned": 0, "has_more": False}
        }

    def _append_pagination_info(
        self,
        summary: str,
        pagination: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Append pagination info to the response summary.

        ALWAYS informs the user when there's more data available.

        Args:
            summary: Current summary text
            pagination: Pagination info dict
            context: Context with resource name

        Returns:
            Summary with pagination info appended
        """
        total = pagination.get("total", 0)
        returned = pagination.get("returned", 0)
        has_more = pagination.get("has_more", False)

        # All data shown
        if not has_more and returned == total:
            if total > 0:
                summary += f"\n\n✅ Showing all {total} result(s)."
            return summary

        # Partial page: clarify total vs shown (count vs list consistency)
        if total > returned > 0:
            resource = context.get("resource", "items") if context else "items"
            summary += (
                f"\n\n📊 There are {total} {resource} "
                f"(showing first {returned})."
            )
            if has_more:
                pagination_info = {
                    "total_count": total,
                    "actual_count": returned,
                    "has_more": has_more,
                }
                follow_ups = generate_follow_up_queries(pagination_info, context)
                if follow_ups:
                    summary += f"\n\n💡 **To see more:**"
                    for item in follow_ups:
                        summary += f"\n  • \"{item['query']}\" - {item['label']}"
            return summary

        # More data available - ALWAYS inform user (aligned with follow_up_queries templates)
        if has_more or returned < total:
            resource = context.get("resource", "items") if context else "items"
            pagination_info = {
                "total_count": total,
                "actual_count": returned,
                "has_more": has_more,
            }
            follow_ups = generate_follow_up_queries(pagination_info, context)

            summary += f"\n\n📊 **{constants.PAGINATION_INFO_TEMPLATE.format(shown=returned, total=total, resource=resource)}**"
            if follow_ups:
                summary += f"\n\n💡 **To see more:**"
                for item in follow_ups:
                    summary += f"\n  • \"{item['query']}\" - {item['label']}"

            if total > 100:
                summary += f"\n  • Consider exporting for {total}+ records"

        return summary
