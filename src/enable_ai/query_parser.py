"""
Query Parser / Intent Analyser - LLM-based understanding of natural language queries.

This module is the intent analyser: it turns user text into structured intent, resource,
filters, and display preferences. The orchestrator calls it via _understand_query().

Supports:
- API Schema (REST endpoints)
- Database Schema (Tables & columns)
- Knowledge Graph (Entities & relationships for PDFs/docs)

Features:
- Intent detection (read/create/update/delete) and resource extraction
- Natural language understanding (any phrasing)
- Complex query parsing (multiple conditions, relationships)
- Schema-aware extraction (validates against schema)
- Relationship detection (joins, nested queries)
- Follow-up context: "show me more", "next page", refinement ("of them", filters merge)
- Date/time calculation (relative dates like "last week")
"""

import json
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from .types import APIError
from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP


class QueryParser:
    """
    Intent analyser and query parser: understands natural language and outputs
    structured intent, resource, filters, display_mode, question_type, and
    use_next_page/next_page_url for "show me more".
    
    Used by APIOrchestrator._understand_query(). Advantages over regex:
    - Understands ANY phrasing (not just keywords)
    - Handles complex conditions and relationships
    - Calculates relative dates ("last week" → actual date)
    - Schema-aware (knows valid resources, fields, values)
    - Multi-turn: get_query_context() for refinement and "show me more"
    - Multi-language support potential
    """
    
    def __init__(self):
        """Initialize LLM parser with OpenAI client."""
        self.openai_client = get_openai_client()
        self.logger = setup_logger('enable_ai.parser')
        
        # Cache for parsed queries (reduce costs)
        self.cache = {}
        
        self.logger.info("Parser initialized")
    
    def parse_input(
        self,
        natural_language_input: str,
        schema: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[list] = None,
        user_context: Optional[Dict[str, Any]] = None,  # v0.3.29: User identity for pronoun resolution
        classification_hint: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], APIError]:
        """
        Parse natural language input using LLM with schema context and conversation history.

        Args:
            natural_language_input: User's natural language query
            schema: Active schema (api, database, or knowledge_graph)
            conversation_history: Previous messages for multi-turn context (list of {role, content})
            user_context: Optional user identity for resolving "me"/"my" pronouns (v0.3.29)
                         Structure: {'user_id': int, 'username': str, 'role': str, 'company_id': int, 'company': str}

        Returns:
            {
                'intent': 'read|create|update|delete',
                'resource': 'resource_name',
                'entities': {...},
                'filters': {...},
                'relationships': [...],
                'sort': {...},
                'limit': int,
                'original_input': '...',
                'schema_type': 'api|database|knowledge_graph'
            }

        Examples:
            >>> parse_input("get user with id 5", api_schema)
            {
                'intent': 'read',
                'resource': 'users',
                'entities': {'id': 5},
                'filters': {'id': {'operator': 'equals', 'value': 5}}
            }

            >>> parse_input("show service orders assigned to me", api_schema, user_context={'user_id': 123})
            {
                'intent': 'read',
                'resource': 'service_orders',
                'entities': {'technician': 123},
                'filters': {'technician': {'operator': 'equals', 'value': 123}}
            }
        """
        if not natural_language_input or not natural_language_input.strip():
            return APIError("Empty input provided")

        if not schema:
            return APIError("Schema is required for LLM parsing")

        try:
            # LLM-first: synonyms live in resource_hints inside the prompt.
            # Post-parse semantic_filters remains the safety net (no query rewriting).

            # Check cache first (only if no user_context - personalized queries shouldn't be cached)
            cache_key = self._get_cache_key(natural_language_input, schema)
            if cache_key in self.cache and not user_context and not conversation_history:
                self.logger.info(f"Using cached parse result for: '{natural_language_input[:50]}...'")
                return self.cache[cache_key]

            # Build prompt with schema context and user context
            prompt = self._build_prompt(
                natural_language_input, schema, user_context, classification_hint,
            )
            
            # Build messages list with conversation history for context (Issue #2 fix)
            messages = [{"role": "system", "content": self._get_system_prompt()}]
            
            # Add conversation history if provided (but exclude context markers)
            if conversation_history:
                clean_history = self._clean_conversation_history(conversation_history)
                messages.extend(clean_history)
                self.logger.debug(f"Including {len(clean_history)} previous messages for context")
            
            # Add current query
            messages.append({"role": "user", "content": prompt})
            
            # Call LLM with function calling if conversation history exists
            self.logger.info(f"Parsing query: '{natural_language_input}'")
            
            if conversation_history:
                from .follow_up_detection import classify_follow_up

                if classify_follow_up(natural_language_input, conversation_history).get("is_follow_up"):
                    self.logger.info(
                        "Follow-up detected — forcing context merge for: %r",
                        natural_language_input[:80],
                    )
                    parsed = self._parse_with_forced_context(
                        messages, schema, conversation_history, user_context,
                    )
                else:
                    # Use function calling; LLM may request context via tool
                    parsed = self._parse_with_context_function(
                        messages, schema, conversation_history, user_context,
                    )
            else:
                # No conversation history - regular parsing
                parsed = self.openai_client.parse_json_response(
                    messages=messages,
                    temperature=DETERMINISTIC_TEMP
                )
            
            # Add metadata
            parsed['original_input'] = natural_language_input
            parsed['schema_type'] = schema.get('type')
            
            # Validate against schema
            validated = self._validate_parsed_output(parsed, schema)
            
            # Cache result
            self.cache[cache_key] = validated
            
            self.logger.info(
                f"Parse complete: intent={validated['intent']}, "
                f"resource={validated.get('resource', 'N/A')}"
            )
            
            return validated
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from LLM: {str(e)}")
            return APIError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            self.logger.error(f"Parse failed: {str(e)}")
            return APIError(f"LLM parsing failed: {str(e)}")
    
    # ========================================================================
    # LLM PROMPT BUILDING
    # ========================================================================
    
    def _get_system_prompt(self) -> str:
        """
        System prompt that defines the LLM's role and output format.

        Returns:
            System prompt string
        """
        return """You are an expert query parser for a natural language to API/Database system.

Your task: Understand the user's intent and extract structured information so the system can execute the right operations.

**Intent and context**
- Decide whether the user is starting a new request, continuing or refining the previous one (referring to prior results), asking for a count/total, asking for the next page of results, or changing how results are shown.
- When the user's intent clearly refers to or continues the previous turn (e.g. "them", "those", "the same", "more", "next page", "and which company?", refining or filtering prior results), call get_query_context() to retrieve previous_resource and previous_filters, then use that resource and merge filters. Set merge_with_previous=true.
- Follow-up refinements like "and assigned to which company?" after asking about service orders mean: stay on service-orders, keep previous filters, set question_type="details" — do NOT list all companies.
- Reset queries like "show all service orders" or "list all users" start fresh: set merge_with_previous=false and do NOT carry over previous filters even if conversation history exists.
- When the user is asking for the next page of a prior list (e.g. more results, next page), use context's next_url and set use_next_page=true and next_page_url from context.
- When the user is asking for a total or count (how many, number of, total), set question_type="count".
- When the user wants a small sample of prior results (e.g. "a few", "some"), set a small limit and question_type="list" so they see items, not only a count.

**IMPORTANT: Follow-up questions asking for details**
- When the user asks "which are those?", "what are they?", "show me the list", "what are their names?", "list them", "show them", "which ones?", or similar questions that ask for details about previous results:
  1. Call get_query_context() to get the previous resource and filters
  2. Set question_type="list" or question_type="details" (NOT "count")
  3. Set display_mode="detailed" or display_mode="full" so the system shows actual item details
  4. Set merge_with_previous=true
  5. Do NOT just return a count - the user wants to SEE the actual items
- These follow-up questions mean: "Show me the actual items you just mentioned" - so ensure the response includes item names, IDs, or other identifying information.

**User Identity / Pronoun Resolution (IMPORTANT)**
- When the user says "me", "my", "mine", "assigned to me", "my reports", "my orders", etc., they are referring to THEMSELVES.
- If USER_CONTEXT is provided in the prompt, use the user_id from that context to resolve these pronouns.
- Map pronouns to the appropriate filter field based on the resource:
  - "service orders assigned to me" → filter by technician={user_id}
  - "my reports" or "reports I created" → filter by technician={user_id} or created_by={user_id}
  - "my company's orders" → filter by company={company_id}
  - "customers in my company" → filter by company={company_id}
- Always use the actual user_id/company_id values, never leave "me" or "my" as string values in filters.

**CRITICAL: NO HARDCODED DATA - ANTI-HALLUCINATION RULES**
- NEVER generate example data, user lists, or sample responses
- NEVER return specific IDs, names, or values that are NOT explicitly mentioned in the user's query
- ONLY extract: intent, resource, filters, sort, limit from what the user ACTUALLY said
- If user asks for "orders" without specifying which ones, return EMPTY filters - do NOT guess IDs
- If user says "my orders" but no USER_CONTEXT is provided, set question_type="needs_clarification"
- Filter values MUST come from ONE of these sources:
  1. The user's explicit query text
  2. USER_CONTEXT (for "me"/"my" pronouns)
  3. Conversation history via get_query_context() (for "those"/"them" references)
- NEVER invent example values like "ABC", "123", "John", or sample names
- If you cannot trace a filter value to the sources above, OMIT it entirely

EXTRACT THE FOLLOWING:

1. **intent** (required) - CRUD operation:
   - "read": get, show, find, list, fetch, retrieve, display, view, search
   - "create": create, add, insert, new, make, register, post
   - "update": update, modify, change, edit, set, alter, patch, put
   - "delete": delete, remove, drop, destroy, cancel

2. **resource** (required) - Target entity/table name
   - Use ONLY resources defined in the schema
   - Map synonyms and abbreviations (e.g., "SOs" → "service_orders")
   - Use canonical form (usually plural: "users", "orders")
   - For follow-up or refinement intents, use the resource from get_query_context() unless the user clearly switches topic

3. **entities** (optional) - Field-value pairs for filtering/matching
   - Extract ALL mentioned field values
   - Convert to appropriate data types (int, string, bool, date)
   - Calculate relative dates (e.g., "last week" → actual date range)
   - **For relationship filters**: Also add to entities using pattern {target_entity}_{field}
     Example: Query "service orders for company ABC" → entities: {"companies_name": "ABC"}

4. **filters** (optional) - Query conditions with operators
   - Structure: {"field": {"operator": "...", "value": ...}}
   - Operators: equals, not_equals, gt, gte, lt, lte, contains, starts_with, ends_with, in, not_in
   - For date ranges: use gte/lte for "between", "last N days", etc.
   - **For relationship filters**: Include the flattened field name {target_entity}_{field}

5. **relationships** (optional) - Joins or nested entities
   - Structure: [{"type": "...", "target_entity": "...", "filters": {...}}]
   - Examples: "employees who work at companies in tech sector"
   - Use this for complex joins, foreign keys, nested queries

6. **sort** (optional) - Ordering preference
   - Structure: {"field": "...", "order": "asc" or "desc"}
   - Default: often by created_date desc or id asc

7. **limit** (optional) - Number of results to return
   - Infer from user phrasing (e.g. "top 10", "first 5", or "a few" → small limit so they see example items)
   - If not specified, omit (let system use default)

8. **display_mode** (optional) - How to display results
   - "summary": Brief summary (default)
   - "full": User wants the complete list
   - "detailed": User wants a detailed or expanded view
   - Infer from intent (e.g. "all", "everything", "in detail")

9. **merge_with_previous** (optional) - Whether this continues or refines the previous query
   - true: User is refining, filtering, or continuing the same topic
   - false: New, independent request

10. **question_type** - What the user wants to see
   - "count": User wants a total or number (infer from intent)
   - "list": User wants to see items
   - "details": User wants detailed info about specific item(s)

RULES:
- Use ONLY field names and resources defined in the provided schema
- Map synonyms to schema names; calculate relative dates from today when provided
- Return valid JSON in the exact format below; omit fields you cannot determine
- For relationship filters, populate BOTH entities and relationships

OUTPUT FORMAT (JSON):
{
    "intent": "read|create|update|delete",
    "resource": "resource_name",
    "entities": {
        "field_name": "value",
        "related_entity_field": "value"
    },
    "filters": {
        "field_name": {
            "operator": "equals|gt|gte|lt|lte|contains|...",
            "value": "..."
        }
    },
    "relationships": [
        {
            "type": "RELATIONSHIP_TYPE",
            "target_entity": "entity_name",
            "filters": {...}
        }
    ],
    "sort": {
        "field": "field_name",
        "order": "asc|desc"
    },
    "limit": 10,
    "display_mode": "summary|full|detailed",
    "merge_with_previous": true|false,
    "question_type": "count|list|details",
    "use_next_page": true|false,
    "next_page_url": "url or omit"
}

EXAMPLES:

Example 1 - Simple filter:
Query: "get user with id 5"
Output: {
    "intent": "read",
    "resource": "users",
    "entities": {"id": 5},
    "filters": {"id": {"operator": "equals", "value": 5}}
}

Example 2 - Multi-resource / "X and their Y" (populate relationships for multi-step):
Query: "get users and their orders" or "show me users and their orders"
Output: {
    "intent": "read",
    "resource": "users",
    "entities": {},
    "filters": {},
    "relationships": [
        {
            "type": "has_many",
            "target_entity": "orders",
            "filters": {}
        }
    ],
    "question_type": "list"
}

Example 3 - Relationship filter (populate both entities AND relationships):
Query: "show orders for customer John"
Output: {
    "intent": "read",
    "resource": "orders",
    "entities": {
        "customers_name": "John"
    },
    "filters": {
        "customers_name": {"operator": "equals", "value": "John"}
    },
    "relationships": [
        {
            "type": "belongs_to",
            "target_entity": "customers",
            "filters": {"name": {"operator": "equals", "value": "John"}}
        }
    ]
}

Example 4 - Multiple conditions with relationship:
Query: "high priority tasks assigned to team Alpha created last week"
Output: {
    "intent": "read",
    "resource": "tasks",
    "entities": {
        "priority": "high",
        "teams_name": "Alpha",
        "created_after": "2024-01-13"
    },
    "filters": {
        "priority": {"operator": "equals", "value": "high"},
        "teams_name": {"operator": "equals", "value": "Alpha"},
        "created_date": {"operator": "gte", "value": "2024-01-13"}
    },
    "relationships": [
        {
            "type": "belongs_to",
            "target_entity": "teams",
            "filters": {"name": {"operator": "equals", "value": "Alpha"}}
        }
    ],
    "question_type": "list"
}

Example 5 - Refinement + count intent (user refers to prior results and asks for a total):
Query: "how many of them are named BOROSCOPE?"
Context from get_query_context(): { "previous_resource": "equipment", "previous_filters": {"status": {"operator": "equals", "value": "low_stock"}} }
Output: {
    "intent": "read",
    "resource": "equipment",
    "entities": {"status": "low_stock", "name": "BOROSCOPE"},
    "filters": {
        "status": {"operator": "equals", "value": "low_stock"},
        "name": {"operator": "contains", "value": "BOROSCOPE"}
    },
    "merge_with_previous": true,
    "question_type": "count"
}

Example 6 - User pronoun resolution ("assigned to me"):
Query: "show me the new service orders assigned to me"
USER_CONTEXT: {"user_id": 123, "username": "john@example.com", "role": "Technician"}
Output: {
    "intent": "read",
    "resource": "service_orders",
    "entities": {"technician": 123, "status": "New"},
    "filters": {
        "technician": {"operator": "equals", "value": 123},
        "status__name": {"operator": "equals", "value": "New"}
    },
    "question_type": "list"
}

Example 7 - "Observations for my last report" (multi-step: get last report, then its observations):
Query: "what are the observations for my last report?" or "observations for my last report"
USER_CONTEXT: {"user_id": 456, "username": "jane@example.com", "role": "Technician"}
Output: {
    "intent": "read",
    "resource": "details_reports",
    "entities": {"technician": 456},
    "filters": {
        "technician": {"operator": "equals", "value": 456}
    },
    "sort": {"field": "created_at", "order": "desc"},
    "limit": 1,
    "question_type": "details",
    "relationships": [
        {"type": "child", "target_entity": "observations", "filters": {}}
    ]
}
NOTE: Populate relationships so the planner generates step 1 (get report with technician=me, limit 1, sort desc) and step 2 (get observations for that report). Use the schema's child resource name for target_entity (e.g. "observations").

Example 8 - Follow-up asking for details (IMPORTANT - user wants to see the items, not just count):
Previous conversation: User asked "Show me new service orders assigned to me" → System responded "Found 9 items"
Query: "which are those?" or "what are they?" or "show me the list"
Context from get_query_context(): { "previous_resource": "service_orders", "previous_filters": {"technician": {"operator": "equals", "value": 123}, "status__name": {"operator": "equals", "value": "New"}} }
Output: {
    "intent": "read",
    "resource": "service_orders",
    "entities": {"technician": 123, "status": "New"},
    "filters": {
        "technician": {"operator": "equals", "value": 123},
        "status__name": {"operator": "equals", "value": "New"}
    },
    "merge_with_previous": true,
    "question_type": "list",
    "display_mode": "detailed"
}
NOTE: The user is asking to SEE the actual items - set question_type="list" and display_mode="detailed", NOT question_type="count"!

Return ONLY the JSON object, no explanations or markdown.
"""

    def _get_filterable_fields_from_hints(self, resource_hints: Dict[str, Any]) -> str:
        """
        Build a dynamic list of resource.field that have values/synonyms in resource_hints,
        for injection into the parser prompt so the LLM knows which fields to map from user words.
        """
        parts = []
        for res_name, res_hints in (resource_hints or {}).items():
            if not isinstance(res_hints, dict):
                continue
            fields = []
            for field_name, field_hints in res_hints.items():
                if field_name.startswith("__"):
                    continue
                if isinstance(field_hints, dict) and (
                    field_hints.get("values") or field_hints.get("synonyms")
                ):
                    fields.append(field_name)
            if fields:
                parts.append(f"{res_name}: {', '.join(fields)}")
        return "; ".join(parts) if parts else "(see hints above for each resource)"

    def _build_prompt(
        self,
        query: str,
        schema: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,  # v0.3.29: User identity
        classification_hint: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build user prompt with schema context, query, and user context.

        Args:
            query: Natural language query
            schema: Active schema
            user_context: Optional user identity for pronoun resolution

        Returns:
            Formatted prompt string
        """
        schema_type = schema.get('type')
        today = datetime.now().strftime('%Y-%m-%d')

        # Extract schema information
        if schema_type == 'api':
            schema_info = self._extract_api_schema_info(schema)
        elif schema_type == 'database':
            schema_info = self._extract_database_schema_info(schema)
        elif schema_type == 'knowledge_graph':
            schema_info = self._extract_kg_schema_info(schema)
        else:
            schema_info = {"resources": [], "fields": {}}

        # Optional: allowed values and synonyms per resource/field (configurable via resource_hints)
        hints_section = ""
        resource_hints = schema.get("resource_hints") or {}
        if resource_hints:
            hints_section = "\nALLOWED VALUES AND SYNONYMS (use these exact values in filters; map user words via synonyms):\n"
            hints_section += json.dumps(resource_hints, indent=2)

            # Build dynamic list of filterable fields (resource.field with values/synonyms)
            filterable_list = self._get_filterable_fields_from_hints(resource_hints)
            filterable_instruction = (
                f"   Filterable fields (use ONLY canonical values from the hints above): {filterable_list}\n"
                "   Set filters by mapping the user's words to the canonical value using each field's \"values\" and \"synonyms\". Do not invent values."
            )
            # Placeholder for literal in prompt (avoid f-string interpreting {FIELD}, {operator}, {value}, VALUE)
            _pattern_value = "VALUE"
            _pattern_example = f"filters: {{FIELD: {{operator: \"equals\", value: {_pattern_value}}}}}"

            hints_section += f"""

CRITICAL - SEMANTIC PHRASE MAPPING (v0.3.42):
When the user uses descriptive phrases, map them to the corresponding filter field and value:

1. RESOURCE SYNONYMS (__resource_synonyms__):
   - "items", "supplies", "materials" → resource: "consumables"
   - "company", "customer", "client" → resource: "companies"
   - "flash report", "flash" → resource: "flash-reports"
   - Use the resource name that has these words in its __resource_synonyms__ list

2. SEMANTIC FILTER PHRASES:
   Look for field "synonyms" in the hints above. When user says a synonym KEY, use that field with the KEY as value:
   - "low in stock", "low stock", "running low" → filters: {{"stock_level": {{"operator": "equals", "value": "low"}}}}
   - "out of stock", "empty" → filters: {{"stock_level": {{"operator": "equals", "value": "out of stock"}}}}
   - The backend will translate "low" to the actual API filter (e.g., current_quantity__lt=10)

3. FILTERABLE FIELDS (schema-driven, from hints above):
{filterable_instruction}

4. "OBSERVATIONS FOR MY LAST REPORT" (single-step):
   When the user asks for observations/details of their last/latest report, set resource to the PARENT resource (e.g. details-reports), add filters for "me" from USER_CONTEXT (e.g. technician=user_id), sort by created_at desc, limit 1. The observations are embedded in the report response — do NOT use relationships or add a second step. Just fetch the report with the right filters and the system will display all its fields including observations.

5. EXAMPLES:
   - "show me consumables with stock_level low" → resource: "consumables", filters: {{"stock_level": {{"operator": "equals", "value": "low"}}}}
   - "list items low in stock" → resource: "consumables", filters: {{"stock_level": {{"operator": "equals", "value": "low"}}}}
   - "show companies" → resource: "companies"
   - "list flash reports" → resource: "flash-reports"
   - For any resource and filterable field listed in (3) above: when the user's words match a synonym or value for that field, set the filter using the canonical value from the hints (e.g. users.role, consumables.stock_level).

When you see "with FIELD VALUE" patterns, extract as: {_pattern_example}
When the user says e.g. "pending" or "quoted", use the synonym or value above.
Prefer the exact strings listed in "values" or given by "synonyms".
"""

        classification_section = ""
        if classification_hint:
            classification_section = f"""
CLASSIFICATION_HINT (optional rule-based guess — verify against the query and schema; LLM decision wins):
{json.dumps(classification_hint, indent=2)}
Use this only as a starting hint. Override when the query, follow-up context, or synonyms imply a different resource/filters.
"""

        # v0.3.29: User context section for pronoun resolution
        user_context_section = ""
        if user_context:
            user_context_section = f"""
USER_CONTEXT (use this to resolve "me", "my", "mine", "assigned to me" pronouns):
{json.dumps(user_context, indent=2)}

IMPORTANT: When the query contains "me", "my", "mine", "assigned to me", "my reports", etc.:
- Use user_id={user_context.get('user_id')} for technician/created_by/assigned_to filters
- Use company_id={user_context.get('company_id')} for company filters
- User's role is: {user_context.get('role', 'Unknown')}
- NEVER leave "me" or "my" as string values in filters - always resolve to actual IDs
"""

        return f"""Parse this natural language query:
"{query}"

TODAY'S DATE: {today}
{classification_section}{user_context_section}
AVAILABLE SCHEMA:

Resources/Tables:
{json.dumps(schema_info['resources'], indent=2)}

Fields by resource:
{json.dumps(schema_info['fields'], indent=2)}
{hints_section}
INSTRUCTIONS:
1. Extract intent, resource, entities, and filters from the query
2. Use ONLY resources and fields defined above
3. Calculate any relative dates based on today's date ({today})
4. Return valid JSON matching the format in the system prompt
5. Be precise - map the query to the exact schema structure
6. For status/filter fields with allowed values above, use EXACTLY those values (or their synonym mapping)
7. If USER_CONTEXT is provided and query contains "me"/"my" pronouns, resolve them to actual user_id/company_id values

Return the parsed JSON now:
"""
    
    def _extract_api_schema_info(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant info from API schema for LLM.
        
        Preference order:
        1. Explicit resource-level "fields" metadata (rich schemas)
        2. Fallback to inferring fields from endpoint parameters
        """
        resources = list(schema.get('resources', {}).keys())
        fields: Dict[str, Any] = {}
        
        for resource_name, resource_def in schema.get('resources', {}).items():
            # Prefer explicit fields if provided by the schema / converter
            explicit_fields = resource_def.get('fields')
            if isinstance(explicit_fields, (list, dict)):
                if isinstance(explicit_fields, list):
                    fields[resource_name] = explicit_fields
                else:
                    # dict mapping field -> metadata
                    fields[resource_name] = list(explicit_fields.keys())
                continue
            
            # Fallback: derive from first endpoint's parameters
            endpoints = resource_def.get('endpoints', [])
            if endpoints:
                first_endpoint = endpoints[0]
                params = first_endpoint.get('parameters', {})

                # Collect all parameter names
                field_names = set()

                # Parameters may be provided either as simple strings or as
                # dict objects with a "name" key. Avoid adding dict objects
                # directly to the set, as that would raise
                # "unhashable type: 'dict'". Instead, normalize everything
                # through the loop below.
                for param_list in [
                    params.get('path', []),
                    params.get('query', []),
                    params.get('body', []),
                ]:
                    if isinstance(param_list, list):
                        for param in param_list:
                            if isinstance(param, dict) and 'name' in param:
                                field_names.add(param['name'])
                            elif isinstance(param, str):
                                field_names.add(param)
                
                if field_names:
                    fields[resource_name] = list(field_names)
                else:
                    fields[resource_name] = ['id', 'name', 'created_date']
            else:
                # No endpoints metadata – fall back to generic defaults
                fields[resource_name] = ['id', 'name', 'created_date']
        
        return {
            'resources': resources,
            'fields': fields
        }
    
    def _extract_database_schema_info(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant info from database schema for LLM."""
        tables = list(schema.get('tables', {}).keys())
        fields = {}
        
        for table_name, table_def in schema.get('tables', {}).items():
            columns = list(table_def.get('columns', {}).keys())
            fields[table_name] = columns if columns else ['id', 'name', 'created_at']
        
        return {
            'resources': tables,
            'fields': fields
        }
    
    def _extract_kg_schema_info(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant info from knowledge graph schema for LLM."""
        entities = list(schema.get('entities', {}).keys())
        fields = {}
        
        for entity_type, entity_def in schema.get('entities', {}).items():
            properties = list(entity_def.get('properties', {}).keys())
            fields[entity_type] = properties if properties else ['id', 'name', 'description']
        
        return {
            'resources': entities,
            'fields': fields
        }
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def _validate_parsed_output(self, parsed: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate LLM output against schema.

        Ensures:
        - Intent is valid CRUD operation
        - Resource exists in schema
        - Fields are valid for the resource
        - Operators are supported

        Args:
            parsed: LLM parsed output
            schema: Active schema

        Returns:
            Validated parsed dict

        Raises:
            ValueError: If validation fails
        """
        schema_type = schema.get('type')

        # Validate intent
        valid_intents = ['read', 'create', 'update', 'delete', 'search']
        if parsed.get('intent') not in valid_intents:
            # Default to read if invalid
            parsed['intent'] = 'read'

        # Validate resource
        if schema_type == 'api':
            valid_resources = list(schema.get('resources', {}).keys())
        elif schema_type == 'database':
            valid_resources = list(schema.get('tables', {}).keys())
        elif schema_type == 'knowledge_graph':
            valid_resources = list(schema.get('entities', {}).keys())
        else:
            valid_resources = []

        resource_found = False
        if parsed.get('resource') in valid_resources:
            resource_found = True
        else:
            # Try to find closest match
            resource = parsed.get('resource', '').lower()
            for valid_resource in valid_resources:
                if resource in valid_resource.lower() or valid_resource.lower() in resource:
                    parsed['resource'] = valid_resource
                    resource_found = True
                    break

            # v0.3.29: Also check resource_hints for synonyms
            if not resource_found:
                resource_hints = schema.get('resource_hints', {})
                for res_name, hints in resource_hints.items():
                    synonyms = hints.get('__resource_synonyms__', [])
                    if resource in [s.lower() for s in synonyms]:
                        parsed['resource'] = res_name
                        resource_found = True
                        break

        # v0.3.29: Mark as unknown_intent if resource not found
        if not resource_found and parsed.get('resource'):
            self.logger.warning(f"Resource '{parsed.get('resource')}' not found in schema. Available: {valid_resources}")
            parsed['unknown_intent'] = True
            parsed['available_resources'] = valid_resources

        # Ensure required fields exist
        if 'entities' not in parsed:
            parsed['entities'] = {}

        if 'filters' not in parsed:
            parsed['filters'] = {}

        # BUG FIX (v0.3.36): Validate filter fields exist in schema
        # This prevents hallucinating filter results when a field doesn't exist
        if resource_found and parsed.get('filters') and schema_type == 'api':
            resource_name = parsed.get('resource')
            resource_schema = schema.get('resources', {}).get(resource_name, {})

            # Extract valid query parameters from GET endpoints
            valid_params = set()
            for endpoint in resource_schema.get('endpoints', []):
                if endpoint.get('method', '').upper() == 'GET':
                    params = endpoint.get('parameters', {})
                    query_params = params.get('query', []) if isinstance(params, dict) else []
                    for param in query_params:
                        if isinstance(param, dict):
                            param_name = param.get('name', '')
                            if param_name:
                                valid_params.add(param_name)
                                # Also add base name without Django lookups
                                base_name = param_name.split('__')[0]
                                valid_params.add(base_name)
                        elif isinstance(param, str):
                            valid_params.add(param)
                            valid_params.add(param.split('__')[0])

            # Also add fields from resource_hints
            resource_hints = schema.get('resource_hints', {}).get(resource_name, {})
            if isinstance(resource_hints, dict):
                for field_name in resource_hints.keys():
                    if not field_name.startswith('__'):
                        valid_params.add(field_name)

            # Warn and remove invalid filter fields
            invalid_filters = []
            for filter_field in list(parsed['filters'].keys()):
                base_field = filter_field.split('__')[0]
                if filter_field not in valid_params and base_field not in valid_params:
                    # Check if it's a common field that might be aliased
                    # e.g., 'technician' -> 'technician__id' or 'assigned_to'
                    is_valid = False
                    for vp in valid_params:
                        if base_field in vp or vp.startswith(base_field + '__'):
                            is_valid = True
                            break

                    if not is_valid:
                        invalid_filters.append(filter_field)
                        self.logger.warning(
                            f"Filter field '{filter_field}' not found in schema for {resource_name}. "
                            f"Valid params: {sorted(valid_params)[:10]}... Removing to prevent hallucination."
                        )
                        del parsed['filters'][filter_field]
                        # Also remove from entities
                        if filter_field in parsed.get('entities', {}):
                            del parsed['entities'][filter_field]

            if invalid_filters:
                parsed['_removed_invalid_filters'] = invalid_filters

        # Convert entities to filters if filters are empty
        if parsed['entities'] and not parsed['filters']:
            for key, value in parsed['entities'].items():
                parsed['filters'][key] = {
                    'operator': 'equals',
                    'value': value
                }

        # Anti-hardcoding check: warn and optionally remove suspicious values
        original_input = parsed.get('original_input', '').lower()
        entities_to_check = list(parsed.get('entities', {}).items())

        for field, value in entities_to_check:
            if self._looks_like_hardcoded(value, original_input):
                self.logger.warning(
                    f"Potentially hardcoded value detected: {field}={value}. "
                    f"Value not found in original input. Removing suspicious entity."
                )
                # Remove suspicious values to prevent hallucination
                del parsed['entities'][field]
                # Also remove from filters if present
                if field in parsed.get('filters', {}):
                    del parsed['filters'][field]

        return parsed

    def _looks_like_hardcoded(self, value: Any, original_input: str) -> bool:
        """
        Check if a value appears to be hardcoded (not from user input).

        This helps detect LLM hallucinations where it invents example data.

        Args:
            value: The value to check
            original_input: The original user query (lowercase)

        Returns:
            True if value appears to be hardcoded/hallucinated
        """
        if value is None:
            return False

        value_str = str(value).lower()

        # Check if value appears in original input
        if value_str in original_input:
            return False

        # Allow common defaults and boolean values
        allowed_defaults = {
            'true', 'false', 'null', 'none', 'yes', 'no',
            'asc', 'desc', 'ascending', 'descending',
            'active', 'inactive', 'enabled', 'disabled',
        }
        if value_str in allowed_defaults:
            return False

        # Allow small integers (often legitimate IDs from context)
        if isinstance(value, int) and value < 10:
            return False

        # Allow values that are just digits (might be parsed from input)
        if value_str.isdigit():
            return False

        # Suspicious: specific string names that don't appear in input
        if isinstance(value, str) and len(value) > 2:
            # Check for common example/placeholder patterns
            suspicious_patterns = [
                'example', 'sample', 'test', 'demo', 'john', 'jane',
                'abc', 'xyz', 'foo', 'bar', 'acme', 'widget',
            ]
            for pattern in suspicious_patterns:
                if pattern in value_str:
                    return True

            # If it's a name-like string (capitalized) not in input, suspicious
            if value[0].isupper() and value_str not in original_input:
                return True

        return False
    
    # ========================================================================
    # FUNCTION CALLING FOR CONTEXT (v0.3.6)
    # ========================================================================
    
    def _clean_conversation_history(self, conversation_history: list) -> list:
        """
        Remove context markers from conversation history to avoid confusing the LLM.
        
        Args:
            conversation_history: Raw conversation history with context markers
            
        Returns:
            Cleaned conversation history
        """
        cleaned = []
        for msg in conversation_history:
            content = msg.get('content', '')
            # Remove [Context: ...] markers
            if '\n[Context:' in content:
                content = content.split('\n[Context:')[0]
            cleaned.append({
                'role': msg['role'],
                'content': content
            })
        return cleaned
    
    def _extract_context_from_history(self, conversation_history: list) -> Dict[str, Any]:
        """
        Extract structured context from conversation history.
        
        Looks for [Context: intent operation on resource] markers in assistant messages.
        Enhanced in v0.3.7 to extract filters for query merging.
        
        Args:
            conversation_history: Conversation history with context markers
            
        Returns:
            Dict with previous_resource, previous_intent, previous_query, previous_filters
        """
        context: Dict[str, Any] = {
            "previous_resource": None,
            "previous_intent": None,
            "previous_query": None,
            "previous_filters": None,
            "next_url": None,
        }
        
        # Look through conversation history in reverse (most recent first)
        for msg in reversed(conversation_history):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Look for [Context: intent operation on resource]
                if '[Context:' in content:
                    import re
                    metadata = msg.get('metadata') or {}
                    if isinstance(metadata, dict) and metadata.get('next_url'):
                        context['next_url'] = metadata['next_url']
                    match = re.search(r'\[Context: (\w+) operation on ([\w_]+)\]', content)
                    if match:
                        context['previous_intent'] = match.group(1)
                        context['previous_resource'] = match.group(2)
                        
                        # Try to extract filters if present
                        filters_match = re.search(r'\[Filters: (.*?)\]', content)
                        if filters_match:
                            try:
                                import json
                                context['previous_filters'] = json.loads(filters_match.group(1))
                            except Exception:
                                pass
                        break
            elif msg.get('role') == 'user':
                if context['previous_query'] is None:
                    context['previous_query'] = msg.get('content', '')
        
        return context
    
    def _define_context_tool(self) -> Dict[str, Any]:
        """
        Define the get_query_context tool for OpenAI function calling (v0.3.12 enhanced).
        
        Returns:
            Tool definition dict
        """
        return {
            "type": "function",
            "function": {
                "name": "get_query_context",
                "description": """Call this when the user's intent refers to or continues the previous query: e.g. referring to prior results ("them", "those"), refining or filtering those results, asking for a specific number (e.g. "first 5", "show me 5"), or asking for the next page of results.

Returns: previous_resource, previous_intent, previous_query, previous_filters, and next_url (if the previous response had more pages). Use previous_resource; merge previous_filters with any new conditions from the current query. If the user specifies how many results to return, the final parse must include limit set to that number (the system maps limit to the API's parameter from the schema). If the user is asking for the next page of results and next_url is provided, set use_next_page=true and next_page_url to that value.""",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    def _parse_with_context_function(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        conversation_history: list,
        user_context: Optional[Dict[str, Any]] = None  # v0.3.29: User identity
    ) -> Dict[str, Any]:
        """
        Parse query using OpenAI function calling to provide structured context (v0.3.12 enhanced).

        Args:
            messages: Messages for the LLM
            schema: Schema for validation
            conversation_history: Full conversation history with context markers
            user_context: Optional user identity for pronoun resolution

        Returns:
            Parsed query dict
        """
        # Define the context retrieval tool
        tools = [self._define_context_tool()]
        
        # First call: Let LLM decide if it needs context
        self.logger.info("🤖 Calling LLM with context function available...")
        response = self.openai_client.chat_completion_with_tools(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=DETERMINISTIC_TEMP
        )
        
        message = response.choices[0].message
        
        # Check if LLM called the function
        if message.tool_calls:
            self.logger.info("✅ LLM called get_query_context() - extracting previous context")
            
            # Extract context from conversation history
            context = self._extract_context_from_history(conversation_history)
            
            self.logger.info(f"📋 Context extracted:")
            self.logger.info(f"   - previous_resource: {context.get('previous_resource')}")
            self.logger.info(f"   - previous_intent: {context.get('previous_intent')}")
            self.logger.info(f"   - previous_filters: {context.get('previous_filters')}")
            self.logger.info(f"   - next_url: {context.get('next_url') and 'yes' or 'no'}")
            
            # Build function response
            function_response = {
                "role": "tool",
                "tool_call_id": message.tool_calls[0].id,
                "name": "get_query_context",
                "content": json.dumps(context)
            }
            
            # Add assistant message and function response to conversation
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            messages.append(function_response)
            
            # Add intent-based instruction to use the context
            next_url_instruction = ""
            if context.get("next_url"):
                next_url_instruction = "\n- If the user is asking for the next page of results, set use_next_page=true and next_page_url to the value from context (next_url)."

            # v0.3.29: Add user context reminder for pronoun resolution
            user_context_reminder = ""
            if user_context:
                user_context_reminder = f"""
- IMPORTANT: If the query contains "me", "my", "mine", "assigned to me", resolve these pronouns using:
  - user_id={user_context.get('user_id')} for technician/created_by/assigned_to filters
  - company_id={user_context.get('company_id')} for company filters
  - NEVER leave "me"/"my" as string values - always use the actual IDs"""

            messages.append({
                "role": "user",
                "content": f"""Parse the query using the context provided.

Use previous_resource; merge previous_filters with any new conditions from the current query. Set merge_with_previous=true. Infer question_type from intent (count vs list vs details).
- If the user specifies how many results to return (a number or phrases like first N, top N), set limit to that number; the system will map it to the API's parameter from the schema.{next_url_instruction}{user_context_reminder}

Return the complete parsed JSON with all filters merged and limit set when the user asks for a specific number of results."""
            })
            
            self.logger.info("🔄 Calling LLM again with context to get final parse...")
            
            # Second call: Get final parsing with context
            final_response = self.openai_client.parse_json_response(
                messages=messages,
                temperature=DETERMINISTIC_TEMP
            )
            
            self.logger.info(f"✅ Final parsed output:")
            self.logger.info(f"   - resource: {final_response.get('resource')}")
            self.logger.info(f"   - filters: {final_response.get('filters')}")
            self.logger.info(f"   - limit: {final_response.get('limit')}")
            self.logger.info(f"   - merge_with_previous: {final_response.get('merge_with_previous')}")
            self.logger.info(f"   - question_type: {final_response.get('question_type')}")
            
            return final_response
        else:
            # LLM didn't call function - might be a standalone query
            self.logger.info("ℹ️  LLM did not call get_query_context() - parsing as standalone query")
            if message.content:
                try:
                    return json.loads(message.content)
                except json.JSONDecodeError:
                    # Fallback: Make another call with JSON format
                    return self.openai_client.parse_json_response(
                        messages=messages,
                        temperature=DETERMINISTIC_TEMP
                    )
            else:
                raise ValueError("No content in LLM response")

    def _parse_with_forced_context(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        conversation_history: list,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse a follow-up query with previous resource/filters injected (no tool-call hop)."""
        context = self._extract_context_from_history(conversation_history)

        self.logger.info(
            "Forced follow-up context: resource=%s filters=%s",
            context.get("previous_resource"),
            context.get("previous_filters"),
        )

        user_context_reminder = ""
        if user_context:
            user_context_reminder = f"""
- Resolve "me"/"my" using user_id={user_context.get('user_id')} and company_id={user_context.get('company_id')}"""

        messages.append({
            "role": "user",
            "content": f"""This is a FOLLOW-UP query that continues the previous turn.

PREVIOUS CONTEXT:
{json.dumps(context, indent=2)}

RULES:
- Keep resource = previous_resource unless the user clearly changes topic
- Merge previous_filters with any new conditions from the current query
- Set merge_with_previous=true
- If the user asks about a field on those items (company, customer, technician, status, etc.), keep the SAME resource and fetch details — do NOT switch to listing all companies/users
- For detail questions after a count, set question_type="details" and display_mode="detailed"{user_context_reminder}

Return the complete parsed JSON.""",
        })

        return self.openai_client.parse_json_response(
            messages=messages,
            temperature=DETERMINISTIC_TEMP,
        )

    # ========================================================================
    # SEMANTIC PHRASE PREPROCESSING (v0.3.44 - Enhanced with fuzzy matching)
    # ========================================================================

    def _preprocess_semantic_phrases(self, query: str, schema: Dict[str, Any]) -> str:
        """
        Pre-process query to transform semantic phrases into explicit filter syntax.

        v0.3.44 enhancements:
        - Fuzzy matching for semantic phrases (word overlap matching)
        - Always resolve resource synonyms (not just when semantic phrase found)
        - Better handling of phrase variations ("low in stock" → "low stock")

        Examples:
            "show me items low in stock" → "show me consumables with stock_level low"
            "show low stock consumables" → "show consumables with stock_level low"
            "list items" → "list consumables"
            "list flash reports" → "list flash-reports"

        Args:
            query: Original user query
            schema: Schema with resource_hints

        Returns:
            Transformed query or original if no transformations needed
        """
        resource_hints = schema.get('resource_hints', {})
        if not resource_hints:
            return query

        query_lower = query.lower()
        transformed = query

        # STEP 1: Always resolve resource synonyms first (v0.3.44 enhancement)
        # This handles "items" → "consumables", "flash reports" → "flash-reports", etc.
        transformed = self._resolve_resource_synonyms(transformed, resource_hints)

        # STEP 2: Build semantic phrase mappings with fuzzy matching support
        semantic_mappings = []

        for resource_name, hints in resource_hints.items():
            if not isinstance(hints, dict):
                continue

            resource_synonyms = hints.get('__resource_synonyms__', [])

            for field_name, field_hints in hints.items():
                if field_name.startswith('__') or not isinstance(field_hints, dict):
                    continue

                synonyms = field_hints.get('synonyms', {})
                if not isinstance(synonyms, dict):
                    continue

                # v0.3.46: Find canonical values - shortest synonym key for each mapped value
                # This lets us use "low" instead of "current_quantity__lt=10" in preprocessing
                canonical_values = {}
                for syn_key, syn_val in synonyms.items():
                    syn_val_str = str(syn_val)
                    if syn_val_str not in canonical_values or len(syn_key) < len(canonical_values[syn_val_str]):
                        canonical_values[syn_val_str] = str(syn_key).lower()

                for phrase, mapped_value in synonyms.items():
                    phrase_lower = str(phrase).lower()
                    phrase_words = set(phrase_lower.split())

                    # Process multi-word phrases and significant single-word phrases
                    if ' ' in phrase_lower or len(phrase_lower) > 5:
                        # v0.3.46: Use simple value for preprocessing, not full API translation
                        # "low stock" → filter_value="low" (api_matcher will translate to current_quantity__lt=10)
                        mapped_str = str(mapped_value)
                        if '=' in mapped_str or '__' in mapped_str:
                            # Use the canonical (shortest) key that maps to this value
                            # e.g., "running low" → "low" (both map to current_quantity__lt=10)
                            simple_value = canonical_values.get(mapped_str, phrase_lower.split()[0])
                        else:
                            simple_value = mapped_str

                        semantic_mappings.append({
                            'phrase': phrase_lower,
                            'phrase_words': phrase_words,
                            'resource': resource_name,
                            'resource_synonyms': resource_synonyms,
                            'field': field_name,
                            'mapped_value': mapped_str,
                            'filter_value': simple_value,
                        })

        # Sort by phrase length (longest first) to avoid partial replacements
        semantic_mappings.sort(key=lambda x: len(x['phrase']), reverse=True)

        # STEP 3: Apply transformations with fuzzy matching (v0.3.44)
        transformed_lower = transformed.lower()
        query_words = set(transformed_lower.split())
        applied_fields = set()  # Track which fields have been transformed

        for mapping in semantic_mappings:
            phrase = mapping['phrase']
            phrase_words = mapping['phrase_words']
            field = mapping['field']

            # Skip if we've already applied a transformation for this field
            if field in applied_fields:
                continue

            # Check for exact phrase match first
            if phrase in transformed_lower:
                transformed = self._apply_semantic_replacement(
                    transformed, phrase, mapping, exact_match=True
                )
                transformed_lower = transformed.lower()
                query_words = set(transformed_lower.split())  # Update query_words!
                applied_fields.add(field)
                self.logger.info(
                    f"v0.3.44 exact match: '{phrase}' → 'with {field} {mapping['filter_value']}'"
                )
                continue

            # v0.3.44: Fuzzy matching - check if query contains the key semantic words
            # "low in stock" should match if "low" + "stock" are both in query
            if len(phrase_words) >= 2:
                # Calculate word overlap
                overlap = phrase_words & query_words
                overlap_ratio = len(overlap) / len(phrase_words)

                # If 80%+ of phrase words are in query, consider it a match
                # AND the words should be contextually close (within 4 words of each other)
                if overlap_ratio >= 0.8:
                    if self._words_are_contextually_close(transformed_lower, list(overlap)):
                        self.logger.info(
                            f"v0.3.44 fuzzy match: '{phrase}' matched via words {overlap} "
                            f"(overlap={overlap_ratio:.0%})"
                        )
                        transformed = self._apply_semantic_replacement(
                            transformed, None, mapping, exact_match=False,
                            matched_words=overlap
                        )
                        transformed_lower = transformed.lower()
                        query_words = set(transformed_lower.split())  # Update query_words!
                        applied_fields.add(field)

        # STEP 4: Generic "low stock" phrase handling (API-agnostic, but schema-aware)
        # This specifically fixes queries like:
        #   - "show me items low in stock"
        #   - "show low stock consumables"
        # for any API schema that exposes a "stock_level" field in resource_hints.
        transformed = self._inject_low_stock_filter(transformed, resource_hints)

        return transformed

    def _resolve_resource_synonyms(self, query: str, resource_hints: Dict[str, Any]) -> str:
        """
        Resolve generic terms to specific resource names using __resource_synonyms__.

        v0.3.44: This now runs unconditionally, not just when semantic phrases are found.

        Examples:
            "list items" → "list consumables"
            "show flash reports" → "show flash-reports"
            "list all companies" → "list all companies" (if companies is a valid resource)

        Args:
            query: User query
            resource_hints: Schema resource hints

        Returns:
            Query with generic terms replaced by resource names
        """
        transformed = query
        query_lower = query.lower()

        # Common generic terms that might refer to specific resources
        generic_terms = {
            'items', 'item', 'things', 'thing', 'stuff', 'products', 'product',
            'materials', 'material', 'supplies', 'supply'
        }

        # Build mapping of synonyms to resource names
        synonym_to_resource = {}
        for resource_name, hints in resource_hints.items():
            if not isinstance(hints, dict):
                continue
            synonyms = hints.get('__resource_synonyms__', [])
            if not isinstance(synonyms, (list, tuple)):
                synonyms = [synonyms]
            for syn in synonyms:
                syn_lower = str(syn).lower()
                synonym_to_resource[syn_lower] = resource_name
                # Also add singular/plural variants
                if syn_lower.endswith('s'):
                    synonym_to_resource[syn_lower[:-1]] = resource_name
                else:
                    synonym_to_resource[syn_lower + 's'] = resource_name

        # Replace generic terms if they match a resource synonym
        for term in generic_terms:
            if term in query_lower and term in synonym_to_resource:
                resource_name = synonym_to_resource[term]
                # Only replace if the term is a standalone word
                transformed = re.sub(
                    rf'\b{term}s?\b',
                    resource_name,
                    transformed,
                    flags=re.IGNORECASE
                )
                self.logger.info(
                    f"v0.3.44 resource synonym: '{term}' → '{resource_name}'"
                )
                break

        # Also handle multi-word resource synonyms like "flash reports" → "flash-reports"
        for syn, resource_name in sorted(synonym_to_resource.items(),
                                         key=lambda x: len(x[0]), reverse=True):
            if ' ' in syn and syn in query_lower:
                transformed = re.sub(
                    rf'\b{re.escape(syn)}\b',
                    resource_name,
                    transformed,
                    flags=re.IGNORECASE
                )
                self.logger.info(
                    f"v0.3.44 multi-word resource synonym: '{syn}' → '{resource_name}'"
                )

        return transformed

    def _inject_low_stock_filter(self, query: str, resource_hints: Dict[str, Any]) -> str:
        """
        Inject an explicit stock_level=low filter for "low stock" style phrases.

        This is API-agnostic but only activates when:
        - The schema's resource_hints define a "stock_level" field for at least one resource
        - The natural language query clearly expresses "low stock" semantics

        Examples it helps with (without relying on LLM to infer the field name):
            "show me items low in stock"    → "... with stock_level low"
            "show low stock consumables"    → "... with stock_level low"
        """
        if not resource_hints:
            return query

        q_lower = query.lower()

        # Only act when the API actually advertises a stock_level field in hints
        has_stock_level = False
        for _res_name, hints in resource_hints.items():
            if isinstance(hints, dict) and "stock_level" in hints:
                has_stock_level = True
                break

        if not has_stock_level:
            return query

        # Detect common "low stock" phrasings
        low_stock_patterns = [
            r"\blow in stock\b",
            r"\blow stock\b",
            r"\bstock is low\b",
            r"\brunning low\b",
        ]

        if not any(re.search(p, q_lower) for p in low_stock_patterns):
            return query

        # If the query already explicitly mentions stock_level, don't duplicate it
        if "stock_level" in q_lower:
            return query

        # If there's already an explicit "with ..." clause, append an additional filter;
        # otherwise, introduce a new "with stock_level low" clause.
        if " with " in q_lower:
            return query + " and stock_level low"
        return query + " with stock_level low"

    def _words_are_contextually_close(self, text: str, words: List[str], max_distance: int = 4) -> bool:
        """
        Check if words appear close to each other in the text.

        This helps avoid false positives where words appear in unrelated parts of a query.

        Args:
            text: The text to check
            words: List of words to find
            max_distance: Maximum number of words between any two target words

        Returns:
            True if words are contextually close
        """
        if len(words) < 2:
            return True

        text_words = text.split()
        word_positions = {}

        for i, w in enumerate(text_words):
            w_clean = re.sub(r'[^\w]', '', w.lower())
            for target in words:
                if target.lower() in w_clean or w_clean in target.lower():
                    if target not in word_positions:
                        word_positions[target] = []
                    word_positions[target].append(i)

        # Check if all words were found
        if len(word_positions) < len(words):
            return False

        # Check if any combination of positions is within max_distance
        positions = [min(pos_list) for pos_list in word_positions.values()]
        if max(positions) - min(positions) <= max_distance + len(words) - 1:
            return True

        return False

    def _apply_semantic_replacement(
        self,
        query: str,
        phrase: Optional[str],
        mapping: Dict[str, Any],
        exact_match: bool,
        matched_words: Optional[set] = None
    ) -> str:
        """
        Apply semantic phrase replacement to query.

        Args:
            query: Current query string
            phrase: Exact phrase to replace (if exact_match=True)
            mapping: Semantic mapping with field and value info
            exact_match: Whether this is an exact or fuzzy match
            matched_words: Words that were matched (for fuzzy matching)

        Returns:
            Query with semantic phrase replaced by explicit filter syntax
        """
        field = mapping['field']
        filter_value = mapping['filter_value']
        resource = mapping['resource']

        if exact_match and phrase:
            # Direct replacement of exact phrase
            transformed = re.sub(
                rf'\b{re.escape(phrase)}\b',
                f'with {field} {filter_value}',
                query,
                flags=re.IGNORECASE
            )
            self.logger.debug(
                f"Semantic phrase (exact): '{phrase}' → 'with {field} {filter_value}' "
                f"for resource '{resource}'"
            )
        else:
            # Fuzzy match: remove matched words and add explicit filter
            transformed = query
            if matched_words:
                # Remove the matched semantic words but keep the structure
                for word in matched_words:
                    # Don't remove if it's part of the resource name
                    if word.lower() not in resource.lower():
                        # Remove word but be careful about common words
                        transformed = re.sub(
                            rf'\b{re.escape(word)}\b\s*',
                            '',
                            transformed,
                            flags=re.IGNORECASE
                        )

                # Clean up double spaces and trailing 'in', 'with', etc.
                transformed = re.sub(r'\s+', ' ', transformed)
                transformed = re.sub(r'\s+(in|with|and|or)\s*$', '', transformed, flags=re.IGNORECASE)
                transformed = transformed.strip()

                # Append the explicit filter
                if f'with {field}' not in transformed.lower():
                    transformed = f"{transformed} with {field} {filter_value}"

            self.logger.debug(
                f"Semantic phrase (fuzzy): matched words {matched_words} → "
                f"'with {field} {filter_value}' for resource '{resource}'"
            )

        return transformed

    # ========================================================================
    # CACHING
    # ========================================================================

    def _get_cache_key(self, query: str, schema: Dict[str, Any]) -> str:
        """Generate cache key for query + schema combination."""
        schema_type = schema.get('type', 'unknown')
        return f"{query.lower().strip()}:{schema_type}"
