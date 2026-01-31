"""
LLM-Based Parser - Uses GPT-4 to parse natural language queries.

Supports:
- API Schema (REST endpoints)
- Database Schema (Tables & columns)
- Knowledge Graph (Entities & relationships for PDFs/docs)

Features:
- Natural language understanding (any phrasing)
- Complex query parsing (multiple conditions, relationships)
- Schema-aware extraction (validates against schema)
- Relationship detection (joins, nested queries)
- Date/time calculation (relative dates like "last week")
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from .types import APIError
from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP


class QueryParser:
    """
    LLM-based parser that uses GPT-4 to understand natural language queries.
    
    Advantages over regex:
    - Understands ANY phrasing (not just keywords)
    - Handles complex conditions and relationships
    - Calculates relative dates ("last week" → actual date)
    - Schema-aware (knows valid resources, fields, values)
    - Multi-language support potential
    """
    
    def __init__(self):
        """Initialize LLM parser with OpenAI client."""
        self.openai_client = get_openai_client()
        self.logger = setup_logger('enable_ai.parser')
        
        # Cache for parsed queries (reduce costs)
        self.cache = {}
        
        self.logger.info("Parser initialized")
    
    def parse_input(self, natural_language_input: str, schema: Optional[Dict[str, Any]] = None, conversation_history: Optional[list] = None) -> Dict[str, Any]:
        """
        Parse natural language input using LLM with schema context and conversation history.
        
        Args:
            natural_language_input: User's natural language query
            schema: Active schema (api, database, or knowledge_graph)
            conversation_history: Previous messages for multi-turn context (list of {role, content})
        
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
            
            >>> parse_input("show urgent service orders created last week", api_schema)
            {
                'intent': 'read',
                'resource': 'service_orders',
                'entities': {
                    'priority': 'urgent',
                    'created_after': '2024-01-13'
                },
                'filters': {
                    'priority': {'operator': 'equals', 'value': 'urgent'},
                    'created_date': {'operator': 'gte', 'value': '2024-01-13'}
                }
            }
        """
        if not natural_language_input or not natural_language_input.strip():
            return APIError("Empty input provided")
        
        if not schema:
            return APIError("Schema is required for LLM parsing")
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(natural_language_input, schema)
            if cache_key in self.cache:
                self.logger.info(f"Using cached parse result for: '{natural_language_input[:50]}...'")
                return self.cache[cache_key]
            
            # Build prompt with schema context
            prompt = self._build_prompt(natural_language_input, schema)
            
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
                # Use function calling to provide structured context
                parsed = self._parse_with_context_function(messages, schema, conversation_history)
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

Your task: Extract structured information from user queries to enable programmatic data access.

⚠️ CRITICAL: Detecting Refinement Queries (v0.3.12 FIX)
**IF THE QUERY CONTAINS ANY OF THESE KEYWORDS, YOU MUST CALL get_query_context():**
- Pronouns: "them", "those", "these", "it", "they"
- Phrases: "of them", "among them", "from them", "in them"
- Continuations: "show me more", "the same ones", "give me all"
- Filters on previous: "the one/ones that", "which are", "where"

**WHEN YOU DETECT A REFINEMENT QUERY:**
1. CALL get_query_context() to get: previous_resource, previous_intent, previous_filters
2. Use previous_resource (don't try to infer a new one)
3. Set merge_with_previous=true
4. ADD new filters to the returned previous_filters (don't replace them)
5. **If the user asks to "list a few", "show a few", "name a few", "give me a few/some"**: set limit to 5 or 10 and question_type to "list" so they see example items, not just a count.

**Examples:**
❌ BAD: "how many of them are named BOROSCOPE?"
   → Detected as: {intent: "read", resource: "inventory", filters: {name: "BOROSCOPE"}}
   → Previous filters LOST!

✓ GOOD: "how many of them are named BOROSCOPE?"
   1. Detect "of them" → CALL get_query_context()
   2. Get: {previous_resource: "equipment", previous_filters: {status: "low_stock"}}
   3. Return: {
       intent: "read",
       resource: "equipment",  ← Use previous_resource!
       filters: {
         status: {operator: "equals", value: "low_stock"},  ← Previous filter!
         name: {operator: "contains", value: "BOROSCOPE"}  ← New filter!
       },
       merge_with_previous: true,
       question_type: "count"  ← Detect "how many"
     }

✓ GOOD: "can you list a few of them?" / "show me a few" (after "which equipment has BOROSCOPE?")
   1. Detect "of them" or "a few" → CALL get_query_context()
   2. Get: {previous_resource: "equipment", previous_filters: {name: {"operator": "contains", "value": "BOROSCOPE"}}}
   3. Return: {
       intent: "read",
       resource: "equipment",
       filters: {name: {operator: "contains", value: "BOROSCOPE"}},
       merge_with_previous: true,
       limit: 5,  ← So user sees example names, not only "Found 25 items"
       question_type: "list"
     }

⚠️ CRITICAL: Detecting Count Questions (v0.3.12 FIX)
**IF THE QUERY ASKS FOR A COUNT, SET question_type="count":**
- Starts with: "how many", "count", "what's the count", "number of"
- Contains: "how much", "quantity", "total number"

**This triggers COUNT formatting instead of LIST formatting!**

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
   - **If this is a follow-up query**, check conversation history for [Context: ... on RESOURCE] and use that resource unless explicitly changed

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
   - Extract from: "top 10", "first 5", "show 20", etc.
   - **"list a few", "show a few", "name a few", "give me a few/some", "can you list a few"** → use limit: 5 or 10 so the user sees example items
   - If not specified, omit (let system use default)

8. **display_mode** (optional) - How to display results (v0.3.7)
   - "summary": Brief summary with examples (default for large results)
   - "full": Complete list of all items
   - "detailed": Detailed view with all fields
   - Detect from: "show me all", "give me the full list", "show everything", "all of them"
   - Also detect: "in detail", "detailed view", "show more info"

9. **merge_with_previous** (optional) - Whether to merge with previous query (v0.3.7)
   - true: This is a refinement of the previous query (add filters, change display)
   - false: This is a completely new query
   - Detect from context: If query contains only filters/display preferences without specifying a new resource

10. **question_type** (required in v0.3.12) - Type of question being asked
   - "count": User asking "how many" / "count" / "what's the total" → Answer with number only
   - "list": User wants to see the items → Show list with details
   - "details": User wants detailed info about specific item(s)
   - DEFAULT: If query starts with "how many" / "count" / "what's the total", ALWAYS set to "count"

RULES:
- Use ONLY field names and resources defined in the provided schema
- Map common synonyms to schema field names
- Calculate dates relative to today's date (provided in prompt)
- Return valid JSON matching the exact format below
- If you cannot determine a field, omit it (don't guess)
- For ambiguous queries, prefer 'read' intent
- **CRITICAL**: For relationship filters, populate BOTH entities and relationships

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
    "question_type": "count|list|details"
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

Example 5 - COUNT question with refinement (CRITICAL for v0.3.12):
Query: "how many of them are named BOROSCOPE?"
Context from get_query_context(): {
    "previous_resource": "equipment",
    "previous_filters": {"status": {"operator": "equals", "value": "low_stock"}}
}
Output: {
    "intent": "read",
    "resource": "equipment",  ← From previous_resource!
    "entities": {
        "status": "low_stock",  ← From previous!
        "name": "BOROSCOPE"
    },
    "filters": {
        "status": {"operator": "equals", "value": "low_stock"},  ← MERGED!
        "name": {"operator": "contains", "value": "BOROSCOPE"}
    },
    "merge_with_previous": true,  ← Mark as refinement!
    "question_type": "count"  ← Detect "how many"!
}

CRITICAL: Return ONLY the JSON object, no explanations or markdown.
"""
    
    def _build_prompt(self, query: str, schema: Dict[str, Any]) -> str:
        """
        Build user prompt with schema context and query.
        
        Args:
            query: Natural language query
            schema: Active schema
        
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
            hints_section += (
                "\n\nWhen the user says e.g. \"pending\" or \"quoted\", use the synonym or value above. "
                "For \"pending\" you may need to use multiple values (e.g. status__name__in or one canonical "
                "\"pending\" value per your API). Prefer the exact strings listed in \"values\" or given by \"synonyms\".\n"
                "\nSome resources may also define \"__resource_synonyms__\". When the user uses those nouns "
                "(for example, \"equipment\", \"equipments\", \"consumables\", \"materials\"), choose the "
                "corresponding resource instead of another with a similar shape.\n"
            )
        
        return f"""Parse this natural language query:
"{query}"

TODAY'S DATE: {today}

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
            valid_resources = schema.get('resources', {}).keys()
        elif schema_type == 'database':
            valid_resources = schema.get('tables', {}).keys()
        elif schema_type == 'knowledge_graph':
            valid_resources = schema.get('entities', {}).keys()
        else:
            valid_resources = []
        
        if parsed.get('resource') not in valid_resources:
            # Try to find closest match
            resource = parsed.get('resource', '').lower()
            for valid_resource in valid_resources:
                if resource in valid_resource.lower() or valid_resource.lower() in resource:
                    parsed['resource'] = valid_resource
                    break
        
        # Ensure required fields exist
        if 'entities' not in parsed:
            parsed['entities'] = {}
        
        if 'filters' not in parsed:
            parsed['filters'] = {}
        
        # Convert entities to filters if filters are empty
        if parsed['entities'] and not parsed['filters']:
            for key, value in parsed['entities'].items():
                parsed['filters'][key] = {
                    'operator': 'equals',
                    'value': value
                }
        
        return parsed
    
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
        context = {
            "previous_resource": None,
            "previous_intent": None,
            "previous_query": None,
            "previous_filters": None
        }
        
        # Look through conversation history in reverse (most recent first)
        for msg in reversed(conversation_history):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Look for [Context: intent operation on resource]
                if '[Context:' in content:
                    import re
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
                            except:
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
                "description": """CRITICAL: You MUST call this function if the query contains ANY of these keywords:
- Pronouns: 'them', 'those', 'these', 'it', 'they', 'that', 'which'
- Phrases: 'of them', 'among them', 'from them', 'in them', 'the ones', 'the same'
- Examples requiring this call:
  * "how many of them are named X?" ← MUST CALL
  * "show me those in location Y" ← MUST CALL
  * "which ones are available?" ← MUST CALL
  * "the ones that match Z" ← MUST CALL

Returns context from previous query:
- previous_resource: The resource type user was querying (e.g., "equipment", "users")
- previous_intent: The operation (e.g., "read", "create")
- previous_query: The exact previous query text
- previous_filters: CRITICAL - The filters from the previous query (e.g., {"status": {"operator": "equals", "value": "low_stock"}})

You MUST use previous_resource and MERGE previous_filters with any new filters!""",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    def _parse_with_context_function(
        self, 
        messages: List[Dict[str, str]], 
        schema: Dict[str, Any],
        conversation_history: list
    ) -> Dict[str, Any]:
        """
        Parse query using OpenAI function calling to provide structured context (v0.3.12 enhanced).
        
        Args:
            messages: Messages for the LLM
            schema: Schema for validation
            conversation_history: Full conversation history with context markers
            
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
            
            # Add explicit instruction to use the context
            messages.append({
                "role": "user",
                "content": f"""Now parse the query using the context provided.

CRITICAL INSTRUCTIONS:
1. Use previous_resource as the resource (don't infer a new one)
2. MERGE previous_filters with any new filters from the current query
3. Set merge_with_previous=true
4. If query asks "how many", set question_type="count"

Return the complete parsed JSON now with ALL filters merged."""
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
    
    # ========================================================================
    # CACHING
    # ========================================================================
    
    def _get_cache_key(self, query: str, schema: Dict[str, Any]) -> str:
        """Generate cache key for query + schema combination."""
        schema_type = schema.get('type', 'unknown')
        return f"{query.lower().strip()}:{schema_type}"
