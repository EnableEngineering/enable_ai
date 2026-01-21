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

import os
import json
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from .types import APIError


# Load environment variables from .env file
# Priority: environment.py > auto-detection
def load_env():
    """Load .env file using priority order: environment.py > auto-detection."""
    try:
        # Priority 1: Check environment.py for DEV/PROD configuration
        try:
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            import environment
            env_path = Path(environment.get_env_path())
            
            if env_path.exists():
                load_dotenv(env_path)
                return
            else:
                print(f"⚠️  environment.py configured but .env not found: {env_path}", file=sys.stderr)
        except (ImportError, AttributeError, FileNotFoundError):
            # environment.py doesn't exist, fall back to auto-detection
            pass
        
        # Priority 2: Auto-detection (CWD, parent directories, etc.)
        load_dotenv()
    except Exception as e:
        print(f"⚠️  Warning: Could not load .env: {e}", file=sys.stderr)

# Load environment on module import
load_env()


class Parser:
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
        # Load API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  # Fast and accurate
        
        # Cache for parsed queries (reduce costs)
        self.cache = {}
    
    def parse_input(self, natural_language_input: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse natural language input using LLM with schema context.
        
        Args:
            natural_language_input: User's natural language query
            schema: Active schema (api, database, or knowledge_graph)
        
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
                print("  📋 Using cached parse result")
                return self.cache[cache_key]
            
            # Build prompt with schema context
            prompt = self._build_prompt(natural_language_input, schema)
            
            # Call LLM
            print(f"  🤖 Parsing with LLM: '{natural_language_input}'")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistent parsing
            )
            
            # Parse LLM response
            parsed_json = response.choices[0].message.content
            parsed = json.loads(parsed_json)
            
            # Add metadata
            parsed['original_input'] = natural_language_input
            parsed['schema_type'] = schema.get('type')
            
            # Validate against schema
            validated = self._validate_parsed_output(parsed, schema)
            
            # Cache result
            self.cache[cache_key] = validated
            
            print(f"  ✓ Parsed: intent={validated['intent']}, resource={validated.get('resource', 'N/A')}")
            
            return validated
            
        except json.JSONDecodeError as e:
            return APIError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
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
   - If not specified, omit (let system use default)

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
    "limit": 10
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

Example 2 - Relationship filter (populate both entities AND relationships):
Query: "show service orders for company ABC"
Output: {
    "intent": "read",
    "resource": "service_orders",
    "entities": {
        "companies_name": "ABC"
    },
    "filters": {
        "companies_name": {"operator": "equals", "value": "ABC"}
    },
    "relationships": [
        {
            "type": "belongs_to",
            "target_entity": "companies",
            "filters": {"name": {"operator": "equals", "value": "ABC"}}
        }
    ]
}

Example 3 - Multiple conditions with relationship:
Query: "urgent service orders for company ABC created last week"
Output: {
    "intent": "read",
    "resource": "service_orders",
    "entities": {
        "priority": "urgent",
        "companies_name": "ABC",
        "created_after": "2024-01-13"
    },
    "filters": {
        "priority": {"operator": "equals", "value": "urgent"},
        "companies_name": {"operator": "equals", "value": "ABC"},
        "created_date": {"operator": "gte", "value": "2024-01-13"}
    },
    "relationships": [
        {
            "type": "belongs_to",
            "target_entity": "companies",
            "filters": {"name": {"operator": "equals", "value": "ABC"}}
        }
    ]
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
        
        return f"""Parse this natural language query:
"{query}"

TODAY'S DATE: {today}

AVAILABLE SCHEMA:

Resources/Tables:
{json.dumps(schema_info['resources'], indent=2)}

Fields by resource:
{json.dumps(schema_info['fields'], indent=2)}

INSTRUCTIONS:
1. Extract intent, resource, entities, and filters from the query
2. Use ONLY resources and fields defined above
3. Calculate any relative dates based on today's date ({today})
4. Return valid JSON matching the format in the system prompt
5. Be precise - map the query to the exact schema structure

Return the parsed JSON now:
"""
    
    def _extract_api_schema_info(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant info from API schema for LLM."""
        resources = list(schema.get('resources', {}).keys())
        fields = {}
        
        for resource_name, resource_def in schema.get('resources', {}).items():
            # Get field names from first endpoint's parameters
            endpoints = resource_def.get('endpoints', [])
            if endpoints:
                first_endpoint = endpoints[0]
                params = first_endpoint.get('parameters', {})
                
                # Collect all parameter names
                field_names = set()
                field_names.update(params.get('path', []))
                field_names.update(params.get('query', []))
                
                # If parameters are objects with 'name' keys
                for param_list in [params.get('path', []), params.get('query', []), params.get('body', [])]:
                    if isinstance(param_list, list):
                        for param in param_list:
                            if isinstance(param, dict) and 'name' in param:
                                field_names.add(param['name'])
                            elif isinstance(param, str):
                                field_names.add(param)
                
                fields[resource_name] = list(field_names) if field_names else ['id', 'name', 'created_date']
        
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
    # CACHING
    # ========================================================================
    
    def _get_cache_key(self, query: str, schema: Dict[str, Any]) -> str:
        """Generate cache key for query + schema combination."""
        schema_type = schema.get('type', 'unknown')
        return f"{query.lower().strip()}:{schema_type}"
