"""
NLP API Caller - AI-Powered API Query Processor

Main interface for processing natural language queries against various data sources.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import sys
from .parser import Parser
from .api_matcher import APIMatcher
from .database_matcher import DatabaseMatcher
from .knowledge_graph_matcher import KnowledgeGraphMatcher
from .api_client import APIClient
from .config_loader import get_config
from .types import MissingInformation, APIResponse, APIError


class NLPProcessor:
    """
    Main processor for natural language queries against various data sources.
    
    Usage:
        # Auto-load schemas from config.json
        processor = NLPProcessor()
        
        # Override specific schemas
        processor = NLPProcessor(schemas={'api': 'custom_api.json'})
        
        # Process query (runtime schema optional)
        result = processor.process("get user with id 5", access_token="...")
    """
    
    def __init__(self, config_path: Optional[str] = None, schemas: Optional[Dict[str, Union[str, dict]]] = None):
        """
        Initialize NLP Processor.
        
        Args:
            config_path: Path to config.json (optional, defaults to 'config.json')
            schemas: Optional schema override dict (overrides config)
                     Format: {'api': path_or_dict, 'database': path_or_dict, ...}
        
        Examples:
            # Auto-load from config
            processor = NLPProcessor()
            
            # Override specific schemas
            processor = NLPProcessor(schemas={'api': 'custom_api.json'})
            
            # Completely custom schemas
            processor = NLPProcessor(schemas={
                'api': api_schema_dict,
                'database': 'schemas/db.json'
            })
        """
        # Load configuration
        if config_path:
            # Load config from custom path
            self.config = self._load_config_from_path(config_path)
            self.config_dir = Path(config_path).parent
        else:
            # Use default config loader
            self.config = get_config()
            self.config_dir = None
        
        # Load schemas (config first, then override)
        self.schemas = self._load_schemas(schemas)
        
        # Initialize components
        # Note: Matchers require schemas, so they're initialized on-demand in _create_plan
        self.parser = Parser()
        self.api_matcher = None  # Initialized when needed with api_spec
        self.database_matcher = None  # Initialized when needed with database schema
        self.kg_matcher = None  # Initialized when needed with knowledge graph schema
        self.client = None  # Initialized when needed with base URL
        
        # Print initialization summary
        self._print_init_summary()
    
    def _load_config_from_path(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a specific file path.
        
        Args:
            config_path: Path to config.json file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: Config file not found at {config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing config file: {e}")
            return {}
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            return {}
    
    def _load_schemas(self, override_schemas: Optional[Dict[str, Union[str, dict]]] = None) -> Dict[str, dict]:
        """
        Load schemas from config, ONLY for enabled data sources.
        
        Logic:
        1. Check which data sources are enabled in config
        2. Determine required schema types for enabled sources
        3. Load only those schemas from config
        4. Apply overrides if provided
        
        Args:
            override_schemas: Optional schemas to override config
            
        Returns:
            Dict of loaded schemas by type (only for enabled sources)
        """
        loaded = {}
        
        # Get enabled data sources
        enabled_sources = self._get_enabled_data_sources_list()
        
        if not enabled_sources:
            print("⚠️  Warning: No data sources enabled in config")
            # Still allow override schemas
            if override_schemas:
                for schema_type, schema_input in override_schemas.items():
                    try:
                        if isinstance(schema_input, dict):
                            self._validate_schema(schema_input, schema_type)
                            loaded[schema_type] = schema_input
                        elif isinstance(schema_input, str):
                            loaded_schema = self._load_schema_file(schema_input, schema_type)
                            if loaded_schema:
                                loaded[schema_type] = loaded_schema
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load {schema_type} schema: {e}")
            return loaded
        
        # Map data sources to required schema types
        required_schemas = self._get_required_schema_types(enabled_sources)
        
        # Load only required schemas from config
        config_schemas = self.config.get('schemas', {})
        
        for schema_type in required_schemas:
            schema_path = config_schemas.get(schema_type)
            if schema_path:
                try:
                    loaded_schema = self._load_schema_file(schema_path, schema_type)
                    if loaded_schema:
                        loaded[schema_type] = loaded_schema
                        print(f"✓ Loaded {schema_type} schema (required by enabled data sources)", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load {schema_type} schema from config: {e}")
            else:
                print(f"⚠️  Warning: {schema_type} schema required but not configured")
        
        # Override with provided schemas (regardless of enabled sources)
        if override_schemas:
            for schema_type, schema_input in override_schemas.items():
                try:
                    if isinstance(schema_input, dict):
                        # Already a dict, validate and use
                        self._validate_schema(schema_input, schema_type)
                        loaded[schema_type] = schema_input
                        print(f"✓ Loaded {schema_type} schema (override)")
                    elif isinstance(schema_input, str):
                        # File path, load it
                        loaded_schema = self._load_schema_file(schema_input, schema_type)
                        if loaded_schema:
                            loaded[schema_type] = loaded_schema
                            print(f"✓ Loaded {schema_type} schema (override)")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load {schema_type} schema: {e}")
        
        return loaded
    
    def _get_enabled_data_sources_list(self) -> list:
        """Get list of all enabled data sources from config."""
        data_sources = self.config.get('data_sources', {})
        enabled = []
        
        for source_name, source_config in data_sources.items():
            if isinstance(source_config, dict) and source_config.get('enabled'):
                enabled.append(source_name)
        
        return enabled
    
    def _get_required_schema_types(self, enabled_sources: list) -> set:
        """
        Determine which schema types are needed based on enabled data sources.
        
        Mapping:
        - api → api schema
        - database → database schema
        - json_files → knowledge_graph schema
        - pdf_documents → knowledge_graph schema
        - vector_search → knowledge_graph schema
        - cache → (no schema needed)
        - search_databases → database schema
        
        Args:
            enabled_sources: List of enabled data source names
            
        Returns:
            Set of required schema types
        """
        source_to_schema = {
            'api': 'api',
            'database': 'database',
            'search_databases': 'database',
            'json_files': 'knowledge_graph',
            'pdf_documents': 'knowledge_graph',
            'vector_search': 'knowledge_graph'
        }
        
        required = set()
        for source in enabled_sources:
            schema_type = source_to_schema.get(source)
            if schema_type:
                required.add(schema_type)
        
        return required
    
    def _load_schema_file(self, schema_path: str, schema_type: str) -> Optional[dict]:
        """
        Load schema from file path.
        
        Args:
            schema_path: Path to schema JSON file (absolute or relative to config dir)
            schema_type: Expected schema type
            
        Returns:
            Loaded schema dict or None
        """
        path = Path(schema_path)
        
        # If path is relative and we have a config directory, resolve relative to it
        if not path.is_absolute() and self.config_dir:
            path = self.config_dir / path
        
        if not path.exists():
            print(f"⚠️  Warning: Schema file not found: {schema_path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                self._validate_schema(schema, schema_type)
                return schema
        except json.JSONDecodeError as e:
            print(f"⚠️  Error: Invalid JSON in {schema_path}: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Error loading {schema_path}: {e}")
            return None
    
    def _validate_schema(self, schema: dict, expected_type: str) -> None:
        """
        Validate schema structure.
        
        Args:
            schema: Schema dict to validate
            expected_type: Expected schema type
            
        Raises:
            ValueError: If schema is invalid
        """
        if 'type' not in schema:
            raise ValueError(f"Schema missing 'type' field. Expected '{expected_type}'")
        
        if schema['type'] != expected_type:
            raise ValueError(
                f"Schema type mismatch. Expected '{expected_type}', got '{schema['type']}'"
            )
        
        # Type-specific validation
        if expected_type == 'api':
            if 'resources' not in schema:
                raise ValueError("API schema missing 'resources' field")
        elif expected_type == 'database':
            if 'tables' not in schema:
                raise ValueError("Database schema missing 'tables' field")
        elif expected_type == 'knowledge_graph':
            if 'entities' not in schema or 'relationships' not in schema:
                raise ValueError("Knowledge graph missing 'entities' or 'relationships' field")
    
    def _print_init_summary(self) -> None:
        """Print initialization summary."""
        print("✓ NLP Processor initialized", file=sys.stderr)
        
        # Print enabled data sources
        enabled_sources = self._get_enabled_data_sources_list()
        if enabled_sources:
            print(f"  - Enabled data sources: {', '.join(enabled_sources)}", file=sys.stderr)
        else:
            print("  - ⚠️  No data sources enabled", file=sys.stderr)
        
        # Print loaded schemas
        if self.schemas:
            schema_names = ', '.join(self.schemas.keys())
            print(f"  - Schemas loaded: {schema_names}", file=sys.stderr)
        else:
            if enabled_sources:
                print("  - ⚠️  No schemas loaded (check config)", file=sys.stderr)
            else:
                print("  - No schemas loaded (runtime schema required)", file=sys.stderr)
        
        # Print client support
        client_support = list(self.config.get('client_support', {}).keys())
        if client_support:
            print(f"  - Client support: {', '.join(client_support)}", file=sys.stderr)
    
    def process(self, query: str, access_token: Optional[str] = None, context: Optional[Any] = None, schema: Optional[dict] = None) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Pipeline:
            1. Load schema (determine active schema)
            2. Parse query (understand intent with schema)
            3. Create plan (match to endpoint/entity)
            4. Execute plan (query data source)
            5. Summarize result (format response)
            6. Return result
        
        Args:
            query: Natural language query from user
            access_token: Optional JWT token for authentication
            context: Optional conversation context
            schema: Optional runtime schema (overrides init schemas)
            
        Returns:
            {
                "success": bool,
                "data": dict,
                "summary": str,
                "query": str,
                "endpoint": str,
                "schema_type": str
            }
        
        Example:
            # Using init-time schema
            result = processor.process("get user with id 5")
            
            # Using runtime schema (overrides init schema)
            custom_schema = {"type": "api", "resources": {...}}
            result = processor.process("get user 5", schema=custom_schema)
        """
        try:
            # Step 1: Determine active schema
            active_schema = self._get_active_schema(schema)
            
            if not active_schema:
                return {
                    "success": False,
                    "error": "No schema provided. Pass schema at init or runtime.",
                    "query": query
                }
            
            # Step 2: Understand Query (with schema)
            parsed = self._understand_query(query, active_schema, context)
            
            if isinstance(parsed, dict) and parsed.get('type') == 'missing_info':
                # Need more information
                return {
                    "success": False,
                    "needs_info": True,
                    "message": parsed.get('message'),
                    "missing_fields": parsed.get('missing_fields'),
                    "context": parsed.get('context')
                }
            
            # Step 3: Create Plan (with schema)
            plan = self._create_plan(parsed, active_schema)
            
            if not plan:
                return {
                    "success": False,
                    "error": "Could not create execution plan",
                    "query": query
                }
            
            # Step 4: Execute Plan (with schema)
            result = self._execute_plan(plan, active_schema, access_token, parsed)
            
            # Step 5: Summarize Result
            summary = self._summarize_result(result, query)
            
            # Step 6: Return Result
            # Check if execution had an error
            has_error = result.get('error') is not None
            
            return {
                "success": not has_error,
                "data": result.get('data'),
                "summary": summary,
                "query": query,
                "endpoint": plan.get('endpoint'),
                "method": plan.get('method'),
                "schema_type": active_schema['type'],
                "error": result.get('error') if has_error else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _get_active_schema(self, runtime_schema: Optional[dict] = None) -> Optional[dict]:
        """
        Determine which schema to use.
        
        Priority:
        1. Runtime schema (passed to process())
        2. Schema matching enabled data source
        3. First available init-time schema
        4. None
        
        Args:
            runtime_schema: Optional runtime schema override
            
        Returns:
            Active schema dict or None
        """
        # Priority 1: Runtime override
        if runtime_schema:
            return runtime_schema
        
        # Priority 2: Match enabled data source
        enabled_source = self._get_enabled_data_source()
        
        # Map data source to schema type
        source_to_schema = {
            'api': 'api',
            'database': 'database',
            'json_files': 'knowledge_graph',
            'pdf_documents': 'knowledge_graph',
            'vector_search': 'knowledge_graph'
        }
        
        schema_type = source_to_schema.get(enabled_source)
        if schema_type and schema_type in self.schemas:
            return self.schemas[schema_type]
        
        # Priority 3: First available schema
        if self.schemas:
            return next(iter(self.schemas.values()))
        
        # Priority 4: None
        return None
    
    def _understand_query(self, query: str, schema: dict, context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Step 2: Parse and understand user query with schema.
        
        Args:
            query: User's natural language query
            schema: Active schema for entity extraction
            context: Optional conversation context
        """
        parsed = self.parser.parse_input(query, schema)
        
        if not parsed:
            raise ValueError("Could not parse query")
        
        return parsed
    
    def _create_plan(self, parsed: Dict[str, Any], schema: dict) -> Optional[Dict[str, Any]]:
        """
        Step 3: Create execution plan from parsed query and schema.
        
        Routes to appropriate plan creator based on schema type:
        - api_schema or api → API Matcher (existing)
        - database_schema or database → Database Matcher (new)
        - knowledge_graph → Knowledge Graph Matcher (new)
        
        Args:
            parsed: Parsed query with intent and entities
            schema: Active schema for matching
        
        Returns:
            Execution plan dict or None
        """
        schema_type = schema.get('type')
        
        if schema_type in ['api_schema', 'api']:
            # Use existing API matcher
            return self._create_api_plan(parsed, schema)
        
        elif schema_type in ['database_schema', 'database']:
            # Use new database matcher
            return self._create_database_plan(parsed, schema)
        
        elif schema_type == 'knowledge_graph':
            # Use new knowledge graph matcher (RAG)
            return self._create_knowledge_graph_plan(parsed, schema)
        
        else:
            return None
    
    # ========================================================================
    # PLAN CREATORS (Schema-Type Specific)
    # ========================================================================
    
    def _create_api_plan(self, parsed: Dict[str, Any], schema: dict) -> Optional[Dict[str, Any]]:
        """
        Create API execution plan.
        
        Supports: CREATE, READ, UPDATE, DELETE
        """
        # Initialize matcher if needed
        if self.api_matcher is None:
            self.api_matcher = APIMatcher()
        
        result = self.api_matcher.match_api(parsed, schema)
        
        # Handle different result types from APIMatcher
        from .types import APIRequest, MissingInformation, APIError
        
        if isinstance(result, APIError):
            return {
                "type": "error",
                "error": result.message
            }
        
        if isinstance(result, MissingInformation):
            return {
                "type": "missing_info",
                "message": result.message,
                "missing_fields": result.missing_fields
            }
        
        if isinstance(result, APIRequest):
            # Extract information from APIRequest object
            return {
                "type": "api",
                "endpoint": result.endpoint,
                "method": result.method,
                "params": result.params,
                "authentication_required": result.authentication_required,
                "endpoint_name": "unknown",  # APIRequest doesn't have this
                "module": "unknown",  # APIRequest doesn't have this
                "schema_type": "api_schema"
            }
        
        return None
    
    def _create_database_plan(self, parsed: Dict[str, Any], schema: dict) -> Optional[Dict[str, Any]]:
        """
        Create database query plan.
        
        Supports: CREATE, READ, UPDATE, DELETE
        
        Converts parsed natural language to SQL/query operations.
        """
        # Initialize matcher if needed
        if self.database_matcher is None:
            self.database_matcher = DatabaseMatcher()
        
        query_plan = self.database_matcher.build_query(parsed, schema)
        
        if not query_plan:
            return None
        
        return {
            "type": "database",
            "table": query_plan.get('table'),
            "operation": query_plan.get('operation'),
            "filters": query_plan.get('filters', {}),
            "sql_query": query_plan.get('sql_query'),
            "params": query_plan.get('params', []),
            "data": query_plan.get('data', {}),
            "schema_type": "database_schema"
        }
    
    def _create_knowledge_graph_plan(self, parsed: Dict[str, Any], schema: dict) -> Optional[Dict[str, Any]]:
        """
        Create knowledge graph RAG plan.
        
        Supports: READ ONLY (search using vector embeddings + RAG)
        
        For unstructured data (PDFs, documents).
        """
        if parsed.get('intent') not in ['read', 'search']:
            return {
                "type": "error",
                "error": "Knowledge graph only supports READ/SEARCH operations"
            }
        
        # Initialize matcher if needed
        if self.kg_matcher is None:
            self.kg_matcher = KnowledgeGraphMatcher()
        
        rag_plan = self.kg_matcher.build_rag_query(parsed, schema)
        
        if not rag_plan or rag_plan.get('error'):
            return rag_plan
        
        return {
            "type": "rag",
            "entity_type": rag_plan.get('entity_type'),
            "query_text": parsed.get('original_input'),
            "filters": rag_plan.get('filters', {}),
            "vector_search": True,
            "relationships": rag_plan.get('relationships', []),
            "related_entities": rag_plan.get('related_entities', {}),
            "vector_search_params": rag_plan.get('vector_search_params', {}),
            "search_mode": rag_plan.get('search_mode', 'entity'),
            "schema_type": "knowledge_graph"
        }
    
    def _execute_plan(self, plan: Dict[str, Any], schema: dict, access_token: Optional[str] = None, parsed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Step 4: Execute the plan against appropriate data source.
        
        Routes based on plan type (not schema type).
        
        Args:
            plan: Execution plan with 'type' field
            schema: Active schema
            access_token: Optional authentication token
            parsed: Optional parsed query data (for context)
        
        Returns:
            Execution result dict
        """
        plan_type = plan.get('type')
        
        if plan_type == 'api':
            return self._execute_api(plan, schema, access_token, parsed)
        
        elif plan_type == 'database':
            return self._execute_database(plan, schema, access_token)
        
        elif plan_type == 'rag':
            return self._execute_rag(plan, schema)
        
        elif plan_type == 'error':
            return {
                "data": None,
                "error": plan.get('error', 'Unknown error')
            }
        
        else:
            return {
                "data": None,
                "error": f"Unknown plan type: {plan_type}"
            }
    
    def _get_enabled_data_source(self) -> Optional[str]:
        """Get the first enabled data source from config."""
        data_sources = self.config.get('data_sources', {})
        
        # Priority order: API, Database, JSON, PDF, Vector Search
        for source in ['api', 'database', 'json_files', 'pdf_documents', 'vector_search']:
            if data_sources.get(source, {}).get('enabled'):
                return source
        
        return None
    
    # ========================================================================
    # DATA SOURCE HANDLERS (Updated with schema parameter)
    # ========================================================================
    
    def _execute_api(self, plan: Dict[str, Any], schema: dict, access_token: Optional[str] = None, parsed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute plan against REST API.
        
        FULLY IMPLEMENTED ✓
        
        Flow:
        1. Check if authentication is required from schema/plan
        2. If required and no token provided, authenticate first
        3. Use token to make actual API call
        4. Handle navigation/index pages by following links
        
        Args:
            plan: Execution plan
            schema: API schema
            access_token: Optional JWT token (if not provided, will authenticate)
            parsed: Optional parsed query data (for determining resource)
        """
        # Get API config (with fallback to schema)
        api_config = self.config.get('data_sources', {}).get('api', {})
        base_url = api_config.get('base_url') or schema.get('base_url')
        
        if not base_url:
            return {
                "data": None,
                "error": "API base_url not configured. Set 'base_url' in config.json under data_sources.api or in the API schema."
            }
        
        # Check if authentication is required
        auth_required = plan.get('authentication_required', True)
        
        if auth_required:
            # Get or authenticate token
            if not access_token:
                # Detect authentication type from schema and config
                auth_type = self._detect_auth_type(schema)
                
                if auth_type:
                    # Authenticate and get token
                    auth_result = self._authenticate(base_url, auth_type)
                    
                    if auth_result.get('error'):
                        return {
                            "data": None,
                            "error": f"Authentication failed: {auth_result['error']}"
                        }
                    
                    access_token = auth_result.get('token')
                    
                    if not access_token:
                        return {
                            "data": None,
                            "error": "Authentication succeeded but no token received"
                        }
        
        # Initialize client with access token (if available)
        if self.client is None or access_token:
            self.client = APIClient(base_url, access_token)
        
        # Build APIRequest object
        from .types import APIRequest
        api_request = APIRequest(
            endpoint=plan['endpoint'],
            method=plan['method'],
            params=plan.get('params', {}),
            authentication_required=auth_required
        )
        
        # Make API call using call_api method
        response = self.client.call_api(api_request)
        
        # Check if response is a navigation/index page
        if isinstance(response, APIResponse) and isinstance(response.data, dict):
            # Check if this looks like a navigation page (has URLs as values)
            is_navigation = all(
                isinstance(v, str) and (v.startswith('http://') or v.startswith('https://'))
                for v in response.data.values()
                if v is not None
            )
            
            if is_navigation and len(response.data) > 0:
                # This is a navigation page, try to follow the appropriate link
                # Use the parsed resource name if available
                resource_name = parsed.get('resource') if parsed else None
                
                # Try exact match first
                if resource_name and resource_name in response.data:
                    navigation_url = response.data[resource_name]
                    print(f"  → Following navigation link for '{resource_name}': {navigation_url}", file=sys.stderr)
                else:
                    # Try to find a matching key (handle variations like "service_orders" vs "service-orders")
                    resource_variants = []
                    if resource_name:
                        resource_variants = [
                            resource_name,
                            resource_name.replace('_', '-'),
                            resource_name.replace('-', '_'),
                            resource_name.replace(' ', '-'),
                            resource_name.replace(' ', '_')
                        ]
                    
                    navigation_url = None
                    for variant in resource_variants:
                        if variant in response.data:
                            navigation_url = response.data[variant]
                            print(f"  → Following navigation link for '{variant}': {navigation_url}", file=sys.stderr)
                            break
                
                if navigation_url:
                    
                    # Make a second request to the actual endpoint
                    from .types import APIRequest
                    from urllib.parse import urlparse
                    
                    # Extract the path from the navigation URL
                    parsed_url = urlparse(navigation_url)
                    base_parsed = urlparse(base_url)
                    
                    # Get the endpoint path relative to base_url
                    actual_endpoint = parsed_url.path
                    if base_parsed.path and base_parsed.path != '/':
                        # Remove the base path to get just the endpoint
                        if actual_endpoint.startswith(base_parsed.path):
                            actual_endpoint = actual_endpoint[len(base_parsed.path):]
                    
                    # Ensure endpoint starts with /
                    if not actual_endpoint.startswith('/'):
                        actual_endpoint = '/' + actual_endpoint
                    
                    followup_request = APIRequest(
                        endpoint=actual_endpoint,
                        method='GET',
                        params=api_request.params,
                        authentication_required=auth_required
                    )
                    
                    response = self.client.call_api(followup_request)
        
        return {
            "data": response.data if isinstance(response, APIResponse) else None,
            "status": response.status_code if isinstance(response, APIResponse) else None,
            "error": response.message if isinstance(response, APIError) else None
        }
    
    def _execute_json_files(self, plan: Dict[str, Any], schema: dict) -> Dict[str, Any]:
        """
        Execute plan against JSON files using knowledge graph.
        
        SKELETON - To be implemented
        
        Args:
            plan: Execution plan
            schema: Knowledge graph schema
        
        Steps:
        1. Get JSON file paths from config
        2. Read and parse JSON files
        3. Filter/search based on plan parameters and knowledge graph
        4. Return matching data
        """
        # TODO: Implement JSON file search with knowledge graph
        # json_config = self.config.get('data_sources', {}).get('json_files', {})
        # file_paths = json_config.get('file_paths', [])
        # entities = schema.get('entities', {})
        
        return {
            "data": None,
            "status": None,
            "error": "JSON file search not yet implemented"
        }
    
    def _execute_database(self, plan: Dict[str, Any], schema: dict, credentials: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute database query plan.
        
        Supports: PostgreSQL, MySQL, SQLite, MongoDB
        
        Args:
            plan: {
                'type': 'database',
                'table': 'employees',
                'operation': 'SELECT',
                'sql_query': 'SELECT * FROM employees WHERE ...',
                'params': [...]
            }
            schema: Database schema
            credentials: Optional database credentials
        
        Returns:
            Query results
        """
        # Get database config
        db_config = self.config.get('data_sources', {}).get('database', {})
        db_type = db_config.get('type', 'postgresql').lower()
        
        try:
            # Route to appropriate database handler
            if db_type in ['postgresql', 'postgres']:
                return self._execute_postgresql(plan, db_config, credentials)
            elif db_type == 'mysql':
                return self._execute_mysql(plan, db_config, credentials)
            elif db_type == 'sqlite':
                return self._execute_sqlite(plan, db_config, credentials)
            elif db_type == 'mongodb':
                return self._execute_mongodb(plan, db_config, credentials)
            else:
                return {
                    "data": None,
                    "error": f"Unsupported database type: {db_type}"
                }
        except ImportError as e:
            return {
                "data": None,
                "error": f"Database driver not installed: {str(e)}. Install with: pip install {self._get_db_package(db_type)}"
            }
        except Exception as e:
            return {
                "data": None,
                "error": f"Database execution failed: {str(e)}"
            }
    
    def _get_db_package(self, db_type: str) -> str:
        """Get the pip package name for a database type."""
        packages = {
            'postgresql': 'psycopg2-binary',
            'postgres': 'psycopg2-binary',
            'mysql': 'mysql-connector-python',
            'sqlite': 'sqlite3',  # Built-in
            'mongodb': 'pymongo'
        }
        return packages.get(db_type, 'unknown')
    
    def _execute_postgresql(self, plan: Dict[str, Any], db_config: dict, credentials: Optional[str] = None) -> Dict[str, Any]:
        """Execute query on PostgreSQL database."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            raise ImportError("psycopg2-binary")
        
        connection = None
        cursor = None
        
        try:
            # Build connection string
            conn_params = {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 5432),
                'database': db_config.get('database', 'postgres'),
                'user': db_config.get('user', 'postgres'),
                'password': credentials or db_config.get('password', '')
            }
            
            # Connect to database
            connection = psycopg2.connect(**conn_params)
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Execute query
            cursor.execute(plan.get('sql_query'), plan.get('params', []))
            
            # Get results based on operation
            if plan.get('operation') == 'SELECT':
                results = cursor.fetchall()
                data = [dict(row) for row in results]
            else:
                connection.commit()
                data = {'affected_rows': cursor.rowcount}
            
            return {
                "data": data,
                "status": "success",
                "message": f"Query executed successfully"
            }
            
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def _execute_mysql(self, plan: Dict[str, Any], db_config: dict, credentials: Optional[str] = None) -> Dict[str, Any]:
        """Execute query on MySQL database."""
        try:
            import mysql.connector
        except ImportError:
            raise ImportError("mysql-connector-python")
        
        connection = None
        cursor = None
        
        try:
            # Build connection params
            conn_params = {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 3306),
                'database': db_config.get('database', 'mysql'),
                'user': db_config.get('user', 'root'),
                'password': credentials or db_config.get('password', '')
            }
            
            # Connect to database
            connection = mysql.connector.connect(**conn_params)
            cursor = connection.cursor(dictionary=True)
            
            # Execute query
            cursor.execute(plan.get('sql_query'), plan.get('params', []))
            
            # Get results based on operation
            if plan.get('operation') == 'SELECT':
                results = cursor.fetchall()
                data = results
            else:
                connection.commit()
                data = {'affected_rows': cursor.rowcount}
            
            return {
                "data": data,
                "status": "success",
                "message": f"Query executed successfully"
            }
            
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def _execute_sqlite(self, plan: Dict[str, Any], db_config: dict, credentials: Optional[str] = None) -> Dict[str, Any]:
        """Execute query on SQLite database."""
        import sqlite3
        
        connection = None
        cursor = None
        
        try:
            # Get database file path
            db_path = db_config.get('database', 'database.db')
            
            # Connect to database
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            
            # Execute query
            cursor.execute(plan.get('sql_query'), plan.get('params', []))
            
            # Get results based on operation
            if plan.get('operation') == 'SELECT':
                results = cursor.fetchall()
                data = [dict(row) for row in results]
            else:
                connection.commit()
                data = {'affected_rows': cursor.rowcount}
            
            return {
                "data": data,
                "status": "success",
                "message": f"Query executed successfully"
            }
            
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def _execute_mongodb(self, plan: Dict[str, Any], db_config: dict, credentials: Optional[str] = None) -> Dict[str, Any]:
        """Execute query on MongoDB database."""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo")
        
        client = None
        
        try:
            # Build connection string
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 27017)
            user = db_config.get('user')
            password = credentials or db_config.get('password')
            
            if user and password:
                connection_string = f"mongodb://{user}:{password}@{host}:{port}/"
            else:
                connection_string = f"mongodb://{host}:{port}/"
            
            # Connect to MongoDB
            client = MongoClient(connection_string)
            db = client[db_config.get('database', 'test')]
            collection = db[plan.get('table')]
            
            # Execute query based on operation
            operation = plan.get('operation')
            filters = plan.get('filters', {})
            
            if operation == 'SELECT':
                results = list(collection.find(filters))
                # Convert ObjectId to string
                for result in results:
                    if '_id' in result:
                        result['_id'] = str(result['_id'])
                data = results
            elif operation == 'INSERT':
                result = collection.insert_one(plan.get('data', {}))
                data = {'inserted_id': str(result.inserted_id)}
            elif operation == 'UPDATE':
                result = collection.update_many(filters, {'$set': plan.get('data', {})})
                data = {'modified_count': result.modified_count}
            elif operation == 'DELETE':
                result = collection.delete_many(filters)
                data = {'deleted_count': result.deleted_count}
            else:
                return {"data": None, "error": f"Unknown operation: {operation}"}
            
            return {
                "data": data,
                "status": "success",
                "message": f"Query executed successfully"
            }
            
        except Exception as e:
            raise e
        finally:
            if client:
                client.close()
    
    def _execute_rag(self, plan: Dict[str, Any], schema: dict) -> Dict[str, Any]:
        """
        Execute RAG query plan for knowledge graph.
        
        SKELETON - To be fully implemented with vector search + RAG.
        
        Args:
            plan: {
                'type': 'rag',
                'entity_type': 'Person',
                'query_text': 'find people who work at Acme',
                'filters': {...},
                'vector_search': True,
                'relationships': [...]
            }
            schema: Knowledge graph schema
        
        Returns:
            RAG search results
        """
        # Determine data source (PDF or JSON)
        enabled_source = self._get_enabled_data_source()
        
        if enabled_source == 'pdf_documents':
            return self._execute_pdf_rag(plan, schema)
        elif enabled_source == 'json_files':
            return self._execute_json_rag(plan, schema)
        else:
            return {
                "data": None,
                "error": "RAG requires 'pdf_documents' or 'json_files' to be enabled"
            }
    
    def _execute_json_rag(self, plan: Dict[str, Any], schema: dict) -> Dict[str, Any]:
        """
        Execute RAG search on JSON files.
        
        Uses vector embeddings for semantic search + LLM for answer generation.
        """
        json_config = self.config.get('data_sources', {}).get('json_files', {})
        file_paths = json_config.get('file_paths', [])
        query_text = plan.get('query_text', '')
        
        try:
            # Load JSON files
            documents = []
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Convert to text for embedding
                        if isinstance(data, list):
                            for item in data:
                                documents.append({
                                    'content': json.dumps(item, indent=2),
                                    'metadata': {'source': file_path, 'type': 'json'}
                                })
                        else:
                            documents.append({
                                'content': json.dumps(data, indent=2),
                                'metadata': {'source': file_path, 'type': 'json'}
                            })
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
            
            if not documents:
                return {
                    "data": None,
                    "error": "No JSON documents found or could be loaded"
                }
            
            # Try to use vector search if available
            try:
                return self._rag_with_embeddings(query_text, documents, plan)
            except ImportError:
                # Fallback to simple keyword search
                return self._rag_with_keyword_search(query_text, documents, plan)
                
        except Exception as e:
            return {
                "data": None,
                "error": f"JSON RAG search failed: {str(e)}"
            }
    
    def _rag_with_embeddings(self, query: str, documents: list, plan: dict) -> Dict[str, Any]:
        """RAG using vector embeddings (requires sentence-transformers)."""
        from sentence_transformers import SentenceTransformer, util
        import torch
        
        # Load embedding model (cached after first load)
        if not hasattr(self, '_embedding_model'):
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode query
        query_embedding = self._embedding_model.encode(query, convert_to_tensor=True)
        
        # Encode documents
        doc_texts = [doc['content'] for doc in documents]
        doc_embeddings = self._embedding_model.encode(doc_texts, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Get top-k results
        top_k = plan.get('vector_search_params', {}).get('top_k', 5)
        top_results = torch.topk(similarities, k=min(top_k, len(documents)))
        
        # Filter by similarity threshold
        threshold = plan.get('vector_search_params', {}).get('similarity_threshold', 0.3)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score >= threshold:
                results.append({
                    'content': documents[idx]['content'],
                    'metadata': documents[idx]['metadata'],
                    'score': float(score)
                })
        
        # Generate answer using context (if LLM available)
        answer = self._generate_answer_from_context(query, results)
        
        return {
            "data": {
                'answer': answer,
                'sources': results,
                'method': 'vector_embeddings'
            },
            "status": "success",
            "message": f"Found {len(results)} relevant documents"
        }
    
    def _rag_with_keyword_search(self, query: str, documents: list, plan: dict) -> Dict[str, Any]:
        """Fallback RAG using keyword search (no dependencies)."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Score documents by keyword matches
        scored_docs = []
        for doc in documents:
            content_lower = doc['content'].lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            if matches > 0:
                score = matches / len(query_terms)
                scored_docs.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': score
                })
        
        # Sort by score and take top-k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        top_k = plan.get('vector_search_params', {}).get('top_k', 5)
        results = scored_docs[:top_k]
        
        # Generate simple answer
        answer = f"Found {len(results)} documents matching your query. " + \
                 "Install 'sentence-transformers' for better semantic search."
        
        return {
            "data": {
                'answer': answer,
                'sources': results,
                'method': 'keyword_search',
                'note': 'Using fallback keyword search. Install sentence-transformers for vector search.'
            },
            "status": "success",
            "message": f"Found {len(results)} relevant documents"
        }
    
    def _generate_answer_from_context(self, query: str, context_docs: list) -> str:
        """Generate answer using LLM if available, otherwise return context summary."""
        if not context_docs:
            return "No relevant information found."
        
        # Try to use OpenAI API if available
        try:
            import os
            from openai import OpenAI
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            
            # Prepare context
            context = "\n\n".join([doc['content'] for doc in context_docs[:3]])
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer the question based on the provided context. If the answer is not in the context, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception:
            # Fallback: return first document snippet
            first_doc = context_docs[0]['content'][:500]
            return f"Most relevant information (excerpt):\n{first_doc}..."
    
    def _execute_pdf_rag(self, plan: Dict[str, Any], schema: dict) -> Dict[str, Any]:
        """
        Execute RAG search on PDF documents.
        
        Uses PDF parsing + vector embeddings for semantic search.
        """
        pdf_config = self.config.get('data_sources', {}).get('pdf_documents', {})
        directory = pdf_config.get('directory', '')
        query_text = plan.get('query_text', '')
        
        try:
            # Load and parse PDF files
            documents = []
            pdf_files = self._get_pdf_files(directory)
            
            if not pdf_files:
                return {
                    "data": None,
                    "error": f"No PDF files found in directory: {directory}"
                }
            
            for pdf_file in pdf_files:
                try:
                    text = self._extract_pdf_text(pdf_file)
                    if text:
                        # Split into chunks for better search
                        chunks = self._chunk_text(text, chunk_size=500)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'content': chunk,
                                'metadata': {
                                    'source': pdf_file,
                                    'type': 'pdf',
                                    'chunk': i
                                }
                            })
                except Exception as e:
                    print(f"Warning: Could not parse {pdf_file}: {e}")
            
            if not documents:
                return {
                    "data": None,
                    "error": "No text could be extracted from PDF files"
                }
            
            # Use same RAG logic as JSON
            try:
                return self._rag_with_embeddings(query_text, documents, plan)
            except ImportError:
                return self._rag_with_keyword_search(query_text, documents, plan)
                
        except Exception as e:
            return {
                "data": None,
                "error": f"PDF RAG search failed: {str(e)}"
            }
    
    def _get_pdf_files(self, directory: str) -> list:
        """Get all PDF files from directory."""
        from pathlib import Path
        
        if not directory or not Path(directory).exists():
            return []
        
        path = Path(directory)
        return [str(f) for f in path.glob('**/*.pdf')]
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            # Try PyPDF2 first
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        except ImportError:
            try:
                # Fallback to pdfplumber
                import pdfplumber
                
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                    return text
                    
            except ImportError:
                raise ImportError("PDF parsing requires PyPDF2 or pdfplumber. Install with: pip install PyPDF2")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    # ========================================================================
    # AUTHENTICATION HANDLERS
    # ========================================================================
    
    def _detect_auth_type(self, schema: dict) -> Optional[str]:
        """
        Detect authentication type required by the API.
        
        Checks both schema and config for authentication requirements.
        
        Priority:
        1. Schema authentication field
        2. Config security_credentials settings
        
        Args:
            schema: API schema
            
        Returns:
            Authentication type: 'jwt', 'oauth', 'api_key', or None
        """
        # Check schema for authentication hints
        auth_info = schema.get('authentication', {})
        if auth_info:
            auth_type = auth_info.get('type')
            if auth_type:
                return auth_type.lower()
        
        # Check config security credentials
        security = self.config.get('security_credentials', {}).get('api', {})
        
        # Check which auth method is enabled in config
        if security.get('jwt', {}).get('enabled'):
            return 'jwt'
        elif security.get('oauth', {}).get('enabled'):
            return 'oauth'
        elif security.get('api_keys', {}).get('enabled'):
            return 'api_key'
        
        return None
    
    def _authenticate(self, base_url: str, auth_type: str) -> Dict[str, Any]:
        """
        Authenticate with the API and get access token.
        
        Makes a call to the authentication endpoint and returns the token.
        Caches tokens to avoid repeated authentication calls.
        
        Args:
            base_url: API base URL
            auth_type: Type of authentication ('jwt', 'oauth', 'api_key')
            
        Returns:
            {
                'token': str,  # Access token
                'error': str   # Error message if failed
            }
        """
        # Check token cache first
        cache_key = f"{auth_type}_{base_url}"
        if hasattr(self, '_auth_token_cache') and cache_key in self._auth_token_cache:
            cached = self._auth_token_cache[cache_key]
            # Check if token is still valid (simple time-based check)
            from datetime import datetime
            if datetime.now() < cached.get('expires_at', datetime.now()):
                return {'token': cached['token']}
        
        # Route to appropriate auth handler
        if auth_type == 'jwt':
            return self._authenticate_jwt(base_url)
        elif auth_type == 'oauth':
            return self._authenticate_oauth(base_url)
        elif auth_type == 'api_key':
            return self._authenticate_api_key()
        else:
            return {'error': f"Unsupported authentication type: {auth_type}"}
    
    def _authenticate_jwt(self, base_url: str) -> Dict[str, Any]:
        """
        Authenticate using JWT (obtain token from token endpoint).
        
        Makes a POST request to the JWT token endpoint with credentials.
        
        Args:
            base_url: API base URL
            
        Returns:
            {'token': str} or {'error': str}
        """
        import os
        import requests
        from datetime import datetime, timedelta
        
        try:
            # Get JWT config
            jwt_config = self.config.get('security_credentials', {}).get('api', {}).get('jwt', {})
            
            if not jwt_config.get('enabled'):
                return {'error': 'JWT authentication not enabled in config'}
            
            # Get token endpoint
            token_endpoint = jwt_config.get('token_endpoint', '/api/token/')
            token_url = base_url.rstrip('/') + token_endpoint
            
            # Get credentials from environment or config
            username = os.getenv('API_USERNAME') or jwt_config.get('username')
            password = os.getenv('API_PASSWORD') or jwt_config.get('password')
            email = os.getenv('API_EMAIL') or jwt_config.get('email')
            
            # Try test credentials if available
            if not username and not email:
                username = jwt_config.get('test_username')
                email = jwt_config.get('test_email')
                password = jwt_config.get('test_password')
            
            if (not username and not email) or not password:
                return {'error': 'JWT credentials not found. Set API_USERNAME/API_EMAIL and API_PASSWORD environment variables or configure test credentials in config.json'}
            
            # Build credentials payload (support both username and email)
            credentials_payload = {'password': password}
            if email:
                credentials_payload['email'] = email
            else:
                credentials_payload['username'] = username
            
            # Make authentication request
            response = requests.post(
                token_url,
                json=credentials_payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code not in [200, 201]:
                return {'error': f"Authentication failed with status {response.status_code}: {response.text[:200]}"}
            
            # Parse response
            data = response.json()
            access_token = data.get('access') or data.get('access_token') or data.get('token')
            
            if not access_token:
                return {'error': f"No access token in response. Keys: {list(data.keys())}"}
            
            # Cache token
            if not hasattr(self, '_auth_token_cache'):
                self._auth_token_cache = {}
            
            # Assume token valid for 1 hour (adjust based on your API)
            expires_in = data.get('expires_in', 3600)
            cache_key = f"jwt_{base_url}"
            self._auth_token_cache[cache_key] = {
                'token': access_token,
                'expires_at': datetime.now() + timedelta(seconds=expires_in - 60)  # 60s buffer
            }
            
            print(f"✓ JWT authentication successful", file=sys.stderr)
            return {'token': access_token}
            
        except requests.RequestException as e:
            return {'error': f"Authentication request failed: {str(e)}"}
        except Exception as e:
            return {'error': f"JWT authentication failed: {str(e)}"}
    
    def _authenticate_oauth(self, base_url: str) -> Dict[str, Any]:
        """
        Authenticate using OAuth 2.0 client credentials flow.
        
        Args:
            base_url: API base URL
            
        Returns:
            {'token': str} or {'error': str}
        """
        import os
        import requests
        from datetime import datetime, timedelta
        
        try:
            # Get OAuth config
            oauth_config = self.config.get('security_credentials', {}).get('api', {}).get('oauth', {})
            
            if not oauth_config.get('enabled'):
                return {'error': 'OAuth authentication not enabled in config'}
            
            # Get credentials from environment
            client_id_env = oauth_config.get('client_id_env', 'OAUTH_CLIENT_ID')
            client_secret_env = oauth_config.get('client_secret_env', 'OAUTH_CLIENT_SECRET')
            
            client_id = os.getenv(client_id_env)
            client_secret = os.getenv(client_secret_env)
            
            if not client_id or not client_secret:
                return {'error': f"OAuth credentials not found in environment ({client_id_env}, {client_secret_env})"}
            
            # Get token URL
            token_url = oauth_config.get('token_url')
            if not token_url:
                return {'error': 'OAuth token_url not configured'}
            
            # Make token request
            response = requests.post(
                token_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'scope': oauth_config.get('scope', '')
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            access_token = data.get('access_token')
            if not access_token:
                return {'error': 'No access_token in OAuth response'}
            
            # Cache token
            if not hasattr(self, '_auth_token_cache'):
                self._auth_token_cache = {}
            
            expires_in = data.get('expires_in', 3600)
            cache_key = f"oauth_{base_url}"
            self._auth_token_cache[cache_key] = {
                'token': access_token,
                'expires_at': datetime.now() + timedelta(seconds=expires_in - 60)
            }
            
            print(f"✓ OAuth authentication successful")
            return {'token': access_token}
            
        except Exception as e:
            return {'error': f"OAuth authentication failed: {str(e)}"}
    
    def _authenticate_api_key(self) -> Dict[str, Any]:
        """
        Get API key from environment or config.
        
        Note: API keys are typically passed directly, not obtained from an endpoint.
        
        Returns:
            {'token': str} or {'error': str}
        """
        import os
        
        try:
            api_key_config = self.config.get('security_credentials', {}).get('api', {}).get('api_keys', {})
            
            if not api_key_config.get('enabled'):
                return {'error': 'API key authentication not enabled in config'}
            
            # Get API key from environment
            key_env_var = api_key_config.get('key_env_variable', 'API_KEY')
            api_key = os.getenv(key_env_var)
            
            if not api_key:
                return {'error': f"API key not found in environment variable {key_env_var}"}
            
            print(f"✓ API key loaded from environment")
            return {'token': api_key}
            
        except Exception as e:
            return {'error': f"API key authentication failed: {str(e)}"}
    
    def _get_auth_headers(self, data_source: str, credentials: str = None) -> Dict[str, str]:
        """
        Get authentication headers based on configured security method.
        
        Routes to the correct auth handler based on what's enabled in config.
        """
        security = self.config.get('security_credentials', {})
        source_security = security.get(data_source, {})
        
        # Check which auth method is enabled
        if source_security.get('jwt', {}).get('enabled'):
            return self._auth_jwt(credentials, source_security.get('jwt', {}))
        elif source_security.get('oauth', {}).get('enabled'):
            return self._auth_oauth(source_security.get('oauth', {}))
        elif source_security.get('api_keys', {}).get('enabled'):
            return self._auth_api_key(credentials, source_security.get('api_keys', {}))
        else:
            return {}
    
    def _auth_jwt(self, token: str, jwt_config: Dict[str, Any]) -> Dict[str, str]:
        """
        JWT authentication.
        
        FULLY IMPLEMENTED ✓
        """
        if not token:
            return {}
        
        header_format = jwt_config.get('header_format', 'Bearer {token}')
        auth_value = header_format.replace('{token}', token)
        
        return {
            'Authorization': auth_value
        }
    
    def _auth_oauth(self, oauth_config: Dict[str, Any]) -> Dict[str, str]:
        """
        OAuth 2.0 authentication.
        
        Supports client credentials flow (most common for API access).
        """
        import os
        import requests
        from datetime import datetime, timedelta
        
        try:
            # Get OAuth credentials from environment
            client_id_env = oauth_config.get('client_id_env', 'OAUTH_CLIENT_ID')
            client_secret_env = oauth_config.get('client_secret_env', 'OAUTH_CLIENT_SECRET')
            
            client_id = os.getenv(client_id_env)
            client_secret = os.getenv(client_secret_env)
            
            if not client_id or not client_secret:
                print(f"Warning: OAuth credentials not found in environment ({client_id_env}, {client_secret_env})")
                return {}
            
            # Check if we have a cached token
            cache_key = f"oauth_token_{client_id}"
            if hasattr(self, '_token_cache') and cache_key in self._token_cache:
                token_data = self._token_cache[cache_key]
                if datetime.now() < token_data['expires_at']:
                    return {'Authorization': f"Bearer {token_data['token']}"}
            
            # Get new token using client credentials flow
            token_url = oauth_config.get('token_url')
            if not token_url:
                print("Warning: OAuth token_url not configured")
                return {}
            
            response = requests.post(
                token_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'scope': oauth_config.get('scope', '')
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            access_token = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)
            
            # Cache token
            if not hasattr(self, '_token_cache'):
                self._token_cache = {}
            
            self._token_cache[cache_key] = {
                'token': access_token,
                'expires_at': datetime.now() + timedelta(seconds=expires_in - 60)  # 60s buffer
            }
            
            return {'Authorization': f"Bearer {access_token}"}
            
        except Exception as e:
            print(f"Warning: OAuth authentication failed: {e}")
            return {}
    
    def _auth_api_key(self, api_key: str, api_key_config: Dict[str, Any]) -> Dict[str, str]:
        """
        API Key authentication.
        
        Supports multiple header formats and environment variables.
        """
        import os
        
        try:
            # Get API key from parameter or environment
            key_env_var = api_key_config.get('key_env_variable', 'API_KEY')
            key = api_key or os.getenv(key_env_var)
            
            if not key:
                print(f"Warning: API key not provided and not found in environment ({key_env_var})")
                return {}
            
            # Get header configuration
            key_header = api_key_config.get('key_header', 'X-API-Key')
            key_prefix = api_key_config.get('key_prefix', '')  # e.g., "Bearer ", "ApiKey "
            
            # Format the header value
            if key_prefix:
                header_value = f"{key_prefix}{key}"
            else:
                header_value = key
            
            return {key_header: header_value}
            
        except Exception as e:
            print(f"Warning: API key authentication failed: {e}")
            return {}
    
    def _summarize_result(self, result: Dict[str, Any], query: str) -> str:
        """
        Step 5: Create human-readable summary of result.
        
        Handles all data source types:
        - API: Standard REST responses
        - Database: Query results with affected_rows
        - RAG (JSON/PDF): Answer with sources and scores
        """
        # Check for errors first
        if result.get('error'):
            return f"Error: {result['error']}"
        
        # Check status (for pending operations)
        status = result.get('status')
        if status == 'pending':
            message = result.get('message', 'Operation pending')
            return f"⚠️  {message}"
        
        data = result.get('data')
        
        if not data:
            # Check if there's a message instead
            if result.get('message'):
                return result['message']
            return "No data returned"
        
        # Handle different data structures
        if isinstance(data, dict):
            # RAG responses (JSON/PDF vector search)
            if 'answer' in data and 'sources' in data:
                answer = data['answer']
                source_count = len(data.get('sources', []))
                method = data.get('method', 'unknown')
                
                if method == 'vector_embeddings':
                    return f"✓ Found answer using semantic search ({source_count} relevant sources)"
                elif method == 'keyword_search':
                    return f"✓ Found answer using keyword search ({source_count} matches)"
                else:
                    return f"✓ Found answer ({source_count} sources)"
            
            # Database responses (affected rows)
            elif 'affected_rows' in data:
                count = data['affected_rows']
                if count == 0:
                    return "No rows affected"
                elif count == 1:
                    return "Successfully affected 1 row"
                else:
                    return f"Successfully affected {count} rows"
            
            # API responses with count (paginated lists)
            elif 'count' in data:
                total = data['count']
                results = data.get('results', [])
                if results:
                    return f"Found {len(results)} of {total} total items"
                else:
                    return f"Found {total} items"
            
            # Single object responses (API/Database)
            elif 'id' in data:
                return f"✓ Successfully retrieved item (ID: {data['id']})"
            
            # MongoDB responses
            elif 'inserted_id' in data:
                return f"✓ Successfully inserted document (ID: {data['inserted_id']})"
            elif 'modified_count' in data:
                count = data['modified_count']
                return f"✓ Modified {count} document(s)"
            elif 'deleted_count' in data:
                count = data['deleted_count']
                return f"✓ Deleted {count} document(s)"
            
            # Generic success
            else:
                return "✓ Operation completed successfully"
        
        # List responses (API results, Database rows)
        elif isinstance(data, list):
            if len(data) == 0:
                return "No items found"
            elif len(data) == 1:
                return "Found 1 item"
            else:
                return f"Found {len(data)} items"
        
        # String or other types
        else:
            return "Operation completed"
    
    def continue_conversation(self, context: Any, follow_up: str, access_token: str = None) -> Dict[str, Any]:
        """
        Continue a conversation with missing information.
        
        Args:
            context: Previous context from process() call
            follow_up: User's follow-up response
            access_token: Optional JWT token
            
        Returns:
            Same format as process()
        """
        # Merge follow-up into context and reprocess
        return self.process(follow_up, access_token, context)


# Convenience function for simple usage
def process_query(query: str, access_token: Optional[str] = None, schema: Optional[dict] = None) -> Dict[str, Any]:
    """
    Quick function to process a single query.
    
    Usage:
        from nlp_api_caller import process_query
        
        # With schema
        result = process_query("get user 5", schema=my_schema)
    """
    processor = NLPProcessor()
    return processor.process(query, access_token, schema=schema)
