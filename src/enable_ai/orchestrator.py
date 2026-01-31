"""
Enable AI - AI-Powered API Query Processor

Main interface for processing natural language queries against REST APIs.
"""

from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
import json
import sys

from .query_parser import QueryParser
from .api_matcher import APIMatcher
from .api_client import APIClient
from .config_loader import get_config
from .types import MissingInformation, APIResponse, APIError
from .workflow import build_api_workflow
from .utils import setup_logger
from .schema_validator import SchemaValidator
from .progress_tracker import ProgressTracker, ProgressUpdate, ProgressStage
from . import constants


class APIOrchestrator:
    """
    Orchestrator for planning and executing multi-step API workflows.
    
    Features:
    - Multi-step planning with dependency resolution
    - Sequential API execution with context passing
    - State persistence via LangGraph checkpointing
    - MCP server integration for AI assistants
    
    Usage:
        # Auto-load schemas from config.json
        orchestrator = APIOrchestrator()
        
        # Process single query
        result = orchestrator.process("get user with id 5")
        
        # Process with state persistence (multi-turn)
        result = orchestrator.process("list users", session_id="abc123")
        result = orchestrator.process("show me the first one", session_id="abc123")
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        schemas: Optional[Dict[str, Union[str, dict]]] = None,
        conversation_store=None,
        formatter_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize API Orchestrator.
        
        Args:
            config_path: Path to config.json (optional, defaults to 'config.json')
            schemas: Optional schema override dict (overrides config)
                     Format: {'api': path_or_dict}
            conversation_store: Optional ConversationStore instance for persistent conversation history.
                               If None, uses InMemoryConversationStore (NOT production safe).
                               For production, use DjangoConversationStore or RedisConversationStore.
        
        Examples:
            # Development/testing (in-memory storage)
            orchestrator = APIOrchestrator()
            
            # Production with Django (recommended)
            from enable_ai.conversation_store import DjangoConversationStore
            from myapp.models import ConversationMessage
            store = DjangoConversationStore(ConversationMessage)
            orchestrator = APIOrchestrator(conversation_store=store)
            
            # Production with Redis
            from enable_ai.conversation_store import RedisConversationStore
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            store = RedisConversationStore(redis_client)
            orchestrator = APIOrchestrator(conversation_store=store)
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
        
        # Initialize logger
        self.logger = setup_logger('enable_ai.orchestrator')
        
        # Initialize components
        self.parser = QueryParser()
        self.api_matcher = None  # Initialized when needed with api_spec
        self.client = None  # Initialized when needed with base URL
        
        # Conversation storage (v0.3.13: replaced in-memory dict with pluggable storage)
        if conversation_store is None:
            # Default to in-memory storage for backward compatibility
            # WARNING: Not production safe - data lost on restart!
            from .conversation_store import InMemoryConversationStore
            self.conversation_store = InMemoryConversationStore()
            self.logger.warning(
                "Using InMemoryConversationStore - NOT production safe! "
                "Pass a DjangoConversationStore or RedisConversationStore for production."
            )
        else:
            self.conversation_store = conversation_store
            self.logger.info(f"Using {type(conversation_store).__name__} for conversation storage")
        
        # Initialize workflow (LangGraph) - no checkpointing needed
        # Conversation state is managed externally via conversation_store
        self.workflow = build_api_workflow(self, checkpointer=None, formatter_config=formatter_config)
        self.logger.info("LangGraph workflow initialized")

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
                        # Check if it's OpenAPI format and convert if needed
                        if self._is_openapi_schema(schema_input):
                            schema_input = self._convert_openapi_to_enable_ai(schema_input)
                        
                        # Validate and use
                        self._validate_schema(schema_input, schema_type)
                        loaded[schema_type] = schema_input
                        print(f"✓ Loaded {schema_type} schema (override)")
                    elif isinstance(schema_input, str):
                        # File path, load it
                        loaded_schema = self._load_schema_file(schema_input, schema_type)
                        if loaded_schema:
                            # Check if loaded schema is OpenAPI and convert
                            if self._is_openapi_schema(loaded_schema):
                                loaded_schema = self._convert_openapi_to_enable_ai(loaded_schema)
                            
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
    
    def _is_openapi_schema(self, schema: dict) -> bool:
        """Check if schema is in OpenAPI/Swagger format."""
        return 'openapi' in schema or 'swagger' in schema
    
    def _convert_openapi_to_enable_ai(self, openapi_schema: dict) -> dict:
        """
        Convert OpenAPI/Swagger schema to Enable AI format.
        
        Args:
            openapi_schema: OpenAPI 3.x or Swagger 2.0 schema
            
        Returns:
            Enable AI format schema
        """
        try:
            from .schema_generator import OpenAPIConverter
            
            converter = OpenAPIConverter()
            enable_ai_schema = converter.generate(openapi_schema)
            
            version = openapi_schema.get('openapi', openapi_schema.get('swagger', 'unknown'))
            print(f"✓ Auto-converted OpenAPI {version} schema to Enable AI format")
            
            return enable_ai_schema
        except ImportError:
            raise ValueError(
                "OpenAPI converter not available. Install with: pip install enable-ai[schema]"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to convert OpenAPI schema: {e}\n"
                f"Try manual conversion: enable-schema convert your-openapi.json"
            )
    
    def _validate_schema(self, schema: dict, expected_type: str) -> None:
        """
        Validate schema structure with comprehensive validation (v0.3.8).
        
        Args:
            schema: Schema dict to validate
            expected_type: Expected schema type
            
        Raises:
            ValueError: If schema is invalid (with helpful message)
        """
        # Check for OpenAPI/Swagger format (suggest auto-conversion)
        if 'openapi' in schema or 'swagger' in schema:
            version = schema.get('openapi', schema.get('swagger', 'unknown'))
            raise ValueError(
                f"Detected OpenAPI/Swagger schema (version {version}).\n"
                f"This needs to be converted to Enable AI format.\n"
                f"Options:\n"
                f"  1. Use CLI: enable-schema convert your-openapi.json\n"
                f"  2. Use Python: from enable_ai.schema_generator import convert_openapi; schema = convert_openapi('your-openapi.json')\n"
                f"  3. Enable AI will auto-convert if you pass OpenAPI dict to schemas parameter"
            )
        
        # Use comprehensive schema validator (v0.3.8)
        validator = SchemaValidator()
        is_valid, errors, warnings = validator.validate_schema(schema, strict=False)
        
        if not is_valid:
            error_msg = f"Schema validation failed for type '{expected_type}':\n\n"
            error_msg += "ERRORS:\n"
            for idx, error in enumerate(errors, 1):
                error_msg += f"  {idx}. {error}\n"
            
            if warnings:
                error_msg += "\nWARNINGS:\n"
                for idx, warning in enumerate(warnings, 1):
                    error_msg += f"  {idx}. {warning}\n"
            
            error_msg += "\nTip: Use 'enable-schema convert' to generate a valid schema from OpenAPI"
            raise ValueError(error_msg)
        
        # Log warnings if any
        if warnings:
            for warning in warnings:
                self.logger.warning(f"Schema validation warning: {warning}")
    
    def _print_init_summary(self) -> None:
        """Print initialization summary."""
        print("✓ NLP Processor initialized", file=sys.stderr)
        
        # Print enabled data sources
        enabled_sources = self._get_enabled_data_sources_list()
        if enabled_sources:
            print(f"  - Enabled data sources: {', '.join(enabled_sources)}", file=sys.stderr)
        else:
            print("  - ⚠️  No data sources enabled", file=sys.stderr)
        
        # Print loaded schemas (from config)
        if self.schemas:
            schema_names = ', '.join(self.schemas.keys())
            print(f"  - Schemas loaded from config: {schema_names}", file=sys.stderr)
        else:
            if enabled_sources:
                # In many setups, schemas are provided at runtime via runtime_schema.
                print("  - No init-time schemas loaded from config (runtime_schema will be required)", file=sys.stderr)
            else:
                print("  - No schemas loaded (runtime schema required)", file=sys.stderr)
        
        # Print client support
        client_support = list(self.config.get('client_support', {}).keys())
        if client_support:
            print(f"  - Client support: {', '.join(client_support)}", file=sys.stderr)
    
    def process(
        self, 
        query: str, 
        access_token: Optional[str] = None, 
        context: Optional[Any] = None, 
        runtime_schema: Optional[dict] = None,
        schema: Optional[dict] = None,  # Deprecated, use runtime_schema
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None  # v0.3.10: Real-time progress
    ) -> Dict[str, Any]:
        """
        Process a natural language query with multi-step planning and execution.
        
        Pipeline (LangGraph):
            1. Load schema (determine active schema)
            2. Parse query (understand intent with schema) - OpenAI
            3. Create execution plan (match to endpoints, resolve dependencies)
            4. Execute plan (sequential API calls with context passing)
            5. Summarize result (format response) - OpenAI
            6. Return result
        
        Args:
            query: Natural language query from user
            access_token: Optional JWT token for authentication
            context: Optional extra context passed to the parser (_understand_query). Use for
                     additional hints; conversation history is loaded from conversation_store when session_id is set.
            runtime_schema: Optional schema to use for this query only (overrides init schemas)
            schema: Deprecated, use runtime_schema instead
            session_id: Optional session ID for multi-turn context (conversation_history loaded/saved via conversation_store)
            progress_callback: Optional callback for real-time progress updates (v0.3.10)
                              Signature: callback(stage: str, message: str, progress: float, metadata: dict)
                              Example: callback("parsing_query", "Understanding...", 0.1, {})
            
        Returns:
            {
                "success": bool,
                "data": dict | list,  # Can be multiple results from sequential calls
                "summary": str,
                "query": str,
                "execution_plan": list,  # List of API calls executed
                "schema_type": str
            }
        
        Examples:
            # Single query
            result = orchestrator.process("get user with id 5")
            
            # Multi-turn conversation
            result1 = orchestrator.process("list users", session_id="abc123")
            result2 = orchestrator.process("show me the first one", session_id="abc123")
            
            # Complex multi-step query
            result = orchestrator.process("Get user 5 and all their orders")
            # Executes: GET /users/5/ → GET /orders/?user_id=5
            
            # Runtime schema override (NEW in v0.3.3)
            result = orchestrator.process("query", runtime_schema=custom_schema)
        """
        # High-level entrypoint log
        self.logger.info(
            "process() called | query=%r | session_id=%r | runtime_schema=%s",
            query[: constants.QUERY_PREVIEW_LENGTH],
            session_id,
            "yes" if runtime_schema is not None or schema is not None else "no",
        )

        # Backward compatibility: support old 'schema' parameter
        if schema is not None and runtime_schema is None:
            import warnings
            warnings.warn(
                "Parameter 'schema' is deprecated, use 'runtime_schema' instead",
                DeprecationWarning,
                stacklevel=2
            )
            runtime_schema = schema
        
        # Auto-convert OpenAPI if needed
        if runtime_schema and self._is_openapi_schema(runtime_schema):
            self.logger.info("Runtime schema detected as OpenAPI/Swagger – auto-converting to Enable AI format")
            runtime_schema = self._convert_openapi_to_enable_ai(runtime_schema)
        
        # Get conversation history if session_id provided (v0.3.13: using external store)
        conversation_history = []
        if session_id:
            conversation_history = self.conversation_store.get_history(session_id)
            self.logger.debug(f"Loaded {len(conversation_history)} messages from session {session_id}")
        
        # Setup progress tracking (v0.3.10)
        progress_tracker = None
        if progress_callback:
            def tracker_callback(update: ProgressUpdate):
                """Adapter to convert ProgressUpdate to simpler callback signature"""
                try:
                    progress_callback(
                        update.stage.value,
                        update.message,
                        update.progress,
                        update.metadata
                    )
                except TypeError:
                    # Fallback for callbacks that don't accept metadata
                    try:
                        progress_callback(update.stage.value, update.message, update.progress)
                    except Exception as e:
                        self.logger.error(f"Progress callback error: {e}")
            
            progress_tracker = ProgressTracker(callback=tracker_callback)
            progress_tracker.update(ProgressStage.STARTED, "Starting your request...")
        
        try:
            # No checkpointing config needed - conversation state managed externally
            state = self.workflow.invoke(
                {
                    "query": query,
                    "access_token": access_token,
                    "context": context,
                    "runtime_schema": runtime_schema,
                    "session_id": session_id,
                    "conversation_history": conversation_history,
                    "progress_tracker": progress_tracker,  # v0.3.10: pass tracker to workflow
                }
            )
            
            # Detailed debug log of internal state (truncated to avoid huge payloads)
            parsed = state.get("parsed")
            if parsed:
                try:
                    parsed_preview = json.dumps(parsed)[: constants.LOG_PREVIEW_LENGTH]
                except TypeError:
                    parsed_preview = str(parsed)[: constants.LOG_PREVIEW_LENGTH]
                self.logger.debug("Parsed query (workflow state): %s", parsed_preview)

            response = state.get("response", {"success": False, "error": constants.ERROR_NO_RESPONSE_GENERATED, "query": query})

            # Log high-level outcome
            data = response.get("data")
            if isinstance(data, list):
                data_info = f"list(len={len(data)})"
            elif isinstance(data, dict):
                # Paginated or single-object
                if "results" in data and isinstance(data["results"], list):
                    data_info = f"dict(results_len={len(data['results'])})"
                else:
                    data_info = f"dict(keys={list(data.keys())})"
            else:
                data_info = type(data).__name__

            self.logger.info(
                "process() completed | success=%s | schema_type=%s | data=%s",
                response.get("success"),
                response.get("schema_type"),
                data_info,
            )
            
            # Store conversation history if session_id provided and query was successful (v0.3.13: using external store)
            if session_id and response.get("success"):
                # Add user message
                self.conversation_store.add_message(session_id, 'user', query)
                
                # Build assistant response with semantic context
                parsed = state.get('parsed', {})
                resource = parsed.get('resource', 'items')
                intent = parsed.get('intent', 'read')
                filters = parsed.get('filters', {})
                summary = response.get('summary', 'Completed successfully')
                
                # Enhanced assistant message with semantic context (v0.3.7: include filters)
                assistant_message = f"{summary}\n[Context: {intent} operation on {resource}]"
                if filters:
                    assistant_message += f"\n[Filters: {json.dumps(filters)}]"
                
                # Add assistant message with metadata
                self.conversation_store.add_message(
                    session_id, 
                    'assistant', 
                    assistant_message,
                    metadata={'resource': resource, 'intent': intent, 'filters': filters}
                )
            
            return response
        except Exception as e:
            self.logger.error("process() failed: %s", e)
            return {"success": False, "error": str(e), "query": query}  # error string is dynamic

    def process_stream(
        self,
        query: str,
        access_token: Optional[str] = None,
        context: Optional[Any] = None,
        runtime_schema: Optional[dict] = None,
        session_id: Optional[str] = None,
    ):
        """
        Process a query and stream state updates after each workflow node.
        Yields (node_name, state) for each node; use the last state for the final response (state['response']).
        Progress is streamed via state updates; for callback-based progress use process(progress_callback=...).
        """
        if runtime_schema and self._is_openapi_schema(runtime_schema):
            runtime_schema = self._convert_openapi_to_enable_ai(runtime_schema)
        conversation_history = []
        if session_id:
            conversation_history = self.conversation_store.get_history(session_id)
        initial_state = {
            "query": query,
            "access_token": access_token,
            "context": context,
            "runtime_schema": runtime_schema,
            "session_id": session_id,
            "conversation_history": conversation_history,
            "progress_tracker": None,
        }
        try:
            for event in self.workflow.stream(initial_state, stream_mode="values"):
                yield event
        except Exception as e:
            self.logger.error("process_stream() failed: %s", e)
            yield "__error__", {"response": {"success": False, "error": str(e), "query": query}}
    
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
    
    def _understand_query(self, query: str, schema: dict, context: Optional[Any] = None, conversation_history: Optional[list] = None) -> Dict[str, Any]:
        """
        Step 2: Parse and understand user query with schema.
        
        Args:
            query: User's natural language query
            schema: Active schema for entity extraction
            context: Optional conversation context
            conversation_history: Optional conversation history for multi-turn queries
        """
        parsed = self.parser.parse_input(query, schema, conversation_history=conversation_history)
        
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

        # Merge filter values into entities so optional query params (e.g. status__name=Quoted)
        # are sent. The matcher builds query params from entities only; filters are not used.
        merged = dict(parsed.get("entities") or {})
        for key, val in (parsed.get("filters") or {}).items():
            if isinstance(val, dict) and "value" in val:
                merged[key] = val["value"]
            else:
                merged[key] = val
        parsed_for_match = {**parsed, "entities": merged}

        result = self.api_matcher.match_api(parsed_for_match, schema)
        
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
                "missing_fields": result.missing_fields,
                "context": result.context
            }
        
        if isinstance(result, APIRequest):
            # Extract information from APIRequest object
            params = dict(result.params) if result.params else {}
            # When user asks "list a few" etc., parsed has limit — send as page_size for DRF/pagination
            limit = parsed.get("limit")
            if limit is not None:
                try:
                    n = min(int(limit), constants.PAGE_SIZE_CAP)
                    params["page_size"] = n
                    self.logger.info(f"Adding page_size={n} from parsed limit for 'list a few' style query")
                except (TypeError, ValueError):
                    pass
            return {
                "type": "api",
                "endpoint": result.endpoint,
                "method": result.method,
                "params": params,
                "authentication_required": result.authentication_required,
                "endpoint_name": "unknown",  # APIRequest doesn't have this
                "module": "unknown",  # APIRequest doesn't have this
                "schema_type": "api_schema"
            }
        
        return None
    
    def _execute_plan(self, plan: Dict[str, Any], schema: dict, access_token: Optional[str] = None, parsed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the plan against API data source.
        
        API-only implementation. Database and RAG support planned for future.
        
        Args:
            plan: Execution plan with 'type' field
            schema: Active schema
            access_token: Optional authentication token
            parsed: Optional parsed query data (for context)
        
        Returns:
            Execution result dict
        """
        plan_type = plan.get('type')
        self.logger.debug(f"Executing plan of type: {plan_type}")
        
        if plan_type == 'api':
            return self._execute_api(plan, schema, access_token, parsed)
        
        elif plan_type == 'error':
            self.logger.warning(f"Plan has error: {plan.get('error')}")
            return {
                "data": None,
                "error": plan.get('error', 'Unknown error')
            }
        
        else:
            error_msg = f"Unsupported plan type: {plan_type}. Only 'api' is currently supported."
            self.logger.error(error_msg)
            return {
                "data": None,
                "error": error_msg
            }
    
    def _get_enabled_data_source(self) -> Optional[str]:
        """Get the enabled API data source from config."""
        data_sources = self.config.get('data_sources', {})
        
        # API-only support
        if data_sources.get('api', {}).get('enabled'):
            return 'api'
        
        self.logger.warning("No API data source enabled in config")
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
            error_msg = constants.ERROR_BASE_URL_NOT_CONFIGURED
            self.logger.error(error_msg)
            return {
                "data": None,
                "error": f"{error_msg}. Set 'base_url' in config.json under data_sources.api or in the API schema."
            }
        
        # Check if authentication is required
        auth_required = plan.get('authentication_required', True)
        self.logger.debug(f"Executing API call: {plan.get('method')} {plan.get('endpoint')}, auth_required={auth_required}")
        
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
                            "error": constants.ERROR_AUTH_NO_TOKEN_RECEIVED
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
        self.logger.info(f"Calling API: {api_request.method} {api_request.endpoint}")
        response = self.client.call_api(api_request)
        
        if isinstance(response, APIResponse):
            self.logger.info(f"API response: status={response.status_code}")
        elif isinstance(response, APIError):
            self.logger.error(f"API error: {response.message}")
        
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
                    
                    self.logger.info(f"Following navigation to: {followup_request.method} {followup_request.endpoint}")
                    response = self.client.call_api(followup_request)
                    api_request = followup_request  # For logging/return metadata
        
        return {
            "data": response.data if isinstance(response, APIResponse) else None,
            "status": response.status_code if isinstance(response, APIResponse) else None,
            "error": response.message if isinstance(response, APIError) else None,
            "endpoint": api_request.endpoint,
            "method": api_request.method,
            "params": api_request.params,
        }

    def _fetch_next_page(self, next_url: str, schema: dict, access_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the next page from a pagination 'next' URL (e.g. Django REST style).
        Used for automatic pagination when response has has_more/next.
        """
        api_config = self.config.get('data_sources', {}).get('api', {})
        base_url = api_config.get('base_url') or schema.get('base_url')
        if not base_url:
            return {"data": None, "error": constants.ERROR_BASE_URL_NOT_CONFIGURED}
        if self.client is None or access_token:
            self.client = APIClient(base_url, access_token)
        response = self.client.get_full_url(next_url)
        if isinstance(response, APIError):
            return {"data": None, "error": response.message}
        return {"data": response.data, "error": None}

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
                timeout=constants.REQUEST_TIMEOUT
            )
            
            if response.status_code not in [200, 201]:
                return {'error': f"Authentication failed with status {response.status_code}: {response.text[: constants.AUTH_ERROR_PREVIEW_LENGTH]}"}
            
            # Parse response
            data = response.json()
            access_token = data.get('access') or data.get('access_token') or data.get('token')
            
            if not access_token:
                return {'error': f"No access token in response. Keys: {list(data.keys())}"}
            
            # Cache token
            if not hasattr(self, '_auth_token_cache'):
                self._auth_token_cache = {}
            
            # Assume token valid for 1 hour (adjust based on your API)
            expires_in = data.get('expires_in', constants.AUTH_EXPIRES_IN_DEFAULT)
            cache_key = f"jwt_{base_url}"
            self._auth_token_cache[cache_key] = {
                'token': access_token,
                'expires_at': datetime.now() + timedelta(seconds=expires_in - constants.AUTH_TOKEN_BUFFER_SECONDS)
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
                timeout=constants.REQUEST_TIMEOUT
            )
            
            response.raise_for_status()
            data = response.json()
            
            access_token = data.get('access_token')
            if not access_token:
                return {'error': 'No access_token in OAuth response'}
            
            # Cache token
            if not hasattr(self, '_auth_token_cache'):
                self._auth_token_cache = {}
            
            expires_in = data.get('expires_in', constants.AUTH_EXPIRES_IN_DEFAULT)
            cache_key = f"oauth_{base_url}"
            self._auth_token_cache[cache_key] = {
                'token': access_token,
                'expires_at': datetime.now() + timedelta(seconds=expires_in - constants.AUTH_TOKEN_BUFFER_SECONDS)
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
            expires_in = token_data.get('expires_in', constants.AUTH_EXPIRES_IN_DEFAULT)
            
            # Cache token
            if not hasattr(self, '_token_cache'):
                self._token_cache = {}
            
            self._token_cache[cache_key] = {
                'token': access_token,
                'expires_at': datetime.now() + timedelta(seconds=expires_in - constants.AUTH_TOKEN_BUFFER_SECONDS)
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
        Step 5: Create intelligent human-readable summary of result.
        
        Handles all data source types:
        - API: Standard REST responses with resource detection and examples
        - Database: Query results with affected_rows
        - RAG (JSON/PDF): Answer with sources and scores
        """
        # Check for errors first
        if result.get('error'):
            return f"{constants.ERROR_PREFIX}{result['error']}"
        
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
            return constants.ORCH_NO_DATA_RETURNED
        
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
                    return constants.ORCH_NO_ROWS_AFFECTED
                elif count == 1:
                    return constants.ORCH_SUCCESS_ONE_ROW
                else:
                    return f"Successfully affected {count} rows"
            
            # API responses with count (paginated lists) - ENHANCED
            elif 'count' in data:
                total = data['count']
                results = data.get('results', [])
                
                if total == 0:
                    return constants.ORCH_NO_ITEMS_FOUND
                
                # Detect resource type and generate intelligent summary
                resource_type = self._detect_resource_type(results)
                summary = constants.ORCH_FOUND_TOTAL_RESOURCE.format(total=total, resource_type=resource_type)
                
                # Add examples if available
                if results:
                    examples = self._extract_examples(results, max_examples=constants.MAX_EXAMPLES)
                    if examples:
                        examples_str = ', '.join(examples)
                        summary += f". {constants.ORCH_EXAMPLES.format(examples=examples_str)}"
                        if len(results) > constants.MAX_EXAMPLES:
                            summary += constants.ORCH_AND_MORE.format(count=len(results) - constants.MAX_EXAMPLES)
                
                return summary
            
            # Single object responses (API/Database) - ENHANCED
            elif 'id' in data:
                resource_type = self._detect_resource_type([data])
                name = self._extract_name(data)
                if name:
                    return f"✓ Found {resource_type}: {name}"
                else:
                    return f"✓ Successfully retrieved {resource_type} (ID: {data['id']})"
            
            # MongoDB responses
            elif 'inserted_id' in data:
                return f"✓ Successfully inserted document (ID: {data['inserted_id']})"
            elif 'modified_count' in data:
                count = data['modified_count']
                return f"✓ Modified {count} document(s)"
            elif 'deleted_count' in data:
                count = data['deleted_count']
                return f"✓ Deleted {count} document(s)"
            
            # Generic dict response - ENHANCED
            else:
                name = self._extract_name(data)
                if name:
                    return f"✓ Operation completed: {name}"
                else:
                    return "✓ " + constants.ORCH_OPERATION_COMPLETED_SUCCESS
        
        # List responses (API results, Database rows) - ENHANCED
        elif isinstance(data, list):
            if len(data) == 0:
                return "No items found matching your query"
            
            resource_type = self._detect_resource_type(data)
            summary = constants.ORCH_FOUND_LEN_RESOURCE.format(count=len(data), resource_type=resource_type)
            
            # Add examples
            examples = self._extract_examples(data, max_examples=constants.MAX_EXAMPLES)
            if examples:
                examples_str = ', '.join(examples)
                summary += f". {constants.ORCH_EXAMPLES.format(examples=examples_str)}"
                if len(data) > constants.MAX_EXAMPLES:
                    summary += constants.ORCH_AND_MORE.format(count=len(data) - constants.MAX_EXAMPLES)
            
            return summary
        
        # String or other types
        else:
                return constants.ORCH_OPERATION_COMPLETED
    
    def _summarize_result_v2(
        self, 
        result: Dict[str, Any], 
        query: str,
        parsed: Dict[str, Any],
        pagination_info: Dict[str, Any]
    ) -> str:
        """
        Enhanced summary with accurate count handling (v0.3.11).
        
        Fixes Issues #2-4: Accurate count calculations and summaries.
        
        Args:
            result: API result
            query: User query
            parsed: Parsed query data
            pagination_info: Pagination analysis from _analyze_pagination
            
        Returns:
            Accurate human-readable summary
        """
        # Check for errors first
        if result.get('error'):
            return f"{constants.ERROR_PREFIX}{result['error']}"
        
        data = result.get('data')
        
        if not data:
            return constants.ORCH_NO_DATA_RETURNED
        
        total = pagination_info['total_count']
        shown = pagination_info['actual_count']
        has_more = pagination_info['has_more']
        
        # Get resource name
        resource = parsed.get('resource', 'items').replace('_', ' ')
        
        # Generate accurate summary based on counts (Issue #4 fix)
        if total == 0:
            return constants.ORCH_NO_RESOURCE_FOUND.format(resource=resource)
        elif total == 1:
            # Single item
            if isinstance(data, dict) and 'results' in data and data['results']:
                name = self._extract_name(data['results'][0])
            elif isinstance(data, list) and data:
                name = self._extract_name(data[0])
            elif isinstance(data, dict):
                name = self._extract_name(data)
            else:
                name = None
            
            if name:
                return constants.ORCH_FOUND_RESOURCE_NAME.format(resource_singular=resource[:-1] if resource.endswith('s') else resource, name=name)
            else:
                return constants.ORCH_FOUND_ONE_RESOURCE.format(resource_singular=resource[:-1] if resource.endswith('s') else resource)
        elif shown == total:
            # All items shown
            base_summary = constants.ORCH_FOUND_TOTAL_RESOURCE.format(total=total, resource_type=resource)
        elif has_more:
            # Paginated with more available (Issue #2 fix: proper pagination detection)
            base_summary = f"Showing {shown} of {total} {resource} (more available)"
        else:
            # Subset shown, no pagination
            base_summary = f"Showing {shown} of {total} {resource}"
        
        # Enrich summary with concrete examples when possible
        examples_source = None
        if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
            examples_source = data['results']
        elif isinstance(data, list):
            examples_source = data
        
        if examples_source:
            examples = self._extract_examples(examples_source, max_examples=constants.MAX_EXAMPLES)
            if examples:
                examples_str = ', '.join(examples)
                if total > len(examples):
                    base_summary += f". {constants.ORCH_EXAMPLES.format(examples=examples_str)}{constants.ORCH_AND_MORE.format(count=total - len(examples))}"
                else:
                    base_summary += f". {constants.ORCH_EXAMPLES.format(examples=examples_str)}"
        
        return base_summary
    
    def _detect_resource_type(self, results: list) -> str:
        """
        Detect resource type from data structure.
        
        Args:
            results: List of result objects
            
        Returns:
            Human-readable resource type (plural)
        """
        if not results:
            return "items"
        
        first = results[0]
        if not isinstance(first, dict):
            return "items"
        
        # Common patterns for KQSPL and other APIs
        field_patterns = {
            ('service_type', 'service_order'): "service orders",
            ('equipment_type', 'serial_number'): "equipment items",
            ('consumable_type', 'quantity'): "consumables",
            ('inventory_item', 'stock_level'): "inventory items",
            ('email', 'role', 'username'): "users",
            ('first_name', 'last_name', 'email'): "people",
            ('customer_name', 'company'): "customers",
            ('order_number', 'order_date'): "orders",
            ('product_name', 'price'): "products",
            ('invoice_number', 'amount'): "invoices",
            ('transaction_id', 'amount'): "transactions",
        }
        
        first_keys = set(first.keys())
        
        # Check patterns
        for pattern_keys, resource_name in field_patterns.items():
            if any(key in first_keys for key in pattern_keys):
                return resource_name
        
        # Fallback to generic
        return "items"
    
    def _extract_examples(self, results: list, max_examples: int = None) -> list:
        """
        Extract representative names from results.
        
        Args:
            results: List of result objects
            max_examples: Maximum number of examples to extract
            
        Returns:
            List of example strings
        """
        if max_examples is None:
            max_examples = constants.MAX_EXAMPLES
        examples = []
        for item in results[:max_examples]:
            name = self._extract_name(item)
            if name:
                examples.append(name)
        return examples
    
    def _extract_name(self, item: dict) -> str:
        """
        Extract the most relevant name/identifier from an item.
        
        Args:
            item: Result object (dict)
            
        Returns:
            Human-readable identifier or empty string
        """
        if not isinstance(item, dict):
            return ""
        
        # Try common name fields in order of preference
        name_fields = [
            'name', 'display_name', 'title', 'label',
            'email', 'username', 'user_email',
            'serial_number', 'equipment_id', 'consumable_code',
            'order_number', 'invoice_number', 'transaction_id',
            'code', 'sku', 'id'
        ]
        
        for field in name_fields:
            if field in item and item[field]:
                value = str(item[field])
                # Truncate if too long
                if len(value) > 50:
                    value = value[: constants.VALUE_PREVIEW_LENGTH] + "..."
                return value
        
        # Heuristic fallback: any key containing "name" that has a non-empty value
        for key, value in item.items():
            if isinstance(key, str) and 'name' in key.lower() and value:
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[: constants.VALUE_PREVIEW_LENGTH] + "..."
                return value_str
        
        return ""
    
    def _summarize_multi_step_result(self, step_results: List[Dict[str, Any]], query: str) -> str:
        """
        Create human-readable summary for multi-step execution results.
        
        Args:
            step_results: List of results from each step
            query: Original user query
            
        Returns:
            Human-readable summary string
        """
        total_steps = len(step_results)
        successful_steps = sum(1 for sr in step_results if sr.get("status") == "success")
        failed_steps = total_steps - successful_steps
        
        if failed_steps > 0:
            return f"Completed {successful_steps}/{total_steps} steps. {failed_steps} step(s) failed."
        
        # All steps successful
        if total_steps == 1:
            # Single step - use regular summarizer
            result = step_results[0].get("result", {})
            return self._summarize_result({"data": result}, query)
        else:
            # Multi-step - provide aggregate summary
            total_items = 0
            for sr in step_results:
                result_data = sr.get("result")
                if isinstance(result_data, list):
                    total_items += len(result_data)
                elif isinstance(result_data, dict):
                    total_items += 1
            
            return f"✓ Successfully completed {total_steps} steps. Retrieved {total_items} item(s) total."
    
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
    
    def clear_conversation(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier to clear
        """
        self.conversation_store.clear_history(session_id)
        self.logger.info(f"Cleared conversation history for session {session_id}")


def process_query(query: str, access_token: Optional[str] = None, schema: Optional[dict] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience wrapper to process a single query.
    
    Note:
        This creates a new APIOrchestrator instance for each call
        and uses in-memory conversation storage (NOT production safe).
    
    For production, create a single APIOrchestrator instance with
    a persistent ConversationStore implementation.
    """
    orchestrator = APIOrchestrator()
    return orchestrator.process(query, access_token, schema=schema, session_id=session_id)
