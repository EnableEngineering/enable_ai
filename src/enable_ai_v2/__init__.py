"""
Enable AI v2 - Natural Language to API

A minimal, API-driven library that translates natural language
queries into API calls using LLM native tool calling (OpenAI default, Anthropic optional).

Zero hardcoding - all configuration from parent module.

Usage:
    from enable_ai_v2 import Orchestrator, Config, UserContext, JWTAuth

    # Parent module provides all config
    config = Config(
        openapi_schema=schema_dict,  # From GET /api/schema/
        base_url="https://api.example.com",
        resource_hints=hints,  # Built from API data
        status_synonyms=synonyms,
    )

    ai = Orchestrator(config=config, auth=JWTAuth(token="..."))

    # Process with user context
    user_ctx = UserContext(user_id=123, role="Technician")
    result = ai.process("show pending orders", user_context=user_ctx)
"""

from .orchestrator import Orchestrator
from .auth import Auth, JWTAuth, APIKeyAuth, BasicAuth, NoAuth
from .config import (
    Config,
    UserContext,
    ResourceHint,
    QueryExample,
    IntentPhrases,
    DEFAULT_INTENT_PHRASES,
    build_resource_hints_from_api,
    build_status_synonyms,
    apply_filter_guidance,
    build_query_examples_suffix,
    resolve_intent_phrases,
    intent_phrases_from_dict,
)
from .types import (
    Response,
    QueryTrace,
    ConfidenceScore,
    ToolCall,
    ValidationResult,
    Endpoint,
    CachedQuery,
)
from .formatter import ResponseFormatter, FormatType
from .progress import ProgressTracker, ProgressStage, ProgressUpdate
from .tool_converter import convert_spec_to_tools, load_openapi_spec
from .tool_filter import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_ID_CODE_PATTERNS,
    build_filter_query,
    extract_business_codes,
    extract_resource_ids,
    filter_openapi_spec,
    filter_tools_for_query,
    get_max_tools_for_provider,
    has_business_code,
    is_count_query,
    is_read_only_query,
    is_mutation_tool,
)
from .llm_client import LLMClient, ClaudeClient  # ClaudeClient deprecated alias
from .conversation_store import (
    ConversationStore,
    InMemoryConversationStore,
    Message,
)
from .query_intent import (
    QueryIntent,
    classify_query_intent,
    intent_prompt_hint,
    needs_follow_up_round,
    is_limited_list_args,
)
from .tool_args import flatten_tool_arguments, enforce_count_tool_args, build_list_cache
from .follow_up import (
    is_referential_follow_up,
    build_referential_tool_calls,
    extract_list_cache_from_history,
    referential_prompt_hint,
)
from .aggregate import format_aggregate_response
from .date_range import (
    extract_date_range,
    date_range_prompt_hint,
    inject_date_filters,
    inject_temporal_filters,
    extract_duration_filter,
    has_duration_phrase,
)

__version__ = "1.4.0"

__all__ = [
    # Main entry point
    "Orchestrator",
    # Auth
    "Auth",
    "JWTAuth",
    "APIKeyAuth",
    "BasicAuth",
    "NoAuth",
    # Config (parent provides these)
    "Config",
    "UserContext",
    "ResourceHint",
    "QueryExample",
    "IntentPhrases",
    "DEFAULT_INTENT_PHRASES",
    "resolve_intent_phrases",
    "intent_phrases_from_dict",
    # Helper functions for parent
    "build_resource_hints_from_api",
    "build_status_synonyms",
    "apply_filter_guidance",
    "build_query_examples_suffix",
    # Types
    "Response",
    "QueryTrace",
    "ConfidenceScore",
    "ToolCall",
    "ValidationResult",
    "Endpoint",
    "CachedQuery",
    # Formatting
    "ResponseFormatter",
    "FormatType",
    # Progress
    "ProgressTracker",
    "ProgressStage",
    "ProgressUpdate",
    # Utilities
    "convert_spec_to_tools",
    "load_openapi_spec",
    "filter_openapi_spec",
    "filter_tools_for_query",
    "get_max_tools_for_provider",
    "DEFAULT_EXCLUDE_PATTERNS",
    "DEFAULT_ID_CODE_PATTERNS",
    "extract_resource_ids",
    "extract_business_codes",
    "has_business_code",
    "build_filter_query",
    "is_count_query",
    "is_read_only_query",
    "is_mutation_tool",
    # LLM client
    "LLMClient",
    "ClaudeClient",
    # Conversation
    "ConversationStore",
    "InMemoryConversationStore",
    "Message",
    # Query intent
    "QueryIntent",
    "classify_query_intent",
    "intent_prompt_hint",
    "needs_follow_up_round",
    "is_limited_list_args",
    "extract_date_range",
    "date_range_prompt_hint",
    "inject_date_filters",
    "flatten_tool_arguments",
    "enforce_count_tool_args",
    "build_list_cache",
    "is_referential_follow_up",
    "build_referential_tool_calls",
    "extract_list_cache_from_history",
    "format_aggregate_response",
    "inject_temporal_filters",
    "extract_duration_filter",
    "has_duration_phrase",
]
