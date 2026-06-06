"""
Centralized constants for limits and static response strings.
Update these to change behavior or copy without touching business logic.
See todo.md § Limits affecting correctness for impact of limit constants.

Key limits can be overridden by environment variables (see README § Configurable limits):
- ENABLE_AI_SAFETY_MAX_PAGES
- ENABLE_AI_PAGE_SIZE_CAP
- ENABLE_AI_CONVERSATION_HISTORY_LIMIT
- ENABLE_AI_IN_MEMORY_MAX_MESSAGES
- ENABLE_AI_REDIS_MAX_MESSAGES
- ENABLE_AI_REQUEST_TIMEOUT
"""

import os

# -----------------------------------------------------------------------------
# Helper: read int from env with default (invalid/missing -> default)
# -----------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# =============================================================================
# LIMITS (configurable via env; see todo.md § Limits affecting correctness)
# =============================================================================

# Pagination: safety cap to avoid infinite loop on buggy APIs (workflow.py)
SAFETY_MAX_PAGES = _env_int("ENABLE_AI_SAFETY_MAX_PAGES", 500)

# Max page_size when user says "list a few" / limit (orchestrator.py)
PAGE_SIZE_CAP = _env_int("ENABLE_AI_PAGE_SIZE_CAP", 100)

# Default page_size for list queries when user does not specify a limit (config.json / env)
DEFAULT_PAGE_SIZE = _env_int("ENABLE_AI_DEFAULT_PAGE_SIZE", 25)

# Known query/body param names for result limit (orchestrator picks from API schema; fallback if not in schema)
LIMIT_PARAM_NAMES = ("page_size", "limit", "per_page", "pageSize", "size")

# Max examples in summaries (orchestrator._extract_examples)
MAX_EXAMPLES = 3

# Conversation history: messages to load per session (conversation_store get_history)
CONVERSATION_HISTORY_LIMIT = _env_int("ENABLE_AI_CONVERSATION_HISTORY_LIMIT", 10)

# In-memory store: max messages to keep per session (InMemoryConversationStore)
IN_MEMORY_MAX_MESSAGES = _env_int("ENABLE_AI_IN_MEMORY_MAX_MESSAGES", 10)

# Redis store: max messages to keep per session (RedisConversationStore)
REDIS_MAX_MESSAGES = _env_int("ENABLE_AI_REDIS_MAX_MESSAGES", 20)

# Chart: max items in labels/datasets (response_formatter)
CHART_MAX_ITEMS = 20

# LLM max_tokens for format/summary (response_formatter)
MAX_TOKENS_CONCISE = 50
MAX_TOKENS_SUMMARY = 150
# 500 was too low for 25-row tables (25 × ~50 tokens/row ≈ 1250+ tokens)
# so we allow a more realistic detailed budget here.
MAX_TOKENS_DETAILED = 4000

# HTTP and schema fetch timeout (seconds) (api_client, orchestrator, schema_loader)
REQUEST_TIMEOUT = _env_int("ENABLE_AI_REQUEST_TIMEOUT", 30)

# API client retry: max attempts (including first), delay in seconds between attempts (backoff)
REQUEST_RETRY_ATTEMPTS = 3
REQUEST_RETRY_BACKOFF_SECONDS = 1.0

# Suggestions: max number to return (workflow _generate_suggestions)
SUGGESTIONS_MAX = 3

# Truncation for logs/display (utils.truncate_string)
TRUNCATE_MAX_LENGTH = 100

# Log preview length for parsed state etc.
LOG_PREVIEW_LENGTH = 1000

# Query / content preview lengths (logs, auth errors, workflow)
QUERY_PREVIEW_LENGTH = 200
AUTH_ERROR_PREVIEW_LENGTH = 200
LOG_CONTENT_PREVIEW = 200
PARSER_INPUT_PREVIEW = 50

# LLM prompt data sample/preview sizes (response_formatter)
LLM_DATA_SAMPLE_SMALL = 2
LLM_DATA_SAMPLE_MEDIUM = 50
LLM_DATA_PREVIEW_500 = 500
LLM_DATA_PREVIEW_1000 = 1000
LLM_DATA_PREVIEW_2000 = 2000
TABLE_FIELD_PREVIEW = 50
# Max rows in code-built markdown tables (LLM tables are replaced by _format_as_table)
TABLE_ROW_SAMPLE = _env_int("ENABLE_AI_TABLE_ROW_SAMPLE", 50)

# Chat window: rows shown per assistant turn (summary display_mode)
CHAT_WINDOW_SIZE = _env_int("ENABLE_AI_CHAT_WINDOW_SIZE", 10)

# Max items to retain in session list_cache (fetch pages until cap)
LIST_CACHE_MAX_ITEMS = _env_int("ENABLE_AI_LIST_CACHE_MAX", 500)
# Use deterministic table format for list responses (1=always table for lists, 0=LLM chooses)
LIST_FORMAT_TABLE = _env_int("ENABLE_AI_LIST_FORMAT_TABLE", 1)
TABLE_FIELDS_MAX = 6
ANALYSIS_FIELDS_MAX = 10
GROUPED_ITEMS_SAMPLE = 3

# utils.py truncation for LLM content / tool desc
UTILS_CONTENT_TRUNCATE = 500
UTILS_TOOL_DESC_PREVIEW = 100
UTILS_CONTENT_DEBUG = 250

# Orchestrator _extract_name value preview
VALUE_PREVIEW_LENGTH = 47

# Redis conversation store default expiry (7 days, in seconds)
REDIS_EXPIRY_SECONDS_DEFAULT = 7 * 24 * 60 * 60

# Auth: default expires_in when API omits it (seconds); buffer subtracted for refresh (orchestrator)
AUTH_EXPIRES_IN_DEFAULT = 3600
AUTH_TOKEN_BUFFER_SECONDS = 60

# Schema loader: default refresh interval in example config (seconds)
SCHEMA_REFRESH_INTERVAL_DEFAULT = 3600

# Response formatter: list length range for "medium sample" hint (6–50 items)
MIN_LIST_LENGTH_MEDIUM_SAMPLE = 6

# Response formatter: data is "simple" (no AI) only when trivial (see _is_simple_data)
SIMPLE_DICT_MAX_KEYS = 3
SIMPLE_LIST_MAX_PRIMITIVE_ITEMS = 10


# =============================================================================
# STRINGS – Errors (workflow, format_error, schema)
# =============================================================================

ERROR_PREFIX = "Error: "
ERROR_UNKNOWN = "Unknown error"
ERROR_NO_SCHEMA = "No schema provided. Pass schema at init or runtime."
ERROR_ONLY_API_SCHEMAS = "Only API schemas are supported in the current version."
ERROR_NO_RESPONSE_GENERATED = "No response generated"
ERROR_OCCURRED = "Error occurred"
ERROR_QUERY_REQUIRED = "Error: 'query' parameter is required"
ERROR_SCHEMA_TYPE_REQUIRED = "Error: 'schema_type' parameter is required"
ERROR_ONLY_API_SCHEMA_SUPPORTED = "Error: Only 'api' schema_type is supported in the current version."
ERROR_NO_API_SCHEMA_LOADED = "Error: No API schema loaded. Cannot authenticate."
ERROR_NO_AUTH_CONFIGURED = "Error: No authentication method configured in config.json"
ERROR_NO_BASE_URL = "Error: No base_url configured"
ERROR_NO_RESOURCES_IN_SCHEMA = "No resources found in API schema"
ERROR_NO_MATCHING_API = "No matching API found for intent: {intent}, resource: {resource}"
ERROR_BASE_URL_NOT_CONFIGURED = "API base_url not configured"
ERROR_AUTH_NO_TOKEN_RECEIVED = "Authentication succeeded but no token received"

# LLM / init (surfaced as-is; no generic fallbacks that mask real errors)
ERROR_OPENAI_KEY_MISSING = (
    "OPENAI_API_KEY not set. Set it in .env (OPENAI_API_KEY=sk-...) or export it in your shell."
)


# =============================================================================
# STRINGS – Suggestions (workflow _generate_suggestions)
# =============================================================================

SUGGESTION_TRY_BROADER = "Try a broader search or different criteria"
SUGGESTION_SHOW_DETAILS = "Say 'show details' for more information"
SUGGESTION_SHOW_DETAILS_ITEM = "Say 'show details on item X' for more info"
SUGGESTION_FILTER_SPECIFIC = "Say 'show only [specific name]' to filter results"
SUGGESTION_NARROW_FILTERS = "Try narrowing down by adding specific filters"
SUGGESTION_SHOW_MORE = "Say 'show more' to see remaining items"


# =============================================================================
# STRINGS – Workflow summaries / progress
# =============================================================================

# Template summaries (workflow); see todo.md § Summarization & presentation – goal is question-aware
# summarization and smart format (table/chart/text) instead of these boilerplate strings.
SUMMARY_RETRIEVED_ALL_ITEMS_STEPS = "Retrieved all {count} item(s) from {steps} step(s) as requested"
SUMMARY_RETRIEVED_COUNT_OF_TOTAL = "Retrieved {count} of {total} item(s) (showing current page)"
SUMMARY_RETRIEVED_ALL_COUNT = "Retrieved all {count} item(s) as requested"
SUMMARY_RETRIEVED_ALL_LEN = "Retrieved all {count} item(s) as requested"
SUMMARY_RETRIEVED_DATA = "Retrieved data as requested"
PROGRESS_DONE = "Done! ✓"
PROGRESS_ERROR_OCCURRED = "Error occurred"
PAGINATION_SAFETY_CAP_WARNING = "Automatic pagination hit safety cap ({max_pages} pages); more pages may exist"
PAGINATION_STEP_SAFETY_CAP_WARNING = "Pagination-as-step hit safety cap ({max_pages} pages); more pages may exist"
PROGRESS_FETCHING_PAGE = "Fetching page {page}..."


# =============================================================================
# STRINGS – Progress tracker (progress_tracker.create_progress_message)
# =============================================================================

PROGRESS_STARTED = "Starting your request..."
PROGRESS_PARSING_QUERY = "Understanding your question..."
PROGRESS_INTENT_DETECTED = "Intent identified: {intent}"
PROGRESS_MATCHING_API = "Finding the right API..."
PROGRESS_API_MATCHED = "Found API: {api_name}"
PROGRESS_PLANNING = "Planning execution steps..."
PROGRESS_PLAN_READY = "Plan ready: {step_count} steps"
PROGRESS_EXECUTING_API = "Calling {api_name}..."
PROGRESS_API_COMPLETED = "API call completed: {api_name}"
PROGRESS_SUMMARIZING = "Preparing your results..."
PROGRESS_COMPLETED = "Done! ✓"
PROGRESS_ERROR = "Error: {error}"
PROGRESS_DEFAULT = "Processing..."

# Minimum ms to show each progress stage so frontend can display it (0 = no delay; set e.g. 400 to avoid stages flashing by)
PROGRESS_MIN_DISPLAY_MS = _env_int("ENABLE_AI_PROGRESS_MIN_DISPLAY_MS", 0)


# =============================================================================
# STRINGS – Response formatter
# =============================================================================

FORMAT_NO_DATA_RETURNED = "No data returned"
FORMAT_FOUND_ITEMS = "Found {count} items"
FORMAT_FOUND_ITEMS_CATEGORIES = "Found {count} items across {categories} categories"
FORMAT_FOUND_ITEMS_PLURAL = "Found {count} item(s)"
FORMAT_RETRIEVED_ONE_ITEM = "Retrieved 1 item with {count} field(s)"
FORMAT_OPERATION_COMPLETED_SUCCESS = "Operation completed successfully"


# =============================================================================
# STRINGS – Orchestrator summaries
# =============================================================================

ORCH_NO_ITEMS_FOUND = "No items found matching your query"
ORCH_NO_DATA_RETURNED = "No data returned"
ORCH_NO_ROWS_AFFECTED = "No rows affected"
ORCH_FOUND_TOTAL_RESOURCE = "Found {total} {resource_type}"
ORCH_FOUND_LEN_RESOURCE = "Found {count} {resource_type}"
ORCH_FOUND_ONE_RESOURCE = "Found 1 {resource_singular}"
ORCH_FOUND_RESOURCE_NAME = "Found {resource_singular}: {name}"
ORCH_AND_MORE = ", and {count} more"
ORCH_EXAMPLES = "Examples: {examples}"
ORCH_NO_RESOURCE_FOUND = "No {resource} found matching your query"
ORCH_SUCCESS_ONE_ROW = "Successfully affected 1 row"
ORCH_OPERATION_COMPLETED = "Operation completed"
ORCH_OPERATION_COMPLETED_SUCCESS = "Operation completed successfully"


# =============================================================================
# STRINGS – API matcher (missing info)
# =============================================================================

API_MATCHER_NEED_INFO_PREFIX = "I need some additional information:\n"


# =============================================================================
# STRINGS – MCP server
# =============================================================================

MCP_ERROR_TOOL_EXEC = "Error executing tool '{name}': {error}"
MCP_AUTH_SUCCESS_MESSAGE = "Authentication successful. Use this token with process_query."


# =============================================================================
# STRINGS – API client / orchestrator (errors)
# =============================================================================

API_FAILED_CONNECT = "Failed to connect to API at {base_url}"
API_TIMEOUT_MESSAGE = "API request timed out after {timeout} seconds"
API_REQUEST_TIMEOUT = "Request timed out"


# =============================================================================
# STRINGS – Unknown Intent Handling (v0.3.29 → v0.3.68 schema-driven)
# =============================================================================

# Generic intro when query cannot be understood — no hardcoded domain examples
UNKNOWN_INTENT_INTRO = "I couldn't understand that query."

# No matching resource found — available resources filled dynamically from schema
NO_MATCHING_RESOURCE = "I couldn't find a matching resource for '{resource}'. Available resources: {available}"

# No results found (but query was understood)
NO_RESULTS_FOUND = "No {resource} found matching your criteria."
NO_RESULTS_SUGGESTION = "Try broadening your search or using different filters."


# =============================================================================
# INTENT CLASSIFIER (rule-based classification)
# =============================================================================

# Confidence thresholds for intent classification
CLASSIFIER_HIGH_CONFIDENCE = 0.85
CLASSIFIER_MEDIUM_CONFIDENCE = 0.65
CLASSIFIER_LOW_CONFIDENCE = 0.40

# Intent keywords for rule-based classification (domain-agnostic)
CLASSIFIER_READ_KEYWORDS = frozenset({
    'list', 'show', 'get', 'find', 'search', 'fetch', 'display', 'view',
    'what', 'which', 'who', 'retrieve', 'see', 'look'
})
CLASSIFIER_CREATE_KEYWORDS = frozenset({
    'create', 'add', 'new', 'insert', 'make', 'register', 'post'
})
CLASSIFIER_UPDATE_KEYWORDS = frozenset({
    'update', 'modify', 'change', 'edit', 'set', 'patch', 'alter'
})
CLASSIFIER_DELETE_KEYWORDS = frozenset({
    'delete', 'remove', 'drop', 'cancel', 'destroy'
})
CLASSIFIER_COUNT_KEYWORDS = frozenset({
    'how many', 'count', 'total', 'number of', 'how much'
})


# =============================================================================
# CONVERSATION LOOP (confirmation and correction)
# =============================================================================

# Confidence threshold below which we ask for confirmation
CONFIRMATION_CONFIDENCE_THRESHOLD = 0.65

# User responses that mean "yes" (case-insensitive)
CONFIRMATION_YES_RESPONSES = frozenset({
    'yes', 'correct', 'y', 'yep', 'yeah', 'right', 'that\'s right',
    'yes that\'s correct', 'exactly', 'proceed', 'go ahead'
})

# User responses that mean "no" (case-insensitive)
CONFIRMATION_NO_RESPONSES = frozenset({
    'no', 'wrong', 'incorrect', 'n', 'nope', 'not quite', 'no that\'s wrong'
})

# Confirmation message templates
CONFIRMATION_MESSAGE_TEMPLATE = "I understood: {interpretation}. Is that correct?"
CONFIRMATION_CLARIFY_MESSAGE = "What did you mean instead?"


# =============================================================================
# CORRECTION STORE (learning from user corrections)
# =============================================================================

# Max corrections to store per session (in-memory store)
CORRECTION_STORE_MAX_PER_SESSION = 100

# Number of similar corrections to retrieve for context
CORRECTION_SIMILAR_LIMIT = 5

# Minimum times a correction must occur to be considered a learned pattern
CORRECTION_PATTERN_MIN_FREQUENCY = 2


# =============================================================================
# SCHEMA INTROSPECTOR (filter discovery)
# =============================================================================

# Default filter operators supported
DEFAULT_FILTER_OPERATORS = (
    'equals', 'not_equals', 'contains', 'starts_with', 'ends_with',
    'gt', 'gte', 'lt', 'lte', 'in', 'not_in'
)

# Django-style lookup suffixes to try when mapping filters
DJANGO_LOOKUP_SUFFIXES = (
    '__name', '__id', '__iexact', '__icontains', '__in',
    '__gte', '__lte', '__gt', '__lt'
)

# Semantic role injection moved to resource_hints (v0.3.68)
# Configure users.role.synonyms in your schema's resource_hints instead of hardcoding here.


# =============================================================================
# PAGINATION INFO (response formatting)
# =============================================================================

# Template for pagination info in responses
PAGINATION_INFO_TEMPLATE = "Showing {shown} of {total} {resource}."
PAGINATION_HAS_MORE_TEMPLATE = "Say 'show more' or 'next page' to see more."
PAGINATION_SHOW_ALL_HINT = "Say 'show all' to see the complete list."

# Empty response templates
EMPTY_RESPONSE_MESSAGE = "No {resource} found matching your criteria."
EMPTY_RESPONSE_FILTERS_APPLIED = "Filters applied:"
EMPTY_RESPONSE_SUGGESTION = "Try broadening your search or using different filters."
EMPTY_RESPONSE_VALID_VALUES_HINT = "Valid values for '{field}': {values}"

# When filter_warnings are present (filter may not be supported by API)
FILTER_WARNING_NOTE = "Results might not be filtered as requested if the API does not support these filters."
