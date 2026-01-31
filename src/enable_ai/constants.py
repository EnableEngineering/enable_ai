"""
Centralized constants for limits and static response strings.
Update these to change behavior or copy without touching business logic.
See todo.md § Limits affecting correctness for impact of limit constants.
"""

# =============================================================================
# LIMITS (configurable; see todo.md § Limits affecting correctness)
# =============================================================================

# Pagination: safety cap to avoid infinite loop on buggy APIs (workflow.py)
SAFETY_MAX_PAGES = 500

# Max page_size when user says "list a few" / limit (orchestrator.py)
PAGE_SIZE_CAP = 100

# Max examples in summaries (orchestrator._extract_examples)
MAX_EXAMPLES = 3

# Conversation history: messages to load per session (conversation_store get_history)
CONVERSATION_HISTORY_LIMIT = 10

# In-memory store: max messages to keep per session (InMemoryConversationStore)
IN_MEMORY_MAX_MESSAGES = 10

# Redis store: max messages to keep per session (RedisConversationStore)
REDIS_MAX_MESSAGES = 20

# Chart: max items in labels/datasets (response_formatter)
CHART_MAX_ITEMS = 20

# LLM max_tokens for format/summary (response_formatter)
MAX_TOKENS_CONCISE = 50
MAX_TOKENS_SUMMARY = 150
MAX_TOKENS_DETAILED = 500

# HTTP and schema fetch timeout (seconds) (api_client, orchestrator, schema_loader)
REQUEST_TIMEOUT = 30

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
TABLE_ROW_SAMPLE = 20
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
