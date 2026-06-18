# Enable AI v2 - Context & Architecture

## Overview

**enable-ai** is a Python library that translates natural language queries into REST API calls using LLM tool calling.

- **Version**: 1.0.9
- **Default LLM**: OpenAI (gpt-4o) - cheaper
- **Optional LLM**: Anthropic (Claude) - `pip install enable-ai[anthropic]`
- **PyPI**: https://pypi.org/project/enable-ai/

## Core Principle

**Zero hardcoding** - Parent module (e.g., KQSPL) provides all configuration:
- OpenAPI schema
- Resource hints (status values, enums)
- Status synonyms
- User context

**Server handles data scoping** - enable-ai does NOT inject user filters (like `technician=`, `customer=`). The server's `get_queryset()` permissions handle which data each user sees.

---

## Installation

```bash
pip install enable-ai

# For Anthropic/Claude support:
pip install enable-ai[anthropic]
```

## Environment Variables

```bash
# Default provider (OpenAI)
export OPENAI_API_KEY="sk-..."

# If using Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Architecture

```
Parent Module (KQSPL)
    │
    ├── Fetches from APIs:
    │   ├── GET /api/schema/                    → openapi_schema
    │   ├── GET /api/service-orders/statuses/   → status values
    │   └── GET /api/service-orders/priorities/ → priority values
    │
    ├── Builds Config with:
    │   ├── resource_hints (from API data)
    │   └── status_synonyms (from API data)
    │
    └── Calls enable_ai_v2.Orchestrator.process()
            │
            ├── LLMClient (OpenAI or Anthropic)
            │   ├── Filters tools by query relevance
            │   ├── Calls LLM with tool definitions
            │   └── Handles truncation/continuation
            │
            ├── APIClient
            │   └── Executes selected tool calls
            │
            └── ResponseFormatter
                └── Formats API results for chat
```

---

## Key Components

### Config

```python
from enable_ai_v2 import Config, ResourceHint

config = Config(
    # Required
    openapi_schema=schema_dict,  # or path to file
    base_url="https://api.example.com",

    # Recommended (improves accuracy)
    resource_hints={
        "service-orders": ResourceHint(
            status_field="status",
            status_values=["New", "InProgress", "Completed"],
        ),
    },
    status_synonyms={
        "open": "New",
        "pending": "New",
        "done": "Completed",
    },

    # Optional
    llm_provider="openai",  # or "anthropic"
    model="gpt-4o",
    max_tokens=16384,
    temperature=0.0,
    include_trace=False,

    # Tool filtering (optional - parent/KQSPL decides)
    tool_exclude_patterns=[r"bulk", r"admin"],  # or None
    allowed_http_methods=["GET"],  # read-only users; None = all methods
    filter_tools_for_non_admin=True,
    tool_filter_enabled=True,  # query-time filtering (shared all providers)
    max_tools_per_query=None,  # override 128/200 defaults
)
```

### Orchestrator

```python
from enable_ai_v2 import Orchestrator, JWTAuth, UserContext

ai = Orchestrator(
    config=config,
    auth=JWTAuth(token="user-jwt-token"),
)

user_ctx = UserContext(
    user_id=123,
    role="Technician",
    is_admin=False,
)

result = ai.process("show my open orders", user_context=user_ctx)
print(result.message)  # Natural language response
print(result.data)     # Raw API data
print(result.success)  # True/False
```

### LLMClient

Unified client supporting provider swap:

```python
# Automatically selected based on config.llm_provider
# - OpenAI: 128 tool limit, filters to most relevant
# - Anthropic: 200 tool limit, filters to most relevant

# Tool filtering uses:
# - Keyword matching (query words → tool name/description)
# - Action word mapping (list/show → GET, create → POST)
```

---

## Response Structure

```python
@dataclass
class Response:
    message: str           # Natural language response
    data: Any             # Raw API data (dict or list)
    success: bool         # True if API call succeeded
    suggestions: list     # Follow-up suggestions
    trace: QueryTrace     # Debug info (if include_trace=True)
```

---

## System Prompt

The system prompt instructs the LLM:

```
You are an API assistant that helps users interact with a REST API.

CRITICAL RULES:
IMPORTANT: The server automatically scopes data based on user permissions.
Do NOT add user ID filters (like technician=, customer=, created_by=) unless
the user explicitly asks to filter by a specific person.
Only add filters the user explicitly requests (e.g., 'SENT invoices' → status=SENT).

Available resources and their valid filter values:
- service-orders: status=New, InProgress, Completed

Status synonyms (use the value on the right):
- "pending" → New
- "done" → Completed
```

---

## Helper Functions

For parent module to build config from API data:

```python
from enable_ai_v2 import (
    build_resource_hints_from_api,
    build_status_synonyms,
)

# Fetch from your APIs
statuses = requests.get(f"{base_url}/api/service-orders/service-order-statuses/").json()
priorities = requests.get(f"{base_url}/api/service-orders/service-order-priorities/").json()

# Build hints
resource_hints = build_resource_hints_from_api(
    openapi_schema,
    statuses,
    priorities,
)

status_synonyms = build_status_synonyms(statuses)
```

---

## Authentication

```python
from enable_ai_v2 import JWTAuth, APIKeyAuth, BasicAuth, NoAuth

# JWT (most common)
auth = JWTAuth(token="your-jwt-token")

# API Key
auth = APIKeyAuth(api_key="key", header_name="X-API-Key")

# Basic Auth
auth = BasicAuth(username="user", password="pass")

# No auth
auth = NoAuth()
```

---

## Progress Tracking

```python
def on_progress(message: str, progress: float):
    print(f"[{int(progress*100)}%] {message}")

config = Config(
    ...,
    progress_callback=on_progress,
)
```

---

## Tool Filtering

Two layers — parent (KQSPL) chooses spec filtering; enable-ai always does query filtering.

### 1. Parent-side spec filtering (optional)

**Ask KQSPL:** Does your OpenAPI spec need pre-filtering before passing to enable-ai?

Use this when you have many endpoints (bulk, admin, webhooks) that most users never query.

**Option A — pre-filter before Config:**

```python
from enable_ai_v2 import filter_openapi_spec

schema = filter_openapi_spec(
    raw_schema,
    exclude_patterns=[r"bulk", r"admin", r"webhook"],
    allowed_methods=["GET"],  # read-only users
    is_admin=user.is_staff,
)

config = Config(openapi_schema=schema, base_url=base_url)
```

**Option B — let Config filter per request:**

```python
config = Config(
    openapi_schema=raw_schema,
    base_url=base_url,
    tool_exclude_patterns=[r"bulk", r"admin"],  # or None
    allowed_http_methods=["GET"] if not user.is_staff else None,
    filter_tools_for_non_admin=True,  # applies DEFAULT_EXCLUDE_PATTERNS for non-admin
)
```

### 2. Query-time filtering (automatic, all LLM providers)

Shared code in `tool_filter.py` — used by OpenAI and Anthropic:

- Keyword matching (query words → tool name/description)
- Normalized resource matching (`service-orders` ↔ `service_orders`)
- Action mapping (list/show → GET, create → POST, etc.)
- OpenAI: 128 tools max, Anthropic: 200 tools max (configurable via `max_tools_per_query`)

Disable with `tool_filter_enabled=False` only if you pre-filtered the spec heavily.

---

## Truncation Handling

- `max_tokens` default: 16384 (high to avoid truncation)
- If output truncated (`finish_reason=length`), automatically continues
- For large API responses, formatter truncates to 2000 chars in LLM context

---

## DRF Pagination Handling

Automatically unwraps DRF paginated responses:

```json
{
  "count": 136,
  "next": "http://...",
  "results": [...]
}
```

Becomes: "Found 136 results: ..." (not "Count: 136, Next: ...")

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0.9 | SO-code search vs retrieve, validator blocks id=169 for SO-169, naming cleanup, trace api_calls |
| 1.0.8 | llm_reasoning, CALLING_LLM, get_llm_messages (provider-agnostic naming) |
| 1.0.7 | Formatter fixes: label from tool/config, pluralization, paginated count handling |
| 1.0.6 | Fix duplicate enrich bug, conversation history, ID/count tool selection, multi-API formatting |
| 1.0.5 | Shared tool_filter module, parent-side spec filtering config, pytest fixes |
| 1.0.4 | Intelligent tool filtering for both OpenAI (128) and Anthropic (200) |
| 1.0.3 | Type hint fixes for pyright |
| 1.0.2 | OpenAI default, Anthropic optional, truncation handling |
| 1.0.1 | Return Claude's clarification text, unwrap DRF pagination |
| 1.0.0 | v2 only release, removed v1 |

---

## File Structure

```
src/enable_ai_v2/
├── __init__.py          # Public exports
├── orchestrator.py      # Main entry point
├── llm_client.py        # Unified LLM client (OpenAI/Anthropic)
├── config.py            # Config, UserContext, ResourceHint
├── tool_converter.py    # OpenAPI → LLM tools
├── tool_filter.py       # Shared spec + query tool filtering
├── api_client.py        # HTTP client for API calls
├── auth.py              # Auth handlers (JWT, APIKey, Basic)
├── formatter.py         # Response formatting
├── validator.py         # Tool call validation
├── query_cache.py       # Query caching
├── progress.py          # Progress tracking
├── types.py             # Type definitions
├── conversation_store.py # Conversation history (Django, Redis, InMemory)
└── mcp_server.py        # MCP server support
```

---

## KQSPL Integration

```python
from enable_ai_v2 import (
    Orchestrator,
    Config,
    JWTAuth,
    UserContext,
    build_resource_hints_from_api,
    build_status_synonyms,
)

# 1. Fetch from APIs
schema = requests.get(f"{base_url}/api/schema/").json()
statuses = requests.get(f"{base_url}/api/service-orders/service-order-statuses/").json()
priorities = requests.get(f"{base_url}/api/service-orders/service-order-priorities/").json()

# 2. Build config
config = Config(
    openapi_schema=schema,
    base_url=base_url,
    resource_hints=build_resource_hints_from_api(schema, statuses, priorities),
    status_synonyms=build_status_synonyms(statuses),
    llm_provider="openai",  # cheaper
)

# 3. Initialize orchestrator with user's JWT
ai = Orchestrator(config=config, auth=JWTAuth(token=request.user.jwt))

# 4. Process query with user context
user_ctx = UserContext(
    user_id=request.user.id,
    role=request.user.role,
    is_admin=request.user.is_staff,
)

result = ai.process(query, user_context=user_ctx)
return {"message": result.message, "data": result.data}
```

---

## Key Design Decisions

1. **Server handles scoping**: Don't inject `technician=` filters. Server's `get_queryset()` uses `has_module_permission()` to scope data.

2. **Parent provides all config**: enable-ai doesn't fetch anything. Parent fetches statuses, priorities, roles from its APIs.

3. **OpenAI default**: Cheaper than Anthropic, same tool calling capability.

4. **Tool filtering**: Reduces token usage and improves accuracy by sending only relevant tools.

5. **No hardcoded role mappings**: Removed `Technician → technician=` logic. Server permissions handle this.

6. **Clarifications are valid**: When LLM asks "which service order?", return that text (not generic fallback).
