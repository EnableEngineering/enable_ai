# Enable AI

Natural language to REST API translation using LLM tool calling.

**Zero hardcoding** - works with any OpenAPI spec. Parent module provides all configuration.

## Installation

```bash
pip install enable-ai

# For Anthropic/Claude support:
pip install enable-ai[anthropic]
```

## Quick Start

```python
from enable_ai_v2 import Orchestrator, Config, JWTAuth

config = Config(
    openapi_schema="path/to/openapi.json",  # or dict
    base_url="https://api.example.com",
    # llm_provider="openai",  # default
    # llm_provider="anthropic",  # optional
)

ai = Orchestrator(
    config=config,
    auth=JWTAuth(token="your-jwt-token"),
)

result = ai.process("show all pending orders")
print(result.message)
```

## Features

- **LLM tool calling** - OpenAPI spec converted to LLM tools (OpenAI default, Anthropic optional)
- **Server-side scoping** - doesn't inject user filters; server's `get_queryset()` handles permissions
- **Resource hints** - status values, enums from your API for better accuracy
- **Status synonyms** - map natural language ("pending") to actual values ("DRAFT")
- **Query caching** - exact match + pattern match for fast repeat queries
- **Progress tracking** - real-time callbacks for UI updates

## Configuration

Parent module provides everything:

```python
from enable_ai_v2 import (
    Config,
    ResourceHint,
    UserContext,
    build_resource_hints_from_api,
    build_status_synonyms,
)

# Fetch from your APIs (parent module's responsibility)
openapi_schema = your_api.get_schema()
statuses = your_api.get_service_order_statuses()
priorities = your_api.get_service_order_priorities()

config = Config(
    # Required
    openapi_schema=openapi_schema,
    base_url="https://api.example.com",

    # Recommended (for accuracy)
    resource_hints=build_resource_hints_from_api(
        openapi_schema,
        statuses,
        priorities,
    ),
    status_synonyms=build_status_synonyms(statuses),

    # Optional
    # Optional
    model="gpt-4o",  # default; or "claude-sonnet-4-20250514" with llm_provider="anthropic"
    temperature=0.0,
    cache_enabled=True,
    include_trace=False,  # True for debugging
)
```

## User Context

Parent module provides user context (doesn't fetch from APIs):

```python
from enable_ai_v2 import UserContext

user_ctx = UserContext(
    user_id=123,
    username="john.doe",
    role="Technician",
    company_id=456,
    is_admin=False,
)

result = ai.process("show my orders", user_context=user_ctx)
```

## Authentication

```python
from enable_ai_v2 import JWTAuth, APIKeyAuth, BasicAuth, NoAuth

# JWT (most common)
auth = JWTAuth(token="your-jwt-token")

# API Key
auth = APIKeyAuth(api_key="your-api-key", header_name="X-API-Key")

# Basic Auth
auth = BasicAuth(username="user", password="pass")

# No auth
auth = NoAuth()

ai = Orchestrator(config=config, auth=auth)
```

## Progress Tracking

```python
def on_progress(message: str, progress: float):
    print(f"[{int(progress*100)}%] {message}")

config = Config(
    openapi_schema=schema,
    base_url="https://api.example.com",
    progress_callback=on_progress,
)
```

## Response

```python
result = ai.process("show pending orders")

result.message   # Natural language response
result.data      # Raw API data
result.success   # True/False
result.trace     # Debug info (if include_trace=True)
```

## Key Principle

**Server handles data scoping.** The system prompt instructs Claude:

> Do NOT add user ID filters (like technician=, customer=, created_by=) unless the user explicitly asks to filter by a specific person.

Your backend's `get_queryset()` permissions handle which data each user sees.

## Requirements

- Python 3.9+
- `OPENAI_API_KEY` environment variable (default provider)
- Or `ANTHROPIC_API_KEY` if using `llm_provider="anthropic"`

## License

MIT
