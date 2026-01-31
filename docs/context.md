# Enable AI - Complete Context

**For v1 remaining work and checklist:** see **[todo.md](todo.md)**. **For hardcoded limits and static strings:** all are in **`constants.py`** (update there as needed); impact described in **todo.md** § Limits affecting correctness.

---

## Project Overview

**Enable AI** is a Python pip module that provides a natural language interface for REST APIs using AI-powered orchestration.

### Core Purpose
- Backend applications send user queries from frontend to this module
- Module processes natural language queries and executes appropriate API calls
- Uses LangGraph for state management and multi-step workflows
- Powered by OpenAI for intent understanding and planning
- Exposes functionality via MCP (Model Context Protocol) server

### Scope
**Current:** API orchestration only  
**Future:** Database and document analysis as optional extensions

---

## Key Features

### 1. Works with any API documentation
- **Feed OpenAPI/Swagger** (file path, URL, or dict) and a base URL
- Module auto-converts to internal schema format (or use CLI: `enable-schema generate --input openapi.json --base-url https://api.example.com --output api_schema.json`)
- User asks in natural language → parser extracts intent → matcher finds endpoint → client calls API → response formatted for the user
- No hardcoded endpoints; the same flow works for any documented API

### 2. Natural Language Query Processing
- Users ask questions in plain English: "list all users", "get user with id 5"
- LLM-based parser extracts intent, resources, and parameters
- No need to know API endpoints or structure

### 3. Multi-Step API Orchestration
- **ExecutionPlanner** creates execution plans for complex queries (see `execution_planner.py`) with `steps`, `depends_on`, and `extract` (e.g. `{"user_id": "$.id"}`)
- Workflow runs steps one-by-one; `_resolve_step_dependencies` in `workflow.py` applies each step's `extract` (via `_extract_by_path`) to previous step results so `{user_id}` etc. are set correctly
- Conversation state is in `conversation_store` (no LangGraph checkpointer); LangGraph state is ephemeral per `invoke()`

### 4. MCP Server Integration
- Exposes functionality as MCP tools for AI assistants
- Provides: query processing, authentication, schema management
- Easy integration with Claude Desktop and other MCP clients

### 5. Centralized Architecture
- **OpenAI calls** centralized in `utils.py` for easy model updates
- **Comprehensive logging** throughout all components
- **Single source of truth** for configuration and behavior

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     APIOrchestrator                          │
│  (Main interface - renamed from NLPProcessor)                │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─► Parser (LLM-based query understanding)
             ├─► APIMatcher (Endpoint matching)
             ├─► Planner (Multi-step execution planning)
             ├─► APIClient (HTTP requests)
             └─► LangGraph Workflow (State management)
```

### Workflow (LangGraph)

```
User Query
    ↓
Load Schema → Parse Query → Create Plan → Execute API → Summarize
    ↓             ↓              ↓             ↓            ↓
Schema OK?   Intent found?   Multi-step?   API call OK?   Result ready
```

### File Structure

```
enable_ai/
├── src/enable_ai/
│   ├── constants.py          # Limits and static strings (see todo.md § Limits)
│   ├── utils.py              # Centralized OpenAI & logging
│   ├── orchestrator.py       # APIOrchestrator (main class)
│   ├── query_parser.py       # LLM-based parser
│   ├── execution_planner.py  # Multi-step planner
│   ├── api_matcher.py        # Endpoint matcher
│   ├── api_client.py         # HTTP client
│   ├── workflow.py           # LangGraph workflow (execute_step, summarize, pagination detection)
│   ├── response_formatter.py # Summarization / display_mode
│   ├── progress_tracker.py   # Progress stages
│   ├── conversation_store.py # InMemory / Django / Redis stores
│   ├── schema_loader.py     # Load schema from file/URL; OpenAPI conversion
│   ├── schema_validator.py   # Schema validation
│   ├── types.py              # Type definitions
│   ├── config_loader.py      # config.json loading
│   ├── mcp_server.py        # MCP server
│   └── schema_generator/     # OpenAPI → API schema
│       ├── schema_converter.py
│       ├── cli.py
│       └── README.md
│
├── examples/
│   ├── README.md
│   ├── simple_usage.py
│   ├── streaming_backend.py
│   └── streaming_frontend.html
│
├── tests/
└── docs/                     # This folder (context.md, todo.md)
```

---

## What's Been Done

### Phase 1: Initial Setup ✅
- Cloned project from GitHub
- Renamed from `nlp-api-caller` to `enable-ai`
- Reorganized project structure (flattened nested folders)
- Moved all tests to `tests/` folder
- Moved all docs to `docs/` folder

### Phase 2: API-Only Focus ✅
**Removed ~2,085 lines of unused code:**
- Database execution methods from `orchestrator.py` (605 lines)
- `database_inspector.py`, `pdf_analyzer.py`, `json_analyzer.py` (1,050 lines)
- Simplified `schema_generator/cli.py` (188 lines)
- Cleaned up documentation (242 lines)

**Result:** Clean, focused codebase

### Phase 3: Centralized OpenAI ✅
**Created `utils.py` with:**
- `OpenAIClient` class (singleton pattern)
- `get_openai_client()` function
- `setup_logger()` for consistent logging
- Model configuration constants (DETERMINISTIC_TEMP, CREATIVE_TEMP, etc.)

**Updated all components to use centralized OpenAI:**
- `query_parser.py` - Uses `get_openai_client()`
- `execution_planner.py` - Uses `get_openai_client()`
- `workflow.py` - Uses centralized utilities

**Benefits:**
- Change OpenAI model in ONE place
- Automatic logging of all LLM calls
- Token usage tracking
- Consistent error handling

### Phase 4: Comprehensive Logging ✅
**Added logging to all components:**
- `orchestrator.py` (APIOrchestrator) - Full workflow logging
- `query_parser.py` - Query parsing, LLM calls
- `execution_planner.py` - Plan creation, multi-step coordination
- `api_matcher.py` - Endpoint matching
- `api_client.py` - HTTP requests/responses
- `workflow.py` - LangGraph state transitions

**Log format:**
```
2026-01-23 15:30:45 - enable_ai.parser - INFO - Parsing query: 'list all users'
2026-01-23 15:30:46 - enable_ai.openai - INFO - OpenAI response: tokens=150
2026-01-23 15:30:47 - enable_ai.orchestrator - INFO - API response: status=200
```

### Phase 5: Renamed Core Components ✅
- `NLPProcessor` → `APIOrchestrator` (no alias; use APIOrchestrator everywhere)
- Updated all imports and references

### Phase 6: Project Cleanup ✅
**Organized structure:**
- Source code: `src/`
- Tests: `tests/`
- Documentation: `docs/`
- Examples: `examples/`

**Created missing files:**
- `.gitignore` - Proper Python gitignore
- `tests/__init__.py` - Make tests a package
- README files for subdirectories

**Removed:**
- Nested `enable_ai/enable_ai/` folder
- Scattered test files
- Multiple duplicate .md files

### Recent (v1) ✅
- **LangGraph stream:** process_stream(query, ...) yields (node_name, state) after each node; progress_callback for stages.
- **Automatic pagination:** Single-step path and pagination-as-step (fetch_all_pages: true) fetch next page(s) when has_more/next; safety cap in constants.SAFETY_MAX_PAGES; api_client.get_full_url, orchestrator._fetch_next_page.
- **Multi-step robustness:** Parser example "get users and their orders" → relationships; planner checks relationships and multiple_resources.
- **Step failure:** All steps failed → format_error; some failed → summarize with partial results + errors list.
- **Error handling:** Parse/plan exceptions return error as-is in state (no generic fallbacks); ResponseFormatter re-raises on LLM failure; workflow summarize sets response["format_error"] when formatter fails; OPENAI_API_KEY hint in utils. **Retry:** api_client retries on timeout, connection error, 429/502/503/504 with exponential backoff (constants.REQUEST_RETRY_ATTEMPTS, REQUEST_RETRY_BACKOFF_SECONDS).
- **Constants:** limits and static strings centralized in constants.py (see todo.md § Limits).
- **Docs:** README environment setup and Troubleshooting section; filter merge (metadata.filters), progress stages (PLAN_READY, API_COMPLETED), backend_simulator example.

---

## Technical Details

### Dependencies

**Core (required):**
- `requests` - HTTP client
- `openai` - LLM for parsing/planning
- `mcp` - MCP server protocol
- `langgraph` - Workflow orchestration
- `python-dotenv` - Environment variables

**Development:**
- `pytest` - Testing
- `pytest-asyncio` - Async testing
- `black` - Code formatting
- `flake8` - Linting

### Configuration

**Required:** `OPENAI_API_KEY` (env). **Schema:** pass at init `APIOrchestrator(schemas={"api": path_or_dict})` or set in `config.json` (`schemas.api`, `data_sources.api.enabled: true`). **base_url:** from `config.json` (`data_sources.api.base_url`) or from the schema (`base_url`).  
**Optional:** `config.json` (auth, schema paths), `.env` (JWT: `API_EMAIL`, `API_PASSWORD`), `conversation_store` (multi-turn). Full checklist: **[todo.md](todo.md)** § Configuration.

**Schema format (internal):**
```json
{
  "type": "api",
  "base_url": "https://api.example.com",
  "resources": {
    "users": {
      "endpoints": [
        {
          "path": "/users/{id}",
          "method": "GET",
          "intent": "read",
          "parameters": {...}
        }
      ]
    }
  }
}
```

### Authentication Support
- JWT (Bearer tokens)
- OAuth 2.0
- API Keys
- Auto-authentication when needed

---

## Usage Examples

### Basic Usage
```python
from enable_ai import APIOrchestrator

# Initialize
orchestrator = APIOrchestrator()

# Process query
result = orchestrator.process("list all users")

print(result['summary'])  # Human-readable summary
print(result['data'])     # Actual API response
```

### Multi-Turn Conversations
```python
# With session_id; conversation_store holds history (default: in-memory)
orchestrator = APIOrchestrator()

result1 = orchestrator.process("list users", session_id="user-123")
result2 = orchestrator.process("show me the first one", session_id="user-123")

# Production: use DjangoConversationStore or RedisConversationStore
from enable_ai.conversation_store import DjangoConversationStore
store = DjangoConversationStore(ConversationMessage)
orchestrator = APIOrchestrator(conversation_store=store)
```

### FastAPI Integration
```python
from fastapi import FastAPI
from enable_ai import APIOrchestrator

app = FastAPI()
orchestrator = APIOrchestrator()

@app.post("/query")
async def process_query(request: QueryRequest):
    result = orchestrator.process(
        query=request.query,
        session_id=request.session_id
    )
    return result
```

### Schema Generation
```bash
# Generate API schema from OpenAPI spec
enable-schema generate \
    --input swagger.json \
    --output api_schema.json \
    --base-url https://api.example.com
```

---

## Key Decisions

### Why API-only?
- **Focus:** Better to do one thing excellently
- **Simplicity:** Fewer dependencies, easier maintenance
- **Performance:** Smaller package, faster installation
- **Future:** Database/documents can be optional extensions

### Why LangGraph?
- **State management:** Built-in checkpointing for multi-turn conversations
- **Flexibility:** Easy to add/modify workflow steps
- **Debugging:** Clear state transitions, easy to inspect
- **Production-ready:** Battle-tested framework

### Why Centralized OpenAI?
- **Maintainability:** Change model in one place
- **Observability:** All LLM calls logged automatically
- **Cost tracking:** Monitor token usage easily
- **Flexibility:** Easy to add fallbacks, A/B testing

### Why MCP Server?
- **Standardization:** Industry-standard protocol for AI tools
- **Integration:** Works with Claude Desktop, other MCP clients
- **Discoverability:** AI assistants can discover capabilities automatically
- **Future-proof:** Growing ecosystem, increasing adoption

---

## Current Status

### ✅ Complete & Working (v1 scope)
- Single-query → single API; multi-step (3–4+ APIs in sequence) with context passing; summarization (single and multi-step)
- **Pagination:** Detection (total_count, has_more, "show more" suggestion) and **automatic fetch** of next page(s) when response has has_more/next (single-step and pagination-as-step; safety cap in constants)
- OpenAPI as input (auto-convert); CLI `enable-schema generate`; auth (JWT/OAuth/API key); conversation_store (Django/Redis/InMemory)
- Centralized OpenAI; logging; MCP server; schema_loader, schema_validator, response_formatter, progress_tracker
- **LangGraph:** process_stream() yields (node_name, state) after each node; step failure (all failed → format_error; some failed → partial results + errors list)
- **Error handling:** Errors returned as-is (no generic fallbacks); parse/plan/LLM failures surface actual exception; formatter failures set response["format_error"]
- **Constants:** All limits and static strings in `constants.py` (see todo.md § Limits affecting correctness)
- **Docs:** README environment setup (venv, .env, OPENAI_API_KEY, cwd) and Troubleshooting table

### 📋 What's left for v1
See **[todo.md](todo.md)** for the v1 checklist: summarization & presentation (question-aware summary, smart format selection), testing & validation, package deployment; retry for API calls done; conditional/parallel deferred.

---

## LangGraph & streaming (summary)

- **Graph:** `StateGraph(APIQueryState)` with nodes: load_schema → parse → create_plan → execute_step (loop) → summarize; conditional edges for schema error, planning error, and step loop vs done. Use `invoke()` for a single final result, or `process_stream()` for state updates after each node.
- **State:** Input state (query, access_token, runtime_schema, session_id, conversation_history, progress_tracker) is passed in; conversation_history is loaded from `conversation_store` before invoke and user/assistant messages are saved after. LangGraph state is not persisted (no checkpointer).
- **Streaming:** Progress is streamed via `progress_callback` (ProgressTracker). **Progress stages (frontend contract):** STARTED, PARSING_QUERY, INTENT_DETECTED, MATCHING_API, PLANNING, PLAN_READY, EXECUTING_API, API_COMPLETED, SUMMARIZING, COMPLETED, ERROR. Callback receives (stage, message, progress, metadata). Final response is returned once at the end. **Optional:** `orchestrator.process_stream(query, ...)` yields (node_name, state) for each workflow node; use the last state for the final response. For SSE, see `examples/streaming_backend.py`. We do not stream summary tokens.
- **Planning:** ExecutionPlanner outputs steps with `extract` (e.g. `{"user_id": "$.id"}`). Step resolution in `_resolve_step_dependencies` uses each step's `extract` via `_extract_by_path` to set variables from previous step results.

---

## Response & parsed shape (contracts)

- **Response (to caller):** `success`, `data`, `summary`, `query`, `display_mode`, `pagination` (total_count, actual_count, has_more), `suggested_actions`, `total_steps`, `schema_type`; optional `formatted`, `format` (from ResponseFormatter); multi-step adds `execution_plan` (list of step summaries), `errors` if any step failed.
- **Parsed query (parser output):** `intent`, `resource`, `entities`, `filters`, `display_mode` (summary|full|detailed), `merge_with_previous`, `question_type` (count|list|details), `limit`. These drive planning and summarization; `merge_with_previous` + conversation_history trigger filter/entity merging from previous turn.
- **Conversation store:** `get_history(session_id, limit=10)`, `add_message(session_id, role, content, metadata)`, `clear_history(session_id)`. Implementations: InMemoryConversationStore, DjangoConversationStore (requires model with session_id, role, content, metadata, created_at), RedisConversationStore.

---

## Other features (for doc updates)

- **Public API:** `APIOrchestrator.process(query, access_token=None, context=None, runtime_schema=None, session_id=None, progress_callback=None)`; `process_query(query, ...)` (standalone, new orchestrator each time); `orchestrator.clear_conversation(session_id)`; `load_schema(source)`; SchemaLoader, SchemaValidator, ResponseFormatter, ProgressTracker, types (APIRequest, APIResponse, APIError, MissingInformation).
- **environment.py:** DEV config points to `examples/backend_simulator/` (folder exists with config.json, env.example, schema, README).
- **Schema conversion:** Full OpenAPI → Enable AI is in orchestrator (`_convert_openapi_to_enable_ai` + schema_generator). Public `load_schema()` uses SchemaLoader with minimal OpenAPI conversion; for OpenAPI prefer orchestrator with path/dict or `enable-schema generate`.
- **MCP tools:** process_query, get_schema_resources, authenticate, get_config_info. `get_orchestrator(config_path)` no longer passes unsupported params.
- **Progress stages:** Workflow emits STARTED, PARSING_QUERY, INTENT_DETECTED, MATCHING_API, PLANNING, PLAN_READY, EXECUTING_API, API_COMPLETED, SUMMARIZING, COMPLETED, ERROR.
- **Filter merging:** get_history returns optional `metadata`; _extract_previous_filters prefers metadata.filters and uses balanced-bracket fallback for content.
- **Audit (improve & use correctly):** See **todo.md** § "Other features – audit" for each feature: makes sense?, using correctly?, what to improve?

---

## Package Information

**Name:** `enable-ai`  
**Version:** 0.3.27 (see `pyproject.toml`)  
**License:** MIT  
**Author:** Enable Engineering  
**Repository:** https://github.com/EnableEngineering/enable_ai  

**Installation:** `pip install enable-ai`  

**Entry Points:** `enable-schema` (schema generation); MCP server: `python -m enable_ai.mcp_server`  

---

## Testing

```bash
pytest
pytest tests/test_api_endpoint.py
```

---

## Implementing v1 todo items

Use this map to implement pending items from **[todo.md](todo.md)** in the right place.

| Todo item | Where to implement | Notes |
|-----------|--------------------|--------|
| **Summarization & presentation** | `response_formatter.py`, `workflow.py` (summarize), `orchestrator.py` (_summarize_*) | Question-aware summary: analyse user question and data; produce concise, suitable summary (not template strings). Smart format: choose table vs chart vs text from question + result shape. |
| **Retry for failed API calls** | Done: `api_client.py` | Retries on timeout, connection error, 429/502/503/504 with exponential backoff; timeout from constants.REQUEST_TIMEOUT. |
| **Conditional / parallel** (deferred) | `execution_planner.py` (plan format), `workflow.py` (routing) | Extend plan schema with conditionals; branch in graph or run parallel steps when `depends_on` allows. |
| **Testing & deployment** | `tests/`, `pyproject.toml`, CI | Run full test suite with OPENAI_API_KEY; test MCP with Claude Desktop; verify schema generation; pip install from local build; publish to TestPyPI then PyPI. |

---

This context provides the architecture and implementation pointers for the enable_ai project; use **todo.md** for the v1 checklist.
