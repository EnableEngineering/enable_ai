# Enable AI - TODO (v1)

Remaining v1 work is listed below. For what's already implemented, see **[context.md](context.md)** § What's Been Done and Current Status.

**For architecture, config details, and where to implement each item:** see **[context.md](context.md)** (and the "Implementing v1 todo items" section there).

---

## Configuration: Required & Optional

### Required
- **OPENAI_API_KEY** (env) – For query parsing, planning, and summarization.
- **API schema** – Pass at init: `APIOrchestrator(schemas={"api": path_or_dict})`, or set in **config.json** (`schemas.api`, `data_sources.api.enabled: true`).
- **base_url** – From **config.json** (`data_sources.api.base_url`) or from the **schema** (`base_url`).

### Optional
- **config.json** – Not required if you pass `schemas=` and set `base_url` in the schema. Used for auth (JWT/OAuth/API keys) and schema paths.
- **.env** – JWT: `API_EMAIL`, `API_PASSWORD`; OpenAI: `OPENAI_API_KEY`.
- **conversation_store** – Multi-turn; default is in-memory (not production-safe).
- **formatter_config** – Optional response formatting.

### Minimal "any API docs" setup
1. OpenAPI/Swagger (file or URL) + API base URL.
2. `APIOrchestrator(schemas={"api": openapi_path_or_dict})` with `base_url` in schema or config.
3. `OPENAI_API_KEY` in environment.

---

## What's Working

- Single-query → single API (parse, match, call, format).
- Multi-step (3–4+ APIs in sequence) with context passing and summarisation.
- Summarisation (single-step and multi-step).
- **Pagination:** Detection (total_count, has_more, "show more" suggestion) and **automatic fetch** of next page(s) when response has has_more/next (single-step and pagination-as-step; safety cap in `constants.SAFETY_MAX_PAGES`).
- OpenAPI as input (auto-convert); CLI: `enable-schema generate`.
- Auth (JWT/OAuth/API key from config).
- Conversation context (session_id + conversation_store).
- **LangGraph:** StateGraph with load_schema → parse → create_plan → execute_step (loop) → summarize; conditional routing; `process_stream()` yields (node_name, state) after each node.
- **State:** Conversation history loaded from `conversation_store` before invoke and saved (user + assistant message) after; LangGraph state is ephemeral per invoke (no checkpointer).
- **Streaming:** Progress via `progress_callback` (stages + message + progress %); optional `process_stream()`; SSE example in `examples/streaming_backend.py`. Final response returned once at the end.
- **Error handling:** Errors returned as-is (no generic fallbacks that mask the real error); LLM/parse/plan failures surface the actual exception message; formatter failures set `format_error` on the response.

---

### Remaining work (v1)

- [ ] **Conditional / parallel** (optional for v1) – If/else in plans; parallel steps. Deferred.

### Limits affecting correctness (audit)

**All limits and static response strings are in `constants.py`** so they can be updated in one place. Change values there (or make them config-driven) as needed.

Hardcoded limits that can produce incorrect or incomplete results. All should be configurable or documented; consider making them config/options.

| Location | Limit | Impact |
|----------|--------|--------|
| **workflow.py** | `SAFETY_MAX_PAGES = 500` (automatic pagination, single-step) | If API has >500 pages, we stop and merge only 500; `has_more` may still be true but we present merged data. Logged warning. |
| **workflow.py** | `SAFETY_MAX_PAGES = 500` (pagination-as-step, fetch_all_pages) | Same as above for multi-step steps with fetch_all_pages. |
| **orchestrator.py** | `min(int(limit), 100)` when building page_size from parsed limit (~L779) | User asking e.g. "show me 200 items" gets only 100 per request; first page only or wrong total if pagination merges. |
| **orchestrator.py** | `_extract_examples(..., max_examples=3)` (L1488, L1533, L1619) | Summaries show only 3 examples; text says "and N more" so count is correct, but display is capped. |
| **conversation_store.py** | `get_history(session_id, limit=10)` (InMemory, Django, Redis) | Only last 10 messages loaded for context; long conversations lose earlier filters/intent → wrong merge or wrong plan. |
| **conversation_store.py** | InMemoryConversationStore `max_messages=10` (L193) | Older messages dropped; context for merge/refinement can be wrong. |
| **conversation_store.py** | RedisConversationStore `max_messages=20` (L135) | Same; only last 20 kept per session. |
| **response_formatter.py** | Chart: `min(20, len(data))` for labels/datasets (L562, L565) | Charts show at most 20 items; larger datasets misrepresented. |
| **response_formatter.py** | `max_tokens=50` / `150` / `500` in LLM calls (L342, L409, L597) | Summaries can be truncated; incomplete wording, not wrong count. |
| **response_formatter.py** | `data[:2]`, `data[:50]`, `data[:20]`, fields `[:6]`, `[:10]` in prompts (L322, L325, L396, L435, L444, L453) | Fewer items/fields sent to LLM → summary may miss patterns. |
| **api_client.py** | `timeout=30` for all requests (L115–123, L176, L191) | Slow APIs return error instead of data → incorrect "failure" for valid long-running calls. |
| **orchestrator.py** | `timeout=30` in auth/schema fetch (L1151, L1230); schema_loader `timeout=30` | Same. |
| **orchestrator.py** | `AUTH_EXPIRES_IN_DEFAULT=3600`, `AUTH_TOKEN_BUFFER_SECONDS=60` (JWT/OAuth cache) | Wrong defaults can cause early refresh or stale tokens → 401s. |
| **workflow.py** | `_generate_suggestions` uses `TABLE_ROW_SAMPLE` (20) and returns `suggestions[:3]` (L93) | Only 3 suggestions shown; 20 is threshold for "filter" vs "narrow" suggestion. |
| **utils.py** | `truncate_string(..., max_length=100)` (L338) | Logs/display truncate; can hide important detail in errors. |

**Recommendation:** Make pagination safety cap, conversation history limit, page_size cap, and timeouts configurable (e.g. config.json or env); document defaults in README.

### Error handling
- [x] Retry for failed API calls – `api_client.py`: retry on timeout, connection error, and 429/502/503/504 with exponential backoff (`REQUEST_RETRY_ATTEMPTS`, `REQUEST_RETRY_BACKOFF_SECONDS` in constants). Timeout remains `constants.REQUEST_TIMEOUT` (configurable there).

### Summarization & presentation
- [ ] **Question-aware summarization** – Replace generic template summaries (e.g. "Retrieved all {count} item(s) as requested", "Retrieved data as requested") with a proper analysis of the user's question and a concise, suitable summary of the answer (not long paragraphs; "good enough" to answer what was asked). Summaries should reflect the question and the data, not boilerplate.
- [ ] **Smart format selection** – Choose the most suitable presentation from the data and question: **table** when comparison/list is natural, **chart** when trends/distribution/numbers suit it, **text** when narrative is better. Drive format (table vs chart vs text etc) from an analysis of the question and result shape, not a single default.

### Testing & validation
- [ ] Run full test suite with OpenAI API key.
- [ ] Test MCP server with Claude Desktop; FastAPI example end-to-end.
- [ ] Verify schema generation with real OpenAPI specs; test multi-step and session/state.

### Package deployment
- [ ] Test pip install from local build; verify entry points; test in fresh venv.
- [ ] Build packages; publish to TestPyPI then PyPI.

---

## Out of scope for v1

*(Keep for later versions.)*

- **Performance** – Schema/query caching, workflow optimisations, metrics.
- **Security** – Key handling hardening, input validation, rate limiting, audit logging.
- **Other APIs** – GraphQL, Postman import.
- **Extra features** – Batch processing, query history/replay, analytics/token dashboards.
- **Integrations** – Express/Django examples, Docker/K8s guides.
- **Phase 2** – Database support, document analysis (PDF/RAG), custom/local models, multi-language.

---

**Last Updated:** 2026-01-31
