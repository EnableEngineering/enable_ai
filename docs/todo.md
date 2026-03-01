# Enable AI - TODO (v1)

Remaining v1 work and the SLM accuracy plan are below. For what's implemented and architecture, see **[context.md](context.md)**.

---

## Configuration (required)

- **OPENAI_API_KEY** (env) – For query parsing, planning, and summarization.
- **API schema** – Pass at init: `APIOrchestrator(schemas={"api": path_or_dict})`, or set in **config.json** (`schemas.api`, `data_sources.api.enabled: true`).
- **base_url** – From **config.json** (`data_sources.api.base_url`) or from the **schema** (`base_url`).

See **context.md** for optional config (auth, conversation_store, formatter).

---

## What's working

- Single-query → single API; multi-step (3–4+ APIs) with context and summarisation; pagination (detection + automatic fetch, cap in `constants.SAFETY_MAX_PAGES`).
- OpenAPI input; CLI `enable-schema generate`; auth (JWT/OAuth/API key); conversation_store; LangGraph StateGraph (load_schema → parse → create_plan → execute_step → summarize); streaming via `progress_callback` and `process_stream()`; error handling and retry (api_client); constants in `constants.py`.

---

## Remaining work (v1)

- [ ] **Testing & validation** – Run full test suite with OpenAI API key; test MCP with Claude Desktop; verify schema generation and multi-step/session.
- [ ] **Package deployment** – Test pip install from local build and entry points; build and publish to TestPyPI then PyPI.

**Limits affecting correctness:** All limits and static strings are in **`constants.py`** (pagination cap, page_size, conversation history, timeouts, etc.). See **context.md** for configurable limits; make pagination cap, history limit, page_size cap, and timeouts configurable where needed.

---

## SLM plan: 90% → 95% accuracy

Use **Small Language Models (SLMs)** as precision layers alongside the main LLM. SLMs handle narrow, deterministic sub-tasks; the LLM keeps reasoning and synthesis. This improves reliability, cost, and regression stability.

### 1. Intent classification (before parse/plan)

- **Problem:** LLM does parse + plan + tool choice in one go → more mistakes.
- **Change:** Insert an SLM step before the current parser: user query → **SLM intent classifier** → then existing parser/planner.
- **SLM role:** Classify intent (e.g. list vs get-one vs multi-resource vs refinement) and optionally high-level route. Output: deterministic label(s) or routing hint.
- **Where:** New module or step in `orchestrator.py` / workflow (e.g. before `parse` node); or thin wrapper that calls SLM then passes result into parser/planner.
- **Outcome:** Fewer wrong workflows and tool choices; easier regression tests (fixed intent set).

### 2. Tool parameter extraction / validation (validate & correct)

- **Problem:** Parser output (resource, entities, filters) can be wrong or inconsistent; errors propagate into plans and API calls.
- **Change:** SLM validates against endpoint signature **and produces a corrected parameter object** (or "missing info"); not just flag.
- **SLM role:** Validate parser output; output corrected params or structured missing-information. Closed-loop quality gate.
- **Where:** After parser (e.g. in `query_parser.py` or a new `param_extractor.py`); feed into matcher/planner.
- **Outcome:** Fewer wrong or missing parameters; better alignment with API schema.

### 3. Output / schema validation (validate & correct)

- **Problem:** Summaries or structured outputs can drift, hallucinate, or violate format.
- **Change:** SLM validates summary against **returned data**; returns **OK** or **"Fix summary with these constraints…"**.
- **SLM role:** Check required fields, format, faithfulness to data; output pass or concrete fix instructions.
- **Where:** In `response_formatter.py` or in the summarize node in `workflow.py`; run after formatter LLM call.
- **Outcome:** Fewer bad responses shipped; closed-loop critic.


### 4. Guardrails (policy / safety)

- **Problem:** Relying only on the LLM for policy and safety is variable.
- **Change:** Use an **SLM guardrail** to detect policy violations, prompt injection, off-topic or unsafe output.
- **SLM role:** Binary or categorical checks (e.g. allow / flag / block); no open-ended generation.
- **Where:** Optional step after user input (before parse) and/or after final response (before returning to caller).
- **Outcome:** More consistent and auditable safety and policy enforcement.

### Implementation order (suggested)

1. **Intent classification** – Biggest impact on routing and plan correctness; clear contract for tests.
2. **Parameter extraction / validation** – Directly reduces API and matcher errors.
3. **Output validator** – Improves summary and structured-output reliability.
4. **Guardrails** – Policy/safety and prompt-injection checks.

### Technical notes

- **Model choice:** Use a single small model (e.g. Phi-3, LLaMA 3 8B, Mistral 7B) or dedicated fine-tuned SLMs per task; keep calls stateless and deterministic (low temp, fixed schema).
- **Cost/latency:** Run SLM steps in-process or on a small endpoint; keep LLM for heavy reasoning only.
- **Regression:** Define fixed intents, parameter schemas, and validation rules so SLM layers are easy to unit and regression test.

### Accuracy improvements (measurable 90% → 95%)

Do these alongside (or before) adding more models; see **context.md** § Accuracy improvements on top of SLM for detail.

- [ ] **Eval harness** – Golden set ~200–500 queries per schema; track routing / param / execution / faithfulness metrics per layer; error taxonomy (wrong endpoint, wrong param, missing param, pagination, extract paths, summary contradicts data); fix biggest bucket first.
- [ ] **Schema-constrained outputs + repair** – Parser/planner/validator: strict JSON schema, allowed enums from API schema; reject/repair non-conforming output.
- [ ] **SLM validate & correct** – SLM #2: corrected params or "missing info"; SLM #3: OK or "fix summary with these constraints".
- [ ] **Deterministic short-circuit** – When query clearly maps to one endpoint, use rules/embeddings + SLM intent; call LLM only when ambiguous.
- [ ] **Grounding contract** – Summary claims must have data pointer (jsonpath/row index) or say "Not available in API response".
- [ ] **MissingInformation** – When params missing/ambiguous, return structured MissingInformation with 1–3 specific questions; don't guess.

**Top 3 for biggest jump:** (1) Schema-constrained outputs + repair loop, (2) SLM param validator that corrects, (3) Eval harness + error taxonomy.

---

**Last Updated:** 2026-02-19
