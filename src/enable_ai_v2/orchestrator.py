"""
Main orchestrator for Enable AI v2.

API-driven, zero hardcoding:
- All config from parent module
- Server handles data scoping
- Only add filters user explicitly requests
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .api_client import APIClient
from .auth import Auth, NoAuth
from .llm_client import LLMClient
from .config import Config, UserContext, ResourceHint, build_query_examples_suffix
from .formatter import ResponseFormatter, FormatType
from .progress import ProgressTracker, ProgressStage, ProgressUpdate
from .query_cache import QueryCache
from .tool_converter import convert_spec_to_tools, get_endpoint_by_tool_name
from .tool_filter import filter_openapi_spec, _tool_resource_segment, narrow_list_result_to_exact_code
from .conversation_store import ConversationStore
from .types import (
    ConfidenceScore,
    Endpoint,
    QueryTrace,
    Response,
    ToolCall,
)
from .validator import Validator

# Intent, temporal, multi-step, aggregate support
from .query_intent import (
    QueryIntent,
    classify_query_intent,
    intent_prompt_hint,
    needs_follow_up_round,
    resolve_phrases_from_config,
)
from .date_range import inject_date_filters, date_range_prompt_hint
from .multi_step import (
    build_scoped_flash_report_list,
    build_compound_second_list,
    build_workload_service_orders_list,
    build_technician_directory_list,
    build_retrieve_for_embedded,
)
from .aggregate import format_aggregate_response
from .tool_args import flatten_tool_arguments, enforce_count_tool_args, build_list_cache
from .follow_up import (
    is_referential_follow_up,
    extract_list_cache_from_history,
    build_referential_tool_calls,
    referential_prompt_hint,
)


# Base system prompt - domain knowledge injected from config
SYSTEM_PROMPT_BASE = """You are an API assistant that helps users interact with a REST API using natural language.

Your task:
1. Understand what the user wants
2. Select the appropriate API tool(s)
3. Extract filter parameters from the query

CRITICAL RULES:
{scoping_instruction}

Parameter extraction:
- For dates, use ISO format (YYYY-MM-DD)
- For status filters, use the exact values provided in tool descriptions
- If user says a synonym (e.g., "pending"), map it to the actual status value

Business reference codes (SO-159, REQ-123, etc.):
- Use LIST tools with search=SO-159 or req_id=... — NOT path {{id}} unless the user gives the numeric database primary key explicitly
- The number after a prefix (e.g. 159 in SO-159) is NOT the database primary key
- A code like SO-159 must match exactly — do not let it also return SO-159-A or other sub-orders unless the user asks for those too

Numeric database ids:
- Only use retrieve/{{id}}/ when the user explicitly provides the internal database id (e.g. "id 42", "database id 42")

Tool selection:
- When the user asks "how many" or for a count, use a list tool with filters — do NOT set page_size=1; read the total from the paginated count field
- When the user asks about multiple resources (e.g. orders AND invoices), select one tool per resource
- Use conversation history to resolve follow-ups (e.g. "who is the technician?" refers to the service order discussed earlier)
- Multi-step queries: if the first API result is incomplete, select additional tools using data from prior results

{resource_hints}

{status_synonyms}

{custom_suffix}
"""

RESPONSE_PROMPT = """You are a helpful assistant presenting API results conversationally.

Rules:
- Be concise but informative
- Use natural language, not technical jargon
- If many results, summarize them
- If error, explain simply
- Don't mention APIs, endpoints, or technical details
"""


class Orchestrator:
    """
    Main entry point for natural language to API translation.

    Usage:
        # Parent module provides all config
        config = Config(
            openapi_schema=schema_dict,
            base_url="https://api.example.com",
            resource_hints=hints,  # From API
            status_synonyms=synonyms,  # From API
        )

        ai = Orchestrator(config=config, auth=JWTAuth(token="..."))

        # Process with user context (parent provides this)
        user_ctx = UserContext(
            user_id=123,
            role="Technician",
            is_admin=False,
        )
        result = ai.process("show pending orders", user_context=user_ctx)
    """

    def __init__(
        self,
        config: Config,
        auth: Optional[Auth] = None,
        api_key: Optional[str] = None,
        conversation_store: Optional[ConversationStore] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Configuration with openapi_schema, resource_hints, etc.
            auth: Authentication handler for API calls
            api_key: LLM API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env)
            conversation_store: Optional store for multi-turn session context
        """
        self.config = config
        self._conversation_store = conversation_store

        # Load OpenAPI schema
        if isinstance(config.openapi_schema, str):
            from .tool_converter import load_openapi_spec
            schema = load_openapi_spec(config.openapi_schema)
        else:
            schema = config.openapi_schema

        self._schema = schema
        self.tools, self.endpoints = self._build_tools(schema)

        # Initialize components
        self._llm = LLMClient(
            provider=self.config.llm_provider,
            api_key=api_key,
            config=self.config,
        )
        self._api = APIClient(
            base_url=config.base_url,
            auth=auth or NoAuth(),
            config=self.config,
        )
        self._validator = Validator(self.endpoints)
        self._formatter = ResponseFormatter(self.config)
        self._cache = QueryCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        ) if self.config.cache_enabled else None

    def _build_tools(
        self,
        schema: dict,
        user_context: Optional[UserContext] = None,
    ) -> tuple[list[dict], list]:
        """Convert OpenAPI schema to tools, applying optional parent-side filters."""
        filtered_schema = self._apply_spec_filters(schema, user_context)
        tools, endpoints = convert_spec_to_tools(filtered_schema)
        self._enrich_tools_with_hints(tools)
        return tools, endpoints

    def _apply_spec_filters(self, schema: dict, user_context: Optional[UserContext] = None) -> dict:
        """
        Apply optional parent-configured OpenAPI spec filters.

        Parent (KQSPL) can set tool_exclude_patterns / allowed_http_methods on Config,
        or pre-filter the schema before passing it in.
        """
        exclude_patterns = self.config.tool_exclude_patterns
        allowed_methods = self.config.allowed_http_methods
        is_admin = user_context.is_admin if user_context else True

        if (
            exclude_patterns is None
            and allowed_methods is None
            and (is_admin or not self.config.filter_tools_for_non_admin)
        ):
            return schema

        return filter_openapi_spec(
            schema,
            exclude_patterns=exclude_patterns,
            allowed_methods=allowed_methods,
            is_admin=is_admin,
        )

    def _enrich_tools_with_hints(self, tools: list[dict]) -> None:
        """Enrich tool descriptions with resource hints from config."""
        for tool in tools:
            resource = self._tool_to_resource(tool["name"])
            hint = self.config.resource_hints.get(resource)

            if hint:
                extra_desc = []

                if hint.status_values:
                    extra_desc.append(
                        f"Valid {hint.status_field or 'status'} values: {', '.join(hint.status_values)}"
                    )

                for field, values in hint.enum_fields.items():
                    extra_desc.append(f"Valid {field} values: {', '.join(str(v) for v in values)}")

                if hint.filter_guidance:
                    extra_desc.append(hint.filter_guidance)

                if extra_desc:
                    tool["description"] += "\n\n" + "\n".join(extra_desc)

    def _tool_to_resource(self, tool_name: str) -> str:
        """Extract resource name from tool name."""
        # service_orders_list -> service-orders
        # invoices_retrieve -> invoices
        parts = tool_name.lower().split("_")
        # Remove action suffixes
        actions = {"list", "retrieve", "create", "update", "destroy", "partial"}
        parts = [p for p in parts if p not in actions]
        return "-".join(parts)

    def process(
        self,
        query: str,
        user_context: Optional[UserContext] = None,
        session_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        prior_list_cache: Optional[list[dict]] = None,
    ) -> Response:
        """
        Process a natural language query.

        Args:
            query: User's natural language query
            user_context: User context from parent module
            session_id: Optional session ID for conversation context
            auth_token: Optional auth token override
            conversation_history: Optional prior messages [{"role","content"}]
                Overrides conversation_store when provided.
            prior_list_cache: Optional rows from the last list response, used to
                resolve referential follow-ups ("show the first 3 of those").
                Falls back to extracting from conversation_history when omitted.

        Returns:
            Response with message, data, and optional trace
        """
        # Update auth if token provided
        if auth_token:
            from .auth import JWTAuth
            self._api.auth = JWTAuth(token=auth_token)

        # Rebuild tools if user-specific spec filtering applies
        if user_context and (
            self.config.tool_exclude_patterns is not None
            or self.config.allowed_http_methods is not None
            or (not user_context.is_admin and self.config.filter_tools_for_non_admin)
        ):
            self.tools, self.endpoints = self._build_tools(self._schema, user_context)

        history = self._get_conversation_history(session_id, conversation_history)

        # Progress tracking
        tracker = None
        if self.config.progress_callback:
            tracker = ProgressTracker(
                callback=lambda u: self.config.progress_callback(u.message, u.progress)
            )

        trace = QueryTrace(
            query=query,
            timestamp=datetime.now(),
            session_id=session_id,
            llm_provider=self.config.llm_provider,
        )
        start_time = datetime.now()

        if tracker:
            tracker.update(ProgressStage.STARTED, "Processing your request...")

        try:
            phrases = resolve_phrases_from_config(self.config)
            intent = classify_query_intent(query, phrases)
            trace.intent = intent.value

            raw_tool_calls: list[ToolCall] = []
            reasoning = ""
            # Referential and cached tool calls are never re-cached: referential
            # calls are meaningless outside this conversation ("those"), and a
            # cache hit is already in the cache.
            cacheable = True

            # Step 1: Referential follow-up ("those", "first 5 of above") — resolved
            # deterministically from prior list rows, no LLM call needed.
            list_cache_in = prior_list_cache or extract_list_cache_from_history(history)
            if is_referential_follow_up(query) and list_cache_in:
                ref_calls = build_referential_tool_calls(query, list_cache_in, self.endpoints)
                if ref_calls:
                    raw_tool_calls = ref_calls
                    cacheable = False

            # Step 2: Check cache (skip if the referential step above already resolved it)
            if not raw_tool_calls and self._cache:
                if tracker:
                    tracker.update(ProgressStage.CHECKING_CACHE, "Checking cache...")
                cached = self._cache.get(query)
                if cached:
                    if tracker:
                        tracker.update(ProgressStage.CACHE_HIT, "Found cached result!")
                    trace.cache_hit = True
                    trace.cache_key = cached.pattern
                    trace.confidence = ConfidenceScore.high()
                    raw_tool_calls = cached.tool_calls
                    cacheable = False

            # Step 3: Call LLM for tool selection (skip if steps 1-2 already resolved it)
            if not raw_tool_calls:
                if tracker:
                    tracker.update(ProgressStage.CALLING_LLM, "Understanding your request...")

                system_prompt = self._build_system_prompt(
                    user_context=user_context, query=query, intent=intent
                )
                if list_cache_in:
                    system_prompt += f"\n\n{referential_prompt_hint(list_cache_in)}"

                raw_tool_calls, reasoning, stop_reason = self._llm.process_query(
                    query=query,
                    tools=self.tools,
                    system_prompt=system_prompt,
                    conversation_history=history,
                )

            if tracker:
                tracker.update(
                    ProgressStage.TOOLS_SELECTED,
                    f"Identified: {', '.join(tc.name for tc in raw_tool_calls)}" if raw_tool_calls else "Processing...",
                )

            trace.reasoning = reasoning
            trace.tool_calls = raw_tool_calls

            # No tools selected - return the LLM's direct response (clarification, etc.)
            if not raw_tool_calls:
                trace.response_time_ms = self._elapsed_ms(start_time)
                if tracker:
                    tracker.update(ProgressStage.COMPLETED, "Done!")
                message = reasoning if reasoning else self._formatter.format_no_tools(query)
                response = Response(
                    message=message,
                    success=True,
                    trace=trace if self.config.include_trace else None,
                )
                self._save_conversation(session_id, query, message)
                return response

            # Step 4: Apply the same deterministic pipeline (date filters, status
            # synonyms, user-context scoping, keyword filter injection, argument
            # normalization) whether raw_tool_calls came from the LLM, the cache,
            # or a referential follow-up. This is what keeps a cache hit from
            # going stale (e.g. "this week" resolved at write time) or leaking one
            # user's resolved placeholders (e.g. company_id) into another user's
            # cache hit for the same query text.
            tool_calls = self._prepare_tool_calls(raw_tool_calls, query, user_context, phrases, intent)

            # Step 5: Validate tool calls
            if tracker:
                tracker.update(ProgressStage.VALIDATING, "Validating parameters...")

            clarification_questions: list[str] = []
            for tc in tool_calls:
                endpoint = get_endpoint_by_tool_name(tc.name, self.endpoints)
                if endpoint:
                    result = self._validator.validate(tc, query)
                    if not result.valid:
                        trace.validation_errors.extend(result.errors)
                    if result.needs_clarification and result.clarification_question:
                        clarification_questions.append(result.clarification_question)

            # Ambiguous or risky operation (e.g. a mutation, a duplicate-looking
            # create) — ask before executing rather than guessing.
            if clarification_questions:
                trace.response_time_ms = self._elapsed_ms(start_time)
                if tracker:
                    tracker.update(ProgressStage.COMPLETED, "Done!")
                msg = (
                    clarification_questions[0]
                    if len(clarification_questions) == 1
                    else self._formatter.format_validation_errors(clarification_questions)
                )
                response = Response(
                    message=msg,
                    success=True,
                    needs_input=True,
                    trace=trace if self.config.include_trace else None,
                )
                self._save_conversation(session_id, query, msg)
                return response

            # Step 6: Execute tools and handle multi-step chaining
            response = self._execute_with_follow_up(
                tool_calls=tool_calls,
                query=query,
                intent=intent,
                phrases=phrases,
                trace=trace,
                start_time=start_time,
                tracker=tracker,
                user_context=user_context,
            )

            # Step 7: Cache successful query — store the RAW (pre-injection) tool
            # calls so a future hit re-runs the same deterministic pipeline rather
            # than reusing another user's resolved values or a stale date range.
            if cacheable and response.success and self._cache and trace.confidence:
                self._cache.put(query, raw_tool_calls, trace.confidence.overall)

            if tracker:
                tracker.update(ProgressStage.COMPLETED, "Done!")

            self._save_conversation(session_id, query, response.message)
            return response

        except Exception as e:
            trace.response_time_ms = self._elapsed_ms(start_time)
            if tracker:
                tracker.update(ProgressStage.ERROR, f"Error: {str(e)}")
            return Response(
                message=self._formatter.format_error(str(e), query),
                success=False,
                trace=trace if self.config.include_trace else None,
            )

    def _build_system_prompt(
        self,
        user_context: Optional[UserContext] = None,
        query: Optional[str] = None,
        intent: Optional[QueryIntent] = None,
    ) -> str:
        """Build system prompt with config-driven hints and intent signals."""
        # Resource hints section
        hints_text = ""
        if self.config.resource_hints:
            hints_lines = ["Available resources and their valid filter values:"]
            for resource, hint in self.config.resource_hints.items():
                if hint.status_values:
                    hints_lines.append(
                        f"- {resource}: {hint.status_field}={', '.join(hint.status_values)}"
                    )
                if hint.filter_guidance:
                    hints_lines.append(f"  Guidance: {hint.filter_guidance}")
            hints_text = "\n".join(hints_lines)

        # Status synonyms section
        synonyms_text = ""
        if self.config.status_synonyms:
            syn_lines = ["Status synonyms (use the value on the right):"]
            for natural, actual in self.config.status_synonyms.items():
                syn_lines.append(f'- "{natural}" → {actual}')
            synonyms_text = "\n".join(syn_lines)

        # User context section
        context_text = ""
        if user_context:
            context_text = f"\nCurrent user: {user_context.role or 'User'}"
            if user_context.is_admin:
                context_text += " (admin - sees all data)"

        examples_text = build_query_examples_suffix(self.config.query_examples)
        suffix_parts = [self.config.system_prompt_suffix or "", examples_text]
        custom_suffix = "\n\n".join(p for p in suffix_parts if p)

        prompt = SYSTEM_PROMPT_BASE.format(
            scoping_instruction=self.config.scoping_instruction,
            resource_hints=hints_text,
            status_synonyms=synonyms_text,
            custom_suffix=custom_suffix,
        )

        if context_text:
            prompt += context_text

        # Intent-specific hints
        if intent:
            hint = intent_prompt_hint(intent)
            if hint:
                prompt += f"\n\n{hint}"

        # Temporal filter hints
        if query:
            date_hint = date_range_prompt_hint(query)
            if date_hint:
                prompt += f"\n\n{date_hint}"

        return prompt

    def _get_conversation_history(
        self,
        session_id: Optional[str],
        conversation_history: Optional[list[dict]] = None,
    ) -> Optional[list[dict]]:
        """Load conversation history from explicit arg or conversation store."""
        if conversation_history is not None:
            return conversation_history
        if session_id and self._conversation_store:
            return self._conversation_store.get_llm_messages(
                session_id,
                limit=self.config.conversation_history_limit,
            )
        return None

    def _save_conversation(
        self,
        session_id: Optional[str],
        query: str,
        message: str,
    ) -> None:
        """Persist user/assistant turns when a conversation store is configured."""
        if not session_id or not self._conversation_store:
            return
        self._conversation_store.add_message(session_id, "user", query)
        self._conversation_store.add_message(session_id, "assistant", message)

    def _translate_status_synonyms(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        """Translate natural language status values to actual values."""
        if not self.config.status_synonyms:
            return tool_calls

        result = []
        for tc in tool_calls:
            new_args = dict(tc.arguments)

            for key, value in tc.arguments.items():
                if isinstance(value, str):
                    # Check if it's a known synonym
                    lower_val = value.lower()
                    if lower_val in self.config.status_synonyms:
                        new_args[key] = self.config.status_synonyms[lower_val]

            result.append(ToolCall(id=tc.id, name=tc.name, arguments=new_args))

        return result

    def _resolve_user_placeholders(
        self,
        tool_calls: list[ToolCall],
        user_context: Optional[UserContext],
    ) -> list[ToolCall]:
        """
        Resolve LLM placeholder values with actual user context.

        Placeholders:
        - __current_user_id__ → user_context.user_id
        - __current_company_id__ → user_context.company_id
        - __current_username__ → user_context.username
        """
        if not user_context:
            return tool_calls

        placeholders = {
            "__current_user_id__": user_context.user_id,
            "__current_company_id__": user_context.company_id,
            "__current_username__": user_context.username,
        }

        result = []
        for tc in tool_calls:
            new_args = dict(tc.arguments)

            for key, value in tc.arguments.items():
                if isinstance(value, str) and value in placeholders:
                    resolved = placeholders[value]
                    if resolved is not None:
                        new_args[key] = resolved
                    else:
                        # Remove the arg if placeholder can't be resolved
                        del new_args[key]

            result.append(ToolCall(id=tc.id, name=tc.name, arguments=new_args))

        return result

    def _inject_context_filters(
        self,
        tool_calls: list[ToolCall],
        query: str,
        user_context: Optional[UserContext],
        phrases: Any = None,
    ) -> list[ToolCall]:
        """
        Inject filters based on context phrases in query and user role.

        - "my company" → explicit company filter.
        - Self-scope keywords ("my jobs", "assigned to me", configurable via
          phrases.self_scope_keywords) → filter by the current user, on
          whichever param phrases.self_scope_param_map names for that resource
          (e.g. technician= for service orders).
        - For non-admin users, phrases.implicit_company_scope_map opts specific
          resources into always scoping by the user's own company, even
          without an explicit "my" phrase (e.g. a customer's plain "show me
          all invoices"). Off by default — the parent module must configure it.

        All keywords/param names are parent-configured (zero hardcoding); this
        method only wires the mechanics together.
        """
        if not user_context:
            return tool_calls

        q = query.lower()
        self_scope_keywords = getattr(phrases, "self_scope_keywords", ()) if phrases else ()
        self_scope_param_map = getattr(phrases, "self_scope_param_map", {}) if phrases else {}
        implicit_company_scope_map = (
            getattr(phrases, "implicit_company_scope_map", {}) if phrases else {}
        )
        wants_self_scope = any(kw.lower() in q for kw in self_scope_keywords)
        wants_my_company = "my company" in q

        if not (
            wants_my_company
            or (wants_self_scope and self_scope_param_map)
            or (not user_context.is_admin and implicit_company_scope_map)
        ):
            return tool_calls

        result = []
        for tc in tool_calls:
            # Only inject into list tools
            if "list" not in tc.name.lower():
                result.append(tc)
                continue

            new_args = dict(tc.arguments)
            segment = _tool_resource_segment(tc.name)

            if wants_my_company and user_context.company_id is not None and "company" not in new_args:
                new_args["company"] = user_context.company_id

            if wants_self_scope and user_context.user_id is not None:
                for seg, param in self_scope_param_map.items():
                    if seg in segment and param not in new_args:
                        new_args[param] = user_context.user_id
                        break

            if not user_context.is_admin and user_context.company_id is not None:
                for seg, param in implicit_company_scope_map.items():
                    if seg in segment and param not in new_args:
                        new_args[param] = user_context.company_id
                        break

            result.append(ToolCall(id=tc.id, name=tc.name, arguments=new_args))

        return result

    def _inject_status_filter(
        self,
        tool_calls: list[ToolCall],
        query: str,
        phrases: Any,
    ) -> list[ToolCall]:
        """
        Inject status filter based on keywords in query.

        Uses phrases.status_injection_map and phrases.status_field_map
        configured by parent.
        """
        status_map = getattr(phrases, "status_injection_map", {})
        field_map = getattr(phrases, "status_field_map", {})

        if not status_map:
            return tool_calls

        q = query.lower()
        matched_status: Optional[str] = None

        # Find first matching keyword
        for keyword, status_value in status_map.items():
            if keyword.lower() in q:
                matched_status = status_value
                break

        if not matched_status:
            return tool_calls

        result = []
        for tc in tool_calls:
            # Only inject into list tools
            if "list" not in tc.name.lower():
                result.append(tc)
                continue

            # Determine field name from resource segment
            tc_lower = tc.name.lower()
            field_name = "status"  # default
            for segment, fname in field_map.items():
                if segment in tc_lower:
                    field_name = fname
                    break

            new_args = dict(tc.arguments)

            # Skip if ANY status-related field already set (avoid conflict)
            has_status = any(
                k == field_name or k.startswith("status") for k in new_args
            )
            if not has_status:
                new_args[field_name] = matched_status

            result.append(ToolCall(id=tc.id, name=tc.name, arguments=new_args))

        return result

    def _inject_equipment_filter(
        self,
        tool_calls: list[ToolCall],
        query: str,
        phrases: Any,
    ) -> list[ToolCall]:
        """
        Inject equipment in_use filter when query mentions equipment usage.

        Uses phrases.equipment_in_use_keywords and phrases.equipment_in_use_param
        configured by parent.
        """
        keywords = getattr(phrases, "equipment_in_use_keywords", ())
        param = getattr(phrases, "equipment_in_use_param", "in_use")

        if not keywords:
            return tool_calls

        q = query.lower()
        should_inject = any(kw.lower() in q for kw in keywords)

        if not should_inject:
            return tool_calls

        result = []
        for tc in tool_calls:
            # Only inject into equipment list tools
            if "equipment" not in tc.name.lower() or "list" not in tc.name.lower():
                result.append(tc)
                continue

            new_args = dict(tc.arguments)
            if param not in new_args:
                new_args[param] = True

            result.append(ToolCall(id=tc.id, name=tc.name, arguments=new_args))

        return result

    def _inject_report_type_filter(
        self,
        tool_calls: list[ToolCall],
        query: str,
        phrases: Any,
    ) -> list[ToolCall]:
        """
        Inject report_type filter based on keywords in query.

        Uses phrases.report_type_injection_map and phrases.report_type_param
        configured by parent.
        """
        type_map = getattr(phrases, "report_type_injection_map", {})
        param = getattr(phrases, "report_type_param", "report_type")

        if not type_map:
            return tool_calls

        q = query.lower()
        matched_type: Optional[str] = None

        for keyword, type_value in type_map.items():
            if keyword.lower() in q:
                matched_type = type_value
                break

        if not matched_type:
            return tool_calls

        result = []
        for tc in tool_calls:
            # Only inject into report/flash list tools
            if "list" not in tc.name.lower():
                result.append(tc)
                continue
            if "report" not in tc.name.lower() and "flash" not in tc.name.lower():
                result.append(tc)
                continue

            new_args = dict(tc.arguments)
            if param not in new_args:
                new_args[param] = matched_type

            result.append(ToolCall(id=tc.id, name=tc.name, arguments=new_args))

        return result

    def _execute_with_follow_up(
        self,
        tool_calls: list[ToolCall],
        query: str,
        intent: QueryIntent,
        phrases: Any,
        trace: QueryTrace,
        start_time: datetime,
        tracker: Optional[ProgressTracker] = None,
        user_context: Optional[UserContext] = None,
    ) -> Response:
        """
        Execute tool calls with deterministic multi-step follow-up.

        Handles:
        - Flash report chains (SO list → flash-reports list)
        - Compound queries (service orders AND invoices)
        - Technician availability (users list + service orders)
        """
        all_tool_calls: list[ToolCall] = list(tool_calls)
        all_results: list[dict] = []
        max_rounds = self.config.max_tool_rounds or 3

        for round_num in range(max_rounds):
            # Execute current batch of tool calls
            current_calls = all_tool_calls[len(all_results):]
            if not current_calls:
                break

            if tracker:
                tracker.update(
                    ProgressStage.EXECUTING_API,
                    f"Calling API (round {round_num + 1})..."
                )

            for tc in current_calls:
                endpoint = get_endpoint_by_tool_name(tc.name, self.endpoints)
                if not endpoint:
                    all_results.append({"error": f"Unknown tool: {tc.name}"})
                    continue
                result, api_trace = self._api.execute(tc, endpoint)
                if "list" in tc.name.lower():
                    result = narrow_list_result_to_exact_code(
                        result, query, self.config.id_code_patterns
                    )
                all_results.append(result)
                trace.api_calls.append(api_trace)

            # Check for deterministic follow-up tools (no LLM call)
            last_tc = current_calls[-1] if current_calls else None
            last_result = all_results[-1] if all_results else {}

            follow_up: Optional[ToolCall] = None

            # Flash report chain
            if last_tc and "error" not in last_result:
                follow_up = build_scoped_flash_report_list(
                    query, last_tc.name, last_result, self.endpoints, phrases
                )

            # Compound second resource
            if not follow_up:
                follow_up = build_compound_second_list(
                    query, all_tool_calls[:len(all_results)], self.endpoints, phrases
                )

            # Technician workload (after users list)
            if not follow_up:
                follow_up = build_workload_service_orders_list(
                    query, all_tool_calls[:len(all_results)], self.endpoints, phrases
                )

            # Technician directory (after service orders list)
            if not follow_up:
                follow_up = build_technician_directory_list(
                    query, all_tool_calls[:len(all_results)], self.endpoints, phrases
                )

            # Auto-retrieve for embedded fields (observations, flash reports, etc.)
            if not follow_up and last_tc:
                follow_up = build_retrieve_for_embedded(
                    query, last_tc.name, last_result, self.endpoints, phrases
                )

            if follow_up:
                all_tool_calls.append(follow_up)
                trace.follow_up_rounds += 1
            else:
                break

        # Update trace with all tool calls
        trace.tool_calls = all_tool_calls

        # Generate final response
        return self._finalize_response(
            all_tool_calls, all_results, query, intent, phrases, trace, start_time, tracker,
            user_context=user_context,
        )

    def _finalize_response(
        self,
        tool_calls: list[ToolCall],
        results: list[dict],
        query: str,
        intent: QueryIntent,
        phrases: Any,
        trace: QueryTrace,
        start_time: datetime,
        tracker: Optional[ProgressTracker] = None,
        user_context: Optional[UserContext] = None,
    ) -> Response:
        """Generate final response after all tool calls complete."""
        all_success = all("error" not in r for r in results)

        if tracker:
            tracker.update(ProgressStage.API_COMPLETED, "Got data!")

        # Calculate confidence
        if tool_calls and trace.api_calls:
            first_tc = tool_calls[0]
            first_endpoint = get_endpoint_by_tool_name(first_tc.name, self.endpoints)
            if first_endpoint:
                validation = self._validator.validate(first_tc, query)
                trace.confidence = self._validator.calculate_full_confidence(
                    first_tc, first_endpoint, validation, cache_hit=trace.cache_hit
                )

        if tracker:
            tracker.update(ProgressStage.FORMATTING, "Preparing response...")

        # Generate response message
        if all_success:
            # Try aggregate formatter first (handles compound, technician availability)
            if intent in (QueryIntent.AGGREGATE, QueryIntent.MULTI_STEP) or len(tool_calls) > 1:
                aggregate_msg = format_aggregate_response(
                    results, query, tool_calls=tool_calls, phrases=phrases
                )
                if aggregate_msg:
                    data = [r.get("data") for r in results] if len(results) > 1 else results[0].get("data")
                    trace.response_time_ms = self._elapsed_ms(start_time)
                    list_cache = build_list_cache(
                        tool_calls, results, max_rows=self.config.count_list_page_size
                    )
                    trace.list_cache = list_cache
                    return Response(
                        message=aggregate_msg,
                        data=data,
                        success=True,
                        suggestions=self._formatter.build_suggestions(query, tool_calls, results),
                        list_cache=list_cache,
                        trace=trace if self.config.include_trace else None,
                    )

            # Standard formatting
            data = results[0].get("data") if len(results) == 1 else [r.get("data") for r in results]
            if len(tool_calls) > 1:
                message = self._formatter.format_multi_success(tool_calls, results, query=query)
            elif self._is_simple_result(data, query=query, tool_calls=tool_calls):
                message = self._formatter.format_success_simple(
                    data, tool_calls[0].name, query=query
                )
            else:
                message = self._llm.generate_response(
                    query=query,
                    api_results=results,
                    tool_calls=tool_calls,
                    system_prompt=RESPONSE_PROMPT,
                )
        else:
            # Error handling
            role = user_context.role if user_context else None
            for result in results:
                if "error" in result:
                    message = self._formatter.format_api_error(
                        tool_calls[0].name if tool_calls else "unknown",
                        result.get("error", "Unknown error"),
                        result.get("status"),
                        hint=(
                            "Try searching by business code (e.g. SO-159) instead of numeric id."
                            if result.get("status") == 404 and tool_calls and "retrieve" in tool_calls[0].name
                            else None
                        ),
                        role=role,
                    )
                    break
            else:
                message = "Something went wrong."
            data = None

        # Extract data
        if not all_success:
            data = None
        elif len(results) == 1:
            data = results[0].get("data")
        else:
            data = [r.get("data") for r in results]

        list_cache = build_list_cache(tool_calls, results, max_rows=self.config.count_list_page_size)
        trace.list_cache = list_cache

        trace.response_time_ms = self._elapsed_ms(start_time)

        return Response(
            message=message,
            data=data,
            success=all_success,
            suggestions=self._formatter.build_suggestions(query, tool_calls, results),
            list_cache=list_cache,
            trace=trace if self.config.include_trace else None,
        )

    def _build_continuation_message(
        self,
        query: str,
        tool_calls: list[ToolCall],
        results: list[dict],
    ) -> str:
        """Build a continuation prompt for multi-step chains."""
        lines = [f"Original user request: {query}", "", "Tool calls so far:"]
        for tc in tool_calls:
            lines.append(f"- {tc.name}({tc.arguments})")
        lines.append("")
        lines.append("Results:")
        for i, result in enumerate(results):
            data = result.get("data")
            if isinstance(data, dict) and "results" in data:
                row = data["results"][0] if data["results"] else {}
                label = row.get("name") or row.get("display_name") or f"id={row.get('id')}"
                lines.append(f"  {i + 1}. {label}")
            else:
                lines.append(f"  {i + 1}. {data}")
        return "\n".join(lines)

    def _prepare_tool_calls(
        self,
        tool_calls: list[ToolCall],
        query: str,
        user_context: Optional[UserContext],
        phrases: Any,
        intent: QueryIntent,
    ) -> list[ToolCall]:
        """
        Apply the full deterministic pipeline to raw tool calls: date filters,
        status synonyms, user-context scoping/placeholders, keyword-driven
        filter injection, and argument normalization/flattening. Runs
        identically regardless of whether tool_calls came from a fresh LLM
        call, the query cache, or a referential follow-up (see process()).
        """
        tool_calls = inject_date_filters(tool_calls, query)
        tool_calls = self._translate_status_synonyms(tool_calls)
        tool_calls = self._resolve_user_placeholders(tool_calls, user_context)
        tool_calls = self._inject_context_filters(tool_calls, query, user_context, phrases)
        tool_calls = self._inject_status_filter(tool_calls, query, phrases)
        tool_calls = self._inject_equipment_filter(tool_calls, query, phrases)
        tool_calls = self._inject_report_type_filter(tool_calls, query, phrases)

        normalized: list[ToolCall] = []
        for tc in tool_calls:
            n = self._validator.normalize_arguments(tc, self.config.status_synonyms)
            flat_args = flatten_tool_arguments(n.arguments)
            normalized.append(ToolCall(id=n.id, name=n.name, arguments=flat_args))

        return enforce_count_tool_args(
            normalized,
            intent,
            retain_page_size=self.config.count_list_page_size,
            aggregate_page_size=self.config.aggregate_list_page_size,
        )

    def _is_simple_result(
        self,
        data: Any,
        query: str = "",
        tool_calls: Optional[list[ToolCall]] = None,
    ) -> bool:
        """Check if result is simple enough to format locally."""
        if tool_calls and len(tool_calls) > 1:
            return False
        if data is None:
            return True
        if isinstance(data, list):
            # Multi-API combined data
            if data and isinstance(data[0], (dict, list)):
                return all(self._is_simple_result(item, query=query) for item in data)
            return len(data) <= 10
        if isinstance(data, dict):
            return True
        return True

    def _elapsed_ms(self, start: datetime) -> int:
        """Calculate elapsed milliseconds."""
        return int((datetime.now() - start).total_seconds() * 1000)

    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self._cache:
            self._cache.clear()

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats()
        return {"enabled": False}

    def update_auth(self, auth: Auth) -> None:
        """Update authentication."""
        self._api.auth = auth

    def close(self) -> None:
        """Clean up resources."""
        self._api.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
