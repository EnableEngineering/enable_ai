from typing import Any, Dict, Optional, TypedDict, List
import re
import json

from langgraph.graph import StateGraph, END  # type: ignore[import]

from .types import APIError
from .execution_planner import ExecutionPlanner
from .intent_classifier import IntentClassifier
from .utils import setup_logger, get_openai_client, CREATIVE_TEMP
from .progress_tracker import ProgressStage
from .response_formatter import ResponseFormatter
from .tracing import QueryTracer, get_tracer_store, create_tracer
from .follow_up_suggestions import (
    enrich_response_with_follow_ups,
    generate_follow_up_queries,
    default_error_suggestions,
    default_error_follow_up_queries,
)
from .follow_up_detection import (
    apply_follow_up_context,
    classify_follow_up,
    extract_last_result_metadata,
    has_prior_context,
)
from .query_execution import merge_execution_context
from .semantic_filters import apply_semantic_filters
from .response_envelope import enrich_api_response
from .user_context_resolver import resolve_user_context_in_parsed
from . import constants

# Module-level logger
logger = setup_logger('enable_ai.workflow')

# Global tracer store
_tracer_store = get_tracer_store()


def _extract_limit_from_query(query: str) -> int:
    """
    Extract numeric limit from follow-up queries like "next 2" or "show me first 5".

    Args:
        query: User's query string

    Returns:
        Extracted limit or default (10)
    """
    import re
    query_lower = query.lower()

    # Match patterns like "next 2", "first 5", "show me 10"
    patterns = [
        r'\b(?:next|first|last|show me|show)\s+(\d+)\b',
        r'\b(\d+)\s+(?:more|results|items)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return int(match.group(1))

    return 10  # Default limit


def _unwrap_json_response(summary: str) -> str:
    """
    Fix JSON wrapper issue (v0.3.37).

    Sometimes the LLM returns a JSON wrapper like:
        {"format": "table", "response": "Here are all 25..."}

    Instead of clean text. This function extracts the actual response.

    Args:
        summary: Summary text that may contain JSON wrapper

    Returns:
        Clean text without JSON wrapper
    """
    if not isinstance(summary, str):
        return str(summary) if summary else ""

    summary = summary.strip()

    # Check if it looks like JSON
    if summary.startswith('{') and summary.endswith('}'):
        try:
            parsed = json.loads(summary)
            if isinstance(parsed, dict):
                # Extract 'response' field if present
                if 'response' in parsed:
                    return str(parsed['response'])
                # Try other common fields
                for field in ['summary', 'text', 'content', 'message']:
                    if field in parsed:
                        return str(parsed[field])
        except json.JSONDecodeError:
            pass

    # Check for partial JSON (starting with { but malformed)
    if summary.startswith('{"format"') or summary.startswith('{"response"'):
        # Try to extract the response value using regex
        match = re.search(r'"response"\s*:\s*"(.*)"', summary, re.DOTALL)
        if match:
            # Unescape the string
            extracted = match.group(1)
            extracted = extracted.replace('\\"', '"').replace('\\n', '\n')
            return extracted

    return summary


def _analyze_pagination(data: Any) -> Dict[str, Any]:
    """
    Analyze API response to extract pagination information (v0.3.11).
    
    Fixes Issues #2-4: Accurate count calculation.
    
    Args:
        data: API response data
        
    Returns:
        Dict with:
            - total_count: Total items available
            - actual_count: Items in current response
            - has_more: Whether more pages exist
            - is_paginated: Whether response is paginated
    """
    info = {
        'total_count': 0,
        'actual_count': 0,
        'has_more': False,
        'is_paginated': False,
        'next_url': None,
    }
    
    # Check for paginated response (Django REST framework style)
    if isinstance(data, dict) and 'results' in data:
        info['is_paginated'] = True
        info['total_count'] = data.get('count', 0)
        results = data.get('results', [])
        info['actual_count'] = len(results)
        info['has_more'] = data.get('next') is not None
        if data.get('next'):
            info['next_url'] = data.get('next')
    # Simple list
    elif isinstance(data, list):
        info['actual_count'] = len(data)
        info['total_count'] = len(data)
    # Single item
    elif isinstance(data, dict):
        info['actual_count'] = 1
        info['total_count'] = 1
    else:
        info['actual_count'] = 1
        info['total_count'] = 1
    
    return info


def _extract_previous_filters(conversation_history: list) -> Dict[str, Any]:
    """
    Extract previous filters from conversation history (v0.3.9).
    Prefers metadata.filters when present (robust); falls back to [Filters: {...}] in content.
    """
    import re
    import json as json_module

    for msg in reversed(conversation_history):
        if msg.get("role") != "assistant":
            continue
        # Prefer metadata.filters when present (avoids regex on content)
        metadata = msg.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("filters"):
            return metadata["filters"]
        # Fallback: parse [Filters: {...}] from content (balanced-brace for robustness)
        content = msg.get("content", "")
        if "[Filters:" in content:
            idx = content.find("[Filters:")
            brace_start = content.find("{", idx)
            if brace_start != -1:
                depth = 0
                end = brace_start
                for i in range(brace_start, len(content)):
                    if content[i] == "{":
                        depth += 1
                    elif content[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                try:
                    return json_module.loads(content[brace_start:end])
                except (json_module.JSONDecodeError, ValueError):
                    pass
            match = re.search(r"\[Filters: (.*?)\]", content, re.DOTALL)
            if match:
                try:
                    return json_module.loads(match.group(1))
                except (json_module.JSONDecodeError, ValueError):
                    pass
    return {}


def _extract_previous_entities(conversation_history: list) -> Dict[str, Any]:
    """
    Extract previous entities from conversation history (v0.3.9).
    
    Args:
        conversation_history: Conversation history
        
    Returns:
        Dict of previous entities or empty dict
    """
    # For now, we primarily use filters
    # Entities can be derived from filters if needed
    filters = _extract_previous_filters(conversation_history)
    
    # Convert filters to entities format
    entities = {}
    for field, filter_obj in filters.items():
        if isinstance(filter_obj, dict) and 'value' in filter_obj:
            entities[field] = filter_obj['value']
    
    return entities


def _extract_by_path(data: Any, path: str) -> Any:
    """
    Extract a value from dict/list using a simple path (e.g. $.id, $.data.user_id).
    Supports $.key and $.key1.key2; no array indexing for v1.
    """
    if not path.startswith("$.") or not isinstance(data, dict):
        return None
    keys = path[2:].strip().split(".")
    current = data
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return None
        current = current[k]
    return current


def _resolve_step_dependencies(
    step_def: Dict[str, Any],
    previous_results: List[Dict[str, Any]],
    all_steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Resolve variable dependencies in a step by substituting values from previous steps.
    Uses each step's "extract" (e.g. {"user_id": "$.id"}) when available, then fallback
    to last result keys and step_N_id from result["id"].
    """
    resolved = json.loads(json.dumps(step_def))  # Deep copy
    variables = {}
    all_steps = all_steps or []

    # Apply planner's extract rules from each previous step
    for prev_step in previous_results:
        step_id = prev_step.get("step_id")
        result_data = prev_step.get("result")
        step_spec = next((s for s in all_steps if s.get("step_id") == step_id), None)
        if step_spec and isinstance(result_data, dict):
            for var_name, path in (step_spec.get("extract") or {}).items():
                if isinstance(path, str) and path.startswith("$."):
                    value = _extract_by_path(result_data, path)
                    if value is not None:
                        variables[var_name] = value

        # Fallback: simple id and step_N_id
        if isinstance(result_data, dict) and "id" in result_data:
            variables[f"step_{step_id}_id"] = result_data["id"]

    # Also add all keys from the most recent result (so {id}, {name} etc. work)
    if previous_results:
        last_result = previous_results[-1].get("result", {})
        if isinstance(last_result, dict):
            for key, value in last_result.items():
                if key not in variables:
                    variables[key] = value

    # Substitute {variable_name} in the step definition
    step_json = json.dumps(resolved)
    for var_name, var_value in variables.items():
        pattern = r"\{"
        pattern += re.escape(var_name)
        pattern += r"\}"
        try:
            replacement = json.dumps(var_value)
        except TypeError:
            replacement = json.dumps(str(var_value))
        step_json = re.sub(pattern, lambda m, r=replacement: r, step_json)

    return json.loads(step_json)


class APIQueryState(TypedDict, total=False):
    """
    State for LangGraph API workflow (v0.3.37: added tracing and follow-up context).

    Input fields:
        query: User's natural language query
        session_id: Session identifier for conversation context
        access_token: JWT token for API authentication
        runtime_schema: Optional schema override for this query
        conversation_history: Previous messages (loaded externally, passed through)
        user_context: User identity for resolving "me"/"my" pronouns (v0.3.29)

    Processing fields:
        active_schema: Schema being used for this query
        parsed: Parsed query (intent, resource, filters)
        execution_plan: Multi-step execution plan
        current_step: Current step index in plan
        step_results: Results from executed steps
        result: Final execution result

    Output fields:
        summary: Human-readable summary
        response: Complete response dict
        error: Error message if any

    Progress & debugging:
        progress_tracker: ProgressTracker instance (not serialized in checkpoints)
        _tracer: QueryTracer instance for debugging (v0.3.37)

    Follow-up context (v0.3.37):
        last_result_metadata: {resource, count, has_more, next_url, filters}
        is_follow_up: Whether this query continues a previous one

    Note: Removed in v0.3.13:
        - context: Unused
        - plan: Legacy field, replaced by execution_plan
    """
    # Input
    query: str
    session_id: Optional[str]
    access_token: Optional[str]
    runtime_schema: Optional[dict]
    conversation_history: Optional[list]
    user_context: Optional[dict]  # v0.3.29: {user_id, username, role, company_id, company}

    # Processing
    active_schema: dict
    parsed: dict
    execution_plan: dict
    current_step: int
    step_results: List[dict]
    result: dict

    # Output
    summary: str
    response: dict
    error: Optional[str]

    # Progress & debugging (not serialized)
    progress_tracker: Any
    _tracer: Any  # v0.3.37: QueryTracer instance

    # Follow-up context (v0.3.37)
    last_result_metadata: Optional[dict]
    is_follow_up: bool
    filter_warnings: List[str]  # v0.3.37: Warnings about unmatched filters

    # Intent classification hint (rule-based pre-parse; LLM always parses)
    classification_confidence: float
    classification_hint: Optional[dict]


def build_api_workflow(processor, checkpointer=None, formatter_config: Optional[Dict[str, Any]] = None) -> "StateGraph":
    """
    Build the LangGraph workflow for API-only query processing with multi-step orchestration.
    
    Workflow:
    1. Load schema
    2. Parse query (OpenAI)
    3. Create execution plan (multi-step planning)
    4. Execute steps sequentially (with dependency resolution)
    5. Summarize results
    """
    graph = StateGraph(APIQueryState)
    
    # Initialize planner and LLM-based response formatter
    planner = ExecutionPlanner()
    formatter = ResponseFormatter(config=formatter_config)

    def load_schema(state: APIQueryState) -> Dict[str, Any]:
        """Node duty: Resolve and validate active schema only. No parsing or execution."""
        active_schema = processor._get_active_schema(state.get("runtime_schema"))
        if not active_schema:
            error_msg = constants.ERROR_NO_SCHEMA
            return {
                "response": enrich_response_with_follow_ups({
                    "success": False,
                    "data": None,
                    "summary": f"{constants.ERROR_PREFIX}{error_msg}",
                    "error": error_msg,
                    "query": state.get("query"),
                    "total_steps": 0,
                    "schema_type": "unknown",
                    "suggested_actions": default_error_suggestions(),
                    "follow_up_queries": default_error_follow_up_queries(),
                }),
            }

        if active_schema.get("type") not in ["api", "api_schema"]:
            error_msg = constants.ERROR_ONLY_API_SCHEMAS
            return {
                "active_schema": active_schema,
                "response": enrich_response_with_follow_ups({
                    "success": False,
                    "data": None,
                    "summary": f"{constants.ERROR_PREFIX}{error_msg}",
                    "error": error_msg,
                    "query": state.get("query"),
                    "total_steps": 0,
                    "schema_type": active_schema.get("type", "unknown"),
                    "suggested_actions": default_error_suggestions(),
                    "follow_up_queries": default_error_follow_up_queries(),
                }),
            }

        return {"active_schema": active_schema}

    def parse_query(state: APIQueryState) -> Dict[str, Any]:
        """Node duty: Intent analysis and query parsing only. Output: structured parsed (intent, resource, filters, etc.). No planning or execution."""
        # Progress update (v0.3.10)
        tracker = state.get("progress_tracker")
        tracer = state.get("_tracer")
        if tracker:
            tracker.update(ProgressStage.PARSING_QUERY, constants.PROGRESS_PARSING_QUERY)

        # v0.3.37: Detect follow-up queries
        query_text = state.get("query") or ""
        conversation_history = state.get("conversation_history") or []
        follow_up_classification = state.get("follow_up_classification") or classify_follow_up(
            query_text, conversation_history,
        )
        is_follow_up = follow_up_classification.get("is_follow_up", False)
        last_metadata = state.get("last_result_metadata")

        try:
            active_schema = state.get("active_schema") or {}
            conversation_history = state.get("conversation_history") or []
            parsed = processor._understand_query(
                query_text,
                active_schema,
                state.get("context"),
                conversation_history,
                state.get("user_context"),  # v0.3.29: pass user context for pronoun resolution
                classification_hint=state.get("classification_hint"),
            )

            # v0.3.37: Log to tracer
            if tracer:
                tracer.log_stage("parse", {
                    "intent": parsed.get("intent") if isinstance(parsed, dict) else None,
                    "resource": parsed.get("resource") if isinstance(parsed, dict) else None,
                    "filters": parsed.get("filters", {}) if isinstance(parsed, dict) else {},
                    "is_follow_up": is_follow_up,
                }, success=not isinstance(parsed, APIError))

        except Exception as e:
            logger.exception("Query parsing (LLM) failed: %s", e)
            error_msg = str(e)
            if tracer:
                tracer.log_stage("parse", {"error": error_msg}, success=False, error=error_msg)
            if tracker:
                tracker.update(ProgressStage.ERROR, f"{constants.ERROR_PREFIX}{error_msg}")
            return {
                "response": {
                    "success": False,
                    "data": None,
                    "summary": f"{constants.ERROR_PREFIX}{error_msg}",
                    "error": error_msg,
                    "query": state.get("query"),
                    "total_steps": 0,
                    "schema_type": state.get("active_schema", {}).get("type", "unknown"),
                }
            }
        
        # Log parsed structure (truncated for safety)
        try:
            parsed_preview = json.dumps(parsed)[: constants.LOG_PREVIEW_LENGTH]
        except TypeError:
            parsed_preview = str(parsed)[: constants.LOG_PREVIEW_LENGTH]
        logger.debug("Parsed query state: %s", parsed_preview)
        
        # Progress update (v0.3.10)
        if tracker and not isinstance(parsed, APIError):
            intent = parsed.get("intent", "unknown")
            resource = parsed.get("resource", "unknown")
            tracker.update(
                ProgressStage.INTENT_DETECTED,
                f"Intent identified: {intent} {resource}",
                intent=intent,
                resource=resource
            )
        
        if isinstance(parsed, APIError):
            if tracker:
                tracker.update(ProgressStage.ERROR, f"{constants.ERROR_PREFIX}{parsed.message}")
            error_msg = parsed.message
            return {
                "response": {
                    "success": False,
                    "data": None,
                    "summary": f"{constants.ERROR_PREFIX}{error_msg}",
                    "error": error_msg,
                    "query": state.get("query"),
                    "total_steps": 0,
                    "schema_type": state.get("active_schema", {}).get("type", "unknown"),
                }
            }
        return {"parsed": parsed}

    def create_execution_plan(state: APIQueryState) -> Dict[str, Any]:
        """Node duty: Turn parsed query into an execution plan (steps). Merges previous filters when merge_with_previous; does not parse or execute."""
        # Progress update (v0.3.10)
        tracker = state.get("progress_tracker")
        if tracker:
            tracker.update(ProgressStage.MATCHING_API, "Finding the right API...")
        
        try:
            # If parsing failed earlier, there will be no 'parsed' key.
            # In that case, just propagate the existing response or return
            # a clean parse error instead of raising KeyError('parsed').
            if "parsed" not in state:
                logger.warning("create_execution_plan called without parsed query; propagating prior response")
                existing_response = state.get("response")
                if existing_response is not None:
                    return {"response": existing_response}
                error_msg = "Query could not be parsed"
                return {
                    "response": {
                        "success": False,
                        "data": None,
                        "summary": f"Error: {error_msg}",
                        "error": error_msg,
                        "query": state.get("query"),
                        "total_steps": 0,
                        "schema_type": state.get("active_schema", {}).get("type", "unknown"),
                    }
                }

            parsed = state.get("parsed", {})

            if parsed.get("_needs_clarification") or parsed.get("question_type") == "needs_clarification":
                msg = parsed.get("_clarification_message") or (
                    "I need more information to complete this query. "
                    "Please be more specific about the resource, filters, or user context."
                )
                return {
                    "response": enrich_response_with_follow_ups({
                        "success": False,
                        "needs_info": True,
                        "message": msg,
                        "query": state.get("query"),
                        "suggested_actions": default_error_suggestions(),
                        "follow_up_queries": default_error_follow_up_queries(),
                    }),
                }
            
            # Check if we need to merge with previous query filters (v0.3.9, enhanced v0.3.12)
            conversation_history = state.get("conversation_history") or []
            query_text = state.get("query") or ""
            follow_up_classification = state.get("follow_up_classification") or {}
            should_merge = (
                parsed.get("merge_with_previous")
                or state.get("is_follow_up")
                or follow_up_classification.get("merge_with_previous")
                or (
                    follow_up_classification.get("is_follow_up")
                    and has_prior_context(conversation_history)
                )
            )
            if should_merge and conversation_history:
                logger.info("🔄 Merging filters from previous query (v0.3.12 enhanced)")
                
                # Extract previous filters from conversation history
                previous_filters = _extract_previous_filters(conversation_history)
                
                logger.info(f"   Previous filters extracted: {previous_filters}")
                
                if previous_filters:
                    # Merge filters: previous + current (current takes precedence for conflicts)
                    current_filters = parsed.get("filters", {})
                    
                    # Deep merge: Start with previous, then add/override with current
                    merged_filters = {}
                    
                    # Add all previous filters first
                    for key, value in previous_filters.items():
                        merged_filters[key] = value
                        logger.info(f"   ✓ Preserving previous filter: {key} = {value}")
                    
                    # Add/override with current filters
                    for key, value in current_filters.items():
                        if key in merged_filters:
                            logger.info(f"   ⚠️  Overriding filter {key}: {merged_filters[key]} → {value}")
                        else:
                            logger.info(f"   ✓ Adding new filter: {key} = {value}")
                        merged_filters[key] = value
                    
                    logger.info(f"   Final merged filters: {merged_filters}")
                    
                    # Update parsed query with merged filters
                    parsed["filters"] = merged_filters
                    
                    # Also merge entities for compatibility
                    previous_entities = _extract_previous_entities(conversation_history)
                    
                    if previous_entities:
                        current_entities = parsed.get("entities", {})
                        merged_entities = {**previous_entities, **current_entities}
                        parsed["entities"] = merged_entities
                        logger.info(f"   Merged entities: {merged_entities}")
                    
                    logger.info(f"✅ Filter merge complete! Total filters: {len(merged_filters)}")
                else:
                    logger.warning("   ⚠️  No previous filters found in conversation history")
                    logger.warning(f"   Conversation history length: {len(conversation_history)}")
                    # Debug: Print last assistant message
                    for msg in reversed(conversation_history):
                        if msg.get('role') == 'assistant':
                            logger.warning(f"   Last assistant message: {msg.get('content', '')[:200]}")
                            break
            elif should_merge:
                logger.warning("⚠️  Follow-up merge requested but no conversation_history available!")
            else:
                logger.info(
                    "ℹ️  No filter merging needed (merge_with_previous=%s, is_follow_up=%s)",
                    parsed.get("merge_with_previous"),
                    state.get("is_follow_up"),
                )

            # Anchor follow-ups to previous resource/filters (safety net after LLM parse)
            parsed = apply_follow_up_context(
                parsed,
                query_text,
                conversation_history,
                state.get("is_follow_up", False),
                classification=follow_up_classification,
            )

            active_schema = state.get("active_schema") or {}

            # Resolve __current_user_id__ etc. before planning (covers classify shortcut path)
            parsed = resolve_user_context_in_parsed(
                parsed, state.get("user_context"), query_text,
            )

            # Semantic filters (idempotent) — covers classify shortcut that skips LLM parse
            parsed = apply_semantic_filters(parsed, query_text, active_schema)

            execution_plan = planner.create_execution_plan(parsed, active_schema)
            total_steps = len(execution_plan.get("steps", []))
            if tracker:
                tracker.update(
                    ProgressStage.API_MATCHED,
                    constants.PROGRESS_API_MATCHED.format(api_name="request"),
                )
                tracker.update(
                    ProgressStage.PLAN_READY,
                    constants.PROGRESS_PLAN_READY.format(step_count=total_steps),
                    step_count=total_steps,
                )
            return {
                "execution_plan": execution_plan,
                "current_step": 0,
                "step_results": [],
                "parsed": parsed  # Updated parsed with merged filters
            }
        except Exception as e:
            logger.exception("Execution planning failed: %s", e)
            return {"error": str(e)}
    
    def execute_next_step(state: APIQueryState) -> Dict[str, Any]:
        """Node duty: Execute one step (API call or fetch_next_page). No parsing, planning, or summarization."""
        tracker = state.get("progress_tracker")
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_step_idx = state.get("current_step", 0)
        step_results = state.get("step_results", [])
        
        if current_step_idx >= len(steps):
            # All steps completed
            return {"current_step": current_step_idx}
        
        current_step_def = steps[current_step_idx]
        
        # "Show me more" / fetch next page: use stored next_url instead of building API plan
        if current_step_def.get("type") == "fetch_next_page":
            url = current_step_def.get("url")
            if not url:
                step_results.append({
                    "step_id": current_step_def.get("step_id"),
                    "step_description": current_step_def.get("description"),
                    "result": None,
                    "status": "failed",
                    "error": "No next page URL available",
                })
                return {
                    "current_step": current_step_idx + 1,
                    "step_results": step_results,
                    "result": {"data": None, "error": "No next page URL available"},
                    "error": "No next page URL available",
                }
            active_schema = state.get("active_schema") or {}
            page_result = processor._fetch_next_page(url, active_schema, state.get("access_token"))
            data = page_result.get("data") if not page_result.get("error") else None
            err = page_result.get("error")
            step_result = {
                "step_id": current_step_def.get("step_id"),
                "step_description": current_step_def.get("description"),
                "result": data,
                "status": "success" if not err else "failed",
                "error": err,
            }
            step_results.append(step_result)
            if tracker:
                tracker.update(ProgressStage.API_COMPLETED, "Next page fetched", endpoint="next page")
            return {
                "current_step": current_step_idx + 1,
                "step_results": step_results,
                "result": {"data": data, "error": err},
            }
        
        # Resolve dependencies - substitute variables from previous steps (uses planner's extract when present)
        resolved_step = _resolve_step_dependencies(current_step_def, step_results, all_steps=steps)
        resolved_step = merge_execution_context(resolved_step, state.get("parsed", {}))
        
        # Convert step to API plan format
        plan = {
            "type": "api",
            "intent": resolved_step.get("intent"),
            "resource": resolved_step.get("resource"),
            "entities": resolved_step.get("entities", {}),
            "filters": resolved_step.get("filters", {})
        }
        
        # Create API request using the matcher
        active_schema = state.get("active_schema") or {}
        api_plan = processor._create_api_plan(resolved_step, active_schema)
        
        if api_plan and api_plan.get("type") == "missing_info":
            return {
                "response": enrich_response_with_follow_ups({
                    "success": False,
                    "needs_info": True,
                    "message": api_plan.get("message"),
                    "missing_fields": api_plan.get("missing_fields"),
                    "context": api_plan.get("context"),
                    "query": state.get("query"),
                    "suggested_actions": default_error_suggestions(),
                    "follow_up_queries": default_error_follow_up_queries(),
                }),
            }

        if not api_plan or api_plan.get("type") == "error":
            return {
                "error": f"Step {current_step_idx + 1} planning failed: {api_plan.get('error', 'Unknown error')}"
            }
        
        # Execute the API call
        result = processor._execute_api(
            api_plan,
            active_schema,
            state.get("access_token"),
            resolved_step,
        )
        data = result.get("data")
        # Pagination as a step: when step has fetch_all_pages, fetch next page(s) and merge.
        # Safety cap only to avoid infinite loop on buggy APIs; see todo.md § Limits affecting correctness.
        if current_step_def.get("fetch_all_pages") and isinstance(data, dict) and data.get("next"):
            pages = 1
            while data.get("next") and pages < constants.SAFETY_MAX_PAGES:
                next_url = data.get("next")
                page_result = processor._fetch_next_page(next_url, active_schema, state.get("access_token"))
                if page_result.get("error"):
                    break
                next_data = page_result.get("data") or {}
                if "results" in data and "results" in next_data and isinstance(data.get("results"), list) and isinstance(next_data.get("results"), list):
                    data["results"].extend(next_data["results"])
                data["next"] = next_data.get("next")
                data["count"] = next_data.get("count", data.get("count", 0))
                pages += 1
                if tracker:
                    tracker.update(ProgressStage.EXECUTING_API, constants.PROGRESS_FETCHING_PAGE.format(page=pages))
            if pages >= constants.SAFETY_MAX_PAGES and data.get("next"):
                logger.warning(constants.PAGINATION_STEP_SAFETY_CAP_WARNING.format(max_pages=constants.SAFETY_MAX_PAGES))
            result = {**result, "data": data}
        
        # Store result (include API call details for debugging/inspection)
        step_result = {
            "step_id": current_step_def.get("step_id"),
            "step_description": current_step_def.get("description"),
            "result": result.get("data"),
            "status": "success" if not result.get("error") else "failed",
            "error": result.get("error"),
            "api_call": {
                "endpoint": api_plan.get("endpoint"),
                "method": api_plan.get("method"),
                "params": api_plan.get("params"),
            },
        }
        
        step_results.append(step_result)
        if tracker:
            endpoint = api_plan.get("endpoint", "API") if isinstance(api_plan, dict) else "API"
            tracker.update(
                ProgressStage.API_COMPLETED,
                f"Step {current_step_idx + 1} completed: {endpoint}",
                endpoint=endpoint,
            )
        # v0.3.37: Propagate filter warnings from API result to state
        api_warnings = result.get("warnings", [])
        existing_warnings = state.get("filter_warnings", [])
        all_warnings = existing_warnings + api_warnings

        out = {
            "current_step": current_step_idx + 1,
            "step_results": step_results,
            "result": result,  # Last result for compatibility
            "filter_warnings": all_warnings,  # v0.3.37: Accumulate filter warnings
        }
        # When all steps failed, set error so route_after_step can send to format_error
        if step_result.get("status") == "failed" and (current_step_idx + 1) >= len(steps):
            if all(sr.get("status") == "failed" for sr in step_results):
                err_parts = [f"Step {i + 1}: {sr.get('error', 'Unknown')}" for i, sr in enumerate(step_results)]
                out["error"] = "All steps failed. " + "; ".join(err_parts)
        return out

    def execute_plan(state: APIQueryState) -> Dict[str, Any]:
        """Legacy: not registered as a graph node. The graph uses execute_next_step for execution. Kept for reference only."""
        plan = state.get("plan", {})
        if plan.get("type") != "api":
            return {
                "result": {
                    "data": None,
                    "error": plan.get("error", "Unsupported plan type"),
                }
            }

        # Progress update (v0.3.10)
        tracker = state.get("progress_tracker")
        if tracker:
            endpoint = plan.get("endpoint", "API")
            tracker.update(ProgressStage.EXECUTING_API, f"Calling {endpoint}...", endpoint=endpoint)

        # Enhanced logging for v0.3.9 - see what filters are being used
        parsed = state.get("parsed", {})
        logger.info(f"Executing API call with filters: {parsed.get('filters', {})}")
        logger.info(f"Display mode: {parsed.get('display_mode', 'summary')}")
        logger.info(f"Merge with previous: {parsed.get('merge_with_previous', False)}")

        active_schema = state.get("active_schema") or {}
        result = processor._execute_api(plan, active_schema, state.get("access_token"), parsed)
        
        # Log the endpoint that was called
        if result:
            logger.info(f"API call result: endpoint={result.get('endpoint')}, "
                       f"method={result.get('method')}, "
                       f"data_count={len(result.get('data', [])) if isinstance(result.get('data'), list) else 'N/A'}")
        
        return {"result": result}

    def format_missing_info(state: APIQueryState) -> Dict[str, Any]:
        """Node duty: Return a structured response when required info is missing. No execution."""
        plan = state.get("plan", {})
        return {
            "response": enrich_response_with_follow_ups({
                "success": False,
                "needs_info": True,
                "message": plan.get("message"),
                "missing_fields": plan.get("missing_fields"),
                "context": plan.get("context"),
                "query": state.get("query"),
                "suggested_actions": default_error_suggestions(),
                "follow_up_queries": default_error_follow_up_queries(),
            }),
        }

    def format_error(state: APIQueryState) -> Dict[str, Any]:
        """Node duty: Format error response with consistent structure. No parsing, execution, or summarization."""
        plan_error = state.get("plan", {}).get("error")
        error = plan_error or state.get("error") or constants.ERROR_UNKNOWN

        # v0.3.29: Check if this is an unknown intent / no matching resource error
        # If so, return helpful suggestions instead of a raw error
        is_unknown_intent = (
            "No matching API" in str(error) or
            "could not be parsed" in str(error).lower() or
            "unknown" in str(error).lower() or
            "not found" in str(error).lower()
        )

        if is_unknown_intent:
            # Return the detailed matcher/orchestrator error as the main summary so the
            # user sees available resources and suggestions instead of a generic message.
            return {
                "response": enrich_response_with_follow_ups({
                    "success": False,
                    "data": None,
                    "summary": str(error),
                    "error": error,
                    "query": state.get("query"),
                    "total_steps": 0,
                    "schema_type": state.get("active_schema", {}).get("type") if state.get("active_schema") else "unknown",
                    "suggested_actions": [
                        "Try rephrasing your question",
                        "Ask about service orders, reports, inventory, or technicians",
                        "Use specific resource names like 'service orders' or 'consumables'",
                    ],
                    "follow_up_queries": default_error_follow_up_queries(),
                    "is_unknown_intent": True,
                }),
            }

        return {
            "response": enrich_response_with_follow_ups({
                "success": False,
                "data": None,
                "summary": f"{constants.ERROR_PREFIX}{error}",
                "error": error,
                "query": state.get("query"),
                "total_steps": 0,
                "schema_type": state.get("active_schema", {}).get("type") if state.get("active_schema") else "unknown",
                "suggested_actions": default_error_suggestions(),
                "follow_up_queries": default_error_follow_up_queries(),
            }),
        }

    def summarize(state: APIQueryState) -> Dict[str, Any]:
        """Node duty: Format and summarize execution results only (count handling, pagination info, LLM summary). No parsing, planning, or execution."""
        # Progress update (v0.3.10)
        tracker = state.get("progress_tracker")
        if tracker:
            tracker.update(ProgressStage.SUMMARIZING, constants.PROGRESS_SUMMARIZING)
        
        execution_plan = state.get("execution_plan", {})
        step_results = state.get("step_results", [])
        result = state.get("result", {})
        parsed = state.get("parsed", {})
        display_mode = parsed.get("display_mode", "summary")  # v0.3.7
        question_type = parsed.get("question_type", "list")  # v0.3.12
        
        # Log high-level summarize context
        logger.info(
            "Summarizing result | is_multi_step=%s | display_mode=%s | question_type=%s",
            execution_plan.get("is_multi_step", False),
            display_mode,
            question_type,
        )

        # Check if multi-step
        is_multi_step = execution_plan.get("is_multi_step", False)
        
        # Initialize pagination_info so it's always defined for trace logging and
        # last_result_metadata, regardless of which branch we take below.
        pagination_info: Dict[str, Any] = {}

        if is_multi_step:
            # Summarize multi-step execution
            all_data = [sr.get("result") for sr in step_results if sr.get("status") == "success"]
            has_error = any(sr.get("status") == "failed" for sr in step_results)
            
            # v0.3.9: Better handling of display_mode for multi-step
            if display_mode == "full":
                summary = constants.SUMMARY_RETRIEVED_ALL_ITEMS_STEPS.format(count=len(all_data), steps=len(step_results))
                logger.info(f"Multi-step display mode = full: returning all {len(all_data)} results")
            elif display_mode == "detailed":
                summary = processor._summarize_multi_step_result(
                    step_results,
                    state.get("query", ""),
                )
                logger.info(f"Multi-step display mode = detailed: detailed view")
            else:
                # Summary mode: provide concise multi-step summary
                summary = processor._summarize_multi_step_result(
                    step_results,
                    state.get("query", ""),
                )
                logger.info(f"Multi-step display mode = summary: standard summary")
            
            # Attempt richer LLM-based formatting for multi-step results
            formatted = None
            fmt_format = None
            format_error = None
            try:
                if all_data:
                    fmt = formatter.format_response(
                        data=all_data if len(all_data) > 1 else all_data[0],
                        query=state.get("query", ""),
                        format_type="auto",
                        context={
                            "resource": parsed.get("resource"),
                            "schema": state.get("active_schema", {}),
                            "question_type": question_type,
                            "display_mode": display_mode,
                        },
                    )
                    if fmt.get("summary"):
                        summary = fmt["summary"]
                    formatted = fmt.get("formatted")
                    fmt_format = fmt.get("format")
            except Exception as e:
                logger.error("ResponseFormatter (multi-step) failed: %s", e)
                format_error = str(e)
            
            # v0.3.37: Get filter warnings for multi-step
            filter_warnings = state.get("filter_warnings", [])

            # Derive pagination from final step for follow-up suggestions
            last_step_data = step_results[-1].get("result") if step_results else None
            pagination_info = _analyze_pagination(last_step_data) if last_step_data else {}

            response = enrich_response_with_follow_ups({
                "success": not has_error,
                "data": all_data if len(all_data) > 1 else (all_data[0] if all_data else None),
                "summary": summary,
                "query": state.get("query"),
                "display_mode": display_mode,  # v0.3.7
                "pagination": pagination_info,
                "execution_plan": [
                    {
                        "step": sr.get("step_id"),
                        "description": sr.get("step_description"),
                        "status": sr.get("status")
                    }
                    for sr in step_results
                ],
                "total_steps": len(step_results),
                "schema_type": state.get("active_schema", {}).get("type"),
            }, pagination_info, parsed, last_step_data)

            # v0.3.37: Add filter warnings to multi-step response
            if filter_warnings:
                response["filter_warnings"] = filter_warnings
                warning_text = "\n".join(f"⚠️ {w}" for w in filter_warnings)
                response["summary"] = f"{warning_text}\n\n{constants.FILTER_WARNING_NOTE}\n\n{summary}"

            if formatted is not None:
                response["formatted"] = formatted
            if fmt_format is not None:
                response["format"] = fmt_format
            if format_error is not None:
                response["format_error"] = format_error

            if has_error:
                failed_steps = [sr for sr in step_results if sr.get("status") == "failed"]
                response["errors"] = [sr.get("error") for sr in failed_steps]
        else:
            # Single-step execution (legacy path)
            # v0.3.11: Enhanced handling with accurate counts and pagination (Issues #2-7)
            # v0.3.12: Count-specific formatting
            data = result.get("data", {})
            # Automatic pagination: only when display_mode is "full" do we fetch all pages.
            # When display_mode is "summary" or "detailed", return first page so user can say "show me more" for next page.
            pages_fetched = 1
            if display_mode == "full":
                while isinstance(data, dict) and data.get("next") and pages_fetched < constants.SAFETY_MAX_PAGES:
                    next_url = data.get("next")
                    active_schema = state.get("active_schema") or {}
                    page_result = processor._fetch_next_page(next_url, active_schema, state.get("access_token"))
                    if page_result.get("error"):
                        logger.warning("Automatic pagination stopped: %s", page_result.get("error"))
                        break
                    next_data = page_result.get("data")
                    if not next_data or not isinstance(next_data, dict):
                        break
                    if "results" in data and "results" in next_data and isinstance(data["results"], list) and isinstance(next_data.get("results"), list):
                        data["results"].extend(next_data["results"])
                    data["next"] = next_data.get("next")
                    data["count"] = next_data.get("count", data.get("count", 0))
                    pages_fetched += 1
                    if tracker:
                        tracker.update(ProgressStage.EXECUTING_API, constants.PROGRESS_FETCHING_PAGE.format(page=pages_fetched))
                if pages_fetched >= constants.SAFETY_MAX_PAGES and isinstance(data, dict) and data.get("next"):
                    logger.warning(constants.PAGINATION_SAFETY_CAP_WARNING.format(max_pages=constants.SAFETY_MAX_PAGES))
            # Analyze response to extract accurate counts (Issue #2-4)
            pagination_info = _analyze_pagination(data)
            
            # v0.3.12: Special handling for COUNT questions
            if question_type == "count":
                total_count = pagination_info['total_count']
                resource = parsed.get('resource', 'items').replace('_', ' ')
                
                # Get filter details for context
                filters = parsed.get('filters', {})
                filter_desc = ""
                
                # Build filter description (e.g., "named BOROSCOPE in low-stock inventory")
                filter_parts = []
                if 'name' in filters:
                    name_value = filters['name'].get('value', '')
                    filter_parts.append(f"named {name_value}")
                if 'status' in filters:
                    status_value = filters['status'].get('value', '')
                    filter_parts.append(f"{status_value}")
                
                if filter_parts:
                    filter_desc = " ".join(filter_parts)
                
                # Generate count-specific summary
                if total_count == 0:
                    if filter_desc:
                        summary = f"There are no {resource} {filter_desc}"
                    else:
                        summary = f"There are no {resource} matching your criteria"
                elif total_count == 1:
                    if filter_desc:
                        summary = f"There is 1 {resource.rstrip('s')} {filter_desc}"
                    else:
                        summary = f"There is 1 {resource.rstrip('s')}"
                else:
                    if filter_desc:
                        summary = f"There are {total_count} {resource} {filter_desc}"
                    else:
                        summary = f"There are {total_count} {resource}"
                
                logger.info(f"🔢 COUNT question detected: {summary}")
                
                # For count questions, return minimal data structure
                final_data = {
                    "count": total_count,
                    "resource": resource,
                    "filters_applied": filters,
                    "items": data.get('results', data) if isinstance(data, dict) and 'results' in data else (data if isinstance(data, list) else [])
                }
                
                response = enrich_response_with_follow_ups({
                    "success": True,
                    "data": final_data,
                    "summary": summary,
                    "query": state.get("query"),
                    "question_type": "count",
                    "display_mode": "count",
                    "pagination": pagination_info,
                    "total_steps": 1,
                    "schema_type": state.get("active_schema", {}).get("type"),
                }, pagination_info, parsed, data)
                
                # Complete progress
                if tracker:
                    tracker.update(ProgressStage.COMPLETED, f"Done! Found {total_count} items ✓")

                response = enrich_api_response(
                    response,
                    {
                        "parsed": parsed,
                        "step_results": step_results,
                        "filter_warnings": state.get("filter_warnings", []),
                        "result": result,
                    },
                )
                return {"summary": summary, "response": response}
            
            # Non-count questions continue with normal logic
            if display_mode == "full":
                # Return full data without summarization (Issue #3: ALL items)
                if isinstance(data, dict) and 'results' in data:
                    full_results = data['results']
                    count = pagination_info['actual_count']
                    total = pagination_info['total_count']
                    
                    if total > count:
                        summary = constants.SUMMARY_RETRIEVED_COUNT_OF_TOTAL.format(count=count, total=total)
                    else:
                        summary = constants.SUMMARY_RETRIEVED_ALL_COUNT.format(count=count)
                    final_data = full_results  # Return ALL items from response (Issue #3)
                elif isinstance(data, list):
                    summary = constants.SUMMARY_RETRIEVED_ALL_LEN.format(count=len(data))
                    final_data = data  # Return ALL items (Issue #3)
                else:
                    summary = constants.SUMMARY_RETRIEVED_DATA
                    final_data = data
                    
                logger.info(f"Display mode = full: returning {len(final_data) if isinstance(final_data, list) else 'N/A'} items")
            elif display_mode == "detailed":
                # Return full data with detailed summary
                summary = processor._summarize_result_v2(
                    result,
                    state.get("query", ""),
                    parsed,
                    pagination_info,
                    schema=state.get("active_schema"),
                )
                final_data = data
                logger.info(f"Display mode = detailed: returning detailed view")
            else:
                # Default: summary mode (Issue #4: accurate summary)
                summary = processor._summarize_result_v2(
                    result,
                    state.get("query", ""),
                    parsed,
                    pagination_info,
                    schema=state.get("active_schema"),
                )
                
                # For summary mode, return structured data with pagination info (Issue #6)
                if isinstance(data, dict) and 'results' in data:
                    final_data = {
                        "total_count": pagination_info['total_count'],
                        "shown_count": pagination_info['actual_count'],
                        "has_more": pagination_info['has_more'],
                        "results": data['results']  # Return ALL items from current page (Issue #3)
                    }
                elif isinstance(data, list):
                    final_data = {
                        "total_count": len(data),
                        "shown_count": len(data),
                        "has_more": False,
                        "results": data  # Return ALL items (Issue #3)
                    }
                else:
                    final_data = data
                    
                logger.info(f"Display mode = summary: returning structured data with counts")
            
            has_error = result.get("error") is not None
            
            # Attempt richer LLM-based formatting for non-count, single-step results
            formatted = None
            fmt_format = None
            format_error = None
            if question_type != "count":
                try:
                    # BUG FIX (v0.3.36): Pass the actual results array to formatter, not wrapper
                    # This ensures LLM sees ALL items and can list them all
                    if isinstance(final_data, dict) and 'results' in final_data:
                        # Pass the results array directly, with pagination context
                        format_data = final_data['results']
                    else:
                        format_data = final_data if final_data is not None else data

                    fmt = formatter.format_response(
                        data=format_data,
                        query=state.get("query", ""),
                        format_type="auto",
                        context={
                            "resource": parsed.get("resource"),
                            "schema": state.get("active_schema", {}),
                            "question_type": question_type,
                            "display_mode": display_mode,
                            "filters": parsed.get("filters", {}),
                        },
                    )
                    if fmt.get("summary"):
                        summary = fmt["summary"]

                    # v0.3.37: Fix JSON wrapper issue - extract clean text
                    summary = _unwrap_json_response(summary)

                    formatted = fmt.get("formatted")
                    fmt_format = fmt.get("format")
                except Exception as e:
                    logger.error("ResponseFormatter (single-step) failed: %s", e)
                    format_error = str(e)

            # v0.3.37: Add filter warnings if any
            filter_warnings = state.get("filter_warnings", [])

            response = enrich_response_with_follow_ups({
                "success": not has_error,
                "data": final_data,
                "summary": summary,
                "query": state.get("query"),
                "display_mode": display_mode,
                "pagination": pagination_info,  # v0.3.11: Include pagination info (Issue #2)
                "total_steps": 1,
                "schema_type": state.get("active_schema", {}).get("type"),
            }, pagination_info, parsed, data)

            # v0.3.37: Add filter warnings to response
            if filter_warnings:
                response["filter_warnings"] = filter_warnings
                # Prepend warning to summary
                warning_text = "\n".join(f"⚠️ {w}" for w in filter_warnings)
                response["summary"] = f"{warning_text}\n\n{constants.FILTER_WARNING_NOTE}\n\n{summary}"

            if formatted is not None:
                response["formatted"] = formatted
            if fmt_format is not None:
                response["format"] = fmt_format
            if format_error is not None:
                response["format_error"] = format_error

            if has_error:
                response["error"] = result.get("error")

        # v0.3.37: Store result metadata for follow-up queries
        last_result_metadata = {
            "resource": parsed.get("resource"),
            "filters": parsed.get("filters", {}),
            "count": pagination_info.get("total_count", 0) if 'pagination_info' in dir() else 0,
            "has_more": pagination_info.get("has_more", False) if 'pagination_info' in dir() else False,
            "next_url": pagination_info.get("next_url") if 'pagination_info' in dir() else None,
        }

        # v0.3.37: Log to tracer and save
        tracer = state.get("_tracer")
        if tracer:
            tracer.log_stage("summarize", {
                "success": response.get("success", False),
                "data_count": pagination_info.get("actual_count", 0) if 'pagination_info' in dir() else 0,
                "has_filter_warnings": len(filter_warnings) > 0 if 'filter_warnings' in dir() else False,
            })
            _tracer_store.save_trace(tracer)

        # Final progress update (v0.3.10)
        if tracker:
            if response.get("success"):
                tracker.update(ProgressStage.COMPLETED, constants.PROGRESS_DONE)
            else:
                tracker.update(ProgressStage.ERROR, constants.PROGRESS_ERROR_OCCURRED)

        response = enrich_api_response(
            response,
            {
                "parsed": parsed,
                "step_results": step_results,
                "filter_warnings": state.get("filter_warnings", []),
                "result": result,
            },
        )

        return {"summary": summary, "response": response, "last_result_metadata": last_result_metadata, "parsed": parsed}

    def handle_follow_up(state: APIQueryState) -> Dict[str, Any]:
        """
        Handle follow-up queries like "show me next 2" or "show me first 3" (v0.3.38).

        This node:
        1. Checks if the query is a follow-up
        2. Extracts context from conversation history
        3. If follow-up + has next_url: fetches next page
        4. If follow-up + no context: returns helpful error
        5. If not follow-up: continues to normal parsing
        """
        query = state.get("query") or ""
        conversation_history = state.get("conversation_history") or []
        tracker = state.get("progress_tracker")

        # LLM classifies whether this query continues the previous turn
        follow_up_classification = classify_follow_up(query, conversation_history)
        is_follow_up = follow_up_classification.get("is_follow_up", False)

        if not is_follow_up:
            return {"is_follow_up": False, "follow_up_classification": follow_up_classification}

        # Extract metadata from previous results
        last_metadata = extract_last_result_metadata(conversation_history)
        follow_up_type = follow_up_classification.get("follow_up_type", "standalone")
        requested_limit = _extract_limit_from_query(query)

        logger.info(
            f"Follow-up detected: type={follow_up_type}, limit={requested_limit}, "
            f"has_context={bool(last_metadata)}, next_url={last_metadata.get('next_url', 'None')}"
        )

        if tracker:
            tracker.update(
                ProgressStage.PARSING_QUERY,
                f"Processing follow-up query: {follow_up_type}"
            )

        # Handle based on follow-up type
        if follow_up_type == "next_page":
            next_url = last_metadata.get("next_url")
            if next_url:
                # Fetch next page using the stored URL
                try:
                    if tracker:
                        tracker.update(ProgressStage.EXECUTING_API, "Fetching next page...")

                    active_schema = state.get("active_schema") or {}
                    page_result = processor._fetch_next_page(
                        next_url,
                        active_schema,
                        state.get("access_token"),
                    )

                    if page_result.get("error"):
                        return {
                            "is_follow_up": True,
                            "response": enrich_response_with_follow_ups({
                                "success": False,
                                "error": page_result["error"],
                                "summary": f"Failed to fetch next page: {page_result['error']}",
                                "query": query,
                                "suggested_actions": default_error_suggestions(),
                                "follow_up_queries": default_error_follow_up_queries(),
                            }),
                        }

                    # Store result for summarization
                    data = page_result.get("data", {})

                    # v0.3.40: Apply requested_limit to slice results
                    # When user says "show me next 2", we should only show 2 items
                    if requested_limit and requested_limit < 100:  # Sanity check
                        if isinstance(data, dict) and 'results' in data:
                            original_count = len(data.get('results', []))
                            data = {
                                **data,
                                'results': data['results'][:requested_limit]
                            }
                            logger.info(
                                f"Applied follow-up limit: showing {len(data['results'])} of "
                                f"{original_count} results (requested: {requested_limit})"
                            )
                        elif isinstance(data, list):
                            original_count = len(data)
                            data = data[:requested_limit]
                            logger.info(
                                f"Applied follow-up limit: showing {len(data)} of "
                                f"{original_count} results (requested: {requested_limit})"
                            )
                        # Update page_result with sliced data
                        page_result = {**page_result, 'data': data}

                    pagination_info = _analyze_pagination(data)

                    # Create parsed context for summarization
                    parsed = {
                        "intent": "read",
                        "resource": last_metadata.get("resource", "items"),
                        "filters": last_metadata.get("filters", {}),
                        "display_mode": "summary",
                        "question_type": "list",
                        "limit": requested_limit,  # v0.3.40: Pass limit for summary
                    }

                    return {
                        "is_follow_up": True,
                        "parsed": parsed,
                        "result": page_result,
                        "step_results": [{
                            "step_id": 1,
                            "step_description": "Fetch next page",
                            "result": data,
                            "status": "success",
                        }],
                        "last_result_metadata": {
                            "resource": last_metadata.get("resource"),
                            "filters": last_metadata.get("filters", {}),
                            "count": pagination_info.get("total_count", 0),
                            "has_more": pagination_info.get("has_more", False),
                            "next_url": pagination_info.get("next_url"),
                        },
                        # Skip to summarize
                        "execution_plan": {"steps": [{"step_id": 1}], "is_multi_step": False},
                        "current_step": 1,
                    }
                except Exception as e:
                    logger.error(f"Failed to fetch next page: {e}")
                    return {
                        "is_follow_up": True,
                        "response": enrich_response_with_follow_ups({
                            "success": False,
                            "error": str(e),
                            "summary": f"Failed to fetch next page: {e}",
                            "query": query,
                            "suggested_actions": default_error_suggestions(),
                            "follow_up_queries": default_error_follow_up_queries(),
                        }),
                    }
            else:
                # No next_url - provide helpful message
                resource = last_metadata.get("resource", "items")
                if last_metadata:
                    msg = (
                        f"There are no more {resource} to show. "
                        f"The previous query returned all available results."
                    )
                else:
                    msg = (
                        "I don't have context from a previous query to show more results. "
                        "Please start with a new query like 'list service orders' or 'show all users'."
                    )

                no_more_pagination = {
                    "total_count": last_metadata.get("count", 0),
                    "actual_count": last_metadata.get("count", 0),
                    "has_more": False,
                }
                follow_ups = generate_follow_up_queries(
                    no_more_pagination,
                    {"resource": last_metadata.get("resource", resource)},
                )
                return {
                    "is_follow_up": True,
                    "response": enrich_response_with_follow_ups({
                        "success": True,
                        "data": None,
                        "summary": msg,
                        "query": query,
                        "total_steps": 0,
                        "suggested_actions": [
                            f"Start fresh: list {resource}",
                            "Try 'show me next 2' after a list query",
                        ],
                        "follow_up_queries": follow_ups or default_error_follow_up_queries(),
                    }),
                }

        elif follow_up_type in ("first_n", "last_n"):
            # User wants first/last N items from previous results
            if not last_metadata or not last_metadata.get("resource"):
                return {
                    "is_follow_up": True,
                    "response": enrich_response_with_follow_ups({
                        "success": True,
                        "data": None,
                        "summary": (
                            "I don't have context from a previous query. "
                            "Please start with a query like 'list service orders' first, "
                            "then ask for 'show me first 3'."
                        ),
                        "query": query,
                        "total_steps": 0,
                        "suggested_actions": default_error_suggestions(),
                        "follow_up_queries": default_error_follow_up_queries(),
                    }),
                }

            # Re-run the previous query with a limit
            # Continue to normal flow but with injected context
            return {
                "is_follow_up": True,
                "last_result_metadata": last_metadata,
                # Let normal parsing handle it, but with context hint
                "context": {
                    "previous_resource": last_metadata.get("resource"),
                    "previous_filters": last_metadata.get("filters", {}),
                    "requested_limit": requested_limit,
                    "follow_up_type": follow_up_type,
                },
            }

        elif follow_up_type in ("reference", "refinement"):
            # Reference or refinement ("show those", "and which company?")
            if not last_metadata or not last_metadata.get("resource"):
                return {
                    "is_follow_up": True,
                    "response": enrich_response_with_follow_ups({
                        "success": True,
                        "data": None,
                        "summary": (
                            "I'm not sure what you're referring to. "
                            "Please be more specific, like 'show all service orders' or 'list users'."
                        ),
                        "query": query,
                        "total_steps": 0,
                        "suggested_actions": default_error_suggestions(),
                        "follow_up_queries": default_error_follow_up_queries(),
                    }),
                }

            # Continue to normal flow with context
            return {
                "is_follow_up": True,
                "follow_up_classification": follow_up_classification,
                "last_result_metadata": last_metadata,
                "context": {
                    "previous_resource": last_metadata.get("resource"),
                    "previous_filters": last_metadata.get("filters", {}),
                },
            }

        # Unknown follow-up type - continue normal flow
        return {
            "is_follow_up": True,
            "follow_up_classification": follow_up_classification,
            "last_result_metadata": last_metadata,
        }

    def route_after_follow_up(state: APIQueryState) -> str:
        """Route after follow-up handling (v0.3.38)."""
        # If we already have a response (follow-up was fully handled), go to end
        if state.get("response"):
            return "end"
        # If we have result from follow-up (fetched next page), go to summarize
        if state.get("is_follow_up") and state.get("result"):
            return "summarize"
        # Follow-ups need LLM parsing with conversation history — skip fast classifier
        if state.get("is_follow_up"):
            return "parse"
        return "classify"

    def route_after_schema(state: APIQueryState) -> str:
        if state.get("response"):
            return "end"
        return "handle_follow_up"  # v0.3.38: Check for follow-up queries first

    def route_after_planning(state: APIQueryState) -> str:
        """Route after execution planning."""
        if state.get("response"):
            return "end"
        if state.get("error"):
            return "format_error"
        execution_plan = state.get("execution_plan", {})
        if not execution_plan or not execution_plan.get("steps"):
            return "format_error"
        return "execute_step"
    
    def route_after_step(state: APIQueryState) -> str:
        """Route after executing a step - continue, summarize (with partial results), or format_error (all failed)."""
        if state.get("response"):
            return "end"
        execution_plan = state.get("execution_plan", {})
        current_step = state.get("current_step", 0)
        total_steps = len(execution_plan.get("steps", []))
        step_results = state.get("step_results", [])

        if current_step < total_steps:
            return "execute_step"  # More steps to execute
        # All steps complete: if all failed, route to format_error for a single clear error response
        if step_results and all(sr.get("status") == "failed" for sr in step_results) and state.get("error"):
            return "format_error"
        return "summarize"  # Partial or full success → summarize (with errors list when some failed)

    def classify_intent(state: APIQueryState) -> Dict[str, Any]:
        """
        Rule-based pre-classification hint for the LLM parser (never skips LLM).

        Returns optional classification_hint — the LLM parse step always runs.
        """
        active_schema = state.get("active_schema")
        if not active_schema:
            return {"classification_confidence": 0.0, "classification_hint": None}

        try:
            classifier = IntentClassifier(active_schema)
            query_text = state.get("query") or ""
            classification, confidence = classifier.classify(query_text)

            logger.debug(
                f"Intent classification hint: confidence={confidence:.2f}, "
                f"resource={classification.resource if classification else 'None'}"
            )

            hint = None
            if classification and confidence >= constants.CLASSIFIER_LOW_CONFIDENCE:
                hint = {**classification.to_dict(), "confidence": confidence}
                logger.info(
                    "Classification hint for LLM: %s on %s (confidence=%.2f)",
                    classification.intent,
                    classification.resource,
                    confidence,
                )

            return {"classification_confidence": confidence, "classification_hint": hint}

        except Exception as e:
            logger.warning(f"Intent classification failed: %s", e)
            return {"classification_confidence": 0.0, "classification_hint": None}

    # Add nodes
    graph.add_node("load_schema", load_schema)
    graph.add_node("handle_follow_up", handle_follow_up)  # v0.3.38: Follow-up query handler
    graph.add_node("classify", classify_intent)
    graph.add_node("parse", parse_query)
    graph.add_node("create_plan", create_execution_plan)
    graph.add_node("execute_step", execute_next_step)
    graph.add_node("summarize", summarize)
    graph.add_node("format_error", format_error)

    # Set entry point
    graph.set_entry_point("load_schema")

    # Add edges
    graph.add_conditional_edges(
        "load_schema",
        route_after_schema,
        {
            "handle_follow_up": "handle_follow_up",  # v0.3.38: Route to follow-up handler
            "end": END,
        },
    )
    # v0.3.38: Route after follow-up handling
    graph.add_conditional_edges(
        "handle_follow_up",
        route_after_follow_up,
        {
            "classify": "classify",
            "parse": "parse",
            "summarize": "summarize",  # Follow-up already fetched data
            "end": END,  # Follow-up returned response directly
        },
    )
    graph.add_edge("classify", "parse")
    graph.add_edge("parse", "create_plan")
    graph.add_conditional_edges(
        "create_plan",
        route_after_planning,
        {
            "format_error": "format_error",
            "execute_step": "execute_step",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "execute_step",
        route_after_step,
        {
            "execute_step": "execute_step",  # Loop for sequential execution
            "summarize": "summarize",
            "format_error": "format_error",   # All steps failed
            "end": END,
        },
    )
    graph.add_edge("summarize", END)
    graph.add_edge("format_error", END)

    # Compile with checkpointer if provided
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()
