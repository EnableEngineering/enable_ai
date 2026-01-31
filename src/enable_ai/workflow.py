from typing import Any, Dict, Optional, TypedDict, List
import re
import json

from langgraph.graph import StateGraph, END

from .types import APIError
from .execution_planner import ExecutionPlanner
from .utils import setup_logger, get_openai_client, CREATIVE_TEMP
from .progress_tracker import ProgressStage
from .response_formatter import ResponseFormatter
from . import constants

# Module-level logger
logger = setup_logger('enable_ai.workflow')


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
        'is_paginated': False
    }
    
    # Check for paginated response (Django REST framework style)
    if isinstance(data, dict) and 'results' in data:
        info['is_paginated'] = True
        info['total_count'] = data.get('count', 0)
        results = data.get('results', [])
        info['actual_count'] = len(results)
        info['has_more'] = data.get('next') is not None
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


def _generate_suggestions(pagination_info: Dict[str, Any], parsed: Dict[str, Any], data: Any) -> List[str]:
    """
    Generate contextual suggestions based on result count (v0.3.11).
    
    Fixes Issue #7: Contextual help.
    
    Args:
        pagination_info: Pagination analysis
        parsed: Parsed query data
        data: Response data
        
    Returns:
        List of suggested next actions
    """
    suggestions = []
    total = pagination_info['total_count']
    has_more = pagination_info['has_more']
    
    if total == 0:
        suggestions.append(constants.SUGGESTION_TRY_BROADER)
    elif total == 1:
        suggestions.append(constants.SUGGESTION_SHOW_DETAILS)
    elif total <= 5:
        suggestions.append(constants.SUGGESTION_SHOW_DETAILS_ITEM)
    elif total <= constants.TABLE_ROW_SAMPLE:
        suggestions.append(constants.SUGGESTION_FILTER_SPECIFIC)
    elif total > constants.TABLE_ROW_SAMPLE:
        suggestions.append(constants.SUGGESTION_NARROW_FILTERS)
        if has_more:
            suggestions.append(constants.SUGGESTION_SHOW_MORE)
    
    return suggestions[: constants.SUGGESTIONS_MAX]


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
        step_json = re.sub(pattern, replacement, step_json)

    return json.loads(step_json)


class APIQueryState(TypedDict, total=False):
    """
    State for LangGraph API workflow (v0.3.13: cleaned up unused fields).
    
    Input fields:
        query: User's natural language query
        session_id: Session identifier for conversation context
        access_token: JWT token for API authentication
        runtime_schema: Optional schema override for this query
        conversation_history: Previous messages (loaded externally, passed through)
    
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
    
    Progress tracking:
        progress_tracker: ProgressTracker instance (not serialized in checkpoints)
    
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
    
    # Progress (not serialized)
    progress_tracker: Any


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
        active_schema = processor._get_active_schema(state.get("runtime_schema"))
        if not active_schema:
            error_msg = constants.ERROR_NO_SCHEMA
            return {
                "response": {
                    "success": False,
                    "data": None,
                    "summary": f"{constants.ERROR_PREFIX}{error_msg}",
                    "error": error_msg,
                    "query": state.get("query"),
                    "total_steps": 0,
                    "schema_type": "unknown",
                }
            }

        if active_schema.get("type") not in ["api", "api_schema"]:
            error_msg = constants.ERROR_ONLY_API_SCHEMAS
            return {
                "active_schema": active_schema,
                "response": {
                    "success": False,
                    "data": None,
                    "summary": f"{constants.ERROR_PREFIX}{error_msg}",
                    "error": error_msg,
                    "query": state.get("query"),
                    "total_steps": 0,
                    "schema_type": active_schema.get("type", "unknown"),
                },
            }

        return {"active_schema": active_schema}

    def parse_query(state: APIQueryState) -> Dict[str, Any]:
        # Progress update (v0.3.10)
        tracker = state.get("progress_tracker")
        if tracker:
            tracker.update(ProgressStage.PARSING_QUERY, "Understanding your question...")
        
        try:
            parsed = processor._understand_query(
                state["query"],
                state["active_schema"],
                state.get("context"),
                state.get("conversation_history", []),
            )
        except Exception as e:
            logger.exception("Query parsing (LLM) failed: %s", e)
            error_msg = str(e)
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
        """
        Create multi-step execution plan with dependency resolution.
        Enhanced in v0.3.9 to handle filter merging when merge_with_previous=true.
        Enhanced in v0.3.12 to better handle filter merging and preserve all previous context.
        """
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

            parsed = state["parsed"]
            
            # Check if we need to merge with previous query filters (v0.3.9, enhanced v0.3.12)
            if parsed.get("merge_with_previous") and state.get("conversation_history"):
                logger.info("🔄 Merging filters from previous query (v0.3.12 enhanced)")
                
                # Extract previous filters from conversation history
                conversation_history = state.get("conversation_history", [])
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
            elif parsed.get("merge_with_previous"):
                logger.warning("⚠️  merge_with_previous=true but no conversation_history available!")
            else:
                logger.info(f"ℹ️  No filter merging needed (merge_with_previous={parsed.get('merge_with_previous')})")
            
            execution_plan = planner.create_execution_plan(
                parsed,
                state["active_schema"]
            )
            total_steps = len(execution_plan.get("steps", []))
            if tracker:
                tracker.update(
                    ProgressStage.PLAN_READY,
                    f"Plan ready: {total_steps} step(s)" if total_steps else "Plan ready",
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
        """
        Execute the next step in the execution plan.
        Handles context passing from previous steps.
        """
        tracker = state.get("progress_tracker")
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_step_idx = state.get("current_step", 0)
        step_results = state.get("step_results", [])
        
        if current_step_idx >= len(steps):
            # All steps completed
            return {"current_step": current_step_idx}
        
        current_step_def = steps[current_step_idx]
        
        # Resolve dependencies - substitute variables from previous steps (uses planner's extract when present)
        resolved_step = _resolve_step_dependencies(current_step_def, step_results, all_steps=steps)
        
        # Convert step to API plan format
        plan = {
            "type": "api",
            "intent": resolved_step.get("intent"),
            "resource": resolved_step.get("resource"),
            "entities": resolved_step.get("entities", {}),
            "filters": resolved_step.get("filters", {})
        }
        
        # Create API request using the matcher
        api_plan = processor._create_api_plan(resolved_step, state["active_schema"])
        
        if not api_plan or api_plan.get("type") == "error":
            return {
                "error": f"Step {current_step_idx + 1} planning failed: {api_plan.get('error', 'Unknown error')}"
            }
        
        # Execute the API call
        result = processor._execute_api(
            api_plan,
            state["active_schema"],
            state.get("access_token"),
            resolved_step
        )
        data = result.get("data")
        # Pagination as a step: when step has fetch_all_pages, fetch next page(s) and merge.
        # Safety cap only to avoid infinite loop on buggy APIs; see todo.md § Limits affecting correctness.
        if current_step_def.get("fetch_all_pages") and isinstance(data, dict) and data.get("next"):
            pages = 1
            while data.get("next") and pages < constants.SAFETY_MAX_PAGES:
                next_url = data.get("next")
                page_result = processor._fetch_next_page(next_url, state["active_schema"], state.get("access_token"))
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
        
        # Store result
        step_result = {
            "step_id": current_step_def.get("step_id"),
            "step_description": current_step_def.get("description"),
            "result": result.get("data"),
            "status": "success" if not result.get("error") else "failed",
            "error": result.get("error")
        }
        
        step_results.append(step_result)
        if tracker:
            endpoint = api_plan.get("endpoint", "API") if isinstance(api_plan, dict) else "API"
            tracker.update(
                ProgressStage.API_COMPLETED,
                f"Step {current_step_idx + 1} completed: {endpoint}",
                endpoint=endpoint,
            )
        out = {
            "current_step": current_step_idx + 1,
            "step_results": step_results,
            "result": result  # Last result for compatibility
        }
        # When all steps failed, set error so route_after_step can send to format_error
        if step_result.get("status") == "failed" and (current_step_idx + 1) >= len(steps):
            if all(sr.get("status") == "failed" for sr in step_results):
                err_parts = [f"Step {i + 1}: {sr.get('error', 'Unknown')}" for i, sr in enumerate(step_results)]
                out["error"] = "All steps failed. " + "; ".join(err_parts)
        return out

    def execute_plan(state: APIQueryState) -> Dict[str, Any]:
        """Legacy: not registered as a graph node. The graph uses execute_next_step for execution. Kept for reference only."""
        plan = state["plan"]
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

        result = processor._execute_api(
            plan,
            state["active_schema"],
            state.get("access_token"),
            parsed,
        )
        
        # Log the endpoint that was called
        if result:
            logger.info(f"API call result: endpoint={result.get('endpoint')}, "
                       f"method={result.get('method')}, "
                       f"data_count={len(result.get('data', [])) if isinstance(result.get('data'), list) else 'N/A'}")
        
        return {"result": result}

    def format_missing_info(state: APIQueryState) -> Dict[str, Any]:
        plan = state.get("plan", {})
        return {
            "response": {
                "success": False,
                "needs_info": True,
                "message": plan.get("message"),
                "missing_fields": plan.get("missing_fields"),
                "context": plan.get("context"),
                "query": state.get("query"),
            }
        }

    def format_error(state: APIQueryState) -> Dict[str, Any]:
        """
        Format error response with consistent structure (Issue #4 fix).
        
        All responses should have the same fields for consistent error handling.
        """
        plan_error = state.get("plan", {}).get("error")
        error = plan_error or state.get("error") or constants.ERROR_UNKNOWN
        return {
            "response": {
                "success": False,
                "data": None,
                "summary": f"{constants.ERROR_PREFIX}{error}",
                "error": error,
                "query": state.get("query"),
                "total_steps": 0,
                "schema_type": state.get("active_schema", {}).get("type") if state.get("active_schema") else "unknown",
            }
        }

    def summarize(state: APIQueryState) -> Dict[str, Any]:
        """
        Summarize results from all executed steps.
        v0.3.7: Respects display_mode from parsed query.
        v0.3.10: Progress tracking for completion.
        v0.3.11: Enhanced count handling, pagination detection, and contextual help (Issues #2-7 fix).
        v0.3.12: Count-specific formatting when question_type="count".
        """
        # Progress update (v0.3.10)
        tracker = state.get("progress_tracker")
        if tracker:
            tracker.update(ProgressStage.SUMMARIZING, "Preparing your results...")
        
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
        
        if is_multi_step:
            # Summarize multi-step execution
            all_data = [sr.get("result") for sr in step_results if sr.get("status") == "success"]
            has_error = any(sr.get("status") == "failed" for sr in step_results)
            
            # v0.3.9: Better handling of display_mode for multi-step
            if display_mode == "full":
                summary = constants.SUMMARY_RETRIEVED_ALL_ITEMS_STEPS.format(count=len(all_data), steps=len(step_results))
                logger.info(f"Multi-step display mode = full: returning all {len(all_data)} results")
            elif display_mode == "detailed":
                summary = processor._summarize_multi_step_result(step_results, state["query"])
                logger.info(f"Multi-step display mode = detailed: detailed view")
            else:
                # Summary mode: provide concise multi-step summary
                summary = processor._summarize_multi_step_result(step_results, state["query"])
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
            
            response = {
                "success": not has_error,
                "data": all_data if len(all_data) > 1 else (all_data[0] if all_data else None),
                "summary": summary,
                "query": state.get("query"),
                "display_mode": display_mode,  # v0.3.7
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
            }
            
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
            # Automatic pagination: when response has has_more/next, fetch next page(s) and merge.
            # Safety cap only to avoid infinite loop on buggy APIs; see todo.md § Limits affecting correctness.
            pages_fetched = 1
            while isinstance(data, dict) and data.get("next") and pages_fetched < constants.SAFETY_MAX_PAGES:
                next_url = data.get("next")
                page_result = processor._fetch_next_page(next_url, state["active_schema"], state.get("access_token"))
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
                
                response = {
                    "success": True,
                    "data": final_data,
                    "summary": summary,
                    "query": state.get("query"),
                    "question_type": "count",
                    "display_mode": "count",
                    "pagination": pagination_info,
                    "total_steps": 1,
                    "schema_type": state.get("active_schema", {}).get("type"),
                }
                
                # Complete progress
                if tracker:
                    tracker.update(ProgressStage.COMPLETED, f"Done! Found {total_count} items ✓")
                
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
                summary = processor._summarize_result_v2(result, state["query"], parsed, pagination_info)
                final_data = data
                logger.info(f"Display mode = detailed: returning detailed view")
            else:
                # Default: summary mode (Issue #4: accurate summary)
                summary = processor._summarize_result_v2(result, state["query"], parsed, pagination_info)
                
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
                    fmt = formatter.format_response(
                        data=final_data if final_data is not None else data,
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
                    logger.error("ResponseFormatter (single-step) failed: %s", e)
                    format_error = str(e)
            
            response = {
                "success": not has_error,
                "data": final_data,
                "summary": summary,
                "query": state.get("query"),
                "display_mode": display_mode,
                "pagination": pagination_info,  # v0.3.11: Include pagination info (Issue #2)
                "suggested_actions": _generate_suggestions(pagination_info, parsed, data),  # v0.3.11: Contextual help (Issue #7)
                "total_steps": 1,
                "schema_type": state.get("active_schema", {}).get("type"),
            }
            
            if formatted is not None:
                response["formatted"] = formatted
            if fmt_format is not None:
                response["format"] = fmt_format
            if format_error is not None:
                response["format_error"] = format_error
            
            if has_error:
                response["error"] = result.get("error")
        
        # Final progress update (v0.3.10)
        if tracker:
            if response.get("success"):
                tracker.update(ProgressStage.COMPLETED, constants.PROGRESS_DONE)
            else:
                tracker.update(ProgressStage.ERROR, constants.PROGRESS_ERROR_OCCURRED)
        
        return {"summary": summary, "response": response}

    def route_after_schema(state: APIQueryState) -> str:
        if state.get("response"):
            return "end"
        return "parse"

    def route_after_planning(state: APIQueryState) -> str:
        """Route after execution planning."""
        if state.get("error"):
            return "format_error"
        execution_plan = state.get("execution_plan", {})
        if not execution_plan or not execution_plan.get("steps"):
            return "format_error"
        return "execute_step"
    
    def route_after_step(state: APIQueryState) -> str:
        """Route after executing a step - continue, summarize (with partial results), or format_error (all failed)."""
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

    # Add nodes
    graph.add_node("load_schema", load_schema)
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
            "parse": "parse",
            "end": END,
        },
    )
    graph.add_edge("parse", "create_plan")
    graph.add_conditional_edges(
        "create_plan",
        route_after_planning,
        {
            "format_error": "format_error",
            "execute_step": "execute_step",
        },
    )
    graph.add_conditional_edges(
        "execute_step",
        route_after_step,
        {
            "execute_step": "execute_step",  # Loop for sequential execution
            "summarize": "summarize",
            "format_error": "format_error",   # All steps failed
        },
    )
    graph.add_edge("summarize", END)
    graph.add_edge("format_error", END)

    # Compile with checkpointer if provided
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()
