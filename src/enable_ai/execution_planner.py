"""
Multi-Step API Planner

Creates execution plans for complex queries that require multiple API calls.
Handles dependency resolution and sequential execution ordering.
"""

import json
from typing import Dict, Any, Optional

from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP


class ExecutionPlanner:
    """
    Plans multi-step API execution workflows with dependency resolution.
    
    Uses OpenAI to determine:
    - Which API calls are needed
    - The order of execution
    - Dependencies between calls
    - Data passing between steps
    """
    
    def __init__(self):
        """Initialize the planner with OpenAI client."""
        self.openai_client = get_openai_client()
        self.logger = setup_logger('enable_ai.planner')
        self.logger.info("ExecutionPlanner initialized")
    
    def create_execution_plan(
        self, 
        parsed_query: Dict[str, Any], 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a multi-step execution plan from parsed query and schema.
        
        Args:
            parsed_query: Parsed query with intent, resource, entities
            schema: API schema with available endpoints
            
        Returns:
            {
                "steps": [
                    {
                        "step_id": 1,
                        "action": "GET /users/5/",
                        "depends_on": [],
                        "extract": {"user_id": "$.id"},
                        "description": "Fetch user with ID 5"
                    },
                    {
                        "step_id": 2,
                        "action": "GET /orders/?user_id={user_id}",
                        "depends_on": [1],
                        "description": "Fetch orders for user"
                    }
                ],
                "is_multi_step": bool,
                "total_steps": int
            }
        """
        # "Show me more" / next page: single step that fetches the stored next_url
        if parsed_query.get("use_next_page") and parsed_query.get("next_page_url"):
            return {
                "steps": [{
                    "step_id": 1,
                    "type": "fetch_next_page",
                    "url": parsed_query["next_page_url"],
                    "description": "Fetch next page of results",
                }],
                "is_multi_step": False,
                "total_steps": 1,
            }
        # Check if query requires multiple steps
        requires_multiple_steps = self._analyze_complexity(parsed_query)
        
        if not requires_multiple_steps:
            # Simple single-step query
            return {
                "steps": [self._create_single_step(parsed_query, schema)],
                "is_multi_step": False,
                "total_steps": 1
            }
        
        # Complex multi-step query - use LLM to plan
        return self._plan_with_llm(parsed_query, schema)
    
    def _analyze_complexity(self, parsed_query: Dict[str, Any]) -> bool:
        """
        Determine if query requires multiple API calls.
        
        Indicators of complexity:
        - Multiple resources mentioned
        - Relationships/joins needed
        - Sequential operations (e.g., "get X and then Y")
        """
        # Check for multiple resources
        resource = parsed_query.get('resource', '')
        relationships = parsed_query.get('relationships', [])
        entities = parsed_query.get('entities', {})
        
        # Multiple relationships indicate joins/multi-step
        if len(relationships) > 0:
            return True
        
        # Check if entities reference multiple resources
        resource_refs = set()
        for key in entities.keys():
            if '_' in key:  # e.g., "companies_name", "users_id"
                resource_refs.add(key.split('_')[0])
        
        if len(resource_refs) > 1:
            return True
        
        # Explicit multiple resources (parser may output list for "X and their Y")
        multiple_resources = parsed_query.get('multiple_resources', [])
        if isinstance(multiple_resources, list) and len(multiple_resources) > 1:
            return True
        
        return False
    
    def _create_single_step(
        self, 
        parsed_query: Dict[str, Any], 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a single-step execution plan."""
        intent = parsed_query.get('intent', 'read')
        resource = parsed_query.get('resource', '')
        
        return {
            "step_id": 1,
            "intent": intent,
            "resource": resource,
            "entities": parsed_query.get('entities', {}),
            "filters": parsed_query.get('filters', {}),
            "depends_on": [],
            "description": f"{intent.upper()} {resource}"
        }
    
    def _plan_with_llm(
        self, 
        parsed_query: Dict[str, Any], 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to create a multi-step execution plan.
        
        The LLM determines:
        - Which endpoints to call
        - The order of execution
        - Dependencies between steps
        - Data to extract and pass between steps
        """
        prompt = self._build_planning_prompt(parsed_query, schema)
        
        try:
            self.logger.info(f"Creating execution plan for query: {parsed_query.get('original_input', 'N/A')}")
            
            plan = self.openai_client.parse_json_response(
                messages=[
                    {"role": "system", "content": self._get_planner_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=DETERMINISTIC_TEMP
            )
            
            # Validate plan structure
            if "steps" not in plan:
                raise ValueError("Plan missing 'steps' field")
            
            plan["is_multi_step"] = len(plan["steps"]) > 1
            plan["total_steps"] = len(plan["steps"])
            
            self.logger.info(
                f"Plan created: {plan['total_steps']} step(s), "
                f"multi-step={plan['is_multi_step']}"
            )
            
            return plan
            
        except Exception as e:
            self.logger.warning(f"LLM planning failed: {e}. Falling back to single-step.")
            # Fallback to single-step
            return {
                "steps": [self._create_single_step(parsed_query, schema)],
                "is_multi_step": False,
                "total_steps": 1
            }
    
    def _get_planner_system_prompt(self) -> str:
        """System prompt for the LLM planner."""
        return """You are an API execution planner. Your job is to turn a parsed user query and API schema into a step-by-step execution plan.

Given the parsed query and schema, produce a plan with:
1. **steps**: Ordered list of API calls to execute
2. **dependencies**: Which steps depend on previous steps (depends_on)
3. **data_extraction**: What to extract from each step and pass to the next (extract with JSONPath)

OUTPUT FORMAT (JSON):
{
    "steps": [
        {
            "step_id": 1,
            "intent": "read|create|update|delete",
            "resource": "resource_name",
            "entities": {"field": "value"},
            "filters": {"field": {"operator": "equals", "value": "..."}},
            "depends_on": [],
            "extract": {"variable_name": "$.json.path"},
            "description": "Human-readable step description",
            "fetch_all_pages": false
        }
    ]
}

RULES:
- step_id starts at 1 and increments
- depends_on lists step_ids that must complete first
- extract uses JSONPath to get data from the previous step's response; use {variable_name} in later steps
- fetch_all_pages (optional): set true when the user wants the full set of results (e.g. all items or all pages) for a list endpoint; the executor will fetch pages until none left

EXAMPLE (Query: "Get user 5 and all their orders"):
{
    "steps": [
        {"step_id": 1, "intent": "read", "resource": "users", "entities": {"id": 5}, "filters": {"id": {"operator": "equals", "value": 5}}, "depends_on": [], "extract": {"user_id": "$.id"}, "description": "Fetch user with ID 5"},
        {"step_id": 2, "intent": "read", "resource": "orders", "entities": {"user_id": "{user_id}"}, "filters": {"user_id": {"operator": "equals", "value": "{user_id}"}}, "depends_on": [1], "description": "Fetch orders for the user"}
    ]
}

Return ONLY the JSON plan, no explanations.
"""
    
    def _build_planning_prompt(
        self, 
        parsed_query: Dict[str, Any], 
        schema: Dict[str, Any]
    ) -> str:
        """Build the planning prompt for the LLM."""
        # Extract available resources from schema
        resources = list(schema.get('resources', {}).keys())
        
        prompt = f"""Create an execution plan for this query:

PARSED QUERY:
{json.dumps(parsed_query, indent=2)}

AVAILABLE API RESOURCES:
{json.dumps(resources, indent=2)}

SCHEMA DETAILS:
{json.dumps(self._simplify_schema_for_prompt(schema), indent=2)}

Create a step-by-step execution plan. Return ONLY the JSON plan.
"""
        return prompt
    
    def _simplify_schema_for_prompt(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify schema to reduce token usage in prompt."""
        simplified = {}
        
        for resource_name, resource_data in schema.get('resources', {}).items():
            endpoints = resource_data.get('endpoints', [])
            simplified[resource_name] = {
                "description": resource_data.get('description', ''),
                "methods": [ep.get('method') for ep in endpoints if isinstance(ep, dict)]
            }
        
        return simplified
