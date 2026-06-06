"""
Multi-Step API Planner

Creates execution plans for complex queries that require multiple API calls.
Handles dependency resolution and sequential execution ordering.
"""

import json
from typing import Dict, Any, Optional, List

from .hint_utils import expand_query_resources, get_embedded_fields
from .utils import get_openai_client, setup_logger, DETERMINISTIC_TEMP
from .query_execution import enrich_step_from_parsed, merge_execution_context
from .user_context_resolver import is_user_id_placeholder, is_company_id_placeholder


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

        parsed_query = expand_query_resources(parsed_query, parsed_query.get("original_input", ""), schema)

        # Smart FK detection - auto-generate lookup steps if needed
        fk_lookups = self._detect_fk_lookups_needed(parsed_query, schema)

        if fk_lookups:
            self.logger.info(f"Auto-detected FK lookups needed: {[l['field'] for l in fk_lookups]}")
            return self._build_fk_lookup_plan(parsed_query, fk_lookups, schema)

        # Multi-resource count ("how many X and Y")
        multi_count = self._plan_multi_resource_count(parsed_query)
        if multi_count:
            return multi_count

        # Parent → child relationship (e.g. observations for my last report)
        rel_plan = self._plan_relationship_query(parsed_query, schema)
        if rel_plan:
            return rel_plan

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

    def _detect_fk_lookups_needed(self, parsed: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict]:
        """
        Intelligently detect FK fields that need ID lookup.

        Auto-detection logic:
        1. If field name matches a resource name (singular/plural), it's likely a FK
           - "company" -> "companies" resource
           - "technician" -> "users" resource (via resource_hints synonyms)
        2. If field ends with "_id", the base name might be a resource
        3. If value is a string (not numeric), likely needs lookup

        Returns:
            List of dicts with field, value, target_resource, search_field
        """
        lookups_needed = []
        resources = schema.get("resources", {})
        resource_hints = schema.get("resource_hints", {})
        filters = parsed.get("filters", {}) or {}

        for field, filter_val in filters.items():
            value = filter_val.get("value") if isinstance(filter_val, dict) else filter_val

            # Skip if already numeric (ID)
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                continue

            # Skip user/company context placeholders — resolved before planning
            if is_user_id_placeholder(value) or is_company_id_placeholder(value):
                continue

            # Skip if it's a known enum field (has values in resource_hints)
            current_resource = parsed.get("resource", "")
            field_hints = resource_hints.get(current_resource, {}).get(field, {})
            if isinstance(field_hints, dict) and field_hints.get("values"):
                continue  # This is an enum field, not a FK

            # Try to find matching resource for this field
            target_resource = self._find_fk_target_resource(field, resources, resource_hints)

            if target_resource:
                lookups_needed.append({
                    "field": field,
                    "value": value,
                    "target_resource": target_resource,
                    "search_field": self._get_search_field(target_resource, resources)
                })

        return lookups_needed

    def _find_fk_target_resource(
        self,
        field: str,
        resources: Dict[str, Any],
        resource_hints: Dict[str, Any]
    ) -> Optional[str]:
        """
        Find the target resource for a FK field.

        Matching strategies:
        1. Exact match: field "companies" -> resource "companies"
        2. Singular/plural: field "company" -> resource "companies"
        3. Synonym match: field "technician" -> users (via __resource_synonyms__)
        4. Common patterns: field "assigned_to" -> users, "created_by" -> users
        """
        field_lower = field.lower().replace("_id", "")

        # Strategy 1 & 2: Direct or plural match
        for res_name in resources.keys():
            res_lower = res_name.lower()
            if field_lower == res_lower:
                return res_name
            if field_lower + "s" == res_lower:
                return res_name
            if field_lower + "es" == res_lower:
                return res_name
            if field_lower == res_lower.rstrip("s"):
                return res_name
            # Handle "ies" -> "y" (e.g., "company" -> "companies")
            if res_lower.endswith("ies") and field_lower == res_lower[:-3] + "y":
                return res_name

        # Strategy 3: Check resource_hints synonyms
        for res_name, hints in resource_hints.items():
            if not isinstance(hints, dict):
                continue
            synonyms = hints.get("__resource_synonyms__", [])
            if not isinstance(synonyms, list):
                synonyms = [synonyms]
            if field_lower in [str(s).lower() for s in synonyms]:
                return res_name

        # Strategy 4: Common FK patterns
        user_fields = ["technician", "assigned_to", "created_by", "updated_by", "owner", "assignee"]
        if field_lower in user_fields and "users" in resources:
            return "users"

        return None

    def _get_search_field(self, resource: str, resources: Dict[str, Any]) -> str:
        """
        Determine the best field to search by for a resource.
        Priority: name > title > username > email > id
        """
        resource_data = resources.get(resource, {})
        fields = resource_data.get("fields", [])

        # If fields is a dict (field definitions), get keys
        if isinstance(fields, dict):
            fields = list(fields.keys())

        for preferred in ["name", "title", "username", "email"]:
            if preferred in fields:
                return preferred
        return "name"  # Default fallback

    def _build_fk_lookup_plan(
        self,
        parsed: Dict[str, Any],
        fk_lookups: List[Dict],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a multi-step plan with FK lookup steps.

        Creates:
        1. One lookup step per FK field
        2. Final main query step with variable substitutions
        """
        steps = []

        # Generate lookup steps
        for i, lookup in enumerate(fk_lookups):
            steps.append({
                "step_id": i + 1,
                "intent": "read",
                "resource": lookup["target_resource"],
                "filters": {
                    lookup["search_field"]: {
                        "operator": "icontains",  # Case-insensitive search
                        "value": lookup["value"]
                    }
                },
                "entities": {},
                "extract": {f"{lookup['field']}_id": "$.results[0].id"},
                "depends_on": [],
                "description": f"Lookup {lookup['target_resource']} ID for '{lookup['value']}'"
            })

        # Main query step with variable substitution
        main_filters = dict(parsed.get("filters", {}) or {})
        for lookup in fk_lookups:
            # Replace string value with variable reference
            main_filters[lookup["field"]] = {
                "operator": "equals",
                "value": f"{{{lookup['field']}_id}}"  # Will be substituted
            }

        steps.append(merge_execution_context({
            "step_id": len(steps) + 1,
            "intent": parsed.get("intent", "read"),
            "resource": parsed.get("resource"),
            "entities": parsed.get("entities", {}),
            "filters": main_filters,
            "depends_on": list(range(1, len(steps) + 1)),
            "description": f"Get {parsed.get('resource')} with resolved FK IDs",
        }, parsed))

        return {
            "steps": steps,
            "is_multi_step": True,
            "total_steps": len(steps)
        }
    
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
    
    def _plan_multi_resource_count(self, parsed_query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build parallel count steps for multiple_resources queries."""
        multiple = parsed_query.get("multiple_resources", [])
        if not isinstance(multiple, list) or len(multiple) < 2:
            return None
        if parsed_query.get("question_type") != "count":
            return None

        steps = []
        for i, res in enumerate(multiple):
            steps.append(enrich_step_from_parsed({
                "step_id": i + 1,
                "intent": parsed_query.get("intent", "read"),
                "resource": res,
                "entities": parsed_query.get("entities", {}),
                "filters": parsed_query.get("filters", {}),
                "depends_on": [],
                "description": f"Count {res}",
                "question_type": "count",
            }, parsed_query))

        self.logger.info("Multi-resource count plan: %s", multiple)
        return {
            "steps": steps,
            "is_multi_step": True,
            "total_steps": len(steps),
            "plan_type": "multi_resource_count",
        }

    def _plan_relationship_query(
        self,
        parsed_query: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministic parent→child plan when parser outputs relationships.

        Embedded children (nested fields, not API resources) use:
          step 1 list parent → step 2 retrieve parent /{id}/ and extract field.
        """
        relationships = parsed_query.get("relationships") or []
        if not relationships:
            return None

        parent_resource = parsed_query.get("resource")
        if not parent_resource:
            return None

        rel = relationships[0]
        child_target = rel.get("target_entity")
        if not child_target:
            return None

        resource_hints = (schema or {}).get("resource_hints") or {}
        schema_resources = set((schema or {}).get("resources") or {})
        embedded_fields = get_embedded_fields(parent_resource, resource_hints)
        is_embedded = (
            child_target in embedded_fields
            or child_target not in schema_resources
        )

        step1 = enrich_step_from_parsed({
            "step_id": 1,
            "intent": parsed_query.get("intent", "read"),
            "resource": parent_resource,
            "entities": parsed_query.get("entities", {}),
            "filters": parsed_query.get("filters", {}),
            "depends_on": [],
            "extract": {"parent_id": "$.results[0].id"},
            "description": f"List {parent_resource} for lookup",
        }, parsed_query)

        if is_embedded:
            step2 = enrich_step_from_parsed({
                "step_id": 2,
                "intent": "read",
                "resource": parent_resource,
                "entities": {"id": "{parent_id}"},
                "filters": {},
                "depends_on": [1],
                "description": f"Retrieve {parent_resource} detail for {child_target}",
                "embedded_field": child_target,
                "question_type": "details",
                "display_mode": "detailed",
            }, parsed_query)
            plan_type = "embedded_child"
        else:
            child_entities = dict(parsed_query.get("entities", {}))
            child_entities["id"] = "{parent_id}"
            child_filters = dict(rel.get("filters") or {})
            child_filters["id"] = {"operator": "equals", "value": "{parent_id}"}
            step2 = enrich_step_from_parsed({
                "step_id": 2,
                "intent": "read",
                "resource": child_target,
                "entities": child_entities,
                "filters": child_filters,
                "depends_on": [1],
                "description": f"Fetch {child_target} for parent",
            }, parsed_query)
            plan_type = "parent_child"

        self.logger.info(
            "Relationship plan: %s → %s (embedded=%s, type=%s)",
            parent_resource, child_target, is_embedded, rel.get("type"),
        )
        return {
            "steps": [step1, step2],
            "is_multi_step": True,
            "total_steps": 2,
            "plan_type": plan_type,
            "embedded_field": child_target if is_embedded else None,
        }

    def _create_single_step(
        self, 
        parsed_query: Dict[str, Any], 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a single-step execution plan."""
        intent = parsed_query.get('intent', 'read')
        resource = parsed_query.get('resource', '')
        
        return enrich_step_from_parsed({
            "step_id": 1,
            "intent": intent,
            "resource": resource,
            "entities": parsed_query.get('entities', {}),
            "filters": parsed_query.get('filters', {}),
            "depends_on": [],
            "description": f"{intent.upper()} {resource}",
        }, parsed_query)
    
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
            
            plan["steps"] = [
                enrich_step_from_parsed(step, parsed_query)
                for step in plan.get("steps", [])
            ]
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
            "fetch_all_pages": false,
            "sort": {"field": "created_at", "order": "desc"},
            "limit": 10
        }
    ]
}

RULES:
- Include sort and limit on steps when the user asks for ordering or a specific count (e.g. "last report" -> sort desc, limit 1)
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
