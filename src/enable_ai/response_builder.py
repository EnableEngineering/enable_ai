"""
Response Builder - Build responses with reasoning trail for transparency.

This module creates responses that include:
1. Query understanding - How the system interpreted the query
2. Actions taken - What API calls were made
3. Result explanation - Why the result is what it is
4. Suggestions - How to improve/refine the query

This transparency helps users:
- Understand what the AI did
- Correct misunderstandings faster
- Learn how to phrase queries
- Debug issues
"""

from typing import Dict, Any, List, Optional
from .utils import setup_logger


class ResponseBuilder:
    """
    Build responses with full reasoning trail.

    Creates structured responses that explain:
    - How the query was understood
    - What actions were taken
    - Why the result is what it is

    Usage:
        builder = ResponseBuilder()
        response = builder.build_response(
            query="show me technician users",
            parsed_intent={"intent": "read", "resource": "users", "filters": {...}},
            api_calls=[{"method": "GET", "endpoint": "/api/users/", "params": {...}}],
            api_responses=[{"status": 200, "data": {...}}],
            final_summary="Found 3 technician users...",
        )

        # Response includes:
        # - summary: "Found 3 technician users..."
        # - reasoning: {query_understanding, actions_taken, result_explanation}
        # - debug: {parsed_intent, api_calls, raw_responses} (optional)
    """

    def __init__(self, include_debug: bool = False):
        """
        Initialize the response builder.

        Args:
            include_debug: Whether to include debug info (raw API data)
        """
        self.include_debug = include_debug
        self.logger = setup_logger('enable_ai.response_builder')

    def build_response(
        self,
        query: str,
        parsed_intent: Dict[str, Any],
        api_calls: List[Dict[str, Any]],
        api_responses: List[Dict[str, Any]],
        final_summary: str,
        success: bool = True,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build response with full reasoning trail.

        Args:
            query: Original user query
            parsed_intent: How the query was parsed
            api_calls: List of API calls made
            api_responses: List of API responses received
            final_summary: The formatted summary for the user
            success: Whether the operation was successful
            schema: Optional schema for additional context

        Returns:
            Dict with summary, reasoning, and optional debug info
        """
        response = {
            "summary": final_summary,
            "success": success,
            "reasoning": {
                "query_understanding": self._explain_parsing(query, parsed_intent),
                "actions_taken": self._explain_actions(api_calls, api_responses),
                "result_explanation": self._explain_result(api_responses, success, parsed_intent),
            }
        }

        # Add debug info if enabled
        if self.include_debug:
            response["debug"] = {
                "parsed_intent": parsed_intent,
                "api_calls": api_calls,
                "raw_responses": api_responses,
            }

        return response

    def _explain_parsing(
        self,
        query: str,
        parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain how the query was understood.

        Args:
            query: Original query
            parsed: Parsed intent

        Returns:
            Dict explaining the parsing
        """
        intent = parsed.get("intent", "read")
        resource = parsed.get("resource", "items")
        filters = parsed.get("filters", {})
        confidence = parsed.get("confidence", 1.0)

        return {
            "original_query": query,
            "understood_as": {
                "intent": intent,
                "resource": resource,
                "filters": filters,
                "fields": parsed.get("fields", []),
            },
            "confidence": confidence,
            "explanation": self._generate_parsing_explanation(parsed),
            "classified_by": parsed.get("classified_by", "llm"),
        }

    def _generate_parsing_explanation(self, parsed: Dict[str, Any]) -> str:
        """Generate human-readable explanation of parsing."""
        resource = parsed.get("resource", "items")
        filters = parsed.get("filters", {})
        intent = parsed.get("intent", "read")

        # Map intents to verbs
        intent_verbs = {
            "read": "Looking for",
            "create": "Creating",
            "update": "Updating",
            "delete": "Deleting",
        }
        verb = intent_verbs.get(intent, "Processing")

        if not filters:
            return f"{verb} all {resource}."

        # Build filter description
        filter_parts = []
        for field, val in filters.items():
            if isinstance(val, dict):
                value = val.get("value")
            else:
                value = val
            filter_parts.append(f"{field}='{value}'")

        return f"{verb} {resource} where {' and '.join(filter_parts)}."

    def _explain_actions(
        self,
        api_calls: List[Dict[str, Any]],
        responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Explain each action taken.

        Args:
            api_calls: List of API calls made
            responses: Corresponding responses

        Returns:
            List of action explanations
        """
        actions = []

        # Ensure we have matching calls and responses
        for i, call in enumerate(api_calls):
            response = responses[i] if i < len(responses) else {}

            action = {
                "step": i + 1,
                "action": f"Called {call.get('method', 'GET')} {call.get('endpoint', '')}",
                "params": call.get("params", {}),
                "result": self._summarize_api_result(response),
            }
            actions.append(action)

        return actions

    def _summarize_api_result(self, response: Dict[str, Any]) -> str:
        """Summarize an API response for display."""
        # Check for error
        if "error" in response:
            return f"❌ Error: {response['error']}"

        status = response.get("status_code") or response.get("status")
        if status and status >= 400:
            return f"❌ Error: HTTP {status}"

        # Check data
        data = response.get("data", {})

        # Paginated response
        if isinstance(data, dict) and "count" in data:
            count = data.get("count", 0)
            return f"✅ Found {count} results"

        # List response
        if isinstance(data, list):
            return f"✅ Found {len(data)} results"

        # Single item
        if isinstance(data, dict) and data:
            return "✅ Retrieved 1 item"

        return "✅ Success"

    def _explain_result(
        self,
        responses: List[Dict[str, Any]],
        success: bool,
        parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain why the result is what it is.

        Args:
            responses: API responses
            success: Overall success status
            parsed: Parsed intent (for context)

        Returns:
            Result explanation with success, reason, suggestion
        """
        if not success:
            # Find the failure reason
            for resp in responses:
                if "error" in resp:
                    return {
                        "success": False,
                        "reason": resp["error"],
                        "suggestion": self._suggest_fix(resp),
                    }

            return {
                "success": False,
                "reason": "Operation failed for unknown reason.",
                "suggestion": "Try rephrasing your query or contact support.",
            }

        # Aggregate results
        total_count = 0
        for resp in responses:
            data = resp.get("data", {})
            if isinstance(data, dict) and "count" in data:
                total_count = data["count"]
            elif isinstance(data, dict) and "results" in data:
                api_count = data.get("count") or data.get("total_count")
                if api_count is not None:
                    total_count = int(api_count)
                else:
                    total_count = len(data.get("results") or [])
            elif isinstance(data, list):
                total_count = len(data)

        resource = parsed.get("resource", "items")

        if total_count == 0:
            filters = parsed.get("filters", {})
            suggestion = "Try broadening your search or removing filters."
            if filters:
                suggestion = f"Try removing some filters. You searched for {resource} with specific criteria that returned no matches."

            return {
                "success": True,
                "reason": f"Query executed successfully but no {resource} found matching your criteria.",
                "suggestion": suggestion,
            }

        return {
            "success": True,
            "reason": f"Found {total_count} matching {resource}.",
            "suggestion": None,
        }

    def _suggest_fix(self, error_response: Dict[str, Any]) -> str:
        """Suggest how to fix an error."""
        error = str(error_response.get("error", ""))
        status = error_response.get("status_code") or error_response.get("status", 0)

        # Authentication errors
        if status == 401 or "401" in error or "unauthorized" in error.lower():
            return "Your session may have expired. Try logging in again."

        # Permission errors
        if status == 403 or "403" in error or "forbidden" in error.lower():
            return "You may not have permission to access this resource."

        # Not found
        if status == 404 or "404" in error or "not found" in error.lower():
            return "The resource was not found. Check if the ID is correct."

        # Bad request
        if status == 400 or "400" in error or "bad request" in error.lower():
            return "The request was invalid. Check your filter values."

        # Server error
        if status >= 500:
            return "Server error occurred. Try again later or contact support."

        return "Try rephrasing your query or contact support."

    def build_error_response(
        self,
        query: str,
        error: str,
        parsed_intent: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build an error response with reasoning.

        Args:
            query: Original query
            error: Error message
            parsed_intent: Parsed intent (if available)

        Returns:
            Error response with reasoning
        """
        return {
            "summary": f"Error: {error}",
            "success": False,
            "reasoning": {
                "query_understanding": self._explain_parsing(
                    query,
                    parsed_intent or {"intent": "unknown", "resource": "unknown"}
                ) if parsed_intent else {
                    "original_query": query,
                    "understood_as": None,
                    "confidence": 0.0,
                    "explanation": "Could not understand the query.",
                },
                "actions_taken": [],
                "result_explanation": {
                    "success": False,
                    "reason": error,
                    "suggestion": self._suggest_fix({"error": error}),
                },
            }
        }

    def build_empty_response(
        self,
        query: str,
        parsed_intent: Dict[str, Any],
        api_calls: List[Dict[str, Any]],
        api_responses: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a response for empty results with helpful suggestions.

        Args:
            query: Original query
            parsed_intent: How query was parsed
            api_calls: API calls made
            api_responses: Responses received
            schema: Optional schema for suggestions

        Returns:
            Response with empty result explanation and suggestions
        """
        resource = parsed_intent.get("resource", "items")
        filters = parsed_intent.get("filters", {})

        # Build suggestions
        suggestions = []
        if filters:
            suggestions.append(f"Try removing filters: 'Show all {resource}'")

            # Suggest valid values from schema if available
            if schema:
                resource_hints = schema.get("resource_hints", {}).get(resource, {})
                for field in filters.keys():
                    field_hints = resource_hints.get(field, {})
                    if isinstance(field_hints, dict):
                        valid_values = field_hints.get("values", [])
                        if valid_values:
                            values_str = ", ".join(str(v) for v in valid_values[:5])
                            suggestions.append(f"Valid values for '{field}': {values_str}")
        else:
            suggestions.append(f"This {resource} collection may be empty.")

        # Build summary
        summary = f"No {resource} found matching your criteria."
        if suggestions:
            summary += "\n\n💡 **Suggestions:**"
            for sug in suggestions:
                summary += f"\n  • {sug}"

        return self.build_response(
            query=query,
            parsed_intent=parsed_intent,
            api_calls=api_calls,
            api_responses=api_responses,
            final_summary=summary,
            success=True,  # Query succeeded, just no results
            schema=schema
        )


def create_response_builder(include_debug: bool = False) -> ResponseBuilder:
    """Factory function to create a ResponseBuilder instance."""
    return ResponseBuilder(include_debug=include_debug)
