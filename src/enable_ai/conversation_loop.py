"""
Conversation Loop - Closed-loop confirmation and correction handling.

This module manages multi-turn clarification and correction flow:
1. Detect when confirmation is needed (low confidence, ambiguous)
2. Generate confirmation questions
3. Process user responses (yes/no/correction)
4. Store corrections for learning

Integrates with:
- CorrectionStore: For learning from corrections
- SchemaIntrospector: For suggesting valid values
"""

from typing import Dict, Any, List, Optional
from .types import MissingInformation
from .correction_store import CorrectionStore, InMemoryCorrectionStore
from .schema_introspector import SchemaIntrospector
from .utils import setup_logger
from . import constants


class ConversationLoop:
    """
    Handles confirmation and correction flow for uncertain queries.

    Flow:
    1. System parses query with QueryParser
    2. ConversationLoop checks if confirmation needed (low confidence)
    3. If needed, generates confirmation question
    4. User confirms or corrects
    5. Correction is stored for learning
    6. Returns corrected interpretation for execution

    Usage:
        loop = ConversationLoop(correction_store, schema_introspector)

        # Check if confirmation needed
        if loop.needs_confirmation(parsed, confidence):
            question = loop.generate_confirmation_question(parsed, schema)
            # Send question to user...

        # Handle user response
        result = loop.process_user_response(session_id, response, pending_parsed, schema)
        if result["confirmed"]:
            # Execute with parsed intent
        elif result["needs_reparse"]:
            # Re-parse with user's correction as context
    """

    def __init__(
        self,
        correction_store: Optional[CorrectionStore] = None,
        schema_introspector: Optional[SchemaIntrospector] = None
    ):
        """
        Initialize the conversation loop.

        Args:
            correction_store: Store for saving and learning from corrections
            schema_introspector: For suggesting valid filter values
        """
        self.correction_store = correction_store or InMemoryCorrectionStore()
        self.schema_introspector = schema_introspector
        self.logger = setup_logger('enable_ai.conversation_loop')

        # Pending confirmations by session
        self._pending: Dict[str, Dict[str, Any]] = {}

        self.logger.info("ConversationLoop initialized")

    def needs_confirmation(
        self,
        parsed: Dict[str, Any],
        confidence: float
    ) -> bool:
        """
        Determine if we should ask for confirmation before executing.

        Confirmation is needed when:
        - Confidence is below threshold
        - Parsed result has ambiguous filters
        - Resource is unknown
        - Intent is unclear

        Args:
            parsed: Parsed query result
            confidence: Classification/parsing confidence (0.0-1.0)

        Returns:
            True if confirmation should be requested
        """
        # Low confidence
        if confidence < constants.CONFIRMATION_CONFIDENCE_THRESHOLD:
            self.logger.debug(f"Needs confirmation: low confidence ({confidence:.2f})")
            return True

        # Unknown or ambiguous resource
        if parsed.get("unknown_intent"):
            self.logger.debug("Needs confirmation: unknown intent/resource")
            return True

        # Ambiguous filters (multiple possible interpretations)
        if parsed.get("ambiguous_filters"):
            self.logger.debug("Needs confirmation: ambiguous filters")
            return True

        # Missing required filters
        if parsed.get("needs_clarification"):
            self.logger.debug("Needs confirmation: needs clarification")
            return True

        return False

    def generate_confirmation_question(
        self,
        parsed: Dict[str, Any],
        schema: Dict[str, Any],
        original_query: Optional[str] = None
    ) -> MissingInformation:
        """
        Generate a confirmation question for the user.

        Creates a human-readable interpretation of what the system
        understood, asking the user to confirm or correct.

        Args:
            parsed: Parsed query result
            schema: API schema
            original_query: Original user query (for context)

        Returns:
            MissingInformation with confirmation question and options
        """
        resource = parsed.get("resource", "items")
        intent = parsed.get("intent", "read")
        filters = parsed.get("filters", {})

        # Build human-readable interpretation
        interpretation = self._build_interpretation(intent, resource, filters)

        # Build confirmation message
        message = constants.CONFIRMATION_MESSAGE_TEMPLATE.format(
            interpretation=interpretation
        )

        # Create options
        options = [
            {"key": "confirm", "label": "✅ Yes, proceed"},
            {"key": "modify", "label": "✏️ Modify filters"},
            {"key": "cancel", "label": "❌ Cancel"},
        ]

        # Add suggestions for valid values if available
        suggestions = self._get_filter_suggestions(resource, filters, schema)
        if suggestions:
            message += "\n\n💡 **Suggestions:**"
            for suggestion in suggestions[:3]:
                message += f"\n  • {suggestion}"

        return MissingInformation(
            message=message,
            missing_fields=[],
            matched_endpoint=None,
            context={
                "type": "confirmation",
                "parsed": parsed,
                "requires_yes_no": True,
                "options": options,
                "original_query": original_query,
            }
        )

    def process_user_response(
        self,
        session_id: str,
        user_response: str,
        pending_parsed: Dict[str, Any],
        schema: Dict[str, Any],
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user's confirmation or correction response.

        Args:
            session_id: User session identifier
            user_response: User's response text
            pending_parsed: The pending parsed interpretation
            schema: API schema
            original_query: Original query (for storing correction)

        Returns:
            Dict with:
            - confirmed: True if user confirmed
            - needs_clarification: True if more info needed
            - needs_reparse: True if should re-parse with correction
            - correction: User's correction text (if correcting)
            - parsed: Updated parsed result (if corrected)
        """
        response_lower = user_response.lower().strip()

        # Check for confirmation
        if response_lower in constants.CONFIRMATION_YES_RESPONSES:
            self.logger.info(f"Session {session_id}: User confirmed interpretation")
            return {
                "confirmed": True,
                "parsed": pending_parsed
            }

        # Check for rejection
        if response_lower in constants.CONFIRMATION_NO_RESPONSES:
            self.logger.info(f"Session {session_id}: User rejected interpretation")
            return {
                "confirmed": False,
                "needs_clarification": True,
                "question": constants.CONFIRMATION_CLARIFY_MESSAGE
            }

        # Treat response as a correction
        self.logger.info(f"Session {session_id}: Treating response as correction")

        # Store the correction for learning
        if original_query:
            self.store_correction(
                session_id=session_id,
                original_query=original_query,
                original_parsed=pending_parsed,
                correction=user_response,
                corrected_parsed={}  # Will be filled after re-parse
            )

        return {
            "confirmed": False,
            "needs_reparse": True,
            "correction": user_response,
            "correction_context": {
                "previous_parsed": pending_parsed,
                "user_correction": user_response,
            }
        }

    def store_correction(
        self,
        session_id: str,
        original_query: str,
        original_parsed: Dict[str, Any],
        correction: str,
        corrected_parsed: Dict[str, Any]
    ):
        """
        Store a correction for learning.

        Args:
            session_id: User session identifier
            original_query: Original user query
            original_parsed: How system originally parsed it
            correction: User's correction text
            corrected_parsed: The corrected interpretation
        """
        if self.correction_store:
            self.correction_store.add_correction(
                session_id=session_id,
                original_query=original_query,
                system_interpretation=original_parsed,
                user_correction=correction,
                correct_interpretation=corrected_parsed
            )
            self.logger.info(f"Stored correction for session {session_id}")

    def update_correction_with_final_parse(
        self,
        session_id: str,
        original_query: str,
        corrected_parsed: Dict[str, Any]
    ):
        """
        Update a stored correction with the final parsed result.

        Called after re-parsing with the user's correction.

        Args:
            session_id: User session identifier
            original_query: Original query (for lookup)
            corrected_parsed: Final correct interpretation
        """
        # This would update the stored correction with the final parse
        # For now, we store corrections with empty corrected_parsed
        # and could update them later if needed
        pass

    def get_learned_corrections(self, resource: str) -> Dict[str, str]:
        """
        Get learned synonyms from past corrections for a resource.

        Returns synonyms that have been corrected multiple times,
        which can be used to improve future parsing.

        Args:
            resource: Resource name

        Returns:
            Dict mapping original terms to corrected values
        """
        if self.correction_store:
            return self.correction_store.get_learned_synonyms(resource)
        return {}

    def find_alternatives(
        self,
        parsed: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find alternative queries when original returns empty results.

        Suggests:
        1. Removing filters (broaden search)
        2. Similar filter values
        3. Related resources

        Args:
            parsed: Current parsed query
            schema: API schema

        Returns:
            List of alternative query suggestions
        """
        alternatives = []
        resource = parsed.get("resource")
        filters = parsed.get("filters", {})

        # Alternative 1: Remove filters
        if filters:
            alternatives.append({
                "description": f"Show all {resource} (no filters)",
                "modified_intent": {**parsed, "filters": {}},
                "type": "broaden"
            })

        # Alternative 2: Suggest similar filter values
        if self.schema_introspector and resource:
            for field, val in filters.items():
                value = val.get("value") if isinstance(val, dict) else val
                similar = self._find_similar_values(resource, field, value)
                for sim_val in similar[:2]:
                    alternatives.append({
                        "description": f"Try {field}='{sim_val}' instead",
                        "modified_intent": {
                            **parsed,
                            "filters": {**filters, field: {"operator": "equals", "value": sim_val}}
                        },
                        "type": "similar_value"
                    })

        # Alternative 3: Related resources
        similar_resources = self._find_similar_resources(resource, schema)
        for res in similar_resources[:2]:
            alternatives.append({
                "description": f"Search in {res} instead",
                "modified_intent": {**parsed, "resource": res, "filters": {}},
                "type": "different_resource"
            })

        return alternatives

    def _build_interpretation(
        self,
        intent: str,
        resource: str,
        filters: Dict[str, Any]
    ) -> str:
        """Build human-readable interpretation of parsed query."""
        intent_verbs = {
            "read": "list",
            "create": "create",
            "update": "update",
            "delete": "delete",
        }
        verb = intent_verbs.get(intent, intent)

        if not filters:
            return f"{verb} all {resource}"

        # Build filter description
        filter_parts = []
        for field, val in filters.items():
            if isinstance(val, dict):
                value = val.get("value")
                operator = val.get("operator", "equals")
            else:
                value = val
                operator = "equals"

            if operator == "equals":
                filter_parts.append(f"{field}='{value}'")
            elif operator == "contains":
                filter_parts.append(f"{field} contains '{value}'")
            else:
                filter_parts.append(f"{field} {operator} '{value}'")

        return f"{verb} {resource} where {' and '.join(filter_parts)}"

    def _get_filter_suggestions(
        self,
        resource: str,
        filters: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[str]:
        """Get suggestions for filter values from schema."""
        suggestions = []

        if not self.schema_introspector:
            return suggestions

        for field, val in filters.items():
            allowed = self.schema_introspector.get_allowed_values(resource, field)
            if allowed:
                values_str = ", ".join(str(v) for v in allowed[:5])
                if len(allowed) > 5:
                    values_str += f" (+{len(allowed) - 5} more)"
                suggestions.append(f"Valid values for '{field}': {values_str}")

        return suggestions

    def _find_similar_values(
        self,
        resource: str,
        field: str,
        value: Any
    ) -> List[str]:
        """Find similar valid values for a field."""
        if not self.schema_introspector:
            return []

        allowed = self.schema_introspector.get_allowed_values(resource, field)
        if not allowed:
            return []

        # Return values that are different from the current one
        value_str = str(value).lower()
        return [v for v in allowed if str(v).lower() != value_str][:3]

    def _find_similar_resources(
        self,
        resource: str,
        schema: Dict[str, Any]
    ) -> List[str]:
        """Find resources with similar names."""
        if not resource:
            return []

        all_resources = list(schema.get("resources", {}).keys())
        resource_lower = resource.lower()

        similar = []
        for res in all_resources:
            if res.lower() != resource_lower:
                # Check for common prefixes or word overlap
                res_words = set(res.lower().replace("-", "_").split("_"))
                resource_words = set(resource_lower.replace("-", "_").split("_"))
                if res_words & resource_words:
                    similar.append(res)

        return similar

    def set_pending(self, session_id: str, parsed: Dict[str, Any], query: str):
        """Store a pending confirmation for a session."""
        self._pending[session_id] = {
            "parsed": parsed,
            "query": query,
        }

    def get_pending(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get pending confirmation for a session."""
        return self._pending.get(session_id)

    def clear_pending(self, session_id: str):
        """Clear pending confirmation for a session."""
        self._pending.pop(session_id, None)
