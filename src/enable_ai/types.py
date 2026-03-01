from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field


class APIRequest:
    def __init__(
        self,
        endpoint: str,
        params: dict,
        method: str = 'GET',
        authentication_required: bool = True,
        warnings: list = None  # v0.3.37: Filter warnings to show user
    ):
        self.endpoint = endpoint
        self.params = params
        self.method = method
        self.authentication_required = authentication_required
        self.warnings = warnings or []  # v0.3.37: Store filter validation warnings

class APIResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self.data = data

class APIError:
    def __init__(self, message: str):
        self.message = message

class MissingInformation:
    """
    Represents a request for additional information from the user.
    Used when the user's input is incomplete and we need clarification.
    """
    def __init__(self, message: str, missing_fields: list, matched_endpoint: dict = None, context: dict = None):
        """
        Args:
            message: Human-readable question to ask the user
            missing_fields: List of field names that are required but missing
            matched_endpoint: The endpoint that was matched (for context)
            context: Current parsed context (intent, entities extracted so far)
        """
        self.message = message
        self.missing_fields = missing_fields
        self.matched_endpoint = matched_endpoint
        self.context = context or {}
    
    def __repr__(self):
        return f"MissingInformation(message='{self.message}', missing_fields={self.missing_fields})"


@dataclass
class FilterFieldInfo:
    """
    Information about a filterable field discovered from schema.

    Used by SchemaIntrospector to provide field metadata for
    filter validation and query building.
    """
    name: str
    type: str = "string"  # string, integer, boolean, number, array
    allowed_values: Optional[List[str]] = None
    operators: List[str] = field(default_factory=lambda: ["equals"])
    required: bool = False
    description: str = ""
    synonyms: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.allowed_values is None:
            self.allowed_values = []


@dataclass
class ValidationResult:
    """
    Result of validating a filter value against schema constraints.

    Used by SchemaIntrospector.validate_filter_value() to return
    validation status with optional corrections.
    """
    is_valid: bool
    corrected_value: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    original_value: Optional[Any] = None

    @classmethod
    def valid(cls, value: Any) -> "ValidationResult":
        """Create a valid result."""
        return cls(is_valid=True, corrected_value=value, original_value=value)

    @classmethod
    def invalid(cls, value: Any, warning: str) -> "ValidationResult":
        """Create an invalid result with warning."""
        return cls(is_valid=False, original_value=value, warnings=[warning])

    @classmethod
    def corrected(cls, original: Any, corrected: Any, warning: str = "") -> "ValidationResult":
        """Create a result with corrected value."""
        warnings = [warning] if warning else []
        return cls(is_valid=True, corrected_value=corrected, original_value=original, warnings=warnings)


@dataclass
class ClassificationResult:
    """
    Result of rule-based intent classification.

    Used by IntentClassifier to return classification with confidence score.
    """
    intent: str
    resource: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    question_type: str = "list"  # list, count, details
    classified_by: str = "rules"  # rules or llm

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for integration with existing code."""
        return {
            "intent": self.intent,
            "resource": self.resource,
            "entities": self.entities,
            "filters": self.filters,
            "question_type": self.question_type,
            "classified_by": self.classified_by,
        }


@dataclass
class CorrectionRecord:
    """
    Record of a user correction for learning.

    Stores what the system interpreted vs what the user corrected it to,
    for future improvement of the intent analyzer.
    """
    session_id: str
    original_query: str
    system_interpretation: Dict[str, Any]
    user_correction: str
    correct_interpretation: Dict[str, Any]
    timestamp: Optional[str] = None
    resource: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.utcnow().isoformat()
        if self.resource is None:
            self.resource = self.correct_interpretation.get("resource")