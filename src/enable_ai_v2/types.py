"""
Data types for Enable AI v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class APICallTrace:
    """Trace of a single API call."""
    endpoint: str
    method: str
    params: dict[str, Any]
    status_code: int
    response_time_ms: int
    success: bool


@dataclass
class ConfidenceScore:
    """Multi-factor confidence scoring."""
    tool_match: float       # How well tool name matches query (0-1)
    param_coverage: float   # % of extracted params that are valid (0-1)
    schema_valid: float     # Schema validation pass rate (0-1)
    cache_boost: float      # Boost for cached patterns (0-0.1)
    overall: float          # Weighted average (0-1)

    @classmethod
    def high(cls) -> "ConfidenceScore":
        """Create a high confidence score (for cache hits)."""
        return cls(
            tool_match=1.0,
            param_coverage=1.0,
            schema_valid=1.0,
            cache_boost=0.1,
            overall=1.0,
        )


@dataclass
class QueryTrace:
    """Full trace of query processing for debugging/logging."""
    query: str
    timestamp: datetime
    session_id: Optional[str] = None
    cache_hit: bool = False
    cache_key: Optional[str] = None
    # LLM thought process before tool selection (provider-agnostic; not Claude-specific)
    reasoning: Optional[str] = None
    llm_provider: Optional[str] = None
    # Classified query intent (count, single_detail, list, multi_step, aggregate)
    intent: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    api_calls: list[APICallTrace] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    confidence: Optional[ConfidenceScore] = None
    response_time_ms: int = 0
    # Number of multi-step rounds executed
    follow_up_rounds: int = 0
    # Rows from last list response(s) for multi-turn drill-down
    list_cache: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "cache_hit": self.cache_hit,
            "reasoning": self.reasoning,
            "llm_reasoning": self.reasoning,
            "llm_provider": self.llm_provider,
            "intent": self.intent,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ],
            "api_calls": [
                {
                    "endpoint": ac.endpoint,
                    "method": ac.method,
                    "status_code": ac.status_code,
                    "success": ac.success,
                }
                for ac in self.api_calls
            ],
            "confidence": self.confidence.overall if self.confidence else None,
            "response_time_ms": self.response_time_ms,
            "follow_up_rounds": self.follow_up_rounds,
        }


@dataclass
class Response:
    """Response from the orchestrator."""
    # Chat message to display to user
    message: str

    # Structured data (optional, for programmatic access)
    data: Optional[Any] = None

    # Was this a successful query?
    success: bool = True

    # Does the system need more info from user?
    needs_input: bool = False

    # Suggested follow-up questions (for chat UI)
    suggestions: list[str] = field(default_factory=list)

    # Structured list rows for parent multi-turn drill-down (count/list follow-ups)
    list_cache: list[dict] = field(default_factory=list)

    # Full trace for debugging (optional)
    trace: Optional[QueryTrace] = None


@dataclass
class CachedQuery:
    """A cached query pattern."""
    pattern: str                    # Normalized query pattern
    tool_calls: list[ToolCall]      # Tool calls to execute
    confidence: float               # Confidence score
    hits: int = 1                   # Number of times this pattern was used
    last_used: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ],
            "confidence": self.confidence,
            "hits": self.hits,
        }


@dataclass
class ValidationResult:
    """Result of validating a tool call."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 1.0
    needs_clarification: bool = False
    clarification_question: Optional[str] = None  # Chat-friendly question


@dataclass
class Endpoint:
    """An API endpoint extracted from OpenAPI spec."""
    path: str
    method: str
    operation_id: str
    summary: str
    description: str
    parameters: list[dict]
    request_body: Optional[dict] = None
    responses: dict = field(default_factory=dict)

    @property
    def tool_name(self) -> str:
        """Generate tool name from endpoint."""
        if self.operation_id:
            return self.operation_id

        # Generate from path: GET /users/{id} → get_users
        parts = self.path.strip("/").split("/")
        name_parts = [self.method.lower()]
        for part in parts:
            if not part.startswith("{"):
                name_parts.append(part.replace("-", "_"))

        return "_".join(name_parts)
