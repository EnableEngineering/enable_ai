"""
Enable AI - AI-Powered Natural Language Interface for REST APIs

A production-ready library that transforms natural language queries into API calls,
with support for:
- Multi-step query planning and execution
- OpenAPI/Swagger schema support
- JWT authentication
- Multi-turn conversations with persistent context
- Real-time streaming responses
- Pluggable conversation storage (Django, Redis, or custom)
- User context and pronoun resolution ("me", "my", "assigned to me")

Usage:
    from enable_ai import APIOrchestrator
    from enable_ai.conversation_store import DjangoConversationStore

    # Development (in-memory storage - NOT production safe)
    orchestrator = APIOrchestrator()

    # Production with Django (recommended)
    from myapp.models import ConversationMessage
    store = DjangoConversationStore(ConversationMessage)
    orchestrator = APIOrchestrator(conversation_store=store)

    # Process a query
    result = orchestrator.process("get user with id 5")

    # Multi-turn conversation with persistent context
    result1 = orchestrator.process("list users", session_id="abc123")
    result2 = orchestrator.process("show me the first", session_id="abc123")

    # Process with user context (v0.3.29) - resolves "me"/"my" pronouns
    user_context = {
        'user_id': 123,
        'username': 'john@example.com',
        'role': 'Technician',
        'company_id': 456,
        'company': 'ACME Corp'
    }
    result = orchestrator.process(
        "show me service orders assigned to me",
        user_context=user_context,
        session_id="abc123"
    )
    # Result will filter by technician=123 automatically

Version: 0.3.32
"""

__version__ = "0.3.32"

from .query_parser import QueryParser
from .api_matcher import APIMatcher
from .api_client import APIClient
from .orchestrator import APIOrchestrator, process_query
from .execution_planner import ExecutionPlanner
from .schema_loader import SchemaLoader, load_schema
from .schema_validator import SchemaValidator, validate_schema
from .response_formatter import ResponseFormatter
from .progress_tracker import ProgressTracker, ProgressStage, ProgressUpdate
from .types import APIRequest, APIResponse, APIError, MissingInformation
from . import constants

# Conversation stores (v0.3.13)
from .conversation_store import (
    ConversationStore,
    DjangoConversationStore,
    RedisConversationStore,
    InMemoryConversationStore,
)

__all__ = [
    'APIOrchestrator',
    'QueryParser',
    'APIMatcher', 
    'APIClient', 
    'ExecutionPlanner',
    'SchemaLoader',
    'load_schema',
    'SchemaValidator',
    'validate_schema',
    'ResponseFormatter',
    'ProgressTracker',
    'ProgressStage',
    'ProgressUpdate',
    'process_query',
    'APIRequest', 
    'APIResponse', 
    'APIError', 
    'MissingInformation',
    'constants',
    # Conversation stores (v0.3.13)
    'ConversationStore',
    'DjangoConversationStore',
    'RedisConversationStore',
    'InMemoryConversationStore',
]

