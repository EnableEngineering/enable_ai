"""
Conversation storage for multi-turn context.

Provides interfaces for storing conversation history:
- Django ORM (recommended for Django apps)
- Redis (for high-performance caching)
- In-memory (for testing only)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user" or "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationStore(ABC):
    """
    Abstract interface for conversation storage.

    Implement this using Django models, Redis, or any other backend.
    """

    @abstractmethod
    def get_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[Message]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum messages to return (most recent)

        Returns:
            List of Messages in chronological order (oldest first)
        """
        pass

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to conversation history.

        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (tool calls, trace, etc.)
        """
        pass

    @abstractmethod
    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        pass

    def get_llm_messages(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """
        Get history formatted for LLM chat APIs (OpenAI, Anthropic, etc.).

        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts
        """
        history = self.get_history(session_id, limit)
        return [{"role": m.role, "content": m.content} for m in history]

    def get_claude_messages(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Deprecated: use get_llm_messages()."""
        return self.get_llm_messages(session_id, limit)


class InMemoryConversationStore(ConversationStore):
    """
    In-memory conversation storage (for testing only).

    WARNING: Not suitable for production - data is lost on restart
    and not shared across instances.
    """

    def __init__(self, max_messages: int = 100):
        self._store: dict[str, list[Message]] = {}
        self.max_messages = max_messages

    def get_history(self, session_id: str, limit: int = 20) -> list[Message]:
        messages = self._store.get(session_id, [])
        return messages[-limit:]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if session_id not in self._store:
            self._store[session_id] = []

        self._store[session_id].append(
            Message(role=role, content=content, metadata=metadata or {})
        )

        # Trim to max messages
        if len(self._store[session_id]) > self.max_messages:
            self._store[session_id] = self._store[session_id][-self.max_messages:]

    def clear_history(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class DjangoConversationStore(ConversationStore):
    """
    Django ORM implementation of conversation storage.

    Usage:
        from myapp.models import ConversationMessage
        store = DjangoConversationStore(ConversationMessage)

    Required model structure:
        class ConversationMessage(models.Model):
            session_id = models.CharField(max_length=255, db_index=True)
            role = models.CharField(max_length=20)
            content = models.TextField()
            metadata = models.JSONField(default=dict)
            created_at = models.DateTimeField(auto_now_add=True)
    """

    def __init__(self, model_class: Any):
        """
        Initialize with Django model class.

        Args:
            model_class: Django model class for storing messages
        """
        self.model = model_class

    def get_history(self, session_id: str, limit: int = 20) -> list[Message]:
        messages = self.model.objects.filter(
            session_id=session_id
        ).order_by("-created_at")[:limit]

        # Return in chronological order (oldest first)
        result = []
        for msg in reversed(messages):
            result.append(
                Message(
                    role=msg.role,
                    content=msg.content,
                    metadata=getattr(msg, "metadata", {}) or {},
                    timestamp=msg.created_at,
                )
            )
        return result

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model.objects.create(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )

    def clear_history(self, session_id: str) -> None:
        self.model.objects.filter(session_id=session_id).delete()


class RedisConversationStore(ConversationStore):
    """
    Redis implementation of conversation storage.

    Usage:
        import redis
        redis_client = redis.Redis(host='localhost', port=6379)
        store = RedisConversationStore(redis_client)

    Features:
        - Fast in-memory storage with persistence
        - Automatic expiry (7 days default)
        - Horizontal scaling support
    """

    def __init__(
        self,
        redis_client: Any,
        expiry_seconds: int = 7 * 24 * 60 * 60,  # 7 days
        max_messages: int = 100,
        key_prefix: str = "enable_ai:conv:",
    ):
        """
        Initialize with Redis client.

        Args:
            redis_client: Redis client instance
            expiry_seconds: TTL for conversations (default: 7 days)
            max_messages: Maximum messages to keep per session
            key_prefix: Redis key prefix
        """
        self.redis = redis_client
        self.expiry_seconds = expiry_seconds
        self.max_messages = max_messages
        self.key_prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def get_history(self, session_id: str, limit: int = 20) -> list[Message]:
        key = self._key(session_id)
        raw_messages = self.redis.lrange(key, -limit, -1)

        messages = []
        for raw in raw_messages:
            data = json.loads(raw)
            messages.append(
                Message(
                    role=data["role"],
                    content=data["content"],
                    metadata=data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                )
            )
        return messages

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        key = self._key(session_id)

        message = Message(role=role, content=content, metadata=metadata or {})
        self.redis.rpush(key, json.dumps(message.to_dict()))

        # Trim to max messages
        self.redis.ltrim(key, -self.max_messages, -1)

        # Set/refresh expiry
        self.redis.expire(key, self.expiry_seconds)

    def clear_history(self, session_id: str) -> None:
        self.redis.delete(self._key(session_id))
