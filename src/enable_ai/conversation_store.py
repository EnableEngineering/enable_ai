"""
Conversation Storage Abstraction for Enable AI

Provides interfaces for storing conversation history with different backends:
- Django ORM (recommended for most Django apps)
- Redis (for high-performance caching)
- In-memory (for testing only - NOT production safe)

This replaces the in-memory dict in orchestrator to provide:
- Persistence across server restarts
- Load balancer compatibility
- Horizontal scaling support
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from . import constants


class ConversationStore(ABC):
    """
    Abstract interface for conversation storage.
    
    Implement this using Django models, Redis, or any other backend.
    """
    
    @abstractmethod
    def get_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of {role: 'user'|'assistant', content: '...', metadata?: {...}} dicts
            in chronological order (oldest first). metadata is optional (e.g. filters, intent).
        """
        pass
    
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to conversation history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (e.g., filters, intent)
        """
        pass
    
    @abstractmethod
    def clear_history(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier to clear
        """
        pass


class DjangoConversationStore(ConversationStore):
    """
    Django ORM implementation of conversation storage.
    
    Usage:
        from myapp.models import ConversationMessage
        store = DjangoConversationStore(ConversationMessage)
        
    Required model structure:
        class ConversationMessage(models.Model):
            session_id = models.CharField(max_length=255, db_index=True)
            role = models.CharField(max_length=20)  # 'user' or 'assistant'
            content = models.TextField()
            metadata = models.JSONField(default=dict)  # Optional
            created_at = models.DateTimeField(auto_now_add=True)
    """
    
    def __init__(self, model_class):
        """
        Initialize with Django model class.
        
        Args:
            model_class: Django model class for storing messages
        """
        self.model = model_class
    
    def get_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get last N messages from Django DB. Includes metadata when model has metadata field."""
        if limit is None:
            limit = constants.CONVERSATION_HISTORY_LIMIT
        messages = self.model.objects.filter(
            session_id=session_id
        ).order_by('-created_at')[:limit]
        
        # Return in chronological order (oldest first)
        out = []
        for msg in reversed(messages):
            entry = {"role": msg.role, "content": msg.content}
            if hasattr(msg, "metadata") and msg.metadata:
                entry["metadata"] = msg.metadata
            out.append(entry)
        return out
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Save message to Django DB."""
        self.model.objects.create(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
    
    def clear_history(self, session_id: str):
        """Delete all messages for a session."""
        self.model.objects.filter(session_id=session_id).delete()


class RedisConversationStore(ConversationStore):
    """
    Redis implementation of conversation storage.
    
    Usage:
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        store = RedisConversationStore(redis_client)
    
    Features:
        - Fast in-memory storage with persistence
        - Automatic expiry (7 days default)
        - Keeps last 20 messages per session
    """
    
    def __init__(self, redis_client, expiry_seconds: int = 7 * 24 * 60 * 60, max_messages: int = None):
        """
        Initialize with Redis client.
        
        Args:
            redis_client: Redis client instance
            expiry_seconds: How long to keep conversation (default: 7 days)
            max_messages: Maximum messages to keep per session (default: REDIS_MAX_MESSAGES)
        """
        self.redis = redis_client
        self.expiry_seconds = expiry_seconds
        self.max_messages = max_messages if max_messages is not None else constants.REDIS_MAX_MESSAGES
    
    def get_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get last N messages from Redis. Stored messages may include metadata."""
        import json
        if limit is None:
            limit = constants.CONVERSATION_HISTORY_LIMIT
        key = f"conversation:{session_id}"
        messages_json = self.redis.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages_json]
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Save message to Redis."""
        import json
        
        key = f"conversation:{session_id}"
        message = {"role": role, "content": content}
        
        # Add metadata if provided
        if metadata:
            message["metadata"] = metadata
        
        self.redis.rpush(key, json.dumps(message))
        
        # Keep only last N messages
        self.redis.ltrim(key, -self.max_messages, -1)
        
        # Set expiry
        self.redis.expire(key, self.expiry_seconds)
    
    def clear_history(self, session_id: str):
        """Delete conversation from Redis."""
        key = f"conversation:{session_id}"
        self.redis.delete(key)


class InMemoryConversationStore(ConversationStore):
    """
    In-memory conversation storage (for testing/development only).
    
    WARNING: This is NOT production safe!
    - Data is lost on server restart
    - Does not work with load balancers
    - Not horizontally scalable
    
    Use DjangoConversationStore or RedisConversationStore for production.
    """
    
    def __init__(self, max_messages: int = None):
        """
        Initialize in-memory storage.
        
        Args:
            max_messages: Maximum messages to keep per session (default: IN_MEMORY_MAX_MESSAGES)
        """
        self._storage = {}  # {session_id: [messages]}
        self.max_messages = max_messages if max_messages is not None else constants.IN_MEMORY_MAX_MESSAGES
    
    def get_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get messages from memory. Messages may include metadata."""
        if limit is None:
            limit = constants.CONVERSATION_HISTORY_LIMIT
        messages = self._storage.get(session_id, [])
        return messages[-limit:]
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to memory."""
        if session_id not in self._storage:
            self._storage[session_id] = []
        
        message = {"role": role, "content": content}
        if metadata:
            message["metadata"] = metadata
        
        self._storage[session_id].append(message)
        
        # Keep only last N messages
        if len(self._storage[session_id]) > self.max_messages:
            self._storage[session_id] = self._storage[session_id][-self.max_messages:]
    
    def clear_history(self, session_id: str):
        """Clear messages from memory."""
        if session_id in self._storage:
            del self._storage[session_id]
