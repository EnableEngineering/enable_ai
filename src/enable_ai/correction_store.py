"""
Correction Store - Store and learn from user corrections.

This module persists corrections for:
1. Improving future query parsing
2. Updating synonyms dynamically
3. Analytics on common misunderstandings

Supports multiple backends:
- InMemory (development/testing)
- Redis (production - high volume)
- Database (production - persistence)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from .types import CorrectionRecord
from .utils import setup_logger
from . import constants


class CorrectionStore(ABC):
    """
    Abstract base class for correction storage.

    Corrections capture:
    - Original query
    - System's interpretation (parsed result)
    - User's correction/clarification
    - Final correct interpretation

    Subclasses implement storage backend (memory, Redis, database).
    """

    @abstractmethod
    def add_correction(
        self,
        session_id: str,
        original_query: str,
        system_interpretation: Dict[str, Any],
        user_correction: str,
        correct_interpretation: Dict[str, Any]
    ) -> str:
        """
        Store a correction for learning.

        Args:
            session_id: User session identifier
            original_query: What the user originally asked
            system_interpretation: How the system parsed it
            user_correction: What the user said to correct
            correct_interpretation: The corrected parsing

        Returns:
            Correction ID
        """
        pass

    @abstractmethod
    def get_correction(self, correction_id: str) -> Optional[CorrectionRecord]:
        """
        Get a specific correction by ID.

        Args:
            correction_id: The correction's unique ID

        Returns:
            CorrectionRecord or None if not found
        """
        pass

    @abstractmethod
    def get_similar_corrections(
        self,
        query: str,
        resource: Optional[str] = None,
        limit: int = 5
    ) -> List[CorrectionRecord]:
        """
        Find similar past corrections to inform current parsing.

        Args:
            query: Current query to find similar corrections for
            resource: Optional resource to filter by
            limit: Maximum corrections to return

        Returns:
            List of similar CorrectionRecords
        """
        pass

    @abstractmethod
    def get_correction_patterns(self, resource: str) -> Dict[str, Any]:
        """
        Get aggregated correction patterns for a resource.

        Analyzes corrections to find common patterns like:
        - Users often mean X when they say Y
        - Field Z is frequently mis-parsed

        Args:
            resource: Resource name to analyze

        Returns:
            Dict of patterns and frequencies
        """
        pass

    @abstractmethod
    def get_learned_synonyms(self, resource: str) -> Dict[str, str]:
        """
        Get synonyms learned from corrections.

        Only returns synonyms that have been corrected multiple times
        (threshold defined by CORRECTION_PATTERN_MIN_FREQUENCY).

        Args:
            resource: Resource name

        Returns:
            Dict mapping user terms to canonical values
            e.g., {"customer": "Client", "pending": "New"}
        """
        pass


class InMemoryCorrectionStore(CorrectionStore):
    """
    In-memory implementation of CorrectionStore.

    Suitable for:
    - Development and testing
    - Single-instance deployments
    - Prototyping

    Note: Data is lost on restart. For production, use Redis or database backend.
    """

    def __init__(self, max_per_session: int = None):
        """
        Initialize in-memory store.

        Args:
            max_per_session: Max corrections per session (prevents memory bloat)
        """
        self._corrections: List[Dict[str, Any]] = []
        self._session_counts: Dict[str, int] = {}
        self._max_per_session = max_per_session or constants.CORRECTION_STORE_MAX_PER_SESSION
        self.logger = setup_logger('enable_ai.correction_store')

        self.logger.info("InMemoryCorrectionStore initialized")

    def add_correction(
        self,
        session_id: str,
        original_query: str,
        system_interpretation: Dict[str, Any],
        user_correction: str,
        correct_interpretation: Dict[str, Any]
    ) -> str:
        """Store a correction."""
        # Check session limit
        session_count = self._session_counts.get(session_id, 0)
        if session_count >= self._max_per_session:
            self.logger.warning(
                f"Session {session_id} reached correction limit ({self._max_per_session})"
            )
            # Remove oldest correction for this session
            self._remove_oldest_for_session(session_id)

        correction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        resource = correct_interpretation.get("resource")

        correction = {
            "id": correction_id,
            "session_id": session_id,
            "original_query": original_query,
            "system_interpretation": system_interpretation,
            "user_correction": user_correction,
            "correct_interpretation": correct_interpretation,
            "timestamp": timestamp,
            "resource": resource,
        }

        self._corrections.append(correction)
        self._session_counts[session_id] = session_count + 1

        self.logger.info(f"Stored correction {correction_id} for query: {original_query[:50]}...")

        return correction_id

    def get_correction(self, correction_id: str) -> Optional[CorrectionRecord]:
        """Get a specific correction."""
        for corr in self._corrections:
            if corr["id"] == correction_id:
                return self._dict_to_record(corr)
        return None

    def get_similar_corrections(
        self,
        query: str,
        resource: Optional[str] = None,
        limit: int = None
    ) -> List[CorrectionRecord]:
        """Find similar past corrections using simple keyword matching."""
        limit = limit or constants.CORRECTION_SIMILAR_LIMIT
        query_words = set(query.lower().split())
        scored = []

        for corr in self._corrections:
            # Filter by resource if specified
            if resource and corr.get("resource") != resource:
                continue

            # Simple keyword overlap scoring
            orig_words = set(corr["original_query"].lower().split())
            overlap = len(query_words & orig_words)

            if overlap > 0:
                # Boost score if resource matches
                score = overlap
                if corr.get("resource") == resource:
                    score += 2
                scored.append((score, corr))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Return top results as CorrectionRecords
        return [self._dict_to_record(c) for _, c in scored[:limit]]

    def get_correction_patterns(self, resource: str) -> Dict[str, Any]:
        """Analyze corrections to find common patterns."""
        patterns: Dict[str, Any] = {
            "filter_corrections": {},
            "resource_corrections": {},
            "intent_corrections": {},
            "total_corrections": 0,
        }

        resource_corrections = [
            c for c in self._corrections
            if c.get("resource") == resource
        ]
        patterns["total_corrections"] = len(resource_corrections)

        for corr in resource_corrections:
            sys_interp = corr.get("system_interpretation", {})
            correct_interp = corr.get("correct_interpretation", {})

            # Track filter value corrections
            sys_filters = sys_interp.get("filters", {})
            correct_filters = correct_interp.get("filters", {})

            for field in set(list(sys_filters.keys()) + list(correct_filters.keys())):
                sys_val = sys_filters.get(field, {}).get("value") if isinstance(sys_filters.get(field), dict) else sys_filters.get(field)
                correct_val = correct_filters.get(field, {}).get("value") if isinstance(correct_filters.get(field), dict) else correct_filters.get(field)

                if sys_val != correct_val:
                    key = f"{field}_corrections"
                    if key not in patterns["filter_corrections"]:
                        patterns["filter_corrections"][key] = []
                    patterns["filter_corrections"][key].append({
                        "from": sys_val,
                        "to": correct_val,
                    })

            # Track intent corrections
            if sys_interp.get("intent") != correct_interp.get("intent"):
                key = f"{sys_interp.get('intent')}_to_{correct_interp.get('intent')}"
                patterns["intent_corrections"][key] = patterns["intent_corrections"].get(key, 0) + 1

        return patterns

    def get_learned_synonyms(self, resource: str) -> Dict[str, str]:
        """Get synonyms learned from repeated corrections."""
        synonyms: Dict[str, str] = {}
        correction_counts: Dict[str, Dict[str, int]] = {}

        for corr in self._corrections:
            if corr.get("resource") != resource:
                continue

            sys_interp = corr.get("system_interpretation", {})
            correct_interp = corr.get("correct_interpretation", {})

            sys_filters = sys_interp.get("filters", {})
            correct_filters = correct_interp.get("filters", {})

            for field in set(list(sys_filters.keys()) + list(correct_filters.keys())):
                sys_val = sys_filters.get(field, {}).get("value") if isinstance(sys_filters.get(field), dict) else sys_filters.get(field)
                correct_val = correct_filters.get(field, {}).get("value") if isinstance(correct_filters.get(field), dict) else correct_filters.get(field)

                if sys_val and correct_val and str(sys_val).lower() != str(correct_val).lower():
                    # Track this correction pair
                    key = f"{field}:{str(sys_val).lower()}"
                    if key not in correction_counts:
                        correction_counts[key] = {}
                    correction_counts[key][str(correct_val)] = correction_counts[key].get(str(correct_val), 0) + 1

        # Only include synonyms that have been corrected enough times
        min_freq = constants.CORRECTION_PATTERN_MIN_FREQUENCY

        for key, corrected_values in correction_counts.items():
            for corrected_val, count in corrected_values.items():
                if count >= min_freq:
                    original_term = key.split(":", 1)[1]  # Extract the original value
                    synonyms[original_term] = corrected_val
                    self.logger.info(f"Learned synonym: '{original_term}' -> '{corrected_val}' (seen {count} times)")

        return synonyms

    def get_common_misunderstandings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Analytics: what queries commonly fail?

        Returns:
            List of (query_pattern, count) tuples sorted by frequency
        """
        failures: Dict[str, int] = {}

        for corr in self._corrections:
            query = corr.get("original_query", "")
            # Normalize query for grouping
            normalized = " ".join(query.lower().split()[:5])  # First 5 words
            failures[normalized] = failures.get(normalized, 0) + 1

        # Sort by count and return top N
        sorted_failures = sorted(failures.items(), key=lambda x: -x[1])[:limit]

        return [
            {"query_pattern": query, "count": count}
            for query, count in sorted_failures
        ]

    def clear(self):
        """Clear all corrections (for testing)."""
        self._corrections.clear()
        self._session_counts.clear()
        self.logger.info("Cleared all corrections")

    def _remove_oldest_for_session(self, session_id: str):
        """Remove the oldest correction for a session."""
        for i, corr in enumerate(self._corrections):
            if corr.get("session_id") == session_id:
                del self._corrections[i]
                return

    def _dict_to_record(self, d: Dict[str, Any]) -> CorrectionRecord:
        """Convert dict to CorrectionRecord."""
        return CorrectionRecord(
            session_id=d.get("session_id", ""),
            original_query=d.get("original_query", ""),
            system_interpretation=d.get("system_interpretation", {}),
            user_correction=d.get("user_correction", ""),
            correct_interpretation=d.get("correct_interpretation", {}),
            timestamp=d.get("timestamp"),
            resource=d.get("resource"),
        )


# Placeholder for production backends
class RedisCorrectionStore(CorrectionStore):
    """
    Redis-backed correction store.

    For production use with high volume.
    Requires redis-py package.
    """

    def __init__(self, redis_url: str = None, **kwargs):
        # TODO: Implement Redis backend
        raise NotImplementedError("RedisCorrectionStore not yet implemented")

    def add_correction(self, *args, **kwargs):
        raise NotImplementedError()

    def get_correction(self, *args, **kwargs):
        raise NotImplementedError()

    def get_similar_corrections(self, *args, **kwargs):
        raise NotImplementedError()

    def get_correction_patterns(self, *args, **kwargs):
        raise NotImplementedError()

    def get_learned_synonyms(self, *args, **kwargs):
        raise NotImplementedError()


class DatabaseCorrectionStore(CorrectionStore):
    """
    Database-backed correction store.

    For production use with persistence requirements.
    Requires SQLAlchemy or Django ORM.
    """

    def __init__(self, connection_string: str = None, **kwargs):
        # TODO: Implement database backend
        raise NotImplementedError("DatabaseCorrectionStore not yet implemented")

    def add_correction(self, *args, **kwargs):
        raise NotImplementedError()

    def get_correction(self, *args, **kwargs):
        raise NotImplementedError()

    def get_similar_corrections(self, *args, **kwargs):
        raise NotImplementedError()

    def get_correction_patterns(self, *args, **kwargs):
        raise NotImplementedError()

    def get_learned_synonyms(self, *args, **kwargs):
        raise NotImplementedError()
