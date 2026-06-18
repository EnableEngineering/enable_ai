"""
Query pattern caching for low-latency repeated queries.

Two-layer cache:
1. Exact match - hash of normalized query
2. Pattern match - parameterized patterns with variable slots
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .types import CachedQuery, ToolCall


class QueryCache:
    """Two-layer cache for query patterns."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Exact match cache: hash -> CachedQuery
        self._exact_cache: dict[str, CachedQuery] = {}

        # Pattern cache: pattern_hash -> CachedQuery
        self._pattern_cache: dict[str, CachedQuery] = {}

    def get(self, query: str) -> Optional[CachedQuery]:
        """
        Look up query in cache.

        Returns CachedQuery if found and not expired, None otherwise.
        """
        normalized = self._normalize(query)

        # Try exact match first
        key = self._hash(normalized)
        if key in self._exact_cache:
            cached = self._exact_cache[key]
            if not self._is_expired(cached):
                cached.hits += 1
                cached.last_used = datetime.now()
                return cached
            else:
                del self._exact_cache[key]

        # Try pattern match
        pattern, extracted = self._extract_pattern(normalized)
        pattern_key = self._hash(pattern)

        if pattern_key in self._pattern_cache:
            cached = self._pattern_cache[pattern_key]
            if not self._is_expired(cached):
                # Substitute extracted values into cached tool calls
                substituted = self._substitute_values(
                    cached.tool_calls,
                    extracted,
                )
                cached.hits += 1
                cached.last_used = datetime.now()

                # Return a copy with substituted values
                return CachedQuery(
                    pattern=cached.pattern,
                    tool_calls=substituted,
                    confidence=cached.confidence,
                    hits=cached.hits,
                    last_used=cached.last_used,
                )
            else:
                del self._pattern_cache[pattern_key]

        return None

    def put(
        self,
        query: str,
        tool_calls: list[ToolCall],
        confidence: float,
    ) -> None:
        """
        Cache a query result.

        Stores both exact match and pattern match.
        """
        self._evict_if_needed()

        normalized = self._normalize(query)

        # Store exact match
        key = self._hash(normalized)
        self._exact_cache[key] = CachedQuery(
            pattern=normalized,
            tool_calls=tool_calls,
            confidence=confidence,
        )

        # Extract and store pattern
        pattern, extracted = self._extract_pattern(normalized)
        if extracted:  # Only store pattern if there are variables
            pattern_key = self._hash(pattern)
            self._pattern_cache[pattern_key] = CachedQuery(
                pattern=pattern,
                tool_calls=self._parameterize_tool_calls(tool_calls, extracted),
                confidence=confidence * 0.95,  # Slightly lower confidence for patterns
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._exact_cache.clear()
        self._pattern_cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "exact_entries": len(self._exact_cache),
            "pattern_entries": len(self._pattern_cache),
            "max_size": self.max_size,
        }

    def _normalize(self, query: str) -> str:
        """Normalize query for matching."""
        # Lowercase
        q = query.lower().strip()
        # Remove extra whitespace
        q = re.sub(r"\s+", " ", q)
        # Remove punctuation at end
        q = re.sub(r"[.!?]+$", "", q)
        return q

    def _hash(self, text: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _is_expired(self, cached: CachedQuery) -> bool:
        """Check if cached entry is expired."""
        age = (datetime.now() - cached.last_used).total_seconds()
        return age > self.ttl_seconds

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        total = len(self._exact_cache) + len(self._pattern_cache)

        if total >= self.max_size:
            # Evict 10% of oldest exact entries
            to_evict = max(1, len(self._exact_cache) // 10)
            sorted_keys = sorted(
                self._exact_cache.keys(),
                key=lambda k: self._exact_cache[k].last_used,
            )
            for key in sorted_keys[:to_evict]:
                del self._exact_cache[key]

    def _extract_pattern(self, query: str) -> tuple[str, dict[str, str]]:
        """
        Extract variable pattern from query.

        Returns (pattern_string, extracted_values)

        Examples:
            "show order 12345" -> ("show order {id}", {"id": "12345"})
            "invoices for john" -> ("invoices for {name}", {"name": "john"})
        """
        extracted = {}
        pattern = query

        # Extract IDs (numbers)
        id_match = re.search(r"\b(\d{3,})\b", pattern)
        if id_match:
            extracted["id"] = id_match.group(1)
            pattern = pattern.replace(id_match.group(1), "{id}")

        # Extract quoted strings
        quote_match = re.search(r'"([^"]+)"', pattern)
        if quote_match:
            extracted["value"] = quote_match.group(1)
            pattern = pattern.replace(f'"{quote_match.group(1)}"', "{value}")

        # Extract dates (common formats)
        date_match = re.search(
            r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b",
            pattern
        )
        if date_match:
            extracted["date"] = date_match.group(1)
            pattern = pattern.replace(date_match.group(1), "{date}")

        return pattern, extracted

    def _parameterize_tool_calls(
        self,
        tool_calls: list[ToolCall],
        extracted: dict[str, str],
    ) -> list[ToolCall]:
        """Replace extracted values with placeholders in tool calls."""
        parameterized = []

        for tc in tool_calls:
            new_args = {}
            for key, value in tc.arguments.items():
                if isinstance(value, str):
                    for var_name, var_value in extracted.items():
                        if value == var_value:
                            value = f"{{{var_name}}}"
                            break
                new_args[key] = value

            parameterized.append(ToolCall(
                id=tc.id,
                name=tc.name,
                arguments=new_args,
            ))

        return parameterized

    def _substitute_values(
        self,
        tool_calls: list[ToolCall],
        extracted: dict[str, str],
    ) -> list[ToolCall]:
        """Replace placeholders with extracted values."""
        substituted = []

        for tc in tool_calls:
            new_args = {}
            for key, value in tc.arguments.items():
                if isinstance(value, str):
                    for var_name, var_value in extracted.items():
                        if value == f"{{{var_name}}}":
                            value = var_value
                            break
                new_args[key] = value

            substituted.append(ToolCall(
                id=tc.id,
                name=tc.name,
                arguments=new_args,
            ))

        return substituted
