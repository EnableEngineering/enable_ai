"""
Unified semantic filter injection driven by resource_hints only.

Phrase → filter mappings come from each field's synonyms in config/schema
(e.g. resource_hints.inventory-consumables.stock_level.synonyms).
No hardcoded phrases or field names.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .hint_utils import apply_count_default_filters
from .query_normalize import strip_quotes_for_matching
from .utils import setup_logger

logger = setup_logger("enable_ai.semantic_filters")


def _normalize_resource_name(name: str) -> str:
    return (name or "").lower().replace("-", "_").replace(" ", "_")


def _get_resource_hints(
    resource_hints: Dict[str, Any],
    resource: str,
) -> Dict[str, Any]:
    """Look up hints for a resource, trying exact and normalized names."""
    if not resource_hints or not resource:
        return {}
    if resource in resource_hints and isinstance(resource_hints[resource], dict):
        return resource_hints[resource]
    target = _normalize_resource_name(resource)
    for name, hints in resource_hints.items():
        if isinstance(hints, dict) and _normalize_resource_name(name) == target:
            return hints
    return {}


def _filter_value_from_synonym(mapped_value: Any, phrase: str) -> Any:
    """
    Resolve the filter value to inject from a synonym mapping.

    Simple mappings use the value as-is (e.g. "low stock" -> "low").
    Complex API mappings (field__op=value) use the phrase when it is a
    single token; api_matcher._validate_filter_values expands them later.
    """
    if not isinstance(mapped_value, str):
        return mapped_value
    if "=" in mapped_value and "__" in mapped_value.split("=", 1)[0]:
        # e.g. current_quantity__lt=10 — keep the spoken token from the phrase
        parts = phrase.strip().split()
        return parts[0] if parts else mapped_value
    return mapped_value


def _phrase_in_query(phrase: str, text: str) -> bool:
    """Match a synonym phrase in query text (word-boundary for single tokens)."""
    if not phrase or not text:
        return False
    if " " in phrase:
        return phrase in text
    # Optional trailing 's' covers plurals (technician/technicians) without matching substrings (renew/new)
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?:s|es)?(?!\w)", text))


def _synonym_phrases_for_field(field_hints: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Return (phrase_lower, mapped_value) pairs, longest phrases first."""
    synonyms = field_hints.get("synonyms") or {}
    if not isinstance(synonyms, dict):
        return []
    pairs = [(str(phrase).lower().strip(), value) for phrase, value in synonyms.items() if phrase]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def inject_semantic_filters(
    filters: Optional[Dict[str, Any]],
    resource: str,
    query: str,
    resource_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add filters when the query text matches a synonym phrase from resource_hints.

    Idempotent — skips fields already present in filters.
    """
    filters = dict(filters or {})
    text = (query or "").lower()
    rh = _get_resource_hints(resource_hints or {}, resource)
    if not rh:
        return filters

    for field_name, field_hints in rh.items():
        if field_name.startswith("__") or not isinstance(field_hints, dict):
            continue
        if field_name in filters:
            continue
        base = field_name.split("__")[0]
        if base in filters:
            continue

        matched = False
        for phrase, mapped_value in _synonym_phrases_for_field(field_hints):
            if _phrase_in_query(phrase, text):
                value = _filter_value_from_synonym(mapped_value, phrase)
                filters[field_name] = {"operator": "equals", "value": value}
                logger.info(
                    "Semantic filter from hint: %s=%r (phrase=%r, resource=%s)",
                    field_name, value, phrase, resource,
                )
                matched = True
                break

        # Also match canonical values from hints (e.g. user says "New" directly)
        if not matched:
            for value in field_hints.get("values") or []:
                val_str = str(value).lower()
                if _phrase_in_query(val_str, text):
                    filters[field_name] = {"operator": "equals", "value": value}
                    logger.info(
                        "Semantic filter from hint value: %s=%r (resource=%s)",
                        field_name, value, resource,
                    )
                    break

    return filters


def apply_semantic_filters(
    parsed: Dict[str, Any],
    query: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply semantic filter injection to a parsed query dict."""
    if not isinstance(parsed, dict):
        return parsed
    result = dict(parsed)
    hints = (schema or {}).get("resource_hints") or {}
    result["filters"] = inject_semantic_filters(
        result.get("filters"),
        result.get("resource", ""),
        strip_quotes_for_matching(query),
        hints,
    )
    result = apply_count_default_filters(result, query, hints)
    return result
