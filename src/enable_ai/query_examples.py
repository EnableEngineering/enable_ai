"""
Match user queries against schema query_examples for routing bias.
"""

import re
from typing import Any, Dict, List, Optional

from .utils import setup_logger

logger = setup_logger("enable_ai.query_examples")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _token_overlap(a: str, b: str) -> float:
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def match_query_example(
    query: str,
    schema: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Find the best matching query_examples entry for routing bias.

    Examples shape (flexible):
      {"query": "...", "resource": "...", "filters": {...}, "limit": 1, "sort": {...}}
    """
    examples = (schema or {}).get("query_examples") or []
    if not examples or not query:
        return None

    q_norm = _normalize(query)
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for ex in examples:
        if not isinstance(ex, dict):
            continue
        ex_query = ex.get("query") or ex.get("example") or ""
        ex_norm = _normalize(str(ex_query))
        if not ex_norm:
            continue

        score = 0.0
        if q_norm == ex_norm:
            score = 1.0
        elif ex_norm in q_norm or q_norm in ex_norm:
            score = 0.92
        else:
            score = _token_overlap(q_norm, ex_norm)

        if score > best_score and score >= 0.55:
            best_score = score
            best = ex

    if best:
        logger.info(
            "Matched query_example (score=%.2f): %r",
            best_score, best.get("query") or best.get("example"),
        )
    return best


def apply_query_example_defaults(
    parsed: Dict[str, Any],
    query: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge query_example hints into parsed when match is strong enough."""
    if not isinstance(parsed, dict):
        return parsed

    match = match_query_example(query, schema)
    if not match:
        return parsed

    result = dict(parsed)
    for key in (
        "resource", "question_type", "display_mode", "limit", "sort",
        "summary_field", "intent",
    ):
        if key in match and match[key] is not None and not result.get(key):
            result[key] = match[key]

    if match.get("filters") and isinstance(match["filters"], dict):
        merged = dict(match["filters"])
        merged.update(result.get("filters") or {})
        result["filters"] = merged

    return result
