"""
Query preprocessing utilities for Enable AI.

Combines punctuation cleanup and compound query splitting.
"""

import re
from typing import List, Optional

# -----------------------------------------------------------------------------
# Quote stripping (from query_normalize.py)
# -----------------------------------------------------------------------------

_QUOTE_CHARS_RE = re.compile(r"['''\"`]")


def strip_quotes_for_matching(query: Optional[str]) -> str:
    """Remove quote characters so resource_hints synonym matching sees bare tokens."""
    if not query:
        return query or ""
    return _QUOTE_CHARS_RE.sub(" ", query.strip())


# -----------------------------------------------------------------------------
# Compound query splitting (from compound_query.py)
# -----------------------------------------------------------------------------

_QUESTION_SIGNAL = re.compile(
    r"\b(?:how many|how much|what|which|show|list|count|get|give|tell|are there|is there)\b",
    re.IGNORECASE,
)

_PREFIX_STRIP = re.compile(
    r"^(?:or\s+)?(?:[\w-]+\s+){0,4}questions?\s+like[-\s]*",
    re.IGNORECASE,
)


def _normalize_compound_text(query: str) -> str:
    text = (query or "").strip()
    text = _PREFIX_STRIP.sub("", text).strip()
    return text


def _signaled_parts(parts: List[str]) -> List[str]:
    return [p.strip() for p in parts if p.strip() and _QUESTION_SIGNAL.search(p)]


def split_compound_questions(query: str) -> List[str]:
    """
    Split compound messages into sub-questions.

    Supports comma/semicolon lists, ``and``-joined questions, and strips
    prefixes like ``or invoicing questions like-``.
    """
    text = _normalize_compound_text(query)
    if not text:
        return []

    and_parts = re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
    if len(and_parts) >= 2:
        signaled = _signaled_parts(and_parts)
        if len(signaled) >= 2:
            return signaled

    comma_parts = re.split(r"[,;]\s+", text)
    if len(comma_parts) >= 2:
        signaled = _signaled_parts(comma_parts)
        if len(signaled) >= 2:
            return signaled

    return [text]
