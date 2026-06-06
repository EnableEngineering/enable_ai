"""
Punctuation-only query cleanup for synonym matching (not semantic rewriting).

Semantic understanding stays with the LLM parser and resource_hints — never rewrite
user intent here (e.g. do not map "in X stage" to a status field).
"""

import re
from typing import Optional

_QUOTE_CHARS_RE = re.compile(r"['''\"`]")


def strip_quotes_for_matching(query: Optional[str]) -> str:
    """Remove quote characters so resource_hints synonym matching sees bare tokens."""
    if not query:
        return query or ""
    return _QUOTE_CHARS_RE.sub(" ", query.strip())
