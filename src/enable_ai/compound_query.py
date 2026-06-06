"""
Split compound user messages into independent sub-questions.
"""

import re
from typing import List

_QUESTION_SIGNAL = re.compile(
    r"\b(?:how many|how much|what|which|show|list|count|get|give|tell|are there|is there)\b",
    re.IGNORECASE,
)


def split_compound_questions(query: str) -> List[str]:
    """
    Split comma/semicolon-separated messages into sub-questions when each
    part looks like an independent query.
    """
    text = (query or "").strip()
    if not text:
        return []

    parts = re.split(r"[,;]\s+", text)
    if len(parts) <= 1:
        return [text]

    signaled = [p.strip() for p in parts if p.strip() and _QUESTION_SIGNAL.search(p)]
    if len(signaled) >= 2:
        return signaled

    return [text]
