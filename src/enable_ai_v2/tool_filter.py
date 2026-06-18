"""
Shared tool filtering for all LLM providers.

Two layers (parent chooses how much to do upstream):

1. Spec filtering (optional, parent/KQSPL) - reduce tool count before conversion
2. Query filtering (automatic) - pick most relevant tools per user query
"""

from __future__ import annotations

import copy
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .query_intent import QueryIntent
    from .types import ToolCall

# Provider tool limits (OpenAI hard cap; Anthropic benefits from smaller sets)
PROVIDER_MAX_TOOLS = {
    "openai": 128,
    "anthropic": 200,
}

DEFAULT_EXCLUDE_PATTERNS = [
    r"bulk",
    r"batch",
    r"export",
    r"import",
    r"admin",
    r"internal",
    r"webhook",
    r"audit",
    r"log",
    r"migration",
    r"available_transitions",
    r"activities",
    r"create_retrieve",
]

READ_ACTION_WORDS = ("list", "show", "get", "find", "search", "fetch", "display", "all")
CREATE_ACTION_WORDS = ("create", "add", "new", "insert")
UPDATE_ACTION_WORDS = ("update", "edit", "change", "modify", "patch")
DELETE_ACTION_WORDS = ("delete", "remove", "destroy", "cancel")
MUTATION_TOOL_MARKERS = (
    "partial_update", "update", "create", "destroy", "delete", "patch",
)
EMBEDDED_LOOKUP_BAD_MARKERS = (
    "by_service_id", "by_service_order", "service_order_id", "by_so",
)
TRANSITION_TOOL_MARKERS = (
    "check_transition", "transition_impact", "available_transitions", "transition",
)
COUNT_QUERY_WORDS = ("how many", "count", "number of", "total", "how much")
# Default business-code patterns (parent can override via Config.id_code_patterns)
DEFAULT_ID_CODE_PATTERNS = [
    r"\b([a-z]{2,})-(\d+)\b",   # SO-159, REQ-123
    r"\b([a-z]{2,})(\d+)\b",    # SO159
]

# Legacy patterns for standalone numeric refs when no business code present
STANDALONE_ID_PATTERNS = [
    re.compile(r"\b(?:order|invoice|ticket|report)[\s#-]*(\d+)\b", re.IGNORECASE),
    re.compile(r"#(\d+)\b"),
]

SEARCH_TOOL_HINTS = ("list", "search", "filter")
RETRIEVE_TOOL_HINTS = ("retrieve", "get", "read", "detail")

# Tokens that cause false-positive tool name matches (SO-159 → "so", "159")
_LOW_SIGNAL_TOKENS = frozenset({
    "so", "id", "no", "at", "on", "in", "to", "or", "an", "as", "by",
})


def get_max_tools_for_provider(provider: str, override: Optional[int] = None) -> int:
    """Return max tools to send for a provider."""
    if override is not None:
        return override
    return PROVIDER_MAX_TOOLS.get(provider, PROVIDER_MAX_TOOLS["openai"])


def filter_openapi_spec(
    spec: dict,
    *,
    exclude_patterns: Optional[list[str]] = None,
    allowed_methods: Optional[list[str]] = None,
    is_admin: bool = True,
) -> dict:
    """
    Pre-filter OpenAPI spec before tool conversion.

    Parent modules (e.g. KQSPL) can call this when building Config, or pass
    equivalent options via Config.tool_exclude_patterns / allowed_http_methods.

    Args:
        spec: OpenAPI spec dict
        exclude_patterns: Regex patterns matched against path + operationId
        allowed_methods: HTTP methods to keep (e.g. ["GET"] for read-only users)
        is_admin: If False, default exclude patterns are applied when none given
    """
    filtered = copy.deepcopy(spec)
    paths = filtered.get("paths", {})
    if not paths:
        return filtered

    patterns = list(exclude_patterns or [])
    if not is_admin and not patterns:
        patterns = list(DEFAULT_EXCLUDE_PATTERNS)

    allowed = {m.lower() for m in allowed_methods} if allowed_methods else None
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    new_paths: dict = {}
    for path, methods in paths.items():
        kept_methods: dict = {}
        for method, details in methods.items():
            method_lower = method.lower()
            if method_lower not in ("get", "post", "put", "patch", "delete"):
                continue
            if allowed and method_lower not in allowed:
                continue

            haystack = " ".join(
                [
                    path,
                    details.get("operationId", ""),
                    details.get("summary", ""),
                    details.get("description", ""),
                ]
            )
            if any(p.search(haystack) for p in compiled):
                continue

            kept_methods[method] = details

        if kept_methods:
            new_paths[path] = kept_methods

    filtered["paths"] = new_paths
    return filtered


def _normalize_token(token: str) -> str:
    """Normalize resource tokens for matching (service-orders -> serviceorders)."""
    return re.sub(r"[_\-\s]+", "", token.lower())


def extract_business_codes(
    text: str,
    patterns: Optional[list[str]] = None,
) -> list[str]:
    """
    Extract prefixed business reference codes (SO-159, REQ-123).

    Does NOT emit bare numeric IDs when a prefixed code is present.
    """
    codes: list[str] = []
    seen: set[str] = set()
    raw_patterns = patterns or DEFAULT_ID_CODE_PATTERNS

    for raw in raw_patterns:
        for match in re.finditer(raw, text, re.IGNORECASE):
            if match.lastindex and match.lastindex >= 2:
                prefix = match.group(1).lower()
                number = match.group(2)
                for candidate in (f"{prefix}-{number}", f"{prefix}{number}"):
                    if candidate not in seen:
                        seen.add(candidate)
                        codes.append(candidate)
    return codes


def has_business_code(text: str, patterns: Optional[list[str]] = None) -> bool:
    """True when text contains a prefixed business reference code."""
    return bool(extract_business_codes(text, patterns))


def numeric_suffix_from_code(code: str) -> Optional[str]:
    """Extract numeric suffix from SO-159 or SO159."""
    match = re.search(r"(\d+)$", code)
    return match.group(1) if match else None


def extract_resource_ids(
    text: str,
    patterns: Optional[list[str]] = None,
) -> list[str]:
    """
    Extract resource identifiers for tool matching.

    When business codes exist (SO-159), returns only prefixed forms — not bare 159.
    """
    codes = extract_business_codes(text, patterns)
    if codes:
        return codes

    ids: list[str] = []
    seen: set[str] = set()
    for pattern in STANDALONE_ID_PATTERNS:
        for match in pattern.finditer(text):
            number = match.group(1)
            if number not in seen:
                seen.add(number)
                ids.append(number)
    return ids


def _is_bare_code_reply(text: str, patterns: Optional[list[str]] = None) -> bool:
    """True when message is essentially only a business code (follow-up reply)."""
    stripped = text.strip()
    codes = extract_business_codes(stripped, patterns)
    if not codes:
        return False
    normalized = re.sub(r"[\s.,!?]+", "", stripped.lower())
    for code in codes:
        if normalized in (code.lower(), code.replace("-", "").lower()):
            return True
    return False


def is_count_query(query: str) -> bool:
    """True when user is asking for a count/total."""
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in COUNT_QUERY_WORDS)


def is_read_only_query(query: str) -> bool:
    """True when the user is asking to view/list/count, not mutate."""
    q = query.lower()
    if any(w in q for w in CREATE_ACTION_WORDS + UPDATE_ACTION_WORDS + DELETE_ACTION_WORDS):
        return False
    return any(w in q for w in READ_ACTION_WORDS) or is_count_query(query)


def is_mutation_tool(tool_name: str) -> bool:
    """True for update/create/delete style tools."""
    name = tool_name.lower()
    if "list" in name or "retrieve" in name or "search" in name:
        return False
    return any(marker in name for marker in MUTATION_TOOL_MARKERS)


def build_filter_query(
    query: str,
    conversation_history: Optional[list[dict]] = None,
    id_code_patterns: Optional[list[str]] = None,
) -> str:
    """
    Enrich query for tool scoring using conversation context.

    Helps follow-ups like "who is the technician?" or bare "SO-169" after clarification.
    """
    enriched = query

    if conversation_history and _is_bare_code_reply(query, id_code_patterns):
        for msg in reversed(conversation_history):
            if msg.get("role") != "user":
                continue
            prior = msg.get("content", "").strip()
            if prior and not _is_bare_code_reply(prior, id_code_patterns):
                enriched = f"{prior} {query}"
                break

    parts = [enriched]
    if conversation_history:
        recent = conversation_history[-6:]
        for msg in recent:
            content = msg.get("content", "")
            if content:
                parts.append(content)

    combined = " ".join(parts)
    ids = extract_resource_ids(combined, id_code_patterns)
    if ids and not extract_resource_ids(query, id_code_patterns):
        parts.append(" ".join(ids))
    return " ".join(parts)


def _query_tokens(query: str) -> set[str]:
    """Extract meaningful tokens from a query."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    tokens: set[str] = set()
    for word in words:
        if len(word) > 2 or word.isdigit():
            tokens.add(word)
    for rid in extract_resource_ids(query):
        tokens.add(rid.lower())
        if "-" in rid:
            tokens.update(part for part in rid.lower().split("-") if part)
    return tokens


def _tool_search_text(tool: dict) -> str:
    name = tool.get("name", "")
    desc = tool.get("description", "")
    return f"{name} {desc}".lower()


def _tool_resource_segment(tool_name: str) -> str:
    """Primary resource segment from tool name (service_orders_list → service_orders)."""
    parts = tool_name.lower().split("_")
    actions = {"list", "retrieve", "create", "update", "destroy", "partial", "get", "read", "search"}
    resource_parts = [p for p in parts if p not in actions]
    return "_".join(resource_parts) or tool_name.lower()


def _is_workflow_tool(tool_name: str) -> bool:
    name = tool_name.lower()
    return any(marker in name for marker in (
        "statuses", "activities", "available_transitions", "create_retrieve", "bulk", "webhook"
    ))


def score_tool_for_query(
    query: str,
    tool: dict,
    *,
    id_code_patterns: Optional[list[str]] = None,
    intent: Optional["QueryIntent"] = None,
) -> int:
    """
    Score a tool's relevance to a query.

    Used by all LLM providers via filter_tools_for_query().
    """
    query_lower = query.lower()
    query_words = _query_tokens(query)
    name = tool.get("name", "").lower()
    desc = tool.get("description", "").lower()
    search_text = _tool_search_text(tool)
    normalized_search = _normalize_token(search_text)
    segment = _tool_resource_segment(name)

    score = 0

    business_codes = extract_business_codes(query, id_code_patterns)
    code_numeric_suffixes: set[str] = set()
    if business_codes:
        for code in business_codes:
            suffix = numeric_suffix_from_code(code)
            if suffix:
                code_numeric_suffixes.add(suffix)

    for word in query_words:
        if word in _LOW_SIGNAL_TOKENS:
            continue
        if word.isdigit() and (business_codes or word in code_numeric_suffixes):
            continue
        word_norm = _normalize_token(word)
        if word in name:
            score += 10
        elif word_norm and word_norm == _normalize_token(segment):
            score += 12
        elif word in desc:
            score += 3
        elif word_norm and word_norm in normalized_search:
            score += 5

    # Resource affinity — service orders vs statuses/activities
    from .query_intent import QueryIntent, mentions_service_orders, query_wants_embedded_fields

    if mentions_service_orders(query):
        if segment in ("service_orders", "service_order"):
            score += 18
        if _is_workflow_tool(name):
            score -= 25
        if "statuses" in segment and "service_order" not in segment:
            score -= 20

    # Intent-aware boosts
    if intent is not None:
        if intent == QueryIntent.COUNT and "list" in name:
            score += 10
        elif intent == QueryIntent.SINGLE_DETAIL and ("list" in name or "retrieve" in name):
            score += 8
        elif intent == QueryIntent.MULTI_STEP and "list" in name:
            score += 6
        elif intent == QueryIntent.AGGREGATE and "list" in name:
            score += 4

    # SO/detail queries should not pick workflow endpoints
    q_lower = query.lower()
    if has_business_code(query) or any(w in q_lower for w in ("technician", "assigned", "scheduled", "address", "inspection")):
        if _is_workflow_tool(name):
            score -= 30

    # Read-only queries → penalize mutation endpoints (Q53 partial_update)
    if is_read_only_query(query) and is_mutation_tool(name):
        score -= 45

    # Embedded-field follow-up → penalize alternate lookup endpoints (Q01 by_service_id)
    if query_wants_embedded_fields(query) and any(m in name for m in EMBEDDED_LOOKUP_BAD_MARKERS):
        score -= 50

    from .query_intent import is_scheduling_status_query

    if is_scheduling_status_query(query):
        if segment in ("service_orders", "service_order") and "list" in name:
            score += 20
        if any(m in name for m in TRANSITION_TOOL_MARKERS):
            score -= 50

    # Aggregate queries — "each technician", "per technician" want data, not users list
    if any(p in q_lower for p in ("each technician", "per technician", "by technician")):
        if "user" in segment or "technician" in segment:
            score -= 15  # Penalize users/technicians list
        if "service_order" in segment and "list" in name:
            score += 10  # Prefer service orders for aggregation

    # Action word → HTTP verb hints in tool names
    if any(w in query_lower for w in READ_ACTION_WORDS):
        if any(v in name for v in ("list", "get", "retrieve", "search")):
            score += 2
    if any(w in query_lower for w in CREATE_ACTION_WORDS):
        if any(v in name for v in ("create", "post")):
            score += 2
    if any(w in query_lower for w in UPDATE_ACTION_WORDS):
        if any(v in name for v in ("update", "patch", "put")):
            score += 2
    if any(w in query_lower for w in DELETE_ACTION_WORDS):
        if any(v in name for v in ("delete", "destroy")):
            score += 2

    # Prefer list endpoints when user asks plural/resource questions without an ID
    if any(w in query_lower for w in READ_ACTION_WORDS) and "list" in name:
        score += 1

    # Count queries → prefer list endpoints
    if is_count_query(query) and "list" in name:
        score += 8

    business_codes = extract_business_codes(query, id_code_patterns)
    if business_codes:
        # Business codes → search/list, NOT path {id} retrieve
        if "list" in name or "search" in name or "search" in desc:
            score += 18
        if any(h in name for h in RETRIEVE_TOOL_HINTS) and "{id}" not in desc:
            score += 4
        if "retrieve" in name and not is_count_query(query):
            score -= 12

        for code in business_codes:
            code_norm = _normalize_token(code)
            if code_norm in normalized_search or code.lower() in search_text:
                score += 12

    elif extract_resource_ids(query, id_code_patterns):
        # Standalone numeric id — retrieve may be appropriate
        resource_ids = extract_resource_ids(query, id_code_patterns)
        if any(h in name for h in RETRIEVE_TOOL_HINTS):
            score += 15
        if "list" in name and not is_count_query(query):
            score -= 5
        for rid in resource_ids:
            rid_norm = _normalize_token(rid)
            if rid_norm in normalized_search or rid.lower() in search_text:
                score += 12

    return score


def filter_tools_for_query(
    query: str,
    tools: list[dict],
    *,
    max_tools: int = 128,
    conversation_history: Optional[list[dict]] = None,
    id_code_patterns: Optional[list[str]] = None,
) -> list[dict]:
    """
    Filter tools to the most relevant ones for a query.

    Shared by OpenAI, Anthropic, and any future LLM client.
    """
    if len(tools) <= max_tools:
        return tools

    filter_query = build_filter_query(query, conversation_history, id_code_patterns)

    from .query_intent import classify_query_intent
    intent = classify_query_intent(query)

    scored = [
        (
            score_tool_for_query(
                filter_query, tool, id_code_patterns=id_code_patterns, intent=intent
            ),
            tool,
        )
        for tool in tools
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    # Keep all positively scored tools first for accuracy
    positive = [tool for score, tool in scored if score > 0]

    # Multi-resource queries: keep top tools per distinct resource prefix
    if len(positive) > max_tools:
        positive = _cap_tools_by_resource(positive, scored, max_tools)

    if len(positive) >= max_tools:
        return positive[:max_tools]

    # Fill remaining slots with highest-ranked tools (including zero-score fallbacks)
    selected = list(positive)
    seen = {tool["name"] for tool in selected}
    for _, tool in scored:
        if len(selected) >= max_tools:
            break
        if tool["name"] not in seen:
            selected.append(tool)
            seen.add(tool["name"])

    return selected


def _tool_resource_prefix(tool_name: str) -> str:
    """Group tools by resource prefix (service_orders_list -> service_orders)."""
    return _tool_resource_segment(tool_name)


def _cap_tools_by_resource(
    positive: list[dict],
    scored: list[tuple[int, dict]],
    max_tools: int,
) -> list[dict]:
    """Ensure multi-resource queries keep tools from each mentioned resource."""
    score_map = {tool["name"]: score for score, tool in scored}
    by_resource: dict[str, list[dict]] = {}
    for tool in positive:
        prefix = _tool_resource_prefix(tool["name"])
        by_resource.setdefault(prefix, []).append(tool)

    if len(by_resource) <= 1:
        return positive

    # Take best tool per resource first, then fill remaining slots
    selected: list[dict] = []
    seen: set[str] = set()
    for tools in by_resource.values():
        tools.sort(key=lambda t: score_map.get(t["name"], 0), reverse=True)
        best = tools[0]
        if best["name"] not in seen:
            selected.append(best)
            seen.add(best["name"])

    for score, tool in sorted(scored, key=lambda item: item[0], reverse=True):
        if len(selected) >= max_tools:
            break
        if tool["name"] not in seen:
            selected.append(tool)
            seen.add(tool["name"])

    return selected
