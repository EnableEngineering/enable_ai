"""
Configuration for Enable AI v2.

All domain-specific data must be provided by the parent module.
enable_ai_v2 does NOT call any APIs for configuration - parent provides everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


@dataclass
class QueryExample:
    """Rich query routing example (parent/KQSPL integration)."""

    query: str
    resource: str = ""
    tools: str = ""
    filters: Optional[dict[str, Any]] = None
    sort: Optional[str] = None
    limit: Optional[int] = None
    question_type: Optional[str] = None  # count|details|list|multi_step
    notes: Optional[str] = None


@dataclass
class IntentPhrases:
    """
    Domain-specific NL phrases for intent detection.

    Parent module (KQSPL) should provide these from config/bootstrap.
    Pip supplies sensible defaults when fields are omitted.
    """

    multi_step_keywords: tuple[str, ...] = (
        "observations",
        "flash report",
        "flash-report",
        "consumables",
        "equipment",
        "probe_details",
        "probe details",
        "nested",
    )
    single_detail_keywords: tuple[str, ...] = (
        "my ",
        "my latest",
        "latest",
        "most recent",
        "last ",
        "recent ",
    )
    aggregate_keywords: tuple[str, ...] = (
        "most ",
        "each ",
        "per technician",
        "per customer",
        " and ",
    )
    # Patterns that should NOT trigger aggregate even if aggregate_keywords match
    aggregate_exclude_patterns: tuple[str, ...] = (
        "most recent",
        "most recently",
        "most recent service order",
        "most recent report",
        "most recent invoice",
    )
    compound_resource_phrases: tuple[str, ...] = (
        "service order",
        "service orders",
        "invoice",
        "invoices",
        "flash report",
        "technician",
        "technicians",
        "user",
        "users",
        "equipment",
        "report",
        "reports",
    )
    compound_resource_segments: dict[str, str] = field(default_factory=lambda: {
        "service order": "service_order",
        "service orders": "service_order",
        "invoice": "invoice",
        "invoices": "invoice",
        "flash report": "flash",
        "technician": "technician",
        "technicians": "technician",
        "user": "user",
        "users": "user",
        "equipment": "equipment",
        "report": "report",
        "reports": "report",
    })
    availability_phrases: tuple[str, ...] = (
        "no jobs",
        "no job",
        "have no jobs",
        "with no jobs",
        "no service orders",
        "no assignments",
        "without assignments",
        "not assigned",
        "who is available",
        "who is free",
        "technicians with no",
        "technician with no",
        "without jobs",
        "not busy",
        "who has no",
        "availability",
        "no open jobs",
        "not working on",
        "idle technician",
    )
    scheduling_phrases: tuple[str, ...] = (
        "scheduled yet",
        "been scheduled",
        "is it scheduled",
        "has it been scheduled",
        "was it scheduled",
    )
    service_order_keywords: tuple[str, ...] = (
        "service order",
        "service orders",
        "so-",
        "so ",
    )
    flash_report_phrases: tuple[str, ...] = (
        "flash report",
        "flash-report",
        "flash reports",
    )
    # Parent resources that trigger flash report follow-up (configurable)
    flash_report_parent_segments: tuple[str, ...] = (
        "service_order",
        "service-order",
    )
    embedded_field_names: tuple[str, ...] = (
        "observations",
        "notes",
        "probe_details",
        "consumables",
        "equipment",
    )
    # AR/Accounts Receivable context keywords (for dashboard formatting)
    ar_context_keywords: tuple[str, ...] = (
        "outstanding",
        "ar dashboard",
        "accounts receivable",
        "highest outstanding",
        "overdue",
    )
    # AR field names for extraction (configurable by parent)
    ar_company_fields: tuple[str, ...] = (
        "company",
        "customer",
        "client",
        "company_name",
        "customer_name",
    )
    ar_amount_fields: tuple[str, ...] = (
        "outstanding",
        "balance",
        "amount",
        "total",
        "outstanding_amount",
        "total_outstanding",
    )

    # === FILTER INJECTION CONFIG (parent must provide) ===

    # Status injection: keyword → status value to inject
    # e.g., {"open": "New", "pending": "Pending", "closed": "Completed"}
    status_injection_map: dict[str, str] = field(default_factory=dict)

    # Status field name per resource segment
    # e.g., {"service_order": "status", "invoice": "status"}
    status_field_map: dict[str, str] = field(default_factory=dict)

    # Equipment in-use keywords (triggers in_use=true filter)
    equipment_in_use_keywords: tuple[str, ...] = ()

    # Equipment filter param name
    equipment_in_use_param: str = "in_use"

    # Company search keywords (triggers company search filter)
    company_search_keywords: tuple[str, ...] = ()

    # Company search param name
    company_search_param: str = "company"

    # Report type injection: keyword → report_type value
    # e.g., {"inspection": "inspection", "maintenance": "maintenance"}
    report_type_injection_map: dict[str, str] = field(default_factory=dict)

    # Report type param name
    report_type_param: str = "report_type"

    # Per-entity aggregation keywords (e.g., "per technician", "by customer")
    # Maps keyword phrase → (entity_type, filter_field)
    # e.g., {"per technician": ("technician", "assigned_to"), "by customer": ("customer", "company")}
    aggregate_per_entity_map: dict[str, tuple[str, str]] = field(default_factory=dict)


DEFAULT_INTENT_PHRASES = IntentPhrases()


def resolve_intent_phrases(config: Optional["Config"] = None) -> IntentPhrases:
    """Merge parent Config.intent_phrases over pip defaults (per-field)."""
    if not config or not config.intent_phrases:
        return DEFAULT_INTENT_PHRASES
    parent = config.intent_phrases
    defaults = DEFAULT_INTENT_PHRASES

    def pick(name: str):
        value = getattr(parent, name)
        if value:
            return value
        return getattr(defaults, name)

    def pick_dict(name: str):
        value = getattr(parent, name, None)
        if value:
            return value
        return getattr(defaults, name)

    return IntentPhrases(
        multi_step_keywords=pick("multi_step_keywords"),
        single_detail_keywords=pick("single_detail_keywords"),
        aggregate_keywords=pick("aggregate_keywords"),
        aggregate_exclude_patterns=pick("aggregate_exclude_patterns"),
        compound_resource_phrases=pick("compound_resource_phrases"),
        compound_resource_segments=pick("compound_resource_segments") or defaults.compound_resource_segments,
        availability_phrases=pick("availability_phrases"),
        scheduling_phrases=pick("scheduling_phrases"),
        service_order_keywords=pick("service_order_keywords"),
        flash_report_phrases=pick("flash_report_phrases"),
        flash_report_parent_segments=pick("flash_report_parent_segments"),
        embedded_field_names=pick("embedded_field_names"),
        ar_context_keywords=pick("ar_context_keywords"),
        ar_company_fields=pick("ar_company_fields"),
        ar_amount_fields=pick("ar_amount_fields"),
        # Filter injection config
        status_injection_map=pick_dict("status_injection_map"),
        status_field_map=pick_dict("status_field_map"),
        equipment_in_use_keywords=pick("equipment_in_use_keywords"),
        equipment_in_use_param=pick("equipment_in_use_param") or defaults.equipment_in_use_param,
        company_search_keywords=pick("company_search_keywords"),
        company_search_param=pick("company_search_param") or defaults.company_search_param,
        report_type_injection_map=pick_dict("report_type_injection_map"),
        report_type_param=pick("report_type_param") or defaults.report_type_param,
        aggregate_per_entity_map=pick_dict("aggregate_per_entity_map"),
    )


def intent_phrases_from_dict(raw: dict[str, Any]) -> IntentPhrases:
    """
    Build IntentPhrases from a KQSPL config.json-style dict.

    Keys match IntentPhrases field names; omitted keys use pip defaults.
    """
    if not raw:
        return DEFAULT_INTENT_PHRASES
    defaults = DEFAULT_INTENT_PHRASES
    segments = dict(defaults.compound_resource_segments)
    if raw.get("compound_resource_segments"):
        segments.update(raw["compound_resource_segments"])

    def as_tuple(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
        val = raw.get(key)
        if not val:
            return fallback
        return tuple(val)

    def as_dict(key: str, fallback: dict) -> dict:
        val = raw.get(key)
        if not val:
            return fallback
        return dict(val)

    return IntentPhrases(
        multi_step_keywords=as_tuple("multi_step_keywords", defaults.multi_step_keywords),
        single_detail_keywords=as_tuple("single_detail_keywords", defaults.single_detail_keywords),
        aggregate_keywords=as_tuple("aggregate_keywords", defaults.aggregate_keywords),
        aggregate_exclude_patterns=as_tuple("aggregate_exclude_patterns", defaults.aggregate_exclude_patterns),
        compound_resource_phrases=as_tuple("compound_resource_phrases", defaults.compound_resource_phrases),
        compound_resource_segments=segments,
        availability_phrases=as_tuple("availability_phrases", defaults.availability_phrases),
        scheduling_phrases=as_tuple("scheduling_phrases", defaults.scheduling_phrases),
        service_order_keywords=as_tuple("service_order_keywords", defaults.service_order_keywords),
        flash_report_phrases=as_tuple("flash_report_phrases", defaults.flash_report_phrases),
        flash_report_parent_segments=as_tuple("flash_report_parent_segments", defaults.flash_report_parent_segments),
        embedded_field_names=as_tuple("embedded_field_names", defaults.embedded_field_names),
        ar_context_keywords=as_tuple("ar_context_keywords", defaults.ar_context_keywords),
        ar_company_fields=as_tuple("ar_company_fields", defaults.ar_company_fields),
        ar_amount_fields=as_tuple("ar_amount_fields", defaults.ar_amount_fields),
        # Filter injection config
        status_injection_map=as_dict("status_injection_map", defaults.status_injection_map),
        status_field_map=as_dict("status_field_map", defaults.status_field_map),
        equipment_in_use_keywords=as_tuple("equipment_in_use_keywords", defaults.equipment_in_use_keywords),
        equipment_in_use_param=raw.get("equipment_in_use_param") or defaults.equipment_in_use_param,
        company_search_keywords=as_tuple("company_search_keywords", defaults.company_search_keywords),
        company_search_param=raw.get("company_search_param") or defaults.company_search_param,
        report_type_injection_map=as_dict("report_type_injection_map", defaults.report_type_injection_map),
        report_type_param=raw.get("report_type_param") or defaults.report_type_param,
        aggregate_per_entity_map=as_dict("aggregate_per_entity_map", defaults.aggregate_per_entity_map),
    )


@dataclass
class ResourceHint:
    """Hints for a specific resource (e.g., invoices, service-orders)."""

    # Field that contains status
    status_field: Optional[str] = None

    # Valid status values (from API/DB, not hardcoded)
    status_values: list[str] = field(default_factory=list)

    # Additional enum fields and their values
    enum_fields: dict[str, list[str]] = field(default_factory=dict)

    # Guidance for the LLM on how to filter this resource
    filter_guidance: Optional[str] = None


@dataclass
class UserContext:
    """
    User context provided by parent module.

    Parent module is responsible for:
    - Fetching user data from its APIs
    - Determining permissions
    - Passing this context to process()
    """

    user_id: Optional[int | str] = None
    username: Optional[str] = None
    role: Optional[str] = None
    company_id: Optional[int | str] = None
    company_name: Optional[str] = None

    # Permissions - parent module determines these
    is_admin: bool = False  # If True, user sees all data (no scoping needed)
    permissions: list[str] = field(default_factory=list)  # e.g., ["invoicing.view", "service_orders.change"]

    # Extra context from parent
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """
    Configuration for the Orchestrator.

    REQUIRED - Parent module MUST provide:
    - openapi_schema: The OpenAPI spec (dict or path)
    - base_url: API base URL
    - auth_token: JWT token for API calls

    RECOMMENDED - For better accuracy:
    - resource_hints: Status values, enums per resource
    - status_synonyms: Natural language → actual status value

    OPTIONAL - Has sensible defaults:
    - model, temperature, cache settings, etc.
    """

    # === REQUIRED ===

    # OpenAPI schema - parent must provide (dict or file path)
    openapi_schema: dict | str = field(default_factory=dict)

    # API base URL
    base_url: str = ""

    # === RECOMMENDED (for accuracy) ===

    # Resource hints - status values, enums from API
    # Key: resource name (e.g., "invoices", "service-orders")
    # Value: ResourceHint with status_values, enum_fields
    resource_hints: dict[str, ResourceHint] = field(default_factory=dict)

    # Status synonyms - natural language → actual value
    # e.g., {"pending": "DRAFT", "paid": "PAID", "open": "New"}
    status_synonyms: dict[str, str] = field(default_factory=dict)

    # All available roles (from /api/users/roles/)
    available_roles: list[str] = field(default_factory=list)

    # === OPTIONAL (has defaults) ===

    # LLM provider: "openai" (default, cheaper) or "anthropic"
    llm_provider: str = "openai"

    # Model (auto-selected based on provider if not specified)
    # OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo
    # Anthropic: claude-sonnet-4-20250514, claude-3-5-sonnet-20241022
    model: str = "gpt-4o"
    max_tokens: int = 16384  # High default to avoid truncation
    temperature: float = 0.0

    # API settings
    request_timeout: int = 30
    max_retries: int = 2
    retry_delay_seconds: float = 0.5

    # Caching
    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600

    # Response
    max_items_in_response: int = 20

    # Debugging
    include_trace: bool = False

    # === TOOL FILTERING (optional - parent/KQSPL decides) ===
    #
    # Parent can pre-filter the OpenAPI spec for better performance:
    # - exclude admin/bulk/internal endpoints for non-admin users
    # - limit to GET-only for read-only users
    #
    # enable-ai always applies query-time filtering in addition to this.

    # Regex patterns matched against path + operationId (None = no spec filtering)
    tool_exclude_patterns: Optional[list[str]] = None

    # HTTP methods to keep, e.g. ["GET"] for read-only users (None = all methods)
    allowed_http_methods: Optional[list[str]] = None

    # Apply default exclude patterns for non-admin users when tool_exclude_patterns unset
    filter_tools_for_non_admin: bool = True

    # Query-time tool filtering (shared across all LLM providers)
    tool_filter_enabled: bool = True

    # Override provider max tools (OpenAI default 128, Anthropic 200)
    max_tools_per_query: Optional[int] = None

    # Conversation context
    conversation_history_limit: int = 20

    # Multi-step tool loop (1 = single pass; 3 recommended for chained queries)
    max_tool_rounds: int = 3

    # Business reference code patterns (SO-159, REQ-123) for search-vs-retrieve logic
    id_code_patterns: Optional[list[str]] = None

    # Nested fields to summarize in detail responses (resource → field names)
    embedded_response_fields: dict[str, list[str]] = field(default_factory=dict)

    # Example queries for routing (parent provides domain-specific examples)
    query_examples: list[Union[dict[str, Any], QueryExample]] = field(default_factory=list)

    # Domain NL phrases for intent detection (parent/KQSPL overrides pip defaults)
    intent_phrases: Optional[IntentPhrases] = None

    # One retry round when API returns HTTP 400 (LLM fixes bad filters)
    max_400_retries: int = 1

    # Rows to retain on COUNT list calls for follow-up drill-down
    count_list_page_size: int = 20

    # Progress callback - parent can provide to show progress in UI
    progress_callback: Optional[Callable[[str, float], None]] = None

    # === SYSTEM PROMPT CUSTOMIZATION ===

    # Additional instructions for the LLM (appended to system prompt)
    system_prompt_suffix: Optional[str] = None

    # Critical instruction about scoping (default: let server handle it)
    scoping_instruction: str = (
        "IMPORTANT: The server automatically scopes data based on user permissions. "
        "Do NOT add user ID filters (like technician=, customer=, created_by=) unless "
        "the user explicitly asks to filter by a specific person. "
        "Only add filters the user explicitly requests (e.g., 'SENT invoices' → status=SENT)."
    )


# === HELPER FUNCTIONS FOR PARENT MODULE ===

def build_resource_hints_from_api(
    openapi_schema: dict,
    service_order_statuses: list[dict],
    service_order_priorities: list[dict],
) -> dict[str, ResourceHint]:
    """
    Helper for parent module to build resource_hints from API data.

    Args:
        openapi_schema: OpenAPI schema dict
        service_order_statuses: List from /api/service-orders/service-order-statuses/
        service_order_priorities: List from /api/service-orders/service-order-priorities/

    Returns:
        dict of resource name → ResourceHint
    """
    hints: dict[str, ResourceHint] = {}

    # Extract enums from OpenAPI schema
    components = openapi_schema.get("components", {}).get("schemas", {})
    for schema_name, schema_def in components.items():
        if "properties" not in schema_def:
            continue

        resource = _schema_to_resource(schema_name)
        hint = hints.get(resource, ResourceHint())

        for field_name, field_def in schema_def.get("properties", {}).items():
            if "enum" in field_def:
                if field_name == "status":
                    hint.status_field = "status"
                    hint.status_values = field_def["enum"]
                else:
                    hint.enum_fields[field_name] = field_def["enum"]

        if hint.status_field or hint.enum_fields:
            hints[resource] = hint

    # Add service order statuses from API (DB-driven, can change)
    if service_order_statuses:
        so_hint = hints.get("service-orders", ResourceHint())
        so_hint.status_field = "status"
        so_hint.status_values = [s.get("name") or s.get("status") for s in service_order_statuses if s]
        hints["service-orders"] = so_hint

    # Add priorities
    if service_order_priorities:
        so_hint = hints.get("service-orders", ResourceHint())
        so_hint.enum_fields["priority"] = [p.get("name") or p.get("priority") for p in service_order_priorities if p]
        hints["service-orders"] = so_hint

    return hints


def build_status_synonyms(
    service_order_statuses: list[dict],
) -> dict[str, str]:
    """
    Helper for parent module to build status_synonyms.

    Returns common natural language → actual value mappings.
    """
    synonyms: dict[str, str] = {}

    # Common invoice synonyms (from model choices - stable)
    synonyms.update({
        "draft": "DRAFT",
        "pending": "DRAFT",
        "sent": "SENT",
        "unpaid": "SENT",
        "paid": "PAID",
        "overdue": "OVERDUE",
        "void": "VOID",
        "cancelled": "VOID",
    })

    # Service order status synonyms (from API data)
    for status in service_order_statuses:
        name = status.get("name") or status.get("status")
        if name:
            # Add lowercase version
            synonyms[name.lower()] = name
            # Add common alternatives
            if name.lower() == "inprogress" or name.lower() == "in progress":
                synonyms["in progress"] = name
                synonyms["ongoing"] = name
            if name.lower() == "completed":
                synonyms["done"] = name
                synonyms["finished"] = name
            if name.lower() == "new":
                synonyms["open"] = name

    return synonyms


def apply_filter_guidance(
    hints: dict[str, ResourceHint],
    guidance_map: dict[str, str],
) -> dict[str, ResourceHint]:
    """
    Inject per-resource filter_guidance from parent module (e.g. KQSPL).

    Example:
        hints = apply_filter_guidance(hints, {
            "service-orders": "For SO-### codes use list with search=SO-###, not path id",
        })
    """
    for resource, guidance in guidance_map.items():
        hint = hints.get(resource, ResourceHint())
        hint.filter_guidance = guidance
        hints[resource] = hint
    return hints


def build_query_examples_suffix(
    examples: list[Union[dict[str, Any], QueryExample]],
) -> str:
    """
    Format parent-provided query examples for the system prompt.

    Accepts QueryExample dataclass or dict with keys:
    query, tools/tool, resource, filters, sort, limit, question_type, notes
    """
    if not examples:
        return ""

    lines = ["Example queries and expected tools:"]
    for raw in examples:
        ex = raw if isinstance(raw, QueryExample) else _example_from_dict(raw)
        if not ex.query:
            continue
        target = ex.tools or ex.resource or "appropriate list/detail tool"
        line = f'- "{ex.query}" → {target}'
        extras: list[str] = []
        if ex.question_type:
            extras.append(f"type={ex.question_type}")
        if ex.filters:
            extras.append(f"filters={ex.filters}")
        if ex.sort:
            extras.append(f"sort={ex.sort}")
        if ex.limit is not None:
            extras.append(f"limit={ex.limit}")
        if ex.notes:
            extras.append(ex.notes)
        if extras:
            line += f" ({'; '.join(extras)})"
        lines.append(line)
    return "\n".join(lines)


def _example_from_dict(raw: dict[str, Any]) -> QueryExample:
    return QueryExample(
        query=raw.get("query", ""),
        resource=raw.get("resource", ""),
        tools=raw.get("tools", raw.get("tool", "")),
        filters=raw.get("filters"),
        sort=raw.get("sort"),
        limit=raw.get("limit"),
        question_type=raw.get("question_type"),
        notes=raw.get("notes"),
    )


def _schema_to_resource(schema_name: str) -> str:
    """Convert schema name to resource name."""
    import re
    name = schema_name.replace("Serializer", "").replace("Request", "").replace("Response", "")
    name = re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()
    if not name.endswith("s") and not name.endswith("data"):
        name += "s"
    return name
