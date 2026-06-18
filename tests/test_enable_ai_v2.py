"""
Tests for enable_ai_v2 module.
"""

from enable_ai_v2 import (
    Config,
    ToolCall,
    Endpoint,
    Response,
    ConfidenceScore,
)
from enable_ai_v2.tool_converter import (
    convert_spec_to_tools,
    extract_endpoints,
    endpoint_to_tool,
)
from enable_ai_v2.query_cache import QueryCache
from enable_ai_v2.validator import Validator
from enable_ai_v2.formatter import ResponseFormatter
from enable_ai_v2.tool_filter import (
    DEFAULT_EXCLUDE_PATTERNS,
    build_filter_query,
    extract_business_codes,
    extract_resource_ids,
    filter_openapi_spec,
    filter_tools_for_query,
    score_tool_for_query,
    get_max_tools_for_provider,
    has_business_code,
    is_count_query,
)
from enable_ai_v2.progress import ProgressStage
from enable_ai_v2.conversation_store import InMemoryConversationStore
from enable_ai_v2.types import QueryTrace
from enable_ai_v2.query_intent import (
    QueryIntent,
    classify_query_intent,
    needs_follow_up_round,
)
from datetime import datetime
from unittest.mock import MagicMock


# Sample OpenAPI spec for testing
SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "get": {
                "operationId": "list_users",
                "summary": "List all users",
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["active", "inactive"]},
                    },
                    {
                        "name": "page",
                        "in": "query",
                        "schema": {"type": "integer"},
                    },
                ],
                "responses": {"200": {"description": "OK"}},
            },
        },
        "/users/{id}": {
            "get": {
                "operationId": "get_user",
                "summary": "Get a user by ID",
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    },
                ],
                "responses": {"200": {"description": "OK"}},
            },
        },
        "/orders": {
            "post": {
                "operationId": "create_order",
                "summary": "Create a new order",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "product_id": {"type": "integer"},
                                    "quantity": {"type": "integer"},
                                },
                                "required": ["product_id", "quantity"],
                            }
                        }
                    },
                },
                "responses": {"201": {"description": "Created"}},
            },
        },
    },
}


class TestToolConverter:
    """Tests for OpenAPI to tool conversion."""

    def test_extract_endpoints(self):
        endpoints = extract_endpoints(SAMPLE_SPEC)
        assert len(endpoints) == 3

        names = {ep.tool_name for ep in endpoints}
        assert "list_users" in names
        assert "get_user" in names
        assert "create_order" in names

    def test_endpoint_to_tool(self):
        endpoints = extract_endpoints(SAMPLE_SPEC)
        get_user = next(ep for ep in endpoints if ep.tool_name == "get_user")

        tool = endpoint_to_tool(get_user, SAMPLE_SPEC)

        assert tool["name"] == "get_user"
        assert "id" in tool["input_schema"]["properties"]
        assert "id" in tool["input_schema"]["required"]

    def test_convert_spec_to_tools(self):
        tools, endpoints = convert_spec_to_tools(SAMPLE_SPEC)

        assert len(tools) == 3
        assert len(endpoints) == 3

        # Check tool structure
        list_users_tool = next(t for t in tools if t["name"] == "list_users")
        assert "input_schema" in list_users_tool
        assert list_users_tool["input_schema"]["type"] == "object"

    def test_tool_with_request_body(self):
        tools, _ = convert_spec_to_tools(SAMPLE_SPEC)
        create_order = next(t for t in tools if t["name"] == "create_order")

        props = create_order["input_schema"]["properties"]
        assert "product_id" in props
        assert "quantity" in props

    def test_enum_in_parameters(self):
        tools, _ = convert_spec_to_tools(SAMPLE_SPEC)
        list_users = next(t for t in tools if t["name"] == "list_users")

        status_prop = list_users["input_schema"]["properties"]["status"]
        assert "enum" in status_prop
        assert status_prop["enum"] == ["active", "inactive"]


class TestQueryCache:
    """Tests for query caching."""

    def test_cache_put_and_get(self):
        cache = QueryCache(max_size=100, ttl_seconds=3600)

        tool_calls = [ToolCall(id="1", name="list_users", arguments={"status": "active"})]
        cache.put("show active users", tool_calls, 0.95)

        result = cache.get("show active users")
        assert result is not None
        assert result.tool_calls[0].name == "list_users"

    def test_cache_miss(self):
        cache = QueryCache()
        result = cache.get("something not in cache")
        assert result is None

    def test_cache_normalization(self):
        cache = QueryCache()
        tool_calls = [ToolCall(id="1", name="list_users", arguments={})]
        cache.put("Show Active Users!", tool_calls, 0.9)

        # Should match despite case and punctuation differences
        result = cache.get("show active users")
        assert result is not None

    def test_pattern_matching(self):
        cache = QueryCache()

        # Cache a query with an ID
        tool_calls = [ToolCall(id="1", name="get_user", arguments={"id": "12345"})]
        cache.put("show user 12345", tool_calls, 0.95)

        # Query with different ID should match pattern
        result = cache.get("show user 67890")
        assert result is not None
        # The ID should be substituted
        assert result.tool_calls[0].arguments["id"] == "67890"

    def test_cache_clear(self):
        cache = QueryCache()
        tool_calls = [ToolCall(id="1", name="list_users", arguments={})]
        cache.put("show users", tool_calls, 0.9)

        cache.clear()
        assert cache.get("show users") is None

    def test_cache_stats(self):
        cache = QueryCache()
        tool_calls = [ToolCall(id="1", name="list_users", arguments={})]
        cache.put("show users", tool_calls, 0.9)

        stats = cache.stats()
        assert stats["exact_entries"] == 1


class TestValidator:
    """Tests for tool call validation."""

    def test_valid_tool_call(self):
        _, endpoints = convert_spec_to_tools(SAMPLE_SPEC)
        validator = Validator(endpoints)

        tool_call = ToolCall(id="1", name="get_user", arguments={"id": 123})
        result = validator.validate(tool_call, "show user 123")

        assert result.valid
        assert len(result.errors) == 0

    def test_missing_required_param(self):
        _, endpoints = convert_spec_to_tools(SAMPLE_SPEC)
        validator = Validator(endpoints)

        tool_call = ToolCall(id="1", name="get_user", arguments={})
        result = validator.validate(tool_call, "show user")

        assert not result.valid
        assert any("id" in err.lower() for err in result.errors)

    def test_unknown_tool(self):
        _, endpoints = convert_spec_to_tools(SAMPLE_SPEC)
        validator = Validator(endpoints)

        tool_call = ToolCall(id="1", name="unknown_tool", arguments={})
        result = validator.validate(tool_call, "do something")

        assert not result.valid
        assert "Unknown tool" in result.errors[0]

    def test_enum_validation(self):
        _, endpoints = convert_spec_to_tools(SAMPLE_SPEC)
        validator = Validator(endpoints)

        # Valid enum value
        tool_call = ToolCall(
            id="1", name="list_users", arguments={"status": "active"}
        )
        result = validator.validate(tool_call, "show active users")
        assert result.valid

        # Invalid enum value
        tool_call = ToolCall(
            id="1", name="list_users", arguments={"status": "unknown"}
        )
        result = validator.validate(tool_call, "show unknown users")
        assert not result.valid

    def test_confidence_calculation(self):
        _, endpoints = convert_spec_to_tools(SAMPLE_SPEC)
        validator = Validator(endpoints)

        tool_call = ToolCall(id="1", name="get_user", arguments={"id": 123})
        result = validator.validate(tool_call, "show user 123")

        assert result.confidence > 0
        assert result.confidence <= 1.0


class TestResponseFormatter:
    """Tests for response formatting."""

    def test_format_error(self):
        formatter = ResponseFormatter()
        msg = formatter.format_error("Connection failed", "show users")
        assert "Connection failed" in msg

    def test_format_empty_list(self):
        formatter = ResponseFormatter()
        msg = formatter.format_success_simple([], "list_users")
        assert "No results" in msg

    def test_format_single_item(self):
        formatter = ResponseFormatter()
        data = {"id": 1, "name": "John Doe", "email": "john@example.com"}
        msg = formatter.format_success_simple(data, "get_user")
        assert "John Doe" in msg

    def test_format_list(self):
        formatter = ResponseFormatter()
        data = [
            {"id": 1, "name": "John"},
            {"id": 2, "name": "Jane"},
            {"id": 3, "name": "Bob"},
        ]
        msg = formatter.format_success_simple(data, "list_users")
        assert "3 results" in msg

    def test_format_api_error_401(self):
        formatter = ResponseFormatter()
        msg = formatter.format_api_error("get_user", "Unauthorized", 401)
        assert "log in" in msg.lower()

    def test_format_api_error_404(self):
        formatter = ResponseFormatter()
        msg = formatter.format_api_error("get_user", "Not found", 404)
        assert "find" in msg.lower() or "exist" in msg.lower()


class TestConfig:
    """Tests for configuration."""

    def test_default_config(self):
        config = Config()
        assert config.temperature == 0.0  # Deterministic
        assert config.cache_enabled is True

    def test_config_with_trace(self):
        config = Config(include_trace=True)
        assert config.include_trace is True

    def test_config_defaults(self):
        config = Config()
        assert config.include_trace is False
        assert config.cache_enabled is True
        assert config.model == "gpt-4o"
        assert config.llm_provider == "openai"
        assert config.tool_filter_enabled is True
        assert config.filter_tools_for_non_admin is True
        assert config.tool_exclude_patterns is None
        assert config.allowed_http_methods is None


class TestTypes:
    """Tests for type classes."""

    def test_confidence_score_high(self):
        score = ConfidenceScore.high()
        assert score.overall == 1.0
        assert score.cache_boost == 0.1

    def test_response_with_trace(self):
        from datetime import datetime
        from enable_ai_v2.types import QueryTrace

        trace = QueryTrace(
            query="show users",
            timestamp=datetime.now(),
        )
        response = Response(
            message="Found 5 users",
            data=[{"id": 1}],
            success=True,
            trace=trace,
        )

        assert response.success
        assert response.trace is not None
        assert response.trace.query == "show users"

    def test_tool_call_dataclass(self):
        tc = ToolCall(id="1", name="test", arguments={"a": 1})
        assert tc.id == "1"
        assert tc.arguments["a"] == 1

    def test_endpoint_tool_name(self):
        ep = Endpoint(
            path="/users/{id}",
            method="GET",
            operation_id="",
            summary="Get user",
            description="",
            parameters=[],
        )
        # Without operation_id, generates from path
        assert ep.tool_name == "get_users"

        ep2 = Endpoint(
            path="/users/{id}",
            method="GET",
            operation_id="get_user_by_id",
            summary="Get user",
            description="",
            parameters=[],
        )
        # With operation_id, uses it directly
        assert ep2.tool_name == "get_user_by_id"


class TestToolFilter:
    """Tests for shared tool filtering (all LLM providers)."""

    FILTER_SPEC = {
        "openapi": "3.0.0",
        "paths": {
            "/service-orders/": {
                "get": {
                    "operationId": "list_service_orders",
                    "summary": "List service orders",
                },
            },
            "/service-orders/bulk-export/": {
                "post": {
                    "operationId": "bulk_export_service_orders",
                    "summary": "Bulk export service orders",
                },
            },
            "/admin/users/": {
                "get": {
                    "operationId": "admin_list_users",
                    "summary": "Admin list users",
                },
            },
            "/invoices/": {
                "post": {
                    "operationId": "create_invoice",
                    "summary": "Create invoice",
                },
            },
        },
    }

    SAMPLE_TOOLS = [
        {"name": "list_service_orders", "description": "List service orders"},
        {"name": "get_service_order", "description": "Get a service order by ID"},
        {"name": "list_invoices", "description": "List invoices"},
        {"name": "create_invoice", "description": "Create a new invoice"},
        {"name": "bulk_export_orders", "description": "Bulk export orders"},
    ]

    def test_filter_openapi_spec_exclude_patterns(self):
        filtered = filter_openapi_spec(
            self.FILTER_SPEC,
            exclude_patterns=[r"bulk", r"admin"],
            is_admin=True,
        )
        paths = filtered["paths"]
        assert "/service-orders/" in paths
        assert "/service-orders/bulk-export/" not in paths
        assert "/admin/users/" not in paths

    def test_filter_openapi_spec_allowed_methods(self):
        filtered = filter_openapi_spec(
            self.FILTER_SPEC,
            allowed_methods=["GET"],
            is_admin=True,
        )
        paths = filtered["paths"]
        assert "get" in paths["/service-orders/"]
        assert "/invoices/" not in paths

    def test_filter_openapi_spec_non_admin_defaults(self):
        filtered = filter_openapi_spec(
            self.FILTER_SPEC,
            is_admin=False,
        )
        paths = filtered["paths"]
        assert "/service-orders/bulk-export/" not in paths
        assert "/admin/users/" not in paths

    def test_score_tool_for_query_resource_match(self):
        tool = {"name": "list_service_orders", "description": "List service orders"}
        score = score_tool_for_query("show my service orders", tool)
        assert score > 0

    def test_score_tool_action_mapping(self):
        list_tool = {"name": "list_invoices", "description": "List invoices"}
        create_tool = {"name": "create_invoice", "description": "Create invoice"}

        assert score_tool_for_query("list all invoices", list_tool) > score_tool_for_query(
            "list all invoices", create_tool
        )
        assert score_tool_for_query("create a new invoice", create_tool) > score_tool_for_query(
            "create a new invoice", list_tool
        )

    def test_filter_tools_for_query_limits_and_prioritizes(self):
        tools = [
            {"name": f"tool_{i}", "description": f"Generic tool {i}"}
            for i in range(150)
        ]
        tools.append({"name": "list_service_orders", "description": "List service orders"})

        filtered = filter_tools_for_query(
            "show service orders",
            tools,
            max_tools=128,
        )
        assert len(filtered) == 128
        assert any(t["name"] == "list_service_orders" for t in filtered)

    def test_get_max_tools_for_provider(self):
        assert get_max_tools_for_provider("openai") == 128
        assert get_max_tools_for_provider("anthropic") == 200
        assert get_max_tools_for_provider("openai", override=50) == 50

    def test_default_exclude_patterns_defined(self):
        assert "bulk" in DEFAULT_EXCLUDE_PATTERNS
        assert "admin" in DEFAULT_EXCLUDE_PATTERNS

    def test_extract_resource_ids(self):
        ids = extract_resource_ids("show SO-159 details")
        assert "so-159" in ids or "so159" in ids
        assert "159" not in ids

    def test_extract_resource_ids_no_bare_number(self):
        assert "159" not in extract_resource_ids("show SO-159")
        assert extract_business_codes("show SO-159") != []

    def test_build_filter_query_from_history(self):
        history = [
            {"role": "user", "content": "Tell me about SO-159"},
            {"role": "assistant", "content": "SO-159 is in progress."},
        ]
        enriched = build_filter_query("who is the technician?", history)
        assert "159" in enriched or "so-159" in enriched

    def test_build_filter_query_bare_code_follow_up(self):
        history = [
            {"role": "user", "content": "observations for my last report"},
            {"role": "assistant", "content": "Which service order? Please provide the SO code."},
        ]
        enriched = build_filter_query("SO-169", history)
        assert "observations" in enriched.lower()
        assert "so-169" in enriched.lower() or "so169" in enriched.lower()

    def test_so_code_prefers_search_not_retrieve(self):
        retrieve = {"name": "service_orders_retrieve", "description": "Get service order by id"}
        list_tool = {"name": "service_orders_list", "description": "List service orders with search"}
        query = "show details for SO-159"
        assert score_tool_for_query(query, list_tool) > score_tool_for_query(query, retrieve)

    def test_is_count_query(self):
        assert is_count_query("how many open service orders do I have")
        assert not is_count_query("show open service orders")


class TestFormatterMulti:
    """Tests for count and multi-API formatting."""

    def test_format_count_from_paginated_response(self):
        formatter = ResponseFormatter()
        data = {"count": 5, "results": [{"id": 1, "name": "SO-1"}]}
        msg = formatter.format_success_simple(
            data, "service_orders_list", query="how many open orders"
        )
        assert "5" in msg
        assert "You have" in msg
        assert "service orders" in msg

    def test_resource_label_from_tool_name_not_query(self):
        formatter = ResponseFormatter()
        assert formatter._resource_label("service_orders_list") == "service orders"
        assert formatter._resource_label("list_invoices") == "invoices"

    def test_count_noun_pluralization(self):
        formatter = ResponseFormatter()
        assert formatter._count_noun("service orders", 1) == "service order"
        assert formatter._count_noun("service orders", 5) == "service orders"
        assert formatter._count_noun("invoice", 2) == "invoices"

    def test_format_count_singular(self):
        formatter = ResponseFormatter()
        data = {"count": 1, "results": [{"id": 1, "name": "SO-1"}]}
        msg = formatter.format_success_simple(
            data, "service_orders_list", query="how many open orders"
        )
        assert msg == "You have 1 service order."

    def test_format_multi_success(self):
        formatter = ResponseFormatter()
        tool_calls = [
            ToolCall(id="1", name="list_service_orders", arguments={}),
            ToolCall(id="2", name="list_invoices", arguments={}),
        ]
        results = [
            {"data": {"count": 2, "results": [{"name": "SO-1"}, {"name": "SO-2"}]}},
            {"data": {"count": 1, "results": [{"name": "INV-1"}]}},
        ]
        msg = formatter.format_multi_success(tool_calls, results)
        assert "List Service Orders" in msg
        assert "List Invoices" in msg


class TestNamingAndTrace:
    """Tests for 1.0.8+ provider-agnostic naming."""

    def test_query_trace_llm_reasoning_key(self):
        trace = QueryTrace(
            query="test",
            timestamp=datetime.now(),
            reasoning="Need more info",
            llm_provider="openai",
        )
        d = trace.to_dict()
        assert "llm_reasoning" in d
        assert d["llm_reasoning"] == "Need more info"
        assert "claude_reasoning" not in d
        assert d["llm_provider"] == "openai"
        assert "api_calls" in d

    def test_progress_calling_llm(self):
        assert ProgressStage.CALLING_LLM.value == "calling_llm"
        assert ProgressStage.CALLING_CLAUDE.value == "calling_llm"

    def test_get_llm_messages(self):
        store = InMemoryConversationStore()
        store.add_message("s1", "user", "hello")
        store.add_message("s1", "assistant", "hi")
        msgs = store.get_llm_messages("s1")
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert store.get_claude_messages("s1") == msgs


class TestQueryIntent:
    def test_classify_count(self):
        assert classify_query_intent("how many open service orders") == QueryIntent.COUNT

    def test_classify_single_detail(self):
        assert classify_query_intent("status of my latest service order") == QueryIntent.SINGLE_DETAIL
        assert classify_query_intent("show SO-159 details") == QueryIntent.SINGLE_DETAIL

    def test_classify_multi_step(self):
        assert classify_query_intent("observations for my latest report") == QueryIntent.MULTI_STEP
        assert classify_query_intent("flash report for latest SO") == QueryIntent.MULTI_STEP

    def test_most_recent_not_aggregate(self):
        """'most recent' should trigger SINGLE_DETAIL, not AGGREGATE."""
        # "most" alone triggers aggregate, but "most recent" is excluded
        assert classify_query_intent("flash report for my most recent service order") == QueryIntent.MULTI_STEP
        assert classify_query_intent("show most recent report") != QueryIntent.AGGREGATE
        # But "who has most" should still be aggregate
        assert classify_query_intent("who has the most service orders") == QueryIntent.AGGREGATE

    def test_needs_follow_up_when_embedded_missing(self):
        tc = ToolCall(id="1", name="service_orders_list", arguments={"page_size": 1})
        results = [{"data": {"count": 1, "results": [{"id": 42, "name": "SO-1"}]}}]
        assert needs_follow_up_round(
            "observations for my latest report",
            QueryIntent.MULTI_STEP,
            [tc],
            results,
        )


class TestFormatterPageSizeOne:
    def test_page_size_one_formats_row_not_count(self):
        formatter = ResponseFormatter()
        data = {
            "count": 16,
            "results": [{"id": 1, "name": "SO-159", "status": "open"}],
        }
        msg = formatter.format_success_simple(
            data,
            "service_orders_list",
            query="status of my latest service order",
            tool_args={"page_size": 1},
            intent=QueryIntent.SINGLE_DETAIL,
        )
        assert "You have" not in msg
        assert "16" not in msg
        assert "SO-159" in msg or "open" in msg


class TestToolFilterResourceAffinity:
    def test_service_orders_beats_statuses_for_so_query(self):
        statuses = {
            "name": "service_order_statuses_list",
            "description": "List service order statuses",
        }
        orders = {
            "name": "service_orders_list",
            "description": "List service orders with search",
        }
        query = "technician assigned to SO-159"
        assert score_tool_for_query(query, orders) > score_tool_for_query(query, statuses)


class TestValidatorExtended:
    def test_statuses_tool_rejected_for_so_query(self):
        endpoints = [
            Endpoint(
                path="/service-order-statuses/",
                method="GET",
                operation_id="service_order_statuses_list",
                summary="List statuses",
                description="",
                parameters=[],
            )
        ]
        validator = Validator(endpoints)
        tc = ToolCall(
            id="1",
            name="service_order_statuses_list",
            arguments={"search": "SO-159"},
        )
        result = validator.validate(tc, "technician on SO-159")
        assert result.needs_clarification
        assert "service-orders" in (result.clarification_question or "").lower()

    def test_technician_zero_rejected(self):
        endpoints = [
            Endpoint(
                path="/service-orders/",
                method="GET",
                operation_id="service_orders_list",
                summary="List service orders",
                description="",
                parameters=[],
            )
        ]
        validator = Validator(endpoints)
        tc = ToolCall(id="1", name="service_orders_list", arguments={"technician": 0})
        result = validator.validate(tc, "my service orders")
        assert result.needs_clarification
        assert "technician" in (result.clarification_question or "").lower()


class TestAgentLoop:
    AGENT_SPEC = {
        "openapi": "3.0.0",
        "paths": {
            "/service-orders/": {
                "get": {
                    "operationId": "service_orders_list",
                    "summary": "List service orders",
                    "parameters": [
                        {"name": "page_size", "in": "query", "schema": {"type": "integer"}},
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
            "/service-orders/{id}/": {
                "get": {
                    "operationId": "service_orders_retrieve",
                    "summary": "Get service order",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        },
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
        },
    }

    def test_multi_step_loop_executes_follow_up(self):
        from enable_ai_v2 import Orchestrator

        config = Config(
            openapi_schema=self.AGENT_SPEC,
            base_url="http://test",
            cache_enabled=False,
            max_tool_rounds=3,
            include_trace=True,
        )
        orch = Orchestrator(config=config, api_key="fake")

        round1 = [
            ToolCall(id="1", name="service_orders_list", arguments={"page_size": 1})
        ]
        round2 = [
            ToolCall(id="2", name="service_orders_retrieve", arguments={"id": 42})
        ]
        llm_round = [0]

        def mock_process_query(query, tools, system_prompt, conversation_history=None):
            llm_round[0] += 1
            if llm_round[0] == 1:
                return round1, "list latest", "tool_use"
            return round2, "get detail", "tool_use"

        orch._llm.process_query = mock_process_query

        def mock_execute(tc, endpoint):
            if tc.name == "service_orders_list":
                return (
                    {"data": {"count": 19, "results": [{"id": 42, "name": "SO-1"}]}},
                    MagicMock(success=True),
                )
            return (
                {
                    "data": {
                        "id": 42,
                        "observations": [{"text": "Crack found"}],
                    }
                },
                MagicMock(success=True),
            )

        orch._api.execute = mock_execute
        orch._llm.generate_response = MagicMock(return_value="Observations: Crack found")

        response = orch.process("observations for my latest report")

        assert llm_round[0] == 1
        assert len(response.trace.tool_calls) == 2
        assert response.trace.tool_calls[1].name == "service_orders_retrieve"


class TestDateRange:
    def test_this_month(self):
        from enable_ai_v2.date_range import extract_date_range
        from datetime import datetime

        ref = datetime(2026, 6, 14)
        filters = extract_date_range("service orders completed this month", reference=ref)
        assert filters["updated_at__gte"] == "2026-06-01"
        assert filters["updated_at__lte"] == "2026-06-14"

    def test_this_week_uses_created_at(self):
        from enable_ai_v2.date_range import extract_date_range
        from datetime import datetime

        ref = datetime(2026, 6, 14)  # Sunday
        filters = extract_date_range("orders created this week", reference=ref)
        assert "created_at__gte" in filters
        assert filters["created_at__lte"] == "2026-06-14"

    def test_inject_date_filters_on_list_tool(self):
        from enable_ai_v2.date_range import inject_date_filters
        from datetime import datetime

        tc = ToolCall(id="1", name="service_orders_list", arguments={"status": "open"})
        updated = inject_date_filters(
            [tc],
            "how many orders completed this month",
        )
        assert "updated_at__gte" in updated[0].arguments
        assert updated[0].arguments["status"] == "open"

    def test_last_month(self):
        from enable_ai_v2.date_range import extract_date_range
        from datetime import datetime

        ref = datetime(2026, 6, 14)
        filters = extract_date_range("SOs from last month", reference=ref)
        assert filters["created_at__gte"] == "2026-05-01"
        assert filters["created_at__lte"] == "2026-05-31"

    def test_next_n_days(self):
        from enable_ai_v2.date_range import extract_date_range
        from datetime import datetime

        ref = datetime(2026, 6, 14)
        filters = extract_date_range("invoices due next 7 days", reference=ref)
        assert filters["due_date__gte"] == "2026-06-14"
        assert filters["due_date__lte"] == "2026-06-21"

    def test_greater_than_n_days(self):
        from enable_ai_v2.date_range import extract_date_range
        from datetime import datetime

        ref = datetime(2026, 6, 14)
        filters = extract_date_range("InProgress > 7 days", reference=ref)
        assert "created_at__lte" in filters
        assert filters["created_at__lte"] == "2026-06-07"


class TestReadOnlyMutationGuard:
    def test_partial_update_rejected_on_read_query(self):
        endpoints = [
            Endpoint(
                path="/invoicing/invoices/{id}/",
                method="PATCH",
                operation_id="invoices_partial_update",
                summary="Partial update invoice",
                description="",
                parameters=[
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
            )
        ]
        validator = Validator(endpoints)
        tc = ToolCall(id="1", name="invoices_partial_update", arguments={"id": 1})
        result = validator.validate(tc, "show all invoices with partial payments")
        assert result.needs_clarification

    def test_mutation_tool_penalized_in_scoring(self):
        from enable_ai_v2.tool_filter import is_read_only_query, score_tool_for_query

        update_tool = {"name": "invoices_partial_update", "description": "Update invoice"}
        list_tool = {"name": "invoices_list", "description": "List invoices"}
        query = "show all invoices with partial payments"
        assert is_read_only_query(query)
        assert score_tool_for_query(query, list_tool) > score_tool_for_query(query, update_tool)


class TestEmbeddedAutoRetrieve:
    AGENT_SPEC = {
        "openapi": "3.0.0",
        "paths": {
            "/details-reports/": {
                "get": {
                    "operationId": "details_reports_list",
                    "summary": "List details reports",
                    "responses": {"200": {"description": "OK"}},
                },
            },
            "/details-reports/{id}/": {
                "get": {
                    "operationId": "details_reports_retrieve",
                    "summary": "Get details report",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        },
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
            "/details-reports/by_service_id/": {
                "get": {
                    "operationId": "details_reports_by_service_id_list",
                    "summary": "By service id",
                    "responses": {"200": {"description": "OK"}},
                },
            },
        },
    }

    def test_auto_retrieve_after_list_for_observations(self):
        from enable_ai_v2 import Orchestrator

        config = Config(
            openapi_schema=self.AGENT_SPEC,
            base_url="http://test",
            cache_enabled=False,
            max_tool_rounds=3,
            include_trace=True,
        )
        orch = Orchestrator(config=config, api_key="fake")

        round1 = [
            ToolCall(id="1", name="details_reports_list", arguments={"page_size": 1})
        ]
        llm_round = [0]

        def mock_process_query(query, tools, system_prompt, conversation_history=None):
            llm_round[0] += 1
            if llm_round[0] == 1:
                return round1, "list report", "tool_use"
            raise AssertionError("LLM should not run round 2 — auto retrieve expected")

        orch._llm.process_query = mock_process_query

        def mock_execute(tc, endpoint):
            if tc.name == "details_reports_list":
                return (
                    {"data": {"count": 19, "results": [{"id": 99, "name": "DR-1"}]}},
                    MagicMock(success=True),
                )
            assert tc.name == "details_reports_retrieve"
            assert tc.arguments["id"] == 99
            return (
                {"data": {"id": 99, "observations": [{"text": "OK"}]}},
                MagicMock(success=True),
            )

        orch._llm.generate_response = MagicMock(return_value="fallback")
        orch._api.execute = mock_execute

        response = orch.process("observations for my latest details report")

        assert llm_round[0] == 1
        assert len(response.trace.tool_calls) == 2
        assert response.trace.tool_calls[1].name == "details_reports_retrieve"


class TestToolArgs:
    def test_flatten_nested_filters(self):
        from enable_ai_v2.tool_args import flatten_tool_arguments

        args = {"filters": {"filters": {"status__name": "Completed"}}}
        flat = flatten_tool_arguments(args)
        assert flat == {"status__name": "Completed"}

    def test_flatten_single_filters_wrapper(self):
        from enable_ai_v2.tool_args import flatten_tool_arguments

        args = {"filters": {"status": "open"}}
        flat = flatten_tool_arguments(args)
        assert flat == {"status": "open"}

    def test_enforce_count_removes_page_size_one(self):
        from enable_ai_v2.tool_args import enforce_count_tool_args

        tc = ToolCall(
            id="1",
            name="service_orders_list",
            arguments={"page_size": 1, "ordering": "-created_at"},
        )
        updated = enforce_count_tool_args([tc], QueryIntent.COUNT, retain_page_size=20)
        assert updated[0].arguments["page_size"] == 20
        assert updated[0].arguments.get("ordering") == "-created_at"


class TestFormatterCountIntent:
    def test_count_intent_uses_pagination_count_not_first_row(self):
        formatter = ResponseFormatter()
        data = {
            "count": 13,
            "results": [{"id": 1, "display_name": "SO-5", "description": "test"}],
        }
        msg = formatter.format_success_simple(
            data,
            "service_orders_list",
            query="how many new service orders came in this week",
            tool_args={"page_size": 1, "ordering": "-created_at"},
            intent=QueryIntent.COUNT,
        )
        assert msg == "You have 13 service orders."
        assert "SO-5" not in msg

    def test_list_labels_prefer_display_name(self):
        formatter = ResponseFormatter()
        data = [
            {"id": 19, "description": "TPI", "display_name": "SO-159"},
            {"id": 18, "description": "Third party", "display_name": "SO-160"},
        ]
        msg = formatter.format_success_simple(data, "service_orders_list")
        assert "SO-159" in msg
        assert "SO-160" in msg
        assert "TPI" not in msg or "SO-159" in msg

    def test_suggestions_deduped_and_contextual(self):
        formatter = ResponseFormatter()
        tool_calls = [
            ToolCall(id="1", name="service_orders_list", arguments={}),
        ]
        results = [
            {
                "data": {
                    "count": 13,
                    "results": [
                        {"id": 1, "display_name": "SO-1"},
                        {"id": 2, "display_name": "SO-2"},
                    ],
                }
            }
        ]
        suggestions = formatter.build_suggestions(
            "how many open service orders",
            tool_calls,
            results,
        )
        assert len(suggestions) <= 3
        assert len(suggestions) == len(set(s.lower() for s in suggestions))
        assert not all("Show more details about this" in s for s in suggestions)


class TestListCache:
    def test_build_list_cache_from_paginated(self):
        from enable_ai_v2.tool_args import build_list_cache

        tc = ToolCall(id="1", name="service_orders_list", arguments={})
        results = [{"data": {"count": 2, "results": [{"id": 1}, {"id": 2}]}}]
        cache = build_list_cache([tc], results)
        assert len(cache) == 2
        assert cache[0]["id"] == 1


class Test130Features:
    def test_scoped_flash_report_list(self):
        from enable_ai_v2.multi_step import build_scoped_flash_report_list
        from enable_ai_v2.types import Endpoint

        endpoints = [
            Endpoint(
                path="/flash-reports/",
                method="GET",
                operation_id="flash_reports_list",
                summary="List flash reports",
                description="",
                parameters=[],
            )
        ]
        prior = {"data": {"results": [{"id": 5, "display_name": "SO-100"}]}}
        tc = build_scoped_flash_report_list(
            "flash report for my most recent service order",
            "service_orders_list",
            prior,
            endpoints,
        )
        assert tc is not None
        assert tc.name == "flash_reports_list"
        assert tc.arguments["service_order"] == 5
        assert tc.arguments["search"] == "SO-100"

    def test_referential_tool_calls_with_ids(self):
        from enable_ai_v2.follow_up import build_referential_tool_calls
        from enable_ai_v2.types import Endpoint

        endpoints = [
            Endpoint(
                path="/service-orders/",
                method="GET",
                operation_id="service_orders_list",
                summary="List SOs",
                description="",
                parameters=[],
            )
        ]
        cache = [
            {"id": 1, "display_name": "SO-1"},
            {"id": 2, "display_name": "SO-2"},
        ]
        calls = build_referential_tool_calls(
            "show first 2 of those above",
            cache,
            endpoints,
        )
        assert calls is not None
        assert calls[0].name == "service_orders_list"
        assert calls[0].arguments["id__in"] == [1, 2]

    def test_aggregate_technician_ranking(self):
        from enable_ai_v2.aggregate import format_aggregate_response

        results = [{
            "data": {
                "results": [
                    {"technician_name": "Rahul D", "id": 1},
                    {"technician_name": "Rahul D", "id": 2},
                    {"technician_name": "Jane", "id": 3},
                ]
            }
        }]
        msg = format_aggregate_response(results, "which technician has the most service orders")
        assert msg is not None
        assert "Rahul D" in msg
        assert "most" in msg.lower()

    def test_scheduling_rejects_transition_tool(self):
        endpoints = [
            Endpoint(
                path="/check-transition/",
                method="GET",
                operation_id="check_transition_impact",
                summary="Check transition",
                description="",
                parameters=[],
            )
        ]
        validator = Validator(endpoints)
        tc = ToolCall(id="1", name="check_transition_impact", arguments={})
        result = validator.validate(tc, "Has my service order been scheduled yet?")
        assert result.needs_clarification

    def test_duration_filter_extraction(self):
        from enable_ai_v2.date_range import extract_duration_filter
        from datetime import datetime

        filters = extract_duration_filter(
            "service orders InProgress for more than 7 days",
            reference=datetime(2026, 6, 14),
        )
        assert filters["updated_at__lte"] == "2026-06-07"

    def test_ar_dashboard_format(self):
        formatter = ResponseFormatter()
        data = {
            "results": [{
                "company_name": "Acme Corp",
                "outstanding_amount": 125000,
            }]
        }
        msg = formatter.format_success_simple(
            data,
            "invoicing_ar_dashboard_list",
            query="which company has the highest outstanding AR",
        )
        assert "Acme Corp" in msg
        assert "125000" in msg

    def test_ar_dashboard_avoids_generic_count_noun(self):
        formatter = ResponseFormatter()
        data = {
            "count": 1,
            "results": [{"customer": {"name": "Beta LLC"}, "total_outstanding": 9900}],
        }
        msg = formatter.format_success_simple(
            data,
            "invoicing_ar_dashboard_list",
            query="AR dashboard outstanding balances",
        )
        assert "ar dashboard" not in msg.lower() or "Beta LLC" in msg
        assert "9900" in msg or "Beta LLC" in msg

    def test_status_field_in_single_detail(self):
        formatter = ResponseFormatter()
        data = {"display_name": "SO-5", "status": "In Progress", "id": 5}
        msg = formatter.format_success_simple(
            data,
            "service_orders_list",
            query="what is the status of SO-5",
            tool_args={"page_size": 1},
        )
        assert "Status: In Progress" in msg

    def test_count_with_page_size_one(self):
        formatter = ResponseFormatter()
        data = {"count": 13, "results": [{"display_name": "SO-1", "id": 1}]}
        msg = formatter.format_success_simple(
            data,
            "service_orders_list",
            query="how many open service orders",
            tool_args={"page_size": 1},
        )
        assert msg == "You have 13 service orders."

    def test_count_with_sample_rows(self):
        """Count query with multiple results shows sample rows."""
        formatter = ResponseFormatter()
        data = {
            "count": 5,
            "results": [
                {"id": 1, "display_name": "SO-173", "status": "InProgress", "priority": "High"},
                {"id": 2, "display_name": "SO-168", "status": "Scheduled"},
                {"id": 3, "display_name": "SO-165", "status": "New"},
            ],
        }
        msg = formatter.format_success_simple(
            data,
            "service_orders_list",
            query="how many open service orders",
        )
        assert "You have 5 service orders." in msg
        assert "SO-173" in msg
        assert "InProgress" in msg

    def test_technician_availability_synthesis(self):
        from enable_ai_v2.aggregate import format_aggregate_response
        from enable_ai_v2.types import ToolCall

        users = [
            {"id": 11, "first_name": "Alice", "last_name": "A"},
            {"id": 12, "first_name": "Bob", "last_name": "B"},
            {"id": 13, "first_name": "Carol", "last_name": "C"},
        ]
        orders = [
            {"id": 1, "technician": {"id": 11, "display_name": "Alice A"}},
            {"id": 2, "technician_name": "Bob B"},
        ]
        tool_calls = [
            ToolCall(id="1", name="users_list", arguments={}),
            ToolCall(id="2", name="service_orders_list", arguments={}),
        ]
        results = [
            {"data": {"results": users}},
            {"data": {"results": orders}},
        ]
        msg = format_aggregate_response(
            results,
            "which technicians have no jobs assigned",
            tool_calls=tool_calls,
        )
        assert msg is not None
        assert "Carol C" in msg
        assert "Alice" not in msg or "no assigned" in msg.lower()

    def test_workload_second_step_builder(self):
        from enable_ai_v2.multi_step import build_workload_service_orders_list
        from enable_ai_v2.types import Endpoint, ToolCall

        endpoints = [
            Endpoint(
                path="/service-orders/",
                method="GET",
                operation_id="service_orders_list",
                summary="List SOs",
                description="",
                parameters=[],
            )
        ]
        prior = [ToolCall(id="1", name="users_list", arguments={})]
        tc = build_workload_service_orders_list(
            "which technicians have no jobs",
            prior,
            endpoints,
        )
        assert tc is not None
        assert tc.name == "service_orders_list"
        assert tc.arguments["page_size"] == 100

    def test_compound_query_intent(self):
        from enable_ai_v2.query_intent import classify_query_intent, QueryIntent, is_compound_query

        q = "show open service orders and overdue invoices"
        assert is_compound_query(q)
        assert classify_query_intent(q) == QueryIntent.AGGREGATE

    def test_compound_second_step_builder(self):
        from enable_ai_v2.multi_step import build_compound_second_list
        from enable_ai_v2.types import Endpoint, ToolCall

        endpoints = [
            Endpoint(
                path="/service-orders/",
                method="GET",
                operation_id="service_orders_list",
                summary="List SOs",
                description="",
                parameters=[],
            ),
            Endpoint(
                path="/invoices/",
                method="GET",
                operation_id="invoicing_invoices_list",
                summary="List invoices",
                description="",
                parameters=[],
            ),
        ]
        prior = [ToolCall(id="1", name="service_orders_list", arguments={})]
        tc = build_compound_second_list(
            "show open service orders and overdue invoices",
            prior,
            endpoints,
        )
        assert tc is not None
        assert "invoice" in tc.name.lower()

    def test_availability_formatter_fallback(self):
        formatter = ResponseFormatter()
        data = {
            "count": 3,
            "results": [
                {"id": 13, "first_name": "Carol", "last_name": "C"},
                {"id": 12, "first_name": "Bob", "last_name": "B"},
                {"id": 11, "first_name": "Alice", "last_name": "A"},
            ],
        }
        msg = formatter.format_success_simple(
            data,
            "users_list",
            query="which technicians have no jobs assigned",
        )
        assert "Found 3 results: 13" not in msg
        assert "Carol" in msg or "directory" in msg.lower()

    def test_compound_aggregate_format(self):
        from enable_ai_v2.aggregate import format_aggregate_response
        from enable_ai_v2.types import ToolCall

        tool_calls = [
            ToolCall(id="1", name="service_orders_list", arguments={}),
            ToolCall(id="2", name="invoicing_invoices_list", arguments={}),
        ]
        results = [
            {"data": {"results": [{"display_name": "SO-1"}, {"display_name": "SO-2"}]}},
            {"data": {"results": [{"invoice_number": "INV-9"}]}},
        ]
        msg = format_aggregate_response(
            results,
            "show open service orders and overdue invoices",
            tool_calls=tool_calls,
        )
        assert msg is not None
        assert "SO" in msg or "service" in msg.lower()
        assert "INV" in msg or "invoice" in msg.lower()


class TestContinuationMessage:
    def test_build_continuation_message_no_name_error(self):
        from enable_ai_v2 import Orchestrator

        config = Config(openapi_schema={"openapi": "3.0.0", "paths": {}}, base_url="http://test")
        orch = Orchestrator(config=config, api_key="fake")
        tool_calls = [
            ToolCall(id="1", name="flash_reports_list", arguments={"page_size": 1}),
        ]
        results = [{"data": {"count": 1, "results": [{"id": 7, "name": "FR-1"}]}}]
        msg = orch._build_continuation_message(
            "show flash report for latest service order",
            tool_calls,
            results,
        )
        assert "Original user request" in msg
        assert "flash_reports_list" in msg


class TestCountNounGrammar:
    def test_deduplicate_stuttered_label(self):
        formatter = ResponseFormatter()
        assert formatter._count_noun("service orders service orders", 1) == "service order"


class TestEnumNormalization:
    def test_fuzzy_enum_mapping(self):
        endpoints = [
            Endpoint(
                path="/reports/",
                method="GET",
                operation_id="reports_list",
                summary="List reports",
                description="",
                parameters=[
                    {
                        "name": "report_type",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["UT_WELD", "RT", "MT"]},
                    }
                ],
            )
        ]
        validator = Validator(endpoints)
        tc = ToolCall(id="1", name="reports_list", arguments={"report_type": "UT Weld"})
        normalized = validator.normalize_arguments(tc)
        assert normalized.arguments["report_type"] == "UT_WELD"


class TestPermissionHints:
    def test_403_customer_hint(self):
        formatter = ResponseFormatter()
        msg = formatter.format_api_error("users_list", "Forbidden", 403, role="Customer")
        assert "permission" in msg.lower()
        assert "service orders" in msg.lower() or "invoices" in msg.lower()


class TestIntentPhrasesConfig:
    def test_parent_overrides_availability_phrases(self):
        from enable_ai_v2 import Config, IntentPhrases, classify_query_intent, QueryIntent
        from enable_ai_v2.query_intent import is_technician_availability_query, resolve_intent_phrases

        custom = IntentPhrases(availability_phrases=("who has zero assignments",))
        config = Config(intent_phrases=custom)
        phrases = resolve_intent_phrases(config)

        assert is_technician_availability_query("who has zero assignments today", phrases)
        assert not is_technician_availability_query("who has zero assignments today")
        assert classify_query_intent("who has zero assignments", phrases) == QueryIntent.AGGREGATE

    def test_intent_phrases_from_dict(self):
        from enable_ai_v2 import intent_phrases_from_dict

        phrases = intent_phrases_from_dict({
            "scheduling_phrases": ["is it on the schedule"],
            "compound_resource_phrases": ["work order", "work orders", "invoice", "invoices"],
        })
        from enable_ai_v2.query_intent import is_scheduling_status_query, is_compound_query

        assert is_scheduling_status_query("is it on the schedule for SO-5", phrases)
        assert is_compound_query("show work orders and invoices", phrases)


class TestQueryExampleFormat:
    def test_rich_query_examples_suffix(self):
        from enable_ai_v2 import QueryExample, build_query_examples_suffix

        suffix = build_query_examples_suffix([
            QueryExample(
                query="how many open orders",
                resource="service-orders",
                question_type="count",
                notes="use list without page_size=1",
            ),
            {"query": "SO-159 status", "tools": "service_orders_list", "filters": {"search": "SO-159"}},
        ])
        assert "how many open orders" in suffix
        assert "type=count" in suffix
        assert "service_orders_list" in suffix


class TestValidatorBusinessCode:
    def test_business_code_as_path_id_needs_clarification(self):
        endpoints = [
            Endpoint(
                path="/service-orders/{id}/",
                method="GET",
                operation_id="service_orders_retrieve",
                summary="Get SO",
                description="",
                parameters=[],
            )
        ]
        validator = Validator(endpoints)
        tc = ToolCall(id="1", name="service_orders_retrieve", arguments={"id": 169})
        result = validator.validate(tc, "show SO-169")
        assert result.needs_clarification
        assert "business reference code" in (result.clarification_question or "").lower()
