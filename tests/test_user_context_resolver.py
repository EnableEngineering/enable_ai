from enable_ai.execution_planner import ExecutionPlanner
from enable_ai.user_context_resolver import resolve_user_context_in_parsed
from unittest.mock import patch


def test_resolve_current_user_id_placeholder():
    parsed = {
        "intent": "read",
        "resource": "service-orders",
        "filters": {
            "status__name": {"operator": "equals", "value": "New"},
            "technician": {"operator": "equals", "value": "__current_user_id__"},
        },
    }
    user_context = {"user_id": 17, "username": "tech@example.com"}

    result = resolve_user_context_in_parsed(parsed, user_context)

    assert result["filters"]["technician"]["value"] == 17
    assert result["filters"]["status__name"]["value"] == "New"


def test_no_fk_lookup_after_user_context_resolution():
    parsed = {
        "intent": "read",
        "resource": "service-orders",
        "filters": {
            "status__name": {"operator": "equals", "value": "New"},
            "technician": {"operator": "equals", "value": "__current_user_id__"},
        },
    }
    schema = {
        "resources": {
            "users": {"endpoints": [{"method": "GET", "path": "/api/users/"}]},
            "service-orders": {"endpoints": [{"method": "GET", "path": "/api/service-orders/"}]},
        },
        "resource_hints": {},
    }
    user_context = {"user_id": 17}

    resolved = resolve_user_context_in_parsed(parsed, user_context)

    with patch("enable_ai.execution_planner.get_openai_client"):
        planner = ExecutionPlanner()
    lookups = planner._detect_fk_lookups_needed(resolved, schema)
    plan = planner.create_execution_plan(resolved, schema)

    assert lookups == []
    assert plan["total_steps"] == 1
    assert plan["steps"][0]["filters"]["technician"]["value"] == 17
