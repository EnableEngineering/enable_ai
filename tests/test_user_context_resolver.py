from unittest.mock import patch

from enable_ai.execution_planner import ExecutionPlanner
from enable_ai.user_context_resolver import (
    normalize_user_context,
    resolve_params_dict,
    resolve_user_context_in_parsed,
    resolve_user_context_placeholders,
)


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


def test_normalize_user_context_aliases():
    ctx = normalize_user_context({"id": 17, "username": "tech@example.com"})
    assert ctx["user_id"] == 17


def test_resolve_current_user_id_without_underscores():
    parsed = {
        "filters": {
            "technician": {"operator": "equals", "value": "current_user_id"},
        },
    }
    result = resolve_user_context_in_parsed(parsed, {"user_id": 17})
    assert result["filters"]["technician"]["value"] == 17


def test_resolve_params_dict_flat_placeholder():
    params = {"technician": "__current_user_id__", "status__name": "New"}
    resolved = resolve_params_dict(params, {"user_id": 17})
    assert resolved["technician"] == 17
    assert resolved["status__name"] == "New"


def test_resolve_user_context_placeholders_alias():
    parsed = {
        "filters": {"technician": {"operator": "equals", "value": "__current_user_id__"}},
    }
    result = resolve_user_context_placeholders(parsed, {"id": 17})
    assert result["filters"]["technician"]["value"] == 17


def test_admin_role_gets_no_scoped_injection():
    hints = {
        "service-orders": {
            "__user_scoped_fields__": ["technician"],
            "__user_scoped_fields_by_role__": {
                "Technician": ["technician"],
                "Admin": [],
            },
        },
    }
    parsed = {
        "resource": "service-orders",
        "filters": {},
        "entities": {},
    }
    result = resolve_user_context_in_parsed(
        parsed,
        {"user_id": 1, "role": "Admin"},
        query="show my service orders",
        resource_hints=hints,
    )
    assert "technician" not in result["filters"]


def test_customer_company_scoped_injection():
    hints = {
        "invoicing-invoices": {
            "__user_scoped_fields_by_role__": {
                "Customer": ["company"],
            },
        },
    }
    parsed = {
        "resource": "invoicing-invoices",
        "filters": {},
        "entities": {},
    }
    result = resolve_user_context_in_parsed(
        parsed,
        {"user_id": 5, "company_id": 42, "role": "Customer"},
        query="show my invoices",
        resource_hints=hints,
    )
    assert result["filters"]["company"]["value"] == 42

