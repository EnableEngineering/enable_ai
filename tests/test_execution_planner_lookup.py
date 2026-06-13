from unittest.mock import patch

from enable_ai.execution_planner import ExecutionPlanner, _lookup_search_operator


def test_reference_code_uses_exact_lookup():
    assert _lookup_search_operator("SO-159") == "exact"
    assert _lookup_search_operator("WO-42-A") == "exact"
    assert _lookup_search_operator("Acme Corp") == "icontains"


def test_fk_lookup_plan_uses_exact_for_so_number():
    parsed = {
        "intent": "read",
        "resource": "service-orders",
        "filters": {
            "number": {"operator": "equals", "value": "SO-159"},
        },
    }
    schema = {
        "resources": {
            "service-orders": {
                "fields": ["number"],
                "endpoints": [{"method": "GET", "path": "/api/service-orders/"}],
            },
            "companies": {
                "fields": ["name"],
                "endpoints": [{"method": "GET", "path": "/api/companies/"}],
            },
        },
        "resource_hints": {},
    }
    with patch("enable_ai.execution_planner.get_openai_client"):
        planner = ExecutionPlanner()
    fk = planner._detect_fk_lookups_needed(parsed, schema)
    assert fk == []
