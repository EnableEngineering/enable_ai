from enable_ai.hint_utils import (
    expand_query_resources,
    find_aggregate_resources_for_query,
    get_extra_query_params,
    get_user_scoped_fields,
    query_implies_breadth,
    query_implies_user_scope,
    strip_user_scoped_filters_on_breadth,
)
from enable_ai.user_context_resolver import resolve_user_context_in_parsed


def test_query_implies_user_scope_excludes_show_me():
    assert not query_implies_user_scope("Show me the low stock items to refill")
    assert not query_implies_user_scope("pls show me all service orders")
    assert not query_implies_user_scope("show me all of them")


def test_query_implies_breadth_without_user_scope():
    assert query_implies_breadth("show me all service orders")
    assert not query_implies_user_scope("show me all service orders")
    assert query_implies_user_scope("show all my service orders")


def test_strip_user_scoped_on_breadth():
    parsed = {
        "resource": "service-orders",
        "filters": {"technician": {"operator": "equals", "value": 29}},
        "entities": {"technician": 29},
    }
    hints = {"service-orders": {"__user_scoped_fields__": ["technician"]}}
    result = strip_user_scoped_filters_on_breadth(
        parsed, "pls show me all service orders", hints,
    )
    assert "technician" not in result["filters"]
    assert "technician" not in result["entities"]


def test_get_extra_query_params():
    hints = {"users": {"__extra_query_params__": ["role", "is_active"]}}
    assert get_extra_query_params("users", hints) == {"role", "is_active"}


def test_query_implies_user_scope_includes_assigned_to_me():
    assert query_implies_user_scope("Show me new service orders assigned to me")
    assert query_implies_user_scope("what are the observations for my last report?")


def test_no_technician_injection_without_user_scoped_fields():
    parsed = {
        "intent": "read",
        "resource": "inventory-consumables",
        "filters": {"stock_level": {"operator": "equals", "value": "low"}},
    }
    hints = {
        "service-orders": {"__user_scoped_fields__": ["technician"]},
        "inventory-consumables": {},
    }
    result = resolve_user_context_in_parsed(
        parsed,
        {"user_id": 29},
        "Show me the low stock items to refill",
        resource_hints=hints,
    )
    assert "technician" not in result["filters"]
    assert "technician" not in result.get("entities", {})


def test_technician_injection_with_user_scoped_fields():
    parsed = {
        "intent": "read",
        "resource": "service-orders",
        "filters": {},
    }
    hints = {"service-orders": {"__user_scoped_fields__": ["technician"]}}
    result = resolve_user_context_in_parsed(
        parsed,
        {"user_id": 17},
        "Show me service orders assigned to me",
        resource_hints=hints,
    )
    assert result["filters"]["technician"]["value"] == 17
    assert result["entities"]["technician"] == 17


def test_filters_for_display_uses_resolved_entities():
    from enable_ai.user_context_resolver import filters_for_display

    parsed = {
        "entities": {"technician": 17},
        "filters": {"technician": {"operator": "equals", "value": "__current_user_id__"}},
    }
    display = filters_for_display(parsed)
    assert display["technician"]["value"] == 17


def test_find_aggregate_resources():
    hints = {
        "inventory": {
            "__aggregate_resources__": ["inventory-equipment", "inventory-consumables"],
            "__resource_synonyms__": ["inventory", "inventory items"],
        }
    }
    agg = find_aggregate_resources_for_query("how many inventory items are there?", hints)
    assert agg == ["inventory-equipment", "inventory-consumables"]


def test_expand_query_resources_aggregate():
    schema = {
        "resources": {
            "inventory-equipment": {},
            "inventory-consumables": {},
        },
        "resource_hints": {
            "inventory": {
                "__aggregate_resources__": ["inventory-equipment", "inventory-consumables"],
                "__resource_synonyms__": ["inventory items"],
            }
        },
    }
    parsed = {"question_type": "count", "resource": "inventory-equipment"}
    expanded = expand_query_resources(parsed, "how many inventory items?", schema)
    assert expanded["multiple_resources"] == ["inventory-equipment", "inventory-consumables"]


def test_get_user_scoped_fields():
    hints = {"service-orders": {"__user_scoped_fields__": ["technician"]}}
    assert get_user_scoped_fields("service-orders", hints) == ["technician"]
    assert get_user_scoped_fields("inventory-consumables", hints) == []
