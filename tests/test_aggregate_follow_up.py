from enable_ai.execution_planner import ExecutionPlanner
from enable_ai.follow_up_detection import (
    apply_follow_up_context,
    build_session_metadata,
    expand_aggregate_follow_up,
)
from enable_ai.hint_utils import get_aggregate_resources
from enable_ai.schema_splitter import split_grouped_resources
from unittest.mock import patch


HINTS = {
    "inventory": {
        "__aggregate_resources__": ["inventory-equipment", "inventory-consumables"],
        "__resource_synonyms__": ["inventory", "inventory items"],
        "__list_display_fields__": ["name", "stock_level"],
    },
}


def test_get_aggregate_resources_alias():
    assert get_aggregate_resources({
        "__aggregate__": ["a", "b"],
    }) == ["a", "b"]
    assert get_aggregate_resources({
        "__aggregate_resources__": ["x", "y"],
    }) == ["x", "y"]


def test_expand_aggregate_follow_up_after_count():
    meta = {
        "resource": "inventory",
        "question_type": "count",
        "count": 26,
        "filters": {},
    }
    parsed = {
        "intent": "read",
        "resource": "inventory",
        "question_type": "list",
        "merge_with_previous": True,
    }
    clf = {
        "is_follow_up": True,
        "follow_up_type": "reference",
        "question_type_override": "list",
        "merge_with_previous": True,
        "keep_previous_resource": True,
    }
    result = expand_aggregate_follow_up(parsed, meta, clf, HINTS)
    assert result["multiple_resources"] == ["inventory-equipment", "inventory-consumables"]
    assert result["resource"] == "inventory-equipment"


def test_expand_aggregate_follow_up_from_session_multiple_resources():
    meta = {
        "resource": "inventory",
        "question_type": "count",
        "multiple_resources": ["inventory-equipment", "inventory-consumables"],
    }
    parsed = {"resource": "inventory", "question_type": "list"}
    clf = {"follow_up_type": "reference", "question_type_override": "list"}
    result = expand_aggregate_follow_up(parsed, meta, clf, {})
    assert result["multiple_resources"] == ["inventory-equipment", "inventory-consumables"]


def test_apply_follow_up_context_sets_multiple_resources():
    history = [
        {"role": "user", "content": "how many inventory items?"},
        {
            "role": "assistant",
            "content": "There are 26 inventory items",
            "metadata": {
                "resource": "inventory",
                "question_type": "count",
                "count": 26,
                "filters": {},
            },
        },
    ]
    parsed = {
        "intent": "read",
        "resource": "inventory",
        "question_type": "details",
        "filters": {},
    }
    classification = {
        "is_follow_up": True,
        "follow_up_type": "reference",
        "merge_with_previous": True,
        "keep_previous_resource": True,
        "question_type_override": "list",
    }
    with patch(
        "enable_ai.follow_up_detection.classify_follow_up",
        return_value=classification,
    ):
        result = apply_follow_up_context(
            parsed,
            "show me those",
            history,
            is_follow_up=True,
            classification=classification,
            resource_hints=HINTS,
        )
    assert result["question_type"] == "list"
    assert result["multiple_resources"] == ["inventory-equipment", "inventory-consumables"]


def test_strip_virtual_aggregate_parent():
    schema = {
        "resources": {
            "inventory": {
                "endpoints": [
                    {"path": "/api/inventory/equipment/", "method": "GET"},
                    {"path": "/api/inventory/consumables/", "method": "GET"},
                    {"path": "/api/inventory/low-stock/", "method": "GET"},
                ],
            },
            "inventory-equipment": {
                "endpoints": [{"path": "/api/inventory/equipment/", "method": "GET"}],
            },
        },
        "resource_hints": {
            "inventory": {
                "__aggregate_resources__": ["inventory-equipment", "inventory-consumables"],
            },
        },
        "resource_split_rules": {
            "inventory": {
                "inventory-equipment": "/equipment",
                "inventory-consumables": "/consumables",
            },
        },
    }
    result = split_grouped_resources(schema)
    assert "inventory" not in result["resources"]
    assert "inventory-equipment" in result["resources"]


def test_plan_multi_resource_list():
    with patch("enable_ai.execution_planner.get_openai_client"):
        planner = ExecutionPlanner()
    parsed = {
        "intent": "read",
        "resource": "inventory-equipment",
        "multiple_resources": ["inventory-equipment", "inventory-consumables"],
        "question_type": "list",
        "filters": {},
    }
    plan = planner._plan_multi_resource_list(parsed)
    assert plan is not None
    assert plan["plan_type"] == "multi_resource_list"
    assert plan["total_steps"] == 2


def test_build_session_metadata_persists_multiple_resources():
    meta = build_session_metadata(
        {
            "resource": "inventory",
            "question_type": "count",
            "multiple_resources": ["inventory-equipment", "inventory-consumables"],
        },
        {"data": {"count": 26}, "pagination": {"total_count": 26}},
    )
    assert meta["multiple_resources"] == ["inventory-equipment", "inventory-consumables"]
