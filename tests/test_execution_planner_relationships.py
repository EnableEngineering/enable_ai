from unittest.mock import patch

from enable_ai.execution_planner import ExecutionPlanner


def test_plan_relationship_query_parent_child():
    with patch("enable_ai.execution_planner.get_openai_client"):
        planner = ExecutionPlanner()
    parsed = {
        "intent": "read",
        "resource": "details-reports",
        "filters": {"technician": {"operator": "equals", "value": 17}},
        "sort": {"field": "created_at", "order": "desc"},
        "limit": 1,
        "relationships": [
            {"type": "child", "target_entity": "observations", "filters": {}},
        ],
    }
    plan = planner._plan_relationship_query(parsed)
    assert plan is not None
    assert plan["total_steps"] == 2
    assert plan["steps"][0]["resource"] == "details-reports"
    assert plan["steps"][1]["resource"] == "observations"
    assert plan["steps"][0]["extract"]["parent_id"] == "$.results[0].id"
    assert plan["steps"][1]["depends_on"] == [1]


def test_plan_multi_resource_count():
    with patch("enable_ai.execution_planner.get_openai_client"):
        planner = ExecutionPlanner()
    parsed = {
        "intent": "read",
        "resource": "flash-reports",
        "multiple_resources": ["flash-reports", "details-reports"],
        "question_type": "count",
        "filters": {},
    }
    plan = planner.create_execution_plan(parsed, {"resources": {}, "resource_hints": {}})
    assert plan["is_multi_step"] is True
    assert plan["plan_type"] == "multi_resource_count"
    assert plan["total_steps"] == 2
    assert plan["steps"][0]["resource"] == "flash-reports"
    assert plan["steps"][1]["resource"] == "details-reports"
