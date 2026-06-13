from enable_ai.follow_up_detection import (
    build_count_list_pivot_parsed,
    should_pivot_count_to_list,
)
from enable_ai.query_examples import apply_query_example_defaults, match_query_example
from enable_ai.temporal_filters import apply_temporal_filters, detect_temporal_phrase


def test_count_to_list_pivot():
    meta = {
        "resource": "service-orders",
        "question_type": "count",
        "filters": {"status__name": {"operator": "equals", "value": "Completed"}},
    }
    assert should_pivot_count_to_list(meta, "reference")
    pivot = build_count_list_pivot_parsed(meta, "show me those", "reference", 5)
    assert pivot["resource"] == "service-orders"
    assert pivot["question_type"] == "list"
    assert pivot["filters"]["status__name"]["value"] == "Completed"


def test_query_example_match():
    schema = {
        "query_examples": [
            {
                "query": "most recent service order",
                "resource": "service-orders",
                "limit": 1,
                "sort": {"field": "created_at", "order": "desc"},
            },
        ],
    }
    match = match_query_example("what is the most recent service order?", schema)
    assert match is not None
    parsed = apply_query_example_defaults({"intent": "read"}, "most recent service order", schema)
    assert parsed["resource"] == "service-orders"
    assert parsed["limit"] == 1


def test_temporal_this_month():
    parsed = apply_temporal_filters(
        {"resource": "service-orders", "filters": {}},
        "how many service orders this month?",
        {"resource_hints": {}},
    )
    assert detect_temporal_phrase("this month") == "this_month"
    assert any(k.endswith("__gte") for k in parsed["filters"])
    assert any(k.endswith("__lte") for k in parsed["filters"])
