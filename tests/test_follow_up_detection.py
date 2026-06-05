from enable_ai.follow_up_detection import (
    apply_follow_up_context,
    is_follow_up_query,
    get_follow_up_type,
)

HISTORY = [
    {"role": "user", "content": "how many service orders are with new"},
    {
        "role": "assistant",
        "content": "There is 1 service-order\n[Context: read operation on service-orders]",
        "metadata": {
            "resource": "service-orders",
            "intent": "read",
            "filters": {
                "status__name": {"operator": "equals", "value": "New"},
            },
            "count": 1,
        },
    },
]


def test_refinement_follow_up_detected():
    assert is_follow_up_query("and assigned to which company?", HISTORY) is True
    assert get_follow_up_type("and assigned to which company?") == "refinement"


def test_standalone_company_query_not_follow_up():
    assert is_follow_up_query("list all companies", []) is False


def test_apply_follow_up_context_keeps_service_orders():
    wrong_parse = {
        "intent": "read",
        "resource": "companies",
        "filters": {},
        "question_type": "list",
    }
    fixed = apply_follow_up_context(
        wrong_parse,
        "and assigned to which company?",
        HISTORY,
        is_follow_up=True,
    )
    assert fixed["resource"] == "service-orders"
    assert fixed["filters"]["status__name"]["value"] == "New"
    assert fixed["merge_with_previous"] is True
    assert fixed["question_type"] == "details"


def test_pagination_still_detected_without_history():
    assert is_follow_up_query("show me more", []) is True
