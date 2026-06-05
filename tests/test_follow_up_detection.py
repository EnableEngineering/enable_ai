from unittest.mock import patch

from enable_ai.follow_up_detection import (
    apply_follow_up_context,
    classify_follow_up,
    clear_classification_cache,
    is_follow_up_query,
    get_follow_up_type,
    should_merge_previous_filters,
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

REFINEMENT_CLASSIFICATION = {
    "is_follow_up": True,
    "follow_up_type": "refinement",
    "merge_with_previous": True,
    "keep_previous_resource": True,
    "question_type_override": "details",
    "display_mode_override": "detailed",
}

STANDALONE_CLASSIFICATION = {
    "is_follow_up": False,
    "follow_up_type": "standalone",
    "merge_with_previous": False,
    "keep_previous_resource": False,
    "question_type_override": None,
    "display_mode_override": None,
}

RESET_CLASSIFICATION = {
    "is_follow_up": False,
    "follow_up_type": "reset",
    "merge_with_previous": False,
    "keep_previous_resource": False,
    "question_type_override": None,
    "display_mode_override": "full",
}

PAGINATION_CLASSIFICATION = {
    "is_follow_up": True,
    "follow_up_type": "next_page",
    "merge_with_previous": True,
    "keep_previous_resource": True,
    "question_type_override": None,
    "display_mode_override": None,
}


@patch("enable_ai.follow_up_detection.get_openai_client")
def test_refinement_follow_up_detected(mock_client):
    clear_classification_cache()
    mock_client.return_value.parse_json_response.return_value = REFINEMENT_CLASSIFICATION

    assert is_follow_up_query("and assigned to which company?", HISTORY) is True
    assert get_follow_up_type("and assigned to which company?", HISTORY) == "refinement"


@patch("enable_ai.follow_up_detection.get_openai_client")
def test_standalone_company_query_not_follow_up(mock_client):
    clear_classification_cache()
    mock_client.return_value.parse_json_response.return_value = STANDALONE_CLASSIFICATION

    assert is_follow_up_query("list all companies", []) is False


@patch("enable_ai.follow_up_detection.get_openai_client")
def test_apply_follow_up_context_keeps_service_orders(mock_client):
    mock_client.return_value.parse_json_response.return_value = REFINEMENT_CLASSIFICATION

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
        classification=REFINEMENT_CLASSIFICATION,
    )
    assert fixed["resource"] == "service-orders"
    assert fixed["filters"]["status__name"]["value"] == "New"
    assert fixed["merge_with_previous"] is True
    assert fixed["question_type"] == "details"


def test_should_merge_on_refinement():
    parsed = {"merge_with_previous": True, "filters": {}}
    assert should_merge_previous_filters(parsed, REFINEMENT_CLASSIFICATION, HISTORY) is True


def test_should_not_merge_on_reset_query():
    parsed = {
        "intent": "read",
        "resource": "service-orders",
        "merge_with_previous": True,  # parser mistakenly set this
        "filters": {},
    }
    assert should_merge_previous_filters(parsed, RESET_CLASSIFICATION, HISTORY) is False


def test_apply_reset_clears_merge_flag():
    parsed = {
        "intent": "read",
        "resource": "service-orders",
        "filters": {"status__name": {"operator": "equals", "value": "New"}},
        "merge_with_previous": True,
    }
    result = apply_follow_up_context(
        parsed,
        "show all service orders",
        HISTORY,
        classification=RESET_CLASSIFICATION,
    )
    assert result["merge_with_previous"] is False
    assert result["filters"] == parsed["filters"]  # no inherited filters


@patch("enable_ai.follow_up_detection.get_openai_client")
def test_pagination_still_detected(mock_client):
    clear_classification_cache()
    mock_client.return_value.parse_json_response.return_value = PAGINATION_CLASSIFICATION

    assert is_follow_up_query("show me more", []) is True
    assert classify_follow_up("show me more", [])["follow_up_type"] == "next_page"
