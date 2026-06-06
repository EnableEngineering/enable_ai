from unittest.mock import patch

from enable_ai.follow_up_detection import (
    apply_follow_up_context,
    apply_referent_context,
    build_session_metadata,
    classify_follow_up,
    clear_classification_cache,
    extract_result_items_from_data,
    is_follow_up_query,
    get_follow_up_type,
    should_merge_previous_filters,
    strip_inherited_session_filters,
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
    assert result["filters"] == {}


def test_strip_inherited_session_filters_removes_prev_and_technician():
    """Test that user-scoped filters are removed using schema-driven __user_scoped_fields__."""
    parsed = {
        "intent": "read",
        "resource": "service-orders",
        "filters": {
            "status__name": {"operator": "equals", "value": "New"},
            "technician": {"operator": "equals", "value": 17},
        },
        "merge_with_previous": True,
    }
    history_with_tech = [
        *HISTORY[:-1],
        {
            **HISTORY[-1],
            "metadata": {
                **HISTORY[-1]["metadata"],
                "filters": {
                    "status__name": {"operator": "equals", "value": "New"},
                    "technician": {"operator": "equals", "value": 17},
                },
            },
        },
    ]
    # v0.3.68: user-scoped fields now come from resource_hints.__user_scoped_fields__
    resource_hints = {
        "service-orders": {
            "__user_scoped_fields__": ["technician", "assigned_to"],
        },
    }
    result = strip_inherited_session_filters(
        parsed, history_with_tech, user_context={"user_id": 17}, resource_hints=resource_hints,
    )
    assert result["filters"] == {}
    assert result["merge_with_previous"] is False


def test_apply_referent_context_sets_id_filter():
    classification = {
        "question_type_override": "details",
        "display_mode_override": "detailed",
        "referent": {
            "resource": "service-orders",
            "id": 28,
            "id_field": "id",
            "label": "SO-5",
        },
    }
    parsed = {"intent": "read", "resource": "companies", "filters": {}}
    result = apply_referent_context(parsed, classification)
    assert result["resource"] == "service-orders"
    assert result["filters"]["id"]["value"] == 28
    assert result["question_type"] == "details"
    assert result["limit"] == 1


def test_extract_result_items_from_list_response():
    data = {
        "results": [
            {"id": 28, "code": "SO-5", "company": {"id": 3, "name": "Bharat Engineering Works"}},
        ]
    }
    items = extract_result_items_from_data(data)
    assert len(items) == 1
    assert items[0]["id"] == 28
    assert items[0]["code"] == "SO-5"
    assert items[0]["company_name"] == "Bharat Engineering Works"


def test_build_session_metadata_includes_primary_item():
    parsed = {"resource": "service-orders", "intent": "read", "filters": {}}
    response = {
        "data": {"results": [{"id": 28, "code": "SO-5"}]},
        "pagination": {"total_count": 1, "has_more": False},
    }
    meta = build_session_metadata(parsed, response)
    assert meta["primary_item"]["id"] == 28
    assert meta["result_items"][0]["code"] == "SO-5"


@patch("enable_ai.follow_up_detection.get_openai_client")
def test_pagination_still_detected(mock_client):
    clear_classification_cache()
    mock_client.return_value.parse_json_response.return_value = PAGINATION_CLASSIFICATION

    assert is_follow_up_query("show me more", []) is True
    assert classify_follow_up("show me more", [])["follow_up_type"] == "next_page"


# Regression test for v0.3.68 bug: same resource + different filters = standalone
DIFFERENT_FILTER_STANDALONE = {
    "is_follow_up": False,
    "follow_up_type": "standalone",
    "merge_with_previous": False,
    "keep_previous_resource": False,
    "question_type_override": None,
    "display_mode_override": None,
    "referent": None,
}


@patch("enable_ai.follow_up_detection.get_openai_client")
def test_same_resource_different_filters_is_standalone(mock_client):
    """
    Bug fix: 'low priority service orders' after 'new service orders' should be standalone.
    Same resource but different filter scope = no merge.
    """
    clear_classification_cache()
    mock_client.return_value.parse_json_response.return_value = DIFFERENT_FILTER_STANDALONE

    history_new_orders = [
        {"role": "user", "content": "Show me new service orders assigned to me"},
        {
            "role": "assistant",
            "content": "Found 5 new service orders",
            "metadata": {
                "resource": "service-orders",
                "filters": {
                    "status__name": {"operator": "equals", "value": "New"},
                    "technician": {"operator": "equals", "value": 17},
                },
            },
        },
    ]

    # Query about low priority — different filter scope, should NOT inherit status__name=New
    clf = classify_follow_up("Which service orders are in low priority?", history_new_orders)
    assert clf["follow_up_type"] == "standalone"
    assert clf["merge_with_previous"] is False
    assert should_merge_previous_filters({}, clf, history_new_orders) is False
