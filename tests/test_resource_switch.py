from enable_ai.follow_up_detection import classify_follow_up, clear_classification_cache
from enable_ai.hint_utils import (
    apply_count_default_filters,
    should_force_standalone_for_resource_switch,
)
from unittest.mock import patch


HINTS = {
    "users": {"__resource_synonyms__": ["user", "technician"]},
    "service-orders": {"__resource_synonyms__": ["service order", "service orders"]},
    "reports": {
        "__aggregate_resources__": ["flash-reports", "details-reports"],
        "__resource_synonyms__": ["reports", "total reports"],
    },
    "flash-reports": {"__resource_synonyms__": ["flash report", "flash reports"]},
    "details-reports": {
        "__resource_synonyms__": ["detailed report", "detailed reports", "details report"],
        "__count_default_filters__": {
            "status": {"operator": "not_equals", "value": "archived"},
        },
        "status": {
            "values": ["active", "archived"],
            "synonyms": {"active": "active", "archived": "archived"},
        },
    },
}

RESOURCES = {
    "users", "service-orders", "flash-reports", "details-reports",
}


def test_force_standalone_different_resource():
    meta = {"resource": "users", "question_type": "list"}
    assert should_force_standalone_for_resource_switch(
        "show me 5 recent service orders",
        meta,
        HINTS,
        RESOURCES,
    )


def test_force_standalone_aggregate_narrowing():
    meta = {
        "resource": "reports",
        "question_type": "count",
        "multiple_resources": ["flash-reports", "details-reports"],
        "count": 218,
    }
    assert should_force_standalone_for_resource_switch(
        "how many detailed reports are there?",
        meta,
        HINTS,
        RESOURCES,
    )


def test_no_force_standalone_deixis_only():
    meta = {
        "resource": "inventory-equipment",
        "question_type": "count",
        "count": 15,
    }
    assert not should_force_standalone_for_resource_switch(
        "show me those",
        meta,
        HINTS,
        {"inventory-equipment"},
    )


def test_classify_forces_standalone_on_resource_switch():
    clear_classification_cache()
    history = [
        {"role": "user", "content": "list users"},
        {
            "role": "assistant",
            "content": "Found 10 users",
            "metadata": {"resource": "users", "question_type": "list", "count": 10},
        },
    ]
    with patch("enable_ai.follow_up_detection.get_openai_client") as mock_client:
        mock_client.return_value.parse_json_response.return_value = {
            "is_follow_up": True,
            "follow_up_type": "reference",
            "merge_with_previous": True,
            "keep_previous_resource": True,
        }
        result = classify_follow_up(
            "show me 5 recent service orders",
            history,
            resource_hints=HINTS,
            schema_resources=RESOURCES,
        )
    assert result["is_follow_up"] is False
    assert result["follow_up_type"] == "standalone"


def test_count_default_filters_exclude_archived():
    parsed = {
        "resource": "details-reports",
        "question_type": "count",
        "filters": {},
    }
    result = apply_count_default_filters(
        parsed,
        "how many detailed reports are there?",
        HINTS,
    )
    assert result["filters"]["status"]["operator"] == "not_equals"
    assert result["filters"]["status"]["value"] == "archived"


def test_count_default_filters_skipped_when_archived_mentioned():
    parsed = {
        "resource": "details-reports",
        "question_type": "count",
        "filters": {},
    }
    result = apply_count_default_filters(
        parsed,
        "how many archived detailed reports?",
        HINTS,
    )
    assert "status" not in result["filters"]
