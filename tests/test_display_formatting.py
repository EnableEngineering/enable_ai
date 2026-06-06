from enable_ai.display_formatting import format_display_value
from enable_ai.follow_up_detection import apply_summary_metric_follow_up
from enable_ai.hint_utils import (
    build_summary_response_text,
    find_best_explicit_resource,
)
from enable_ai.response_projector import format_projected_table


AR_HINTS = {
    "invoicing-ar-summary": {
        "__endpoint_role__": "summary",
        "__currency_code__": "INR",
        "__currency_fields__": ["total_outstanding", "collected_this_month"],
        "__response_summary_fields__": {
            "total_outstanding": "Total outstanding amount",
            "collected_this_month": "Amount collected this month",
            "days_1_30_bucket.amount": "1-30 days overdue amount",
        },
        "__response_summary_field_synonyms__": {
            "outstanding": "total_outstanding",
            "collected": "collected_this_month",
        },
    },
    "inventory-items": {
        "__resource_synonyms__": ["item", "items"],
        "__endpoint_role__": "list",
    },
    "inventory-available": {
        "__resource_synonyms__": ["available", "available stock"],
        "__endpoint_role__": "list",
    },
}


def test_currency_formatting_in_summary():
    data = {"total_outstanding": 12500.50, "collected_this_month": 8000}
    text = build_summary_response_text(
        data,
        "invoicing-ar-summary",
        AR_HINTS,
        query="what is the outstanding amount?",
    )
    assert "₹12,500.50" in text
    assert "Total outstanding amount is" in text


def test_nested_bucket_summary_field():
    data = {
        "days_1_30_bucket": {"amount": 4200, "count": 3},
    }
    text = build_summary_response_text(
        data,
        "invoicing-ar-summary",
        AR_HINTS,
        query="1-30 days overdue amount",
        parsed={"summary_field": "days_1_30_bucket.amount"},
    )
    assert "1-30 days overdue amount is" in text
    assert "4,200" in text


def test_table_human_labels_and_booleans():
    hints = {
        "users": {
            "__list_display_labels__": {
                "role.name": "Role",
                "is_active": "Status",
            },
        },
    }
    rows = [{"role.name": "Technician", "is_active": True}]
    table = format_projected_table(
        rows, ["role.name", "is_active"], resource="users", resource_hints=hints,
    )
    assert "Role" in table
    assert "Status" in table
    assert "Active" in table
    assert "role.name" not in table


def test_longest_match_resource_resolution():
    resources = {"inventory-items", "inventory-available"}
    best = find_best_explicit_resource(
        "how many available items are there?",
        AR_HINTS,
        resources,
    )
    assert best in ("inventory-items", "inventory-available")

    tied = find_best_explicit_resource(
        "show available items",
        AR_HINTS,
        resources,
        prior_resource="inventory-available",
    )
    assert tied == "inventory-available"


def test_summary_metric_follow_up_sets_field():
    history = [
        {"role": "assistant", "content": "Summary.", "metadata": {
            "resource": "invoicing-ar-summary",
            "question_type": "summary",
        }},
    ]
    parsed = apply_summary_metric_follow_up(
        {"resource": "invoicing-ar-invoices", "question_type": "read"},
        "and what's the collected amount?",
        history,
        AR_HINTS,
    )
    assert parsed["resource"] == "invoicing-ar-summary"
    assert parsed["question_type"] == "summary"
    assert parsed["summary_field"] == "collected_this_month"


def test_format_display_value_without_currency_config():
    assert format_display_value(99.5, "total", "other", {}) == "99.5"


def test_currency_string_decimal_drf():
    hints = {
        "invoicing-ar-summary": {
            "__currency_code__": "INR",
            "__currency_fields__": [
                "total_outstanding",
                "collected_this_month",
                "total_invoiced",
            ],
        },
    }
    assert format_display_value(
        "51364.00", "total_outstanding", "invoicing-ar-summary", hints,
    ) == "₹51,364"
    assert format_display_value(
        "14643.80", "collected_this_month", "invoicing-ar-summary", hints,
    ) == "₹14,643.80"
    assert format_display_value(
        "455228.90", "total_invoiced", "invoicing-ar-summary", hints,
    ) == "₹455,228.90"
