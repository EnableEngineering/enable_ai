from enable_ai.follow_up_suggestions import (
    generate_follow_up_queries,
    generate_suggestions,
)
from enable_ai.hint_utils import (
    apply_resource_question_defaults,
    build_summary_response_text,
    resolve_summary_field_from_query,
)
from enable_ai.follow_up_detection import (
    build_list_pivot_parsed,
    should_pivot_summary_to_list,
)

AR_HINTS = {
    "invoicing-ar-summary": {
        "__endpoint_role__": "summary",
        "__related_list_resource__": "invoicing-ar-invoices",
        "__response_summary_fields__": {
            "total_outstanding": "Total outstanding amount",
            "collected_this_month": "Amount collected this month",
            "total_overdue": "Total overdue amount",
            "invoice_count": "Invoice count",
        },
        "__response_summary_field_synonyms__": {
            "outstanding": "total_outstanding",
            "collected": "collected_this_month",
            "overdue": "total_overdue",
        },
    },
    "invoicing-ar-invoices": {
        "__endpoint_role__": "list",
        "__related_summary_resource__": "invoicing-ar-summary",
    },
}

AR_DATA = {
    "invoice_count": 3,
    "total_outstanding": 12500.50,
    "collected_this_month": 8000,
    "total_overdue": 0,
}


def test_follow_up_summary_single_metric_pending():
    text = build_summary_response_text(
        AR_DATA,
        "invoicing-ar-summary",
        AR_HINTS,
        query="and what's the pending amount?",
        follow_up_only=True,
    )
    assert "Total outstanding amount is 12500.5" in text
    assert "collected" not in text.lower()
    assert "Invoice count" not in text


def test_follow_up_only_refuses_all_fields_dump():
    text = build_summary_response_text(
        AR_DATA,
        "invoicing-ar-summary",
        AR_HINTS,
        query="tell me more",
        follow_up_only=True,
    )
    assert "couldn't tell which metric" in text
    assert "Invoice count is" not in text


def test_builtin_pending_amount_synonym():
    field = resolve_summary_field_from_query(
        "what is the pending amount?",
        "invoicing-ar-summary",
        AR_HINTS,
    )
    assert field == "total_outstanding"


def test_summary_follow_up_chips_no_pagination():
    queries = generate_follow_up_queries(
        {"total_count": 3, "actual_count": 3, "has_more": True},
        {"resource": "invoicing-ar-summary", "question_type": "summary"},
        session_metadata={"question_type": "summary", "list_cache": []},
        resource_hints=AR_HINTS,
    )
    labels = [q["label"] for q in queries]
    assert not any("next" in l.lower() for l in labels)
    assert any("outstanding" in l.lower() or "show" in l.lower() for l in labels)


def test_summary_suggestions_no_show_more():
    suggestions = generate_suggestions(
        {"total_count": 3, "has_more": True},
        {"resource": "invoicing-ar-summary", "question_type": "summary"},
        session_metadata={"question_type": "summary", "list_cache": []},
        resource_hints=AR_HINTS,
    )
    assert not any("more" in s.lower() for s in suggestions)


def test_amount_query_routes_to_summary_resource():
    schema = {"resource_hints": AR_HINTS}
    parsed = apply_resource_question_defaults(
        {
            "resource": "invoicing-ar-invoices",
            "question_type": "count",
        },
        "what's the total invoiced amount?",
        schema,
    )
    assert parsed["resource"] == "invoicing-ar-summary"
    assert parsed["question_type"] == "summary"


def test_summary_next_page_pivots_to_list():
    meta = {
        "resource": "invoicing-ar-summary",
        "question_type": "summary",
        "list_cache": [],
    }
    assert should_pivot_summary_to_list(meta, "next_page")
    pivot = build_list_pivot_parsed(
        meta, AR_HINTS, "show me next 3", "next_page", requested_limit=3,
    )
    assert pivot["resource"] == "invoicing-ar-invoices"
    assert pivot["limit"] == 3
