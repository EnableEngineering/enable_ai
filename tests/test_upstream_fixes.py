from enable_ai.query_utils import split_compound_questions
from enable_ai.hint_utils import (
    build_summary_response_text,
    expand_query_resources,
    find_aggregate_resources_for_query,
    find_child_resource_over_aggregate,
    is_summary_response,
)
from enable_ai.schema_splitter import split_grouped_resources
from enable_ai.workflow import _analyze_pagination, _api_total_count


REPORT_HINTS = {
    "reports": {
        "__aggregate_resources__": ["flash-reports", "details-reports"],
        "__resource_synonyms__": ["reports", "total reports"],
    },
    "flash-reports": {"__resource_synonyms__": ["flash report", "flash reports"]},
    "details-reports": {
        "__resource_synonyms__": ["detailed report", "detailed reports", "details report"],
    },
}

INVOICING_HINTS = {
    "invoicing-ar-summary": {
        "__endpoint_role__": "summary",
        "__response_summary_fields__": {
            "invoice_count": "Invoice count",
            "total_overdue": "Total overdue amount",
            "total_outstanding": "Total outstanding amount",
            "collected_this_month": "Amount collected this month",
        },
        "__response_summary_field_synonyms__": {
            "outstanding": "total_outstanding",
            "oustanding": "total_outstanding",
            "collected": "collected_this_month",
            "collected this month": "collected_this_month",
            "overdue": "total_overdue",
        },
        "__response_count_fields__": ["invoice_count"],
    },
}


def test_child_over_aggregate_detailed_reports():
    resources = {"flash-reports", "details-reports"}
    child = find_child_resource_over_aggregate(
        "how many detailed reports are there?",
        REPORT_HINTS,
        resources,
    )
    assert child == "details-reports"

    parsed = expand_query_resources(
        {"resource": "reports", "question_type": "count"},
        "how many detailed reports are there?",
        {"resource_hints": REPORT_HINTS, "resources": {r: {} for r in resources}},
    )
    assert parsed["resource"] == "details-reports"
    assert "multiple_resources" not in parsed


def test_aggregate_still_matches_generic_reports():
    agg = find_aggregate_resources_for_query(
        "how many total reports are there?",
        REPORT_HINTS,
    )
    assert agg == ["flash-reports", "details-reports"]


def test_summary_response_detection():
    data = {"invoice_count": 42, "total_overdue": 1500.50}
    assert is_summary_response(data, "invoicing-ar-summary", INVOICING_HINTS)


def test_api_total_count_custom_fields():
    data = {"invoice_count": 42, "total_overdue": 1500}
    assert _api_total_count(data, ["invoice_count"]) == 42


def test_analyze_pagination_summary_not_one():
    data = {"invoice_count": 42, "total_overdue": 1500}
    info = _analyze_pagination(data, "invoicing-ar-summary", INVOICING_HINTS)
    assert info["is_summary"] is True
    assert info["total_count"] == 42


def test_build_summary_response_text_all_fields_fallback():
    data = {"invoice_count": 42, "total_overdue": 1500}
    text = build_summary_response_text(data, "invoicing-ar-summary", INVOICING_HINTS)
    assert "Invoice count is 42" in text
    assert "Total overdue amount is 1500" in text


def test_build_summary_response_text_query_aware_outstanding():
    data = {
        "invoice_count": 3,
        "total_overdue": 0,
        "total_outstanding": 12500.50,
        "collected_this_month": 8000,
    }
    text = build_summary_response_text(
        data,
        "invoicing-ar-summary",
        INVOICING_HINTS,
        query="what is the total outstanding amount?",
    )
    assert "Total outstanding amount is 12500.5" in text
    assert "Invoice count" not in text
    assert "collected" not in text.lower()


def test_build_summary_response_text_query_aware_collected():
    data = {
        "total_outstanding": 100,
        "collected_this_month": 4200,
    }
    text = build_summary_response_text(
        data,
        "invoicing-ar-summary",
        INVOICING_HINTS,
        query="amount collected this month",
    )
    assert "Amount collected this month is 4200" in text
    assert "outstanding" not in text.lower()


def test_typo_tolerance_outstanding():
    from enable_ai.hint_utils import resolve_summary_field_from_query
    field = resolve_summary_field_from_query(
        "what is total oustanding?",
        "invoicing-ar-summary",
        INVOICING_HINTS,
    )
    assert field == "total_outstanding"


def test_invoicing_schema_split():
    schema = {
        "resources": {
            "invoicing": {
                "endpoints": [
                    {"path": "/invoicing/invoices/", "method": "GET"},
                    {"path": "/invoicing/ar-dashboard/", "method": "GET"},
                ],
            },
        },
        "resource_hints": {
            "invoicing-invoices": {},
            "invoicing-ar-summary": {},
        },
    }
    split = split_grouped_resources(schema)
    assert "invoicing-invoices" in split["resources"]
    assert "invoicing-ar-summary" in split["resources"]
    inv_paths = [ep["path"] for ep in split["resources"]["invoicing-invoices"]["endpoints"]]
    assert any("/invoices" in p for p in inv_paths)


def test_split_compound_questions():
    q = (
        "how many overdue invoices are there, "
        "what is the total overdue amount, "
        "how many draft invoices exist"
    )
    parts = split_compound_questions(q)
    assert len(parts) == 3


def test_no_split_single_question_with_commas():
    q = "show me invoices for company A, B, and C"
    parts = split_compound_questions(q)
    assert parts == [q]


def test_split_compound_and_joined():
    q = "how many overdue invoices are there and what is total outstanding"
    parts = split_compound_questions(q)
    assert len(parts) == 2


def test_split_compound_strips_prefix():
    q = "or invoicing questions like- how many draft invoices, what is total overdue"
    parts = split_compound_questions(q)
    assert len(parts) == 2
    assert "draft invoices" in parts[0]


def test_align_parsed_resource_with_query():
    from enable_ai.hint_utils import align_parsed_resource_with_query

    hints = {
        "users": {},
        "service-orders": {"__resource_synonyms__": ["service orders"]},
    }
    parsed = align_parsed_resource_with_query(
        {"resource": "users", "question_type": "list"},
        "show me 5 recent service orders",
        hints,
        {"users", "service-orders"},
    )
    assert parsed["resource"] == "service-orders"


def test_apply_resource_question_defaults_summary():
    from enable_ai.hint_utils import apply_resource_question_defaults

    schema = {
        "resource_hints": {
            "invoicing-ar-summary": {"__endpoint_role__": "summary"},
        },
    }
    parsed = apply_resource_question_defaults(
        {"resource": "invoicing-ar-summary", "question_type": "count"},
        "what is total outstanding?",
        schema,
    )
    assert parsed["question_type"] == "summary"


def test_auto_discover_summary_fields_from_schema():
    from enable_ai.hint_utils import get_response_summary_fields

    hints = {"ar-dash": {"__endpoint_role__": "summary"}}
    schema_res = {
        "fields": [
            {"name": "total_outstanding", "label": "Total outstanding"},
            {"name": "collected_this_month"},
        ],
    }
    fields = get_response_summary_fields("ar-dash", hints, schema_res)
    assert fields["total_outstanding"] == "Total outstanding"
    assert "collected_this_month" in fields
