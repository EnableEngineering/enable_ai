from enable_ai.compound_query import split_compound_questions
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


def test_build_summary_response_text():
    data = {"invoice_count": 42, "total_overdue": 1500}
    text = build_summary_response_text(data, "invoicing-ar-summary", INVOICING_HINTS)
    assert "Invoice count is 42" in text
    assert "Total overdue amount is 1500" in text


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
