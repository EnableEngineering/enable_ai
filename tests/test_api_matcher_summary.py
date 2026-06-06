from enable_ai.api_matcher import APIMatcher
from enable_ai.hint_utils import build_summary_list_mismatch_message


INVOICING_SCHEMA = {
    "resources": {
        "invoicing-ar-summary": {
            "endpoints": [
                {
                    "path": "/api/invoicing/ar-summary/",
                    "method": "GET",
                    "intent": "read",
                },
            ],
        },
        "invoicing-ar-invoices": {
            "endpoints": [
                {
                    "path": "/api/invoicing/invoices/",
                    "method": "GET",
                    "intent": "read",
                },
            ],
        },
    },
    "resource_hints": {
        "invoicing-ar-summary": {
            "__endpoint_role__": "summary",
            "__resource_synonyms__": ["invoice", "invoices", "ar summary"],
            "__response_summary_fields__": {
                "total_outstanding": "Total outstanding amount",
                "total_invoiced": "Total invoiced amount",
                "collected_this_month": "Amount collected this month",
            },
            "__response_summary_field_synonyms__": {
                "outstanding": "total_outstanding",
                "collected": "collected_this_month",
                "invoiced": "total_invoiced",
            },
        },
        "invoicing-ar-invoices": {
            "__endpoint_role__": "list",
            "__resource_synonyms__": ["invoice", "invoices"],
            "__related_summary_resource__": "invoicing-ar-summary",
        },
    },
}


def test_matcher_locks_summary_resource_despite_invoice_synonyms():
    matcher = APIMatcher()
    parsed = {
        "intent": "read",
        "resource": "invoicing-ar-summary",
        "question_type": "summary",
        "summary_field": "total_outstanding",
        "original_input": "what is total oustanding amount?",
        "filters": {},
        "entities": {},
    }
    result = matcher.match_api(parsed, INVOICING_SCHEMA)
    assert result.endpoint.endswith("/ar-summary/")


def test_matcher_routes_amount_query_to_related_summary():
    matcher = APIMatcher()
    parsed = {
        "intent": "read",
        "resource": "invoicing-ar-invoices",
        "question_type": "count",
        "original_input": "What's the total invoice amount?",
        "filters": {},
        "entities": {},
    }
    result = matcher.match_api(parsed, INVOICING_SCHEMA)
    assert result.endpoint.endswith("/ar-summary/")


def test_summary_list_mismatch_message():
    msg = build_summary_list_mismatch_message(
        "invoicing-ar-invoices",
        INVOICING_SCHEMA["resource_hints"],
        {"summary_field": "total_invoiced"},
    )
    assert "dashboard metrics" in msg.lower()
    assert "invoicing ar summary" in msg.lower()
