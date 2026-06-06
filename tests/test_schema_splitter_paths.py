"""Tests for schema splitter path segment matching."""

from enable_ai.follow_up_detection import (
    build_list_pivot_parsed,
    should_pivot_summary_to_list,
)
from enable_ai.schema_splitter import _path_matches, split_grouped_resources


def test_path_matches_trailing_segments():
    assert _path_matches("/invoicing/invoices/", "/invoices")
    assert _path_matches("/api/master-data/companies/", "/companies")
    assert _path_matches("/invoicing/ar-dashboard/invoices/", "/ar-dashboard/invoices")
    assert _path_matches("/invoicing/ar-dashboard/", "/ar-dashboard")
    # Shorter pattern may still match; split_grouped_resources uses longest-match first
    assert _path_matches("/invoicing/ar-dashboard/invoices/", "/invoices")


def test_invoicing_split_no_path_overlap():
    schema = {
        "resources": {
            "invoicing": {
                "endpoints": [
                    {"path": "/invoicing/invoices/", "method": "GET"},
                    {"path": "/invoicing/ar-dashboard/", "method": "GET"},
                    {"path": "/invoicing/ar-dashboard/invoices/", "method": "GET"},
                    {"path": "/invoicing/ar-dashboard/credit-notes/", "method": "GET"},
                ],
            },
        },
        "resource_hints": {},
    }
    result = split_grouped_resources(schema)
    assert "invoicing-ar-invoices" in result["resources"]
    assert "invoicing-ar-credit-notes" in result["resources"]
    assert "invoicing-invoices" in result["resources"]
    assert "invoicing-ar-summary" in result["resources"]

    ar_inv = [ep["path"] for ep in result["resources"]["invoicing-ar-invoices"]["endpoints"]]
    plain_inv = [ep["path"] for ep in result["resources"]["invoicing-invoices"]["endpoints"]]
    assert any("ar-dashboard/invoices" in p for p in ar_inv)
    assert all("ar-dashboard" not in p for p in plain_inv)


def test_summary_follow_up_pivot_to_related_list():
    meta = {
        "resource": "invoicing-ar-summary",
        "question_type": "summary",
        "list_cache": [],
        "filters": {},
    }
    hints = {
        "invoicing-ar-summary": {
            "__related_list_resource__": "invoicing-ar-invoices",
        },
    }
    assert should_pivot_summary_to_list(meta, "reference")
    pivot = build_list_pivot_parsed(
        meta, hints, "show me the first one", "reference",
    )
    assert pivot["resource"] == "invoicing-ar-invoices"
    assert pivot["question_type"] == "list"
    assert pivot["limit"] == 1


def test_no_pivot_when_list_cache_present():
    meta = {
        "resource": "invoicing-ar-summary",
        "question_type": "summary",
        "list_cache": [{"id": 1}],
    }
    assert not should_pivot_summary_to_list(meta, "reference")
