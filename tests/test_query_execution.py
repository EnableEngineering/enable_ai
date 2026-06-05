"""Tests for query execution helpers."""

from enable_ai.query_execution import (
    dedupe_fk_lookup_filters,
    format_sort_param,
    merge_execution_context,
    split_filters_for_endpoint,
    enrich_step_from_parsed,
)


def test_format_sort_param_desc():
    assert format_sort_param({"field": "created_at", "order": "desc"}) == "-created_at"


def test_format_sort_param_asc():
    assert format_sort_param({"field": "name", "order": "asc"}) == "name"


def test_format_sort_param_string_passthrough():
    assert format_sort_param("-id") == "-id"


def test_merge_execution_context():
    step = {"step_id": 1, "resource": "reports", "filters": {}}
    parsed = {"sort": {"field": "created_at", "order": "desc"}, "limit": 1}
    merged = merge_execution_context(step, parsed)
    assert merged["sort"] == parsed["sort"]
    assert merged["limit"] == 1


def test_enrich_step_from_parsed():
    parsed = {
        "intent": "read",
        "resource": "details-reports",
        "sort": {"field": "created_at", "order": "desc"},
        "limit": 1,
        "question_type": "details",
    }
    step = enrich_step_from_parsed({"step_id": 1, "filters": {}}, parsed)
    assert step["resource"] == "details-reports"
    assert step["limit"] == 1
    assert step["sort"]["order"] == "desc"


def test_dedupe_fk_lookup_filters():
    filters = {
        "status__name": {"operator": "equals", "value": "New"},
        "status": {"operator": "equals", "value": "New"},
        "technician": {"operator": "equals", "value": 17},
    }
    deduped = dedupe_fk_lookup_filters(filters)
    assert "status__name" in deduped
    assert "status" not in deduped
    assert deduped["technician"]["value"] == 17


def test_split_filters_for_endpoint():
    endpoint = {
        "parameters": {
            "query": [
                {"name": "status"},
                {"name": "technician"},
            ]
        }
    }
    filters = {
        "status": {"operator": "equals", "value": "New"},
        "role": {"operator": "equals", "value": "Customer"},
    }
    server, client, warnings = split_filters_for_endpoint(filters, endpoint)
    assert "status" in server
    assert "role" in client
    assert warnings
