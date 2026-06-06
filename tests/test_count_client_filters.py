from enable_ai.hint_utils import (
    build_count_filter_description,
    get_count_page_size,
    should_fetch_all_pages_for_count,
)
from enable_ai.query_execution import (
    apply_count_pagination_params,
    fetch_all_paginated_results,
)


def test_get_count_page_size_from_hints():
    hints = {"inventory-equipment": {"__count_page_size__": 200}}
    assert get_count_page_size("inventory-equipment", hints) == 100  # capped


def test_should_fetch_all_defaults_true_with_client_filters():
    assert should_fetch_all_pages_for_count("r", {}, True)
    assert not should_fetch_all_pages_for_count("r", {}, False)


def test_should_fetch_all_respects_hint_opt_out():
    hints = {"r": {"__count_fetch_all_when_client_filters__": False}}
    assert not should_fetch_all_pages_for_count("r", hints, True)


def test_apply_count_pagination_params_sets_page_size():
    endpoint = {"parameters": {"query": [{"name": "page_size"}]}}
    hints = {"equipment": {"__count_page_size__": 75}}
    parsed = {"question_type": "count", "resource": "equipment"}
    client = {"is_available": {"operator": "equals", "value": True}}
    params = apply_count_pagination_params({}, parsed, endpoint, hints, client)
    assert params["page_size"] == 75


def test_fetch_all_paginated_results_merges_pages():
    data = {
        "count": 3,
        "next": "http://api/page2",
        "results": [{"id": 1}, {"id": 2}],
    }

    def fetch_next(url):
        assert url == "http://api/page2"
        return {"data": {"results": [{"id": 3}], "next": None, "count": 3}}

    merged = fetch_all_paginated_results(data, fetch_next)
    assert len(merged["results"]) == 3
    assert merged["next"] is None


def test_build_count_filter_description_is_available():
    desc = build_count_filter_description(
        {"is_available": {"operator": "equals", "value": True}},
        "inventory-equipment",
        {},
    )
    assert desc == "available"


def test_build_count_filter_description_status_and_name():
    desc = build_count_filter_description(
        {
            "name": {"operator": "contains", "value": "BOROSCOPE"},
            "status": {"operator": "equals", "value": "available"},
        },
        "inventory-equipment",
        {},
    )
    assert "named BOROSCOPE" in desc
    assert "available" in desc
