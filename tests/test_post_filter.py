"""Tests for client-side post-filtering."""

from enable_ai.post_filter import apply_client_side_filters, item_matches_filters


def test_item_matches_role_filter():
    item = {"username": "a@x.com", "role": {"name": "Customer"}}
    filters = {"role": {"operator": "equals", "value": "Customer"}}
    assert item_matches_filters(item, filters)


def test_item_rejects_wrong_role():
    item = {"username": "a@x.com", "role": {"name": "Admin"}}
    filters = {"role": {"operator": "equals", "value": "Customer"}}
    assert not item_matches_filters(item, filters)


def test_apply_client_side_filters_paginated():
    data = {
        "count": 3,
        "results": [
            {"role": "Customer", "username": "c@x.com"},
            {"role": "Admin", "username": "a@x.com"},
            {"role": "Customer", "username": "c2@x.com"},
        ],
    }
    filtered, removed = apply_client_side_filters(
        data,
        {"role": {"operator": "equals", "value": "Customer"}},
    )
    assert len(filtered["results"]) == 2
    assert removed == 1
    assert filtered["count"] == 2


def test_apply_client_side_filters_preserves_count_with_next_page():
    data = {
        "count": 100,
        "next": "http://api/page2",
        "results": [
            {"role": "Customer", "username": "c@x.com"},
            {"role": "Admin", "username": "a@x.com"},
        ],
    }
    filtered, removed = apply_client_side_filters(
        data,
        {"role": {"operator": "equals", "value": "Customer"}},
    )
    assert len(filtered["results"]) == 1
    assert filtered["count"] == 100
    assert filtered.get("_client_filter_partial") is True
    assert filtered.get("_filtered_page_count") == 1


def test_apply_client_side_filters_list():
    data = [
        {"status": "New"},
        {"status": "Closed"},
    ]
    filtered, removed = apply_client_side_filters(
        data,
        {"status": {"operator": "equals", "value": "New"}},
    )
    assert len(filtered) == 1
    assert removed == 1
