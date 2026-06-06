from enable_ai.workflow import _analyze_pagination, _api_total_count


def test_api_total_count_from_count_field():
    data = {"count": 81, "results": [{"id": i} for i in range(50)], "next": None}
    assert _api_total_count(data) == 81


def test_analyze_pagination_uses_api_count_not_page_length():
    data = {"count": 81, "results": [{"id": i} for i in range(50)], "next": "http://x?page=2"}
    info = _analyze_pagination(data)
    assert info["total_count"] == 81
    assert info["actual_count"] == 50
    assert info["has_more"] is True
