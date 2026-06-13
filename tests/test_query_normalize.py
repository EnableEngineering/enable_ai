from enable_ai.query_utils import strip_quotes_for_matching
from enable_ai.filters import inject_semantic_filters


def test_strip_quotes_for_matching_only():
    raw = "how many service orders are in 'new' stage?"
    stripped = strip_quotes_for_matching(raw)
    assert "'" not in stripped
    assert "new" in stripped
    assert "with status" not in stripped.lower()


def test_semantic_filter_matches_quoted_token_via_hints():
    hints = {
        "service-orders": {
            "status__name": {
                "values": ["New", "In Progress"],
                "synonyms": {"new": "New"},
            },
        },
    }
    query = strip_quotes_for_matching("how many service orders are in 'new' stage?")
    filters = inject_semantic_filters({}, "service-orders", query, hints)
    assert filters["status__name"]["value"] == "New"
