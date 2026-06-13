"""Golden-query eval for semantic filters and param validator (no API/LLM needed)."""

import json
from pathlib import Path

from enable_ai.filters import apply_semantic_filters, validate_parsed

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden_queries.json"


def _load_schema():
    with open(FIXTURES / "minimal_schema.json") as f:
        return json.load(f)


def _filter_value(filters, key):
    val = (filters or {}).get(key)
    if isinstance(val, dict):
        return val.get("value")
    return val


def test_golden_queries():
    schema = _load_schema()
    cases = json.loads(GOLDEN.read_text())
    failures = []

    for case in cases:
        parsed = apply_semantic_filters(dict(case["parsed"]), case["query"], schema)
        parsed, _warnings, clarification = validate_parsed(parsed, schema, query=case["query"])

        if case.get("expect_filters"):
            for key, expected in case["expect_filters"].items():
                actual = _filter_value(parsed.get("filters"), key)
                if actual != expected:
                    failures.append(
                        f"{case['id']}: expected filters[{key}]={expected!r}, got {actual!r}"
                    )

        if case.get("expect_resource"):
            if parsed.get("resource") != case["expect_resource"]:
                failures.append(
                    f"{case['id']}: expected resource={case['expect_resource']!r}, "
                    f"got {parsed.get('resource')!r}"
                )

        if case.get("expect_limit"):
            if parsed.get("limit") != case["expect_limit"]:
                failures.append(
                    f"{case['id']}: expected limit={case['expect_limit']}, got {parsed.get('limit')}"
                )

        if case.get("expect_sort_order"):
            sort = parsed.get("sort") or {}
            if sort.get("order") != case["expect_sort_order"]:
                failures.append(f"{case['id']}: sort order mismatch")

        if case.get("expect_clarification"):
            if not clarification and parsed.get("question_type") != "needs_clarification":
                failures.append(f"{case['id']}: expected clarification, got none")

    assert not failures, "Golden query failures:\n" + "\n".join(failures)
