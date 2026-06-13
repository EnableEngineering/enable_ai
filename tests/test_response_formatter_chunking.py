"""Tests for chunked LLM summarisation in ResponseFormatter.

All LLM calls are mocked so no OPENAI_API_KEY is required.
"""

from unittest.mock import MagicMock, patch, call
import json
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enable_ai.response_formatter import ResponseFormatter
from enable_ai import constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_formatter() -> ResponseFormatter:
    with patch("enable_ai.response_formatter.get_openai_client"):
        fmt = ResponseFormatter.__new__(ResponseFormatter)
        fmt.model = "gpt-4o-mini"
        fmt.client = MagicMock()
        return fmt


def _large_items(n: int, chars_per_item: int = 400) -> list:
    """Return n dicts that each serialise to ~chars_per_item characters."""
    pad = "x" * (chars_per_item - 60)
    return [
        {"id": i, "name": f"Item {i}", "description": pad}
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# _split_data_into_chunks
# ---------------------------------------------------------------------------

class TestSplitDataIntoChunks:
    def test_all_fit_in_one_chunk(self):
        fmt = _make_formatter()
        data = [{"id": 1}, {"id": 2}]
        chunks = fmt._split_data_into_chunks(data, max_chars=10_000)
        assert len(chunks) == 1
        assert chunks[0] == data

    def test_splits_when_over_budget(self):
        fmt = _make_formatter()
        # Each item is ~400 chars; budget is 600 → at most 1 per chunk
        items = _large_items(5, chars_per_item=400)
        chunks = fmt._split_data_into_chunks(items, max_chars=600)
        # Every chunk must contain at least one item
        assert len(chunks) == 5
        # All items are present
        reconstructed = [item for chunk in chunks for item in chunk]
        assert reconstructed == items

    def test_oversized_single_item_gets_its_own_chunk(self):
        fmt = _make_formatter()
        big = {"id": 1, "blob": "z" * 2000}
        small = {"id": 2}
        chunks = fmt._split_data_into_chunks([big, small], max_chars=500)
        # big exceeds budget → its own chunk; small follows in another
        assert len(chunks) == 2
        assert chunks[0] == [big]
        assert chunks[1] == [small]

    def test_empty_input(self):
        fmt = _make_formatter()
        assert fmt._split_data_into_chunks([], max_chars=1000) == []


# ---------------------------------------------------------------------------
# _format_concise — single call path
# ---------------------------------------------------------------------------

class TestFormatConciseSingleCall:
    def test_small_data_uses_single_llm_call(self):
        fmt = _make_formatter()
        fmt.client.chat_completion.return_value = "Found 2 items: A, B."

        data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        result = fmt._format_concise(data, "list items", {"display_field": "name"})

        assert result["format"] == "concise"
        assert result["summary"] == "Found 2 items: A, B."
        assert fmt.client.chat_completion.call_count == 1


# ---------------------------------------------------------------------------
# _format_concise — chunked path
# ---------------------------------------------------------------------------

class TestFormatConciseChunked:
    def test_large_data_triggers_multiple_llm_calls_and_merge(self):
        fmt = _make_formatter()
        # Build items large enough to exceed the current budget
        budget = constants.LLM_DATA_PREVIEW_1000
        items = _large_items(30, chars_per_item=budget // 4)
        full_json_size = len(json.dumps(items, default=str))
        assert full_json_size > budget, (
            "Test setup: items must exceed the budget to exercise the chunked path"
        )

        # Provide enough responses for up to 30 chunk calls + 1 merge call
        fmt.client.chat_completion.side_effect = [f"Batch {i} summary" for i in range(31)] + ["Final merged summary"]

        result = fmt._format_concise(items, "show me all items", {"display_field": "name"})

        assert result["format"] == "concise"
        # More than one LLM call must have been made (chunk summaries + merge)
        n_calls = fmt.client.chat_completion.call_count
        assert n_calls >= 2
        # The summary is the final call's return value (the merge call)
        assert result["summary"] == f"Batch {n_calls - 1} summary"

    def test_chunked_path_covers_all_items(self):
        """Every item must appear in some chunk — none silently dropped."""
        fmt = _make_formatter()
        items = _large_items(15, chars_per_item=500)
        chunks_seen: list = []

        def capture_call(messages, **kwargs):
            # Extract the data JSON from the prompt
            prompt = messages[0]["content"]
            chunks_seen.append(prompt)
            return "chunk summary"

        fmt.client.chat_completion.side_effect = capture_call

        fmt._format_concise(items, "list all", {"display_field": "name"})

        # Reconstruct items seen across all prompts
        ids_seen = set()
        for prompt in chunks_seen:
            for item in items:
                if f'"id": {item["id"]}' in prompt:
                    ids_seen.add(item["id"])

        all_ids = {item["id"] for item in items}
        assert ids_seen == all_ids, f"Missing items in LLM calls: {all_ids - ids_seen}"


# ---------------------------------------------------------------------------
# _format_detailed — chunked path
# ---------------------------------------------------------------------------

class TestFormatDetailedChunked:
    def test_large_data_triggers_chunked_detailed(self):
        fmt = _make_formatter()
        # Build items large enough to exceed the current detailed budget
        budget = constants.LLM_DATA_PREVIEW_2000
        items = _large_items(20, chars_per_item=budget // 4)
        full_json_size = len(json.dumps(items, default=str))
        assert full_json_size > budget, (
            "Test setup: items must exceed LLM_DATA_PREVIEW_2000"
        )

        fmt.client.chat_completion.side_effect = [
            f"Section {i}" for i in range(20)
        ] + ["Merged detailed breakdown"]

        result = fmt._format_detailed(items, "explain all items", {"count": len(items)})

        assert result["format"] == "detailed"
        n_calls = fmt.client.chat_completion.call_count
        assert n_calls >= 2
        # Last call is the merge → its return value is used
        assert result["formatted"] == f"Section {n_calls - 1}"

    def test_small_data_single_call(self):
        fmt = _make_formatter()
        fmt.client.chat_completion.return_value = "Item 1 detail: ..."

        result = fmt._format_detailed(
            [{"id": 1, "name": "A"}], "explain item", {"count": 1}
        )

        assert result["format"] == "detailed"
        assert fmt.client.chat_completion.call_count == 1
