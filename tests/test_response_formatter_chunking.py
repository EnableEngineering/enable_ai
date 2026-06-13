"""Tests for chunked LLM summarisation in ResponseFormatter.

All LLM calls are mocked so no OPENAI_API_KEY is required.
"""

import json
from typing import Tuple
from unittest.mock import MagicMock, patch

from enable_ai import constants
from enable_ai.response_formatter import ResponseFormatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_formatter() -> Tuple[ResponseFormatter, MagicMock]:
    mock_client = MagicMock()
    with patch("enable_ai.response_formatter.get_openai_client"):
        fmt = ResponseFormatter.__new__(ResponseFormatter)
        fmt.model = "gpt-4o-mini"
        fmt.client = mock_client
        return fmt, mock_client


def _large_items(n: int, chars_per_item: int = 400) -> list:
    """Return n dicts that each serialise to ~chars_per_item characters."""
    pad = "x" * max(chars_per_item - 60, 1)
    return [
        {"id": i, "name": f"Item {i}", "description": pad}
        for i in range(1, n + 1)
    ]


def _items_exceeding_budget(n: int, budget: int) -> list:
    """Build n items whose combined JSON exceeds *budget*."""
    per_item = max(budget // 3, 200)
    items = _large_items(n, chars_per_item=per_item)
    assert len(json.dumps(items, indent=2, default=str)) > budget
    return items


# ---------------------------------------------------------------------------
# _split_data_into_chunks
# ---------------------------------------------------------------------------

class TestSplitDataIntoChunks:
    def test_all_fit_in_one_chunk(self):
        fmt, _ = _make_formatter()
        data = [{"id": 1}, {"id": 2}]
        chunks = fmt._split_data_into_chunks(data, max_chars=10_000)
        assert len(chunks) == 1
        assert chunks[0] == data

    def test_splits_when_over_budget(self):
        fmt, _ = _make_formatter()
        items = _large_items(5, chars_per_item=400)
        chunks = fmt._split_data_into_chunks(items, max_chars=600)
        reconstructed = [item for chunk in chunks for item in chunk]
        assert reconstructed == items
        assert len(chunks) >= 2

    def test_each_chunk_respects_budget(self):
        """Every chunk must serialise to ≤ max_chars (same format as prompts)."""
        fmt, _ = _make_formatter()
        budget = 600
        items = _large_items(8, chars_per_item=350)
        chunks = fmt._split_data_into_chunks(items, budget)
        for chunk in chunks:
            size = fmt._serialized_chunk_size(chunk)
            assert size <= budget or len(chunk) == 1, (
                f"Chunk of {len(chunk)} item(s) is {size} chars, budget is {budget}"
            )

    def test_oversized_single_item_gets_its_own_chunk(self):
        fmt, _ = _make_formatter()
        big = {"id": 1, "blob": "z" * 2000}
        small = {"id": 2}
        chunks = fmt._split_data_into_chunks([big, small], max_chars=500)
        assert len(chunks) == 2
        assert chunks[0] == [big]
        assert chunks[1] == [small]

    def test_empty_input(self):
        fmt, _ = _make_formatter()
        assert fmt._split_data_into_chunks([], max_chars=1000) == []


# ---------------------------------------------------------------------------
# _format_concise
# ---------------------------------------------------------------------------

class TestFormatConciseSingleCall:
    def test_small_data_uses_single_llm_call(self):
        fmt, mock_client = _make_formatter()
        mock_client.chat_completion.return_value = "Found 2 items: A, B."

        data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        result = fmt._format_concise(data, "list items", {"display_field": "name"})

        assert result["format"] == "concise"
        assert result["summary"] == "Found 2 items: A, B."
        assert mock_client.chat_completion.call_count == 1


class TestFormatConciseChunked:
    def test_large_data_calls_merge_with_final_result(self):
        fmt, mock_client = _make_formatter()
        budget = constants.LLM_DATA_PREVIEW_1000
        items = _items_exceeding_budget(30, budget)
        n_chunks = len(fmt._split_data_into_chunks(items, budget))

        def respond(messages, **kwargs):
            prompt = messages[0]["content"]
            if "Final combined response" in prompt:
                return "Final merged summary"
            return "chunk summary"

        mock_client.chat_completion.side_effect = respond

        result = fmt._format_concise(items, "show me all items", {"display_field": "name"})

        assert result["format"] == "concise"
        assert result["summary"] == "Final merged summary"
        assert mock_client.chat_completion.call_count == n_chunks + 1

    def test_all_items_sent_to_llm_none_dropped(self):
        """Every item id must appear in at least one LLM prompt."""
        fmt, mock_client = _make_formatter()
        items = _items_exceeding_budget(15, constants.LLM_DATA_PREVIEW_1000)
        prompts: list = []

        def capture(messages, **kwargs):
            prompts.append(messages[0]["content"])
            return "chunk summary"

        mock_client.chat_completion.side_effect = capture
        fmt._format_concise(items, "list all", {"display_field": "name"})

        ids_seen = set()
        for prompt in prompts:
            for item in items:
                if f'"id": {item["id"]}' in prompt:
                    ids_seen.add(item["id"])

        assert ids_seen == {item["id"] for item in items}

    def test_more_than_sample_medium_limit_all_processed(self):
        """Regression: _format_concise must not slice to LLM_DATA_SAMPLE_MEDIUM (50)."""
        fmt, mock_client = _make_formatter()
        # 60 small items — under char budget individually but > 50 count
        # Force chunked path with large per-item payload
        budget = constants.LLM_DATA_PREVIEW_1000
        items = _items_exceeding_budget(60, budget)
        assert len(items) > constants.LLM_DATA_SAMPLE_MEDIUM

        prompts: list = []

        def capture(messages, **kwargs):
            prompts.append(messages[0]["content"])
            return "ok"

        mock_client.chat_completion.side_effect = capture
        fmt._format_concise(items, "list all", {})

        ids_seen = set()
        for prompt in prompts:
            for item in items:
                if f'"id": {item["id"]}' in prompt:
                    ids_seen.add(item["id"])

        missing = {item["id"] for item in items} - ids_seen
        assert not missing, f"Items beyond sample-medium limit were dropped: {missing}"

    def test_item_ranges_in_chunk_prompts_are_correct(self):
        """Batch prompts must report correct 1-based item ranges."""
        fmt, mock_client = _make_formatter()
        items = _large_items(5, chars_per_item=400)
        chunks = fmt._split_data_into_chunks(items, max_chars=600)
        assert len(chunks) >= 2

        prompts: list = []

        def capture(messages, **kwargs):
            prompts.append(messages[0]["content"])
            return "ok"

        mock_client.chat_completion.side_effect = capture

        item_offset = 0
        for i, chunk in enumerate(chunks):
            fmt._summarize_one_chunk(
                chunk, "list all", len(items), item_offset, i, len(chunks)
            )
            item_offset += len(chunk)

        assert "items 1–" in prompts[0] or "items 1-" in prompts[0]
        last_end = len(items)
        assert f"items {last_end}–{last_end}" in prompts[-1] or f"items {last_end}-{last_end}" in prompts[-1]


# ---------------------------------------------------------------------------
# _format_detailed — chunked path
# ---------------------------------------------------------------------------

class TestFormatDetailedChunked:
    def test_large_data_triggers_chunked_detailed_with_merge(self):
        fmt, mock_client = _make_formatter()
        budget = constants.LLM_DATA_PREVIEW_2000
        items = _items_exceeding_budget(20, budget)

        n_chunks = len(fmt._split_data_into_chunks(items, budget))
        section_responses = [f"Section {i}" for i in range(n_chunks)]
        mock_client.chat_completion.side_effect = section_responses + ["Merged detailed breakdown"]

        result = fmt._format_detailed(items, "explain all items", {"count": len(items)})

        assert result["format"] == "detailed"
        assert result["formatted"] == "Merged detailed breakdown"
        assert mock_client.chat_completion.call_count == n_chunks + 1

    def test_small_data_single_call(self):
        fmt, mock_client = _make_formatter()
        mock_client.chat_completion.return_value = "Item 1 detail: ..."

        result = fmt._format_detailed(
            [{"id": 1, "name": "A"}], "explain item", {"count": 1}
        )

        assert result["format"] == "detailed"
        assert mock_client.chat_completion.call_count == 1


# ---------------------------------------------------------------------------
# _generate_intelligent_response — main production path
# ---------------------------------------------------------------------------

class TestGenerateIntelligentResponseChunked:
    def test_large_dataset_uses_chunked_path_not_truncation(self):
        """Main format_response path must not silently drop items beyond 25."""
        fmt, mock_client = _make_formatter()
        budget = constants.LLM_DATA_PREVIEW_LARGE
        items = _items_exceeding_budget(30, budget)
        n_chunks = len(fmt._split_data_into_chunks(items, budget))

        def respond(messages, **kwargs):
            prompt = messages[0]["content"]
            if "Final combined response" in prompt:
                return "Merged response"
            return "chunk"

        mock_client.chat_completion.side_effect = respond

        with patch.object(constants, "LIST_FORMAT_TABLE", 0):
            result = fmt._generate_intelligent_response(
                items, "list all service orders", context={}
            )

        assert result["summary"] == "Merged response"
        assert mock_client.chat_completion.call_count == n_chunks + 1

    def test_small_dataset_single_call_with_full_data(self):
        fmt, mock_client = _make_formatter()
        mock_client.chat_completion.return_value = (
            '{"format": "bullets", "response": "Found 3 items."}'
        )

        items = [{"id": i, "name": f"Item {i}"} for i in range(1, 4)]
        with patch.object(constants, "LIST_FORMAT_TABLE", 0):
            result = fmt._generate_intelligent_response(items, "list items", context={})

        assert mock_client.chat_completion.call_count == 1
        prompt = mock_client.chat_completion.call_args[1]["messages"][0]["content"]
        # All three items must be in the single prompt
        for i in range(1, 4):
            assert f'"id": {i}' in prompt

    def test_thirty_items_not_truncated_to_twenty_five(self):
        """Regression: old code sent only first 25 of 30+ items."""
        fmt, mock_client = _make_formatter()
        items = [{"id": i, "name": f"N{i}"} for i in range(1, 31)]
        data_json = json.dumps(items, indent=2, default=str)

        if len(data_json) <= constants.LLM_DATA_PREVIEW_LARGE:
            mock_client.chat_completion.return_value = (
                '{"format": "text", "response": "ok"}'
            )
            with patch.object(constants, "LIST_FORMAT_TABLE", 0):
                fmt._generate_intelligent_response(items, "list all", context={})
            prompt = mock_client.chat_completion.call_args[1]["messages"][0]["content"]
            assert '"id": 30' in prompt, "Item 30 was truncated from single-call prompt"
            assert '"id": 26' in prompt, "Items 26-30 were truncated (old 25-item limit)"
