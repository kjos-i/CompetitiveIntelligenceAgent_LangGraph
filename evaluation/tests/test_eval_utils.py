"""Unit tests for eval_utils pure helpers."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, ToolMessage

from eval_utils import (
    content_to_text,
    extract_all_tool_outputs,
    extract_message_text,
    extract_messages,
    join_tool_outputs,
    normalize_text,
    precision_for,
    safe_mean,
)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_strips_accents(self):
        assert normalize_text("café") == "cafe"
        assert normalize_text("naïve") == "naive"

    def test_collapses_punctuation_to_space(self):
        assert normalize_text("foo, bar; baz!") == "foo bar baz"

    def test_collapses_whitespace(self):
        assert normalize_text("foo   bar\n\nbaz") == "foo bar baz"

    def test_preserves_digits_and_percent(self):
        assert normalize_text("Score: 7.5%") == "score 7.5%"

    def test_empty_input(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# precision_for
# ---------------------------------------------------------------------------

class TestPrecisionFor:
    def test_float_format(self):
        assert precision_for(".2f") == 2
        assert precision_for(".0f") == 0
        assert precision_for(".4f") == 4

    def test_percent_format_adds_two(self):
        # .2% displays 0.8571 as "85.71%" — needs 4 stored decimals.
        assert precision_for(".2%") == 4
        assert precision_for(".0%") == 2

    def test_integer_format(self):
        assert precision_for("d") == 0

    def test_unknown_format_defaults_to_two(self):
        assert precision_for("garbage") == 2
        assert precision_for("") == 2


# ---------------------------------------------------------------------------
# safe_mean
# ---------------------------------------------------------------------------

class TestSafeMean:
    def test_basic_mean(self):
        assert safe_mean([1.0, 2.0, 3.0]) == 2.0

    def test_skips_none(self):
        assert safe_mean([1.0, None, 3.0]) == 2.0

    def test_all_none_returns_none(self):
        assert safe_mean([None, None]) is None

    def test_empty_list_returns_none(self):
        assert safe_mean([]) is None

    def test_precision_argument(self):
        # 1/3 = 0.33333… — precision controls rounding of the result.
        assert safe_mean([1, 1, 1, 1, 0], precision=2) == 0.8
        assert safe_mean([1, 1, 1, 1, 1, 1, 0], precision=4) == 0.8571
        assert safe_mean([1, 1, 1, 1, 1, 1, 0], precision=2) == 0.86

    def test_handles_int_and_float_mix(self):
        assert safe_mean([1, 2.0, 3]) == 2.0


# ---------------------------------------------------------------------------
# extract_message_text
# ---------------------------------------------------------------------------

class TestExtractMessageText:
    def test_string_content(self):
        assert extract_message_text("hello") == "hello"

    def test_strips_whitespace(self):
        assert extract_message_text("  hello  ") == "hello"

    def test_block_list_with_text(self):
        content = [{"text": "first"}, {"text": "second"}]
        assert extract_message_text(content) == "first\nsecond"

    def test_block_list_with_content_key(self):
        # Some providers use "content" instead of "text".
        assert extract_message_text([{"content": "hi"}]) == "hi"

    def test_block_list_with_strings(self):
        assert extract_message_text(["a", "b"]) == "a\nb"

    def test_mixed_blocks_skip_unparseable(self):
        # Blocks without text/content should be silently dropped.
        content = [{"text": "kept"}, {"image": "ignored"}]
        assert extract_message_text(content) == "kept"

    def test_empty_input(self):
        assert extract_message_text("") == ""
        assert extract_message_text([]) == ""

    def test_falls_back_to_str(self):
        assert extract_message_text(42) == "42"


# ---------------------------------------------------------------------------
# extract_all_tool_outputs
# ---------------------------------------------------------------------------

class TestExtractAllToolOutputs:
    def test_collects_only_tool_messages(self):
        messages = [
            HumanMessage(content="ignored"),
            ToolMessage(content="tool-1", tool_call_id="t1"),
            ToolMessage(content="tool-2", tool_call_id="t2"),
        ]
        assert extract_all_tool_outputs(messages) == ["tool-1", "tool-2"]

    def test_skips_empty_content(self):
        messages = [
            ToolMessage(content="", tool_call_id="t1"),
            ToolMessage(content="kept", tool_call_id="t2"),
        ]
        assert extract_all_tool_outputs(messages) == ["kept"]

    def test_serialises_non_string_content(self):
        payload = [{"url": "https://x.com"}]
        messages = [ToolMessage(content=payload, tool_call_id="t1")]
        out = extract_all_tool_outputs(messages)
        assert len(out) == 1
        # JSON-encoded so downstream regex / substring checks work.
        assert json.loads(out[0]) == payload


# ---------------------------------------------------------------------------
# content_to_text
# ---------------------------------------------------------------------------

class TestContentToText:
    def test_none_returns_empty_string(self):
        # Distinct from extract_message_text, which would str(None).
        assert content_to_text(None) == ""

    def test_string_passthrough_no_strip(self):
        assert content_to_text("  hello  ") == "  hello  "

    def test_block_list_joined_with_no_separator(self):
        # Markdown-friendly: blocks concatenate without injected newlines.
        assert content_to_text([{"text": "foo"}, {"text": "bar"}]) == "foobar"

    def test_block_list_with_content_key(self):
        assert content_to_text([{"content": "hi"}]) == "hi"

    def test_block_list_with_strings(self):
        assert content_to_text(["a", "b"]) == "ab"

    def test_block_list_falls_back_to_str_for_unknown_items(self):
        assert content_to_text([42, "x"]) == "42x"

    def test_block_list_skips_empty_dict_values(self):
        assert content_to_text([{"text": ""}, {"text": "kept"}]) == "kept"

    def test_top_level_dict_extracts_text(self):
        assert content_to_text({"text": "hi"}) == "hi"

    def test_top_level_dict_falls_back_to_content_key(self):
        assert content_to_text({"content": "hi"}) == "hi"

    def test_other_types_str_fallback(self):
        assert content_to_text(42) == "42"


# ---------------------------------------------------------------------------
# extract_messages
# ---------------------------------------------------------------------------

class TestExtractMessages:
    def test_dict_with_messages_key(self):
        msgs = [HumanMessage(content="x")]
        assert extract_messages({"messages": msgs}) == msgs

    def test_dict_without_messages_returns_empty(self):
        assert extract_messages({"other": 1}) == []

    def test_object_with_messages_attribute(self):
        class State:
            messages = [HumanMessage(content="x")]
        assert extract_messages(State()) == State.messages

    def test_object_without_messages_attribute(self):
        assert extract_messages(object()) == []

    def test_messages_value_is_none(self):
        # None should normalise to [] so callers can iterate safely.
        assert extract_messages({"messages": None}) == []


# ---------------------------------------------------------------------------
# join_tool_outputs
# ---------------------------------------------------------------------------

class TestJoinToolOutputs:
    def test_joins_with_newline(self):
        assert join_tool_outputs(["a", "b", "c"]) == "a\nb\nc"

    def test_empty_list_returns_empty_string(self):
        assert join_tool_outputs([]) == ""

    def test_single_element(self):
        assert join_tool_outputs(["only"]) == "only"
