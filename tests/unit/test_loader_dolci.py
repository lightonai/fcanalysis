"""Unit tests for dolci loader-specific predicates and transforms."""

from fcanalysis.loaders.dolci import (
    _has_conflicting_duplicate_tools,
    _has_consecutive_text_text_assistant,
    _merge_text_fc_assistant_messages,
)

from tests.helpers import assistant, call, func, sample, tool_response, user


class TestHasConflictingDuplicateTools:
    def test_no_tools(self) -> None:
        assert _has_conflicting_duplicate_tools(sample()) is False

    def test_unique_names(self) -> None:
        s = sample(tools=[func("a"), func("b")])
        assert _has_conflicting_duplicate_tools(s) is False

    def test_same_name_same_body(self) -> None:
        # Two identical tool definitions are not "conflicting".
        s = sample(tools=[func("a"), func("a")])
        assert _has_conflicting_duplicate_tools(s) is False

    def test_same_name_different_descriptions(self) -> None:
        s = sample(
            tools=[
                func("a", description="first"),
                func("a", description="second"),
            ]
        )
        assert _has_conflicting_duplicate_tools(s) is True

    def test_same_name_different_parameters(self) -> None:
        s = sample(
            tools=[
                func(
                    "a",
                    parameters={
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                ),
                func(
                    "a",
                    parameters={
                        "type": "object",
                        "properties": {"y": {"type": "integer"}},
                    },
                ),
            ]
        )
        assert _has_conflicting_duplicate_tools(s) is True

    def test_mixed_conflict_and_consistent(self) -> None:
        s = sample(
            tools=[
                func("a", description="x"),
                func("b"),
                func("a", description="y"),  # conflicts with first "a"
            ]
        )
        assert _has_conflicting_duplicate_tools(s) is True


class TestHasConsecutiveTextTextAssistant:
    def test_empty(self) -> None:
        assert _has_consecutive_text_text_assistant(sample()) is False

    def test_single_assistant_text(self) -> None:
        s = sample(messages=[user("u"), assistant(content="hi")])
        assert _has_consecutive_text_text_assistant(s) is False

    def test_two_consecutive_text_assistants(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="part1"),
                assistant(content="part2"),
            ]
        )
        assert _has_consecutive_text_text_assistant(s) is True

    def test_text_then_fc_assistant(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="text"),
                assistant(tool_calls=[call()]),
            ]
        )
        assert _has_consecutive_text_text_assistant(s) is False

    def test_fc_then_text_assistant(self) -> None:
        # FC followed by text is also not consecutive-text-text (one has tc).
        s = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call()]),
                tool_response(),
                assistant(content="text"),
            ]
        )
        assert _has_consecutive_text_text_assistant(s) is False

    def test_text_then_tool_then_text(self) -> None:
        # Tool message between two text-assistant messages breaks the chain.
        s = sample(
            messages=[
                user("u"),
                assistant(content="a"),
                {"role": "tool", "content": "r", "tool_call_id": "1"},
                assistant(content="b"),
            ]
        )
        assert _has_consecutive_text_text_assistant(s) is False

    def test_three_consecutive_text_assistants(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="a"),
                assistant(content="b"),
                assistant(content="c"),
            ]
        )
        assert _has_consecutive_text_text_assistant(s) is True


class TestMergeTextFcAssistantMessages:
    def test_no_merge_when_no_pattern(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="text"),
            ]
        )
        merged, count = _merge_text_fc_assistant_messages(s)
        assert count == 0
        assert merged is s

    def test_merges_text_fc_pair(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="thinking"),
                assistant(tool_calls=[call(name="f")]),
            ]
        )
        merged, count = _merge_text_fc_assistant_messages(s)
        assert count == 1
        assert len(merged.messages) == 2
        assert merged.messages[0]["role"] == "user"
        m = merged.messages[1]
        assert m["role"] == "assistant"
        assert m["content"] == "thinking"
        assert m["tool_calls"] == [call(name="f")]

    def test_combines_both_contents_with_blank_line(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="t1"),
                assistant(content="t2", tool_calls=[call(name="f")]),
            ]
        )
        merged, _ = _merge_text_fc_assistant_messages(s)
        assert merged.messages[1]["content"] == "t1\n\nt2"

    def test_uses_second_content_when_first_empty(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content=""),
                assistant(content="t2", tool_calls=[call()]),
            ]
        )
        merged, _ = _merge_text_fc_assistant_messages(s)
        assert merged.messages[1]["content"] == "t2"

    def test_uses_first_content_when_second_empty(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="t1"),
                assistant(content="", tool_calls=[call()]),
            ]
        )
        merged, _ = _merge_text_fc_assistant_messages(s)
        assert merged.messages[1]["content"] == "t1"

    def test_content_becomes_none_when_both_empty(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content=""),
                assistant(content="", tool_calls=[call()]),
            ]
        )
        merged, _ = _merge_text_fc_assistant_messages(s)
        assert merged.messages[1]["content"] is None

    def test_does_not_merge_fc_then_text(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call()]),
                assistant(content="text"),
            ]
        )
        merged, count = _merge_text_fc_assistant_messages(s)
        assert count == 0
        assert merged is s

    def test_does_not_merge_text_then_text(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="a"),
                assistant(content="b"),
            ]
        )
        merged, count = _merge_text_fc_assistant_messages(s)
        assert count == 0

    def test_does_not_merge_across_tool_boundary(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="t1"),
                tool_response(),
                assistant(tool_calls=[call()]),
            ]
        )
        merged, count = _merge_text_fc_assistant_messages(s)
        assert count == 0

    def test_multiple_merge_pairs(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="t1"),
                assistant(tool_calls=[call(call_id="1")]),
                tool_response(tool_call_id="1"),
                user("u2"),
                assistant(content="t2"),
                assistant(tool_calls=[call(call_id="2")]),
                tool_response(tool_call_id="2"),
            ]
        )
        merged, count = _merge_text_fc_assistant_messages(s)
        assert count == 2

    def test_preserves_tools_dataset_sample_id(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="t1"),
                assistant(tool_calls=[call()]),
            ],
            tools=[func("f")],
            dataset="dolci",
            sample_id="abc",
        )
        merged, _ = _merge_text_fc_assistant_messages(s)
        assert merged.tools == s.tools
        assert merged.dataset == "dolci"
        assert merged.sample_id == "abc"
