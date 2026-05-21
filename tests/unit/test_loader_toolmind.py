"""Unit tests for toolmind loader-specific predicates and transforms."""

from fcanalysis.loaders.toolmind import (
    _filter_required_names,
    _has_balanced_cardinality,
    _has_consecutive_text_assistant,
    _has_non_object_arguments,
    _merge_split_assistant_messages_with_counts,
    _strip_think_tool_from_sample_with_counts,
)

from tests.helpers import assistant, call, func, sample, tool_response, user


class TestFilterRequiredNames:
    def test_all_present(self) -> None:
        assert _filter_required_names(["a", "b"], {"a": {}, "b": {}}) == ["a", "b"]

    def test_drops_missing(self) -> None:
        assert _filter_required_names(["a", "missing"], {"a": {}}) == ["a"]

    def test_dedupes_repeats(self) -> None:
        assert _filter_required_names(["a", "a", "b"], {"a": {}, "b": {}}) == ["a", "b"]

    def test_preserves_order(self) -> None:
        assert _filter_required_names(["b", "a"], {"a": {}, "b": {}}) == ["b", "a"]

    def test_empty_required(self) -> None:
        assert _filter_required_names([], {"a": {}}) == []

    def test_empty_properties(self) -> None:
        assert _filter_required_names(["a"], {}) == []


class TestHasNonObjectArguments:
    def test_dict_arguments(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments='{"x":1}')])])
        assert _has_non_object_arguments(s) is False

    def test_list_arguments(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments="[1,2]")])])
        assert _has_non_object_arguments(s) is True

    def test_int_arguments(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments="42")])])
        assert _has_non_object_arguments(s) is True

    def test_string_arguments(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments='"text"')])])
        assert _has_non_object_arguments(s) is True

    def test_null_arguments(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments="null")])])
        assert _has_non_object_arguments(s) is True

    def test_unparseable_arguments_ignored(self) -> None:
        # Unparseable args are a separate concern (require_parseable_arguments).
        s = sample(messages=[assistant(tool_calls=[call(arguments="{not json")])])
        assert _has_non_object_arguments(s) is False

    def test_no_tool_calls(self) -> None:
        s = sample(messages=[assistant(content="hi")])
        assert _has_non_object_arguments(s) is False

    def test_one_bad_among_good(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[
                        call(call_id="1", arguments='{"x":1}'),
                        call(call_id="2", arguments="[1,2]"),
                    ]
                )
            ]
        )
        assert _has_non_object_arguments(s) is True


class TestHasConsecutiveTextAssistant:
    def test_two_text_in_a_row(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="a"),
                assistant(content="b"),
            ]
        )
        assert _has_consecutive_text_assistant(s) is True

    def test_text_then_tc(self) -> None:
        s = sample(
            messages=[
                assistant(content="a"),
                assistant(tool_calls=[call()]),
            ]
        )
        assert _has_consecutive_text_assistant(s) is False

    def test_no_assistant(self) -> None:
        assert _has_consecutive_text_assistant(sample(messages=[user("u")])) is False


class TestStripThinkToolFromSampleWithCounts:
    def test_no_think_tool(self) -> None:
        s = sample(tools=[func("real")], messages=[assistant(content="hi")])
        out, ndefs, ncalls, nobs = _strip_think_tool_from_sample_with_counts(s)
        assert (ndefs, ncalls, nobs) == (0, 0, 0)
        assert out is s

    def test_strips_solo_think_call(self) -> None:
        s = sample(
            tools=[func("think")],
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="think", call_id="t")]),
                tool_response(content="", tool_call_id="t"),
                assistant(content="reply"),
            ],
        )
        out, ndefs, ncalls, nobs = _strip_think_tool_from_sample_with_counts(s)
        assert ndefs == 1
        assert ncalls == 1
        assert nobs == 1
        # Only user + final assistant should remain.
        assert [m["role"] for m in out.messages] == ["user", "assistant"]

    def test_strips_think_call_keeps_real_calls(self) -> None:
        s = sample(
            tools=[func("think"), func("real")],
            messages=[
                user("u"),
                assistant(
                    content="thinking",
                    tool_calls=[
                        call(name="think", call_id="t"),
                        call(name="real", call_id="r"),
                    ],
                ),
                tool_response(content="", tool_call_id="t"),
                tool_response(content="r-result", tool_call_id="r"),
            ],
        )
        out, ndefs, ncalls, nobs = _strip_think_tool_from_sample_with_counts(s)
        assert ndefs == 1  # think tool def
        assert ncalls == 1  # the think call
        assert nobs == 1  # the empty think response
        # Assistant kept with one tool_call (the real one), think response gone.
        assistant_msg = [m for m in out.messages if m["role"] == "assistant"][0]
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "real"
        tool_msgs = [m for m in out.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "r-result"

    def test_all_calls_are_think_keeps_text_assistant(self) -> None:
        s = sample(
            tools=[func("think")],
            messages=[
                assistant(
                    content="visible", tool_calls=[call(name="think", call_id="t")]
                ),
                tool_response(content="", tool_call_id="t"),
            ],
        )
        out, _, ncalls, nobs = _strip_think_tool_from_sample_with_counts(s)
        assert ncalls == 1
        assert nobs == 1
        assert len(out.messages) == 1
        assert out.messages[0]["role"] == "assistant"
        assert out.messages[0]["content"] == "visible"
        assert "tool_calls" not in out.messages[0]

    def test_all_calls_are_think_no_content_drops_message(self) -> None:
        s = sample(
            tools=[func("think")],
            messages=[
                assistant(tool_calls=[call(name="think", call_id="t")]),
                tool_response(content="", tool_call_id="t"),
            ],
        )
        out, _, _, _ = _strip_think_tool_from_sample_with_counts(s)
        assert out.messages == []


class TestMergeSplitAssistantMessagesWithCounts:
    def test_no_merge_when_no_pattern(self) -> None:
        s = sample(messages=[user("u"), assistant(content="a")])
        out, count = _merge_split_assistant_messages_with_counts(s)
        assert count == 0
        assert out is s

    def test_merges_pair(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="thinking"),
                assistant(tool_calls=[call(name="f")]),
            ]
        )
        out, count = _merge_split_assistant_messages_with_counts(s)
        assert count == 1
        assert len(out.messages) == 2
        m = out.messages[1]
        assert m["content"] == "thinking"
        assert m["tool_calls"] == [call(name="f")]

    def test_combines_both_contents(self) -> None:
        s = sample(
            messages=[
                assistant(content="reasoning"),
                assistant(content="<think>more</think>", tool_calls=[call()]),
            ]
        )
        out, _ = _merge_split_assistant_messages_with_counts(s)
        assert out.messages[0]["content"] == "reasoning\n\n<think>more</think>"

    def test_drops_empty_content(self) -> None:
        s = sample(
            messages=[
                assistant(content=""),
                assistant(content="", tool_calls=[call()]),
            ]
        )
        out, _ = _merge_split_assistant_messages_with_counts(s)
        assert out.messages[0]["content"] is None

    def test_does_not_merge_fc_then_text(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call()]),
                assistant(content="post"),
            ]
        )
        _, count = _merge_split_assistant_messages_with_counts(s)
        assert count == 0


class TestHasBalancedCardinality:
    def test_all_balanced(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call(call_id="1")]),
                tool_response(tool_call_id="1"),
                assistant(tool_calls=[call(call_id="2"), call(call_id="3", name="g")]),
                tool_response(tool_call_id="2"),
                tool_response(content="r3", tool_call_id="3"),
            ]
        )
        assert _has_balanced_cardinality(s) is True

    def test_too_few_responses(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call(call_id="1"), call(call_id="2", name="g")]),
                tool_response(tool_call_id="1"),
            ]
        )
        assert _has_balanced_cardinality(s) is False

    def test_too_many_responses(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call(call_id="1")]),
                tool_response(tool_call_id="1"),
                tool_response(content="extra", tool_call_id="1"),
            ]
        )
        assert _has_balanced_cardinality(s) is False

    def test_no_tool_calls(self) -> None:
        s = sample(messages=[user("u"), assistant(content="a")])
        assert _has_balanced_cardinality(s) is True

    def test_zero_responses_for_call(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call()])])
        assert _has_balanced_cardinality(s) is False
