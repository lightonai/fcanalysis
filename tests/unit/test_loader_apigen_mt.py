"""Unit tests for apigen_mt loader-specific predicates and transforms."""

from fcanalysis.loaders.apigen_mt import (
    _has_consecutive_assistant,
    _has_repeated_tool_call_streak,
    _strip_think_from_sample_with_counts,
)

from tests.helpers import assistant, call, func, sample, tool_response, user


class TestHasRepeatedToolCallStreak:
    def test_no_calls(self) -> None:
        s = sample(messages=[user("u"), assistant(content="a")])
        assert _has_repeated_tool_call_streak(s) is False

    def test_two_identical_below_min(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="a")]
                ),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="b")]
                ),
            ]
        )
        assert _has_repeated_tool_call_streak(s) is False

    def test_three_identical_meets_min(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="a")]
                ),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="b")]
                ),
                tool_response(tool_call_id="b"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="c")]
                ),
            ]
        )
        assert _has_repeated_tool_call_streak(s) is True

    def test_different_args_break_streak(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="a")]
                ),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":2}', call_id="b")]
                ),
                tool_response(tool_call_id="b"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="c")]
                ),
            ]
        )
        assert _has_repeated_tool_call_streak(s) is False

    def test_assistant_text_breaks_streak(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="a")]
                ),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="b")]
                ),
                tool_response(tool_call_id="b"),
                assistant(content="comment"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="c")]
                ),
            ]
        )
        assert _has_repeated_tool_call_streak(s) is False

    def test_user_message_breaks_streak(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="a")]
                ),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="b")]
                ),
                tool_response(tool_call_id="b"),
                user("nudge"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="c")]
                ),
            ]
        )
        assert _has_repeated_tool_call_streak(s) is False

    def test_tool_messages_do_not_break_streak(self) -> None:
        # Tool observations between repeated calls are normal and must not reset the streak.
        s = sample(
            messages=[
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="a")]
                ),
                tool_response(tool_call_id="a"),
                tool_response(content="extra", tool_call_id="a"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="b")]
                ),
                tool_response(tool_call_id="b"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="c")]
                ),
            ]
        )
        assert _has_repeated_tool_call_streak(s) is True

    def test_min_length_parameter(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="a")]
                ),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[call(name="f", arguments='{"x":1}', call_id="b")]
                ),
            ]
        )
        assert _has_repeated_tool_call_streak(s, min_length=2) is True
        assert _has_repeated_tool_call_streak(s, min_length=3) is False


class TestHasConsecutiveAssistant:
    def test_text_then_text(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="a"),
                assistant(content="b"),
            ]
        )
        assert _has_consecutive_assistant(s) is True

    def test_text_then_fc(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(content="a"),
                assistant(tool_calls=[call()]),
            ]
        )
        assert _has_consecutive_assistant(s) is False

    def test_fc_then_text_via_tool(self) -> None:
        # Standard pattern: assistant(fc) -> tool -> assistant(text). Not consecutive.
        s = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call()]),
                tool_response(),
                assistant(content="result"),
            ]
        )
        assert _has_consecutive_assistant(s) is False

    def test_no_assistants(self) -> None:
        assert _has_consecutive_assistant(sample(messages=[user("u")])) is False


class TestStripThinkFromSampleWithCounts:
    def test_no_think_tool(self) -> None:
        s = sample(
            messages=[assistant(content="hi")],
            tools=[func("real")],
        )
        out, ndefs, ncalls, nobs = _strip_think_from_sample_with_counts(s)
        assert ndefs == ncalls == nobs == 0
        assert out.tools == s.tools
        assert out.messages == s.messages

    def test_strips_think_tool_definition(self) -> None:
        s = sample(tools=[func("real"), func("think"), func("other")])
        out, ndefs, ncalls, nobs = _strip_think_from_sample_with_counts(s)
        assert ndefs == 1
        assert ncalls == 0
        assert nobs == 0
        assert [t["function"]["name"] for t in out.tools] == ["real", "other"]

    def test_strips_solo_think_call_and_response(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="think", call_id="t")]),
                tool_response(content="ok", tool_call_id="t"),
                assistant(content="done"),
            ],
            tools=[func("think")],
        )
        out, ndefs, ncalls, nobs = _strip_think_from_sample_with_counts(s)
        assert ndefs == 1
        assert ncalls == 1
        assert nobs == 1
        assert len(out.messages) == 2
        assert out.messages[0]["role"] == "user"
        assert out.messages[1]["content"] == "done"

    def test_think_call_without_response(self) -> None:
        s = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="think", call_id="t")]),
                user("next"),
            ],
            tools=[func("think")],
        )
        out, ndefs, ncalls, nobs = _strip_think_from_sample_with_counts(s)
        assert ncalls == 1
        assert nobs == 0
        # User message after the think call must survive.
        assert out.messages[-1]["role"] == "user"

    def test_multi_call_message_not_stripped(self) -> None:
        # Stripping only applies when the assistant message has exactly one
        # call AND that call is "think". Multi-call messages survive intact.
        s = sample(
            messages=[
                assistant(
                    tool_calls=[
                        call(name="think", call_id="t"),
                        call(name="real", call_id="r"),
                    ]
                ),
                tool_response(content="t-out", tool_call_id="t"),
                tool_response(content="r-out", tool_call_id="r"),
            ],
            tools=[func("think"), func("real")],
        )
        out, ndefs, ncalls, nobs = _strip_think_from_sample_with_counts(s)
        assert ndefs == 1  # only the "think" definition removed
        assert ncalls == 0
        assert nobs == 0
        # Messages untouched.
        assert len(out.messages) == 3

    def test_preserves_metadata(self) -> None:
        s = sample(dataset="apigen_mt", sample_id="42")
        out, *_ = _strip_think_from_sample_with_counts(s)
        assert out.dataset == "apigen_mt"
        assert out.sample_id == "42"
