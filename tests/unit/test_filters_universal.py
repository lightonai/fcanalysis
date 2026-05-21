"""Unit tests for universal filters in fcanalysis.loaders.base.

These exercise both the public API (apply_filters) and the predicate
functions that implement each drop reason, using synthetic samples so
filters are isolated from dataset loading.
"""

from fcanalysis.loaders.base import (
    FilterConfig,
    _has_unbalanced_cardinality,
    _has_unparseable_arguments,
    apply_filters,
    strip_thinking_from_sample,
)

from tests.helpers import assistant, call, func, sample, tool_response, user


class TestStripThinking:
    def test_removes_think_tag(self) -> None:
        s = sample(messages=[assistant(content="<think>hidden</think>visible")])
        strip_thinking_from_sample(s)
        assert s.messages[0]["content"] == "visible"

    def test_removes_reasoning_tag(self) -> None:
        s = sample(messages=[assistant(content="<reasoning>r</reasoning>after")])
        strip_thinking_from_sample(s)
        assert s.messages[0]["content"] == "after"

    def test_removes_both_kinds(self) -> None:
        s = sample(
            messages=[
                assistant(content="<think>a</think>mid<reasoning>b</reasoning>end")
            ]
        )
        strip_thinking_from_sample(s)
        assert s.messages[0]["content"] == "midend"

    def test_removes_reasoning_content_field(self) -> None:
        s = sample(messages=[assistant(content="x", reasoning_content="hidden")])
        strip_thinking_from_sample(s)
        assert "reasoning_content" not in s.messages[0]

    def test_empty_content_becomes_none(self) -> None:
        s = sample(messages=[assistant(content="<think>only</think>")])
        strip_thinking_from_sample(s)
        assert s.messages[0]["content"] is None

    def test_preserves_user_messages(self) -> None:
        s = sample(messages=[user("<think>x</think>visible")])
        strip_thinking_from_sample(s)
        assert s.messages[0]["content"] == "<think>x</think>visible"

    def test_unchanged_when_clean(self) -> None:
        s = sample(messages=[assistant(content="plain")])
        strip_thinking_from_sample(s)
        assert s.messages[0]["content"] == "plain"

    def test_multiline_think(self) -> None:
        s = sample(messages=[assistant(content="<think>line1\nline2</think>tail")])
        strip_thinking_from_sample(s)
        assert s.messages[0]["content"] == "tail"

    def test_assistant_with_no_content(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call()])])
        strip_thinking_from_sample(s)
        assert "content" not in s.messages[0] or s.messages[0].get("content") is None


class TestHasUnparseableArguments:
    def test_valid_json_object(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments='{"x":1}')])])
        assert _has_unparseable_arguments(s) is False

    def test_invalid_json(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments="{not json")])])
        assert _has_unparseable_arguments(s) is True

    def test_empty_string_is_invalid(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(arguments="")])])
        assert _has_unparseable_arguments(s) is True

    def test_no_tool_calls(self) -> None:
        s = sample(messages=[assistant(content="hi")])
        assert _has_unparseable_arguments(s) is False

    def test_non_string_arguments_skipped(self) -> None:
        tc = call()
        tc["function"]["arguments"] = {"x": 1}
        s = sample(messages=[assistant(tool_calls=[tc])])
        assert _has_unparseable_arguments(s) is False

    def test_only_first_unparseable_triggers(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[
                        call(call_id="a", arguments='{"x":1}'),
                        call(call_id="b", arguments="{garbage"),
                    ]
                )
            ]
        )
        assert _has_unparseable_arguments(s) is True


class TestHasUnbalancedCardinality:
    def test_single_balanced_pair(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call()]),
                tool_response(),
            ]
        )
        assert _has_unbalanced_cardinality(s) is False

    def test_more_calls_than_responses(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[
                        call(call_id="a"),
                        call(name="g", call_id="b"),
                    ]
                ),
                tool_response(tool_call_id="a"),
            ]
        )
        assert _has_unbalanced_cardinality(s) is True

    def test_calls_with_no_responses(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call()])])
        assert _has_unbalanced_cardinality(s) is True

    def test_multi_turn_all_balanced(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call(call_id="a")]),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[
                        call(call_id="b"),
                        call(name="g", call_id="c"),
                    ]
                ),
                tool_response(tool_call_id="b"),
                tool_response(content="r2", tool_call_id="c"),
            ]
        )
        assert _has_unbalanced_cardinality(s) is False

    def test_multi_turn_second_unbalanced(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call(call_id="a")]),
                tool_response(tool_call_id="a"),
                assistant(
                    tool_calls=[
                        call(call_id="b"),
                        call(name="g", call_id="c"),
                    ]
                ),
                tool_response(tool_call_id="b"),
            ]
        )
        assert _has_unbalanced_cardinality(s) is True

    def test_no_tool_calls(self) -> None:
        s = sample(messages=[user("hi"), assistant(content="hello")])
        assert _has_unbalanced_cardinality(s) is False


class TestApplyFilters:
    def test_empty_input(self) -> None:
        kept, reasons = apply_filters([], FilterConfig())
        assert kept == []
        assert reasons == {}

    def test_all_filters_off_keeps_everything(self) -> None:
        bad = sample(messages=[assistant(tool_calls=[call(arguments="not json")])])
        good = sample(messages=[assistant(content="hi")])
        kept, reasons = apply_filters([good, bad], FilterConfig())
        assert len(kept) == 2
        assert reasons == {}

    def test_strip_thinking_transforms_in_place(self) -> None:
        s = sample(messages=[assistant(content="<think>x</think>visible")])
        kept, _ = apply_filters([s], FilterConfig(strip_thinking=True))
        assert kept[0].messages[0]["content"] == "visible"

    def test_require_parseable_arguments_drops_bad(self) -> None:
        bad = sample(messages=[assistant(tool_calls=[call(arguments="{")])])
        good = sample(messages=[assistant(tool_calls=[call(arguments="{}")])])
        kept, reasons = apply_filters(
            [good, bad], FilterConfig(require_parseable_arguments=True)
        )
        assert len(kept) == 1
        assert kept[0] is good
        assert reasons == {"unparseable_arguments": 1}

    def test_require_balanced_cardinality_drops_unbalanced(self) -> None:
        bad = sample(messages=[assistant(tool_calls=[call()])])
        good = sample(messages=[assistant(tool_calls=[call()]), tool_response()])
        kept, reasons = apply_filters(
            [good, bad], FilterConfig(require_balanced_cardinality=True)
        )
        assert len(kept) == 1
        assert kept[0] is good
        assert reasons == {"unbalanced_cardinality": 1}

    def test_require_defined_functions_drops_unknown_call(self) -> None:
        bad = sample(
            messages=[assistant(tool_calls=[call(name="undefined")])],
            tools=[func(name="known")],
        )
        good = sample(
            messages=[assistant(tool_calls=[call(name="known")])],
            tools=[func(name="known")],
        )
        kept, reasons = apply_filters(
            [good, bad], FilterConfig(require_defined_functions=True)
        )
        assert len(kept) == 1
        assert kept[0] is good
        assert reasons == {"undefined_function_calls": 1}

    def test_require_valid_arguments_drops_schema_violations(self) -> None:
        params = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        missing_required = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments="{}")])],
            tools=[func(name="f", parameters=params)],
        )
        valid = sample(
            messages=[
                assistant(tool_calls=[call(name="f", arguments='{"x":"v"}')]),
                tool_response(),
            ],
            tools=[func(name="f", parameters=params)],
        )
        kept, reasons = apply_filters(
            [valid, missing_required],
            FilterConfig(require_valid_arguments=True),
        )
        assert len(kept) == 1
        assert kept[0] is valid
        assert reasons == {"invalid_arguments": 1}

    def test_overlapping_drops_counted_each_but_removed_once(self) -> None:
        # Same sample fails unparseable AND unbalanced (no tool response)
        bad = sample(messages=[assistant(tool_calls=[call(arguments="{")])])
        kept, reasons = apply_filters(
            [bad],
            FilterConfig(
                require_parseable_arguments=True,
                require_balanced_cardinality=True,
            ),
        )
        assert kept == []
        assert reasons == {
            "unparseable_arguments": 1,
            "unbalanced_cardinality": 1,
        }

    def test_strip_thinking_runs_before_drop_filters(self) -> None:
        s = sample(
            messages=[
                assistant(
                    content="<think>z</think>v",
                    tool_calls=[call(arguments="{}")],
                ),
                tool_response(),
            ]
        )
        kept, reasons = apply_filters(
            [s],
            FilterConfig(
                strip_thinking=True,
                require_parseable_arguments=True,
                require_balanced_cardinality=True,
            ),
        )
        assert len(kept) == 1
        assert kept[0].messages[0]["content"] == "v"
        assert reasons == {}

    def test_drop_reasons_omit_zero_counts(self) -> None:
        good = sample(
            messages=[
                assistant(tool_calls=[call(name="f", arguments="{}")]),
                tool_response(),
            ],
            tools=[func(name="f")],
        )
        _, reasons = apply_filters(
            [good],
            FilterConfig(
                require_parseable_arguments=True,
                require_balanced_cardinality=True,
                require_defined_functions=True,
                require_valid_arguments=True,
            ),
        )
        assert reasons == {}
