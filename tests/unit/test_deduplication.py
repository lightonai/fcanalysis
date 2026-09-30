"""Tests for complete model-visible trajectory deduplication."""

from copy import deepcopy
from typing import cast

import pytest

from fcanalysis.deduplication import (
    DeduplicationLevel,
    DatasetDeduplicationCounts,
    EquivalenceAuditCounts,
    audit_subset_samples,
    curate_subset_samples,
    deduplicate_samples,
)
from fcanalysis.format import ConversationSample


def _tool(name: str, *, enum: list[str] | None = None) -> dict:
    value_schema: dict[str, object] = {
        "type": "string",
        "description": "A value",
    }
    if enum is not None:
        value_schema["enum"] = enum
    parameters: dict = {
        "type": "object",
        "properties": {"value": value_schema},
        "required": ["value"],
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}",
            "parameters": parameters,
        },
    }


def _call(name: str, arguments: str = "{}") -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _sample(
    sample_id: str,
    *,
    dataset: str = "a",
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    annotations: dict | None = None,
    raw: dict | None = None,
) -> ConversationSample:
    return ConversationSample(
        messages=messages
        if messages is not None
        else [
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "Done"},
        ],
        tools=tools if tools is not None else [_tool("f")],
        dataset=dataset,
        sample_id=sample_id,
        annotations=annotations or {},
        raw=raw or {},
    )


def test_exact_dedup_ignores_only_sample_metadata_and_retains_first_object() -> None:
    first = _sample(
        "first",
        dataset="source_a",
        annotations={"quality": "preferred"},
        raw={"source": {"row": 1}},
    )
    duplicate = _sample(
        "second",
        dataset="source_b",
        annotations={"quality": "different"},
        raw={"source": {"row": 999}},
    )

    kept, report = deduplicate_samples([first, duplicate])

    assert kept == [first]
    assert kept[0] is first
    assert report.input_samples == 2
    assert report.output_samples == 1
    assert report.removed_samples == 1
    assert report.unique_trajectories == 1
    assert report.duplicate_groups == 1
    assert report.exact_removed_samples == 1
    assert report.normalized_removed_samples == 0
    assert report.by_dataset == {
        "source_a": DatasetDeduplicationCounts(1, 1, 0, 0, 0),
        "source_b": DatasetDeduplicationCounts(1, 0, 1, 1, 0),
    }


def test_comparison_does_not_mutate_inputs_raw_or_retained_rows() -> None:
    first = _sample(
        "first",
        messages=[
            {"role": "user", "content": "\r\nDo it\r\n"},
            {"role": "assistant", "content": "\r\nDone\r\n"},
        ],
        tools=[_tool("b"), _tool("a")],
        annotations={"nested": {"values": [1, 2]}},
        raw={"messages": [{"content": "\r\nsource\r\n"}]},
    )
    duplicate = _sample(
        "duplicate",
        messages=[
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "Done"},
        ],
        tools=[_tool("a"), _tool("b")],
    )
    before = deepcopy([first, duplicate])

    kept, _ = deduplicate_samples([first, duplicate], level="level_1_5")

    assert [first, duplicate] == before
    assert kept[0] is first
    assert kept[0].messages[0]["content"] == "\r\nDo it\r\n"
    assert [tool["function"]["name"] for tool in kept[0].tools] == ["b", "a"]
    assert kept[0].raw == {"messages": [{"content": "\r\nsource\r\n"}]}


def test_tool_definitions_ignore_list_and_object_key_order() -> None:
    a = _tool("a")
    b = _tool("b")
    reordered_a = {
        "function": {
            "parameters": {
                "required": ["value"],
                "properties": {
                    "value": {
                        "description": "A value",
                        "type": "string",
                    }
                },
                "type": "object",
            },
            "description": "Run a",
            "name": "a",
        },
        "type": "function",
    }
    first = _sample("first", tools=[a, b])
    duplicate = _sample("duplicate", tools=[_tool("b"), reordered_a])

    kept, report = deduplicate_samples([first, duplicate])

    assert kept == [first]
    assert report.removed_samples == 1


def test_tool_definition_order_is_an_unordered_multiset_not_a_set() -> None:
    one_copy = _sample("one", tools=[_tool("a")])
    two_copies = _sample("two", tools=[_tool("a"), _tool("a")])

    kept, report = deduplicate_samples([one_copy, two_copies])

    assert kept == [one_copy, two_copies]
    assert report.removed_samples == 0


def test_message_object_key_order_is_ignored() -> None:
    first = _sample(
        "first",
        messages=[{"role": "assistant", "content": "Done", "name": "agent"}],
    )
    duplicate = _sample(
        "duplicate",
        messages=[{"name": "agent", "content": "Done", "role": "assistant"}],
    )

    kept, _ = deduplicate_samples([first, duplicate])

    assert kept == [first]


def test_parallel_call_order_remains_observable() -> None:
    messages_ab = [
        {"role": "user", "content": "Both"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("a"), _call("b")],
        },
    ]
    messages_ba = deepcopy(messages_ab)
    reversed_calls = messages_ba[1]["tool_calls"]
    assert isinstance(reversed_calls, list)
    reversed_calls.reverse()

    first = _sample("ab", messages=messages_ab, tools=[_tool("a"), _tool("b")])
    second = _sample("ba", messages=messages_ba, tools=[_tool("b"), _tool("a")])

    kept, _ = deduplicate_samples([first, second])

    assert kept == [first, second]


def test_parallel_and_sequential_batch_boundaries_remain_observable() -> None:
    parallel = _sample(
        "parallel",
        messages=[
            {"role": "user", "content": "Both"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("a"), _call("b")],
            },
            {"role": "tool", "content": "A"},
            {"role": "tool", "content": "B"},
        ],
        tools=[_tool("a"), _tool("b")],
    )
    sequential = _sample(
        "sequential",
        messages=[
            {"role": "user", "content": "Both"},
            {"role": "assistant", "content": None, "tool_calls": [_call("a")]},
            {"role": "tool", "content": "A"},
            {"role": "assistant", "content": None, "tool_calls": [_call("b")]},
            {"role": "tool", "content": "B"},
        ],
        tools=[_tool("a"), _tool("b")],
    )

    kept, _ = deduplicate_samples([parallel, sequential])

    assert kept == [parallel, sequential]


def test_tool_result_order_and_position_remain_observable() -> None:
    first = _sample(
        "ab",
        messages=[
            {"role": "assistant", "tool_calls": [_call("a"), _call("b")]},
            {"role": "tool", "content": "A"},
            {"role": "tool", "content": "B"},
        ],
        tools=[_tool("a"), _tool("b")],
    )
    reversed_results = _sample(
        "ba",
        messages=[
            {"role": "assistant", "tool_calls": [_call("a"), _call("b")]},
            {"role": "tool", "content": "B"},
            {"role": "tool", "content": "A"},
        ],
        tools=[_tool("a"), _tool("b")],
    )

    kept, _ = deduplicate_samples([first, reversed_results])

    assert kept == [first, reversed_results]


@pytest.mark.parametrize(
    ("first_enum", "second_enum"),
    [
        (["red", "blue"], ["blue", "red"]),
        (["red", "red"], ["red"]),
    ],
)
def test_arrays_in_tool_definitions_preserve_order_and_multiplicity(
    first_enum: list[str], second_enum: list[str]
) -> None:
    first = _sample("first", tools=[_tool("paint", enum=first_enum)])
    second = _sample("second", tools=[_tool("paint", enum=second_enum)])

    kept, _ = deduplicate_samples([first, second])

    assert kept == [first, second]


def test_call_arguments_are_not_parsed_rewritten_or_reordered() -> None:
    first = _sample(
        "first",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [_call("f", '{"values":[1,2],"flag":true}')],
            }
        ],
    )
    array_reordered = _sample(
        "array",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [_call("f", '{"values":[2,1],"flag":true}')],
            }
        ],
    )
    object_reformatted = _sample(
        "object",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [_call("f", '{"flag":true,"values":[1,2]}')],
            }
        ],
    )

    kept, _ = deduplicate_samples([first, array_reordered, object_reformatted])

    assert kept == [first, array_reordered, object_reformatted]


def test_level_1_5_is_explicit_and_normalizes_only_approved_whitespace() -> None:
    crlf_with_blank_lines = _sample(
        "crlf",
        messages=[
            {"role": "user", "content": "\r\nHello\r\nworld\r\n"},
            {"role": "assistant", "content": "\r\nDone\r\n"},
        ],
    )
    lf = _sample(
        "lf",
        messages=[
            {"role": "user", "content": "Hello\nworld"},
            {"role": "assistant", "content": "Done"},
        ],
    )

    exact_kept, _ = deduplicate_samples([crlf_with_blank_lines, lf])
    normalized_kept, report = deduplicate_samples(
        [crlf_with_blank_lines, lf], level="level_1_5"
    )

    assert exact_kept == [crlf_with_blank_lines, lf]
    assert normalized_kept == [crlf_with_blank_lines]
    assert report.exact_removed_samples == 0
    assert report.normalized_removed_samples == 1
    assert report.by_dataset["a"].normalized_removed_samples == 1


def test_level_1_5_abstains_on_tool_result_content() -> None:
    crlf = _sample(
        "crlf",
        messages=[{"role": "tool", "content": "\r\nok\r\n"}],
    )
    lf = _sample(
        "lf",
        messages=[{"role": "tool", "content": "\nok\n"}],
    )

    kept, report = deduplicate_samples([crlf, lf], level="level_1_5")

    assert kept == [crlf, lf]
    assert report.removed_samples == 0


def test_level_1_5_abstains_on_tool_call_arguments() -> None:
    crlf = _sample(
        "crlf",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [_call("f", '{\r\n"value":"x"\r\n}')],
            }
        ],
    )
    lf = _sample(
        "lf",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [_call("f", '{\n"value":"x"\n}')],
            }
        ],
    )

    kept, report = deduplicate_samples([crlf, lf], level="level_1_5")

    assert kept == [crlf, lf]
    assert report.removed_samples == 0


@pytest.mark.parametrize(
    ("first_content", "second_content"),
    [
        ('\r\n{"value": 1}\r\n', '\n{"value": 1}\n'),
        (
            "\r\n```python\r\nprint('hello')\r\n```\r\n",
            "\n```python\nprint('hello')\n```\n",
        ),
        ("\r\n    indented_code()\r\n", "\n    indented_code()\n"),
    ],
)
def test_level_1_5_abstains_on_structured_and_code_content(
    first_content: str, second_content: str
) -> None:
    first = _sample("first", messages=[{"role": "assistant", "content": first_content}])
    second = _sample(
        "second", messages=[{"role": "assistant", "content": second_content}]
    )

    kept, report = deduplicate_samples([first, second], level="level_1_5")

    assert kept == [first, second]
    assert report.removed_samples == 0


@pytest.mark.parametrize(
    ("first_content", "second_content"),
    [
        (" Hello", "Hello"),
        ("Hello ", "Hello"),
        ("Hello  world", "Hello world"),
        ("Done.", "Done"),
        ("Cannot do that", "cannot do that"),
        ("value: 1", "value: 2"),
    ],
)
def test_level_1_5_does_not_apply_broad_text_normalization(
    first_content: str, second_content: str
) -> None:
    first = _sample("first", messages=[{"role": "assistant", "content": first_content}])
    second = _sample(
        "second", messages=[{"role": "assistant", "content": second_content}]
    )

    kept, _ = deduplicate_samples([first, second], level="level_1_5")

    assert kept == [first, second]


def test_level_1_5_preserves_code_indentation_on_nonblank_lines() -> None:
    four_spaces = _sample(
        "four",
        messages=[
            {
                "role": "assistant",
                "content": "\n```python\nif ready:\n    run()\n```\n",
            }
        ],
    )
    two_spaces = _sample(
        "two",
        messages=[
            {
                "role": "assistant",
                "content": "```python\nif ready:\n  run()\n```",
            }
        ],
    )

    kept, _ = deduplicate_samples([four_spaces, two_spaces], level="level_1_5")

    assert kept == [four_spaces, two_spaces]


def test_level_1_5_preserves_blank_lines_inside_text() -> None:
    one_blank = _sample(
        "one",
        messages=[{"role": "assistant", "content": "First\n\nSecond"}],
    )
    two_blanks = _sample(
        "two",
        messages=[{"role": "assistant", "content": "First\n\n\nSecond"}],
    )

    kept, _ = deduplicate_samples([one_blank, two_blanks], level="level_1_5")

    assert kept == [one_blank, two_blanks]


def test_report_is_deterministic_and_counts_each_removed_source_dataset() -> None:
    first_a = _sample("a1", dataset="a")
    duplicate_b = _sample("b1", dataset="b")
    duplicate_a = _sample("a2", dataset="a")
    distinct_b = _sample(
        "b2",
        dataset="b",
        messages=[{"role": "user", "content": "Different"}],
    )

    kept, report = deduplicate_samples([first_a, duplicate_b, duplicate_a, distinct_b])

    assert kept == [first_a, distinct_b]
    assert [sample.sample_id for sample in kept] == ["a1", "b2"]
    assert list(report.by_dataset) == ["a", "b"]
    assert report.by_dataset == {
        "a": DatasetDeduplicationCounts(2, 1, 1, 1, 0),
        "b": DatasetDeduplicationCounts(2, 1, 1, 1, 0),
    }


def test_level_1_5_reports_ordered_exact_then_normalized_removals() -> None:
    normalized_variant = _sample(
        "normalized",
        dataset="first",
        messages=[{"role": "assistant", "content": "\nfoo\n"}],
    )
    exact_variant = _sample(
        "exact-first",
        dataset="second",
        messages=[{"role": "assistant", "content": "foo"}],
    )
    exact_duplicate = _sample(
        "exact-duplicate",
        dataset="third",
        messages=[{"role": "assistant", "content": "foo"}],
    )

    kept, report = deduplicate_samples(
        [normalized_variant, exact_variant, exact_duplicate], level="level_1_5"
    )

    assert kept == [normalized_variant]
    assert report.exact_removed_samples == 1
    assert report.normalized_removed_samples == 1
    assert report.by_dataset["first"].removed_samples == 0
    assert report.by_dataset["second"].normalized_removed_samples == 1
    assert report.by_dataset["third"].exact_removed_samples == 1


def test_empty_input_and_invalid_level() -> None:
    kept, report = deduplicate_samples([])

    assert kept == []
    assert report.input_samples == 0
    assert report.output_samples == 0
    assert report.removed_samples == 0
    assert report.by_dataset == {}

    with pytest.raises(ValueError, match="unsupported deduplication level"):
        deduplicate_samples([], level=cast(DeduplicationLevel, "level_9"))


def test_level_2_selects_shortest_total_assistant_content_and_stable_ties() -> None:
    longer = _sample(
        "longer",
        messages=[
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": "I will call it.",
                "reasoning_content": "short private reasoning",
                "tool_calls": [_call("f", '{"value":"x"}')],
            },
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Finished."},
        ],
    )
    shortest = _sample(
        "shortest",
        messages=[
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "much longer private reasoning is irrelevant",
                "tool_calls": [_call("f", '{"value":"x"}')],
            },
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Done."},
        ],
    )
    equal_score_later = deepcopy(shortest)
    equal_score_later.sample_id = "equal-score-later"
    equal_score_later.messages[2]["reasoning_content"] = "different again"

    kept, report = deduplicate_samples(
        [longer, shortest, equal_score_later], level="level_2"
    )

    assert kept == [shortest]
    assert kept[0] is shortest
    assert report.input_samples == 3
    assert report.output_samples == 1
    assert report.removed_samples == 2
    assert report.duplicate_groups == 1
    assert report.by_dataset["a"].removed_samples == 2


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "system", "content": "Different policy"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "short"},
        ],
        [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Different question"},
            {"role": "assistant", "content": "short"},
        ],
        [
            {"role": "user", "content": "Question"},
            {"role": "system", "content": "Policy"},
            {"role": "assistant", "content": "short"},
        ],
        [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "short"},
        ],
    ],
)
def test_level_2_preserves_roles_positions_and_exact_context(
    messages: list[dict],
) -> None:
    base = _sample(
        "base",
        messages=[
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "a long response"},
        ],
    )
    changed = _sample("changed", messages=messages)

    kept, _ = deduplicate_samples([base, changed], level="level_2")

    assert kept == [base, changed]


def test_level_2_normalizes_json_arguments_results_and_tool_definition_order() -> None:
    first = _sample(
        "first",
        messages=[
            {"role": "user", "content": "Both"},
            {
                "role": "assistant",
                "content": "a much longer preamble",
                "tool_calls": [_call("a", '{"x":1,"values":[1,2]}')],
            },
            {"role": "tool", "content": '{"answer":2,"ok":true}'},
        ],
        tools=[_tool("a"), _tool("b")],
    )
    canonical_variant = _sample(
        "canonical",
        messages=[
            {"role": "user", "content": "Both"},
            {
                "role": "assistant",
                "content": "short",
                "tool_calls": [_call("a", '{ "values" : [1,2], "x" : 1 }')],
            },
            {"role": "tool", "content": '{"ok":true,"answer":2}'},
        ],
        tools=[_tool("b"), _tool("a")],
    )

    kept, _ = deduplicate_samples([first, canonical_variant], level="level_2")

    assert kept == [canonical_variant]


@pytest.mark.parametrize(
    ("arguments", "result"),
    [
        ('{"x":1,"values":[2,1]}', '{"answer":2,"ok":true}'),
        ('{"x":2,"values":[1,2]}', '{"answer":2,"ok":true}'),
        ("not json ", '{"answer":2,"ok":true}'),
        ('{"x":1,"values":[1,2]}', '{"answer":3,"ok":true}'),
    ],
)
def test_level_2_preserves_json_boundaries_argument_and_result_changes(
    arguments: str, result: str
) -> None:
    base = _sample(
        "base",
        messages=[
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": "long response",
                "tool_calls": [_call("f", '{"x":1,"values":[1,2]}')],
            },
            {"role": "tool", "content": '{"answer":2,"ok":true}'},
        ],
    )
    changed = _sample(
        "changed",
        messages=[
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": "short",
                "tool_calls": [_call("f", arguments)],
            },
            {"role": "tool", "content": result},
        ],
    )

    kept, _ = deduplicate_samples([base, changed], level="level_2")

    assert kept == [base, changed]


def test_level_2_preserves_parallel_and_sequential_call_boundaries() -> None:
    parallel = _sample(
        "parallel",
        messages=[
            {"role": "user", "content": "Both"},
            {
                "role": "assistant",
                "content": "long",
                "tool_calls": [_call("a"), _call("b")],
            },
            {"role": "tool", "content": "A"},
            {"role": "tool", "content": "B"},
        ],
        tools=[_tool("a"), _tool("b")],
    )
    sequential = _sample(
        "sequential",
        messages=[
            {"role": "user", "content": "Both"},
            {"role": "assistant", "content": "", "tool_calls": [_call("a")]},
            {"role": "tool", "content": "A"},
            {"role": "assistant", "content": "", "tool_calls": [_call("b")]},
            {"role": "tool", "content": "B"},
        ],
        tools=[_tool("a"), _tool("b")],
    )

    kept, _ = deduplicate_samples([parallel, sequential], level="level_2")

    assert kept == [parallel, sequential]


def test_level_2_preserves_call_decisions_and_complete_tool_environment() -> None:
    called = _sample(
        "called",
        messages=[
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "long", "tool_calls": [_call("f")]},
        ],
        tools=[_tool("f")],
    )
    no_call = _sample(
        "no-call",
        messages=[
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "x"},
        ],
        tools=[_tool("f")],
    )
    changed_definition = _sample(
        "changed-definition",
        messages=deepcopy(called.messages),
        tools=[_tool("f")],
    )
    changed_definition.tools[0]["function"]["description"] = "Different contract"
    duplicate_definition = _sample(
        "duplicate-definition",
        messages=deepcopy(called.messages),
        tools=[_tool("f"), _tool("f")],
    )

    kept, _ = deduplicate_samples(
        [called, no_call, changed_definition, duplicate_definition], level="level_2"
    )

    assert kept == [called, no_call, changed_definition, duplicate_definition]


def test_level_2_does_not_mutate_retained_or_removed_samples() -> None:
    longer = _sample(
        "longer",
        messages=[
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": "longer",
                "tool_calls": [_call("f", '{ "b": 2, "a": 1 }')],
            },
            {"role": "tool", "content": '{ "ok": true }'},
        ],
        tools=[_tool("f"), _tool("g")],
        annotations={"nested": {"keep": True}},
        raw={"untouched": [1, 2]},
    )
    shorter = _sample(
        "shorter",
        messages=[
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": [_call("f", '{"a":1,"b":2}')],
            },
            {"role": "tool", "content": '{"ok":true}'},
        ],
        tools=[_tool("g"), _tool("f")],
    )
    before = deepcopy([longer, shorter])

    kept, _ = deduplicate_samples([longer, shorter], level="level_2")

    assert [longer, shorter] == before
    assert kept == [shorter]
    assert kept[0] is shorter


def test_cumulative_subset_curation_reports_each_stage_separately() -> None:
    long = _sample(
        "long",
        messages=[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Long response"},
        ],
    )
    exact_duplicate = deepcopy(long)
    exact_duplicate.sample_id = "exact"
    normalized_duplicate = _sample(
        "normalized",
        messages=[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "\nLong response\n"},
        ],
    )
    shortest = _sample(
        "shortest",
        messages=[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Short"},
        ],
    )

    kept, report = curate_subset_samples(
        [long, exact_duplicate, normalized_duplicate, shortest]
    )

    assert kept == [shortest]
    assert kept[0] is shortest
    assert report.input_samples == 4
    assert report.output_samples == 1
    assert report.removed_samples == 3
    assert report.level_1.removed_samples == 1
    assert report.level_1_5.removed_samples == 1
    assert report.level_2.removed_samples == 1
    assert report.level_3.eligible_samples == 0
    assert report.level_4.eligible_samples == 1
    assert report.level_5.eligible_samples == 1
    serialized = report.as_dict()
    assert serialized["level_2"]["removed_samples"] == 1
    assert serialized["level_3"]["eligible_samples"] == 0


def test_level_3_4_5_audits_are_aggregate_only_and_preserve_boundaries() -> None:
    common_tools = [_tool("f")]
    first = _sample(
        "first",
        messages=[
            {"role": "system", "content": "S"},
            {"role": "user", "content": "First wording"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [_call("f", '{"a":1,"b":2}')],
            },
            {"role": "tool", "content": '{"ok":true,"value":2}'},
            {"role": "assistant", "content": "First answer"},
        ],
        tools=common_tools,
    )
    same_level_3 = _sample(
        "same-l3",
        messages=[
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Different wording"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("f", '{ "b": 2, "a": 1 }')],
            },
            {"role": "tool", "content": '{"value":2,"ok":true}'},
            {"role": "assistant", "content": "Different answer"},
        ],
        tools=common_tools,
    )
    same_level_4_only = _sample(
        "same-l4",
        messages=[
            {"role": "system", "content": "Different environment"},
            {"role": "user", "content": "Third"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("f", '{"a":9}')],
            },
            {"role": "tool", "content": '{"ok":false}'},
        ],
        tools=common_tools,
    )
    no_call_f = _sample(
        "no-call-f",
        messages=[{"role": "user", "content": "No call"}],
        tools=common_tools,
    )
    no_call_g = _sample(
        "no-call-g",
        messages=[{"role": "user", "content": "No call either"}],
        tools=[_tool("g")],
    )
    samples = [first, same_level_3, same_level_4_only, no_call_f, no_call_g]
    before = deepcopy(samples)

    audit = audit_subset_samples(samples)

    assert samples == before
    assert audit.level_3 == EquivalenceAuditCounts(3, 2, 1, 2, 1)
    assert audit.level_4 == EquivalenceAuditCounts(5, 2, 2, 5, 3)
    assert audit.level_5 == EquivalenceAuditCounts(5, 2, 1, 4, 3)
    assert audit.as_dict()["level_3"]["eligible_samples"] == 3


def test_level_4_preserves_parallel_batch_boundaries() -> None:
    parallel = _sample(
        "parallel",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("a"), _call("b")],
            }
        ],
    )
    sequential = _sample(
        "sequential",
        messages=[
            {"role": "assistant", "content": "", "tool_calls": [_call("a")]},
            {"role": "assistant", "content": "", "tool_calls": [_call("b")]},
        ],
    )

    audit = audit_subset_samples([parallel, sequential])

    assert audit.level_4 == EquivalenceAuditCounts(2, 2, 0, 0, 0)
