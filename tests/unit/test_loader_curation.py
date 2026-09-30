"""Adversarial curation tests independent of historical corpus fixtures."""

from copy import deepcopy
from pathlib import Path

import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.curation import (
    BoundCallBatch,
    CurationConfig,
    CurationInput,
    CurationScope,
    curate,
)


SCOPE = CurationScope("dataset", "subset", "train")


@pytest.mark.parametrize("level", ["level_1", "level_1_5", "level_2"])
def test_large_integer_arguments_deduplicate_exactly_without_float_rounding(level):
    original = _episode(
        "original", calls=[_call(arguments='{"a":51090942171709440000,"b":987}')]
    )
    reordered = _episode(
        "reordered", calls=[_call(arguments='{ "b":987, "a":51090942171709440000 }')]
    )
    distinct = _episode(
        "distinct", calls=[_call(arguments='{"a":51090942171709440001,"b":987}')]
    )
    kept, _ = _run(
        [original, reordered, distinct],
        level=level,
        batches=(BoundCallBatch(1, (2,), parallel=False),),
    )
    assert kept == [original, distinct]
    assert (
        kept[0].messages[1]["tool_calls"][0]["function"]["arguments"]
        == '{"a":51090942171709440000,"b":987}'
    )


def _tool(name="f", **fields):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "A function",
            "parameters": {"type": "object", "properties": {}},
            **fields,
        },
    }


def _call(name="f", arguments="{}"):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _sample(sample_id, *, messages=None, tools=None, **fields):
    return ConversationSample(
        sample_id=sample_id,
        dataset="dataset",
        messages=messages
        if messages is not None
        else [
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "Done"},
        ],
        tools=tools if tools is not None else [_tool()],
        **fields,
    )


def _episode(sample_id, *, calls=None, results=None, answer="Done"):
    calls = calls if calls is not None else [_call()]
    results = results if results is not None else ["OK"]
    return _sample(
        sample_id,
        messages=[
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            *({"role": "tool", "content": result} for result in results),
            {"role": "assistant", "content": answer},
        ],
    )


def _run(samples, *, level="level_2", audit=True, batches=(), scope=SCOPE):
    with curate(
        (CurationInput(sample, batches) for sample in samples),
        scope=scope,
        config=CurationConfig(max_level=level, audit=audit),
    ) as run:
        kept = list(run)
        return kept, run.reports


def test_complete_values_are_retained_and_metadata_never_affects_equality():
    first = _sample(
        "first", raw={"nested": {"values": [1, 2]}}, annotations={"score": 1}
    )
    second = _sample(
        "second", raw={"nested": {"values": [9]}}, annotations={"score": 99}
    )
    before = deepcopy([first, second])
    kept, reports = _run([first, second], level="level_1")
    assert kept == [first]
    assert [first, second] == before
    assert reports[SCOPE].stages["level_1"].removed_samples == 1
    kept[0].messages[0]["content"] = "Different"
    assert first.messages[0]["content"] == "Do it"
    assert first.raw == before[0].raw


def test_complete_definitions_ignore_object_and_definition_list_order_only():
    a, b = _tool("a"), _tool("b")
    reversed_keys = {
        "function": dict(reversed(list(a["function"].items()))),
        "type": "function",
    }
    first = _sample("first", tools=[a, b])
    reordered = _sample("reordered", tools=[b, reversed_keys])
    strict = _sample(
        "strict", tools=[{**a, "function": {**a["function"], "strict": False}}, b]
    )
    multiplicity = _sample("multiplicity", tools=[a, b, a])
    kept, _ = _run([first, reordered, strict, multiplicity], level="level_1")
    assert kept == [first, strict, multiplicity]
    assert kept[0].tools == [a, b]


def test_empty_absent_and_positioned_system_messages_remain_distinct():
    absent = _sample("absent")
    empty = _sample(
        "empty", messages=[{"role": "system", "content": ""}, *absent.messages]
    )
    moved = _sample(
        "moved",
        messages=[
            absent.messages[0],
            {"role": "system", "content": ""},
            absent.messages[1],
        ],
    )
    kept, _ = _run([absent, empty, moved])
    assert kept == [absent, empty, moved]


def test_verified_parallel_pairs_are_atomic_and_source_order_is_unchanged():
    original = _episode("original", calls=[_call("b"), _call("a")], results=["B", "A"])
    permutation = _episode(
        "permutation", calls=[_call("a"), _call("b")], results=["A", "B"]
    )
    wrong_pairing = _episode(
        "wrong", calls=[_call("a"), _call("b")], results=["B", "A"]
    )
    before = deepcopy(original)
    kept, _ = _run(
        [original, permutation, wrong_pairing],
        level="level_1",
        batches=(BoundCallBatch(1, (2, 3), parallel=True),),
    )
    assert kept == [original, wrong_pairing]
    assert kept[0] == before


def test_nonpositional_source_proof_binds_results_before_parallel_sorting():
    original = _episode("original", calls=[_call("a"), _call("b")], results=["B", "A"])
    ordered = _episode("ordered", calls=[_call("a"), _call("b")], results=["A", "B"])
    with curate(
        [
            CurationInput(original, (BoundCallBatch(1, (3, 2), parallel=True),)),
            CurationInput(ordered, (BoundCallBatch(1, (2, 3), parallel=True),)),
        ],
        scope=SCOPE,
        config=CurationConfig(max_level="level_1"),
    ) as run:
        assert list(run) == [original]


@pytest.mark.parametrize("parallel", [None, False])
def test_ambiguous_or_nonparallel_linkage_never_authorizes_permutation(parallel):
    original = _episode("a", calls=[_call("b"), _call("a")], results=["B", "A"])
    permutation = _episode("b", calls=[_call("a"), _call("b")], results=["A", "B"])
    batches = () if parallel is None else (BoundCallBatch(1, (2, 3), parallel=False),)
    kept, _ = _run([original, permutation], batches=batches)
    assert kept == [original, permutation]


def test_identical_order_compares_equal_with_or_without_parallel_proof():
    original = _episode("a", calls=[_call("a"), _call("b")], results=["A", "B"])
    duplicate = deepcopy(original)
    duplicate.sample_id = "b"
    with curate(
        [
            CurationInput(original),
            CurationInput(duplicate, (BoundCallBatch(1, (2, 3), True),)),
        ],
        scope=SCOPE,
    ) as run:
        assert list(run) == [original]


def test_duplicate_call_names_keep_arguments_results_and_multiplicity():
    a = _episode(
        "a",
        calls=[_call(arguments='{"x":1}'), _call(arguments='{"x":2}')],
        results=["one", "two"],
    )
    b = _episode(
        "b",
        calls=[_call(arguments='{"x":2}'), _call(arguments='{"x":1}')],
        results=["two", "one"],
    )
    wrong = _episode(
        "wrong",
        calls=[_call(arguments='{"x":1}'), _call(arguments='{"x":2}')],
        results=["two", "one"],
    )
    extra = _episode(
        "extra", calls=[_call(arguments='{"x":1}')] * 2, results=["one"] * 2
    )
    kept, _ = _run([a, b, wrong, extra], batches=(BoundCallBatch(1, (2, 3), True),))
    assert kept == [a, wrong, extra]


def test_parallel_boundaries_and_sequential_chronology_are_significant():
    parallel = _episode("parallel", calls=[_call("a"), _call("b")], results=["A", "B"])
    sequential = _sample(
        "sequential",
        messages=[
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": None, "tool_calls": [_call("a")]},
            {"role": "tool", "content": "A"},
            {"role": "assistant", "content": None, "tool_calls": [_call("b")]},
            {"role": "tool", "content": "B"},
            {"role": "assistant", "content": "Done"},
        ],
    )
    with curate(
        [
            CurationInput(parallel, (BoundCallBatch(1, (2, 3), True),)),
            CurationInput(
                sequential,
                (BoundCallBatch(1, (2,), True), BoundCallBatch(3, (4,), True)),
            ),
        ],
        scope=SCOPE,
    ) as run:
        assert list(run) == [parallel, sequential]
        assert run.reports[SCOPE].audit["level_4"].unique_groups == 2


@pytest.mark.parametrize(
    "binding",
    [
        BoundCallBatch(0, (2, 3), True),
        BoundCallBatch(1, (2, 2), True),
        BoundCallBatch(1, (2,), True),
        BoundCallBatch(1, (2, 4), True),
        BoundCallBatch(100, (2, 3), True),
    ],
)
def test_invalid_binding_coordinates_raise_instead_of_asserting_equivalence(binding):
    sample = _episode("a", calls=[_call("a"), _call("b")], results=["A", "B"])
    with pytest.raises(ValueError):
        _run([sample], batches=(binding,))


def test_level_1_5_normalizes_only_plain_prose_and_preserves_selected_text():
    first = _sample(
        "first",
        messages=[
            {"role": "user", "content": "\r\nDo it\r\n"},
            {"role": "assistant", "content": "\nDone\n"},
        ],
    )
    second = _sample("second")
    kept, reports = _run([first, second], level="level_1_5")
    assert kept == [first]
    assert reports[SCOPE].stages["level_1"].removed_samples == 0
    assert reports[SCOPE].stages["level_1_5"].removed_samples == 1
    assert kept[0].messages[0]["content"] == "\r\nDo it\r\n"


@pytest.mark.parametrize(
    "text",
    [
        "```python\r\nprint(1)\r\n```",
        '{"a": 1}\r\n',
        "return value\r\n",
        "    indented\r\n",
        "# Heading\r\n",
        "1. A list\r\n",
    ],
)
def test_level_1_5_abstains_on_structured_or_code_content(text):
    first = _sample("first", messages=[{"role": "assistant", "content": text}])
    second = _sample(
        "second",
        messages=[{"role": "assistant", "content": text.replace("\r\n", "\n")}],
    )
    kept, _ = _run([first, second], level="level_1_5")
    assert kept == [first, second]


@pytest.mark.parametrize(
    "literal",
    [
        "'quoted literal'",
        'b"byte literal"',
        'r"raw literal"',
        "(1, 2)",
        "()",
        "True",
        "False",
        "None",
        "Ellipsis",
        "...",
        "1_000",
        "0xff",
        "1j",
        ".5j",
        "-1j",
        "1 + 2j",
        "true",
        "false",
        "null",
    ],
)
def test_level_1_5_abstains_on_python_and_json_literals_without_parsing(literal):
    first = _sample(
        "first", messages=[{"role": "assistant", "content": "\r\n" + literal + "\r\n"}]
    )
    second = _sample("second", messages=[{"role": "assistant", "content": literal}])
    kept, _ = _run([first, second], level="level_1_5")
    assert kept == [first, second]


def test_arguments_preserve_string_whitespace_array_order_and_numeric_type():
    a = _episode("a", calls=[_call(arguments='{"x":" words ","a":[1,2]}')])
    object_order = _episode(
        "duplicate", calls=[_call(arguments=' {"a":[1,2],"x":" words "} ')]
    )
    whitespace = _episode(
        "whitespace", calls=[_call(arguments='{"x":"words","a":[1,2]}')]
    )
    array_order = _episode(
        "array", calls=[_call(arguments='{"x":" words ","a":[2,1]}')]
    )
    numeric_type = _episode(
        "float", calls=[_call(arguments='{"x":" words ","a":[1.0,2]}')]
    )
    kept, _ = _run([a, object_order, whitespace, array_order, numeric_type])
    assert kept == [a, whitespace, array_order, numeric_type]
    assert (
        kept[0].messages[1]["tool_calls"][0]["function"]["arguments"]
        == '{"x":" words ","a":[1,2]}'
    )


def test_uncached_ambiguous_argument_json_never_establishes_equivalence():
    ambiguous = _episode("ambiguous", calls=[_call(arguments='{"x":1,"x":2}')])
    with pytest.raises(ValueError, match="duplicate_json_key"):
        _run([ambiguous])


@pytest.mark.parametrize("keyword", ["default", "const", "enum"])
def test_level_1_5_abstains_on_structured_literals_inside_definitions(keyword):
    source_value = {"description": "\nExact literal\n", "title": "\r\nExact title\r\n"}
    changed_value = {"description": "Exact literal", "title": "Exact title"}
    if keyword == "enum":
        source_value = [source_value]
        changed_value = [changed_value]
    first = _sample(
        "first", tools=[_tool(parameters={"type": "object", keyword: source_value})]
    )
    second = _sample(
        "second", tools=[_tool(parameters={"type": "object", keyword: changed_value})]
    )
    kept, _ = _run([first, second])
    assert kept == [first, second]


@pytest.mark.parametrize(
    "first_result,second_result",
    [("\r\nOK\r\n", "OK"), ('{"a":1}', '{ "a": 1 }'), ("1", "1.0")],
)
def test_results_are_opaque_at_every_deletion_level(first_result, second_result):
    a = _episode("a", results=[first_result])
    b = _episode("b", results=[second_result])
    kept, _ = _run([a, b])
    assert kept == [a, b]


def test_level_2_selects_shortest_original_stably_and_preserves_global_order():
    long = _episode("long", answer="This is a very long answer")
    unrelated = _episode("unrelated", results=["Different"])
    short = _episode("short", answer="OK")
    tie = _episode("tie", answer="No")
    kept, reports = _run([long, unrelated, short, tie])
    assert kept == [unrelated, short]
    counts = reports[SCOPE].stages["level_2"]
    assert (counts.input_samples, counts.output_samples, counts.removed_samples) == (
        4,
        2,
        2,
    )
    assert counts.duplicate_groups == 1


def test_levels_are_cumulative_and_level_1_5_first_member_is_not_reconsidered():
    first = _episode("first", answer="\nDone\n")
    exact = deepcopy(first)
    exact.sample_id = "exact"
    whitespace = _episode("whitespace", answer="Done")
    shorter = _episode("shorter", answer="OK")
    kept, reports = _run([first, exact, whitespace, shorter])
    assert kept == [shorter]
    assert [
        reports[SCOPE].stages[level].removed_samples
        for level in ("level_1", "level_1_5", "level_2")
    ] == [1, 1, 1]
    kept, _ = _run([first, whitespace])
    assert kept == [first]


def test_level_2_preserves_assistant_positions_and_exact_user_context():
    a = _sample("a")
    b = _sample(
        "b",
        messages=[
            a.messages[0],
            {"role": "assistant", "content": "A"},
            {"role": "assistant", "content": "B"},
        ],
    )
    c = _sample(
        "c", messages=[{"role": "user", "content": "Do something else"}, a.messages[1]]
    )
    kept, _ = _run([a, b, c])
    assert kept == [a, b, c]


def test_levels_3_to_5_are_aggregate_only_and_level_3_excludes_no_call_rows():
    first = _episode("a")
    other_user = deepcopy(first)
    other_user.sample_id = "other"
    other_user.messages[0]["content"] = "Please proceed"
    no_call = _sample("no_call")
    before = deepcopy([first, other_user, no_call])
    kept, reports = _run([first, other_user, no_call])
    assert kept == before
    assert all(sample.annotations == {} for sample in kept)
    audits = reports[SCOPE].audit
    assert audits["level_3"].eligible_samples == 2
    assert audits["level_3"].unique_groups == 1
    assert audits["level_3"].samples_in_repeated_groups == 2
    assert audits["level_4"].eligible_samples == 3
    assert audits["level_4"].unique_groups == 2
    assert audits["level_5"].unique_groups == 1
    assert reports[SCOPE].removed_samples == 0


def test_audit_only_preserves_duplicates_and_reports_final_population():
    first = _episode("a")
    second = _episode("b")
    kept, reports = _run([first, second], level=None)
    assert kept == [first, second]
    assert reports[SCOPE].stages == {}
    assert reports[SCOPE].audit["level_3"].extra_samples == 1
    curated, reports = _run([first, second])
    assert curated == [first]
    assert reports[SCOPE].audit["level_3"].extra_samples == 0


def test_named_scope_keeps_interleaved_subsets_and_splits_separate():
    rows = [_sample(i) for i in range(6)]
    scopes = [
        CurationScope("dataset", "a", "train"),
        CurationScope("dataset", "b", "train"),
        CurationScope("dataset", "a", "test"),
    ]
    kept, reports = _run(rows, scope=lambda record: scopes[record.sample.sample_id % 3])
    assert kept == rows[:3]
    assert len(reports) == 3
    assert all(
        report.input_samples == 2 and report.output_samples == 1
        for report in reports.values()
    )


def test_large_raw_payloads_preserve_scoped_cumulative_counts_and_winner_order():
    rows = []
    scopes = [CurationScope("dataset", subset, "train") for subset in ("a", "b", "c")]
    for variant, answer in enumerate(
        ("A longer answer", "A longer answer", "\nA longer answer\n", "OK")
    ):
        for group in range(2):
            for scope in scopes:
                sample = _episode(f"{scope.subset}:{group}:{variant}", answer=answer)
                sample.messages[0]["content"] = f"Request {group}"
                sample.raw = {
                    "subset": scope.subset,
                    "source_messages": deepcopy(sample.messages),
                    "large_payload": "source data " * 8192,
                }
                rows.append(sample)
    before = deepcopy(rows)
    scope_by_subset = {scope.subset: scope for scope in scopes}
    kept, reports = _run(
        rows,
        scope=lambda record: scope_by_subset[record.sample.raw["subset"]],
        batches=(BoundCallBatch(1, (2,), True),),
    )
    assert kept == rows[-6:]
    assert rows == before
    for scope in scopes:
        report = reports[scope]
        assert (
            report.input_samples,
            report.output_samples,
            report.removed_samples,
        ) == (8, 2, 6)
        for level, population in (
            ("level_1", (8, 6)),
            ("level_1_5", (6, 4)),
            ("level_2", (4, 2)),
        ):
            stage = report.stages[level]
            assert (stage.input_samples, stage.output_samples) == population
            assert stage.removed_samples == 2
            assert stage.duplicate_groups == 2
        for level in ("level_3", "level_4", "level_5"):
            assert report.audit[level].eligible_samples == 2
            assert report.audit[level].unique_groups == 1
            assert report.audit[level].extra_samples == 1


def test_scope_cannot_silently_consolidate_distinct_datasets():
    row = _sample("a")
    row.dataset = "another"
    with pytest.raises(ValueError, match="scope dataset"):
        _run([row])
    with pytest.raises(ValueError, match="nonempty"):
        CurationScope("dataset", "", "train")


def test_stream_is_one_pass_and_temporary_files_are_removed_on_early_close(tmp_path):
    consumed = []

    def records():
        for i in range(100):
            consumed.append(i)
            yield CurationInput(_episode(i, results=[str(i)]))

    with curate(
        records(), scope=SCOPE, config=CurationConfig(temporary_directory=tmp_path)
    ) as run:
        first = next(iter(run))
        assert first.sample_id == 0
        assert consumed == list(range(100))
        assert run.reports[SCOPE].output_samples == 100
        assert len(list(tmp_path.iterdir())) == 1
    assert list(tmp_path.iterdir()) == []


def test_temporary_files_are_removed_after_source_exception(tmp_path):
    def records():
        yield CurationInput(_sample("first"))
        raise RuntimeError("source failure")

    with curate(
        records(), scope=SCOPE, config=CurationConfig(temporary_directory=tmp_path)
    ) as run:
        with pytest.raises(RuntimeError, match="source failure"):
            list(run)
    assert list(tmp_path.iterdir()) == []


def test_empty_input_and_disabled_audit(tmp_path):
    with curate(
        [], scope=SCOPE, config=CurationConfig(temporary_directory=tmp_path)
    ) as run:
        assert list(run) == []
        assert run.reports == {}
    kept, reports = _run([_sample("a")], audit=False)
    assert len(kept) == 1
    assert reports[SCOPE].audit == {}
    assert not any(Path(tmp_path).iterdir())
