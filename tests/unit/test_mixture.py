"""Final retained-union curation, independent of source fixtures."""

from copy import deepcopy
from dataclasses import asdict
import weakref
import sqlite3

import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.curation import (
    BoundCallBatch,
    CurationConfig,
    CurationInput,
    CurationScope,
    MixtureScope,
    curate,
)
import fcanalysis.mixture as mixture
from fcanalysis.mixture import MixtureSource, deduplicate_mixture


SCOPE = MixtureScope("retained-union", "train")


def _sample(
    identifier,
    *,
    dataset="a",
    environment="one",
    user="Do it",
    answer="Done",
    calls=True,
):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": environment,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [{"role": "user", "content": user}]
    if calls:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a":1}'},
                        }
                    ],
                },
                {"role": "tool", "content": "result"},
            ]
        )
    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "reasoning_content": "Native reasoning",
        }
    )
    return ConversationSample(messages, tools, dataset, identifier)


def _run(sources, tmp_path, *, level="level_2", audit=True, partition_size=1):
    with deduplicate_mixture(
        sources,
        scope=SCOPE,
        config=CurationConfig(
            max_level=level, audit=audit, temporary_directory=tmp_path
        ),
        partition_size_bytes=partition_size,
    ) as result:
        samples = list(result)
        report = result.report
    assert list(tmp_path.iterdir()) == []
    return samples, report


def test_cross_dataset_exact_duplicates_priority_metadata_and_raw(tmp_path):
    first = _sample(7, dataset="first")
    first.raw = {"tuple": (b"raw", 10**90), "nested": {"list": [1, 2]}}
    first.annotations = {"keep": True}
    second = deepcopy(first)
    second.dataset = "second"
    second.sample_id = "7"
    second.raw = {"different": True}
    second.annotations = {"keep": False}
    original = deepcopy(first)
    kept, report = _run(
        [
            MixtureSource("priority", iter([first])),
            MixtureSource("later", iter([second])),
        ],
        tmp_path,
    )
    assert kept == [first] and first == original
    assert kept[0] is not first and kept[0].raw is not first.raw
    assert kept[0].messages[-1]["reasoning_content"] == "Native reasoning"
    assert report.complete
    assert report.sources["priority"].output_samples == 1
    assert report.sources["later"].removed_samples == 1
    assert report.curation.stages["level_1"].removed_samples == 1
    kept[0].raw["nested"]["list"].append(3)
    assert first == original


def test_shorter_later_winner_and_equal_length_tie_follow_global_order(tmp_path):
    long = _sample("long", answer="A long answer")
    middle = _sample("middle", environment="other")
    short = _sample("short", dataset="later", answer="Yes")
    tie = _sample("tie", dataset="third", answer="Yep")
    kept, report = _run(
        [
            MixtureSource("first", [long, middle]),
            MixtureSource("later", [short]),
            MixtureSource("last", [tie]),
        ],
        tmp_path,
    )
    assert [s.sample_id for s in kept] == ["middle", "short"]
    assert report.sources["first"].removed_samples == 1
    assert report.sources["later"].output_samples == 1
    assert report.sources["last"].removed_samples == 1
    assert report.curation.stages["level_2"].removed_samples == 2


@pytest.mark.parametrize("level", [None, "level_1", "level_1_5", "level_2"])
@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("partition_size", [1, 2200, 10**9])
def test_complete_environment_partitions_equal_single_global_pass(
    tmp_path, level, audit, partition_size
):
    rows = [
        _sample("a-long", answer="Long assistant answer"),
        _sample("b", environment="two", user="Other"),
        _sample("text-a", environment="text-a", calls=False),
        _sample("near", user="\nDo it\n", answer="Long assistant answer"),
        _sample("a-short", dataset="b", answer="X"),
        _sample("b-copy", dataset="b", environment="two", user="Other"),
        _sample("text-b", environment="text-b", calls=False),
        _sample("same-names-new-environment", environment="three"),
        _sample("array-distinct"),
    ]
    rows[-1].messages[1]["tool_calls"][0]["function"]["arguments"] = '{"a":[1,2]}'
    config = CurationConfig(max_level=level, audit=audit, temporary_directory=tmp_path)
    with curate(
        (CurationInput(s) for s in rows), scope=SCOPE, config=config
    ) as baseline:
        expected = list(baseline)
        expected_report = asdict(baseline.reports[SCOPE])
    kept, report = _run(
        [
            MixtureSource("one", iter(rows[:4])),
            MixtureSource("empty", iter([])),
            MixtureSource("two", iter(rows[4:])),
        ],
        tmp_path,
        level=level,
        audit=audit,
        partition_size=partition_size,
    )
    assert kept == expected
    assert asdict(report.curation) == expected_report
    assert sum(c.input_samples for c in report.sources.values()) == len(rows)
    assert sum(c.output_samples for c in report.sources.values()) == len(expected)
    assert report.sources["empty"].input_samples == 0


def test_exact_tool_environment_includes_types_bigints_and_definition_order(tmp_path):
    first = _sample("integer")
    first.tools[0]["function"]["parameters"]["default"] = 10**90
    reordered = deepcopy(first)
    reordered.sample_id = "reordered"
    reordered.dataset = "b"
    reordered.tools[0] = dict(reversed(list(reordered.tools[0].items())))
    distinct = deepcopy(first)
    distinct.sample_id = "distinct"
    distinct.tools[0]["function"]["parameters"]["default"] += 1
    integer = _sample("one-int")
    floating = _sample("one-float")
    integer.tools[0]["function"]["parameters"]["default"] = 1
    floating.tools[0]["function"]["parameters"]["default"] = 1.0
    kept, report = _run(
        [MixtureSource("source", [first, reordered, distinct, integer, floating])],
        tmp_path,
    )
    assert [s.sample_id for s in kept] == [
        "integer",
        "distinct",
        "one-int",
        "one-float",
    ]
    assert report.curation.audit["level_5"].unique_groups == 4


def _two_calls(identifier, reverse=False):
    sample = _sample(identifier)
    calls = [
        {"type": "function", "function": {"name": name, "arguments": "{}"}}
        for name in ("a", "b")
    ]
    results = [{"role": "tool", "content": value} for value in ("A", "B")]
    if reverse:
        calls.reverse()
        results.reverse()
    sample.messages[1]["tool_calls"] = calls
    sample.messages[2:3] = results
    return sample


def test_canonical_multicall_order_never_grants_permutation(tmp_path):
    a = _two_calls("a")
    b = _two_calls("b", reverse=True)
    kept, report = _run(
        [MixtureSource("one", [a]), MixtureSource("two", [b])], tmp_path
    )
    assert kept == [a, b]
    assert report.curation.audit["level_4"].unique_groups == 2


def test_explicit_parallel_bindings_are_rejected_not_silently_mixed(tmp_path):
    record = CurationInput(
        _two_calls("parallel", reverse=True), (BoundCallBatch(1, (2, 3), True),)
    )
    with pytest.raises(ValueError, match="non-parallel"):
        _run(
            [
                MixtureSource("ordered", [_two_calls("ordered")]),
                MixtureSource("parallel", [record]),
            ],
            tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_explicit_ordered_bindings_and_stale_argument_cache(tmp_path):
    first = _sample("first")
    second = _sample("second")
    second.messages[1]["tool_calls"][0]["function"]["arguments"] = '{"a":2}'
    records = [
        CurationInput(s, (BoundCallBatch(1, (2,)),), {(1, 0): {"a": 0}})
        for s in (first, second)
    ]
    kept, _ = _run([MixtureSource("source", records)], tmp_path)
    assert kept == [first, second]


@pytest.mark.parametrize("indices", [(3, 2), (2,), (2, 3, 4)])
def test_wrong_explicit_coordinates_rejected(tmp_path, indices):
    record = CurationInput(_two_calls("wrong"), (BoundCallBatch(1, indices),))
    with pytest.raises(ValueError, match="canonical order"):
        _run([MixtureSource("source", [record])], tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["orphan", "missing", "extra"])
def test_noncanonical_result_blocks_fail_and_clean_up(tmp_path, change):
    sample = _sample("bad")
    if change == "orphan":
        sample.messages.insert(0, {"role": "tool", "content": "orphan"})
    elif change == "missing":
        del sample.messages[2]
    else:
        sample.messages.insert(3, {"role": "tool", "content": "extra"})
    with pytest.raises(ValueError):
        _run([MixtureSource("source", [sample])], tmp_path)
    assert list(tmp_path.iterdir()) == []


class _OnePass:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0
        self.closed = False

    def __iter__(self):
        self.calls += 1
        assert self.calls == 1
        try:
            yield from self.rows
        finally:
            self.closed = True


class _Raw(dict):
    pass


def test_input_objects_are_not_accumulated_and_stream_is_consumed_once(tmp_path):
    references = []

    def rows():
        for i in range(250):
            assert sum(r() is not None for r in references) <= 2
            sample = _sample(i, user=str(i))
            sample.raw = _Raw(payload="raw" * 300)
            references.append(weakref.ref(sample.raw))
            yield sample

    source = _OnePass(rows())
    kept, report = _run([MixtureSource("source", source)], tmp_path)
    assert len(kept) == 250 and source.calls == 1 and source.closed
    assert report.complete
    assert all(r() is None for r in references)


def test_close_before_start_does_not_open_inputs_or_temp_files(tmp_path):
    source = _OnePass([_sample("a")])
    result = deduplicate_mixture(
        [MixtureSource("source", source)],
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
    )
    result.close()
    assert source.calls == 0 and not result.report.complete
    assert list(tmp_path.iterdir()) == []


def test_final_report_available_before_first_yield_and_early_close_cleans(tmp_path):
    source = _OnePass([_sample("a"), _sample("b", user="Different")])
    with deduplicate_mixture(
        [MixtureSource("source", source)],
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
    ) as result:
        next(iter(result))
        assert result.report.complete
        assert result.report.sources["source"].output_samples == 2
        assert source.closed
        assert list(tmp_path.iterdir())
    assert list(tmp_path.iterdir()) == []


def test_input_exception_closes_iterator_and_leaves_incomplete_report(tmp_path):
    def rows():
        yield _sample("a")
        raise RuntimeError("source failure")

    source = _OnePass(rows())
    result = deduplicate_mixture(
        [MixtureSource("source", source)],
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
    )
    with result, pytest.raises(RuntimeError, match="source failure"):
        list(result)
    assert source.closed and not result.report.complete
    assert list(tmp_path.iterdir()) == []


def test_empty_streams_have_complete_zero_reports(tmp_path):
    kept, report = _run(
        [MixtureSource("empty", []), MixtureSource("also empty", iter([]))], tmp_path
    )
    assert kept == [] and report.complete
    assert report.curation.input_samples == report.curation.output_samples == 0
    assert report.curation.audit["level_4"].unique_groups == 0
    assert list(report.sources) == ["empty", "also empty"]


def test_large_environment_is_never_split_at_partition_target(tmp_path, monkeypatch):
    original = mixture.curate
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(mixture, "curate", observed)
    rows = [_sample(i, user=str(i)) for i in range(20)]
    kept, _ = _run([MixtureSource("source", rows)], tmp_path, partition_size=1)
    assert kept == rows and len(calls) == 1


def test_ordinary_scope_dataset_guard_unchanged(tmp_path):
    with curate(
        [CurationInput(_sample("a", dataset="other"))],
        scope=CurationScope("dataset", "subset", "train"),
        config=CurationConfig(temporary_directory=tmp_path),
    ) as result:
        with pytest.raises(ValueError, match="dataset must match"):
            list(result)
    assert list(tmp_path.iterdir()) == []


def test_selected_callback_uses_original_positions_before_any_yield(tmp_path):
    positions = []
    rows = [
        _sample("long", answer="Very long"),
        _sample("short", answer="X"),
        _sample("other", environment="other"),
    ]
    with curate(
        (CurationInput(s) for s in rows),
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
        on_selected=positions.append,
    ) as result:
        assert next(iter(result)).sample_id == "short"
        assert positions == [1, 2]
    assert list(tmp_path.iterdir()) == []


def test_callback_exception_closes_core_spool(tmp_path):
    def fail(position):
        raise RuntimeError("callback failure")

    with curate(
        [CurationInput(_sample("a"))],
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
        on_selected=fail,
    ) as result:
        with pytest.raises(RuntimeError, match="callback failure"):
            list(result)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["", " ", None, 1])
def test_explicit_scope_names_and_source_names(value):
    with pytest.raises(ValueError):
        MixtureScope(value, "train")
    with pytest.raises(ValueError):
        MixtureScope("mixture", value)
    with pytest.raises(ValueError):
        MixtureSource(value, [])


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_partition_size_validation(value):
    with pytest.raises(ValueError, match="positive integer"):
        deduplicate_mixture([], scope=SCOPE, partition_size_bytes=value)


def test_duplicate_names_and_wrong_scope_fail_before_source_consumption():
    source = _OnePass([_sample("a")])
    with pytest.raises(ValueError, match="unique"):
        deduplicate_mixture(
            [MixtureSource("same", source), MixtureSource("same", [])], scope=SCOPE
        )
    with pytest.raises(TypeError, match="MixtureScope"):
        deduplicate_mixture([], scope=CurationScope("dataset", "subset", "train"))  # ty: ignore[invalid-argument-type]
    assert source.calls == 0


def test_named_stream_attribution_does_not_use_dataset_or_sample_id(tmp_path):
    first = _sample(7)
    second = _sample(7, user="Different visible input")
    kept, report = _run(
        [MixtureSource("one", [first]), MixtureSource("two", [second])], tmp_path
    )
    assert kept == [first, second]
    assert report.sources["one"].output_samples == 1
    assert report.sources["two"].output_samples == 1


def test_progress_is_cumulative_by_phase_and_partition_independent(tmp_path):
    events = []
    with deduplicate_mixture(
        [MixtureSource("one", [_sample("a"), _sample("b", environment="two")])],
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
        partition_size_bytes=1,
        progress=lambda phase, count: events.append((phase, count)),
    ) as result:
        assert len(list(result)) == 2
    assert events == [
        ("spooling", 1),
        ("spooling", 2),
        ("curating", 1),
        ("curating", 2),
        ("yielding", 1),
        ("yielding", 2),
    ]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure_phase", ["spooling", "curating", "yielding"])
def test_progress_guard_failure_cleans_all_resources(tmp_path, failure_phase):
    def guard(phase, count):
        if phase == failure_phase:
            raise RuntimeError("resource guard")

    with deduplicate_mixture(
        [MixtureSource("source", [_sample("a")])],
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
        progress=guard,
    ) as result:
        with pytest.raises(RuntimeError, match="resource guard"):
            list(result)
    assert list(tmp_path.iterdir()) == []


def test_partition_query_searches_selected_environments_not_full_sample_table():
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(
            "CREATE TABLE samples (position INTEGER PRIMARY KEY, environment INTEGER, payload BLOB);"
            "CREATE INDEX samples_environment ON samples(environment);"
            "CREATE TABLE work_environments (environment INTEGER PRIMARY KEY);"
        )
        details = [
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN " + mixture._PARTITION_ROWS
            )
        ]
    assert any("SCAN work_environments" in detail for detail in details)
    assert any(
        "SEARCH samples USING COVERING INDEX samples_environment (environment=?)"
        in detail
        for detail in details
    )
    assert not any("SCAN samples" in detail for detail in details)


@pytest.mark.parametrize("level", [None, "level_1", "level_1_5", "level_2"])
@pytest.mark.parametrize("audit", [False, True])
def test_compressed_comparisons_preserve_reports_positions_and_exact_payloads(
    tmp_path, level, audit
):
    import pickle

    rows = [
        _sample("long", answer="Long response"),
        _sample("exact", dataset="second", answer="Long response"),
        _sample("near", user="\nDo it\n", answer="Long response"),
        _sample("short", answer="X"),
        _sample("another environment", environment="other"),
    ]
    parallel = _two_calls("parallel", reverse=True)
    parallel_copy = _two_calls("parallel-copy")
    rows.extend([parallel, parallel_copy])
    rows[0].raw = {"large": 10**90, "tuple": (1, b"raw")}
    records = [CurationInput(s) for s in rows[:-2]] + [
        CurationInput(s, (BoundCallBatch(1, (2, 3), True),)) for s in rows[-2:]
    ]
    expected = None
    for compressed, payloads in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        positions = []
        with curate(
            iter(records),
            scope=SCOPE,
            config=CurationConfig(
                max_level=level, audit=audit, temporary_directory=tmp_path
            ),
            on_selected=positions.append,
            compress_comparisons=compressed,
            compress_payloads=payloads,
        ) as result:
            outputs = [
                pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL) for s in result
            ]
            current = outputs, asdict(result.reports[SCOPE]), positions
        if expected is None:
            expected = current
        else:
            assert current == expected
        assert list(tmp_path.iterdir()) == []


def test_payload_spool_does_not_update_large_rows_to_select_winners(tmp_path):
    with deduplicate_mixture(
        [MixtureSource("source", [_sample("a")])],
        scope=SCOPE,
        config=CurationConfig(temporary_directory=tmp_path),
    ) as result:
        next(iter(result))
        path = next(tmp_path.glob("fcanalysis-mixture-*/mixture.sqlite3"))
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            assert [
                row[1] for row in connection.execute("PRAGMA table_info(samples)")
            ] == ["position", "environment", "payload"]
            assert connection.execute(
                "SELECT position FROM selected_positions"
            ).fetchall() == [(0,)]
    assert list(tmp_path.iterdir()) == []
