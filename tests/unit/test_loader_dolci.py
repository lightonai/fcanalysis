"""Direct Dolci adversarial tests derived from the pinned source grammar."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.dolci import (
    DATASET_ID,
    DolciConfig,
    _convert_row,
    _parse_function_calls,
    _pipeline,
    iter_load,
    load,
)
from fcanalysis.loaders.normalization import Reject


def definition(name="f", **extra):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            **extra,
        },
    }


def message(role, content=None, calls=None, definitions=None):
    return {
        "role": role,
        "content": content,
        "function_calls": calls,
        "functions": None if definitions is None else json.dumps(definitions),
    }


def row(*, calls="f(x=1)", results=None, final="done", tools=None):
    return {
        "id": "source-1",
        "dataset_source": "source-subset",
        "messages": [
            message("system", "  exact system\n", definitions=tools or [definition()]),
            message("user", "  exact request\r\n"),
            message("assistant", calls=calls),
            *[
                message("environment", result)
                for result in (results if results is not None else ['{"x": 1}'])
            ],
            message("assistant", final),
        ],
    }


def process(raw, **config):
    pipe = _pipeline(FilterConfig(**config))
    return pipe.process(_convert_row(raw)), pipe


@pytest.mark.parametrize(
    "value,expected",
    [
        ("f(x=-1)", {"x": -1}),
        ("f(x=true)", {"x": True}),
        ("f(x=false)", {"x": False}),
        ("f(x=null)", {"x": None}),
        ("f(x=None)", {"x": None}),
        ("f(x=[1,2]*2)", {"x": [1, 2, 1, 2]}),
        ("f(x=1+2)", {"x": 3}),
        ("f(x=4/2)", {"x": 2.0}),
        ('f(x={"a":[1]})', {"x": {"a": [1]}}),
    ],
)
def test_bounded_call_values(value, expected):
    calls = _parse_function_calls(value)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == expected


@pytest.mark.parametrize(
    "source_call,arguments",
    [
        (
            'merge_dictionaries(dict1={"factorial": 720, "fact4": 24, '
            '"name": "Assistant"}, dict2={"fact_of_fact4": 620448401733239439360000})',
            {
                "dict1": {"factorial": 720, "fact4": 24, "name": "Assistant"},
                "dict2": {"fact_of_fact4": 620448401733239439360000},
            },
        ),
        (
            "utils.count_occurrences(lst=[25852016738884976640000, "
            "265252859812191058636308480000000, 119622221865977780250000000000])",
            {
                "lst": [
                    25852016738884976640000,
                    265252859812191058636308480000000,
                    119622221865977780250000000000,
                ]
            },
        ),
        (
            "math.prime_factorization(number=15511210043330985984000000)",
            {"number": 15511210043330985984000000},
        ),
    ],
)
def test_released_large_integer_calls_preserve_exact_values(source_call, arguments):
    # Exact call projections from pinned Dolci rows 85108, 106364 and 150476.
    # Serialization correctness does not certify their surrounding arithmetic.
    calls = _parse_function_calls(source_call)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == arguments


@pytest.mark.parametrize(
    "number", [620448401733239439360000, -620448401733239439360000]
)
def test_large_integer_survives_conversion_and_complete_row_validation(number):
    raw = row(calls=f"f(x={number})", results=[json.dumps({"x": number})])
    before = deepcopy(raw)
    state, _ = process(raw)
    assert state is not None
    arguments = state.sample.messages[2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"x": number}
    assert state.sample.messages[3]["content"] == json.dumps({"x": number})
    assert raw == before


@pytest.mark.parametrize(
    "value",
    [
        "f",
        "a.f",
        "f(1)",
        "f(**x)",
        "f(x=missing)",
        "f(x=1,x=2)",
        'f(x={"a":1,"a":2})',
        "f(x=(1,2))",
        "f(x={1,2})",
        "f(x=[1]*1000000000)",
        "f(x=10**100000000)",
        "f(x=1/0)",
        "f(x=True+1)",
        "f(x=nan)",
        "f(x=1e1000)",
        'f(x={1:"a"})',
        "f(x=[x for x in y])",
        "a = f(x=1)",
        "f(x=1);f(x=2)",
    ],
)
def test_unsupported_or_ambiguous_python_never_guessed(value):
    assert _parse_function_calls(value) is None


def test_dotted_names_and_parallel_order():
    calls = _parse_function_calls("a.f(x=1)\nb.g(x=2)")
    assert calls is not None
    assert [c["function"]["name"] for c in calls] == ["a.f", "b.g"]


def test_source_content_and_raw_isolation():
    raw = row()
    before = deepcopy(raw)
    converted = _convert_row(raw)
    assert converted.raw == before and converted.raw is raw
    assert converted.messages[0]["content"] == "  exact system\n"
    assert converted.messages[1]["content"] == "  exact request\r\n"
    assert converted.messages[2]["content"] is None
    converted.tools[0]["function"]["parameters"]["required"].append("z")
    converted.messages[2]["tool_calls"][0]["function"]["name"] = "changed"
    assert raw == before


@pytest.mark.parametrize("system", [None, "", "policy"])
def test_absent_empty_and_nonempty_system(system):
    raw = row()
    if system is None:
        raw["messages"].pop(0)
        # No calls need definitions in this source absence case.
        raw["messages"] = [message("user", "u"), message("assistant", "a")]
    else:
        raw["messages"][0]["content"] = system
    state, _ = process(raw)
    assert state is not None
    systems = [m for m in state.sample.messages if m["role"] == "system"]
    assert systems == (
        [] if system is None else [{"role": "system", "content": system}]
    )


def test_exact_duplicates_reconciled_before_validation():
    raw = row(tools=[definition(), definition()])
    state, pipe = process(raw)
    assert state is not None and len(state.sample.tools) == 1
    assert pipe.transforms["duplicate_tool_definitions_removed"] == 1


def test_conflict_not_rescued_by_override():
    raw = row(tools=[definition(description="a"), definition(description="b")])
    state, pipe = process(raw, system_message_override="new")
    assert state is None and pipe.drops == {"conflicting_duplicate_tool_names": 1}


def test_future_definition_is_not_front_loaded():
    raw = row()
    raw["messages"].insert(2, message("system", "later", definitions=[definition("g")]))
    with pytest.raises(Reject, match="unsupported_tool_definition_epoch"):
        _convert_row(raw)


@pytest.mark.parametrize("role", ["user", "environment", "system"])
def test_non_assistant_calls_quarantined(role):
    raw = row()
    raw["messages"][2]["role"] = role
    with pytest.raises(Reject, match="function_calls_on_"):
        _convert_row(raw)


@pytest.mark.parametrize("sentinel", ["true", "false", "None"])
def test_unverified_absent_call_sentinels_not_invented(sentinel):
    with pytest.raises(Reject, match="malformed_function_calls"):
        _convert_row(row(calls=sentinel))


def test_verified_null_call_sentinel():
    raw = row(calls="null")
    raw["messages"] = [raw["messages"][0], raw["messages"][1], raw["messages"][-1]]
    raw["messages"][-1]["function_calls"] = "null"
    state, _ = process(raw)
    assert state is not None


def test_exact_bundled_json_preserves_values_and_outside_whitespace():
    raw = row(calls="f(x=1)\nf(x=2)", results=['  {"a": 1}\n {"b": 2}  \n'])
    transforms = Counter()
    converted = _convert_row(raw, transforms)
    assert [m["content"] for m in converted.messages if m["role"] == "tool"] == [
        '  {"a": 1}',
        '{"b": 2}  \n',
    ]
    assert transforms == {"bundled_result_batches_split": 1}
    state, _ = process(raw)
    assert state is not None
    assert state.batches[0].parallel is True


@pytest.mark.parametrize(
    "result",
    [
        'prefix {"a":1}\n{"b":2}',
        '{"a":1}\n{"b":2} suffix',
        "first line\nsecond line",
        '[{"a":1},{"b":2}]',
        '{"a":1}{"b":2}',
        '{"a":1}\n{"b":NaN}',
        '{"a":1,"a":2}\n{"b":2}',
        '{"a":[}\n{"b":2}',
    ],
)
def test_bundled_heuristics_abstain(result):
    with pytest.raises(Reject, match="ambiguous_bundled_results"):
        _convert_row(row(calls="f(x=1)\nf(x=2)", results=[result]))


def test_split_results_keep_source_order():
    raw = row(calls="f(x=1)\nf(x=2)", results=["two", "one"])
    state, _ = process(raw)
    assert state is not None
    assert [m["content"] for m in state.sample.messages if m["role"] == "tool"] == [
        "two",
        "one",
    ]


@pytest.mark.parametrize(
    "results,reason",
    [([], "unbalanced_cardinality"), (["one", "two"], "unbalanced_cardinality")],
)
def test_missing_or_extra_results(results, reason):
    state, pipe = process(row(results=results))
    assert state is None and pipe.drops == {reason: 1}


def test_consecutive_assistants_preserved():
    raw = row()
    raw["messages"].insert(2, message("assistant", "  preparing  "))
    state, _ = process(raw)
    assert state is not None
    assert state.sample.messages[2:4] == [
        {"role": "assistant", "content": "  preparing  "},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": '{"x":1}'}}
            ],
        },
    ]


def test_reasoning_then_final_termination():
    raw = row(final="<think>private</think>")
    state, pipe = process(raw, strip_thinking=True)
    assert state is None and pipe.drops == {"empty_final_assistant": 1}
    state, pipe = process(raw, strip_thinking=False)
    assert state is None and pipe.drops == {"empty_final_assistant": 1}


def test_reasoning_text_outside_spans_unchanged():
    state, _ = process(
        row(final="  <think>private</think>answer\n"), strip_thinking=True
    )
    assert state.sample.messages[-1]["content"] == "  answer\n"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(new="field"),
        lambda r: r["messages"][1].update(new="field"),
        lambda r: r["messages"][1].update(role="alien"),
        lambda r: r["messages"][1].update(content=["unsupported"]),
    ],
)
def test_unknown_source_fields_quarantined(mutate):
    raw = row()
    mutate(raw)
    with pytest.raises(Reject):
        _convert_row(raw)


@pytest.mark.parametrize(
    "flag", ["drop_consecutive_text_text_assistant", "merge_text_fc_assistant"]
)
def test_unsafe_legacy_config_fails_explicitly(flag):
    with pytest.raises(ValueError):
        iter_load(DolciConfig(**{flag: True}))


def test_iter_load_count_scope_and_order(monkeypatch):
    first = row()
    second = deepcopy(first)
    second["id"] = "source-2"
    other = deepcopy(first)
    other["id"] = "source-3"
    other["dataset_source"] = "different"
    monkeypatch.setattr(
        "fcanalysis.loaders.dolci.parquet_rows",
        lambda *a, **k: iter([first, second, other]),
    )
    samples, report = load()
    assert [s.sample_id for s in samples] == ["source-1", "source-3"]
    assert all(s.dataset == DATASET_ID for s in samples)
    assert (
        report.raw_count,
        report.stage1_count,
        report.filtered_count,
        report.final_count,
    ) == (3, 3, 3, 2)
    assert len(report.dataset_config_transform_counts["curation"]) == 2


def test_streaming_report_incomplete_until_exhausted(monkeypatch):
    monkeypatch.setattr(
        "fcanalysis.loaders.dolci.parquet_rows", lambda *a, **k: iter([row()])
    )
    rows, report = iter_load(
        curation_config=CurationConfig(max_level=None, audit=False)
    )
    assert report.final_count is None
    assert next(rows).sample_id == "source-1"
    assert report.final_count is None
    rows.close()
    assert report.final_count is None


@pytest.mark.parametrize(
    "value",
    [
        "f(x=(((10**100)**100)**100)**100)",
        "f(x=([1]*60000)+([2]*60000))",
        "f(x=[[0]*100000]*100000)",
        "f(x=[[('a'+'b')]*100000]*100000)",
        "f(x=1**100000000.0)",
        "f(x=(1e308+1e308)-(1e308+1e308))",
        "f(x=(-2)**0.5)",
    ],
)
def test_intermediate_expansion_and_numeric_limits_reject_before_materializing(value):
    assert _parse_function_calls(value) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("f(x=(2**30)-(2**30))", {"x": 0}),
        ("f(x=4**0.5)", {"x": 2.0}),
        ("f(x=2**-2)", {"x": 0.25}),
        ("f(x=1e308/1e308)", {"x": 1.0}),
        ("f(x=[[1,2]*2]*2)", {"x": [[1, 2, 1, 2], [1, 2, 1, 2]]}),
        ("f(x=([1]+[2])+[3])", {"x": [1, 2, 3]}),
    ],
)
def test_ordinary_bounded_arithmetic_and_expansion_preserve_values(value, expected):
    calls = _parse_function_calls(value)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == expected


def test_evaluation_work_budget_is_shared_across_arguments_and_call_batch(monkeypatch):
    from fcanalysis.loaders import dolci

    budget_type = dolci._EvaluationBudget
    monkeypatch.setattr(dolci, "_EvaluationBudget", lambda: budget_type(remaining=80))
    assert _parse_function_calls("f(x=[0]*10)") is not None
    assert _parse_function_calls("f(x=[0]*10,y=[0]*10)") is None
    assert _parse_function_calls("f(x=[0]*10)\nf(x=[0]*10)") is None
