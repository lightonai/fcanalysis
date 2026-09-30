"""Source-prefix restoration must prove donors and preserve continuations."""

from copy import deepcopy
from dataclasses import replace

import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.curation import CurationScope
from fcanalysis.loaders.reconstruction import (
    PrefixRecord,
    ReconstructionConfig,
    reconstruct_prefixes,
)


SCOPE = CurationScope("source", "effort", "train")
USER = {"role": "user", "content": "Please Check BOTH Items.\n"}
FIRST = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "Exact reasoning. Capitals, punctuation, and  spaces.\n",
    "tool_calls": [
        {"name": "lookup", "arguments": '{ "key": "A" }'},
        {"name": "lookup", "arguments": '{ "key": "B" }'},
    ],
}
RESULTS = [
    {"role": "tool", "content": '{"value":"A"}'},
    {"role": "tool", "content": '{"value":"B"}'},
]
FINAL = {
    "role": "assistant",
    "content": "Both are present.",
    "reasoning_content": "Use the two returned values.",
}
TOOLS = [{"name": "lookup", "parameters": {"type": "object"}}]


def _project(message):
    return {k: v for k, v in message.items() if k != "reasoning_content"}


def _record(name, messages, *, scope=SCOPE, tools=None, required=None, anchor=True):
    messages = deepcopy(messages)
    tools = deepcopy(TOOLS if tools is None else tools)
    raw = {"messages": messages, "tools": tools, "id": name}
    return PrefixRecord(
        value=raw,
        scope=scope,
        context=tools,
        messages=messages,
        projected_messages=[_project(message) for message in messages],
        required=tuple(
            index
            for index, message in enumerate(messages[:-1])
            if message["role"] == "assistant" and "reasoning_content" not in message
        )
        if required is None
        else required,
        anchor=len(messages) - 1 if anchor else None,
    )


def _validate(raw, messages):
    return ConversationSample(
        messages=messages,
        tools=deepcopy(raw["tools"]),
        dataset="source",
        sample_id=raw["id"],
        raw=raw,
    )


def _pair():
    return (
        _record("short", [USER, FIRST]),
        _record("long", [USER, _project(FIRST), *RESULTS, FINAL]),
    )


def _run(records, validate=_validate, **kwargs):
    with reconstruct_prefixes(records, validate, **kwargs) as run:
        return list(run), run.report


def test_nonadjacent_reverse_order_restores_all_bytes_and_keeps_longest_raw():
    short, long = _pair()
    other = _record("other", [{**USER, "content": "Other question"}, FINAL])
    before = deepcopy([long, other, short])
    kept, report = _run(iter([long, other, short]))
    assert [sample.sample_id for sample in kept] == ["long", "other"]
    assert kept[0].messages == [USER, FIRST, *RESULTS, FINAL]
    assert kept[0].raw == long.value
    assert "reasoning_content" not in kept[0].raw["messages"][1]
    assert [long, other, short] == before
    kept[0].messages[1]["tool_calls"][0]["name"] = "changed"
    assert kept[0].raw["messages"][1]["tool_calls"][0]["name"] == "lookup"
    assert report.input_rows == 3
    assert report.restored_messages == 1
    assert report.removed_prefix_rows == 1
    assert report.output_rows == 2


def test_missing_donor_is_quarantined_even_if_validation_would_strip_reasoning():
    _, long = _pair()
    calls = []

    def validate(raw, messages):
        calls.append(raw)
        return _validate(raw, [_project(message) for message in messages])

    kept, report = _run([long], validate)
    assert kept == []
    assert calls == []
    assert report.missing_donor_rows == 1


def test_conflicting_native_donors_are_not_selected_by_order_or_future_results():
    short, long = _pair()
    conflict = _record(
        "alternative",
        [USER, {**FIRST, "reasoning_content": "Different original reasoning."}],
    )
    for records in ([short, conflict, long], [long, conflict, short]):
        kept, report = _run(records)
        assert {sample.sample_id for sample in kept} == {"short", "alternative"}
        assert report.ambiguous_donor_rows == 1
        assert report.removed_prefix_rows == 0


@pytest.mark.parametrize("value", ["", None])
def test_explicit_empty_native_anchor_is_a_real_donor(value):
    short = _record("short", [USER, {**FIRST, "reasoning_content": value}])
    _, long = _pair()
    kept, _ = _run([short, long])
    assert kept[0].messages[1]["reasoning_content"] is value


def test_complete_anchor_without_native_field_can_establish_genuine_absence():
    short = _record("short", [USER, _project(FIRST)])
    _, long = _pair()
    kept, report = _run([short, long])
    assert len(kept) == 1
    assert "reasoning_content" not in kept[0].messages[1]
    assert report.restored_messages == 1


def test_duplicate_identical_donors_are_unambiguous_and_curation_owns_duplicates():
    short, long = _pair()
    duplicate_short = replace(short, value={**short.value, "id": "short-copy"})
    duplicate_long = replace(long, value={**long.value, "id": "long-copy"})
    kept, report = _run([short, duplicate_short, long, duplicate_long])
    assert [sample.sample_id for sample in kept] == ["long", "long-copy"]
    assert report.ambiguous_donor_rows == 0
    assert report.removed_prefix_rows == 2


@pytest.mark.parametrize("change", ["scope", "tools", "user", "calls", "system"])
def test_same_first_question_or_tool_names_do_not_prove_a_donor(change):
    short, long = _pair()
    if change == "scope":
        short.scope = replace(SCOPE, subset="different-effort")
    elif change == "tools":
        short.context = [{**TOOLS[0], "description": "Other contract"}]
    elif change == "user":
        short.messages[0]["content"] += " "
        short.projected_messages = [_project(message) for message in short.messages]
    elif change == "calls":
        short.messages[1]["tool_calls"].reverse()
        short.projected_messages = [_project(message) for message in short.messages]
    else:
        short = _record("short", [{"role": "system", "content": "Policy"}, USER, FIRST])
    kept, report = _run([short, long])
    assert [sample.sample_id for sample in kept] == ["short"]
    assert report.missing_donor_rows == 1


def test_linked_results_and_full_history_must_match_for_later_donors():
    short, middle = _pair()
    later = _record(
        "later",
        [USER, _project(FIRST), *RESULTS, _project(FINAL), USER, FINAL],
    )
    middle.messages[2]["content"] = '{"value":"OTHER"}'
    middle.projected_messages = [_project(message) for message in middle.messages]
    kept, report = _run([short, middle, later])
    assert [sample.sample_id for sample in kept] == ["long"]
    assert report.missing_donor_rows == 1


def test_multiple_assistants_within_one_user_turn_are_restored_independently():
    short, middle = _pair()
    later = _record(
        "later",
        [USER, _project(FIRST), *RESULTS, _project(FINAL), USER, FINAL],
    )
    kept, report = _run([later, short, middle])
    assert [sample.sample_id for sample in kept] == ["later"]
    assert kept[0].messages[1] == FIRST
    assert kept[0].messages[4] == FINAL
    assert len(kept[0].messages[1]["tool_calls"]) == 2
    assert report.restored_messages == 3
    assert report.removed_prefix_rows == 2


def test_invalid_later_candidate_cannot_delete_a_valid_complete_prefix():
    short, long = _pair()

    def validate(raw, messages):
        return None if raw["id"] == "long" else _validate(raw, messages)

    kept, report = _run([long, short], validate)
    assert [sample.sample_id for sample in kept] == ["short"]
    assert report.validation_dropped_rows == 1
    assert report.removed_prefix_rows == 0


def test_two_valid_continuations_stay_separate():
    short, long = _pair()
    alternate = _record(
        "alternate",
        [USER, _project(FIRST), *RESULTS, {**FINAL, "content": "Another answer"}],
    )
    kept, report = _run([short, alternate, long])
    assert [sample.sample_id for sample in kept] == ["alternate", "long"]
    assert report.removed_prefix_rows == 1


def test_final_transform_cannot_create_a_source_prefix_proof():
    short, long = _pair()
    # This is an already complete history of a different reasoning variant.
    long.messages[1] = {**FIRST, "reasoning_content": "Other reasoning"}
    long.required = ()
    long.projected_messages = [*long.messages[:-1], _project(long.messages[-1])]

    def strip(raw, messages):
        for message in messages:
            message.pop("reasoning_content", None)
        return _validate(raw, messages)

    kept, report = _run([short, long], strip)
    assert [sample.sample_id for sample in kept] == ["short", "long"]
    assert report.removed_prefix_rows == 0


def test_failure_and_early_close_remove_spooled_payloads(tmp_path):
    config = ReconstructionConfig(temporary_directory=tmp_path)

    def fail(raw, messages):
        raise RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        _run(_pair(), fail, config=config)
    assert list(tmp_path.iterdir()) == []
    run = reconstruct_prefixes(_pair(), _validate, config=config)
    assert next(iter(run)).sample_id == "long"
    assert list(tmp_path.iterdir())
    run.close()
    assert list(tmp_path.iterdir()) == []


def test_bad_record_contracts_fail_explicitly():
    short, _ = _pair()
    for kwargs in (
        {"anchor": 0},
        {"required": (1,)},
        {"required": (-1,)},
        {"required": (0, 0)},
        {"projected_messages": []},
        {"messages": [], "projected_messages": []},
    ):
        with pytest.raises(ValueError):
            replace(short, **kwargs)


def test_projection_must_not_hide_known_history_content():
    _, long = _pair()
    long.projected_messages[0] = {"role": "user", "content": ""}
    with pytest.raises(ValueError, match="known non-anchor history"):
        _run([long])
