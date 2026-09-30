"""Adversarial native terminal acceptance, preservation and curation contracts."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from fcanalysis.loaders import nemotron_terminal as loader
from fcanalysis.loaders._nemotron_terminal_protocol import (
    CONFIRMATION,
    PROTOCOL_PREFIX,
    RESET_NOTICE,
    HANDOFF_PREFIX,
    expected_error,
    expected_warning_prefix,
    interpret,
    split_native,
)
from fcanalysis.loaders._terminus_json import TerminusJSONPlainParser
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig, CurationScope, curate
from fcanalysis.loaders.normalization import Reject
from fcanalysis.loaders.pipeline import RowState, remove_reasoning

SCREEN = "New Terminal Output:\nroot@host:/app#"
INITIAL = (
    PROTOCOL_PREFIX
    + "Inspect the files.\n\nCurrent terminal state:\nCurrent Terminal Screen:\nroot@host:/app#\n"
)
CONFIRM = "Current terminal state:\n" + SCREEN + "\n\n" + CONFIRMATION


def action(commands=None, complete=False, **fields):
    return json.dumps(
        {
            "analysis": "Inspect.",
            "plan": "Continue.",
            "commands": commands or [],
            "task_complete": complete,
            **fields,
        }
    )


def native(visible, reasoning="Reasoning\n```unfinished"):
    return "<think>" + reasoning + "</think>" + ("\n" + visible if visible else "")


def command(keys="ls\n", duration=0.1):
    return {"keystrokes": keys, "duration": duration}


def row(*middle, reasoning=True):
    texts = [INITIAL, *middle, action(complete=True), CONFIRM, action(complete=True)]
    return {
        "conversations": [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": native(text) if reasoning and i % 2 else text,
            }
            for i, text in enumerate(texts)
        ],
        "agent": "terminus-2",
        "model": "deepseek-ai/DeepSeek-V3.2",
        "model_provider": "hosted_vllm",
        "date": "2026-01-01",
        "task": "task",
        "episode": f"episode-{len(texts) // 2 - 1}",
        "run_id": "run",
        "trial_name": "trial",
        "enable_thinking": True,
    }


def process(raw, strip=False, override=None):
    sample = loader._convert_row(raw, "dataset_adapters", Counter())
    pipeline = loader._pipeline(
        FilterConfig(strip_thinking=strip, system_message_override=override)
    )
    state = pipeline.process(sample)
    return state, pipeline


def rejected(raw, reason, **kwargs):
    try:
        state, pipeline = process(raw, **kwargs)
    except Reject as exc:
        assert exc.reason == reason
    else:
        assert state is None
        assert pipeline.drops == {reason: 1}


@pytest.mark.parametrize("strip", [False, True])
def test_native_protocol_preserved_without_synthetic_tools_or_mutable_aliases(strip):
    raw = row(action([command('printf "<think>literal</think>"\\n\n')]), SCREEN)
    original = deepcopy(raw)
    state, _ = process(raw, strip)
    assert state is not None
    sample = state.sample
    assert sample.tools == []
    assert sample.sample_id == "run/trial"
    assert sample.dataset == loader.DATASET_ID + "/dataset_adapters"
    assert sample.raw == original
    assert sample.annotations == {}
    for a, b in zip(raw["conversations"], sample.messages, strict=True):
        assert a is not b
        assert a["role"] == b["role"]
        if a["role"] == "user":
            assert a == b
        else:
            assert b["content"] == split_native(a["content"])[1]
            assert ("reasoning_content" in b) is not strip
    assert "<think>literal</think>" in sample.messages[1]["content"]
    sample.messages[0]["content"] = "changed"
    sample.messages[-1]["content"] = "changed"
    assert raw == original


@pytest.mark.parametrize("reasoning", [False, True])
def test_valid_confirmation_and_empty_final_commands(reasoning):
    state, _ = process(row(reasoning=reasoning))
    assert state is not None
    assert json.loads(state.sample.messages[-1]["content"])["task_complete"] is True


@pytest.mark.parametrize("visible", ["", "null", "not JSON", '{"analysis": "x"}'])
def test_observed_parser_rejection_is_preserved_and_can_recover(visible):
    feedback = expected_error(interpret(visible).result)
    raw = row(visible, feedback)
    state, _ = process(raw)
    assert state is not None
    assert state.sample.messages[1]["content"] == visible
    assert state.sample.messages[2]["content"] == feedback
    if not visible:
        rejected(raw, "empty_assistant_after_transform", strip=True)
    else:
        assert process(raw, strip=True)[0] is not None


def test_reasoning_only_json_draft_is_never_executed():
    draft = action([command("rm important-file\n")])
    raw = row("", expected_error(interpret("").result))
    raw["conversations"][1]["content"] = native("", draft)
    state, _ = process(raw)
    assert state is not None
    assert not state.responses[1].result.commands
    assert state.sample.messages[1]["reasoning_content"] == draft
    rejected(raw, "empty_assistant_after_transform", strip=True)


@pytest.mark.parametrize(
    "visible",
    [
        action([{"keystrokes": "ls\n"}]),
        action([command(duration="slow")]),
        "Before\n" + action([command()]) + "\nAfter",
        action([command()])[:-1],
        action([{"keystrokes": "ls\n", "duration": 0.1, "extra": True}]),
        action([command("printf a"), command("b\n")]),
    ],
)
def test_warned_and_repaired_actions_preserve_original_text(visible):
    response = interpret(visible)
    assert not response.result.error and response.result.warning
    raw = row(visible, expected_warning_prefix(response.result) + SCREEN)
    state, _ = process(raw)
    assert state is not None
    assert state.sample.messages[1]["content"] == visible


@pytest.mark.parametrize("keys", ["", "C-c", "C-d", "cd /app\n", "printf a", "é\r\n"])
def test_polling_control_input_and_exact_bytes(keys):
    state, _ = process(row(action([command(keys)]), SCREEN))
    assert state is not None
    assert state.responses[1].result.commands[0].keystrokes == keys


def test_empty_command_array_still_has_screen_feedback():
    assert process(row(action(), SCREEN))[0] is not None


def test_error_text_in_a_real_terminal_screen_is_not_a_parser_rejection():
    visible = action([command("cat missing\n")])
    output = SCREEN + "\nERROR: file missing\n" + CONFIRMATION
    assert process(row(visible, output))[0] is not None


def test_supported_pending_confirmation_survives_parser_error():
    bad = "null"
    raw = row(
        action(complete=True), CONFIRM, bad, expected_error(interpret(bad).result)
    )
    # Remove the extra first completion produced by row(): the existing pending
    # request survives the parser error and final confirmation can terminate.
    del raw["conversations"][-3:-1]
    raw["episode"] = f"episode-{len(raw['conversations']) // 2 - 1}"
    assert process(raw)[0] is not None


def test_false_completion_clears_pending_state():
    raw = row(action(complete=True), CONFIRM, action(), SCREEN)
    assert process(raw)[0] is not None  # fresh request + confirmation required
    del raw["conversations"][-3:-1]
    raw["episode"] = f"episode-{len(raw['conversations']) // 2 - 1}"
    rejected(raw, "unconfirmed_terminal_ending")


def test_cannot_continue_after_second_completion():
    rejected(
        row(action(complete=True), CONFIRM, action(complete=True), SCREEN),
        "continuation_after_terminal_submission",
    )


def test_nonliteral_intermediate_completion_is_source_interpreted():
    raw = row()
    raw["conversations"][1]["content"] = action(complete="yes")
    assert process(raw)[0] is not None


@pytest.mark.parametrize(
    "final", [action([command()], True), action([command("")], True)]
)
def test_final_commands_including_polling_lack_observations(final):
    raw = row()
    raw["conversations"][-1]["content"] = native(final)
    rejected(raw, "final_commands_without_observation")


@pytest.mark.parametrize(
    "final",
    [
        action(complete=False),
        action(complete="true"),
        action(complete=1),
        action(complete=True, analysis=[]),
        action(complete=True)[:-1],
        action([{"bad": "ls"}], True),
    ],
)
def test_final_submission_requires_supported_exact_shape(final):
    raw = row()
    raw["conversations"][-1]["content"] = native(final)
    rejected(raw, "unsupported_final_submission")


@pytest.mark.parametrize("final", ["", "null", "done", '{"task_complete": true}'])
def test_final_parser_failures_do_not_establish_completion(final):
    raw = row()
    raw["conversations"][-1]["content"] = native(final)
    rejected(raw, "final_parser_rejection")


@pytest.mark.parametrize(
    "observation",
    [
        "",
        "invented answer",
        "Previous response had parsing errors:\nERROR: bad",
        SCREEN + "\n" + CONFIRMATION,
    ],
)
def test_first_completion_requires_actual_confirmation_envelope(observation):
    raw = row()
    raw["conversations"][2]["content"] = observation
    rejected(raw, "missing_terminal_confirmation")


@pytest.mark.parametrize(
    "bad_feedback", [SCREEN, expected_error(interpret("null").result) + "changed"]
)
def test_parser_rejection_cannot_be_replaced_by_screen_or_wrong_error(bad_feedback):
    rejected(row("null", bad_feedback), "contradictory_parser_error_feedback")


def test_accepted_command_cannot_have_parser_error_feedback():
    rejected(
        row(action([command()]), expected_error(interpret("null").result)),
        "contradictory_terminal_feedback",
    )


@pytest.mark.parametrize("notice", [RESET_NOTICE, HANDOFF_PREFIX + " unknown context"])
def test_known_unrecoverable_context_replacements(notice):
    rejected(row(action([command()]), notice), "unrecoverable_terminal_context_reset")


def test_literal_reset_marker_in_terminal_screen_does_not_create_a_reset():
    assert (
        process(row(action([command()]), SCREEN + "\n" + RESET_NOTICE))[0] is not None
    )


@pytest.mark.parametrize(
    "content",
    ["<think>no close", "<think>x</think>\ny</think>\nz", "<think></think>\n{}"],
)
def test_unproven_native_boundaries_are_excluded(content):
    with pytest.raises(Reject):
        split_native(content)


@pytest.mark.parametrize(
    "thinking",
    ["```unclosed", '"unterminated', "<think>literal open", "  reasoning\r\n"],
)
def test_opaque_native_reasoning_not_parsed_as_markdown(thinking):
    visible = "\n" + action()
    assert split_native(native(visible, thinking)) == (thinking, visible)


def test_visible_literal_tags_are_not_native_fields():
    visible = action([command('echo "<think>x</think>"\n')])
    assert split_native(visible) == (None, visible)
    assert process(row(visible, SCREEN, reasoning=False), strip=True)[0] is not None


@pytest.mark.parametrize(
    "visible",
    [
        '{"analysis":"a","plan":"p","commands":[],"commands":[{"keystrokes":"rm x"}]}',
        '{"analysis":"a","plan":"p","commands":[{"keystrokes":"x","duration":NaN}]}',
        '{"analysis":"a","plan":"p","commands":[{"keystrokes":"x","duration":1e999}]}',
    ],
)
def test_duplicate_keys_and_nonfinite_values_never_silently_bind(visible):
    with pytest.raises(Reject):
        interpret(visible)


@pytest.mark.parametrize("duration", [-1, -0.1])
def test_negative_wait_is_not_supported(duration):
    with pytest.raises(Reject, match="unsupported_command_duration"):
        interpret(action([command(duration=duration)]))


@pytest.mark.parametrize(
    "field,value", [("agent", "other"), ("model", "other"), ("enable_thinking", False)]
)
def test_unrecognized_producer_is_not_assumed_compatible(field, value):
    raw = row()
    raw[field] = value
    rejected(raw, "unsupported_source_producer")


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("episode", "source_episode_mismatch"),
        ("unknown_role", "unknown_source_message_shape"),
        ("extra_message_field", "unknown_source_message_shape"),
        ("extra_row_field", "unknown_source_row_shape"),
        ("interface", "unsupported_terminal_interface"),
        ("initial_screen", "missing_initial_terminal_context"),
    ],
)
def test_source_shape_and_context_gates(mutation, reason):
    raw = row()
    if mutation == "episode":
        raw["episode"] = "episode-999"
    elif mutation == "unknown_role":
        raw["conversations"][2]["role"] = "tool"
    elif mutation == "extra_message_field":
        raw["conversations"][1]["tool_calls"] = []
    elif mutation == "extra_row_field":
        raw["hidden_prompt"] = "unreleased"
    elif mutation == "interface":
        raw["conversations"][0]["content"] = "Use other tools."
    else:
        raw["conversations"][0]["content"] = PROTOCOL_PREFIX + "Task only."
    rejected(raw, reason)


@pytest.mark.parametrize("strip", [False, True])
def test_system_override_is_checked_in_final_view(strip):
    assert process(row(), strip=strip, override="")[0] is not None
    rejected(
        row(),
        "unsupported_terminal_context_override",
        strip=strip,
        override="Authenticate before every action.",
    )


def comparison(raw):
    state, _ = process(raw)
    assert state is not None
    item = loader._TerminalInput(sample=state.sample, responses=state.responses)
    return item, loader._TerminalComparison(item)


def test_curation_preserves_changed_actions_results_order_duration_and_completion():
    base = row(action([command("a\n"), command("b\n")]), SCREEN)
    _, view = comparison(base)
    for changed in [
        row(action([command("b\n"), command("a\n")]), SCREEN),
        row(action([command("a\n"), command("b\n", duration=0.5)]), SCREEN),
        row(action([command("a\n"), command("c\n")]), SCREEN),
        row(action([command("a\n"), command("b\n")]), SCREEN + "changed"),
    ]:
        _, other = comparison(changed)
        assert view.key("level_2") != other.key("level_2")
    other = deepcopy(base)
    other["conversations"][1]["content"] = native(
        action(
            [command("a\n"), command("b\n")],
            analysis="Much longer explanation.",
            plan="Other plan",
        ),
        "different native trace",
    )
    _, different_prose = comparison(other)
    assert view.key("level_1") != different_prose.key("level_1")
    assert view.key("level_2") == different_prose.key("level_2")
    assert view.key("level_1_5") == view.key("level_1")


def test_curation_abstains_from_omitting_rejected_or_repaired_text():
    a = action([command()])[:-1]
    b = a + " "
    _, left = comparison(row(a, expected_warning_prefix(interpret(a).result) + SCREEN))
    _, right = comparison(row(b, expected_warning_prefix(interpret(b).result) + SCREEN))
    assert left.key("level_2") != right.key("level_2")


def test_terminal_audit_views_keep_feedback_and_count_accepted_action_batches():
    failure = "not JSON"
    raw = row(
        action([command("a\n"), command("b\n")]),
        SCREEN,
        failure,
        expected_error(interpret(failure).result),
        action(),
        SCREEN,
    )
    _, view = comparison(raw)
    assert view.has_calls()
    assert json.loads(view.key("level_4")) == [
        ["send_keys", "send_keys"],
        [],
        ["completion_request"],
        ["completion_request"],
    ]
    assert json.loads(view.key("level_5")) == PROTOCOL_PREFIX

    other = deepcopy(raw)
    other["conversations"][0]["content"] = INITIAL.replace(
        "Inspect the files.", "Perform a different task."
    )
    _, other_view = comparison(other)
    assert view.key("level_2") != other_view.key("level_2")
    assert view.key("level_3") == other_view.key("level_3")
    other["conversations"][2]["content"] += "\nDifferent observed output"
    _, changed_feedback = comparison(other)
    assert view.key("level_3") != changed_feedback.key("level_3")
    assert view.key("level_4") == changed_feedback.key("level_4")
    assert view.key("level_5") == changed_feedback.key("level_5")
    _, no_commands = comparison(row())
    assert not no_commands.has_calls()


def test_explicit_wait_above_runtime_cap_is_preserved_as_source_text():
    visible = action([command("", duration=120)])
    state, _ = process(row(visible, SCREEN))
    assert state is not None
    assert state.sample.messages[1]["content"] == visible
    assert state.responses[1].result.commands[0].duration == 120


def test_shared_disk_selector_returns_unchanged_original_winner(tmp_path):
    longer = row(
        action([command()], analysis="Very long explanation of the command."), SCREEN
    )
    shorter = row(action([command()]), SCREEN)
    longer["trial_name"] = "first"
    shorter["trial_name"] = "second"
    items = [comparison(r)[0] for r in [longer, shorter, shorter]]
    scope = CurationScope(items[0].sample.dataset, "dataset_adapters", "train")
    with curate(
        items,
        scope=scope,
        config=CurationConfig(temporary_directory=tmp_path),
        comparison_factory=loader._TerminalComparison,
    ) as result:
        output = list(result)
        assert len(output) == 1
        assert output[0] == items[1].sample
        assert output[0].raw == shorter
        report = result.reports[scope]
        assert report.stages["level_1"].removed_samples == 1
        assert report.stages["level_2"].removed_samples == 1
        assert report.audit["level_3"].eligible_samples == 1
    assert not list(tmp_path.iterdir())


def test_falsey_comparison_factory_cannot_silently_erase_terminal_actions(tmp_path):
    class Factory:
        def __bool__(self):
            return False

        def __call__(self, record):
            return loader._TerminalComparison(record)

    items = [
        comparison(row(action([command(text)]), SCREEN))[0]
        for text in ["ls\n", "pwd\n"]
    ]
    scope = CurationScope(items[0].sample.dataset, "dataset_adapters", "train")
    with curate(
        items,
        scope=scope,
        config=CurationConfig(temporary_directory=tmp_path),
        comparison_factory=Factory(),
    ) as result:
        assert list(result) == [item.sample for item in items]


@pytest.mark.parametrize("config", [[], ["missing"], ["dataset_adapters"] * 2])
def test_invalid_config_selection(config):
    with pytest.raises(ValueError):
        loader.NemotronTerminalConfig(configs=config)


def test_legacy_lossy_options_are_not_silently_accepted():
    with pytest.raises(TypeError):
        loader.NemotronTerminalConfig(strip_malformed=True)  # ty: ignore[unknown-argument]


def test_streaming_report_and_source_order(monkeypatch, tmp_path):
    raw = row(action([command()]), SCREEN)
    bad = row()
    bad["conversations"][-1]["content"] = native(action([command()], True))

    def source(*args, **kwargs):
        yield deepcopy(raw)
        yield deepcopy(bad)

    monkeypatch.setattr(loader, "parquet_rows", source)
    monkeypatch.setattr(loader, "_FILES", {"dataset_adapters": ["one.parquet"]})
    stream, report = loader.iter_load(
        loader.NemotronTerminalConfig(["dataset_adapters"]),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert report.final_count is None and report.raw_count == 0
    out = list(stream)
    assert len(out) == report.final_count == report.filtered_count == 1
    assert report.raw_count == report.stage1_count == 2
    assert report.filter_drop_reasons == {"final_commands_without_observation": 1}
    assert not list(tmp_path.iterdir())


def test_early_close_has_no_completed_report_and_releases_spool(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "parquet_rows", lambda *a, **k: iter([row()]))
    monkeypatch.setattr(loader, "_FILES", {"dataset_adapters": ["one.parquet"]})
    stream, report = loader.iter_load(
        loader.NemotronTerminalConfig(["dataset_adapters"]),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    next(stream)
    stream.close()
    assert report.final_count is None
    assert not list(tmp_path.iterdir())


def test_shared_reasoning_default_unchanged_and_structured_only_is_surgical():
    raw = row()
    sample = loader._convert_row(raw, "dataset_adapters", Counter())
    sample.messages[1]["content"] = "<think>literal</think>content"
    old = deepcopy(sample)
    remove_reasoning(RowState(old))
    assert old.messages[1]["content"] == "content"
    remove_reasoning(RowState(sample), inline=False)
    assert sample.messages[1]["content"] == "<think>literal</think>content"
    assert "reasoning_content" not in sample.messages[1]


@pytest.mark.parametrize(
    "visible",
    [
        action(),
        action([command()]),
        "prefix " + action() + " suffix",
        action([{"keystrokes": "ls\n"}]),
        action([command(duration="later")]),
        action([command()])[:-1],
        "null",
        "",
        '{"analysis": 2}',
        action([{"bad": "key"}], True),
        action(complete="yes"),
    ],
)
def test_fast_parser_matches_unmodified_upstream_interpretation(visible):
    from dataclasses import asdict

    assert asdict(interpret(visible).result) == asdict(
        TerminusJSONPlainParser().parse_response(visible)
    )


@pytest.mark.parametrize(
    "wrap", [lambda text: "before " + text, lambda text: "```json\n" + text + "\n```"]
)
def test_final_wrapped_json_is_a_supported_parser_submission(wrap):
    raw = row()
    raw["conversations"][-1]["content"] = native(wrap(action(complete=True)))
    assert process(raw)[0] is not None


def test_duplicate_visible_prose_keys_preserved_without_ambiguous_actions():
    visible = '{"analysis":"first","analysis":"last","plan":"p","commands":[],"task_complete":false}'
    r = interpret(visible)
    assert r.duplicate_prose_fields == 1
    assert r.result.analysis == "last"
    assert r.original_object is None  # Level 2 must abstain
    state, _ = process(row(visible, SCREEN))
    assert state is not None
    assert state.sample.messages[1]["content"] == visible


def test_unknown_field_warning_order_is_not_execution_order():
    visible = action([{"keystrokes": "ls\n", "duration": 0.1, "z": 1, "a": 2}])
    r = interpret(visible)
    prefix = expected_warning_prefix(r.result)
    names = prefix.split("Unknown fields: ", 1)[1].split("\n", 1)[0]
    changed = prefix.replace(
        "Unknown fields: " + names,
        "Unknown fields: " + ", ".join(reversed(names.split(", "))),
    )
    assert process(row(visible, changed + SCREEN))[0] is not None
    rejected(
        row(
            visible,
            changed.replace("Unknown fields: ", "Unknown fields: extra, ") + SCREEN,
        ),
        "contradictory_terminal_feedback",
    )


def test_legacy_json_trailing_comma_error_requires_exact_source_location():
    from fcanalysis.loaders._nemotron_terminal_protocol import error_feedback_matches

    text = '{\n "analysis":"x",\n}'
    response = interpret(text)
    if "Illegal trailing comma" not in response.result.error:
        pytest.skip("Current Python decoder already uses legacy wording")
    consumed = response.consumed_text
    close = consumed.rindex("}")
    legacy = "Invalid JSON: " + str(
        json.JSONDecodeError(
            "Expecting property name enclosed in double quotes", consumed, close
        )
    )
    suffix = response.result.error.split(" | Content", 1)[1]
    old = expected_error(response.result).replace(
        response.result.error, legacy + " | Content" + suffix
    )
    assert error_feedback_matches(response, old)
    assert process(row(text, old))[0] is not None
    assert not error_feedback_matches(
        response, old.replace(f"(char {close})", "(char 0)")
    )


def test_source_reader_local_root_does_not_resolve_network(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fcanalysis.loaders import source

    pq.write_table(pa.Table.from_pylist([{"a": 1}, {"a": 2}]), tmp_path / "x.parquet")

    def fail(*args):
        raise AssertionError("Unexpected remote resolver")

    monkeypatch.setattr(source, "pinned_file", fail)
    assert list(
        source.parquet_rows(
            "dataset", "revision", ["x.parquet"], local_root=tmp_path, batch_size=1
        )
    ) == [{"a": 1}, {"a": 2}]


def test_unrepresentable_command_unicode_is_a_row_rejection_not_spool_failure():
    with pytest.raises(Reject, match="unsupported_protocol_unicode"):
        interpret(action([command("\ud800")]))


def test_huge_integer_parser_limit_is_a_row_rejection_not_loader_crash():
    text = (
        '{"analysis":"a","plan":"p","commands":[{"keystrokes":"x","duration":'
        + "9" * 5000
        + "}]}"
    )
    with pytest.raises(Reject):
        interpret(text)


@pytest.mark.parametrize("name", ["overlap", "semantic"])
def test_native_call_analysis_entry_points_do_not_misclassify_terminal_data(name):
    from fcanalysis.overlap import _load_dataset
    from fcanalysis.semantic import _load_samples

    with pytest.raises(ValueError, match="textual terminal actions"):
        if name == "overlap":
            _load_dataset("nemotron_terminal")
        else:
            _load_samples("nemotron_terminal", "")


def test_invalid_unicode_json_key_cannot_escape_into_curation():
    text = json.dumps({"analysis": "a", "plan": "p", "commands": [], "\ud800": "value"})
    with pytest.raises(Reject, match="unsupported_protocol_unicode"):
        interpret(text)


def test_completion_discards_entire_invalid_batch_without_inventing_execution():
    raw = row()
    bad = action([command("touch file\n"), {"bad": "missing keystrokes"}], True)
    raw["conversations"][1]["content"] = native(bad)
    state, _ = process(raw)
    assert state is not None
    assert state.responses[1].result.commands == []
    assert state.responses[1].result.warning
    assert state.sample.messages[1]["content"] == bad
    assert state.sample.messages[2]["content"] == CONFIRM


def test_parser_error_discards_valid_commands_before_a_malformed_one():
    text = action([command("touch file\n"), {"bad": "missing keystrokes"}])
    response = interpret(text)
    assert response.result.error and not response.result.commands
    assert process(row(text, expected_error(response.result)))[0] is not None


def test_error_marker_in_parser_warning_follows_the_observed_harness_branch():
    visible = action(
        [{"keystrokes": "ls\n", "duration": 0.1, "ERROR:": "unknown field"}]
    )
    response = interpret(visible)
    assert not response.result.error and response.result.commands
    assert "ERROR:" in response.result.warning
    state, _ = process(row(visible, expected_error(response.result)))
    assert state is not None
    assert state.issues["interpreted_commands"] == 0
    assert state.issues["parser_rejected_assistants"] == 1
