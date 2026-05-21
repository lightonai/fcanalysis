import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .._gc import gc_disabled
from ..format import ConversationSample
from .base import FilterConfig, LoadReport, apply_filters


DATASET_ID = "nvidia/Nemotron-Terminal-Corpus"
DATASET_REVISION = "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9"

_FILES: dict[str, list[str]] = {
    "dataset_adapters": [
        "dataset_adapters/code.parquet",
        "dataset_adapters/math.parquet",
        "dataset_adapters/swe.parquet",
    ],
    "skill_based_easy": [
        "synthetic_tasks/skill_based/easy/data_processing/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/data_querying/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/data_science/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/debugging/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/dependency_management/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/file_operations/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/scientific_computing/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/security/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/software_engineering/data_filtered.parquet",
    ],
    "skill_based_medium": [
        "synthetic_tasks/skill_based/medium/data_processing/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/data_querying/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/data_science/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/debugging/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/dependency_management/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/file_operations/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/scientific_computing/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/security/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/software_engineering/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/system_administration/data_filtered.parquet",
    ],
    "skill_based_mixed": [
        "synthetic_tasks/skill_based/mixed/data_processing/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/data_science/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/debugging/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/file_operations/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/scientific_computing/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/security/data_filtered.parquet",
    ],
}

ALL_CONFIGS = list(_FILES.keys())

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_commands",
        "description": "Execute a batch of shell commands in the Linux terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": "Array of command objects to execute sequentially.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keystrokes": {
                                "type": "string",
                                "description": "Exact keystrokes to send to the terminal.",
                            },
                            "duration": {
                                "type": "number",
                                "description": "Seconds to wait for command completion.",
                            },
                        },
                        "required": ["keystrokes"],
                    },
                },
            },
            "required": ["commands"],
        },
    },
}

_TERMINAL_OUTPUT_PREFIXES = (
    "New Terminal Output:",
    "Current terminal state:",
    "Current Terminal Screen:",
)

# Fatal: the framework could not parse the assistant's JSON, nothing was
# executed. Warnings (non-fatal) always carry embedded terminal output, so
# they are detected positionally without a separate prefix constant.
_PARSE_ERROR_PREFIX = "Previous response had parsing errors:"


@dataclass(slots=True)
class NemotronTerminalConfig:
    strip_malformed: bool = False
    drop_orphans: bool = False
    drop_incomplete: bool = False
    configs: list[str] | None = None


@dataclass(slots=True)
class _ConvertState:
    call_id_counter: int = 0
    orphan_counter: int = 0
    has_task_complete: bool = False


def _parse_assistant_json(content: str) -> dict[str, Any] | None:
    cleaned = _THINK_RE.sub("", content).strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError, TypeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _split_first_user_message(content: str) -> tuple[str, str]:
    # The template normally ends with "Task Description:\n"; the no-newline
    # variant is a documented fallback.
    for marker in ("Task Description:\n", "Task Description:"):
        idx = content.find(marker)
        if idx != -1:
            return content[:idx].strip(), content[idx + len(marker) :].strip()
    return "", content


def _is_terminal_output(content: str) -> bool:
    return content.startswith(_TERMINAL_OUTPUT_PREFIXES)


def _has_valid_commands(parsed: dict[str, Any]) -> bool:
    commands = parsed.get("commands")
    if not isinstance(commands, list) or not commands:
        return False
    return all(isinstance(cmd, dict) and "keystrokes" in cmd for cmd in commands)


def _next_raw_has_terminal(
    raw_conversations: list[dict[str, Any]], msg_idx: int
) -> bool:
    if msg_idx + 1 >= len(raw_conversations):
        return False
    nxt = raw_conversations[msg_idx + 1]
    if nxt.get("role") != "user":
        return False
    content = nxt.get("content") or ""
    if content.startswith(_TERMINAL_OUTPUT_PREFIXES):
        return True
    return any(prefix in content for prefix in _TERMINAL_OUTPUT_PREFIXES)


def _strip_think(content: str) -> str:
    return _THINK_RE.sub("", content).strip()


def _extract_assistant_text(parsed: dict[str, Any]) -> str:
    parts = [p for p in (parsed.get("analysis"), parsed.get("plan")) if p]
    return "\n\n".join(parts)


def _emit_tool_response(
    messages: list[dict[str, Any]],
    content: str,
    pending_call_ids: list[str],
    state: _ConvertState,
) -> None:
    if pending_call_ids:
        tc_id = pending_call_ids.pop(0)
    else:
        state.orphan_counter += 1
        tc_id = f"call_orphan_{state.orphan_counter}"
    messages.append({"role": "tool", "content": content, "tool_call_id": tc_id})


def _handle_user_message(
    content: str,
    msg_idx: int,
    messages: list[dict[str, Any]],
    pending_call_ids: list[str],
    state: _ConvertState,
    issues: Counter[str],
) -> None:
    if msg_idx == 0:
        system_text, task_text = _split_first_user_message(content)
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": task_text})
        return

    if _is_terminal_output(content):
        _emit_tool_response(messages, content, pending_call_ids, state)
        return

    # Feedback message, possibly with embedded terminal output (warning +
    # "New Terminal Output:..." pattern). Split into feedback + tool response
    # when embedded; emit as pure user message otherwise.
    for prefix in _TERMINAL_OUTPUT_PREFIXES:
        embedded_idx = content.find(prefix)
        if embedded_idx > 0:
            messages.append({"role": "user", "content": content[:embedded_idx].strip()})
            _emit_tool_response(
                messages, content[embedded_idx:], pending_call_ids, state
            )
            return

    if content.startswith(_PARSE_ERROR_PREFIX):
        issues["parse_error_feedback_messages"] += 1
    messages.append({"role": "user", "content": content})


def _handle_assistant_message(
    content: str,
    raw_conversations: list[dict[str, Any]],
    msg_idx: int,
    messages: list[dict[str, Any]],
    pending_call_ids: list[str],
    state: _ConvertState,
) -> None:
    parsed = _parse_assistant_json(content)
    if parsed is not None and parsed.get("task_complete") is True:
        state.has_task_complete = True

    # Emit a tool_call only when (1) JSON parsed, (2) commands are valid, and
    # (3) the next raw message contains terminal output, i.e. the framework
    # actually executed the commands. This prevents dangling tool_calls for
    # truncated episodes and framework-rejected commands.
    if (
        parsed is not None
        and _has_valid_commands(parsed)
        and _next_raw_has_terminal(raw_conversations, msg_idx)
    ):
        state.call_id_counter += 1
        tc_id = f"call_{state.call_id_counter}"
        pending_call_ids.clear()
        pending_call_ids.append(tc_id)
        messages.append(
            {
                "role": "assistant",
                "content": _extract_assistant_text(parsed),
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": "execute_commands",
                            "arguments": json.dumps({"commands": parsed["commands"]}),
                        },
                    }
                ],
            }
        )
        return

    if parsed is not None:
        text = _extract_assistant_text(parsed) or _strip_think(content)
    else:
        text = _strip_think(content)
    messages.append({"role": "assistant", "content": text})


def _convert_sample(
    raw_row: dict[str, Any],
    sample_idx: int,
    config: str,
) -> tuple[ConversationSample, dict[str, int], bool]:
    raw_conversations: list[dict[str, Any]] = raw_row["conversations"]
    task_id = raw_row.get("task") or raw_row.get("trial_name") or str(sample_idx)
    run_id = raw_row.get("run_id", "")
    sample_id = f"{task_id}__{run_id[:8]}" if run_id else task_id

    messages: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    state = _ConvertState()
    issues: Counter[str] = Counter()

    for msg_idx, msg in enumerate(raw_conversations):
        role = msg["role"]
        content = msg.get("content") or ""
        match role:
            case "user":
                _handle_user_message(
                    content, msg_idx, messages, pending_call_ids, state, issues
                )
            case "assistant":
                _handle_assistant_message(
                    content,
                    raw_conversations,
                    msg_idx,
                    messages,
                    pending_call_ids,
                    state,
                )
            case _:
                raise ValueError(
                    f"Unknown message role {role!r} in sample {sample_id}, "
                    f"message index {msg_idx}"
                )

    if state.orphan_counter:
        issues["orphan_tool_response_messages"] = state.orphan_counter

    sample = ConversationSample(
        messages=messages,
        tools=[_TOOL_DEFINITION],
        dataset=f"{DATASET_ID}/{config}",
        sample_id=sample_id,
        raw=raw_row,
    )
    return sample, dict(issues), state.has_task_complete


def _apply_strip_malformed(
    sample: ConversationSample,
) -> tuple[ConversationSample, int]:
    # Detect: assistant (no tool_calls) followed by user-role parse-error
    # feedback. Strip both, collapsing the error-recovery cycle.
    messages = sample.messages
    result: list[dict[str, Any]] = []
    stripped = 0
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if (
            i + 1 < n
            and msg["role"] == "assistant"
            and not msg.get("tool_calls")
            and messages[i + 1]["role"] == "user"
            and (messages[i + 1].get("content") or "").startswith(_PARSE_ERROR_PREFIX)
        ):
            stripped += 1
            i += 2
            continue
        result.append(msg)
        i += 1

    if stripped == 0:
        return sample, 0
    return (
        ConversationSample(
            messages=result,
            tools=sample.tools,
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            raw=sample.raw,
        ),
        stripped,
    )


def _apply_drop_orphans(
    sample: ConversationSample,
) -> tuple[ConversationSample, int]:
    # An orphan tool response has tool_call_id starting with "call_orphan_".
    # Also remove the immediately preceding plain-text assistant: the
    # premature "task complete" declaration that triggered the framework's
    # "Are you sure?" confirmation in the raw data.
    messages = sample.messages
    orphan_indices: set[int] = set()
    for i, msg in enumerate(messages):
        if msg["role"] == "tool" and (msg.get("tool_call_id") or "").startswith(
            "call_orphan_"
        ):
            orphan_indices.add(i)
            if (
                i > 0
                and messages[i - 1]["role"] == "assistant"
                and not messages[i - 1].get("tool_calls")
            ):
                orphan_indices.add(i - 1)

    if not orphan_indices:
        return sample, 0

    orphan_tool_count = sum(1 for i in orphan_indices if messages[i]["role"] == "tool")
    result = [msg for i, msg in enumerate(messages) if i not in orphan_indices]
    return (
        ConversationSample(
            messages=result,
            tools=sample.tools,
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            raw=sample.raw,
        ),
        orphan_tool_count,
    )


def _apply_dataset_config(
    samples: list[ConversationSample],
    completions: list[bool],
    config: NemotronTerminalConfig,
) -> tuple[list[ConversationSample], dict[str, int], dict[str, int]]:
    drop_reasons: dict[str, int] = {}
    transform_counts: dict[str, int] = {}

    if config.drop_incomplete:
        kept: list[ConversationSample] = []
        dropped = 0
        for s, completed in zip(samples, completions, strict=True):
            if completed:
                kept.append(s)
            else:
                dropped += 1
        if dropped:
            drop_reasons["incomplete_episodes"] = dropped
        samples = kept

    if config.strip_malformed:
        stripped_samples = 0
        stripped_pairs = 0
        new_samples: list[ConversationSample] = []
        for s in samples:
            transformed, pairs = _apply_strip_malformed(s)
            new_samples.append(transformed)
            if pairs:
                stripped_samples += 1
                stripped_pairs += pairs
        samples = new_samples
        if stripped_samples:
            transform_counts["strip_malformed_samples"] = stripped_samples
            transform_counts["strip_malformed_pairs"] = stripped_pairs

    if config.drop_orphans:
        orphan_samples = 0
        orphan_total = 0
        new_samples = []
        for s in samples:
            transformed, count = _apply_drop_orphans(s)
            new_samples.append(transformed)
            if count:
                orphan_samples += 1
                orphan_total += count
        samples = new_samples
        if orphan_samples:
            transform_counts["drop_orphans_samples"] = orphan_samples
            transform_counts["drop_orphans_messages"] = orphan_total

    return samples, drop_reasons, transform_counts


def _load_parquet(
    path: Path,
    config: str,
    batch_size: int = 500,
) -> tuple[list[ConversationSample], list[bool], dict[str, int]]:
    pf = pq.ParquetFile(path)
    samples: list[ConversationSample] = []
    completions: list[bool] = []
    issue_counts: Counter[str] = Counter()
    global_idx = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        rows = batch.to_pydict()
        for i in range(len(rows["conversations"])):
            row = {k: rows[k][i] for k in rows}
            sample, issues, completed = _convert_sample(row, global_idx, config)
            samples.append(sample)
            completions.append(completed)
            issue_counts.update(issues)
            if issues.get("orphan_tool_response_messages", 0) > 0:
                issue_counts["samples_with_orphan_tool_responses"] += 1
            if issues.get("parse_error_feedback_messages", 0) > 0:
                issue_counts["samples_with_parse_error_feedback"] += 1
            global_idx += 1

    return samples, completions, dict(issue_counts)


def _resolve_file(filename: str, path: str | Path | None) -> Path:
    if path is None:
        return Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                revision=DATASET_REVISION,
                filename=filename,
                repo_type="dataset",
            )
        )
    file_path = Path(path) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    return file_path


def load(
    dataset_config: NemotronTerminalConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    path: str | Path | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    configs_to_load = ALL_CONFIGS
    if dataset_config is not None and dataset_config.configs is not None:
        for c in dataset_config.configs:
            if c not in _FILES:
                raise ValueError(
                    f"Unknown config {c!r}. Valid: {sorted(_FILES.keys())}"
                )
        configs_to_load = dataset_config.configs

    all_samples: list[ConversationSample] = []
    all_completions: list[bool] = []
    all_issue_counts: Counter[str] = Counter()

    with gc_disabled():
        for config_name in configs_to_load:
            for filename in _FILES[config_name]:
                file_path = _resolve_file(filename, path)
                samples, completions, issue_counts = _load_parquet(
                    file_path, config_name
                )
                all_samples.extend(samples)
                all_completions.extend(completions)
                all_issue_counts.update(issue_counts)

    raw_count = len(all_samples)

    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=raw_count,
        stage1_count=raw_count,
        stage1_issue_counts=dict(all_issue_counts),
    )

    if dataset_config is not None:
        all_samples, ds_drops, ds_transforms = _apply_dataset_config(
            all_samples, all_completions, dataset_config
        )
        report.dataset_config_count = len(all_samples)
        report.dataset_config_drop_reasons.update(ds_drops)
        report.dataset_config_transform_counts.update(ds_transforms)

    if filter_config is not None:
        all_samples, filter_drops = apply_filters(all_samples, filter_config)
        report.filtered_count = len(all_samples)
        report.filter_config = filter_config
        report.filter_drop_reasons = filter_drops

    return all_samples, report
