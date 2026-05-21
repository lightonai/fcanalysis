from collections import Counter
from dataclasses import dataclass
from typing import Any

import orjson
from datasets import load_dataset

from ..format import ConversationSample
from ..validation import has_undefined_function_calls
from .base import FilterConfig, LoadReport, apply_filters


DATASET_ID = "Salesforce/APIGen-MT-5k"
DATASET_REVISION = "abc4a517d67c541f85f6470cbd8fd3186b36830e"


@dataclass(slots=True)
class APIGenMTConfig:
    strip_think_tool: bool = False
    drop_undefined_function_calls: bool = False
    drop_repeated_tool_call_streaks: bool = False
    # Deprecated alias for drop_repeated_tool_call_streaks; both contribute to the
    # same drop count under reason key "repeated_tool_call_streaks".
    drop_error_recovery_loops: bool = False
    drop_consecutive_assistant: bool = False


def _parse_tools(tools_json: str) -> list[dict[str, Any]]:
    raw_tools: list[dict[str, Any]] = orjson.loads(tools_json)
    return [{"type": "function", "function": t} for t in raw_tools]


def _convert_conversations(
    conversations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for msg in conversations:
        role = msg["from"]
        value = msg["value"]
        match role:
            case "human":
                messages.append({"role": "user", "content": value})
            case "gpt":
                messages.append({"role": "assistant", "content": value})
            case "function_call":
                fc: dict[str, Any] = orjson.loads(value)
                name = fc["name"]
                arguments = fc.get("arguments", {})
                arguments_str = (
                    arguments
                    if isinstance(arguments, str)
                    else orjson.dumps(arguments).decode()
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": name, "arguments": arguments_str},
                            }
                        ],
                    }
                )
            case "observation":
                messages.append({"role": "tool", "content": value})
            case _:
                raise ValueError(
                    f"Unknown conversation role in APIGen-MT: "
                    f"role={role!r}, value={value[:50]!r}"
                )
    return messages


def _tool_call_signature(msg: dict[str, Any]) -> bytes:
    return orjson.dumps(
        [
            (tc["function"]["name"], tc["function"]["arguments"])
            for tc in msg["tool_calls"]
        ],
        option=orjson.OPT_SORT_KEYS,
    )


def _max_repeated_streak(sample: ConversationSample) -> int:
    # A streak is a run of assistant tool-call messages with identical signatures.
    # Tool observations between them don't break the streak; any user message or
    # text-only assistant message resets it.
    longest = 0
    prev_sig: bytes | None = None
    current = 0
    for msg in sample.messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            sig = _tool_call_signature(msg)
            if sig == prev_sig:
                current += 1
            else:
                prev_sig = sig
                current = 1
            if current > longest:
                longest = current
        elif msg["role"] != "tool":
            prev_sig = None
            current = 0
    return longest


def _has_repeated_tool_call_streak(
    sample: ConversationSample,
    min_length: int = 3,
) -> bool:
    return _max_repeated_streak(sample) >= min_length


def _has_consecutive_assistant(sample: ConversationSample) -> bool:
    prev_role: str | None = None
    prev_has_tool_calls = False
    for msg in sample.messages:
        role = msg["role"]
        has_tool_calls = bool(msg.get("tool_calls"))
        if (
            role == "assistant"
            and prev_role == "assistant"
            and not has_tool_calls
            and not prev_has_tool_calls
        ):
            return True
        prev_role = role
        prev_has_tool_calls = has_tool_calls
    return False


def _strip_think_from_sample_with_counts(
    sample: ConversationSample,
) -> tuple[ConversationSample, int, int, int]:
    filtered_tools = [
        t for t in sample.tools if t.get("function", {}).get("name") != "think"
    ]
    removed_tool_definitions = len(sample.tools) - len(filtered_tools)

    filtered_messages: list[dict[str, Any]] = []
    skip_next_tool = False
    removed_tool_calls = 0
    removed_tool_observations = 0
    for msg in sample.messages:
        if skip_next_tool and msg["role"] == "tool":
            skip_next_tool = False
            removed_tool_observations += 1
            continue
        skip_next_tool = False

        if msg["role"] == "assistant" and msg.get("tool_calls"):
            calls = msg["tool_calls"]
            if len(calls) == 1 and calls[0]["function"]["name"] == "think":
                skip_next_tool = True
                removed_tool_calls += 1
                continue

        filtered_messages.append(msg)

    return (
        ConversationSample(
            messages=filtered_messages,
            tools=filtered_tools,
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            raw=sample.raw,
        ),
        removed_tool_definitions,
        removed_tool_calls,
        removed_tool_observations,
    )


def _apply_dataset_config(
    samples: list[ConversationSample],
    config: APIGenMTConfig,
) -> tuple[list[ConversationSample], dict[str, int], dict[str, int]]:
    transform_counts: Counter[str] = Counter()

    if config.strip_think_tool:
        stripped: list[ConversationSample] = []
        samples_with_think = 0
        for s in samples:
            new_sample, defs, calls, obs = _strip_think_from_sample_with_counts(s)
            stripped.append(new_sample)
            if defs or calls or obs:
                samples_with_think += 1
                transform_counts["think_tool_definitions_removed"] += defs
                transform_counts["think_tool_calls_removed"] += calls
                transform_counts["think_tool_observations_removed"] += obs
        samples = stripped
        if samples_with_think:
            transform_counts["samples_with_think_tool"] = samples_with_think

    drop_repeated = (
        config.drop_repeated_tool_call_streaks or config.drop_error_recovery_loops
    )

    drop_reasons: Counter[str] = Counter()
    kept: list[ConversationSample] = []
    for s in samples:
        drop = False
        if config.drop_undefined_function_calls and has_undefined_function_calls(s):
            drop_reasons["undefined_function_calls"] += 1
            drop = True
        if drop_repeated and _has_repeated_tool_call_streak(s):
            drop_reasons["repeated_tool_call_streaks"] += 1
            drop = True
        if config.drop_consecutive_assistant and _has_consecutive_assistant(s):
            drop_reasons["consecutive_assistant"] += 1
            drop = True
        if not drop:
            kept.append(s)

    return kept, dict(drop_reasons), dict(transform_counts)


def load(
    dataset_config: APIGenMTConfig | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    raw_count = len(ds)

    # Bulk Arrow read; the per-sample conversion is dict shuffling + orjson, light
    # enough that datasets.map() with multiprocessing would lose more to Arrow
    # round-trip than it gains from parallelism.
    table = ds.data.table
    col_system = table.column("system").to_pylist()
    col_tools = table.column("tools").to_pylist()
    col_conversations = table.column("conversations").to_pylist()

    samples: list[ConversationSample] = []
    stage1_issues: Counter[str] = Counter()
    for i in range(raw_count):
        tools = _parse_tools(col_tools[i])
        messages = _convert_conversations(col_conversations[i])
        messages.insert(0, {"role": "system", "content": col_system[i]})

        sample = ConversationSample(
            messages=messages,
            tools=tools,
            dataset=DATASET_ID,
            sample_id=i,
            raw={
                "system": col_system[i],
                "tools": col_tools[i],
                "conversations": col_conversations[i],
            },
        )
        if _has_consecutive_assistant(sample):
            stage1_issues["samples_with_consecutive_assistant_text"] += 1
        max_streak = _max_repeated_streak(sample)
        if max_streak >= 2:
            stage1_issues["samples_with_repeated_tool_call_streak_ge_2"] += 1
        if max_streak >= 3:
            stage1_issues["samples_with_repeated_tool_call_streak_ge_3"] += 1

        samples.append(sample)

    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=raw_count,
        stage1_count=len(samples),
        stage1_issue_counts=dict(stage1_issues),
    )

    if dataset_config is not None:
        samples, ds_drops, ds_transforms = _apply_dataset_config(
            samples, dataset_config
        )
        report.dataset_config_count = len(samples)
        report.dataset_config_drop_reasons.update(ds_drops)
        report.dataset_config_transform_counts.update(ds_transforms)

    if filter_config is not None:
        samples, filter_drops = apply_filters(samples, filter_config)
        report.filtered_count = len(samples)
        report.filter_config = filter_config
        report.filter_drop_reasons = filter_drops

    return samples, report
