import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
from huggingface_hub import hf_hub_download

from .._gc import gc_disabled
from ..format import ConversationSample
from .base import FilterConfig, LoadReport, apply_filters

DATASET_ID = "nvidia/Nemotron-SFT-Agentic-v2"
DATASET_REVISION = "49e79a3be5ab8cf7511a12958b95cfd6408cd8db"

_HF_REPO = "nvidia/Nemotron-SFT-Agentic-v2"
_FILES: dict[str, str] = {
    "interactive_agent": "data/interactive_agent.jsonl",
    "search": "data/search.jsonl",
    "tool_calling": "data/tool_calling.jsonl",
}
_DEFAULT_SPLITS: tuple[str, ...] = ("interactive_agent", "search")
_ALL_SPLITS: tuple[str, ...] = ("interactive_agent", "search", "tool_calling")


@dataclass(slots=True)
class NemotronAgenticV2Config:
    splits: tuple[str, ...] = _DEFAULT_SPLITS
    interactive_agent_domain_cap: int | None = None


@dataclass(slots=True)
class _ConvertStats:
    samples_by_split: dict[str, int] = field(default_factory=dict)
    parse_errors_by_split: dict[str, int] = field(default_factory=dict)
    uuid_collisions_by_split: dict[str, int] = field(default_factory=dict)
    samples_with_orphan_calls: int = 0
    samples_with_consecutive_same_role: int = 0
    samples_with_no_tools: int = 0
    tool_calls_with_id_field: int = 0
    tool_call_arguments_dict: int = 0
    tool_call_arguments_list: int = 0
    tool_call_arguments_string: int = 0
    tool_call_arguments_other: int = 0
    tool_message_content_string: int = 0
    tool_message_content_other: int = 0
    tool_messages_with_tool_call_id_field: int = 0
    tool_messages_with_name_field: int = 0
    assistant_messages_with_function_call_field: int = 0
    assistant_messages_missing_reasoning_content: int = 0
    assistant_messages_empty_reasoning_content: int = 0
    intermediate_assistant_messages_with_text_and_tool_calls: int = 0
    tool_defs_parameters_null: int = 0
    tool_defs_missing_type: int = 0
    tool_defs_missing_properties: int = 0
    tool_defs_empty_properties: int = 0
    tool_defs_with_strict_true: int = 0
    samples_with_duplicate_tool_names: int = 0

    def as_dict(self) -> dict[str, int]:
        # The output keys diverge from the dataclass field names: the
        # historical `*_stripped` / `*_passthrough` / `*_serialized` suffixes
        # describe what happens to the source field in the converted output,
        # while the field names track source-side prevalence.
        return {
            "samples_with_orphan_calls": self.samples_with_orphan_calls,
            "samples_with_consecutive_same_role": self.samples_with_consecutive_same_role,
            "samples_with_no_tools": self.samples_with_no_tools,
            "tool_calls_with_id_field_stripped": self.tool_calls_with_id_field,
            "tool_call_arguments_string_passthrough": self.tool_call_arguments_string,
            "tool_call_arguments_dict_serialized": self.tool_call_arguments_dict,
            "tool_call_arguments_list_serialized": self.tool_call_arguments_list,
            "tool_call_arguments_other_serialized": self.tool_call_arguments_other,
            "tool_message_content_string_passthrough": self.tool_message_content_string,
            "tool_message_content_other_serialized": self.tool_message_content_other,
            "tool_messages_with_tool_call_id_stripped": self.tool_messages_with_tool_call_id_field,
            "tool_messages_with_name_stripped": self.tool_messages_with_name_field,
            "assistant_messages_with_function_call_stripped": self.assistant_messages_with_function_call_field,
            "assistant_messages_missing_reasoning_content": self.assistant_messages_missing_reasoning_content,
            "assistant_messages_empty_reasoning_content": self.assistant_messages_empty_reasoning_content,
            "intermediate_assistant_messages_with_text_and_tool_calls": self.intermediate_assistant_messages_with_text_and_tool_calls,
            "tool_defs_parameters_null": self.tool_defs_parameters_null,
            "tool_defs_missing_type": self.tool_defs_missing_type,
            "tool_defs_missing_properties": self.tool_defs_missing_properties,
            "tool_defs_empty_properties": self.tool_defs_empty_properties,
            "tool_defs_with_strict_true": self.tool_defs_with_strict_true,
            "samples_with_duplicate_tool_names": self.samples_with_duplicate_tool_names,
            "samples_with_uuid_collision": sum(self.uuid_collisions_by_split.values()),
        }


def _convert_tool_call(tc: dict[str, Any], stats: _ConvertStats) -> dict[str, Any]:
    if "id" in tc:
        stats.tool_calls_with_id_field += 1

    func = tc["function"]
    arguments = func["arguments"]
    if isinstance(arguments, str):
        stats.tool_call_arguments_string += 1
        args_str = arguments
    elif isinstance(arguments, dict):
        stats.tool_call_arguments_dict += 1
        args_str = orjson.dumps(arguments).decode()
    elif isinstance(arguments, list):
        stats.tool_call_arguments_list += 1
        args_str = orjson.dumps(arguments).decode()
    else:
        stats.tool_call_arguments_other += 1
        args_str = orjson.dumps(arguments).decode()

    return {
        "type": "function",
        "function": {"name": func["name"], "arguments": args_str},
    }


def _serialize_tool_content(content: Any, stats: _ConvertStats) -> str:
    if isinstance(content, str):
        stats.tool_message_content_string += 1
        return content
    stats.tool_message_content_other += 1
    return orjson.dumps(content).decode() if content is not None else ""


def _convert_assistant_message(
    msg: dict[str, Any],
    is_last: bool,
    stats: _ConvertStats,
) -> dict[str, Any]:
    if "function_call" in msg:
        stats.assistant_messages_with_function_call_field += 1

    raw_content = msg.get("content")
    out: dict[str, Any] = {"role": "assistant", "content": raw_content or ""}

    tc_list = msg.get("tool_calls")
    if tc_list:
        out["tool_calls"] = [_convert_tool_call(tc, stats) for tc in tc_list]
        if isinstance(raw_content, str) and raw_content and not is_last:
            stats.intermediate_assistant_messages_with_text_and_tool_calls += 1

    if "reasoning_content" in msg:
        rc = msg["reasoning_content"]
        out["reasoning_content"] = rc
        if not rc:
            stats.assistant_messages_empty_reasoning_content += 1
    else:
        stats.assistant_messages_missing_reasoning_content += 1

    return out


def _convert_tool_message(
    msg: dict[str, Any],
    stats: _ConvertStats,
) -> dict[str, Any]:
    if "tool_call_id" in msg:
        stats.tool_messages_with_tool_call_id_field += 1
    if "name" in msg:
        stats.tool_messages_with_name_field += 1
    return {
        "role": "tool",
        "content": _serialize_tool_content(msg.get("content"), stats),
    }


def _convert_messages(
    raw_messages: list[dict[str, Any]],
    stats: _ConvertStats,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_index = len(raw_messages) - 1
    for i, msg in enumerate(raw_messages):
        role = msg["role"]
        match role:
            case "system" | "user":
                out.append({"role": role, "content": msg.get("content") or ""})
            case "assistant":
                out.append(_convert_assistant_message(msg, i == last_index, stats))
            case "tool":
                out.append(_convert_tool_message(msg, stats))
            case _:
                raise ValueError(f"Unknown message role: {role!r}")
    return out


def _normalize_tool(tool: dict[str, Any], stats: _ConvertStats) -> dict[str, Any]:
    func = tool.get("function", {})
    func_out: dict[str, Any] = {"name": func.get("name", "")}
    if "description" in func:
        func_out["description"] = func["description"]

    if "parameters" in func:
        parameters = func["parameters"]
        func_out["parameters"] = parameters
        if parameters is None:
            stats.tool_defs_parameters_null += 1
        elif isinstance(parameters, dict):
            if "type" not in parameters:
                stats.tool_defs_missing_type += 1
            if "properties" not in parameters:
                stats.tool_defs_missing_properties += 1
            else:
                props = parameters["properties"]
                if isinstance(props, dict) and not props:
                    stats.tool_defs_empty_properties += 1

    if "strict" in func:
        func_out["strict"] = func["strict"]
        if func["strict"] is True:
            stats.tool_defs_with_strict_true += 1

    return {"type": "function", "function": func_out}


def _has_duplicate_tool_names(raw_tools: list[Any]) -> bool:
    seen: set[str] = set()
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not isinstance(name, str):
            continue
        if name in seen:
            return True
        seen.add(name)
    return False


def _has_orphan_calls(messages: list[dict[str, Any]]) -> bool:
    n = len(messages)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            continue
        n_calls = len(msg["tool_calls"])
        j = i + 1
        while j < n and messages[j]["role"] == "tool":
            j += 1
        if n_calls > j - i - 1:
            return True
    return False


def _has_consecutive_same_role(messages: list[dict[str, Any]]) -> bool:
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            return True
    return False


def _extract_uuid(raw: dict[str, Any]) -> str | None:
    # metadata.uuid is present on all three splits; search additionally carries
    # a top-level uuid equal to metadata.uuid.
    md = raw.get("metadata")
    if isinstance(md, dict):
        u = md.get("uuid")
        if isinstance(u, str) and u:
            return u
    u = raw.get("uuid")
    if isinstance(u, str) and u:
        return u
    return None


def _convert_sample(
    raw: dict[str, Any],
    split: str,
    line_num: int,
    stats: _ConvertStats,
) -> ConversationSample:
    messages = _convert_messages(raw["messages"], stats)
    raw_tools = raw.get("tools") or []
    tools = [_normalize_tool(t, stats) for t in raw_tools]

    if not tools:
        stats.samples_with_no_tools += 1
    if _has_duplicate_tool_names(raw_tools):
        stats.samples_with_duplicate_tool_names += 1
    if _has_orphan_calls(messages):
        stats.samples_with_orphan_calls += 1
    if _has_consecutive_same_role(messages):
        stats.samples_with_consecutive_same_role += 1

    uuid = _extract_uuid(raw)
    sample_id = f"{split}_{uuid}" if uuid else f"{split}_line{line_num}"
    return ConversationSample(
        messages=messages,
        tools=tools,
        dataset=f"{DATASET_ID}/{split}",
        sample_id=sample_id,
        raw=raw,
    )


def _download_split(split: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=_HF_REPO,
            revision=DATASET_REVISION,
            filename=_FILES[split],
            repo_type="dataset",
        )
    )


def _iter_split(
    split: str,
    stats: _ConvertStats,
) -> Iterable[ConversationSample]:
    path = _download_split(split)
    count = 0
    parse_errors = 0
    with open(path, "rb") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = orjson.loads(line)
            except ValueError:
                parse_errors += 1
                continue
            count += 1
            yield _convert_sample(raw, split, line_num, stats)
    stats.samples_by_split[split] = count
    if parse_errors:
        stats.parse_errors_by_split[split] = parse_errors


def _disambiguate_uuid_collisions(
    samples: list[ConversationSample],
    stats: _ConvertStats,
) -> None:
    # In-place rename of every sample_id that appears more than once, to
    # `{sample_id}_occ{k}`. Preserves the uuid in the id while making the
    # full id unique.
    first_seen: dict[str, int] = {}
    collision_indices: dict[str, list[int]] = {}
    for idx, s in enumerate(samples):
        sid = s.sample_id
        if not isinstance(sid, str):
            continue
        if sid in first_seen:
            collision_indices.setdefault(sid, [first_seen[sid]]).append(idx)
        else:
            first_seen[sid] = idx

    for sid, indices in collision_indices.items():
        for known in _ALL_SPLITS:
            if sid.startswith(f"{known}_"):
                stats.uuid_collisions_by_split[known] = (
                    stats.uuid_collisions_by_split.get(known, 0) + 1
                )
                break
        for occ, idx in enumerate(indices, start=1):
            samples[idx].sample_id = f"{sid}_occ{occ}"


def _apply_domain_cap(
    samples: list[ConversationSample],
    cap: int,
) -> tuple[list[ConversationSample], int]:
    ia_dataset = f"{DATASET_ID}/interactive_agent"
    ia_by_domain: dict[str, list[int]] = {}
    keep: set[int] = set()

    for i, s in enumerate(samples):
        if s.dataset == ia_dataset:
            domain = s.raw.get("domain", "") if isinstance(s.raw, dict) else ""
            ia_by_domain.setdefault(domain, []).append(i)
        else:
            keep.add(i)

    rng = random.Random(42)
    dropped = 0
    for _domain, indices in sorted(ia_by_domain.items()):
        if len(indices) <= cap:
            keep.update(indices)
        else:
            keep.update(rng.sample(indices, cap))
            dropped += len(indices) - cap

    return [s for i, s in enumerate(samples) if i in keep], dropped


def _apply_dataset_config(
    samples: list[ConversationSample],
    config: NemotronAgenticV2Config,
    report: LoadReport,
) -> list[ConversationSample]:
    if config.interactive_agent_domain_cap is not None:
        samples, dropped = _apply_domain_cap(
            samples, config.interactive_agent_domain_cap
        )
        if dropped > 0:
            report.dataset_config_drop_reasons["interactive_agent_domain_cap"] = dropped

    report.dataset_config_count = len(samples)
    return samples


def load(
    dataset_config: NemotronAgenticV2Config | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    if dataset_config is None:
        dataset_config = NemotronAgenticV2Config()

    for s in dataset_config.splits:
        if s not in _FILES:
            raise ValueError(f"Unknown split {s!r}. Valid: {sorted(_FILES)}")

    stats = _ConvertStats()
    all_samples: list[ConversationSample] = []

    with gc_disabled():
        for split in dataset_config.splits:
            all_samples.extend(_iter_split(split, stats))

    _disambiguate_uuid_collisions(all_samples, stats)

    raw_count = sum(stats.samples_by_split.values()) + sum(
        stats.parse_errors_by_split.values()
    )

    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=raw_count,
        stage1_count=len(all_samples),
        stage1_issue_counts=stats.as_dict(),
    )
    if stats.parse_errors_by_split:
        report.stage1_drop_reasons["parse_error"] = sum(
            stats.parse_errors_by_split.values()
        )

    all_samples = _apply_dataset_config(all_samples, dataset_config, report)

    if filter_config is not None:
        all_samples, drops = apply_filters(all_samples, filter_config)
        report.filtered_count = len(all_samples)
        report.filter_config = filter_config
        report.filter_drop_reasons = drops

    return all_samples, report
