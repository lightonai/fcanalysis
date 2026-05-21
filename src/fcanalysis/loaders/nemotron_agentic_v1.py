from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
from huggingface_hub import hf_hub_download

from .._gc import gc_disabled
from ..format import ConversationSample
from .base import FilterConfig, LoadReport, apply_filters


DATASET_ID = "nvidia/Nemotron-Agentic-v1"
DATASET_REVISION = "650d590978ca35c8f1ecea2faf136e5fac421b62"

_FILES: dict[str, str] = {
    "interactive_agent": "data/interactive_agent.jsonl",
    "tool_calling": "data/tool_calling.jsonl",
}
_ALL_SPLITS: tuple[str, ...] = ("interactive_agent", "tool_calling")


@dataclass(slots=True)
class NemotronAgenticV1Config:
    splits: tuple[str, ...] = _ALL_SPLITS
    drop_orphan_samples: bool = False
    drop_empty_system: bool = False
    drop_conflicting_duplicate_tools: bool = False


@dataclass(slots=True)
class _ConvertStats:
    # Field names track source-side prevalence; as_dict() output keys add
    # suffixes describing the converted-output effect
    # (_serialized / _passthrough / _stripped).
    samples_by_split: dict[str, int] = field(default_factory=dict)
    samples_with_orphan_calls: int = 0
    samples_with_consecutive_same_role: int = 0
    samples_with_empty_system: int = 0
    samples_with_no_tools: int = 0
    tool_calls_with_index_field: int = 0
    tool_calls_with_id_field: int = 0
    tool_call_arguments_dict: int = 0
    tool_call_arguments_list: int = 0
    tool_call_arguments_string: int = 0
    tool_message_content_dict: int = 0
    tool_message_content_list: int = 0
    tool_message_content_bool: int = 0
    tool_message_content_string: int = 0
    tool_message_content_other: int = 0
    tool_messages_with_tool_call_id_field: int = 0
    assistant_messages_missing_reasoning_content: int = 0
    tool_defs_function_level_required_lifted: int = 0
    tool_defs_function_level_required_empty_dropped: int = 0
    tool_defs_leaf_type_root_string: int = 0
    tool_defs_leaf_type_root_integer: int = 0
    tool_defs_with_missing_properties: int = 0
    tool_defs_with_empty_properties: int = 0
    samples_with_duplicate_tool_names: int = 0
    samples_with_conflicting_duplicate_tool_names: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "samples_with_orphan_calls": self.samples_with_orphan_calls,
            "samples_with_consecutive_same_role": self.samples_with_consecutive_same_role,
            "samples_with_empty_system": self.samples_with_empty_system,
            "samples_with_no_tools": self.samples_with_no_tools,
            "tool_calls_with_index_field_stripped": self.tool_calls_with_index_field,
            "tool_calls_with_id_field_stripped": self.tool_calls_with_id_field,
            "tool_call_arguments_dict_serialized": self.tool_call_arguments_dict,
            "tool_call_arguments_list_serialized": self.tool_call_arguments_list,
            "tool_call_arguments_string_passthrough": self.tool_call_arguments_string,
            "tool_message_content_dict_serialized": self.tool_message_content_dict,
            "tool_message_content_list_serialized": self.tool_message_content_list,
            "tool_message_content_bool_serialized": self.tool_message_content_bool,
            "tool_message_content_string_passthrough": self.tool_message_content_string,
            "tool_message_content_other_serialized": self.tool_message_content_other,
            "tool_messages_with_tool_call_id_stripped": self.tool_messages_with_tool_call_id_field,
            "assistant_messages_missing_reasoning_content": self.assistant_messages_missing_reasoning_content,
            "tool_defs_function_level_required_lifted": self.tool_defs_function_level_required_lifted,
            "tool_defs_function_level_required_empty_dropped": self.tool_defs_function_level_required_empty_dropped,
            "tool_defs_leaf_type_root_string": self.tool_defs_leaf_type_root_string,
            "tool_defs_leaf_type_root_integer": self.tool_defs_leaf_type_root_integer,
            "tool_defs_with_missing_properties": self.tool_defs_with_missing_properties,
            "tool_defs_with_empty_properties": self.tool_defs_with_empty_properties,
            "samples_with_duplicate_tool_names": self.samples_with_duplicate_tool_names,
            "samples_with_conflicting_duplicate_tool_names": self.samples_with_conflicting_duplicate_tool_names,
        }


def _convert_tool_call(tc: dict[str, Any], stats: _ConvertStats) -> dict[str, Any]:
    if "id" in tc:
        stats.tool_calls_with_id_field += 1
    if "index" in tc:
        stats.tool_calls_with_index_field += 1

    func = tc["function"]
    arguments = func["arguments"]
    if isinstance(arguments, str):
        stats.tool_call_arguments_string += 1
        args_str = arguments
    elif isinstance(arguments, list):
        stats.tool_call_arguments_list += 1
        args_str = orjson.dumps(arguments).decode()
    elif isinstance(arguments, dict):
        stats.tool_call_arguments_dict += 1
        args_str = orjson.dumps(arguments).decode()
    else:
        args_str = orjson.dumps(arguments).decode()

    return {
        "type": "function",
        "function": {"name": func["name"], "arguments": args_str},
    }


def _serialize_tool_content(content: Any, stats: _ConvertStats) -> str:
    # str/bool/dict/list checks in order; bool subclasses int but neither is
    # checked as int, so order is only for clarity. None falls into "other"
    # but is special-cased to "" (orjson.dumps(None) would produce "null").
    if isinstance(content, str):
        stats.tool_message_content_string += 1
        return content
    if isinstance(content, bool):
        stats.tool_message_content_bool += 1
        return orjson.dumps(content).decode()
    if isinstance(content, dict):
        stats.tool_message_content_dict += 1
        return orjson.dumps(content).decode()
    if isinstance(content, list):
        stats.tool_message_content_list += 1
        return orjson.dumps(content).decode()
    stats.tool_message_content_other += 1
    return orjson.dumps(content).decode() if content is not None else ""


def _convert_messages(
    raw_messages: list[dict[str, Any]],
    stats: _ConvertStats,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in raw_messages:
        role = msg["role"]
        match role:
            case "system" | "user":
                out.append({"role": role, "content": msg.get("content") or ""})
            case "assistant":
                converted: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                }
                tc_list = msg.get("tool_calls")
                if tc_list:
                    converted["tool_calls"] = [
                        _convert_tool_call(tc, stats) for tc in tc_list
                    ]
                if "reasoning_content" in msg:
                    converted["reasoning_content"] = msg["reasoning_content"]
                else:
                    stats.assistant_messages_missing_reasoning_content += 1
                out.append(converted)
            case "tool":
                if "tool_call_id" in msg:
                    stats.tool_messages_with_tool_call_id_field += 1
                out.append(
                    {
                        "role": "tool",
                        "content": _serialize_tool_content(msg.get("content"), stats),
                    }
                )
            case _:
                raise ValueError(f"Unknown message role: {role!r}")
    return out


def _lift_function_required(parameters: Any, names: list[Any]) -> Any:
    # Passthrough when parameters is neither None nor a dict: the function-
    # level required list has nowhere to lift INTO. The caller increments the
    # _lifted counter before this call regardless, matching the original
    # loader's accounting (counts attempts, not just successful lifts).
    if parameters is None:
        return {"required": list(names)}
    if not isinstance(parameters, dict):
        return parameters
    existing = parameters.get("required")
    if isinstance(existing, list):
        merged = list(existing)
        for name in names:
            if name not in merged:
                merged.append(name)
    else:
        merged = list(names)
    return {**parameters, "required": merged}


def _normalize_tool(tool: dict[str, Any], stats: _ConvertStats) -> dict[str, Any]:
    func = tool.get("function", {})
    func_out: dict[str, Any] = {"name": func.get("name", "")}
    if "description" in func:
        func_out["description"] = func["description"]

    parameters: Any = func.get("parameters") if "parameters" in func else None

    if "required" in func:
        func_level_required = func["required"]
        if isinstance(func_level_required, list):
            if func_level_required:
                stats.tool_defs_function_level_required_lifted += 1
                parameters = _lift_function_required(parameters, func_level_required)
            else:
                stats.tool_defs_function_level_required_empty_dropped += 1

    if parameters is not None:
        func_out["parameters"] = parameters

    if isinstance(parameters, dict):
        match parameters.get("type"):
            case "string":
                stats.tool_defs_leaf_type_root_string += 1
            case "integer":
                stats.tool_defs_leaf_type_root_integer += 1
        if "properties" not in parameters:
            stats.tool_defs_with_missing_properties += 1
        else:
            props = parameters["properties"]
            if isinstance(props, dict) and not props:
                stats.tool_defs_with_empty_properties += 1

    if "strict" in func:
        func_out["strict"] = func["strict"]
    return {"type": "function", "function": func_out}


def _has_conflicting_duplicate_tools(raw_tools: list[Any]) -> bool:
    # Canonicalize each function dict with sorted keys so pure key-order
    # differences are not treated as conflicts.
    name_to_reprs: dict[str, set[bytes]] = {}
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not isinstance(name, str):
            continue
        canonical = orjson.dumps(func, option=orjson.OPT_SORT_KEYS)
        name_to_reprs.setdefault(name, set()).add(canonical)
    return any(len(reprs) > 1 for reprs in name_to_reprs.values())


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
        n_responses = 0
        j = i + 1
        while j < n and messages[j]["role"] == "tool":
            n_responses += 1
            j += 1
        if n_calls > n_responses:
            return True
    return False


def _has_consecutive_same_role(messages: list[dict[str, Any]]) -> bool:
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            return True
    return False


def _convert_sample(
    raw: dict[str, Any],
    split: str,
    stats: _ConvertStats,
) -> ConversationSample:
    messages = _convert_messages(raw["messages"], stats)
    raw_tools = raw.get("tools") or []
    tools = [_normalize_tool(t, stats) for t in raw_tools]

    if not tools:
        stats.samples_with_no_tools += 1
    if _has_duplicate_tool_names(raw_tools):
        stats.samples_with_duplicate_tool_names += 1
        if _has_conflicting_duplicate_tools(raw_tools):
            stats.samples_with_conflicting_duplicate_tool_names += 1
    if messages and messages[0]["role"] == "system" and messages[0]["content"] == "":
        stats.samples_with_empty_system += 1
    if _has_orphan_calls(messages):
        stats.samples_with_orphan_calls += 1
    if _has_consecutive_same_role(messages):
        stats.samples_with_consecutive_same_role += 1

    uuid = raw.get("uuid", "")
    return ConversationSample(
        messages=messages,
        tools=tools,
        dataset=f"{DATASET_ID}/{split}",
        sample_id=f"{split}_{uuid}" if uuid else f"{split}_{id(raw)}",
        raw=raw,
    )


def _iter_split(split: str, stats: _ConvertStats) -> Iterator[ConversationSample]:
    path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            revision=DATASET_REVISION,
            filename=_FILES[split],
            repo_type="dataset",
        )
    )
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            raw = orjson.loads(line)
            count += 1
            yield _convert_sample(raw, split, stats)
    stats.samples_by_split[split] = count


def _apply_dataset_config(
    samples: list[ConversationSample],
    config: NemotronAgenticV1Config,
    report: LoadReport,
) -> list[ConversationSample]:
    kept: list[ConversationSample] = []
    dropped_orphan = 0
    dropped_conflicting_tools = 0
    removed_empty_system = 0

    for s in samples:
        drop = False
        if config.drop_orphan_samples and _has_orphan_calls(s.messages):
            dropped_orphan += 1
            drop = True
        if config.drop_conflicting_duplicate_tools and _has_conflicting_duplicate_tools(
            s.raw.get("tools") or []
        ):
            dropped_conflicting_tools += 1
            drop = True
        if drop:
            continue

        if config.drop_empty_system:
            msgs = s.messages
            if msgs and msgs[0]["role"] == "system" and msgs[0].get("content") == "":
                s.messages = msgs[1:]
                removed_empty_system += 1

        kept.append(s)

    if dropped_orphan:
        report.dataset_config_drop_reasons["orphan_samples"] = dropped_orphan
    if dropped_conflicting_tools:
        report.dataset_config_drop_reasons["conflicting_duplicate_tools"] = (
            dropped_conflicting_tools
        )
    if removed_empty_system:
        report.dataset_config_transform_counts["removed_empty_system"] = (
            removed_empty_system
        )
    report.dataset_config_count = len(kept)
    return kept


def load(
    dataset_config: NemotronAgenticV1Config | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    if dataset_config is None:
        dataset_config = NemotronAgenticV1Config()

    for s in dataset_config.splits:
        if s not in _FILES:
            raise ValueError(f"Unknown split {s!r}. Valid: {sorted(_FILES)}")

    stats = _ConvertStats()
    all_samples: list[ConversationSample] = []

    with gc_disabled():
        for split in dataset_config.splits:
            all_samples.extend(_iter_split(split, stats))

    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=sum(stats.samples_by_split.values()),
        stage1_count=len(all_samples),
        stage1_issue_counts=stats.as_dict(),
    )

    all_samples = _apply_dataset_config(all_samples, dataset_config, report)

    if filter_config is not None:
        all_samples, drops = apply_filters(all_samples, filter_config)
        report.filtered_count = len(all_samples)
        report.filter_config = filter_config
        report.filter_drop_reasons = drops

    return all_samples, report
