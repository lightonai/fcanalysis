from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import orjson
from huggingface_hub import hf_hub_download

from .._gc import gc_disabled
from ..format import ConversationSample
from .base import FilterConfig, LoadReport, apply_filters


DATASET_ID = "Nanbeige/ToolMind"
DATASET_REVISION = "8020ed1c03c367e4eb720ac3828ab4b0b95d8baf"

SOURCES: tuple[str, ...] = (
    "graph_syn_datasets/graphsyn.jsonl",
    "open_datasets/APIGen-MT-5k-query.jsonl",
    "open_datasets/BUTTONInstruct-query.jsonl",
    "open_datasets/ToolACE-query.jsonl",
    "open_datasets/When2Call-query.jsonl",
    "open_datasets/glaive-function-calling-v2-query.jsonl",
    "open_datasets/tau-train-query.jsonl",
    "open_datasets/xlam-function-calling-60k-query.jsonl",
)

_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "set": "array",
    "tuple": "array",
    "none": "null",
}

_JSON_SCHEMA_TYPES: frozenset[str] = frozenset(
    {"string", "integer", "number", "boolean", "array", "object", "null"}
)


@dataclass(slots=True)
class ToolMindConfig:
    sources: list[str] | None = None
    seed_group_filter: Literal["longest", "longest_clean", "none"] = "none"
    drop_non_object_arguments: bool = False
    drop_consecutive_text_assistant: bool = False
    merge_split_assistant: bool = False
    strip_think_tool: bool = False


def _split_on_top_level_comma(s: str) -> list[str]:
    # Splits on commas outside of any ()/[] pair so nested generics like
    # Union[List[int], str] or Callable[[float], float] stay intact.
    parts: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


def _convert_type_string(type_str: str) -> dict[str, Any]:
    clean = _split_on_top_level_comma(type_str)[0]

    if clean.startswith(("List[", "list[")) and clean.endswith("]"):
        inner = clean[5:-1]
        return {
            "type": "array",
            "items": _convert_type_string(_split_on_top_level_comma(inner)[0]),
        }

    if clean.startswith(("Tuple[", "tuple[")) and clean.endswith("]"):
        inner = clean[clean.index("[") + 1 : -1]
        prefix_items = [
            _convert_type_string(p) for p in _split_on_top_level_comma(inner)
        ]
        return {"type": "array", "prefixItems": prefix_items, "items": False}

    if clean.startswith("Union[") and clean.endswith("]"):
        inner = clean[6:-1]
        return {
            "anyOf": [_convert_type_string(p) for p in _split_on_top_level_comma(inner)]
        }

    if clean.startswith("Dict") or clean.startswith("dict["):
        return {"type": "object"}

    if clean.startswith("Callable"):
        return {"type": "string"}

    if clean in ("List", "list"):
        return {"type": "array"}

    return {"type": _TYPE_MAP.get(clean.lower(), "string")}


def _normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    func = tool.get("function", tool)

    # ToolMind has a small set of tools double-wrapped as
    # {type, function: {type, function: {name, ...}}}; unwrap one level.
    if isinstance(func.get("function"), dict) and "name" in func["function"]:
        func = func["function"]

    params_raw = func.get("parameters") or func.get("arguments") or {}
    func_required = func.get("required")
    explicit_required = func_required if isinstance(func_required, list) else None

    params: Any = (
        _normalize_parameters(params_raw, explicit_required)
        if isinstance(params_raw, dict)
        else params_raw
    )

    return {
        "type": "function",
        "function": {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": params,
        },
    }


def _is_flat_property_map(params: dict[str, Any]) -> bool:
    # A flat map looks like {"city": {"type": "str", ...}, ...} with no
    # "properties" wrapper. Distinguishing edge case: a JSON Schema object's
    # "type" key holds a string ("object"), whereas a flat map containing a
    # parameter literally named "type" holds a dict value.
    if "properties" in params or not params:
        return False
    if "type" in params and isinstance(params["type"], str):
        return False
    has_dict_values = False
    for v in params.values():
        if isinstance(v, dict):
            has_dict_values = True
            if "type" not in v and "description" not in v:
                return False
    return has_dict_values


def _filter_required_names(
    required: list[str],
    properties: dict[str, Any],
) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for name in required:
        if name in properties and name not in seen:
            filtered.append(name)
            seen.add(name)
    return filtered


def _normalize_parameters(
    params: dict[str, Any],
    func_required: list[str] | None = None,
) -> dict[str, Any]:
    if _is_flat_property_map(params):
        return _build_schema_from_flat(params, func_required)

    result: dict[str, Any] = dict(params)
    if result.get("type") == "dict":
        result["type"] = "object"
    if "properties" in result and "type" not in result:
        result["type"] = "object"

    raw_required = params.get("required")
    if isinstance(raw_required, list):
        explicit_required: list[str] | None = raw_required
    elif "required" in params:
        explicit_required = []
    elif func_required is not None:
        explicit_required = [n for n in func_required if isinstance(n, str)]
    else:
        explicit_required = None

    if "properties" in result:
        properties = result["properties"]
        infer_required = explicit_required is None
        required: list[str] = (
            _filter_required_names(explicit_required, properties)
            if explicit_required is not None
            else []
        )
        new_properties: dict[str, Any] = {}
        for pname, pdef in properties.items():
            if isinstance(pdef, dict):
                new_properties[pname] = _normalize_property(
                    pdef, pname, required, infer_required=infer_required
                )
            else:
                new_properties[pname] = pdef
        result["properties"] = new_properties
        result["required"] = required
    elif explicit_required is not None:
        result["required"] = explicit_required

    return result


def _build_schema_from_flat(
    flat: dict[str, Any],
    explicit_required: list[str] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    infer_required = explicit_required is None

    for pname, pdef in flat.items():
        if not isinstance(pdef, dict):
            continue
        properties[pname] = _normalize_property(
            pdef, pname, required, infer_required=infer_required
        )

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if explicit_required is not None:
        result["required"] = _filter_required_names(explicit_required, properties)
    elif required:
        result["required"] = required
    return result


def _normalize_property(
    pdef: dict[str, Any],
    pname: str,
    required: list[str],
    *,
    infer_required: bool,
) -> dict[str, Any]:
    type_str = pdef.get("type", "")
    if not isinstance(type_str, str):
        # Shallow copy so the caller can't accidentally alias the raw pdef
        # back through tool output (the deeper structures stay shared, but
        # current downstream consumers do not mutate tool definitions).
        return dict(pdef)

    is_optional = "optional" in type_str.lower() or "default" in pdef

    if type_str not in _JSON_SCHEMA_TYPES:
        schema = _convert_type_string(type_str)
        if "description" in pdef:
            schema["description"] = pdef["description"]
        if "default" in pdef:
            schema["default"] = pdef["default"]
    else:
        schema = dict(pdef)

    if is_optional:
        if "anyOf" in schema:
            existing = schema["anyOf"]
            has_null = any(
                isinstance(sub, dict) and sub.get("type") == "null" for sub in existing
            )
            if not has_null:
                # New list rather than append; `schema` is a shallow copy of
                # `pdef`, so `existing` is the same list reference and an
                # in-place append would mutate the raw pdef.
                schema["anyOf"] = [*existing, {"type": "null"}]
        else:
            type_schema: dict[str, Any] = {}
            for k in ("type", "items", "prefixItems"):
                if k in schema:
                    type_schema[k] = schema.pop(k)
            if type_schema:
                schema["anyOf"] = [type_schema, {"type": "null"}]

    if infer_required and not is_optional and pname not in required:
        required.append(pname)

    return schema


def _serialize_arguments(arguments: Any) -> str:
    # dict  -> JSON object string
    # str   -> passed through (already serialized; may or may not be valid JSON)
    # None  -> "{}"
    # list/int/other -> orjson.dumps (valid JSON but non-object; malformed data)
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return orjson.dumps(arguments).decode()


def _convert_messages(
    conversations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for msg in conversations:
        role: str = msg["role"]
        content = msg.get("content")
        match role:
            case "assistant" if msg.get("tool_calls"):
                tool_calls: list[dict[str, Any]] = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", tc)
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": _serialize_arguments(
                                    func.get("arguments")
                                ),
                            },
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": content
                        if content and str(content).strip()
                        else None,
                        "tool_calls": tool_calls,
                    }
                )
            case "tool":
                messages.append({"role": "tool", "content": content or ""})
            case _:
                messages.append({"role": role, "content": content or ""})
    return messages


def _has_non_object_arguments(sample: ConversationSample) -> bool:
    for msg in sample.messages:
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            try:
                parsed = orjson.loads(tc["function"]["arguments"])
            except ValueError:
                continue
            if not isinstance(parsed, dict):
                return True
    return False


def _has_consecutive_text_assistant(sample: ConversationSample) -> bool:
    prev_role: str | None = None
    prev_has_tc = False
    for msg in sample.messages:
        role = msg["role"]
        has_tc = bool(msg.get("tool_calls"))
        if (
            role == "assistant"
            and prev_role == "assistant"
            and not has_tc
            and not prev_has_tc
        ):
            return True
        prev_role = role
        prev_has_tc = has_tc
    return False


def _count_consecutive_text_assistant_pairs(sample: ConversationSample) -> int:
    pairs = 0
    prev_role: str | None = None
    prev_has_tc = False
    for msg in sample.messages:
        role = msg["role"]
        has_tc = bool(msg.get("tool_calls"))
        if (
            role == "assistant"
            and prev_role == "assistant"
            and not has_tc
            and not prev_has_tc
        ):
            pairs += 1
        prev_role = role
        prev_has_tc = has_tc
    return pairs


def _count_split_assistant_pairs(sample: ConversationSample) -> int:
    pairs = 0
    messages = sample.messages
    for i in range(len(messages) - 1):
        first = messages[i]
        second = messages[i + 1]
        if (
            first["role"] == "assistant"
            and second["role"] == "assistant"
            and not first.get("tool_calls")
            and bool(second.get("tool_calls"))
        ):
            pairs += 1
    return pairs


def _count_think_tool_stats(sample: ConversationSample) -> tuple[int, int, int]:
    think_defs = sum(
        1 for tool in sample.tools if tool.get("function", {}).get("name") == "think"
    )

    think_calls = 0
    think_calls_without_observation = 0
    messages = sample.messages
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            i += 1
            continue

        tool_calls = msg["tool_calls"]
        think_indices = [
            idx
            for idx, tc in enumerate(tool_calls)
            if tc.get("function", {}).get("name") == "think"
        ]
        if not think_indices:
            i += 1
            continue

        think_calls += len(think_indices)
        j = i + 1
        tool_response_count = 0
        while j < n and messages[j]["role"] == "tool":
            tool_response_count += 1
            j += 1
        think_calls_without_observation += sum(
            1 for idx in think_indices if idx >= tool_response_count
        )
        i = j

    return think_defs, think_calls, think_calls_without_observation


def _strip_think_tool_from_sample_with_counts(
    sample: ConversationSample,
) -> tuple[ConversationSample, int, int, int]:
    filtered_tools = [
        t for t in sample.tools if t.get("function", {}).get("name") != "think"
    ]
    removed_tool_definitions = len(sample.tools) - len(filtered_tools)

    messages = sample.messages
    n = len(messages)
    rebuilt: list[dict[str, Any]] = []
    changed = removed_tool_definitions > 0
    removed_tool_calls = 0
    removed_tool_observations = 0

    i = 0
    while i < n:
        msg = messages[i]
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            rebuilt.append(msg)
            i += 1
            continue

        tcs = msg["tool_calls"]
        j = i + 1
        while j < n and messages[j]["role"] == "tool":
            j += 1
        tool_responses = messages[i + 1 : j]

        think_indices: set[int] = {
            idx
            for idx, tc in enumerate(tcs)
            if tc.get("function", {}).get("name") == "think"
        }

        if not think_indices:
            rebuilt.append(msg)
            rebuilt.extend(tool_responses)
            i = j
            continue

        changed = True
        removed_tool_calls += len(think_indices)
        kept_tcs = [tc for idx, tc in enumerate(tcs) if idx not in think_indices]
        kept_responses: list[dict[str, Any]] = []
        for idx, response in enumerate(tool_responses):
            # Tool responses past len(tcs) have no positional call to pair with;
            # leave them in place rather than discarding mismatched data.
            if idx < len(tcs) and idx in think_indices:
                removed_tool_observations += 1
                continue
            kept_responses.append(response)

        if kept_tcs:
            rebuilt.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": kept_tcs,
                }
            )
            rebuilt.extend(kept_responses)
        else:
            content = msg.get("content")
            if content and str(content).strip():
                rebuilt.append({"role": "assistant", "content": content})
        i = j

    if not changed:
        return sample, 0, 0, 0
    return (
        ConversationSample(
            messages=rebuilt,
            tools=filtered_tools,
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            raw=sample.raw,
        ),
        removed_tool_definitions,
        removed_tool_calls,
        removed_tool_observations,
    )


def _merge_split_assistant_messages_with_counts(
    sample: ConversationSample,
) -> tuple[ConversationSample, int]:
    messages = sample.messages
    n = len(messages)
    merged: list[dict[str, Any]] = []
    merged_pairs = 0

    i = 0
    while i < n:
        msg = messages[i]
        if (
            i + 1 < n
            and msg["role"] == "assistant"
            and not msg.get("tool_calls")
            and messages[i + 1]["role"] == "assistant"
            and messages[i + 1].get("tool_calls")
        ):
            first_content = msg.get("content") or ""
            second_content = messages[i + 1].get("content") or ""
            if first_content and second_content:
                combined = first_content.rstrip() + "\n\n" + second_content
            else:
                combined = first_content or second_content
            merged.append(
                {
                    "role": "assistant",
                    "content": combined if combined.strip() else None,
                    "tool_calls": messages[i + 1]["tool_calls"],
                }
            )
            merged_pairs += 1
            i += 2
        else:
            merged.append(msg)
            i += 1

    if not merged_pairs:
        return sample, 0
    return (
        ConversationSample(
            messages=merged,
            tools=sample.tools,
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            raw=sample.raw,
        ),
        merged_pairs,
    )


def _compute_seed_key(sample: ConversationSample) -> str:
    for msg in sample.messages:
        if msg["role"] == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def _has_balanced_cardinality(sample: ConversationSample) -> bool:
    msgs = sample.messages
    for i, msg in enumerate(msgs):
        if msg["role"] != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not tcs:
            continue
        n_calls = len(tcs)
        n_responses = 0
        for j in range(i + 1, len(msgs)):
            if msgs[j]["role"] == "tool":
                n_responses += 1
            else:
                break
        if n_calls != n_responses:
            return False
    return True


def _select_from_seed_groups(
    samples: list[ConversationSample],
    mode: Literal["longest", "longest_clean"],
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(_compute_seed_key(s), []).append(i)

    kept_indices: list[int] = []
    intermediates_discarded = 0
    clean_selections = 0
    fallback_selections = 0

    for indices in groups.values():
        if mode == "longest_clean":
            # Stable sort by length desc; ties retain ascending sample_id order.
            sorted_desc = sorted(
                indices, key=lambda i: len(samples[i].messages), reverse=True
            )
            chosen = next(
                (idx for idx in sorted_desc if _has_balanced_cardinality(samples[idx])),
                None,
            )
            if chosen is not None:
                kept_indices.append(chosen)
                clean_selections += 1
            else:
                kept_indices.append(sorted_desc[0])
                fallback_selections += 1
        else:
            kept_indices.append(max(indices, key=lambda i: len(samples[i].messages)))
        intermediates_discarded += len(indices) - 1

    kept_indices.sort()

    drop_reasons: dict[str, int] = {}
    if intermediates_discarded:
        drop_reasons["seed_group_intermediate_discarded"] = intermediates_discarded

    transform_counts: dict[str, int] = {
        "seed_groups_total": len(groups),
        "seed_groups_kept": len(groups),
    }
    if mode == "longest_clean":
        transform_counts["seed_group_clean_selections"] = clean_selections
        transform_counts["seed_group_fallback_selections"] = fallback_selections

    return kept_indices, drop_reasons, transform_counts


def _apply_dataset_config(
    samples: list[ConversationSample],
    config: ToolMindConfig,
) -> tuple[list[ConversationSample], dict[str, int], dict[str, int]]:
    # Plain dict (not Counter) so seed-drops keys and per-sample drop keys
    # never silently sum; the contract is that they are disjoint.
    drop_reasons: dict[str, int] = {}
    transform_counts: dict[str, int] = {}

    if config.seed_group_filter != "none":
        kept_indices, seed_drops, seed_transforms = _select_from_seed_groups(
            samples, config.seed_group_filter
        )
        samples = [samples[i] for i in kept_indices]
        drop_reasons.update(seed_drops)
        transform_counts.update(seed_transforms)

    if config.drop_non_object_arguments or config.drop_consecutive_text_assistant:
        kept: list[ConversationSample] = []
        non_object_count = 0
        consec_count = 0
        for s in samples:
            drop = False
            if config.drop_non_object_arguments and _has_non_object_arguments(s):
                non_object_count += 1
                drop = True
            if (
                config.drop_consecutive_text_assistant
                and _has_consecutive_text_assistant(s)
            ):
                consec_count += 1
                drop = True
            if not drop:
                kept.append(s)
        if non_object_count:
            drop_reasons["non_object_arguments"] = non_object_count
        if consec_count:
            drop_reasons["consecutive_text_assistant"] = consec_count
        samples = kept

    # Order matters: strip_think_tool before merge_split_assistant. Stripping
    # a solo "think" call can leave a text-only assistant adjacent to a later
    # FC assistant; the merge step then collapses that pair.
    if config.strip_think_tool:
        stripped: list[ConversationSample] = []
        n_samples = 0
        n_defs = 0
        n_calls = 0
        n_obs = 0
        for s in samples:
            out, defs, calls, obs = _strip_think_tool_from_sample_with_counts(s)
            stripped.append(out)
            if defs or calls or obs:
                n_samples += 1
                n_defs += defs
                n_calls += calls
                n_obs += obs
        samples = stripped
        if n_samples:
            transform_counts["samples_with_think_tool"] = n_samples
            transform_counts["think_tool_definitions_removed"] = n_defs
            transform_counts["think_tool_calls_removed"] = n_calls
            transform_counts["think_tool_observations_removed"] = n_obs

    if config.merge_split_assistant:
        merged_samples: list[ConversationSample] = []
        n_samples = 0
        n_pairs = 0
        for s in samples:
            out, pairs = _merge_split_assistant_messages_with_counts(s)
            merged_samples.append(out)
            if pairs:
                n_samples += 1
                n_pairs += pairs
        samples = merged_samples
        if n_samples:
            transform_counts["merge_split_assistant_samples"] = n_samples
            transform_counts["merge_split_assistant_pairs"] = n_pairs

    return samples, drop_reasons, transform_counts


def _download_files(
    sources: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, str]]:
    file_list = sources if sources is not None else SOURCES
    return [
        (
            fname,
            hf_hub_download(
                repo_id=DATASET_ID,
                revision=DATASET_REVISION,
                filename=fname,
                repo_type="dataset",
            ),
        )
        for fname in file_list
    ]


def _accumulate_stage1_issues(
    sample: ConversationSample,
    issues: Counter[str],
) -> None:
    if _has_non_object_arguments(sample):
        issues["samples_with_non_object_arguments"] += 1

    split_pairs = _count_split_assistant_pairs(sample)
    if split_pairs:
        issues["samples_with_split_assistant_messages"] += 1
        issues["split_assistant_message_pairs"] += split_pairs

    text_pairs = _count_consecutive_text_assistant_pairs(sample)
    if text_pairs:
        issues["samples_with_consecutive_text_assistant"] += 1
        issues["consecutive_text_assistant_pairs"] += text_pairs

    think_defs, think_calls, think_calls_without_obs = _count_think_tool_stats(sample)
    if think_defs:
        issues["samples_with_think_tool_definitions"] += 1
        issues["think_tool_definitions"] += think_defs
    if think_calls:
        issues["samples_with_think_tool_calls"] += 1
        issues["think_tool_calls"] += think_calls
    if think_calls_without_obs:
        issues["samples_with_think_tool_without_observation"] += 1
        issues["think_tool_calls_without_observation"] += think_calls_without_obs


def load(
    dataset_config: ToolMindConfig | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    sources = dataset_config.sources if dataset_config is not None else None
    files = _download_files(sources)

    samples: list[ConversationSample] = []
    stage1_issues: Counter[str] = Counter()
    raw_count = 0

    with gc_disabled():
        for source_name, local_path in files:
            with open(local_path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw: dict[str, Any] = orjson.loads(line)
                    conversations: list[dict[str, Any]] = raw["conversations"]
                    raw_tools: list[dict[str, Any]] | None = raw["tools"]

                    sample = ConversationSample(
                        messages=_convert_messages(conversations),
                        tools=[_normalize_tool(t) for t in (raw_tools or [])],
                        dataset=DATASET_ID,
                        sample_id=raw_count,
                        raw={
                            "conversations": conversations,
                            "tools": raw_tools,
                            "source_file": source_name,
                        },
                    )
                    _accumulate_stage1_issues(sample, stage1_issues)
                    samples.append(sample)
                    raw_count += 1

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
