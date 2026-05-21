import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import orjson
from datasets import load_dataset

from .._gc import gc_disabled
from ..format import ConversationSample
from .base import FilterConfig, LoadReport, apply_filters


DATASET_ID = "LLM360/TxT360-3efforts"
DATASET_REVISION = "bfc4a082d11967cd7810fe0b773be87bf54fb32e"
CONFIG = "agent"
SPLITS = ("high", "medium", "low")

_THINK_FIELDS = ("think", "think_fast", "think_faster")


@dataclass(slots=True)
class TxT360Config:
    seed_group_filter: Literal["last", "last_clean", "latest_clean_prefix", "none"] = (
        "none"
    )
    require_non_empty_user: bool = False
    drop_user_tool_call_samples: bool = False


@dataclass(frozen=True, slots=True)
class _SampleInfo:
    orphan_tool_messages: int
    unmatched_tool_calls: int
    user_tool_call_messages: int
    has_qualifying_user: bool
    seed_key: str | None = None


def _convert_tools(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": t} for t in raw_tools]


def _extract_think_content(msg: dict[str, Any]) -> str | None:
    for field in _THINK_FIELDS:
        val = msg.get(field)
        if val:
            return val
    return None


def _count_user_tool_call_messages(raw_msgs: list[dict[str, Any]]) -> int:
    return sum(
        1 for msg in raw_msgs if msg["role"] == "user" and bool(msg.get("tool_calls"))
    )


def _has_qualifying_user(messages: list[dict[str, Any]]) -> bool:
    return any(
        msg["role"] == "user"
        and isinstance(msg.get("content"), str)
        and msg["content"].strip()
        for msg in messages
    )


def _convert_messages(
    raw_msgs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    # Returns (messages, tools, orphan_count, unmatched_tc_count).
    # Orphan: tool message with no pending tool_calls remaining.
    # Unmatched: tool_calls never paired with a following tool message
    # (typical for prefix-truncation samples that end mid-FC turn).
    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    orphan_count = 0
    unmatched_tc_count = 0
    pending_tc_count = 0

    for m in raw_msgs:
        match m["role"]:
            case "system":
                raw_tools = m.get("tools") or []
                if raw_tools:
                    tools = _convert_tools(raw_tools)
                messages.append({"role": "system", "content": m.get("content") or ""})

            case "user":
                # User-role tool_calls (empty or not) are ignored; only content
                # is taken. Non-empty user tool_calls are tracked separately and
                # may be dropped via drop_user_tool_call_samples.
                messages.append({"role": "user", "content": m.get("content") or ""})

            case "assistant":
                raw_tc = m.get("tool_calls") or []
                content = m.get("content")

                think = _extract_think_content(m)
                if think:
                    think_wrapped = f"<think>{think}</think>"
                    content = (
                        f"{think_wrapped}\n{content}"
                        if content and content.strip()
                        else think_wrapped
                    )

                if raw_tc:
                    # Flush unmatched from any prior assistant FC message.
                    unmatched_tc_count += pending_tc_count
                    converted_tcs = [
                        {
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in raw_tc
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content if content and content.strip() else None,
                            "tool_calls": converted_tcs,
                        }
                    )
                    pending_tc_count = len(raw_tc)
                else:
                    messages.append({"role": "assistant", "content": content or ""})

            case "tool":
                if pending_tc_count > 0:
                    pending_tc_count -= 1
                else:
                    orphan_count += 1
                messages.append({"role": "tool", "content": m.get("content")})

    unmatched_tc_count += pending_tc_count
    return messages, tools, orphan_count, unmatched_tc_count


def _compute_seed_key(raw_msgs: list[dict[str, Any]]) -> str:
    # SHA-256(first_user_content + NUL + tools_json). Consecutive samples
    # sharing this key are prefix truncations of one source conversation.
    first_user = ""
    tools_json = b""
    for m in raw_msgs:
        if m["role"] == "system":
            tools_json = orjson.dumps(m.get("tools") or [], option=orjson.OPT_SORT_KEYS)
        if m["role"] == "user" and not first_user:
            first_user = m.get("content") or ""
    return hashlib.sha256(first_user.encode() + b"\0" + tools_json).hexdigest()


def _build_seed_groups(sample_infos: list[_SampleInfo]) -> list[tuple[int, int]]:
    n = len(sample_infos)
    if n == 0:
        return []
    groups: list[tuple[int, int]] = []
    cur_start = 0
    for i in range(1, n):
        if sample_infos[i].seed_key != sample_infos[cur_start].seed_key:
            groups.append((cur_start, i))
            cur_start = i
    groups.append((cur_start, n))
    return groups


def _is_clean_seed_candidate(
    sample_info: _SampleInfo,
    dataset_config: TxT360Config,
) -> bool:
    if sample_info.orphan_tool_messages > 0:
        return False
    if sample_info.unmatched_tool_calls > 0:
        return False
    if (
        dataset_config.drop_user_tool_call_samples
        and sample_info.user_tool_call_messages > 0
    ):
        return False
    if dataset_config.require_non_empty_user and not sample_info.has_qualifying_user:
        return False
    return True


def _classify_last_clean_drop_reason(
    sample_info: _SampleInfo,
    dataset_config: TxT360Config,
) -> str:
    if sample_info.orphan_tool_messages > 0:
        return "seed_group_candidate_orphan_tool_messages"
    if sample_info.unmatched_tool_calls > 0:
        return "seed_group_candidate_unmatched_tool_calls"
    if (
        dataset_config.drop_user_tool_call_samples
        and sample_info.user_tool_call_messages > 0
    ):
        return "seed_group_candidate_user_tool_calls"
    if dataset_config.require_non_empty_user and not sample_info.has_qualifying_user:
        return "seed_group_candidate_without_qualifying_user"
    raise AssertionError("last_clean drop reason requested for a clean sample")


def _select_seed_group_indices(
    sample_infos: list[_SampleInfo],
    dataset_config: TxT360Config,
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    groups = _build_seed_groups(sample_infos)
    total_groups = len(groups)
    if total_groups == 0:
        return [], {}, {"seed_groups_total": 0, "seed_groups_kept": 0}

    kept_indices: list[int] = []
    drop_reasons: Counter[str] = Counter()
    drop_reasons["seed_group_intermediate_discarded"] = len(sample_infos) - total_groups

    for start, end in groups:
        last = end - 1
        match dataset_config.seed_group_filter:
            case "last":
                kept_indices.append(last)
            case "last_clean":
                last_info = sample_infos[last]
                if _is_clean_seed_candidate(last_info, dataset_config):
                    kept_indices.append(last)
                else:
                    drop_reasons[
                        _classify_last_clean_drop_reason(last_info, dataset_config)
                    ] += 1
            case "latest_clean_prefix":
                chosen: int | None = None
                for idx in range(last, start - 1, -1):
                    if _is_clean_seed_candidate(sample_infos[idx], dataset_config):
                        chosen = idx
                        break
                if chosen is None:
                    drop_reasons["seed_group_no_clean_prefix"] += 1
                else:
                    kept_indices.append(chosen)
            case "none":
                raise AssertionError("seed-group selection called with filter=none")

    transform_counts = {
        "seed_groups_total": total_groups,
        "seed_groups_kept": len(kept_indices),
    }
    return kept_indices, dict(drop_reasons), transform_counts


def _apply_dataset_config(
    samples: list[ConversationSample],
    sample_infos: list[_SampleInfo],
    dataset_config: TxT360Config,
) -> tuple[list[ConversationSample], list[_SampleInfo], dict[str, int], dict[str, int]]:
    drop_reasons: dict[str, int] = {}
    transform_counts: dict[str, int] = {}

    if dataset_config.seed_group_filter != "none":
        kept_indices, seed_drops, seed_transforms = _select_seed_group_indices(
            sample_infos, dataset_config
        )
        samples = [samples[i] for i in kept_indices]
        sample_infos = [sample_infos[i] for i in kept_indices]
        drop_reasons.update(seed_drops)
        transform_counts.update(seed_transforms)

    if (
        not dataset_config.drop_user_tool_call_samples
        and not dataset_config.require_non_empty_user
    ):
        return samples, sample_infos, drop_reasons, transform_counts

    kept_samples: list[ConversationSample] = []
    kept_infos: list[_SampleInfo] = []
    dropped_user_tc = 0
    dropped_empty_user = 0
    for sample, info in zip(samples, sample_infos, strict=True):
        drop = False
        if (
            dataset_config.drop_user_tool_call_samples
            and info.user_tool_call_messages > 0
        ):
            dropped_user_tc += 1
            drop = True
        if dataset_config.require_non_empty_user and not info.has_qualifying_user:
            dropped_empty_user += 1
            drop = True
        if not drop:
            kept_samples.append(sample)
            kept_infos.append(info)

    if dropped_user_tc:
        drop_reasons["user_tool_call_samples"] = dropped_user_tc
    if dropped_empty_user:
        drop_reasons["no_qualifying_user_message"] = dropped_empty_user

    return kept_samples, kept_infos, drop_reasons, transform_counts


def _accumulate_stage1_issues(info: _SampleInfo, issues: Counter[str]) -> None:
    if info.user_tool_call_messages > 0:
        issues["samples_with_user_tool_call_messages"] += 1
        issues["user_tool_call_messages"] += info.user_tool_call_messages
    if info.orphan_tool_messages > 0:
        issues["samples_with_orphan_tool_messages"] += 1
        issues["orphan_tool_messages"] += info.orphan_tool_messages
    if info.unmatched_tool_calls > 0:
        issues["samples_with_unmatched_tool_calls"] += 1
        issues["unmatched_tool_calls"] += info.unmatched_tool_calls
    if not info.has_qualifying_user:
        issues["samples_with_no_qualifying_user_messages"] += 1


def _process_sample(
    idx: int,
    raw_str: str,
    dataset_name: str,
    needs_seed_keys: bool,
) -> tuple[ConversationSample, _SampleInfo]:
    raw_msgs: list[dict[str, Any]] = orjson.loads(raw_str)
    messages, tools, orphans, unmatched = _convert_messages(raw_msgs)
    info = _SampleInfo(
        orphan_tool_messages=orphans,
        unmatched_tool_calls=unmatched,
        user_tool_call_messages=_count_user_tool_call_messages(raw_msgs),
        has_qualifying_user=_has_qualifying_user(messages),
        seed_key=_compute_seed_key(raw_msgs) if needs_seed_keys else None,
    )
    sample = ConversationSample(
        messages=messages,
        tools=tools,
        dataset=dataset_name,
        sample_id=idx,
        raw={"messages": raw_str},
    )
    return sample, info


def load(
    split: str = "high",
    dataset_config: TxT360Config | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    ds = load_dataset(DATASET_ID, CONFIG, revision=DATASET_REVISION, split=split)
    raw_count = len(ds)
    col_messages = ds.data.table.column("messages").to_pylist()
    del ds

    dataset_name = f"{DATASET_ID}/{split}"
    samples: list[ConversationSample] = []
    sample_infos: list[_SampleInfo] = []
    stage1_issues: Counter[str] = Counter()
    needs_seed_keys = (
        dataset_config is not None and dataset_config.seed_group_filter != "none"
    )

    with gc_disabled():
        for i in range(raw_count):
            sample, info = _process_sample(
                i, col_messages[i], dataset_name, needs_seed_keys
            )
            samples.append(sample)
            sample_infos.append(info)
            _accumulate_stage1_issues(info, stage1_issues)

    report = LoadReport(
        dataset=dataset_name,
        raw_count=raw_count,
        stage1_count=len(samples),
        stage1_issue_counts=dict(stage1_issues),
    )

    if dataset_config is not None:
        samples, sample_infos, ds_drops, ds_transforms = _apply_dataset_config(
            samples, sample_infos, dataset_config
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
