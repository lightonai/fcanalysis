import ast
from collections import Counter
from dataclasses import dataclass
from typing import Any

import orjson
from datasets import load_dataset

from .._gc import gc_disabled
from ..format import ConversationSample
from .base import FilterConfig, LoadReport, apply_filters

DATASET_ID = "allenai/Dolci-Instruct-SFT-Tool-Use"
DATASET_REVISION = "dc042846f0f2de0f15eedae3d6ced04223ed47eb"


@dataclass(slots=True)
class DolciConfig:
    drop_consecutive_text_text_assistant: bool = False
    merge_text_fc_assistant: bool = False
    drop_conflicting_duplicate_tools: bool = False


def _is_real_fc(fc_str: str | None) -> bool:
    # The function_calls field carries Python-style call syntax or one of the
    # literal sentinels below. JavaScript-style "true"/"false"/"null" appear
    # as data-generation artifacts and must not be parsed as real calls.
    if not fc_str:
        return False
    return fc_str.strip() not in ("", "null", "None", "true", "false")


def _safe_eval_node(node: ast.expr) -> Any:
    # Restricted AST evaluator for keyword-argument values. Accepts literals,
    # containers, unary numeric ops, a small set of binary ops, and the
    # JavaScript-style true/false/null Name nodes. Anything else raises
    # ValueError so the surrounding call is rejected.
    match node:
        case ast.Constant(value=v):
            return v
        case ast.Name(id="true"):
            return True
        case ast.Name(id="false"):
            return False
        case ast.Name(id="null"):
            return None
        case ast.Name():
            raise ValueError
        case ast.List(elts=elts):
            return [_safe_eval_node(e) for e in elts]
        case ast.Dict(keys=keys, values=vals):
            result: dict[Any, Any] = {}
            for k, v in zip(keys, vals, strict=True):
                if k is None:
                    raise ValueError
                result[_safe_eval_node(k)] = _safe_eval_node(v)
            return result
        case ast.Tuple(elts=elts):
            return tuple(_safe_eval_node(e) for e in elts)
        case ast.Set(elts=elts):
            return [_safe_eval_node(e) for e in elts]
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            val = _safe_eval_node(operand)
            if isinstance(val, (int, float)):
                return -val
            raise ValueError
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            val = _safe_eval_node(operand)
            if isinstance(val, (int, float)):
                return val
            raise ValueError
        case ast.BinOp(left=left, op=ast.Mult(), right=right):
            lval = _safe_eval_node(left)
            rval = _safe_eval_node(right)
            if isinstance(lval, list) and isinstance(rval, int):
                return lval * rval
            if isinstance(lval, int) and isinstance(rval, list):
                return rval * lval
            raise ValueError
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            lval = _safe_eval_node(left)
            rval = _safe_eval_node(right)
            if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
                return lval + rval
            if isinstance(lval, list) and isinstance(rval, list):
                return lval + rval
            raise ValueError
        case ast.BinOp(left=left, op=ast.Sub(), right=right):
            lval = _safe_eval_node(left)
            rval = _safe_eval_node(right)
            if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
                return lval - rval
            raise ValueError
        case ast.BinOp(left=left, op=ast.Div(), right=right):
            lval = _safe_eval_node(left)
            rval = _safe_eval_node(right)
            if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
                if rval == 0:
                    raise ValueError
                return lval / rval
            raise ValueError
        case ast.BinOp(left=left, op=ast.FloorDiv(), right=right):
            lval = _safe_eval_node(left)
            rval = _safe_eval_node(right)
            if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
                if rval == 0:
                    raise ValueError
                return lval // rval
            raise ValueError
        case ast.BinOp(left=left, op=ast.Pow(), right=right):
            lval = _safe_eval_node(left)
            rval = _safe_eval_node(right)
            if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
                # Cap exponent magnitude to defend against blowup like 10**1e9.
                if isinstance(rval, int) and abs(rval) > 100:
                    raise ValueError
                return lval**rval
            raise ValueError
        case _:
            raise ValueError


def _extract_dotted_name(node: ast.expr) -> str | None:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=value, attr=attr):
            prefix = _extract_dotted_name(value)
            return None if prefix is None else f"{prefix}.{attr}"
        case _:
            return None


def _parse_single_call(node: ast.expr) -> dict[str, Any] | None:
    match node:
        case ast.Call(func=func_node, args=pos_args, keywords=keywords):
            name = _extract_dotted_name(func_node)
            if name is None or pos_args:
                return None
            kwargs: dict[str, Any] = {}
            for kw in keywords:
                if kw.arg is None:
                    return None
                try:
                    kwargs[kw.arg] = _safe_eval_node(kw.value)
                except ValueError:
                    return None
            try:
                arguments = orjson.dumps(kwargs).decode()
            except orjson.JSONEncodeError:
                return None
            return {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        case ast.Name(id=name):
            return {
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        case ast.Attribute():
            name = _extract_dotted_name(node)
            if name is None:
                return None
            return {
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        case _:
            return None


def _parse_function_calls(fc_str: str) -> list[dict[str, Any]] | None:
    try:
        tree = ast.parse(fc_str, mode="exec")
    except SyntaxError:
        return None
    if not tree.body:
        return None

    tool_calls: list[dict[str, Any]] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            return None
        result = _parse_single_call(stmt.value)
        if result is None:
            return None
        tool_calls.append(result)
    return tool_calls if tool_calls else None


def _convert_sample(
    sample_id: str,
    raw_messages: list[dict[str, Any]],
    dataset_source: str,
) -> tuple[ConversationSample | None, str | None]:
    if len(raw_messages) < 2:
        return None, "too_few_messages"

    has_user = False
    for msg in raw_messages:
        role = msg["role"]
        if role == "user":
            has_user = True
            if _is_real_fc(msg.get("function_calls")):
                return None, "function_calls_on_user"
        elif role == "environment" and _is_real_fc(msg.get("function_calls")):
            return None, "function_calls_on_environment"

    if not has_user:
        return None, "no_user_message"

    tools: list[dict[str, Any]] = []
    sys_msg = raw_messages[0]
    if sys_msg["role"] == "system":
        funcs_str = sys_msg.get("functions")
        if funcs_str and funcs_str.strip() not in ("", "null", "None"):
            try:
                tools = orjson.loads(funcs_str)
            except orjson.JSONDecodeError:
                return None, "malformed_tool_definitions"

    messages: list[dict[str, Any]] = []
    for msg in raw_messages:
        role = msg["role"]
        # Conflates None and "" for content; non-string content is not seen
        # in this dataset, so the or-fallback never triggers in practice.
        content = msg.get("content") or ""

        match role:
            case "system":
                messages.append({"role": "system", "content": content})
            case "user":
                messages.append({"role": "user", "content": content})
            case "assistant":
                fc_str = msg.get("function_calls")
                if _is_real_fc(fc_str):
                    assert isinstance(fc_str, str)
                    parsed = _parse_function_calls(fc_str)
                    if parsed is None:
                        return None, "malformed_function_calls"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content if content.strip() else None,
                            "tool_calls": parsed,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content if content.strip() else None,
                        }
                    )
            case "environment":
                messages.append({"role": "tool", "content": content})
            case _:
                return None, f"unknown_role_{role}"

    messages = _split_bundled_tool_responses(messages)

    return (
        ConversationSample(
            messages=messages,
            tools=tools,
            dataset=DATASET_ID,
            sample_id=sample_id,
            raw={"messages": raw_messages, "dataset_source": dataset_source},
        ),
        None,
    )


def _extract_json_objects(s: str) -> list[str]:
    # Scan for top-level JSON objects/arrays. Combined `{[` depth is
    # sufficient because JSON never has mismatched delimiters.
    objects: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] not in ("{", "["):
            i += 1
            continue
        depth = 0
        in_string = False
        escape_next = False
        start = i
        j = i
        while j < n:
            c = s[j]
            if escape_next:
                escape_next = False
                j += 1
                continue
            if in_string:
                if c == "\\":
                    escape_next = True
                elif c == '"':
                    in_string = False
                j += 1
                continue
            if c == '"':
                in_string = True
            elif c in ("{", "["):
                depth += 1
            elif c in ("}", "]"):
                depth -= 1
                if depth == 0:
                    objects.append(s[start : j + 1])
                    i = j + 1
                    break
            j += 1
        else:
            break
    return objects


def _split_bundled_tool_responses(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # When a parallel-call turn (N>1 calls) is followed by exactly 1 tool
    # message, try two strategies to split it into N tool messages:
    # (1) JSON-boundary detection via brace/bracket depth, then
    # (2) newline-split fallback. If neither yields N parts, leave bundled.
    result: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            result.append(msg)
            i += 1
            continue

        n_calls = len(msg["tool_calls"])
        result.append(msg)
        i += 1
        if n_calls <= 1:
            continue

        j = i
        while j < n and messages[j]["role"] == "tool":
            j += 1
        n_responses = j - i

        if n_responses != 1:
            # Already pre-split (or no responses); pass through.
            while i < j:
                result.append(messages[i])
                i += 1
            continue

        content = messages[i].get("content") or ""

        parts = _extract_json_objects(content)
        if len(parts) == n_calls:
            for part in parts:
                result.append({"role": "tool", "content": part})
            i = j
            continue

        lines = [p for p in content.split("\n") if p.strip()]
        if len(lines) == n_calls:
            for line in lines:
                result.append({"role": "tool", "content": line.strip()})
            i = j
            continue

        # Unresolvable: keep bundled.
        result.append(messages[i])
        i = j

    return result


def _has_conflicting_duplicate_tools(sample: ConversationSample) -> bool:
    if not sample.tools:
        return False
    name_to_serialized: dict[str, set[bytes]] = {}
    for t in sample.tools:
        name = t.get("function", {}).get("name", "")
        name_to_serialized.setdefault(name, set()).add(orjson.dumps(t))
    return any(len(defs) > 1 for defs in name_to_serialized.values())


def _count_stage1_issues(sample: ConversationSample) -> Counter[str]:
    issues: Counter[str] = Counter()
    if _has_conflicting_duplicate_tools(sample):
        issues["conflicting_duplicate_tool_names"] = 1

    hollow = sum(
        1
        for msg in sample.messages
        if msg["role"] == "assistant"
        and not msg.get("content")
        and not msg.get("tool_calls")
    )
    if hollow:
        issues["hollow_assistant_messages"] = hollow

    empty_tool = sum(
        1
        for msg in sample.messages
        if msg["role"] == "tool" and not (msg.get("content") or "").strip()
    )
    if empty_tool:
        issues["empty_tool_content"] = empty_tool

    return issues


def _has_consecutive_text_text_assistant(sample: ConversationSample) -> bool:
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


def _merge_text_fc_assistant_messages(
    sample: ConversationSample,
) -> tuple[ConversationSample, int]:
    messages = sample.messages
    merged: list[dict[str, Any]] = []
    i = 0
    merged_pairs = 0
    while i < len(messages):
        msg = messages[i]
        if (
            i + 1 < len(messages)
            and msg["role"] == "assistant"
            and not msg.get("tool_calls")
            and messages[i + 1]["role"] == "assistant"
            and messages[i + 1].get("tool_calls")
        ):
            merged_pairs += 1
            first = (msg.get("content") or "").strip()
            second = (messages[i + 1].get("content") or "").strip()
            if first and second:
                combined = first + "\n\n" + second
            else:
                combined = first or second
            merged.append(
                {
                    "role": "assistant",
                    "content": combined or None,
                    "tool_calls": messages[i + 1]["tool_calls"],
                }
            )
            i += 2
        else:
            merged.append(msg)
            i += 1

    if merged_pairs == 0:
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


def _apply_dataset_config(
    samples: list[ConversationSample],
    config: DolciConfig,
) -> tuple[list[ConversationSample], dict[str, int], dict[str, int]]:
    drop_reasons: Counter[str] = Counter()
    kept: list[ConversationSample] = []
    for s in samples:
        drop = False
        if (
            config.drop_consecutive_text_text_assistant
            and _has_consecutive_text_text_assistant(s)
        ):
            drop_reasons["consecutive_text_text_assistant"] += 1
            drop = True
        if config.drop_conflicting_duplicate_tools and _has_conflicting_duplicate_tools(
            s
        ):
            drop_reasons["conflicting_duplicate_tool_names"] += 1
            drop = True
        if not drop:
            kept.append(s)

    samples = kept
    transform_counts: dict[str, int] = {}

    if config.merge_text_fc_assistant:
        merged: list[ConversationSample] = []
        n_samples = 0
        n_pairs = 0
        for s in samples:
            new_sample, pairs = _merge_text_fc_assistant_messages(s)
            merged.append(new_sample)
            if pairs:
                n_samples += 1
                n_pairs += pairs
        samples = merged
        if n_samples:
            transform_counts["merge_text_fc_assistant_pairs"] = n_pairs
            transform_counts["merge_text_fc_assistant_samples"] = n_samples

    return samples, dict(drop_reasons), transform_counts


def load(
    dataset_config: DolciConfig | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    raw_count = len(ds)

    table = ds.data.table
    col_id = table.column("id").to_pylist()
    col_messages = table.column("messages").to_pylist()
    col_source = table.column("dataset_source").to_pylist()

    samples: list[ConversationSample] = []
    drop_reasons: Counter[str] = Counter()
    stage1_issues: Counter[str] = Counter()

    with gc_disabled():
        for i in range(raw_count):
            try:
                sample, drop_reason = _convert_sample(
                    col_id[i], col_messages[i], col_source[i]
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected error converting sample {i} (id={col_id[i]})"
                ) from exc
            if drop_reason is not None:
                drop_reasons[drop_reason] += 1
                continue
            assert sample is not None
            samples.append(sample)
            stage1_issues.update(_count_stage1_issues(sample))

    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=raw_count,
        stage1_count=len(samples),
        stage1_drop_reasons=dict(drop_reasons),
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
