"""TOUCAN: source-faithful complete conversations from three MCP teachers.

Origin, sources and licenses
----------------------------
Agent Ark's ``Agent-Ark/Toucan-1.5M`` is pinned at
``0df3cf37f2abefb380370cfb02eabea2a35ae782`` (see the exact REVISION constant).
Card: https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/blob/0df3cf37f2abefb380370cfb02eabea2a35ae782/README.md
Paper: https://arxiv.org/abs/2510.01179 (TOUCAN, 2025).
Producer: https://github.com/TheAgentArk/Toucan/tree/a1976dea7688c0f36533aa6e3a3fa4c27e017b4b
The pinned card declares Apache-2.0 for the dataset. The producer repository
has an MIT license; its bundled Qwen-Agent files carry Apache-2.0 notices.
These code licenses do not establish licenses of individual MCP responses or
third-party data. Per-response inherited restrictions are unknown.

Generation and source schema
----------------------------
The paper describes question generation/judging and actual MCP execution with
Kimi-K2, GPT-OSS-120B and Qwen3-32B, using Qwen-Agent and OpenAI Agents. Query
diversification, irrelevant server shuffling and multi-turn extension produce
four subset_name strata: single-turn-original, single-turn-diversify,
irrelevant and multi-turn. Quality assessments and target_tools are audit
labels; they neither prove context grounding nor authorize deleting no-call
answers. Kimi has 40 shards/518,516 rows, OSS 47/457,130 and Qwen3 44/551,613.
The complete raw census parsed all 1,527,259 teacher message payloads.
The three-shard SFT selection (119,287 rows) is a separately reformatted,
rebalanced derivative with six columns and tool_call/tool_response roles. It
is censused but is outside this teacher adapter's input contract; it is not
an independent fourth teacher, and no overlap-based deletion is performed.

Teacher rows have nine string columns: uuid, subset_name, messages,
question, available_tools, target_tools, question_quality_assessment,
response_quality_assessment and metadata. messages/available_tools/assessments/
metadata are serialized JSON; target_tools is comma-delimited audit text (one
Qwen value happens to parse as a JSON string). Only messages and verified tool
sources enter model-visible context. uuid supplies existing sample_id; all
original fields remain unchanged in raw. Generation metadata, server snapshots,
question, target labels and scores never supply hidden context. Unknown source
fields, roles, call fields and definition fields quarantine explicitly.

The pinned producer splits one assistant emission into reasoning/text/legacy
function_call messages: Kimi postprocess_fncall_messages lines 134–192 and Nous
lines 127–231 emit an optional reasoning/text prefix followed by call-only
fragments; completion_openai_agent.py lines 326–356 and 407–430 separate native
reasoning from its following call or answer. Only that R? T? C+ / R T grammar
is merged. Consecutive text-only assistants remain separate; a run containing
calls with multiple text targets, trailing text or interleaved reasoning is
quarantined as ambiguous. Source text is never trimmed or separated by invented
newlines. Function results retain exact content and name linkage. Qwen-Agent
executes calls in source order; OSS may execute a parallel batch, but the export
omits result IDs and its fallback may repeat the last call's name. Source name
order must agree and multiplicity must balance before producer-only call_id
values are removed. Full teacher census: Kimi has 1,654,737 calls/results, OSS
1,265,339 and Qwen3 1,491,068. Only OSS calls have IDs; no teacher results
reference them. The system-message totals are 513,970/451,328/541,008,
respectively; absence remains a valid source state. Multi-turn rows can contain
more than two user messages. This adapter conservatively preserves batch order
for curation: source serialization alone does not prove completion independence.
No anonymous result reordering or naive role alternation is assumed.

Exact Kimi sentinel and Nous XML tool templates are extracted from system
content; unrelated text, empty systems and system positions remain. Malformed
or unfamiliar suspected templates quarantine. Future definition epochs are
not moved into the initial tools channel. All definition sources are reconciled
case-sensitively with complete bodies. The only cross-source compatibility
projection is the pinned MCPManager's omission of root title/description/
default/examples/$comment/deprecated/readOnly/writeOnly and empty required.
It is limited to root schema positions and preserves constraints and nested
property names/data. When an actual embedded context exists, an available-only
name is quarantined as unexposed rather than promoted to a callable capability.
System-only names remain visible through their verified definition. In particular
additionalProperties, references and schema dialects are never erased to force agreement. Definition drift is quarantined.

Native reasoning and visible reasoning
-------------------------------------
OSS exports 1,907,946 structured reasoning_content fields across its four
strata. Neither Kimi nor Qwen3 has that source field. Raw assistant tag-bearing
message counts (opening/closing occurrences by message, including literals):
Kimi original has none; diversify has think 3/3 and reasoning 2/2; irrelevant
think 2/1; multi-turn think 24/16 and reasoning 5/5. OSS original has think 1/1;
diversify reasoning 1/0; irrelevant none; multi-turn think 8/0 and reasoning
1/0. Qwen original/diversify have only closing think tags (4 and 3); multi-turn
has think 10,518/10,513; irrelevant has none. These counts do not classify every
literal occurrence as private reasoning. SFT contains think tags in irrelevant
(1/1) and multi-turn (7/4), plus multi-turn reasoning (1/1), but is not loaded.
Optional FilterConfig.strip_thinking removes only structured assistant reasoning_content
and safely closed native spans, shielding literal code and quarantining
ambiguous boundaries. The loader default removes native reasoning; an explicit
false flag retains it. Ordinary prose,
ReAct Thought/Action/Observation text, declared think/sequentialthinking/Clear
Thought tools, their arguments, observations, state and retries always remain.
The loader does not impose anchored loss masks or reasoning visibility on
history. No source-prefix family deletion is proved for this release.

Verified omissions, runtime context and limits
---------------------------------------------
The pinned Qwen producer can append tool hints/irrelevance warnings to inference
input_messages but export original message + responses. Released generation
metadata does not establish those flags for every row; no blanket affected-row
claim is justified. Known true omitted-hint flags quarantine; absent flags
remain an explicit uncertainty. The loader never invents a hint or simulator
state. MCP resource discovery remains visible; the pinned wrapper builds its
callable map before inference and does not grant new callable names from
arbitrary result text. Unknown dynamic-capability protocols quarantine when a
later call is undefined. Exact Exa asynchronous, audited credential, and the
inspected 12306 first-call relative-departure-date checks live in toucan_context.py;
failure/retry observations remain intact. The date check requires a visible
requested date or labeled clock for that exact consuming contract; prior
unclassified results cause abstention and are never promoted to clock evidence.
These bounded context gates do not certify arbitrary natural-language grounding,
policy compliance or the truth of externally returned results.

Exact ordered pipeline
----------------------
1. Iterate pinned config/shard/row order in bounded Arrow batches; classify raw
   fields and parse without duplicate keys, defaults, or mutable raw aliases.
2. Reconstruct source assistant serialization and extract exact initial system
   definitions. Reconcile audited cross-source encodings, then shared complete
   definition reconciliation; preserve all validation constraints.
3. Shared canonical structure/argument parsing, then audited positional/name
   linkage (producer-only IDs checked before removal). Shared result alignment
   defaults on (FilterConfig.align_results); audited positional/name checks
   already require the published order to match. Then shared complete
   supported-schema and static capability checks. Unsupported schema constraints
   are reported separately from invalid arguments.
4. Apply explicit subset/optional upstream-quality selection and exact producer
   artifact/omitted-context/runtime gates. No visible tool episodes are stripped.
5. Optionally remove native reasoning, then apply system override only after
   source validation. Revalidate final structure/linkage/capabilities and every
   affected context gate; a transform cannot rescue earlier invalid context.
6. Require complete response termination with nonempty final assistant and no
   unfinished jobs. A final response containing only recognized balanced native
   reasoning is incomplete even when native reasoning is retained. Native
   control tokens alone also provide no answer. Neither does an open-only native
   tail preceded solely by balanced native spans and whitespace. Code literals
   and other ambiguous boundaries retain their configured handling.
   Cumulative
   shared Levels 1/1.5/2 curation follows, scoped
   separately to teacher/subset/train within the pinned revision. Level 2 keeps
   fewest assistant-content characters with stable source-order ties. Levels
   3–5 are aggregate-only. No cross-teacher/subset/SFT/mixture deletion.

Legacy require_* and drop_* flags cannot disable mandatory production checks.
strip_scaffold_tools=True fails explicitly because those observations can ground
later targets. iter_load uses bounded row processing and disk curation; load
explicitly materializes the final population. Reports are final on exhaustion.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq

from ..format import ConversationSample
from .base import FilterConfig, LoadReport
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .normalization import Reject, json_bytes, normalize_tools, parse_json
from .pipeline import (
    Pipeline,
    RowState,
    Stage,
    link_calls,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_structure,
    validate_termination,
)
from .toucan_context import validate_context, validate_producer_artifacts

DATASET_ID = "Agent-Ark/Toucan-1.5M"
REVISION = "0df3cf37f2abefb380370cfb02eabea2a35ae782"
TEACHER_CONFIGS = ("Kimi-K2", "OSS", "Qwen3")
SUBSETS = ("single-turn-original", "single-turn-diversify", "irrelevant", "multi-turn")


@dataclass
class ToucanConfig:
    subsets: tuple[str, ...] = SUBSETS
    drop_low_quality: bool = False
    min_question_quality: int = 5
    min_scenario_realism: int = 5
    min_completeness: int = 4
    min_conciseness: int = 4
    require_full_tool_use: bool = False
    drop_incomplete_termination: bool = True
    drop_conflicting_duplicate_tools: bool = True
    drop_assistant_protocol_residue: bool = True
    drop_ambiguous_reasoning: bool = True
    drop_incomplete_deep_research: bool = True
    drop_producer_error_targets: bool = True
    drop_invalid_source_tool_linkage: bool = True
    strip_scaffold_tools: bool = False


def _valid_embedded_tool_definition(tool: Any) -> bool:
    return (
        isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
        and bool(tool["function"]["name"])
        and isinstance(tool["function"].get("parameters"), (dict, bool))
    )


_KIMI_TOOL_TEMPLATE_PREFIX = "<|im_system|>tool_declare<|im_middle|>"
_KIMI_TOOL_TEMPLATE_SUFFIX = "<|im_end|>"

_XML_TOOL_TEMPLATE_PREFIX = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)
_XML_TOOL_TEMPLATE_SUFFIX = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def _parse_kimi_tool_body(body: str) -> list[dict[str, Any]] | None:
    try:
        tools = parse_json(body)
    except Reject:
        return None
    if not isinstance(tools, list) or not all(
        _valid_embedded_tool_definition(tool) for tool in tools
    ):
        return None
    return tools


def _parse_xml_tool_body(body: str) -> list[dict[str, Any]] | None:
    """Parse the observed one-JSON-object-per-line XML template body."""

    lines = [line for line in body.splitlines() if line.strip()]
    try:
        tools = [parse_json(line) for line in lines]
    except Reject:
        return None
    if not all(_valid_embedded_tool_definition(tool) for tool in tools):
        return None
    return tools


def _find_valid_template(
    content: str,
    start: int,
    prefix: str,
    suffix: str,
    body_parser: Any,
) -> tuple[int, list[dict[str, Any]]] | None:
    """Return the end and tools of a verified template at ``start``."""

    body_start = start + len(prefix)
    suffix_start = content.find(suffix, body_start)
    while suffix_start >= 0:
        tools = body_parser(content[body_start:suffix_start])
        if tools is not None:
            return suffix_start + len(suffix), tools
        suffix_start = content.find(suffix, suffix_start + 1)
    return None


def _extract_embedded_tool_system_content(
    content: str,
) -> tuple[str, list[dict[str, Any]], bool, int]:
    """Extract verified tool-template spans and preserve surrounding text.

    Both framework templates may occur at the beginning, middle, or end of a
    future system message. A candidate is extracted only when its exact fixed
    framing and serialized tool body validate. Malformed and unfamiliar
    candidates remain byte-for-byte rather than being heuristically truncated.
    Returns ``(remaining_content, extracted_tools, removed, invalid_candidates)``.
    """

    templates = (
        (
            _KIMI_TOOL_TEMPLATE_PREFIX,
            _KIMI_TOOL_TEMPLATE_SUFFIX,
            _parse_kimi_tool_body,
        ),
        (_XML_TOOL_TEMPLATE_PREFIX, _XML_TOOL_TEMPLATE_SUFFIX, _parse_xml_tool_body),
    )
    kept: list[str] = []
    extracted_tools: list[dict[str, Any]] = []
    cursor = 0
    removed = False
    invalid_candidates = 0

    while cursor < len(content):
        candidates = [
            (position, prefix, suffix, parser)
            for prefix, suffix, parser in templates
            if (position := content.find(prefix, cursor)) >= 0
        ]
        if not candidates:
            kept.append(content[cursor:])
            break

        start, prefix, suffix, parser = min(candidates, key=lambda item: item[0])
        match = _find_valid_template(content, start, prefix, suffix, parser)
        if match is None:
            # Keep the unmatched prefix and continue looking after it. This
            # preserves the content exactly while still allowing a later valid
            # template in the same message to be normalized.
            invalid_candidates += 1
            prefix_end = start + len(prefix)
            kept.append(content[cursor:prefix_end])
            cursor = prefix_end
            continue

        end, tools = match
        kept.append(content[cursor:start])
        extracted_tools.extend(tools)
        cursor = end
        removed = True
    else:
        # The final valid template ended exactly at the end of the message.
        pass

    if not removed:
        return content, [], False, invalid_candidates
    remaining = "".join(kept)
    return (
        remaining,
        extracted_tools,
        True,
        invalid_candidates,
    )


def _strip_embedded_tool_system_content(content: str) -> tuple[str, bool]:
    """Compatibility wrapper returning only retained text and removal status."""

    remaining, _tools, removed, _invalid = _extract_embedded_tool_system_content(
        content
    )
    return remaining, removed


def _is_embedded_tool_system_message(content: str) -> bool:
    """Return whether ``content`` consists only of verified tool templates."""

    remaining, removed = _strip_embedded_tool_system_content(content)
    return removed and not remaining


_ROOT_ANNOTATIONS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "$comment",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
_ROW_FIELDS = frozenset(
    {
        "uuid",
        "subset_name",
        "messages",
        "question",
        "available_tools",
        "target_tools",
        "question_quality_assessment",
        "response_quality_assessment",
        "metadata",
    }
)


def _normalize_tool_definitions(value: Any) -> list[dict[str, Any]]:
    tools = normalize_tools(value)
    for tool in tools:
        if tool.keys() != {"type", "function"}:
            raise Reject("unknown_tool_envelope_field")
        if tool["function"].keys() - {"name", "description", "parameters", "strict"}:
            raise Reject("unknown_function_definition_field")
    return tools


def _cross_source_compatible(system: dict[str, Any], available: dict[str, Any]) -> bool:
    """Only the pinned MCPManager's root annotation omissions are comparable."""
    a, b = deepcopy(system), deepcopy(available)
    sa, sb = a["function"].get("parameters"), b["function"].get("parameters")
    if not isinstance(sa, dict) or not isinstance(sb, dict):
        return False
    # A cleaned Qwen schema always has exactly these root fields. Do not apply
    # this equivalence within either source, or recursively to literal data.
    if sa.keys() != {"type", "properties", "required"}:
        return False
    for key in _ROOT_ANNOTATIONS:
        sb.pop(key, None)
    if "required" not in sb and sa["required"] == []:
        sb["required"] = []
    return json_bytes(a) == json_bytes(b)


def _reconcile_tool_definitions(
    system_tools: list[dict[str, Any]],
    available: Any,
    has_template: bool,
    transforms: Counter[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    primary = _normalize_tool_definitions(system_tools)
    secondary = _normalize_tool_definitions(available)
    if not has_template:
        return secondary, False
    by_name = {tool["function"]["name"]: tool for tool in primary}
    mismatch = {json_bytes(t) for t in primary} != {json_bytes(t) for t in secondary}
    if any(tool["function"]["name"] not in by_name for tool in secondary):
        raise Reject("unexposed_available_tool_definition")
    result = list(primary)
    for tool in secondary:
        first = by_name.get(tool["function"]["name"])
        if (
            first is not None
            and json_bytes(first) != json_bytes(tool)
            and _cross_source_compatible(first, tool)
        ):
            if transforms is not None:
                transforms["cross_source_schema_encodings_reconciled"] += 1
            continue
        result.append(tool)
    return result, mismatch


def _convert_messages_with_tool_context(
    raw_msgs: list[dict[str, Any]], transforms: Counter[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], bool]:
    if not isinstance(raw_msgs, list):
        raise Reject("malformed_messages")
    out: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    has_template = False
    seen_conversation = False
    run: list[dict[str, Any]] = []

    def flush() -> None:
        if not run:
            return
        # Pinned Kimi/Nous postprocessors emit R? T? C+ (reasoning, text,
        # call fragments), and OpenAI's exporter emits R C or R T. Text-only
        # assistants have no call-driven split proof and remain separate.
        # A call run with two text targets or trailing text cannot be merged
        # without guessing the original emission boundary.
        kinds = "".join(
            ("C" if message.get("function_call") is not None else "")
            + ("T" if message.get("content") else "")
            + ("R" if message.get("reasoning_content") else "")
            or "E"
            for message in run
        )
        if len(run) > 1:
            if any(message.get("function_call") is not None for message in run):
                if not re.fullmatch(r"R?T?C+", kinds):
                    raise Reject("ambiguous_source_assistant_run")
            elif kinds not in ("RT", "RE"):
                for message in run:
                    out.append(
                        {
                            key: deepcopy(value)
                            for key, value in message.items()
                            if key != "function_call"
                        }
                    )
                run.clear()
                return
        content = "".join(m.get("content") or "" for m in run)
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if any(m.get("content") is None for m in run) and not any(
            "content" in m and m["content"] is not None for m in run
        ):
            assistant["content"] = None
        if any("reasoning_content" in m for m in run):
            values = [m["reasoning_content"] for m in run if "reasoning_content" in m]
            assistant["reasoning_content"] = (
                "".join(v or "" for v in values)
                if any(v is not None for v in values)
                else None
            )
        calls = []
        ids = []
        for source in run:
            fc = source.get("function_call")
            if fc is None:
                continue
            call = {
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": deepcopy(fc["arguments"]),
                },
            }
            # The producer exports these IDs without result references. Validate
            # their shape/uniqueness; source names and order prove linkage below.
            if "call_id" in fc and fc["call_id"] is not None:
                if not isinstance(fc["call_id"], str) or not fc["call_id"]:
                    raise Reject("invalid_producer_call_id")
                ids.append(fc["call_id"])
                call["id"] = fc["call_id"]
            calls.append(call)
        if len(ids) != len(set(ids)):
            raise Reject("invalid_producer_call_id")
        if calls:
            assistant["tool_calls"] = calls
        if len(run) > 1 and transforms is not None:
            transforms["source_assistant_fragments_merged"] += len(run) - 1
        out.append(assistant)
        run.clear()

    for source in raw_msgs:
        if not isinstance(source, dict):
            raise Reject("malformed_message")
        role = source.get("role")
        allowed = {"role", "content"}
        if role == "assistant":
            allowed |= {"function_call", "reasoning_content"}
        elif role == "function":
            allowed.add("name")
        elif role not in ("user", "system"):
            raise Reject("unknown_role")
        if source.keys() - allowed:
            raise Reject("unknown_source_message_field")
        if not isinstance(source.get("content"), (str, type(None))):
            raise Reject("unsupported_content_shape")
        if role == "assistant":
            seen_conversation = True
            if "reasoning_content" in source and not isinstance(
                source["reasoning_content"], (str, type(None))
            ):
                raise Reject("unsupported_reasoning_shape")
            fc = source.get("function_call")
            if fc is not None and (
                not isinstance(fc, dict)
                or fc.keys() - {"name", "arguments", "call_id"}
                or not {"name", "arguments"} <= fc.keys()
            ):
                raise Reject("unknown_source_call_field")
            run.append(source)
            continue
        flush()
        content = source.get("content")
        if role == "system":
            if content is None:
                out.append({"role": "system", "content": None})
                continue
            normalized, extracted, removed, invalid = (
                _extract_embedded_tool_system_content(content)
            )
            # Only exact producer templates may disappear from model context.
            if invalid or (
                not removed
                and ("<|im_system|>tool_declare" in content or "<tools>" in content)
            ):
                raise Reject("malformed_embedded_tool_template")
            if removed and seen_conversation:
                raise Reject("unsupported_later_tool_epoch")
            definitions.extend(extracted)
            has_template |= removed
            if removed and transforms is not None:
                transforms["embedded_tool_templates_extracted"] += 1
            # A standalone tool-declaration system is a protocol envelope; an
            # explicitly empty source system is preserved as a distinct state.
            if normalized or not removed:
                out.append({"role": "system", "content": normalized})
        elif role == "function":
            seen_conversation = True
            message = {"role": "tool", "content": content}
            if "name" in source:
                if not isinstance(source["name"], (str, type(None))):
                    raise Reject("invalid_source_tool_linkage")
                if source["name"] not in (None, ""):
                    message["name"] = source["name"]
            out.append(message)
        else:
            seen_conversation = True
            out.append({"role": role, "content": content})
    flush()
    return out, {}, definitions, has_template


def _convert_messages(
    raw_msgs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    messages, issues, _, _ = _convert_messages_with_tool_context(raw_msgs)
    return messages, issues


def _convert_sample(
    row: dict[str, Any], cfg_name: str, transforms: Counter[str] | None = None
) -> tuple[ConversationSample, dict[str, int]]:
    if row.keys() - _ROW_FIELDS:
        raise Reject("unknown_source_row_field")
    if row.get("subset_name") not in SUBSETS:
        raise Reject("unknown_source_subset")
    raw_messages = (
        parse_json(row["messages"])
        if isinstance(row.get("messages"), str)
        else deepcopy(row.get("messages"))
    )
    if not isinstance(raw_messages, list):
        raise Reject("malformed_messages")
    messages, issues, system_tools, has_template = _convert_messages_with_tool_context(
        raw_messages, transforms
    )
    tools, mismatch = _reconcile_tool_definitions(
        system_tools, row.get("available_tools", "[]"), has_template, transforms
    )
    if mismatch:
        issues["system_available_tool_context_mismatch"] = 1
    if not any(m.get("role") == "system" for m in raw_messages):
        issues["no_system_message"] = 1
    return ConversationSample(
        messages, tools, f"{DATASET_ID}:{cfg_name}", row.get("uuid", ""), raw=row
    ), issues


def _source_linkage(state: RowState, *, align_results: bool = True) -> None:
    """Prove source cardinality and name order before removing producer IDs."""
    messages = state.sample.messages
    for index, message in enumerate(messages):
        calls = message.get("tool_calls", [])
        if not calls:
            continue
        end = index + 1
        while end < len(messages) and messages[end]["role"] == "tool":
            end += 1
        if end - index - 1 != len(calls):
            raise Reject("unbalanced_cardinality")
        ids = [call["id"] for call in calls if "id" in call]
        if len(set(ids)) != len(ids):
            raise Reject("invalid_producer_call_id")
        for offset, call in enumerate(calls, 1):
            echo = messages[index + offset].get("name")
            if echo is not None and echo != call["function"]["name"]:
                raise Reject("invalid_source_tool_linkage")
        for call in calls:
            if "id" in call:
                del call["id"]
                state.transforms["producer_call_ids_removed"] += 1
    link_calls(state, positional=True, parallel=False, align_results=align_results)


def _passes_quality(row: dict[str, Any], config: ToucanConfig) -> bool:
    """Optional audit-score selection; irrelevant rows have no response judge."""

    def score(assessment: Any, key: str) -> int | None:
        item = assessment.get(key) if isinstance(assessment, dict) else None
        value = item.get("score") if isinstance(item, dict) else None
        return value if type(value) is int else None

    try:
        question = parse_json(row.get("question_quality_assessment") or "{}")
        quality = score(question, "question_quality")
        realism = score(question, "scenario_realism")
        if quality is None or quality < config.min_question_quality:
            return False
        if realism is None or realism < config.min_scenario_realism:
            return False
        if row.get("subset_name") == "irrelevant":
            return True
        response = parse_json(row.get("response_quality_assessment") or "{}")
        completeness = score(response, "completeness")
        conciseness = score(response, "conciseness")
        if completeness is None or completeness < config.min_completeness:
            return False
        if conciseness is None or conciseness < config.min_conciseness:
            return False
        if config.require_full_tool_use:
            coverage = response.get("desired_tools_used_percentage")
            return type(coverage) in (int, float) and coverage == 1.0
        return True
    except Reject:
        return False


def _source_policy(state: RowState, *, config: ToucanConfig) -> None:
    if state.sample.raw["subset_name"] not in config.subsets:
        raise Reject("subset_not_selected")
    if config.drop_low_quality and not _passes_quality(state.sample.raw, config):
        raise Reject("low_quality")
    validate_producer_artifacts(state)
    validate_context(state)


def _pipeline(filters: FilterConfig, config: ToucanConfig | None = None) -> Pipeline:
    linkage = partial(_source_linkage, align_results=filters.align_results)
    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("source_structure", validate_structure),
        ("source_linkage", linkage),
        ("source_capabilities", validate_capabilities),
        ("source_policy", partial(_source_policy, config=config or ToucanConfig())),
    ]
    if filters.strip_thinking:
        stages.append(("reasoning", remove_reasoning))
    stages.extend(
        [
            (
                "system_override",
                partial(override_system, override=filters.system_message_override),
            ),
            ("final_structure", validate_structure),
            ("final_linkage", linkage),
            ("final_capabilities", validate_capabilities),
            ("final_artifacts", validate_producer_artifacts),
            ("final_context", validate_context),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def _validate_load_options(configs: Sequence[str], batch_size: int) -> tuple[str, ...]:
    if isinstance(configs, (str, bytes)) or not isinstance(configs, Sequence):
        raise TypeError("configs must be a sequence of teacher config names")
    values = tuple(configs)
    if (
        not values
        or len(set(values)) != len(values)
        or set(values) - set(TEACHER_CONFIGS)
    ):
        raise ValueError(
            f"configs must select distinct teacher configs {TEACHER_CONFIGS}; SFT is a separately reformatted derivative"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    return values


def _resolve_shards(configs: tuple[str, ...], path: str | Path | None) -> list[Path]:
    if path is None:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            DATASET_ID,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=[f"{c}/*.parquet" for c in configs],
        )
    root = Path(path)
    shards = []
    for config in configs:
        selected = sorted((root / config).glob("*.parquet"))
        if not selected:
            raise FileNotFoundError(f"No parquet shards under {root / config}")
        shards.extend(selected)
    return shards


def iter_load(
    dataset_config: ToucanConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    path: str | Path | None = None,
    configs: Sequence[str] = TEACHER_CONFIGS,
    batch_size: int = 32,
    curation_config: CurationConfig | None = None,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    selected = _validate_load_options(configs, batch_size)
    config = dataset_config or ToucanConfig()
    if config.strip_scaffold_tools:
        raise ValueError("TOUCAN cannot strip observable scaffold tool episodes")
    if set(config.subsets) - set(SUBSETS):
        raise ValueError("Unsupported TOUCAN subset selection")
    filters = filter_config or FilterConfig(strip_thinking=True)
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    shards = _resolve_shards(selected, path)
    pipeline = _pipeline(filters, config)
    report = LoadReport(
        f"{DATASET_ID} [{'+'.join(selected)}]", 0, 0, filter_config=filters
    )
    source_counts: Counter[str] = Counter()
    transforms: Counter[str] = Counter()

    def converted(config_shards: list[Path]) -> Generator[CurationInput, None, None]:
        for shard in config_shards:
            with pq.ParquetFile(shard) as source:
                for batch in source.iter_batches(batch_size=batch_size):
                    for raw in batch.to_pylist():
                        report.raw_count += 1
                        source_counts[
                            shard.parent.name
                            + ":"
                            + str(raw.get("subset_name", "unknown"))
                        ] += 1
                        try:
                            sample, issues = _convert_sample(
                                raw, shard.parent.name, transforms
                            )
                        except Reject as exc:
                            report.stage1_drop_reasons[exc.reason] = (
                                report.stage1_drop_reasons.get(exc.reason, 0) + 1
                            )
                            continue
                        report.stage1_count += 1
                        for reason, count in issues.items():
                            report.stage1_issue_counts[reason] = (
                                report.stage1_issue_counts.get(reason, 0) + count
                            )
                        state = pipeline.process(sample)
                        if state is not None:
                            yield CurationInput(
                                state.sample, state.batches, state.parsed_arguments
                            )
        report.dataset_config_count = pipeline.passed["source_policy"]
        report.filtered_count = pipeline.passed["termination"]
        report.filter_drop_reasons = dict(pipeline.drops)
        report.strip_thinking_applied = filters.strip_thinking
        report.dataset_config_transform_counts.update(
            {
                "source": {
                    "revision": REVISION,
                    "configs": list(selected),
                    "files": [f"{p.parent.name}/{p.name}" for p in shards],
                    "split": "train",
                    "subsets": dict(source_counts),
                },
                "pipeline_passed": dict(pipeline.passed),
                "pipeline_drops": dict(pipeline.stage_drops),
                "transformations": dict(transforms + pipeline.transforms),
            }
        )

    def output() -> Generator[ConversationSample, None, None]:
        count = 0
        curation_reports = []
        # Teachers are independent deletion scopes and contiguous in the public
        # source order. Release each teacher's disk spool before reading another.
        for teacher in selected:
            config_shards = [shard for shard in shards if shard.parent.name == teacher]
            with (
                closing(converted(config_shards)) as inputs,
                curate(
                    inputs,
                    scope=lambda record: CurationScope(
                        record.sample.dataset, record.sample.raw["subset_name"], "train"
                    ),
                    config=curation_config or CurationConfig(),
                ) as run,
            ):
                for sample in run:
                    count += 1
                    yield sample
                curation_reports.extend(r.as_dict() for r in run.reports.values())
        report.final_count = count
        report.dataset_config_transform_counts["curation"] = curation_reports

    return output(), report


def load(
    dataset_config: ToucanConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    path: str | Path | None = None,
    configs: Sequence[str] = TEACHER_CONFIGS,
    batch_size: int = 32,
    curation_config: CurationConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    rows, report = iter_load(
        dataset_config,
        filter_config,
        path=path,
        configs=configs,
        batch_size=batch_size,
        curation_config=curation_config,
    )
    return list(rows), report
