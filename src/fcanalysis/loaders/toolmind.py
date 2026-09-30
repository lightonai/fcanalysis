"""ToolMind: exact assistant-prefix reconstruction in eight released subsets.

Origin and license
------------------
Nanbeige's ToolMind report (https://arxiv.org/abs/2511.15718v2, sections 3-4)
and pinned card https://huggingface.co/datasets/Nanbeige/ToolMind/blob/
8020ed1c03c367e4eb720ac3828ab4b0b95d8baf/README.md describe graph-sampled tool
chains, simulated user/assistant/tool actors, trajectory filtering and per-turn
quality filtering. The described generation simulates tool responses. The card declares
Apache-2.0; this does not establish new permissions for inherited source data.
The seven augmented sources are APIGen-MT, BUTTONInstruct, ToolACE, When2Call,
Glaive function-calling v2, tau-bench train and xLAM function-calling 60k. Their
per-row inherited terms and generation-code license are unknown here; no exact
generation repository/revision is supplied by the release. The complete pinned
physical census independently confirms the paper's
368,611 rows: graphsyn 163,180; APIGen-MT 25,109; BUTTONInstruct 21,202;
ToolACE 7,327; When2Call 17,531; Glaive 20,017; tau 12,882; xLAM 101,363.
There are 2,565,299 messages, 832,373 calls and 95,182 multi-call assistant
occurrences. Every row ends in assistant content beginning with <think>, and
no historical assistant content contains that opening marker.

Released schema and reconstruction proof
----------------------------------------
Each row contains conversations and tools. Conversations contain role/content,
plus wrapped function calls on assistants. Known roles are system/user/assistant/
tool. Structured arguments are serialized without changing values; no absent or
null arguments are changed into {}. Source calls/results in the audited release
lack linkage IDs. Singleton adjacency is unambiguous; parallel pairing needs
the explicit source bundle evidence detailed below. Cardinality alone does not
establish it. Separate consecutive assistant messages are preserved.

Section 4.1 establishes splitting at each assistant message, keeping prior
context, and an anchor-only upstream loss mask. Pinned raw prefix pairs directly
show prior native reasoning absent in history. The source does not provide a
splitter implementation establishing arbitrary whitespace rewrites. Therefore
comparison removes only one exactly leading, closed <think>...</think> span;
all remaining content bytes must agree. Differing separators or placeholders
abstain. Systems, complete raw definitions, calls, arguments, results, source
subset and every other message field remain exact. Every historical assistant
requires its own unique matching complete anchor; missing or conflicting donors
are quarantined, never represented as recovered complete conversations. The
selected longest raw row and its existing sample_id remain unchanged. Donors
cannot be inferred from a first question, source order, or future result.

The verified production configuration retains 28,134 conversations, all with
exactly one user message; 6,941 contain restored multi-assistant/tool histories.
No recovered multi-user conversations are claimed. Its 215,174 missing-donor
exclusions measure absent exact full-prefix donors, not their upstream cause.
Filtering and separator differences must not be assumed to explain every such
absence without a separate source-specific proof.

All eight subsets release native assistant <think> spans. Retained native spans
stay in their original content representation. Native removal runs after
reconstruction. Omitting filter_config enables stripping; an explicit
FilterConfig uses its strip_thinking value (whose field default is False).
Only supported native fields/spans are removed. Visible think tools, calls/results, ordinary
prose and literal code survive. An earlier assistant is not discarded because
its upstream anchor failed filtering; missing required reconstruction evidence
and invalid final-view content are separate reasons for exclusion.

BUTTON tool results are bundled arrays of complete {name, arguments, results}
entries; ToolACE uses {name, results}. Full-corpus inspection establishes these
exact grammars in those two files only. Expansion preserves every entry and its
order, then shared linkage proves exact name/argument echoes (BUTTON) or unique
names (ToolACE). Duplicate, missing and contradictory echoes remain exclusions.
Proven results align to unchanged call order by default; disabling
FilterConfig.align_results rejects candidates needing alignment. Whole-pair
order remains significant for curation. Only the results payload supplies
external state. Ordinary JSON lists in other subsets remain unchanged.

Definitions include wrapped and double-wrapped functions, legacy function-level
required, argument-map schemas, Python primitive type aliases, and null parameter
schemas. Only exact str/int/float/bool/dict/list aliases are mapped to JSON Schema
primitives, recursively in schema positions; defaults and other values stay
exact. Legacy arguments or parameters property maps are wrapped in
object/properties without inferring required or nullability. Function required is moved into parameters
only when absent there or exactly equal; explicit conflicts are quarantined.
A null legacy function required is an audited empty serialization placeholder.
Unknown type encodings, null parameter schemas and invalid constraints are not
repaired by guessing. In particular the cagr definition in xLAM row 0 has an
explicit legacy arguments map; absence of a parameters key must not be mistaken
for a null schema.
Graph definitions also carry input/output descriptions, functionality and output
structure outside the function envelope. These known annotations stay
unchanged in their released tool envelope, including function-level response
contracts. They already belong to the source tools channel; none are moved into
function or promoted from row audit fields. Their generation-time renderer is
unreleased; the paper explains their graph construction origin but does not
establish which teacher rendered every annotation.

The visible-input gate for inherited APIGen retail policies reuses that exact
source policy. Existing credentials use 186 exact field/description contracts,
five exact consuming functions for mixed descriptions and nine exact function
descriptions where a property description is absent. Named independently audited
Amazon/login slots are retained. Three explicitly released public access
constants are allowed only under their exact descriptions. Prior user/system
text and linked result scalar values provide evidence, with exact numeric
spellings and no result-map keys, definition examples or assistant reasoning.
New passwords, hashes, strength checks and ordinary key/code/token meanings stay
separate. Ambiguous or undeclared property names do not establish credentials.
The apiTokenInstance authentication field is covered by its exact released
description. A separate source-template stage rejects the unresolved API-key
placeholder in http.post's explicitly documented URL template; substituted URLs
remain untouched, and no generic whole-URL credential rule is inferred.
These checks do not certify arbitrary policy compliance, generated result truth,
asynchronous task completion, or missing teacher state. A downstream call with
no established initial capability is quarantined; this adapter does not infer
callability from incidental discovery prose. Unknown model-visible fields are
reported as explicit source conversion exclusions.

Ordered pipeline
----------------
1. Read the eight pinned JSONL files separately in stable source order. Parse
   strict JSON, classify source messages, and keep each original raw row intact.
2. Build whole-message raw prefix proofs within one file. Restore only exact
   unambiguous released anchor messages. Missing or conflicting required donors
   quarantine the candidate before optional reasoning removal.
3. Convert source calls/results and complete tool definitions without mutable
   raw aliases. Expand only the two audited bundle protocols while retaining
   every full entry and its source order. Preserve system absence/emptiness/position. Apply
   only the named source-schema encodings; reconcile exact repeated definitions
   and reject unresolved same-name collisions.
4. Validate structure and singleton or explicit bundle linkage; align proven
   results to unchanged call order by default (FilterConfig.align_results).
   Validate active capabilities, supported argument schemas, the exact unresolved
   HTTP source template, then bounded source-visible context rules.
   Fifteen audited GraphSyn/BUTTON function contracts also check absolute
   calendar inputs derived from relative requests before any visible clock or
   result. Explicit-date ambiguities and prior results cause abstention; schema
   examples and native reasoning never become clock evidence.
5. Optionally remove native reasoning and apply the caller's system override.
   Recheck final structure, linkage, capabilities and affected context rules;
   require a nonempty final assistant response. Balanced native spans alone
   never constitute that response, including when reasoning is retained. Native
   control tokens alone also provide no answer. Neither does an open-only native
   tail preceded solely by balanced native spans and whitespace. Code literals
   and other ambiguous boundaries retain their configured handling.
   Transformations cannot rescue a
   source-invalid row. Visible think calls and assistant chronology remain.
6. Select longest valid complete source prefixes by comparing the full restored
   histories. Apply configured cumulative Levels 1/1.5/2 within each source
   file; Levels 3-5 are aggregate-only. Report every exclusive reconstruction and
   validation drop, transformation, curation scope, and actual final_count.
"""

from collections import Counter
from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast

from ..format import ConversationSample
from .apigen_mt import validate_visible_inputs as _validate_apigen_visible_inputs
from .base import FilterConfig, LoadReport
from ._toolmind_credentials import (
    credential_requires_evidence,
    validate_source_placeholders,
)
from ._toolmind_temporal import calendar_slots
from ._toolmind_results import expand_result, result_echo, result_values
from .context import (
    json_scalar_values,
    validate_argument_evidence,
    validate_calendar_evidence,
)
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .normalization import (
    Reject,
    normalize_arguments,
    parse_json,
    serialize_result,
)
from .pipeline import (
    Pipeline,
    RowState,
    Stage,
    link_calls,
    native_reasoning_spans,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_structure,
    validate_termination,
)
from .reconstruction import PrefixRecord, ReconstructionConfig, reconstruct_prefixes
from .source import jsonl_lines
from .legacy_tools import normalize_legacy_tool as _normalize_tool

DATASET_ID = "Nanbeige/ToolMind"
DATASET_REVISION = "8020ed1c03c367e4eb720ac3828ab4b0b95d8baf"
SOURCES = (
    "graph_syn_datasets/graphsyn.jsonl",
    "open_datasets/APIGen-MT-5k-query.jsonl",
    "open_datasets/BUTTONInstruct-query.jsonl",
    "open_datasets/ToolACE-query.jsonl",
    "open_datasets/When2Call-query.jsonl",
    "open_datasets/glaive-function-calling-v2-query.jsonl",
    "open_datasets/tau-train-query.jsonl",
    "open_datasets/xlam-function-calling-60k-query.jsonl",
)


@dataclass(slots=True)
class ToolMindConfig:
    """Reconstruction is mandatory. none disables heuristic grouping, which is
    unsupported; legacy longest labels request only exact proven families.
    FilterConfig.require_* flags cannot disable production validation."""

    sources: list[str] | None = None
    seed_group_filter: Literal["longest", "longest_clean", "none"] = "none"
    drop_non_object_arguments: bool = False
    drop_consecutive_text_assistant: bool = False
    merge_split_assistant: bool = False
    strip_think_tool: bool = False


def _source_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(row, dict) or set(row) != {"conversations", "tools"}:
        raise Reject("unknown_source_row_fields")
    messages = row["conversations"]
    if not isinstance(messages, list) or not messages:
        raise Reject("invalid_source_messages")
    if not isinstance(row["tools"], list):
        raise Reject("invalid_source_tools")
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise Reject("invalid_source_message")
        msg = cast(dict[str, Any], msg)
        role = msg.get("role")
        if not isinstance(role, str):
            raise Reject("unknown_source_role")
        allowed = {"role", "content"}
        if role == "assistant":
            allowed |= {"tool_calls"}
        elif role not in {"system", "user", "tool"}:
            raise Reject("unknown_source_role")
        if msg.keys() - allowed:
            raise Reject("unknown_source_message_fields")
        if "content" in msg and not isinstance(msg["content"], (str, type(None))):
            if role != "tool":
                raise Reject("invalid_source_content")
        if "tool_calls" in msg and not isinstance(msg["tool_calls"], list):
            raise Reject("invalid_source_tool_calls")
        if (
            role == "assistant"
            and index != len(messages) - 1
            and (msg.get("content") or "").startswith("<think>")
        ):
            raise Reject("unexpected_historical_reasoning")
    if messages[-1].get("role") != "assistant" or not (
        messages[-1].get("content") or ""
    ).startswith("<think>"):
        raise Reject("missing_source_reasoning_anchor")
    return messages


def _project_message(message: dict[str, Any]) -> dict[str, Any]:
    projected = dict(message)
    content = message.get("content")
    if (
        message.get("role") == "assistant"
        and isinstance(content, str)
        and content.startswith("<think>")
    ):
        try:
            spans = native_reasoning_spans(content)
        except Reject as exc:
            raise Reject("inseparable_source_reasoning") from exc
        if len(spans) != 1 or spans[0][0] != 0:
            raise Reject("inseparable_source_reasoning")
        projected["content"] = content[spans[0][1] :]
    return projected


def _prefix_record(row: dict[str, Any], index: int | str, source: str) -> PrefixRecord:
    messages = _source_messages(row)
    return PrefixRecord(
        value=(row, index),
        scope=CurationScope(DATASET_ID, source, source.split("/", 1)[0]),
        context=row["tools"],
        messages=messages,
        projected_messages=[_project_message(m) for m in messages],
        required=tuple(
            i for i, m in enumerate(messages[:-1]) if m["role"] == "assistant"
        ),
        anchor=len(messages) - 1,
    )


def _convert_row(
    row: dict[str, Any],
    index: int | str,
    messages: list[dict[str, Any]] | None = None,
    *,
    transforms: Counter[str] | None = None,
    source_file: str | None = None,
) -> ConversationSample:
    source = _source_messages(row) if messages is None else messages
    canonical: list[dict[str, Any]] = []
    for msg in source:
        role = msg["role"]
        if role == "tool" and "content" in msg:
            canonical.extend(
                expand_result(
                    msg["content"], source_file=source_file, transforms=transforms
                )
            )
            continue
        out = {"role": role}
        if "content" in msg:
            out["content"] = (
                serialize_result(msg["content"]) if role == "tool" else msg["content"]
            )
        if role == "assistant" and "tool_calls" in msg:
            calls = []
            for call in msg["tool_calls"]:
                if not isinstance(call, dict) or set(call) != {"function"}:
                    raise Reject("unknown_source_call_fields")
                function = call["function"]
                if not isinstance(function, dict) or set(function) != {
                    "name",
                    "arguments",
                }:
                    raise Reject("unknown_source_call_fields")
                calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": normalize_arguments(function["arguments"])[0],
                        },
                    }
                )
            out["tool_calls"] = calls
        canonical.append(out)
    return ConversationSample(
        messages=deepcopy(canonical),
        tools=[_normalize_tool(t, transforms) for t in row["tools"]],
        dataset=DATASET_ID,
        sample_id=index,
        raw=row,
    )


def _validate_visible_context(
    state: RowState, *, source_file: str | None = None
) -> None:
    """Check audited existing credentials against preceding visible evidence.

    JSON result keys, definition examples, assistant prose/native reasoning and
    future results cannot supply credentials. Numeric OTPs use exact JSON
    spelling; booleans and nulls are never numeric evidence.
    """
    validate_argument_evidence(
        state,
        credential_requires_evidence,
        values=json_scalar_values,
        result_values=lambda value, _message: result_values(
            value, source_file=source_file
        ),
        strict_numeric_tokens=True,
    )

    validate_calendar_evidence(
        state, slots=partial(calendar_slots, source_file=source_file)
    )

    if any(
        m["role"] == "system" and "# Retail agent policy" in (m.get("content") or "")
        for m in state.sample.messages
    ):
        _validate_apigen_visible_inputs(state)


def _pipeline(filters: FilterConfig, *, source_file: str | None = None) -> Pipeline:
    def linkage(state: RowState) -> None:
        link_calls(
            state,
            positional=False,
            parallel=False,
            align_results=filters.align_results,
            result_echo=partial(result_echo, source_file=source_file),
        )

    context = partial(_validate_visible_context, source_file=source_file)

    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("structure", validate_structure),
        ("linkage", linkage),
        ("capabilities", validate_capabilities),
        ("source_templates", validate_source_placeholders),
        ("visible_context", context),
    ]
    if filters.strip_thinking:
        stages.append(("reasoning", remove_reasoning))
    if filters.system_message_override is not None:
        stages.append(
            (
                "system_override",
                partial(override_system, override=filters.system_message_override),
            )
        )
    stages.extend(
        [
            ("final_structure", validate_structure),
            ("final_linkage", linkage),
            ("final_capabilities", validate_capabilities),
            ("final_source_templates", validate_source_placeholders),
            ("final_visible_context", context),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def iter_load(
    dataset_config: ToolMindConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    config = dataset_config or ToolMindConfig()
    if (
        config.strip_think_tool
        or config.merge_split_assistant
        or config.drop_consecutive_text_assistant
    ):
        raise ValueError(
            "ToolMind preserves visible think tools and separate assistant messages"
        )
    sources = SOURCES if config.sources is None else tuple(config.sources)
    if len(sources) != len(set(sources)) or any(
        source not in SOURCES for source in sources
    ):
        raise ValueError("sources must be unique pinned ToolMind source files")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    filters = filter_config or FilterConfig(strip_thinking=True)
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=0,
        stage1_count=0,
        filter_config=filters,
        strip_thinking_applied=filters.strip_thinking,
    )
    pipelines: list[Pipeline] = []
    parse_drops: Counter[str] = Counter()
    conversion_drops: Counter[str] = Counter()
    conversion_transforms: Counter[str] = Counter()
    reconstructions: dict[str, Any] = {}
    curations: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    def output() -> Generator[ConversationSample, None, None]:
        final_count = 0
        for source in sources:
            pipeline = _pipeline(filters, source_file=source)
            pipelines.append(pipeline)
            scope = CurationScope(DATASET_ID, source, source.split("/", 1)[0])

            def records() -> Iterator[PrefixRecord]:
                source_counts[source] = 0
                for local_index, line in enumerate(
                    jsonl_lines(DATASET_ID, DATASET_REVISION, [source])
                ):
                    index = f"{source}:{local_index}"
                    report.raw_count += 1
                    source_counts[source] += 1
                    try:
                        row = parse_json(line)
                        record = _prefix_record(row, index, source)
                    except Reject as exc:
                        parse_drops[exc.reason] += 1
                        continue
                    report.stage1_count += 1
                    yield record

            def validate(
                value: tuple[dict[str, Any], int | str], messages: list[dict[str, Any]]
            ) -> CurationInput | None:
                row, index = value
                try:
                    sample = _convert_row(
                        row,
                        index,
                        messages,
                        transforms=conversion_transforms,
                        source_file=source,
                    )
                except Reject as exc:
                    conversion_drops[exc.reason] += 1
                    return None
                state = pipeline.process(sample)
                return (
                    None
                    if state is None
                    else CurationInput(
                        state.sample, state.batches, state.parsed_arguments
                    )
                )

            reconstructed = reconstruct_prefixes(
                records(),
                validate=validate,
                config=ReconstructionConfig(
                    temporary_directory=curation_config.temporary_directory
                    if curation_config
                    else None
                ),
            )
            result = curate(reconstructed, scope=scope, config=curation_config)
            try:
                for sample in result:
                    final_count += 1
                    yield sample
                reconstructions[source] = reconstructed.report.as_dict()
                curations.extend(r.as_dict() for r in result.reports.values())
            finally:
                result.close()
                reconstructed.close()
        passed = sum((pipeline.passed for pipeline in pipelines), Counter())
        drops = sum((pipeline.drops for pipeline in pipelines), Counter())
        stage_drops = sum((pipeline.stage_drops for pipeline in pipelines), Counter())
        transforms = sum((pipeline.transforms for pipeline in pipelines), Counter())
        report.final_count = final_count
        report.stage1_drop_reasons = dict(parse_drops)
        report.dataset_config_drop_reasons = dict(conversion_drops)
        report.filter_drop_reasons = dict(drops)
        report.filtered_count = passed["termination"]
        report.dataset_config_count = sum(
            r["output_rows"] for r in reconstructions.values()
        )
        report.dataset_config_transform_counts = {
            "source": {
                "revision": DATASET_REVISION,
                "files": list(sources),
                "subsets": source_counts,
            },
            "reconstruction": reconstructions,
            "pipeline_passed": dict(passed),
            "pipeline_drops": dict(stage_drops),
            "transformations": dict(conversion_transforms + transforms),
            "curation": curations,
        }

    return output(), report


def load(
    dataset_config: ToolMindConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[list[ConversationSample], LoadReport]:
    rows, report = iter_load(
        dataset_config,
        filter_config,
        curation_config=curation_config,
        batch_size=batch_size,
    )
    try:
        return list(rows), report
    finally:
        rows.close()
