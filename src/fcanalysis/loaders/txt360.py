"""TxT360-3efforts agent: reconstruct released assistant prefixes per effort.

Origin, release, and terms
-------------------------
LLM360's K2-V2 report (https://arxiv.org/abs/2512.06201v1, section 6.2) and
https://huggingface.co/datasets/LLM360/TxT360-3efforts/blob/
bfc4a082d11967cd7810fe0b773be87bf54fb32e/README.md describe mostly GPT-OSS-120B
regenerated answers and inherited Nemotron Post Training v1, xLAM,
CommitPackFT, TOUCAN, Hermes, Glaive and ToolACE tool-use data. The card declares
CC BY 4.0. Inherited-source restrictions are not independently relicensed by
this adapter; their exact per-row applicability is unknown. Both pinned code
repositories below carry Apache-2.0 in their root LICENSE files, verified at the
same revisions. Those code terms are distinct from the dataset's CC BY 4.0.

The authoritative reference splitter is LLM360/k2v2_train at
871c086db94d0b835b75c4330b9edf067bc3ef20,
sft/sft_data/create_sft_mix_3efforts.py:63-115. It emits one prefix per assistant,
retains only role and truthy content/tool_calls on prior assistants, and places
one effort-specific field on the anchor, explicitly empty when think is absent.
The matching trainer is Ber666/k2v2-sft at
d951e01f08b9949c1c00db555436205b77998451,
verl/utils/dataset/multiturn_sft_dataset.py:193. Its anchored loss policy belongs
to the trainer, not this loader. The release does not pin its exact generation
command; the reference code explicitly says the published mixture is sampled.

Source schema, reasoning, and limitations
----------------------------------------
Only config agent is supported. Its high/medium/low splits contain respectively
26/1/7 Parquet files and 1,401,471/133,670/804,047 footer rows. Every source row
has one messages field containing serialized JSON. The four roles are
system/user/assistant/tool. Bare complete function definitions occur in system
tools lists; calls are direct name/arguments objects, not OpenAI envelopes.
Absent system content and explicit empty content remain distinct. Empty
non-assistant tool_calls are serialization artifacts; nonempty user/tool calls
have no accepted canonical role mapping and are quarantined.

The native fields are think (high), think_fast (medium), and think_faster (low).
They map verbatim to canonical reasoning_content, including explicit emptiness.
The complete raw scan finds nonempty strings on 1,401,452 high, 133,660 medium,
and 217,852 low anchors; all other anchors explicitly contain empty strings.
The source field name and full JSON string remain unchanged in raw. A missing
historical field is unknown, not evidence of empty reasoning: every such message
requires a unique complete anchor donor at its exact whole-message prefix.
Comparison excludes only the expected effort field; systems, full raw tools,
arguments, results, message keys and chronology stay exact. Conflicting donors,
missing donors, and other-effort fields cannot be blended. No across-effort
restoration, family deletion or curation occurs. Native stripping is optional
and follows reconstruction. Visible think tools, their arguments/results,
ordinary prose and literal code remain intact.

Definitions appearing before the first assistant form the initial environment.
Later tool lists require a verified temporal replacement protocol, which this
release does not establish, and are quarantined rather than front-loaded.
A complete raw census confirms all 2,339,188 rows end in an assistant with
exactly the expected effort field and have exactly one initial system. No
historical structured effort fields or later systems occur. In low, 2,025,554
function definitions use a flattened type="function" envelope; it is normalized
without changing the function body. In high, 465 legacy function-level required
values are explicitly []; they move to parameters only without a conflict.
Low schema-valued positions contain the exact primitive aliases int (10,474),
float (13,738), and dict (4,955); these map to integer, number, and object while
all constraints and instance defaults/enums/constants remain unchanged. High and
medium contain none. Low also contains three each of unsupported any and an
object-valued type; unknown encodings remain unsupported. There are 90,901 empty
function-name occurrences; no names are invented.
Assistant content contains the literal <think> opening text in 2,308 high
message occurrences (1,680 historical), 611 low (14 historical), and zero medium.
These counts include unclosed and code-literal forms. Some are interrupted by
source call serialization without a closing tag. Content stays exact in prefix comparison;
only the subsequent optional shared reasoning stage strips supported spans.
Raw-verified low endpoints also contain leading single or repeated native
openers followed by text ending mid-sentence or mid-literal, without a closing
native token or outside answer. The shared no-outside-answer gate excludes
these structural forms even when native reasoning is retained;
it does not complete tags, invent an answer or judge ordinary prose semantically.
The entire medium
release has zero tool-role messages despite 234,000 tool-call occurrences.
Its 101,384 post-call user messages comprise 100,200 non-JSON strings, 1,044
content-absent tool_calls objects, 136 JSON objects and four JSON arrays. Reviewed
strings are evaluator corrections; reviewed JSON objects/arrays propose another
call. The published splitter preserves non-assistant roles unchanged, and no
released protocol authorizes converting this feedback into execution results.
The final complete-trajectory gate consequently retains 303 text-only medium
conversations with either native-reasoning setting. This is a bounded release
finding, not a claim about the unreleased runtime or every feedback string.

The low effort also contains 2,964 tool-role arrays of {name, results} entries;
2,894 match complete batches with unique names in source order. Because runtime
labels are absent, only exact complete-batch agreement establishes the bundle
protocol. Such arrays expand to singleton arrays retaining every complete
entry. The proof is rebuilt for final linkage and result-evidence projection.
Unmatched singleton arrays retain ordinary payload semantics; other ambiguous
parallel results remain excluded. Four audited low rows have a distinct parser
defect: Champion Meta Data call syntax is embedded in Wordle's difficulty value
while two named results follow one call. That exact corruption is quarantined
independently of schema validity. High and medium have no such arrays.

Unlinked parallel calls cannot acquire positional pairing merely from equal
counts; singleton adjacency and explicit source linkage are accepted. Unknown
source fields, null schemas, collisions, malformed calls and unsupported schemas
are explicit exclusions. Dynamic calls without a proven active definition are
quarantined. The bounded visible-credential gate uses 1,234 exact released
field/description contracts plus 39 exact function-description contracts where
parameter descriptions are absent. It checks API keys, existing login/session
credentials, returned OTPs and structured access credentials against preceding
user/system text or result scalar values, including exact numeric OTP spellings.
Six explicitly released demo/test access constants are allowed only under their
exact descriptions. Creation, password strength/hash checks and ordinary
key/code/token meanings are separate. This does not certify arbitrary
natural-language task success or hidden simulator state; ambiguous descriptions
remain outside this finite gate. Inherited runtime labels are not released.
A separate calendar gate checks nine low consuming slots from eight exact
released function contracts demonstrated by seven retained conversations, and
213 high slot contracts under 162 exact function name/description pairs. High
also requires exact consuming parameter schemas. Its full raw context census
found 2,629 affected rows across 708 distinct context/call/slot cases; every case
was reviewed. These counts do not prove reconstruction or final retention.
Mixed contracts involving explicit years/quarters, partial dates, static
defaults or indirect planning bounds remain outside this gate. Shared checks
abstain after any tool result and for unsupported explicit dates/ranges; they
never use the machine clock or native reasoning as factual evidence. Medium
has no independently selected calendar contracts.

Ordered pipeline
----------------
1. Read the pinned effort's files in order with bounded Arrow batches; parse
   strict JSON and classify every role/message/call field. Preserve the original
   row and sample_id. Build exact raw prefix comparisons within this effort.
2. Restore historical assistant messages only from unique matching released
   anchors, including explicit empty reasoning. Missing or conflicting required
   donors quarantine the candidate. Raw and all
   donors remain unchanged; normalized mutable values are independent copies.
3. Extract initial system tool lists without moving system prose, preserve all
   complete definitions, reconcile exact repeats, and reject same-name conflicts.
   Expand only proved low-effort result bundles, preserving whole entries/order.
   Validate structure and call/result linkage; align proven results to unchanged
   call order by default (FilterConfig.align_results). Validate capabilities and
   schemas, then the bounded visible-credential and effort-specific calendar gates.
4. Optionally remove supported native reasoning with FilterConfig.strip_thinking
   (enabled when filter_config is omitted; an explicit config uses its field
   value, default False), then apply the caller's system override after validation.
5. Recheck final structure/linkage/capabilities and both applicable context gates,
   then require nonempty final assistant response termination. Balanced native
   spans alone never constitute that response, including when reasoning is kept.
   Native control tokens alone also provide no answer. Neither does an open-only
   native tail preceded solely by balanced native spans and whitespace. Code
   literals and other ambiguous boundaries retain their configured handling.
   Neither a closing tag nor a missing answer may be invented.
   No upstream loss
   mask, context-only assistant deletion, or assistant adjacency merge is used.
6. Select proven complete prefix families using exact restored source histories,
   then apply cumulative configured Levels 1/1.5/2 within this effort. Levels
   3-5 are aggregate-only. Reports expose conversion, reconstruction, validation,
   transformations, curation and the actual returned final_count separately.
"""

from collections import Counter
from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast

from ..format import ConversationSample
from .base import FilterConfig, LoadReport
from ._txt360_credentials import credential_requires_evidence
from ._txt360_results import (
    expanded_results,
    proved_results,
    result_echo,
    result_values,
)
from ._txt360_temporal import calendar_slots
from .context import (
    json_scalar_values,
    validate_argument_evidence,
    validate_calendar_evidence,
)
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .normalization import (
    Reject,
    normalize_arguments,
    normalize_tools,
    parse_json,
    serialize_result,
)
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
from .reconstruction import PrefixRecord, ReconstructionConfig, reconstruct_prefixes
from .schema import normalize_schema_types
from .source import parquet_rows

DATASET_ID = "LLM360/TxT360-3efforts"
DATASET_REVISION = "bfc4a082d11967cd7810fe0b773be87bf54fb32e"
CONFIG = "agent"
SPLITS = ("high", "medium", "low")
_THINK_FIELDS = {"high": "think", "medium": "think_fast", "low": "think_faster"}
_FILES = {
    split: tuple(f"agent/{split}-{i:05d}-of-{n:05d}.parquet" for i in range(n))
    for split, n in (("high", 26), ("medium", 1), ("low", 7))
}


@dataclass(slots=True)
class TxT360Config:
    """Reconstruction is mandatory. none disables heuristic grouping, which is
    unsupported; legacy family labels request exact proven families only.
    FilterConfig.require_* flags cannot disable production validation."""

    seed_group_filter: Literal["last", "last_clean", "latest_clean_prefix", "none"] = (
        "none"
    )
    require_non_empty_user: bool = False
    drop_user_tool_call_samples: bool = False


def _has_qualifying_user(messages: list[dict[str, Any]]) -> bool:
    return any(
        m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and bool(m["content"].strip())
        for m in messages
    )


def _source_messages(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if (
        not isinstance(row, dict)
        or set(row) != {"messages"}
        or not isinstance(row["messages"], str)
    ):
        raise Reject("unknown_source_row_fields")
    messages = parse_json(row["messages"])
    if not isinstance(messages, list) or not messages:
        raise Reject("invalid_source_messages")
    field = _THINK_FIELDS[split]
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise Reject("invalid_source_message")
        msg = cast(dict[str, Any], msg)
        role = msg.get("role")
        if not isinstance(role, str):
            raise Reject("unknown_source_role")
        allowed = {"role", "content", "tool_calls"}
        if role == "system":
            allowed |= {"tools"}
        elif role == "assistant":
            allowed |= {field}
        elif role not in {"user", "tool"}:
            raise Reject("unknown_source_role")
        if msg.keys() - allowed:
            raise Reject("unknown_source_message_fields")
        if "content" in msg and not isinstance(msg["content"], (str, type(None))):
            # Structured tool results are losslessly serialized at conversion.
            if role != "tool":
                raise Reject("invalid_source_content")
        if "tool_calls" in msg and not isinstance(msg["tool_calls"], list):
            raise Reject("invalid_source_tool_calls")
        if role != "assistant" and msg.get("tool_calls"):
            raise Reject("nonassistant_source_tool_calls")
        if field in msg and (
            index != len(messages) - 1 or not isinstance(msg[field], str)
        ):
            raise Reject("invalid_source_reasoning_anchor")
    if messages[-1].get("role") != "assistant" or field not in messages[-1]:
        raise Reject("missing_source_reasoning_anchor")
    return messages


def _prefix_record(row: dict[str, Any], index: int, split: str) -> PrefixRecord:
    messages = _source_messages(row, split)
    field = _THINK_FIELDS[split]
    projected = [
        {
            k: v
            for k, v in msg.items()
            if not (msg["role"] == "assistant" and k == field)
        }
        for msg in messages
    ]
    required = tuple(
        i for i, msg in enumerate(messages[:-1]) if msg["role"] == "assistant"
    )
    return PrefixRecord(
        value=(row, index),
        scope=CurationScope(DATASET_ID, CONFIG, split),
        context=None,
        messages=messages,
        projected_messages=projected,
        required=required,
        anchor=len(messages) - 1,
    )


def _convert_tools(
    value: Any, transforms: Counter[str] | None = None, *, split: str | None = None
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise Reject("invalid_source_tools")
    counts = transforms if transforms is not None else Counter()
    functions = deepcopy(value)
    for function in functions:
        if not isinstance(function, dict) or function.keys() - {
            "name",
            "description",
            "parameters",
            "strict",
            "type",
            "required",
        }:
            raise Reject("unknown_source_definition_fields")
        if "type" in function:
            if function.pop("type") != "function":
                raise Reject("invalid_flattened_function_type")
            counts["flattened_function_envelopes_normalized"] += 1
        if "required" in function:
            required = function.pop("required")
            parameters = function.get("parameters")
            if not isinstance(required, list) or not isinstance(parameters, dict):
                raise Reject("invalid_legacy_required")
            if "required" in parameters and parameters["required"] != required:
                raise Reject("conflicting_legacy_required")
            parameters["required"] = required
            counts["legacy_function_required_lifted"] += 1
        if split == "low" and "parameters" in function:
            function["parameters"] = normalize_schema_types(
                function["parameters"],
                {"int": "integer", "float": "number", "dict": "object"},
                transforms=counts,
            )
    return normalize_tools(functions, bare=True)


def _convert_row(
    row: dict[str, Any],
    index: int,
    split: str,
    messages: list[dict[str, Any]] | None = None,
    *,
    transforms: Counter[str] | None = None,
) -> ConversationSample:
    source = _source_messages(row, split) if messages is None else messages
    canonical: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    seen_assistant = False
    field = _THINK_FIELDS[split]
    expanded = expanded_results(source, split=split, transforms=transforms)
    for index_in_source, msg in enumerate(source):
        role = msg["role"]
        if role == "tool" and "content" in msg:
            canonical.extend(
                expanded.get(
                    index_in_source,
                    [{"role": "tool", "content": serialize_result(msg["content"])}],
                )
            )
            continue
        out = {"role": role}
        if "content" in msg:
            out["content"] = (
                serialize_result(msg["content"]) if role == "tool" else msg["content"]
            )
        if role == "system" and "tools" in msg:
            if seen_assistant and msg["tools"]:
                raise Reject("unresolved_later_tool_environment")
            tools.extend(_convert_tools(msg["tools"], transforms, split=split))
        if role == "assistant":
            seen_assistant = True
            if field not in msg or not isinstance(msg[field], str):
                raise Reject("missing_reconstructed_reasoning")
            out["reasoning_content"] = msg[field]
            if "tool_calls" in msg:
                calls = []
                for call in msg["tool_calls"]:
                    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                        raise Reject("unknown_source_call_fields")
                    calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": normalize_arguments(call["arguments"])[0],
                            },
                        }
                    )
                out["tool_calls"] = calls
        canonical.append(out)
    return ConversationSample(
        messages=deepcopy(canonical),
        tools=tools,
        dataset=f"{DATASET_ID}/{split}",
        sample_id=index,
        raw=row,
    )


def _validate_visible_context(state: RowState, *, split: str | None = None) -> None:
    """Check audited existing credentials against preceding visible evidence.

    JSON result keys, definition examples, assistant prose/native reasoning and
    future results cannot supply credentials. Numeric OTPs use exact JSON
    spelling; booleans and nulls are never numeric evidence.
    """
    validate_argument_evidence(
        state,
        credential_requires_evidence,
        values=json_scalar_values,
        result_values=partial(
            result_values, proved=proved_results(state.sample.messages, split=split)
        ),
        strict_numeric_tokens=True,
    )
    validate_calendar_evidence(state, slots=partial(calendar_slots, split=split))


def _pipeline(filters: FilterConfig, *, split: str | None = None) -> Pipeline:
    def linkage(state: RowState) -> None:
        link_calls(
            state,
            positional=False,
            parallel=False,
            align_results=filters.align_results,
            result_echo=partial(
                result_echo, proved=proved_results(state.sample.messages, split=split)
            ),
        )

    context = partial(_validate_visible_context, split=split)

    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("structure", validate_structure),
        ("linkage", linkage),
        ("capabilities", validate_capabilities),
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
            ("final_visible_context", context),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def iter_load(
    split: str = "high",
    dataset_config: TxT360Config | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    if split not in SPLITS:
        raise ValueError(f"unsupported agent split: {split!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = dataset_config or TxT360Config()
    filters = filter_config or FilterConfig(strip_thinking=True)
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    report = LoadReport(
        dataset=f"{DATASET_ID}/{split}",
        raw_count=0,
        stage1_count=0,
        filter_config=filters,
        strip_thinking_applied=filters.strip_thinking,
    )
    pipeline = _pipeline(filters, split=split)
    conversion_drops: Counter[str] = Counter()
    parse_drops: Counter[str] = Counter()
    conversion_transforms: Counter[str] = Counter()
    scope = CurationScope(f"{DATASET_ID}/{split}", CONFIG, split)

    def records() -> Iterator[PrefixRecord]:
        for index, row in enumerate(
            parquet_rows(
                DATASET_ID, DATASET_REVISION, _FILES[split], batch_size=batch_size
            )
        ):
            report.raw_count += 1
            try:
                record = _prefix_record(row, index, split)
            except Reject as exc:
                parse_drops[exc.reason] += 1
                continue
            report.stage1_count += 1
            yield record

    def validate(
        value: tuple[dict[str, Any], int], messages: list[dict[str, Any]]
    ) -> CurationInput | None:
        row, index = value
        try:
            sample = _convert_row(
                row, index, split, messages, transforms=conversion_transforms
            )
            if config.require_non_empty_user and not _has_qualifying_user(
                sample.messages
            ):
                raise Reject("no_qualifying_user_message")
        except Reject as exc:
            conversion_drops[exc.reason] += 1
            return None
        state = pipeline.process(sample)
        return (
            None
            if state is None
            else CurationInput(state.sample, state.batches, state.parsed_arguments)
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

    def output() -> Generator[ConversationSample, None, None]:
        try:
            count = 0
            for sample in result:
                count += 1
                yield sample
            report.final_count = count
            report.stage1_drop_reasons = dict(parse_drops)
            report.dataset_config_drop_reasons = dict(conversion_drops)
            report.filter_drop_reasons = dict(pipeline.drops)
            report.filtered_count = pipeline.passed["termination"]
            report.dataset_config_count = reconstructed.report.output_rows
            report.dataset_config_transform_counts = {
                "source": {
                    "revision": DATASET_REVISION,
                    "config": CONFIG,
                    "split": split,
                    "files": list(_FILES[split]),
                },
                "reconstruction": reconstructed.report.as_dict(),
                "pipeline_passed": dict(pipeline.passed),
                "pipeline_drops": dict(pipeline.stage_drops),
                "transformations": dict(conversion_transforms + pipeline.transforms),
                "curation": [r.as_dict() for r in result.reports.values()],
            }
        finally:
            result.close()
            reconstructed.close()

    return output(), report


def load(
    split: str = "high",
    dataset_config: TxT360Config | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[list[ConversationSample], LoadReport]:
    rows, report = iter_load(
        split,
        dataset_config,
        filter_config,
        curation_config=curation_config,
        batch_size=batch_size,
    )
    try:
        return list(rows), report
    finally:
        rows.close()
