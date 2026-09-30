"""UltraData Tool-Use: source-faithful, scoped conversion of eight source families.

Origin, release and license
---------------------------
OpenBMB/MiniCPM Team released UltraData-SFT-Agent-2609 for MiniCPM5-2B on
2026-09-07. Card: https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609
Revision: f684cc1a9f3e19f6f4929102cd9b06cc0b895b8a. The associated UltraData
paper (https://arxiv.org/abs/2602.09003) describes the broad L0-L4 framework;
https://github.com/OpenBMB/MiniCPM describes the model recipe. Neither supplies
this release's exact exporter, teacher revisions, environments or verifier code.
The pinned LICENSE contains Apache-2.0. The card additionally requires compliance
with inherited-source terms and restricts unchanged redistribution; inherited
rights and the interaction of those statements remain unresolved here.

Generation and source schema
----------------------------
The card describes seed collection, task rewriting/expansion, environment/tool
configuration, teacher rollouts in real or simulated harnesses, task-specific
verification, then trajectory organization. These are publisher claims, not
locally reproduced generation. Only Tool-Use/train's nine JSONL shards are read.
The complete raw census has 82,760 rows: areal_tau2 15,246; simia_airline 7,199;
simia_retail 12,800; toolmind_graphsyn 15,283; spider_bird_merged_traj 20,445;
synth_memory_20260608 5,000; synth_search_20260609 5,000; and
tau3_banking_sft_glm-5.2_20260625 1,787. Labels are published source labels;
they do not prove seed identity, benchmark split membership or regeneration.
Several source families span shards, so shards are not curation boundaries.

Rows contain uuid/messages/tools/source/domain. Messages use system, user,
assistant and tool roles, string content and structured reasoning_content on
assistants. Every call has wrapped name/object-arguments. Database trajectories
carry complete call/result IDs; the other families have anonymous tool results.
All definitions are preserved, including GraphSyn's outer input/output
annotations. The opt-in shared legacy grammar handles argument/property maps,
redundant function envelopes, function-level required and exact Python primitive
type aliases. Unknown encodings, conflicting definitions and invalid arguments
remain exclusions. No textual case, whitespace, punctuation or values change.

Reasoning, masks and completeness
--------------------------------
All eight sources contain native reasoning_content; explicitly empty strings
also occur. Some historical assistants retain reasoning and others do not.
The available evidence does not establish this export's reasoning-omission or
assistant-prefix-splitting rule. No reasoning is restored from another row and
no completeness claim extends to unavailable producer histories. Published
empty strings remain empty. Native removal is optional: an omitted filter_config
uses strip_thinking=True; an explicit FilterConfig uses its own value (False by
default). Visible think tools occur in simia_airline and toolmind_graphsyn and
remain intact, as do their arguments/results and ordinary reasoning-like prose.
Malformed inline native boundaries are excluded in both modes; literal code is
shielded. Retaining native reasoning does not waive this source-export check.
Areal and banking initial greetings have loss=False. That upstream training mask
stays in raw; it never deletes a message or imposes a trainer loss policy.

Some rows end in calls lacking results. All database and banking rows end in
user messages, including acknowledgements with [FINISHED] or ###STOP###. These are preserved;
no endpoint is manufactured by trimming users, appending an answer or borrowing
results. Such rows fail the final-assistant contract. Banking's nested discovery
runtime is additionally uncertified and explicitly quarantined. General result
truth, policy interpretation, task success and benchmark decontamination are
not certified by mechanical acceptance.

The original ToolMind GraphSyn reconstruction contract cannot be transferred
solely because this release reuses that source label. Inspected hypothetical
donor matches can replace already nonempty historical reasoning with different
reasoning, which does not establish an omitted field. The loader preserves these
released histories independently, including all explicit empty native fields.
Every source family contains both empty and nonempty reasoning fields; this
does not imply that every assistant in a retained trajectory reasons natively.

Tau-family context gates require protected IDs in strictly earlier user text or
recognized result values. Retail also requires a successful visible lookup and
consistent authenticated user ID; known lookup errors do not authenticate, but
can precede recovery. Unknown lookup envelopes reject. The optional retail
order-ID '#' comparison never rewrites source arguments. Inherited GraphSyn
credential/calendar rules and its unsupported tool-update placeholder remain
bounded source policies, not general semantic checks.

Ordered pipeline
----------------
1. Stream the nine pinned files in physical order, parse strict JSON and select
   configured native source labels. Preserve the exact row in raw with no
   mutable canonical aliases; remove only audited loss metadata from messages.
2. Convert the selected legacy tool grammar and serialize object arguments;
   reconcile complete definitions and validate canonical message/call structure.
3. Establish immediate one-result-per-call correspondence using the shared
   linker. Explicit IDs may establish pairing; anonymous singleton adjacency
   is accepted. No anonymous multicall positional contract is established here,
   so those candidates are excluded. Arrays remain individual result payloads.
   Align proven results to unchanged call order by default; whole-pair order
   remains significant for deduplication.
4. Validate supported schemas/capabilities and bounded source context: released
   policy/clock/memory system context, tau identifier and authentication inputs,
   and the exact source prohibition on simultaneous assistant prose and a call,
   and exact inherited GraphSyn credential/calendar/placeholder contracts.
   Unresolved tool-epoch and banking discovery protocols are quarantined.
5. Validate inline native boundaries in both modes, then optionally strip
   native reasoning and apply system override. Recheck exact
   source system preservation, mechanics, capabilities and source context;
   require a nonempty final non-call assistant. Since original system context
   carries policies, clocks or memory, an override changing that context is
   excluded, including adding a system to a source that lacked one. This is a
   deliberately conservative override policy, not arbitrary policy inference.
   Rebuilding final call coordinates uses already-proved pairing: these two
   transforms cannot change calls/results or their relative order.
6. Apply disk-backed Levels 1/1.5/2 deduplication separately per native source
   label across all its shards. Levels 3-5 are audit-only. No cross-dataset or
   cross-source deletion is performed here. Report raw, conversion, validation,
   source and curation counts after exhausting the returned iterator.

Use UltraDataToolUseConfig(sources=("simia_retail",)) to select labels; omission
selects all eight. Selection still traverses all nine shards in their published
order. iter_load() yields only after complete curation; close an abandoned
iterator to release temporary disk. load() materializes the selected output.
Raw IDs/labels are not evidence of benchmark decontamination or cross-source
novelty. Exact counts and configurations are in
``tests/fixtures/loaders/ultradata_tool_use``. The independent full-source test
``tests/e2e/test_ultradata_projection.py`` verifies source occurrences and retained
messages/definitions/raw without using either definition converter; strict
JSON/native-boundary primitives remain shared. It does not independently prove
every rejected-row or deduplication decision.
"""

from collections import Counter
from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import re
from typing import Any

from ..format import ConversationSample
from ._toolmind_credentials import (
    credential_requires_evidence,
    validate_source_placeholders,
)
from ._toolmind_temporal import calendar_slots
from .base import FilterConfig, LoadReport
from .context import (
    contains_literal_token,
    json_scalar_values,
    validate_argument_evidence,
    validate_calendar_evidence,
)
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .legacy_tools import normalize_legacy_tool
from .normalization import Reject, normalize_tools, parse_json
from .pipeline import (
    Pipeline,
    RowState,
    Stage,
    link_calls,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_reasoning_boundaries,
    validate_structure,
    validate_termination,
)
from .source import jsonl_lines

DATASET_ID = "openbmb/UltraData-SFT-Agent-2609"
DATASET_REVISION = "f684cc1a9f3e19f6f4929102cd9b06cc0b895b8a"
FILES = tuple(f"data/Tool_Use/Tool_Use_part-{i}-of-9.jsonl" for i in range(1, 10))
SOURCES = (
    "areal_tau2",
    "simia_airline",
    "simia_retail",
    "toolmind_graphsyn",
    "spider_bird_merged_traj",
    "synth_memory_20260608",
    "synth_search_20260609",
    "tau3_banking_sft_glm-5.2_20260625",
)
_LEGACY = frozenset(
    {"toolmind_graphsyn", "synth_memory_20260608", "synth_search_20260609"}
)
_TAU = frozenset({"areal_tau2", "simia_airline", "simia_retail"})
_AUTH = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
_OPAQUE = frozenset(
    {
        "user_id",
        "order_id",
        "product_id",
        "item_ids",
        "new_item_ids",
        "payment_method_id",
        "payment_id",
        "reservation_id",
        "customer_id",
        "line_id",
        "bill_id",
    }
)
_USER_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.]*_[0-9]+")
# Observed unsuccessful lookup envelopes are not credentials or successful
# authentication. Preserve the payload and allow the subsequent visible retry.
_AUTH_FAILURE = re.compile(
    r"(?:Error: user not found|USER_NOT_FOUND|User not found|user not found|"
    r"user not found by email|User not found by email\.|"
    r"ERROR: No user found for the provided email\.|"
    r"User not found for email [^\s]+|"
    r"USER_NOT_FOUND: No user with email '[^'\n]+'(?: was found\.)?)"
)
_ROLE_FIELDS = {
    "system": {"role", "content"},
    "user": {"role", "content"},
    "assistant": {"role", "content", "reasoning_content", "tool_calls", "loss"},
    "tool": {"role", "content", "tool_call_id"},
}


@dataclass(slots=True)
class UltraDataToolUseConfig:
    sources: tuple[str, ...] = SOURCES

    def __post_init__(self) -> None:
        if (
            not self.sources
            or len(set(self.sources)) != len(self.sources)
            or any(s not in SOURCES for s in self.sources)
        ):
            raise ValueError(
                "sources must be a nonempty selection of unique published source labels"
            )


def _convert_row(raw: Any, transforms: Counter[str]) -> ConversationSample:
    if not isinstance(raw, dict) or set(raw) != {
        "uuid",
        "source",
        "domain",
        "messages",
        "tools",
    }:
        raise Reject("unknown_source_row_shape")
    if raw["source"] not in SOURCES or raw["domain"] != "Tool_Use":
        raise Reject("unknown_source_scope")
    if not isinstance(raw["uuid"], str) or not raw["uuid"]:
        raise Reject("invalid_source_uuid")
    if not isinstance(raw["messages"], list) or not isinstance(raw["tools"], list):
        raise Reject("malformed_source_row")
    messages = []
    for original in raw["messages"]:
        if (
            not isinstance(original, dict)
            or not isinstance(original.get("role"), str)
            or original["role"] not in _ROLE_FIELDS
        ):
            raise Reject("unknown_source_role")
        role = original["role"]
        if original.keys() - _ROLE_FIELDS[role] or not isinstance(
            original.get("content"), str
        ):
            raise Reject("unknown_source_message_shape")
        if role == "assistant" and not isinstance(
            original.get("reasoning_content"), str
        ):
            raise Reject("unsupported_source_reasoning")
        message = deepcopy(original)
        if "loss" in message:
            if (
                raw["source"] not in {"areal_tau2", SOURCES[-1]}
                or message["loss"] is not False
            ):
                raise Reject("unsupported_source_loss")
            del message["loss"]
            transforms["source_loss_metadata_removed"] += 1
        messages.append(message)
    if raw["source"] in _LEGACY:
        tools = [normalize_legacy_tool(tool, transforms) for tool in raw["tools"]]
    else:
        for tool in raw["tools"]:
            if (
                not isinstance(tool, dict)
                or set(tool) != {"type", "function"}
                or not isinstance(tool["function"], dict)
                or set(tool["function"]) != {"name", "description", "parameters"}
            ):
                raise Reject("unknown_source_definition_shape")
        tools = normalize_tools(raw["tools"])
    return ConversationSample(
        messages=messages,
        tools=tools,
        dataset=DATASET_ID,
        sample_id=raw["uuid"],
        raw=raw,
    )


def _system_context(state: RowState) -> None:
    original = [m for m in state.sample.raw["messages"] if m["role"] == "system"]
    final = [m for m in state.sample.messages if m["role"] == "system"]
    if final != original:
        raise Reject("changed_source_system_context")


def _argument_leaves(value: Any, field: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _argument_leaves(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _argument_leaves(child, field)
    elif isinstance(value, str):
        yield field, value


def _tau_context(state: RowState) -> None:
    """Only prior users and actual result values supply protected input evidence."""
    users: list[str] = []
    evidence: set[str] = set()
    authenticated: str | None = None
    auth_results = {
        batch.result_indices[ci]
        for batch in state.batches
        for ci, call in enumerate(
            state.sample.messages[batch.assistant_index].get("tool_calls", [])
        )
        if call["function"]["name"] in _AUTH
    }
    retail = any(
        m["role"] == "system" and "# Retail agent policy" in m["content"]
        for m in state.sample.messages
    )
    for index, message in enumerate(state.sample.messages):
        if message["role"] == "user":
            users.append(message["content"])
        elif message["role"] == "tool":
            try:
                result = parse_json(message["content"])
            except Reject:
                result = None
            failed_auth = False
            if index in auth_results:
                if isinstance(result, dict) and set(result) in (
                    {"result"},
                    {"user_id"},
                ):
                    value = next(iter(result.values()))
                    if isinstance(value, str) and _AUTH_FAILURE.fullmatch(value):
                        failed_auth = True
                    elif not isinstance(value, str) or not _USER_ID.fullmatch(value):
                        raise Reject("unrecognized_source_authentication_result")
                    elif authenticated is not None and value != authenticated:
                        raise Reject("conflicting_authenticated_user")
                    else:
                        authenticated = value
                elif not isinstance(result, dict) or set(result) != {"error"}:
                    raise Reject("unrecognized_source_authentication_result")
            if (
                result is not None
                and not failed_auth
                and not (isinstance(result, dict) and result.get("error"))
            ):
                evidence.update(json_scalar_values(result))
        elif message["role"] == "assistant":
            for ci, call in enumerate(message.get("tool_calls", [])):
                name = call["function"]["name"]
                arguments = state.parsed_arguments[index, ci]
                for field, value in _argument_leaves(arguments):
                    if field in _OPAQUE or (
                        name == "get_details_by_id" and field == "id"
                    ):
                        known = value in evidence or contains_literal_token(
                            value, users
                        )
                        if field == "order_id" and re.fullmatch(r"#W[0-9]+", value):
                            known |= value[1:] in evidence or contains_literal_token(
                                value[1:], users
                            )
                        if not value or not known:
                            raise Reject("ungrounded_identifier")
                        if (
                            retail
                            and field == "user_id"
                            and authenticated is not None
                            and value != authenticated
                        ):
                            raise Reject("conflicting_authenticated_user")
                    if (
                        name in _AUTH
                        and field in {"email", "first_name", "last_name", "zip"}
                        or name == "get_customer_by_phone"
                        and field == "phone_number"
                        or name == "get_customer_by_name"
                        and field in {"full_name", "dob"}
                    ) and not contains_literal_token(
                        value, users, case_insensitive=field != "zip"
                    ):
                        raise Reject("ungrounded_authentication_input")
                if (
                    retail
                    and name
                    not in _AUTH | {"calculate", "think", "transfer_to_human_agents"}
                    and authenticated is None
                ):
                    raise Reject("missing_visible_authentication")


def _visible_context(state: RowState) -> None:
    _system_context(state)
    source = state.sample.raw["source"]
    if source == SOURCES[-1]:
        raise Reject("unverified_source_discovery_protocol")
    if source in _TAU:
        systems = [m["content"] for m in state.sample.messages if m["role"] == "system"]
        text_or_call = any(
            "You cannot do both at the same time." in text for text in systems
        )
        for message in state.sample.messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                if text_or_call and message["content"].strip():
                    raise Reject("source_forbids_text_with_tool_call")
        _tau_context(state)
    elif source == "toolmind_graphsyn":
        for message in state.sample.messages:
            if (
                message["role"] == "user"
                and message["content"]
                == "I have updated some more functions you can choose from. What about now?"
            ):
                raise Reject("unresolved_source_tool_epoch")
        validate_source_placeholders(state)
        validate_argument_evidence(
            state,
            credential_requires_evidence,
            values=json_scalar_values,
            strict_numeric_tokens=True,
        )
        validate_calendar_evidence(
            state,
            slots=partial(
                calendar_slots, source_file="graph_syn_datasets/graphsyn.jsonl"
            ),
        )


def _pipeline(filters: FilterConfig) -> Pipeline:
    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("structure", validate_structure),
        (
            "linkage",
            partial(
                link_calls,
                positional=False,
                parallel=False,
                align_results=filters.align_results,
            ),
        ),
        ("capabilities", validate_capabilities),
        ("visible_context", _visible_context),
        ("native_boundaries", validate_reasoning_boundaries),
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
            # Only reasoning and system positions changed after proved source binding.
            (
                "final_linkage",
                partial(
                    link_calls,
                    positional=True,
                    parallel=False,
                    align_results=filters.align_results,
                ),
            ),
            ("final_capabilities", validate_capabilities),
            ("final_visible_context", _visible_context),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def iter_load(
    dataset_config: UltraDataToolUseConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dataset_config is not None and not isinstance(
        dataset_config, UltraDataToolUseConfig
    ):
        raise TypeError("dataset_config must be UltraDataToolUseConfig or None")
    config = dataset_config or UltraDataToolUseConfig()
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
    pipelines = {source: _pipeline(filters) for source in SOURCES}
    source_counts: Counter[str] = Counter()
    drops: Counter[str] = Counter()
    transforms: Counter[str] = Counter()
    retained: Counter[str] = Counter()

    def candidates() -> Iterator[CurationInput]:
        for line in jsonl_lines(DATASET_ID, DATASET_REVISION, FILES):
            report.raw_count += 1
            try:
                row = parse_json(line)
                if isinstance(row, dict) and isinstance(row.get("source"), str):
                    source_counts[row["source"]] += 1
                    if row["source"] in SOURCES and row["source"] not in config.sources:
                        drops["unselected_source"] += 1
                        continue
                sample = _convert_row(row, transforms)
            except Reject as exc:
                drops[exc.reason] += 1
                continue
            source = row["source"]
            report.stage1_count += 1
            state = pipelines[source].process(sample)
            if state is not None:
                yield CurationInput(state.sample, state.batches, state.parsed_arguments)

    def output() -> Generator[ConversationSample, None, None]:
        result = curate(
            candidates(),
            scope=lambda item: CurationScope(
                DATASET_ID, item.sample.raw["source"], "train"
            ),
            config=curation_config,
        )
        try:
            for sample in result:
                retained[sample.raw["source"]] += 1
                yield sample
        finally:
            result.close()
        report.final_count = sum(retained.values())
        report.stage1_drop_reasons = dict(drops)
        report.dataset_config_count = report.stage1_count
        report.filtered_count = sum(p.passed["termination"] for p in pipelines.values())
        report.filter_drop_reasons = dict(
            sum((p.drops for p in pipelines.values()), Counter())
        )
        report.dataset_config_transform_counts = {
            "source": {
                "revision": DATASET_REVISION,
                "files": list(FILES),
                "subsets": dict(source_counts),
            },
            "source_retained": dict(retained),
            "pipeline_by_source": {
                s: {
                    "passed": dict(p.passed),
                    "drops": dict(p.drops),
                    "stage_drops": dict(p.stage_drops),
                }
                for s, p in pipelines.items()
            },
            "transformations": dict(
                transforms + sum((p.transforms for p in pipelines.values()), Counter())
            ),
            "curation": [entry.as_dict() for entry in result.reports.values()],
        }

    return output(), report


def load(
    dataset_config: UltraDataToolUseConfig | None = None,
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
