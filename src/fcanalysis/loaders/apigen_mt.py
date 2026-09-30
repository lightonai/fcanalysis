"""Salesforce APIGen-MT-5k: complete, sequential simulated conversations.

Origin and authoritative sources
--------------------------------
Prabhakar et al., *APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation
via Simulated Agent-Human Interplay* (2025), https://arxiv.org/abs/2504.03601v1.
The released card is https://huggingface.co/datasets/Salesforce/APIGen-MT-5k/
blob/abc4a517d67c541f85f6470cbd8fd3186b36830e/README.md. The project repository,
https://github.com/apigen-mt/apigen-mt.github.io, contains the website rather than
an executable, revision-pinned generator. Related model/training code is at
https://github.com/SalesforceAIResearch/xLAM. The inherited simulator APIs and
policies are from https://github.com/sierra-research/tau-bench; the release does
not identify its simulator commit or publish per-row blueprints/teacher state.

License
-------
The pinned dataset card above declares Creative Commons
Attribution-NonCommercial 4.0 (``cc-by-nc-4.0``). Its Ethical Considerations
separately describe a research-only release; Data Licenses state that the
GPT-4-generated portion should not be used to develop models competing with
OpenAI. These are the release's declarations, distinct from licenses of
inherited tau-bench policies/APIs or related xLAM code. The raw schema has no
per-row teacher or license metadata.

Generation, subsets, and released schema
---------------------------------------
The paper describes API-graph/context sampling, PersonaHub-conditioned task
blueprints, execution/policy checks, committee review and iterative refinement,
then simulated user/agent/environment interaction with success-based rejection
sampling. Task recombination and multiple successful attempts diversify the
trajectories. The card names GPT-4o and DeepSeek-V3; the paper also names R1.
These are generation claims, not proof that every released target is grounded.
The paper's 3,820-trajectory experiment and 28 APIs are not this file's census.
The 5,000-row release is a subset of xLAM-2 training, distinct from original
single-turn APIGen/xLAM-60k; no row-level reuse relationship is released.

The pinned ``apigen-mt_5k.json`` is one JSON array, config ``dataset``, split
``train``, with exactly ``system``, serialized JSON ``tools``, and
``conversations`` per row. Each message contains exactly ``from`` and ``value``.
Tools are bare function definitions with complete name/description/parameters.
Calls are serialized objects with name and arguments; arguments are either
objects (16,929 calls) or serialized objects (5,026). Results are source strings.
There are 1,589 airline rows and 3,411 retail rows, distinguished by two complete
system policies. Three tool lists occur: airline 14 tools; retail 15 without
think (1,400 rows) or 16 including think (2,011). Across them are 26 distinct
names, not 28. No domain field, source ID, scores, or source-prefix families are
released. Curation scope is this one released train split.

Reasoning representations
-------------------------
Neither domain has a dedicated native reasoning field or native
``<think>``/``<reasoning>``/``<analysis>`` spans or chat-control tokens in the
pinned raw turns. Teacher-model reasoning claims do not establish such released
content. ``FilterConfig.strip_thinking`` remains optional (loader default true);
it removes supported native reasoning only and does not change retained content
or counts at this pin.

The observable ``think`` tool is declared in all 1,589 airline rows and 2,011
retail rows; 1,400 retail rows omit it. Its 849 raw calls (640 airline, 209
retail) each contain a ``thought`` argument and an immediately following empty
observation. The declared operation appends the thought to a log without
retrieving information or changing the database. On retained rows its complete
definition, arguments and results remain intact regardless of native stripping.
The visible ``calculate`` tool supplies arithmetic support (183 raw calls), and
ordinary assistant explanations remain as released. Neither is native private
reasoning. ``strip_think_tool=True`` is explicitly unsupported.

Verified quirks and limits
--------------------------
The complete raw census has 92,311 turns: human 24,229; gpt 24,172;
function_call 21,955; observation 21,955. Every call is followed immediately by
one observation; the source policies explicitly allow only one call at a time.
There are no source linkage IDs or parallel batches. The 39 gpt-to-gpt edges
have no proven serialization-split explanation and remain separate messages.
All 4,904 gpt endings and 96 observation endings are inspected mechanically;
only a nonempty final assistant is an ordinary response-SFT boundary.

Declared tools are static; no callable discovery or asynchronous job protocol
occurs. Repeated calls, visible errors and recovery
are retained when otherwise valid. Unknown source/message/call fields are
rejected, rather than silently removed. Tool definitions are never simplified.

The airline clock is already visible in its system. Retail requires a successful
identity lookup before protected operations. Upstream ``find_user_id_*`` returns
a user ID, or exactly ``Error: user not found``; all released lookup results match
these forms. Opaque IDs cannot be obtained from schema examples, teacher hints,
prior assistant inventions or future results. The order_id schemas explicitly
require a leading '#', so a user-supplied W-number grounds the corresponding
#W-number without changing the retained call. Partial payment-card suffixes do
not determine full payment IDs. Raw rows include unsupported calls, recursively
invalid arguments, invented IDs and operations after failed authentication.
The gates below cover these deterministic defects; they do not certify arbitrary
natural-language confirmations, relative-time calculations, policy compliance,
or semantic task success. The missing generator state prevents replaying the
paper's original success checks. Such semantic uncertainty is not repaired by
scores or keyword deletion.

The source lookup implementations support case-insensitive email/name matching;
ZIP codes and opaque IDs remain exact. The result-evidence grammar admits exact
string values and explicitly recognized keyed payment/item mappings, not all
JSON keys or incidental prose substrings. Empty opaque IDs remain invalid.
Relevant upstream contracts are ``tau_bench/envs/retail/tools/`` in the tau-bench
repository: ``find_user_id_by_email.py``, ``find_user_id_by_name_zip.py`` and
``get_order_details.py``. These explain the released grammar but are not a pin
of the unavailable generation-time simulator.

Final-view rechecking covers only the audited policies, not arbitrary instructions
introduced by a custom system override. Passing it does not certify that removing
the original policy or clock leaves the task equally learnable. Native reasoning
and loss selection remain trainer responsibilities; retaining recovery context
does not require positive loss on every earlier failed assistant response.

Ordered pipeline
----------------
1. Load the pinned file/config/train split in source order; iterate Arrow rows
   without materializing whole columns. Strictly classify source fields, map
   human/gpt/function_call/observation to user/assistant/assistant-call/tool,
   serialize structured arguments only, and preserve the complete original raw
   row without normalized mutable aliases. Preserve source system text, including
   absence/emptiness, complete definitions and separate assistant messages.
2. Reconcile complete normalized definitions: retain first exact repeat, reject
   same-name conflicts. Validate structure and source-proven singleton positional
   call/result linkage, then static capabilities and full supported JSON Schema.
   Shared result alignment defaults on (FilterConfig.align_results); the pinned
   singleton batches need no moves under either setting.
3. Apply APIGen-MT's visible-input and retail-authentication policy: require opaque
   argument IDs in preceding user text or exact structured result values/keys;
   apply only the documented order-prefix comparison; validate identity-lookup
   inputs and successful authentication before protected operations.
4. Remove native reasoning representations when FilterConfig.strip_thinking
   is enabled (the loader default), using the shared stage, then apply the shared
   post-validation system override. Visible think tools, their calls/results,
   ordinary prose and raw remain unchanged.
5. Recheck final structure, linkage and capabilities/arguments, then reapply the
   visible-input and retail-authentication policy to the final model-visible
   context, including authenticated-user consistency. This catches requirements
   introduced by an override; the earlier source check prevents an override from
   rescuing a source-invalid trajectory. Finally require nonempty final assistant
   answer termination; recognized balanced native reasoning alone is not an
   answer in either reasoning mode. Native control tokens alone also provide no
   answer. Neither does an open-only native tail preceded solely by balanced
   native spans and whitespace. Code literals and other ambiguous boundaries
   retain their configured handling.
6. Apply configured shared Level 1/1.5/2 curation inside this train split
   (all three cumulative levels plus aggregate audit are enabled by default),
   selecting original rows without reordering content; requested Levels 3--5
   produce aggregate audit metadata only. No prefix-family collapse is performed.

The loader's omitted filter_config enables native stripping; an explicit
FilterConfig uses its own value (False by default). Mandatory production gates
cannot be bypassed by legacy require_* flags or visible-think/retry deletion
options. iter_load() uses a temporary disk spool and emits only after complete
curation; close it when abandoning iteration. load() materializes the population.
Pinned configuration/report contracts live in ``tests/fixtures/loaders/apigen_mt``;
the source-census and retained-contract checks are in
``tests/unit/test_loader_apigen_mt.py``. These tests establish their stated
mechanical/source contracts, not the paper's unavailable success judgments.
"""

from collections import Counter
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from functools import partial
import re
from typing import Any

from datasets import load_dataset

from ..format import ConversationSample
from .base import FilterConfig, LoadReport
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .context import collect_string_values, contains_literal_token
from .normalization import normalize_arguments, normalize_tools, parse_json
from .pipeline import (
    Pipeline,
    Reject,
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

DATASET_ID = "Salesforce/APIGen-MT-5k"
DATASET_REVISION = "abc4a517d67c541f85f6470cbd8fd3186b36830e"
DATA_FILE = "apigen-mt_5k.json"


@dataclass(slots=True)
class APIGenMTConfig:
    """Compatibility fields cannot disable the mandatory production contract.

    ``strip_think_tool`` is rejected: visible tool use is not native reasoning.
    The former repeat/adjacency deletion policies are likewise unsupported.
    Definition, linkage, argument and termination checks are always enforced;
    legacy FilterConfig.require_* fields cannot disable them.
    """

    strip_think_tool: bool = False
    drop_undefined_function_calls: bool = True
    drop_repeated_tool_call_streaks: bool = False
    drop_error_recovery_loops: bool = False
    drop_consecutive_assistant: bool = False
    drop_incomplete_termination: bool = True


def _parse_tools(tools_json: str) -> list[dict[str, Any]] | None:
    """Compatibility conversion helper; production reconciliation is a stage."""
    try:
        state = RowState(
            ConversationSample(
                messages=[],
                tools=_normalize_source_tools(tools_json),
                dataset=DATASET_ID,
                sample_id=0,
            )
        )
        reconcile_definitions(state)
        return state.sample.tools
    except Reject:
        return None


def _normalize_source_tools(value: str) -> list[dict[str, Any]]:
    tools = normalize_tools(value, bare=True)
    for tool in tools:
        function = tool["function"]
        if function.keys() - {"name", "description", "parameters", "strict"}:
            raise Reject("unknown_source_definition_fields")
        if "description" in function and not isinstance(function["description"], str):
            raise Reject("invalid_source_definition_description")
        if "strict" in function and not isinstance(function["strict"], bool):
            raise Reject("invalid_source_definition_strict")
    return tools


def _convert_conversations(conversations: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not isinstance(conversations, list):
        raise Reject("invalid_source_conversations")
    messages: list[dict[str, Any]] = []
    roles = {"human": "user", "gpt": "assistant", "observation": "tool"}
    for message in conversations:
        if not isinstance(message, dict) or set(message) != {"from", "value"}:
            raise Reject("unknown_source_message_fields")
        role, value = message["from"], message["value"]
        if not isinstance(value, str):
            raise Reject("invalid_source_content")
        if not isinstance(role, str):
            raise Reject("unknown_source_role")
        if role in roles:
            messages.append({"role": roles[role], "content": value})
        elif role == "function_call":
            try:
                call = parse_json(value)
            except ValueError as exc:
                raise Reject("invalid_source_call_json") from exc
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise Reject("unknown_source_call_fields")
            arguments = call["arguments"]
            if not isinstance(arguments, (str, dict)):
                raise Reject("invalid_source_arguments")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else normalize_arguments(arguments)[0],
                            },
                        }
                    ],
                }
            )
        else:
            raise Reject("unknown_source_role")
    return messages


def _convert_row(row: dict[str, Any], sample_id: int) -> ConversationSample:
    if (
        row.keys() - {"system", "tools", "conversations"}
        or not {"tools", "conversations"} <= row.keys()
    ):
        raise Reject("unknown_source_row_fields")
    system = row.get("system")
    if system is not None and not isinstance(system, str):
        raise Reject("invalid_source_system")
    if not isinstance(row["tools"], str):
        raise Reject("invalid_source_tools")
    tools = _normalize_source_tools(row["tools"])
    messages = _convert_conversations(row["conversations"])
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    return ConversationSample(
        messages=messages,
        tools=tools,
        dataset=DATASET_ID,
        sample_id=sample_id,
        raw=row,
    )


def _convert_sample(
    system: str | None,
    tools_json: str,
    conversations: list[dict[str, str]],
    sample_id: int,
) -> ConversationSample | None:
    """Convert/reconcile one source-shaped value, without curating a trajectory."""
    try:
        state = RowState(
            _convert_row(
                {"system": system, "tools": tools_json, "conversations": conversations},
                sample_id,
            )
        )
        reconcile_definitions(state)
        return state.sample
    except Reject:
        return None


# These fields have opaque simulator identity semantics. This is intentionally
# an enumerated source contract, not a universal name/keyword heuristic.
_OPAQUE_FIELDS = frozenset(
    {
        "user_id",
        "reservation_id",
        "order_id",
        "product_id",
        "item_ids",
        "new_item_ids",
        "payment_id",
        "payment_method_id",
        "flight_number",
    }
)
_AUTH_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
_RETAIL_AUTH_EXEMPT = _AUTH_TOOLS | {"think", "transfer_to_human_agents", "calculate"}
_USER_ID = re.compile(r"[a-z]+_[a-z]+_[0-9]+\Z")
_RETAIL_POLICY = "At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code."


def _argument_values(value: Any, field: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _argument_values(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _argument_values(item, field)
    elif isinstance(value, str):
        yield field, value


def _result_key_is_identifier(key: str) -> bool:
    return (
        re.fullmatch(
            r"(?:[0-9]+|(?:credit_card|gift_card|paypal|certificate)_[0-9]+)", key
        )
        is not None
    )


def validate_visible_inputs(state: RowState) -> None:
    """Prove source-specific opaque inputs and identity transitions in order.

    Results become evidence only after their linked call. Unstructured lookup
    user IDs use the fully enumerated released result grammar; arbitrary prose
    results and JSON substrings cannot grant facts by incidental mention.
    """
    evidence: set[str] = set()
    user_text: list[str] = []
    authenticated: str | None = None
    pending_auth: str | None = None
    retail = any(
        message["role"] == "system" and _RETAIL_POLICY in (message.get("content") or "")
        for message in state.sample.messages
    )
    for index, message in enumerate(state.sample.messages):
        if message["role"] == "user":
            user_text.append(message["content"])
        elif message["role"] == "tool":
            content = message["content"]
            if pending_auth:
                if content == "Error: user not found":
                    pass
                elif _USER_ID.fullmatch(content):
                    if authenticated is not None and authenticated != content:
                        raise Reject("conflicting_authenticated_user")
                    authenticated = content
                    evidence.add(content)
                else:
                    raise Reject("unrecognized_authentication_result")
                pending_auth = None
            try:
                value = parse_json(content)
            except Reject as exc:
                if exc.reason != "malformed_json":
                    raise
                continue
            collect_string_values(value, evidence, key_filter=_result_key_is_identifier)
        elif message["role"] == "assistant" and message.get("tool_calls"):
            name = message["tool_calls"][0]["function"]["name"]
            arguments = state.parsed_arguments[index, 0]
            for field, value in _argument_values(arguments):
                if (
                    retail
                    and field == "user_id"
                    and authenticated is not None
                    and value != authenticated
                ):
                    raise Reject("conflicting_authenticated_user")
                if field in _OPAQUE_FIELDS:
                    if not value:
                        raise Reject("ungrounded_identifier")
                    if value in evidence or contains_literal_token(value, user_text):
                        continue
                    # The released order_id definition explicitly directs the
                    # caller to add '#'; comparison does not rewrite the call.
                    if (
                        field == "order_id"
                        and re.fullmatch(r"#W[0-9]+", value)
                        and (
                            value[1:] in evidence
                            or contains_literal_token(value[1:], user_text)
                        )
                    ):
                        continue
                    raise Reject("ungrounded_identifier")
                if name in _AUTH_TOOLS and field in {
                    "email",
                    "first_name",
                    "last_name",
                    "zip",
                }:
                    if not contains_literal_token(
                        value, user_text, case_insensitive=field != "zip"
                    ):
                        raise Reject("ungrounded_authentication_input")
            if retail and name not in _RETAIL_AUTH_EXEMPT and authenticated is None:
                raise Reject("missing_visible_authentication")
            if name in _AUTH_TOOLS:
                pending_auth = name


def _pipeline(filter_config: FilterConfig) -> Pipeline:
    linkage = partial(
        link_calls,
        positional=True,
        parallel=False,
        align_results=filter_config.align_results,
    )
    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("structure", validate_structure),
        ("linkage", linkage),
        ("capabilities", validate_capabilities),
        ("visible_inputs", validate_visible_inputs),
    ]
    if filter_config.strip_thinking:
        stages.append(("reasoning", remove_reasoning))
    if filter_config.system_message_override is not None:
        stages.append(
            (
                "system_override",
                partial(
                    override_system, override=filter_config.system_message_override
                ),
            )
        )
    stages.extend(
        [
            ("final_structure", validate_structure),
            ("final_linkage", linkage),
            ("final_capabilities", validate_capabilities),
            ("final_visible_inputs", validate_visible_inputs),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(tuple(stages))


def iter_load(
    dataset_config: APIGenMTConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    """Return a disk-curated stream and counters finalized when exhausted.

    A caller stopping early must close the returned iterator. Independent
    source scopes may execute in parallel; partitions of this one train scope
    require one final shared curation pass rather than independent deletion.
    """
    config = dataset_config or APIGenMTConfig()
    if config.strip_think_tool:
        raise ValueError(
            "strip_think_tool removes visible behavior and is no longer supported; use FilterConfig.strip_thinking for native reasoning"
        )
    if (
        config.drop_repeated_tool_call_streaks
        or config.drop_error_recovery_loops
        or config.drop_consecutive_assistant
    ):
        raise ValueError(
            "APIGen-MT preserves valid repeated calls and consecutive assistant messages"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    filters = filter_config or FilterConfig(strip_thinking=True)
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    ds = load_dataset(
        DATASET_ID, name="dataset", revision=DATASET_REVISION, split="train"
    )
    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=len(ds),
        stage1_count=0,
        filter_config=filters,
        strip_thinking_applied=filters.strip_thinking,
    )
    pipeline = _pipeline(filters)

    def inputs() -> Iterator[CurationInput]:
        conversion_drops: Counter[str] = Counter()
        valid = 0
        rows = (
            row
            for batch in ds.data.table.to_reader(max_chunksize=batch_size)
            for row in batch.to_pylist()
        )
        for index, row in enumerate(rows):
            try:
                sample = _convert_row(row, index)
            except Reject as exc:
                conversion_drops[exc.reason] += 1
                continue
            report.stage1_count += 1
            state = pipeline.process(sample)
            if state is not None:
                valid += 1
                yield CurationInput(state.sample, state.batches, state.parsed_arguments)
        report.stage1_drop_reasons.update(conversion_drops)
        report.dataset_config_count = report.stage1_count
        report.filtered_count = valid
        report.filter_drop_reasons = dict(pipeline.drops)
        report.dataset_config_transform_counts.update(
            {
                "source": {
                    "revision": DATASET_REVISION,
                    "files": [DATA_FILE],
                    "config": "dataset",
                    "split": "train",
                    "subsets": {"dataset": report.raw_count},
                },
                "pipeline_passed": dict(pipeline.passed),
                "pipeline_drops": dict(pipeline.stage_drops),
                "transformations": dict(pipeline.transforms),
            }
        )

    result = curate(
        inputs(),
        scope=CurationScope(DATASET_ID, "dataset", "train"),
        config=curation_config,
    )

    def output() -> Generator[ConversationSample, None, None]:
        try:
            count = 0
            for sample in result:
                count += 1
                yield sample
            report.final_count = count
            report.dataset_config_transform_counts["curation"] = [
                counts.as_dict() for counts in result.reports.values()
            ]
        finally:
            result.close()

    return output(), report


def load(
    dataset_config: APIGenMTConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[list[ConversationSample], LoadReport]:
    iterator, report = iter_load(
        dataset_config,
        filter_config,
        curation_config=curation_config,
        batch_size=batch_size,
    )
    try:
        return list(iterator), report
    finally:
        iterator.close()
