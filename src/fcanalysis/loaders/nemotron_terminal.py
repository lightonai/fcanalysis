"""Nemotron-Terminal: faithful textual terminal interactions for SFT.

Origin, generation, subsets and license
--------------------------------------
NVIDIA, Pi et al., On Data Engineering for Scaling LLM Terminal Capabilities:
https://arxiv.org/abs/2602.21193 (v1). Corpus/card/upstream data repository:
https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus
Pinned revision: a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9. Its card declares
CC BY 4.0; inherited-source terms are separate and not comprehensively cleared.
The associated synthetic tasks are nvidia/Nemotron-Terminal-Synthetic-Tasks,
revision 7e53648e183cee7bcb0aed623cc91a121d37fa38. The inspected harness is
https://github.com/harbor-framework/harbor (Apache-2.0), dated comparison revisions
8c35bdb782ab741a8aa27a6a596dd8bcb5f8ad08 and
265c303150b08d0e4aab695dd96ff5f6b4159db9. These are NOT certified NVIDIA producer
pins. No complete reproducible Terminal-Task-Gen/export invocation was located.

The paper adapts Nemotron-Cascade math/code/SWE prompts and synthesizes terminal
skill tasks, inputs and tests, then collects DeepSeek-V3.2/Terminus-2 rollouts in
containers. Math inherits OpenMathReasoning; code inherits OpenCodeReasoning;
SWE inherits SWE-Bench-Train, SWE-reBench, SWE-Smith and SWE-Fixer. The paper's
seed-generated family is not in this release. Published filtering/verification
claims do not establish per-row success. Sampled task archives contain mutable
container/dependency references; exact historical replay is not certified.

The 29 files contain 366,154 physical rows: dataset_adapters 226,313 (31,960 code,
162,692 math, 31,661 SWE), skill_based_easy 44,809, skill_based_medium 89,343,
skill_based_mixed 5,689. Difficulty boundaries/configuration differences remain
unknown; these names are not reasoning-effort levels. Code has source labels
OpenCodeReasoning (31,950) or synthetic (10, origin unresolved). Other rows lack
per-example inherited-source labels. All use terminus-2, DeepSeek-V3.2 and
thinking-enabled metadata. Rows have conversations, agent, model, model_provider,
date, task, episode, run_id, trial_name, enable_thinking; code adds source.
The episode-N label counts assistant responses minus one. It does not establish
prefix donors; no cross-row consolidation/restoration is performed.

Model-visible interface, reasoning and trainer responsibilities
--------------------------------------------------------------
Initial USER text contains the full released JSON instructions, task and screen.
Assistants emit textual JSON analysis/plan/commands/task_complete. Feedback is
USER text containing screens, parser errors/warnings or completion confirmation.
These source roles and visible text remain exact; tools=[] means a textual
interface, NOT text-only QA. No synthetic native tool_calls or harness role is
created. The chat template renders these messages as ordinary chat content.
Ordered keystroke batches share a persistent tmux session and one observation.
Empty keystrokes poll; C-c/C-d are control inputs; duration is a wait (capped by
the inspected harness at 60), not proof the process finished. Feedback need not
contain every terminal byte: released screen/truncation text stays unchanged.

Exported leading <think>...</think> contains native reasoning. A unique exporter
closing boundary plus its one separator newline is decoded to reasoning_content;
all interior reasoning and remaining visible content stay exact. Ambiguous
boundaries are excluded. Opaque reasoning may contain unfinished code fences;
command strings may contain literal reasoning tags. strip_thinking is optional
and defaults to False; True removes only the decoded native field. It never
removes JSON analysis/plan or literal tags in visible commands. Not every
assistant has nonempty native reasoning despite enable_thinking=True.
There is no declared thinking tool in this release; analysis/plan are visible
reasoning mechanisms inside the textual protocol and remain in both modes.

Parser failures and recovery remain in accepted histories. A draft inside native
reasoning is never executed by the loader or promoted to a visible action.
Source-observed parser repairs/defaults are interpretation, not target rewrites.
Repeated analysis/plan keys can be retained as opaque visible prose because the
source parser keeps the last value and those keys do not select commands or
completion. All occurrences remain in content; Level 2 abstains. Duplicate
execution/completion fields are excluded. This source-text exception does not
relax shared strict JSON decoding of native function calls or definitions.
Trainers choose whether to mask rejected assistants while supervising subsequent
corrections; masking a target must not erase its explanatory historical context.
Trainers also own loss weights, packing, reasoning-history handling and rendering
qualification. Historical Harbor retains visible chat history separately from
exported reasoning; an archive is not proof that all old thinking was in each
teacher request. Model-specific history policies must be explicit.

A first finish declaration elicits confirmation; a second accepted declaration
ends the attempt. Retain this exchange. A parser rejection leaves the pending
confirmation intact; an accepted response with false/absent completion clears it.
There can be no later model interaction after an accepted second declaration.
The final consumed object must have explicit boolean true completion, commands=[],
string analysis/plan, no duplicate prose fields and a supported pending confirmation.
Source-supported surrounding text or code fences remain intact; final
auto-repairs/coerced flags are not accepted. Final commands whose observations
were omitted are excluded, never erased or given invented
results. Submission is not verified success or semantic correctness. Commands
can fail and still support later recovery; general semantic curation is separate.
The inspected parser can discard a malformed command batch when completion is
true and return a warning; the confirmation prompt can replace that warning in
visible feedback. Supported historical cases retain their original text without
attributing execution to the discarded commands. A malformed final batch fails
the strict empty-command submission rule, even if the parser drops its commands.

A historical native-only assistant can represent a failed generation followed
by the harness's parser rejection and a later correction. Stripping its native
field leaves empty visible content, so the final-view nonempty-assistant gate
rejects the entire conversation. This is a conservative acceptance policy, not
proof that its recovery context is worthless. Nothing is erased or promoted from
native reasoning to an action. Native-only final responses are already excluded
by the visible confirmed-submission requirement in either reasoning mode.

Known reset/handoff markers or incompatible contexts are excluded. No recoverable
compression/reset/participant-view family has been established in this release;
no such views are fabricated. Other terminal sources may expose valid separate
views, whose masking/selection belongs to their trainer under their source rules.

Ordered conversion, validation, transforms and curation
-------------------------------------------------------
1. Read selected configs/files in stable published order, in bounded Parquet
   batches. Validate source schema, roles, model/interface identity and episode
   shape. Copy message containers once, preserve raw unchanged, and use the
   existing sample_id field for the full source (run_id, trial_name) pair.
2. Decode only the proven native export envelope; retain visible text verbatim.
   Run shared canonical structure validation and the exact initial-interface gate.
3. Interpret each visible assistant through the inspected source parser. Require
   corresponding parser-error/warning feedback, allowing only verified unordered
   unknown-field warning names and the earlier CPython trailing-comma diagnostic,
   or the supported terminal screen/confirmation envelope. Reject contradictory
   evidence, ambiguous execution JSON, nonfinite/unsupported durations,
   known context replacements and continuation
   after accepted termination. Never execute commands or consult private tests.
4. Require a confirmed final empty-command submission. No fabricated endpoints,
   arbitrary trimming, hidden success labels or borrowing from other attempts.
5. Optionally remove only structured native reasoning using the shared stage,
   then apply system override. Recheck structure/interface and nonempty assistant
   targets in the final view. A nonempty added system override is excluded because
   its effects on the textual protocol are uncertified; empty override is a no-op.
   Parser/action checks are reusable because these accepted transforms cannot
   change visible assistant JSON, feedback, source roles or their chronology.
6. Curate separately within each selected config across its files using the shared
   disk-backed selector and a terminal comparison view. Level 1 is exact complete
   canonical content; Level 1.5 abstains from whitespace normalization throughout
   terminal protocol/screen text. Level 2 omits native reasoning and analysis/plan
   only for strict, complete, warning-free JSON responses; it preserves all other
   fields, ordered commands, explicit durations, feedback and completion. It abstains on repaired,
   rejected, warned or extra-text assistant responses, preserving the whole message
   including native reasoning for comparison. Shortest original assistant content
   wins (excluding native reasoning from that score), then stable source order.
   Levels 3-5 are aggregate audits only, with the source-specific views below.
   No mixture or benchmark decontamination is implied. Generic native-call analysis/mixture
   comparisons are not qualified for these textual actions.
7. Return original selected canonical samples and complete reports when the
   iterator is exhausted. Partial iteration has no final_count. load() materializes
   output; iter_load() and temporary SQLite curation bound memory by batch/row size.

Audit views, use and verification
---------------------------------
Level 3 uses the conservative Level 2 assistant projection but replaces the
initial task/screen with the exact fixed interface prefix; all subsequent
feedback remains. It is eligible only with interpreted commands. Level 4 records
ordered lists of send_keys action kinds, one per command, plus completion_request
when completion is declared; it preserves empty accepted batches and skips
parser-rejected generations. These are analysis labels, not invented tools.
Level 5 compares the exact fixed textual interface. No level permits command,
batch or observation permutation. None is a proof of equivalent tasks or success.

NemotronTerminalConfig(configs=["dataset_adapters"]) selects a nonempty set of
known configs, traversed in published order regardless of caller list order.
Omission selects all four. iter_load(path=...) can use an explicit local directory
containing the pinned files at their published relative paths; the caller owns
revision selection. Without path it uses pinned Hugging Face retrieval/cache.
The first output arrives only after complete curation; close partial iteration
to release the temporary store. Canonical annotations remain empty, raw remains
unchanged, and reports count conversations rather than trainer loss targets.
validated_protocol_issues counts occurrences in final-view-valid candidates before
curation; its completion_requests counts prompts eliciting confirmation and
excludes final confirming assistants. Decoding/removal counters can include later
exclusions and are neither retained coverage nor token counts.

The old strip_malformed/drop_orphans/drop_incomplete dataset flags are unsupported;
legacy require_* booleans cannot disable production gates. Native-call semantic
and overlap entry points reject this protocol explicitly. Generic mixture and
token-category code is not qualified for it, so direct source loading does not
authorize those analysis paths. Compatible chat rendering must preserve literal
commands and the actual source interface; harness-user feedback is context under
assistant-only loss. Repeated confirmation alone does not prove self-correction.

Pinned reports/configurations are in ``tests/fixtures/loaders/nemotron_terminal``.
``tests/e2e/test_nemotron_terminal_projection.py`` separately derives retained
canonical fields, compares runtime raw and replays fixtures, using the attributed
reference parser for protocol interpretation. It does not independently implement
every exclusion or curation decision. General command/answer correctness, exact
historical runtime and absence of every hidden reset remain uncertified.
"""

from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

from ..format import ConversationSample
from ._nemotron_terminal_protocol import (
    HANDOFF_PREFIX,
    PROTOCOL_PREFIX,
    RESET_NOTICE,
    Response,
    confirmation,
    error_feedback_matches,
    interpret,
    parser_feedback,
    screen,
    split_native,
    warning_feedback_matches,
)
from .base import FilterConfig, LoadReport
from .curation import (
    CurationConfig,
    CurationInput,
    CurationScope,
    canonical_comparison,
    curate,
)
from .normalization import Reject, json_bytes
from .pipeline import (
    Pipeline,
    RowState,
    Stage,
    override_system,
    remove_reasoning,
    validate_structure,
)
from .source import parquet_rows

DATASET_ID = "nvidia/Nemotron-Terminal-Corpus"
DATASET_REVISION = "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9"
_FILES: dict[str, list[str]] = {
    "dataset_adapters": [
        "dataset_adapters/code.parquet",
        "dataset_adapters/math.parquet",
        "dataset_adapters/swe.parquet",
    ],
    "skill_based_easy": [
        "synthetic_tasks/skill_based/easy/data_processing/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/data_querying/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/data_science/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/debugging/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/dependency_management/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/file_operations/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/scientific_computing/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/security/data_filtered.parquet",
        "synthetic_tasks/skill_based/easy/software_engineering/data_filtered.parquet",
    ],
    "skill_based_medium": [
        "synthetic_tasks/skill_based/medium/data_processing/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/data_querying/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/data_science/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/debugging/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/dependency_management/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/file_operations/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/scientific_computing/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/security/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/software_engineering/data_filtered.parquet",
        "synthetic_tasks/skill_based/medium/system_administration/data_filtered.parquet",
    ],
    "skill_based_mixed": [
        "synthetic_tasks/skill_based/mixed/data_processing/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/data_science/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/debugging/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/file_operations/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/scientific_computing/data_filtered.parquet",
        "synthetic_tasks/skill_based/mixed/security/data_filtered.parquet",
    ],
}

ALL_CONFIGS = list(_FILES.keys())


@dataclass(slots=True)
class NemotronTerminalConfig:
    configs: list[str] | None = None

    def __post_init__(self) -> None:
        if self.configs is not None and (
            not self.configs
            or any(c not in _FILES for c in self.configs)
            or len(set(self.configs)) != len(self.configs)
        ):
            raise ValueError("configs must select unique published config names")


@dataclass
class _TerminalState(RowState):
    responses: dict[int, Response] = field(default_factory=dict)
    issues: Counter[str] = field(default_factory=Counter)


@dataclass(slots=True)
class _TerminalInput(CurationInput):
    responses: dict[int, Response] = field(default_factory=dict)


def _convert_row(raw: Any, config: str, transforms: Counter[str]) -> ConversationSample:
    required = {
        "conversations",
        "agent",
        "model",
        "model_provider",
        "date",
        "task",
        "episode",
        "run_id",
        "trial_name",
        "enable_thinking",
    }
    if (
        not isinstance(raw, dict)
        or not required <= raw.keys()
        or raw.keys() - required - {"source"}
    ):
        raise Reject("unknown_source_row_shape")
    if (
        raw["agent"] != "terminus-2"
        or raw["model"] != "deepseek-ai/DeepSeek-V3.2"
        or raw["enable_thinking"] is not True
    ):
        raise Reject("unsupported_source_producer")
    if any(
        not isinstance(raw[k], str) or not raw[k]
        for k in required - {"conversations", "enable_thinking"}
    ):
        raise Reject("invalid_source_metadata")
    source = raw["conversations"]
    if not isinstance(source, list) or len(source) < 4 or len(source) % 2:
        raise Reject("unsupported_source_conversation")
    if raw["episode"] != f"episode-{len(source) // 2 - 1}":
        raise Reject("source_episode_mismatch")
    messages = []
    for index, original in enumerate(source):
        if not isinstance(original, dict):
            raise Reject("unknown_source_message_shape")
        original = cast(dict[str, Any], original)
        if (
            set(original) != {"role", "content"}
            or original["role"] != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(original["content"], str)
        ):
            raise Reject("unknown_source_message_shape")
        message = dict(original)
        if original["role"] == "assistant":
            reasoning, visible = split_native(original["content"])
            message["content"] = visible
            if reasoning is not None:
                message["reasoning_content"] = reasoning
                transforms["native_fields_decoded"] += 1
        messages.append(message)
    return ConversationSample(
        messages=messages,
        tools=[],
        dataset=f"{DATASET_ID}/{config}",
        sample_id=f"{raw['run_id']}/{raw['trial_name']}",
        raw=raw,
    )


def _interface(state: RowState) -> None:
    sample = state.sample
    if sample.tools or any(
        m["role"] not in {"user", "assistant"} for m in sample.messages
    ):
        raise Reject("unsupported_terminal_context_override")
    initial = sample.messages[0]["content"]
    if not initial.startswith(PROTOCOL_PREFIX):
        raise Reject("unsupported_terminal_interface")
    rest = initial[len(PROTOCOL_PREFIX) :]
    if "\n\nCurrent terminal state:\n" not in rest:
        raise Reject("missing_initial_terminal_context")
    task, terminal = rest.rsplit("\n\nCurrent terminal state:\n", 1)
    if not task.strip() or not screen(terminal):
        raise Reject("missing_initial_terminal_context")
    if initial != sample.raw["conversations"][0]["content"]:
        raise Reject("changed_terminal_initial_context")


def _transcript(base: RowState) -> None:
    state = cast(_TerminalState, base)
    messages = state.sample.messages
    pending = False
    for index in range(1, len(messages), 2):
        message = messages[index]
        response = interpret(message["content"])
        state.responses[index] = response
        result = response.result
        feedback = parser_feedback(result)
        rejected = "ERROR:" in feedback
        if rejected:
            state.issues["parser_rejected_assistants"] += 1
        if result.warning:
            state.issues["parser_warning_assistants"] += 1
        if "AUTO-CORRECTED:" in result.warning:
            state.issues["parser_autocorrected_assistants"] += 1
        state.issues["duplicate_prose_fields"] += response.duplicate_prose_fields
        state.issues["interpreted_commands"] += 0 if rejected else len(result.commands)
        if index == len(messages) - 1:
            if rejected:
                raise Reject("final_parser_rejection")
            if result.commands:
                raise Reject("final_commands_without_observation")
            if not pending:
                raise Reject("unconfirmed_terminal_ending")
            obj = response.consumed_object
            if (
                obj is None
                or obj.get("task_complete") is not True
                or obj.get("commands") != []
                or response.duplicate_prose_fields
                or "AUTO-CORRECTED:" in result.warning
                or not isinstance(obj.get("analysis"), str)
                or not isinstance(obj.get("plan"), str)
            ):
                raise Reject("unsupported_final_submission")
            return
        observation = messages[index + 1]["content"]
        if observation == RESET_NOTICE or observation.startswith(HANDOFF_PREFIX):
            raise Reject("unrecoverable_terminal_context_reset")
        if rejected:
            if not error_feedback_matches(response, observation):
                raise Reject("contradictory_parser_error_feedback")
            # The inspected harness leaves the pending state intact on rejection.
            continue
        if result.is_task_complete:
            if pending:
                raise Reject("continuation_after_terminal_submission")
            if not confirmation(observation):
                raise Reject("missing_terminal_confirmation")
            pending = True
            state.issues["completion_requests"] += 1
        else:
            pending = False
            valid = (
                warning_feedback_matches(result, observation)
                if result.warning
                else screen(observation)
            )
            if not valid:
                raise Reject("contradictory_terminal_feedback")
    raise Reject("incomplete_terminal_conversation")


def _final_view(state: RowState) -> None:
    _interface(state)
    for message in state.sample.messages:
        if message["role"] == "assistant" and not (
            message["content"].strip() or message.get("reasoning_content", "").strip()
        ):
            raise Reject("empty_assistant_after_transform")


def _pipeline(filters: FilterConfig) -> Pipeline:
    stages: list[tuple[str, Stage]] = [
        ("structure", validate_structure),
        ("interface", _interface),
        ("terminal_protocol", _transcript),
    ]
    if filters.strip_thinking:
        stages.append(("reasoning", partial(remove_reasoning, inline=False)))
    if filters.system_message_override is not None:
        stages.append(
            (
                "system_override",
                partial(override_system, override=filters.system_message_override),
            )
        )
    stages.extend(
        [("final_structure", validate_structure), ("final_view", _final_view)]
    )
    return Pipeline(stages, state_factory=_TerminalState)


class _TerminalComparison:
    """Source-specific projections; full original samples stay in the spool."""

    def __init__(self, record: CurationInput):
        item = cast(_TerminalInput, record)
        self.record = item
        self.exact = canonical_comparison(record)
        self.cache: dict[str, bytes] = {}

    def assistant_characters(self) -> int:
        return self.exact.assistant_characters()

    def has_calls(self) -> bool:
        return any(
            r.result.commands and "ERROR:" not in parser_feedback(r.result)
            for r in self.record.responses.values()
        )

    def key(self, level: str) -> bytes:
        if level in self.cache:
            return self.cache[level]
        if level in {"level_1", "level_1_5"}:
            value = self.exact.key("level_1")
        elif level == "level_5":
            value = json_bytes(PROTOCOL_PREFIX)
        elif level == "level_4":
            trace = []
            for response in self.record.responses.values():
                result = response.result
                if "ERROR:" in parser_feedback(result):
                    continue
                events = ["send_keys"] * len(result.commands)
                if result.is_task_complete:
                    events.append("completion_request")
                trace.append(events)
            value = json_bytes(trace)
        else:
            projected = []
            for index, message in enumerate(self.record.sample.messages):
                if index == 0 and level == "level_3":
                    projected.append({"role": "user", "content": PROTOCOL_PREFIX})
                    continue
                if message["role"] != "assistant":
                    projected.append(message)
                    continue
                response = self.record.responses[index]
                obj = response.original_object
                if obj is None or response.result.error or response.result.warning:
                    # Repairs/rejections and extra text do not provide an exact
                    # prose/action separation. Abstain rather than erase actions.
                    projected.append(message)
                else:
                    projected.append(
                        {
                            "role": "assistant",
                            "terminal_json": {
                                k: v
                                for k, v in obj.items()
                                if k not in {"analysis", "plan"}
                            },
                        }
                    )
            value = json_bytes(projected)
        self.cache[level] = value
        return value


def iter_load(
    dataset_config: NemotronTerminalConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    path: str | Path | None = None,
    batch_size: int = 64,
    curation_config: CurationConfig | None = None,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    """Stream selected configs; local path must contain the exact pinned files.

    Source gates always run. Legacy deletion/repair flags are unsupported.
    Close an abandoned iterator to release temporary files. Report final_count
    is populated only after complete exhaustion; no background worker is used.
    """
    config = dataset_config or NemotronTerminalConfig()
    filters = filter_config or FilterConfig()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    selected = config.configs or ALL_CONFIGS
    # Config selection never changes the stable published traversal order.
    selected = [name for name in ALL_CONFIGS if name in selected]
    report = LoadReport(
        DATASET_ID,
        0,
        0,
        filter_config=filters,
        strip_thinking_applied=filters.strip_thinking,
    )
    pipelines = {name: _pipeline(filters) for name in selected}
    counts: dict[str, Counter[str]] = {name: Counter() for name in selected}
    drops: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    transforms: Counter[str] = Counter()
    file_counts: dict[str, int] = {}

    def candidates() -> Generator[CurationInput, None, None]:
        for name in selected:
            for filename in _FILES[name]:
                file_counts[filename] = 0
                for raw in parquet_rows(
                    DATASET_ID,
                    DATASET_REVISION,
                    [filename],
                    batch_size=batch_size,
                    local_root=path,
                ):
                    report.raw_count += 1
                    counts[name]["raw"] += 1
                    file_counts[filename] += 1
                    try:
                        sample = _convert_row(raw, name, transforms)
                    except Reject as exc:
                        drops[exc.reason] += 1
                        counts[name]["conversion_dropped"] += 1
                        continue
                    report.stage1_count += 1
                    counts[name]["converted"] += 1
                    state = pipelines[name].process(sample)
                    if state is not None:
                        terminal = cast(_TerminalState, state)
                        issues.update(terminal.issues)
                        counts[name]["validated"] += 1
                        yield _TerminalInput(
                            sample=state.sample, responses=terminal.responses
                        )

    def output() -> Generator[ConversationSample, None, None]:
        source = candidates()
        result = curate(
            source,
            scope=lambda r: CurationScope(
                r.sample.dataset, r.sample.dataset.rsplit("/", 1)[-1], "train"
            ),
            config=curation_config,
            comparison_factory=_TerminalComparison,
            compress_comparisons=True,
            compress_payloads=True,
        )
        try:
            for sample in result:
                counts[sample.dataset.rsplit("/", 1)[-1]]["retained"] += 1
                yield sample
        finally:
            result.close()
            source.close()
        report.stage1_drop_reasons = dict(drops)
        report.dataset_config_count = report.stage1_count
        report.filtered_count = sum(p.passed["final_view"] for p in pipelines.values())
        report.filter_drop_reasons = dict(
            sum((p.drops for p in pipelines.values()), Counter())
        )
        report.final_count = sum(c["retained"] for c in counts.values())
        report.dataset_config_transform_counts = {
            "validated_protocol_issues": dict(issues),
            "source": {
                "revision": DATASET_REVISION,
                "files": file_counts,
                "subsets": {n: dict(c) for n, c in counts.items()},
            },
            "pipeline_by_config": {
                n: {
                    "passed": dict(p.passed),
                    "drops": dict(p.drops),
                    "stage_drops": dict(p.stage_drops),
                }
                for n, p in pipelines.items()
            },
            "transformations": dict(
                transforms + sum((p.transforms for p in pipelines.values()), Counter())
            ),
            "curation": [r.as_dict() for r in result.reports.values()],
        }

    return output(), report


def load(
    dataset_config: NemotronTerminalConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    path: str | Path | None = None,
    batch_size: int = 64,
    curation_config: CurationConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    """Materializing convenience API; use iter_load for full-corpus processing."""
    rows, report = iter_load(
        dataset_config,
        filter_config,
        path=path,
        batch_size=batch_size,
        curation_config=curation_config,
    )
    try:
        return list(rows), report
    finally:
        rows.close()
