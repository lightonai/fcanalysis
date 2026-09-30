"""Dolci Tool Use: source-preserving complete-response SFT adapter.

Origin and authoritative sources
--------------------------------
Allen AI's OLMo 3 tool-use release, ``allenai/Dolci-Instruct-SFT-Tool-Use``,
revision ``dc042846f0f2de0f15eedae3d6ced04223ed47eb``. Dataset card:
https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use/blob/dc042846f0f2de0f15eedae3d6ced04223ed47eb/README.md
Parent: https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT
Paper: https://arxiv.org/html/2512.13961v2 (OLMo 3, sections 5.2.1/A.7.2).
Upstream training repository: https://github.com/allenai/open-instruct;
research-agent source: https://github.com/rlresearch/dr-tulu. Neither is a
pinned, complete SimFC generation recipe for these exact released rows.

Declared license
----------------
The pinned tool-use card declares ODC-BY, without specifying a version. It
also states intended research and educational use under Ai2's Responsible Use
Guidelines: https://allenai.org/responsible-use. This is the released dataset's
declaration, separate from licenses of training or tool implementation code.
The card and raw rows provide no per-record license allocation for inherited
sources; no additional licensing conclusion is inferred from schema origin.

Generation, subsets and schema
------------------------------
The six ``data/train-0000{0..5}-of-00006.parquet`` shards contain 227,579
rows (five times 37,930 plus 37,929), one default config and train split.
The exact three row fields are id, dataset_source, messages. Each message has
role/content/function_calls/functions; roles are system/user/assistant/
environment, content is string or null. All 227,579 rows have exactly one
leading system carrying JSON-serialized function definitions in ``functions``.
Calls use newline-separated Python-style keyword calls; JSON true/false/null
literals also occur inside values. Source ``function_calls='null'`` is an
absent-call sentinel. There are no source linkage IDs.

Five dataset_source partitions: SimFC
``allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated`` (200,000),
science ``allenai/olmo-toolu-s2-sft-m3`` (8,074), ``...-m4v2`` (9,085),
``...-m5v2`` (5,417), and
``allenai/olmo-toolu_deepresearch_no_thinking_DRv4`` (5,003). SimFC uses
GPT-4o/4.1/5 to synthesize users, calls, simulated results and answers from
xLAM/ToolACE/MCP tool schemas; schema inheritance does not establish transcript
reuse. Science uses GPT-4.1-mini and real ASTA/ASC tools. Search derives from
DR Tulu and includes benchmark/user-query sources; producer webpage summaries
are already the released result content and are preserved as such. The
paper's 6.6K search count differs from this pinned release's 5,003 records.

Native and visible reasoning
----------------------------
The complete raw census finds no separate reasoning/analysis field in any
subset. Despite its no_thinking label, search contains 40 assistant contents
with closed ``<think>...</think>`` spans and one with an unmatched closing tag.
SimFC and all three science partitions contain no native think spans. No
``<reasoning>`` tags occur anywhere in the pinned data. Optional
``FilterConfig.strip_thinking`` (loader default true) removes the 40 safely
bounded spans and quarantines the ambiguous trajectory; later validation rejects
resulting empty targets. Setting it false preserves native reasoning unchanged.
Literal think tags in one search user message and one environment message
are preserved. Seven SimFC assistant messages contain ``<analysis>`` XML
elements reporting password-analysis results; these are ordinary visible
answers, not the native reasoning delimiter grammar.

SimFC also exposes reasoning as ordinary callable tools. Parsed raw calls include
``think`` (3), ``sequentialthinking`` (4) and ``sequential_thinking`` (2),
with thoughts in normal arguments and results. Declared, uncalled reasoning
tools include ``chroma_sequential_thinking``, ``gemini_reason``,
``sequentialthinking_tools`` and ``reasoning``. Their supplied definitions
differ: they describe thought logging, iterative or branching reasoning,
tool recommendations, or delegation to a reasoning model. Related declared
operations include ``chroma_get_similar_sessions``, ``chroma_get_thought_history``,
``chroma_get_thought_branches``, ``chroma_continue_thought_chain``,
``get_thinking_summary``, ``clear_thinking_history``, ``evaluateInsight``,
``rollup``, ``get_thoughts``, ``clear_thoughts`` and ``get_thought_stats``.
Another source tool family creates and edits reasoning chains through
``create_thinking_chain``, ``add_thinking_step`` and ``validate_step``, with
``get_chain``, ``generate_visualization``, ``save_to_memory``,
``load_from_memory``, ``search_related_thinking`` and ``apply_template``.
Declaration does not imply use, and raw call counts do not imply retention.

Native reasoning removal never deletes these tool definitions, calls, their
``thought``/``reasoning`` arguments, revision or branch state, history/results,
or ordinary explanatory assistant prose. Such visible content survives in
otherwise retained trajectories; an inspected thought-log/history/clear/stats
trajectory is retained with stripping enabled. This distinction certifies
preservation, not the semantic validity of every reasoning-tool trajectory.
The exact source definitions remain authoritative; no current upstream MCP
implementation is assumed to be the generator revision for these rows.

Verified quirks and limits
--------------------------
Full raw census: 2,571,586 messages, including 626,290 null contents; all tools
are complete function envelopes. Tool schemas include malformed values, local
references without definitions and unfamiliar keywords: these are quarantined,
never repaired by dropping constraints. Calls contain malformed names, invalid
Python, positional values, duplicate keywords, tuple/set values, expressions
and user-role calls. Only bounded literal/container/unary/arithmetic evaluation
is accepted; no names are resolved and no missing arguments are inferred.
Integers beyond 64 bits remain exact within the evaluator's explicit resource
bounds; serializer limits do not turn them into malformed calls or floats.

Parallel calls are one source assistant batch (explicitly described by the
paper). Results use source order: either one environment message per call or,
in SimFC, newline-delimited complete JSON values in one environment message.
Only an entirely consumed newline-delimited sequence with exactly one value
per call is split. A single JSON array is one result, and arbitrary lines or
JSON-looking fragments with surrounding prose never prove linkage. No
independent IDs exist to detect an upstream swap of two anonymous results.
The raw protocol plus inspected examples support positional pairing; exact
producer implementation and semantic fidelity of simulated results remain
uncertain. All original result bytes except audited inter-result separators
survive conversion. Definitions are never inferred from system references to
XML tags. Later definition epochs are unsupported and quarantined.

Consecutive assistants are retained: no producer rule proves they are split
serializations. No prefix-family collapse, heuristic retry removal, dynamic
capability grants, or async job grammar is assumed. Audited SimFC credential
contracts require an exact value in earlier user/system text or linked JSON
result values. Definition examples and future/assistant assertions do not
establish credentials. The exact Bolivia Songs chart contract likewise requires
a supplied date in earlier visible context: literal ISO dates, unambiguous
English calendar dates or linked JSON result values qualify; relative words,
schema defaults and the host clock do not. Other credentials, dates,
authorization and answer claims remain semantically uncertain. Exact audited
Instagram and temporary-upload contracts require existing user/video/account
IDs in prior user/system tokens or linked JSON values. An earlier literal's
presence does not establish its intended entity or clock role; acceptance is
not a correctness label. See ``dolci_context.py`` for the bounded contract set.

The credential gate distinguishes existing secrets from password creation,
password analysis, cryptocurrency tokens and pagination. It conservatively
rejects a requested wrong-password test if its exact test string is not supplied;
this is an evidence limitation, not proof that the negative test is invalid.
Schema-valid null/empty credential values can state absence without asserting
an invented secret. Literal checks do not recursively decode JSON hidden inside
result strings or accept JSON keys as credential evidence. The date gate can
likewise exclude unspecified-year, arbitrary-date or paraphrased requests;
an earlier unrelated date can satisfy its literal check without proving today.
No general clock, consent or entity-resolution certification is implied.

Normalization is structural: retained prose keeps capitalization, punctuation,
whitespace and language. Bundle splitting preserves the first result's leading
whitespace and the last result's trailing whitespace; only the proven separators
between complete values are removed. A correctly linked error result can remain
as context for later recovery. Trainers choose loss targets, masks and weights;
preserving a conversation does not endorse every assistant response as a positive
training target. Canonical fidelity does not certify arbitrary model templates.

Exact ordered pipeline
----------------------
1. Iterate all pinned shards in filename/row order in bounded Arrow batches;
   preserve the complete original row in raw. Classify every source field.
2. Convert roles, parse complete function envelopes and bounded Python calls;
   preserve null/empty/nonempty content and chronology. Split only the audited
   complete bundled-result grammar. Unknown source structure is quarantined.
3. Shared exact definition reconciliation retains first repeats and rejects
   same-name body collisions; no fields/defaults are added to definitions.
4. Shared canonical structure/argument parsing, source-positional call/result
   binding and full supported JSON Schema/capability validation. Shared result
   alignment defaults on (FilterConfig.align_results); positional batches
   already match call order. Unsupported schema constraints have distinct reasons
   from invalid argument values.
5. Source policy checks the audited SimFC credential, chart-date and existing-ID
   contracts against the visible prefix, quarantining unresolved values.
   Consecutive assistants and complete original rows are preserved;
   unsupported legacy merge/drop requests fail explicitly.
6. Optionally remove only safely bounded native reasoning (default enabled),
   then apply the caller's system override only after source validation.
7. Revalidate the exact final structure, pairing, capabilities and source
   context (including any removed source system evidence), then require
   a nonempty final non-call assistant and no empty assistant targets. A final
   response containing only recognized balanced native reasoning is incomplete
   even when that reasoning is retained. Native control tokens alone also provide no answer. Neither does an
   open-only native tail preceded solely by balanced native spans and whitespace.
   Code literals and other ambiguous-boundary policies stay unchanged.
8. Cumulative shared Levels 1/1.5/2 curation (default through Level 2), scoped
   separately to each dataset_source within this revision's train split.
   Level 2 chooses fewest assistant-content characters with stable source-order
   ties; retained rows are unchanged. Parallel comparisons bind atomic units.
   Levels 3--5 are aggregate audit metadata only. No cross-source deletion.

``FilterConfig`` retains its legacy shape, but migrated structural checks are
mandatory; its require_* flags cannot turn invalid rows into production data.
strip_thinking/system_message_override control content and system transforms;
align_results controls proven result alignment, which this pin does not need.
``iter_load`` is bounded before curation and uses a temporary disk spool for
selection; ``load`` explicitly materializes the returned population.
The first retained row is available only after complete curation. Close an
abandoned iterator to release its temporary files. Reports distinguish exclusive
first-failure counts from overlapping source properties; transformations can
include candidates excluded later. Tests and pinned reports are under
``tests/e2e/test_dolci_census.py`` and ``tests/fixtures/loaders/dolci``.
"""

import ast
from collections import Counter
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from functools import partial
import json
import math
from typing import Any

from ..format import ConversationSample
from .base import FilterConfig, LoadReport
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .dolci_context import validate_simfc_credentials
from .normalization import Reject, json_bytes, normalize_tools, parse_json
from .pipeline import (
    Pipeline,
    Stage,
    link_calls,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_structure,
    validate_termination,
)
from .source import parquet_rows

DATASET_ID = "allenai/Dolci-Instruct-SFT-Tool-Use"
DATASET_REVISION = "dc042846f0f2de0f15eedae3d6ced04223ed47eb"
DATA_FILES = tuple(f"data/train-{i:05d}-of-00006.parquet" for i in range(6))


@dataclass(slots=True)
class DolciConfig:
    # Legacy unsafe transformations are explicit errors, not silent no-ops.
    drop_consecutive_text_text_assistant: bool = False
    merge_text_fc_assistant: bool = False
    drop_conflicting_duplicate_tools: bool = True


def _is_real_fc(value: str | None) -> bool:
    return value is not None and value.strip() not in ("", "null")


# Limits apply to every intermediate value and to cumulative expansion across
# the whole source call batch, including repeated references in nested lists.
_MAX_INTEGER_BITS = 4096
_MAX_SEQUENCE_LENGTH = 100_000
_MAX_EXPANDED_UNITS = 1_000_000
_MAX_EVALUATION_UNITS = 8_000_000


@dataclass(slots=True)
class _EvaluationBudget:
    remaining: int = _MAX_EVALUATION_UNITS

    def spend(self, units: int) -> None:
        if units > _MAX_EXPANDED_UNITS or units > self.remaining:
            raise ValueError("function argument expansion exceeds resource bounds")
        self.remaining -= units


def _expanded_bound(units: int) -> None:
    if units > _MAX_EXPANDED_UNITS:
        raise ValueError("function argument expansion exceeds resource bounds")


def _bounded_scalar(value: Any, budget: _EvaluationBudget) -> tuple[Any, int]:
    if type(value) is int:
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise ValueError("function argument integer exceeds resource bounds")
        units = max(1, value.bit_length())
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite function argument arithmetic")
        units = 1
    elif isinstance(value, str):
        units = len(value) + 1
    elif value is None or type(value) is bool:
        units = 1
    else:
        raise ValueError("unsupported function argument literal")
    budget.spend(units)
    return value, units


def _eval_bounded(node: ast.expr, budget: _EvaluationBudget) -> tuple[Any, int]:
    """Return the value and its fully expanded size without following aliases."""
    match node:
        case ast.Constant(value=value):
            return _bounded_scalar(value, budget)
        case ast.Name(id="true"):
            return _bounded_scalar(True, budget)
        case ast.Name(id="false"):
            return _bounded_scalar(False, budget)
        case ast.Name(id="null"):
            return _bounded_scalar(None, budget)
        case ast.List(elts=elements):
            result = []
            units = 1
            for element in elements:
                value, size = _eval_bounded(element, budget)
                units += size + 1
                _expanded_bound(units)
                result.append(value)
            budget.spend(units)
            return result, units
        case ast.Dict(keys=keys, values=values):
            result = {}
            units = 1
            for key_node, value_node in zip(keys, values, strict=True):
                if key_node is None:
                    raise ValueError
                key, key_size = _eval_bounded(key_node, budget)
                if not isinstance(key, str) or key in result:
                    raise ValueError
                value, value_size = _eval_bounded(value_node, budget)
                units += key_size + value_size + 1
                _expanded_bound(units)
                result[key] = value
            budget.spend(units)
            return result, units
        case ast.UnaryOp(op=operation, operand=operand):
            value, _ = _eval_bounded(operand, budget)
            if type(value) not in (int, float):
                raise ValueError
            if isinstance(operation, ast.USub):
                return _bounded_scalar(-value, budget)
            if isinstance(operation, ast.UAdd):
                return _bounded_scalar(value, budget)
            raise ValueError
        case ast.BinOp(left=left, op=operation, right=right):
            lval, left_units = _eval_bounded(left, budget)
            rval, right_units = _eval_bounded(right, budget)
            if isinstance(operation, ast.Mult):
                if isinstance(lval, list) and isinstance(rval, int):
                    sequence, sequence_units, multiplier = lval, left_units, rval
                elif isinstance(lval, int) and isinstance(rval, list):
                    sequence, sequence_units, multiplier = rval, right_units, lval
                else:
                    raise ValueError
                # Cost includes every expanded child, not just outer length.
                if len(sequence) * max(0, multiplier) > _MAX_SEQUENCE_LENGTH:
                    raise ValueError(
                        "function argument sequence exceeds resource bounds"
                    )
                units = 1 + (sequence_units - 1) * max(0, multiplier)
                budget.spend(units)
                return sequence * multiplier, units
            if (
                isinstance(operation, ast.Add)
                and isinstance(lval, list)
                and isinstance(rval, list)
            ):
                if len(lval) + len(rval) > _MAX_SEQUENCE_LENGTH:
                    raise ValueError(
                        "function argument sequence exceeds resource bounds"
                    )
                units = left_units + right_units - 1
                budget.spend(units)
                return lval + rval, units
            if type(lval) not in (int, float) or type(rval) not in (int, float):
                raise ValueError
            if isinstance(operation, ast.Add):
                value = lval + rval
            elif isinstance(operation, ast.Sub):
                value = lval - rval
            elif isinstance(operation, ast.Div):
                value = lval / rval
            elif isinstance(operation, ast.FloorDiv):
                value = lval // rval
            elif isinstance(operation, ast.Pow):
                if abs(rval) > 100:
                    raise ValueError(
                        "function argument exponent exceeds resource bounds"
                    )
                # Check the integer upper bound before exponentiation allocates
                # its result. Bounding the exponent alone does not bound nested
                # powers, even when the entire input expression is tiny.
                if (
                    type(lval) is int
                    and type(rval) is int
                    and rval >= 0
                    and abs(lval) > 1
                ):
                    if lval.bit_length() * rval > _MAX_INTEGER_BITS:
                        raise ValueError(
                            "function argument power exceeds resource bounds"
                        )
                value = lval**rval
            else:
                raise ValueError
            return _bounded_scalar(value, budget)
        case _:
            raise ValueError


def _safe_eval_node(node: ast.expr) -> Any:
    """Evaluate the audited literal/arithmetic grammar with bounded resources."""
    return _eval_bounded(node, _EvaluationBudget())[0]


def _extract_dotted_name(node: ast.expr) -> str | None:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=value, attr=attr):
            prefix = _extract_dotted_name(value)
            return None if prefix is None else f"{prefix}.{attr}"
        case _:
            return None


def _parse_single_call(
    node: ast.expr, budget: _EvaluationBudget | None = None
) -> dict[str, Any] | None:
    budget = budget or _EvaluationBudget()
    match node:
        case ast.Call(func=func_node, args=pos_args, keywords=keywords):
            name = _extract_dotted_name(func_node)
            if name is None or pos_args:
                return None
            kwargs: dict[str, Any] = {}
            argument_units = 1
            for kw in keywords:
                if kw.arg is None or kw.arg in kwargs:
                    return None
                try:
                    value, size = _eval_bounded(kw.value, budget)
                    argument_units += len(kw.arg) + size + 1
                    _expanded_bound(argument_units)
                    kwargs[kw.arg] = value
                except ValueError, ArithmeticError:
                    return None
            try:
                budget.spend(argument_units)
                arguments = json_bytes(kwargs).decode()
            except ValueError:
                return None
            return {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        case _:
            return None


def _parse_function_calls(fc_str: str) -> list[dict[str, Any]] | None:
    if len(fc_str) > 1_000_000:
        return None
    try:
        tree = ast.parse(fc_str, mode="exec")
    except SyntaxError, RecursionError, MemoryError:
        return None
    if sum(1 for _ in ast.walk(tree)) > 100_000:
        return None
    if not tree.body:
        return None
    if len({stmt.lineno for stmt in tree.body}) != len(tree.body):
        # Semicolon-separated Python statements do not prove a source parallel
        # newline batch and must not gain permutation equivalence.
        return None

    tool_calls: list[dict[str, Any]] = []
    budget = _EvaluationBudget()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            return None
        result = _parse_single_call(stmt.value, budget)
        if result is None:
            return None
        tool_calls.append(result)
    return tool_calls if tool_calls else None


def _split_bundled_tool_responses(messages: list[dict[str, Any]]) -> int:
    """Split exact newline-separated JSON values; never scan through prose."""
    splits = 0
    i = 0
    decoder = json.JSONDecoder()
    while i < len(messages):
        n = len(messages[i].get("tool_calls", []))
        if n > 1 and i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
            end = i + 2
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
            if end == i + 2:
                content = messages[i + 1].get("content")
                if not isinstance(content, str):
                    raise Reject("ambiguous_bundled_results")
                parts = []
                cursor = 0
                try:
                    while cursor < len(content):
                        start = cursor
                        while start < len(content) and content[start] in " \t\r\n":
                            start += 1
                        if start == len(content):
                            if parts:
                                parts[-1] += content[cursor:]
                            break
                        if parts and "\n" not in content[cursor:start]:
                            raise Reject("ambiguous_bundled_results")
                        _, stop = decoder.raw_decode(content, start)
                        # Validate JSON duplicate keys/nonfinite numbers too.
                        parse_json(content[start:stop])
                        parts.append(content[0 if not parts else start : stop])
                        cursor = stop
                except (ValueError, RecursionError) as exc:
                    raise Reject("ambiguous_bundled_results") from exc
                if len(parts) != n:
                    raise Reject("ambiguous_bundled_results")
                messages[i + 1 : end] = [
                    {"role": "tool", "content": part} for part in parts
                ]
                splits += 1
        i += 1
    return splits


def _convert_sample(
    sample_id: str,
    raw_messages: list[dict[str, Any]],
    dataset_source: str,
) -> tuple[ConversationSample | None, str | None]:
    """Compatibility conversion entry point; load preserves the full row."""
    try:
        return _convert_row(
            {
                "id": sample_id,
                "messages": raw_messages,
                "dataset_source": dataset_source,
            }
        ), None
    except Reject as exc:
        return None, exc.reason


def _convert_row(
    raw: dict[str, Any], transforms: Counter[str] | None = None
) -> ConversationSample:
    if raw.keys() != {"id", "messages", "dataset_source"}:
        raise Reject("unknown_source_field")
    if not isinstance(raw["messages"], list):
        raise Reject("malformed_source_messages")
    tools = []
    messages = []
    raw_messages: list[Any] = raw["messages"]
    for index, source in enumerate(raw_messages):
        if not isinstance(source, dict) or source.keys() - {
            "role",
            "content",
            "function_calls",
            "functions",
        }:
            raise Reject("unknown_source_message_field")
        role = source.get("role")
        if role not in ("system", "user", "assistant", "environment"):
            raise Reject("unknown_role")
        definitions = source.get("functions")
        if definitions is not None and definitions not in ("", "null"):
            if index != 0 or role != "system":
                raise Reject("unsupported_tool_definition_epoch")
            tools = normalize_tools(definitions)
            for tool in tools:
                if tool.keys() - {"type", "function"} or tool["function"].keys() - {
                    "name",
                    "description",
                    "parameters",
                    "strict",
                }:
                    raise Reject("unknown_source_tool_field")
        if not isinstance(source.get("content"), (str, type(None))):
            raise Reject("unsupported_content_shape")
        message = {
            "role": "tool" if role == "environment" else role,
            "content": source.get("content"),
        }
        calls = source.get("function_calls")
        if calls is not None and not isinstance(calls, str):
            raise Reject("malformed_function_calls")
        if _is_real_fc(calls):
            if role != "assistant":
                raise Reject("function_calls_on_" + str(role))
            try:
                parsed = _parse_function_calls(calls)
            except ValueError, RecursionError, ArithmeticError:
                parsed = None
            if parsed is None:
                raise Reject("malformed_function_calls")
            message["tool_calls"] = parsed
        messages.append(message)
    # Mutable messages/tools were constructed/decoded independently of raw.
    splits = _split_bundled_tool_responses(messages)
    if splits and transforms is not None:
        transforms["bundled_result_batches_split"] += splits
    return ConversationSample(messages, tools, DATASET_ID, raw["id"], raw=raw)


def _pipeline(config: FilterConfig) -> Pipeline:
    linkage = partial(
        link_calls, positional=True, parallel=True, align_results=config.align_results
    )
    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("source_structure", validate_structure),
        ("source_linkage", linkage),
        ("source_capabilities", validate_capabilities),
        ("source_context", validate_simfc_credentials),
    ]
    if config.strip_thinking:
        stages.append(("reasoning", remove_reasoning))
    stages.extend(
        [
            (
                "system_override",
                partial(override_system, override=config.system_message_override),
            ),
            ("final_structure", validate_structure),
            ("final_linkage", linkage),
            ("final_capabilities", validate_capabilities),
            ("final_context", validate_simfc_credentials),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def iter_load(
    dataset_config: DolciConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    config = dataset_config or DolciConfig()
    if config.drop_consecutive_text_text_assistant or config.merge_text_fc_assistant:
        raise ValueError(
            "Dolci has no audited consecutive-assistant deletion/merge rule"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    filters = filter_config or FilterConfig(strip_thinking=True)
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    pipeline = _pipeline(filters)
    report = LoadReport(DATASET_ID, 0, 0, filter_config=filters)
    source_counts: Counter[str] = Counter()
    conversion_transforms: Counter[str] = Counter()

    def converted() -> Iterator[CurationInput]:
        for raw in parquet_rows(
            DATASET_ID, DATASET_REVISION, DATA_FILES, batch_size=batch_size
        ):
            report.raw_count += 1
            source_counts[raw.get("dataset_source", "unknown")] += 1
            try:
                sample = _convert_row(raw, conversion_transforms)
            except Reject as exc:
                report.stage1_drop_reasons[exc.reason] = (
                    report.stage1_drop_reasons.get(exc.reason, 0) + 1
                )
                continue
            report.stage1_count += 1
            state = pipeline.process(sample)
            if state is not None:
                yield CurationInput(state.sample, state.batches, state.parsed_arguments)
        report.dataset_config_count = report.stage1_count
        report.filtered_count = pipeline.passed["termination"]
        report.filter_drop_reasons = dict(pipeline.drops)
        report.strip_thinking_applied = filters.strip_thinking
        report.dataset_config_transform_counts.update(
            {
                "source": {
                    "revision": DATASET_REVISION,
                    "files": list(DATA_FILES),
                    "split": "train",
                    "subsets": dict(source_counts),
                },
                "pipeline_passed": dict(pipeline.passed),
                "pipeline_drops": dict(pipeline.stage_drops),
                "transformations": dict(conversion_transforms + pipeline.transforms),
            }
        )

    def output() -> Generator[ConversationSample, None, None]:
        with curate(
            converted(),
            scope=lambda record: CurationScope(
                DATASET_ID, record.sample.raw["dataset_source"], "train"
            ),
            config=curation_config or CurationConfig(),
        ) as run:
            count = 0
            for sample in run:
                count += 1
                yield sample
            report.final_count = count
            report.dataset_config_transform_counts["curation"] = [
                r.as_dict() for r in run.reports.values()
            ]

    return output(), report


def load(
    dataset_config: DolciConfig | None = None,
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
    return list(rows), report
