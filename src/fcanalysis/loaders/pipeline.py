"""Composable, ordered row stages for function-calling SFT adapters.

Adapters own source decoding and audited runtime grammars. A stage operates on
one owned normalized row, raises ``Reject`` to quarantine it, or returns after
validation/transformation. Insert source stages before, between or after shared
stages; reconstruct prefix-split trajectories before validating their final
training unit. ``Pipeline`` is streaming and partition-local, with mergeable
counters. Corpus curation is a separate disk-backed stage.

Structural validity is not semantic correctness. Discovery, protected-state
grounding, producer sentinels and asynchronous events require audited adapter
callbacks, never keyword guesses. No audit field is promoted into context.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType
from typing import Any, Literal, cast

from ..format import ConversationSample
from .base import _override_system_messages
from .curation import BoundCallBatch
from .normalization import Reject, json_bytes, normalize_arguments, reconcile_tools
from .schema import check_arguments, compile_schema


@dataclass(slots=True)
class RowState:
    sample: ConversationSample
    batches: tuple[BoundCallBatch, ...] = ()
    parsed_arguments: dict[tuple[int, int], dict[str, Any]] = field(
        default_factory=dict
    )
    transforms: Counter[str] = field(default_factory=Counter)
    _argument_text: dict[tuple[int, int], str] = field(default_factory=dict)


Stage = Callable[[RowState], None]
_REASONING_TAG = re.compile(r"</?(?:think|reasoning)>")
_FENCE_START = re.compile(r" {0,3}(`{3,}|~{3,})")
_BACKTICKS = re.compile(r"`+")


class Pipeline:
    def __init__(
        self,
        stages: Iterable[tuple[str, Stage]],
        *,
        state_factory: Callable[[ConversationSample], RowState] = RowState,
    ):
        self.stages = tuple(stages)
        self.state_factory = state_factory
        if len({name for name, _ in self.stages}) != len(self.stages):
            raise ValueError("Stage names must be unique")
        self.passed: Counter[str] = Counter()
        self.drops: Counter[str] = Counter()
        self.stage_drops: Counter[str] = Counter()
        self.transforms: Counter[str] = Counter()

    def process(self, sample: ConversationSample) -> RowState | None:
        state = self.state_factory(sample)
        for name, stage in self.stages:
            try:
                stage(state)
            except Reject as exc:
                self.drops[exc.reason] += 1
                self.stage_drops[name] += 1
                self.transforms.update(state.transforms)
                return None
            self.passed[name] += 1
        self.transforms.update(state.transforms)
        return state

    def __call__(self, samples: Iterable[ConversationSample]) -> Iterator[RowState]:
        for sample in samples:
            state = self.process(sample)
            if state is not None:
                yield state


def reconcile_definitions(state: RowState) -> None:
    tools, repeats = reconcile_tools(state.sample.tools)
    state.sample.tools = tools
    if repeats:
        state.transforms["duplicate_tool_definitions_removed"] += repeats


def validate_structure(state: RowState) -> None:
    """Check canonical field shapes without enforcing naive role alternation."""
    messages = state.sample.messages
    if not isinstance(messages, list):
        raise Reject("malformed_messages")
    if any(not isinstance(message, dict) for message in messages):
        raise Reject("malformed_message")
    if not messages or not any(m.get("role") == "user" for m in messages):
        raise Reject("no_user_message")
    for index, msg in enumerate(messages):
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise Reject("unknown_role")
        allowed = {"role", "content"}
        if role == "assistant":
            allowed |= {"tool_calls", "reasoning_content"}
        elif role == "tool":
            allowed |= {"tool_call_id", "name"}
        if msg.keys() - allowed:
            raise Reject("unknown_message_field")
        if not isinstance(msg.get("content"), (str, type(None))):
            raise Reject("unsupported_content_shape")
        if "reasoning_content" in msg and not isinstance(
            msg["reasoning_content"], (str, type(None))
        ):
            raise Reject("unsupported_reasoning_shape")
        calls: Any = msg.get("tool_calls", [])
        if not isinstance(calls, list):
            raise Reject("malformed_tool_calls")
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise Reject("unknown_call_field")
            unknown_fields = call.keys() - {"type", "id", "function"}
            if unknown_fields:
                raise Reject("unknown_call_field")
            call = cast(dict[str, Any], call)
            function = call.get("function")
            if (
                call.get("type") != "function"
                or not isinstance(function, dict)
                or function.keys() != {"name", "arguments"}
                or not isinstance(function["name"], str)
                or not function["name"]
            ):
                raise Reject("malformed_tool_call")
            key = (index, call_index)
            original = function["arguments"]
            if (
                not isinstance(original, str)
                or state._argument_text.get(key) != original
            ):
                serialized, parsed = normalize_arguments(original)
                if serialized != original:
                    state.transforms["arguments_canonicalized"] += 1
                function["arguments"] = serialized
                state._argument_text[key] = serialized
                state.parsed_arguments[key] = parsed


def link_calls(
    state: RowState,
    *,
    positional: bool = False,
    parallel: bool = False,
    align_results: bool = True,
    result_echo: Callable[[dict[str, Any]], tuple[str, dict[str, Any] | None] | None]
    | None = None,
) -> None:
    """Prove all call/result pairs before dropping source IDs and result names.

    ``positional`` and ``parallel`` are audited source protocol contracts.
    Complete explicit IDs take precedence; supplied contradictory evidence is
    always an error. Once correspondence is proved, ``align_results`` moves
    complete result messages within their immediate batch into unchanged call
    order by default. Disabling this transformation rejects a batch that needs
    alignment; it does not mean the correspondence was unknown. ``parallel``
    independently permits unordered whole-pair comparison during curation and
    is never inferred from result alignment or explicit IDs.

    The returned ``state.batches`` is a proof about these exact final-view
    coordinates. Content-only reasoning removal may reuse this evidence; any
    transform affecting call/result values, order, or positions must rebuild
    it. Re-running this stage after IDs have been stripped requires an audited
    positional source contract or still-present independent echo evidence.
    An optional adapter decoder can prove pairing through complete function-name
    and argument echoes. An echo's argument value may be None for a source
    name-only protocol; distinct names are required without complete IDs.
    Without explicit IDs,
    every result must supply an unambiguous echo of the same kind; partial,
    contradictory or repeated call signatures do not establish that proof.
    Echo content remains unchanged. Echo pairing uses the same alignment policy
    as IDs and names and does not imply unordered whole-pair comparison.
    Mere presence of an earlier coordinate tuple
    cannot prove that arbitrary intervening adapter stages left pairing intact.
    """
    messages = state.sample.messages
    batches = []
    index = 0
    while index < len(messages):
        msg = messages[index]
        if msg["role"] == "tool":
            raise Reject("orphan_tool_result")
        calls = msg.get("tool_calls", [])
        if not calls:
            index += 1
            continue
        end = index + 1
        while end < len(messages) and messages[end]["role"] == "tool":
            end += 1
        results = messages[index + 1 : end]
        if len(results) != len(calls):
            raise Reject("unbalanced_cardinality")
        call_ids = [call.get("id") for call in calls]
        result_ids = [result.get("tool_call_id") for result in results]
        echoes = [result_echo(result) for result in results] if result_echo else []
        if any(echo is not None for echo in echoes):
            for echo in echoes:
                if (
                    echo is None
                    or not isinstance(echo[0], str)
                    or not echo[0]
                    or (echo[1] is not None and not isinstance(echo[1], dict))
                ):
                    raise Reject("invalid_source_tool_linkage")
        else:
            echoes = []
        ids_supplied = any(value is not None for value in (*call_ids, *result_ids))
        if ids_supplied:
            if (
                not all(
                    isinstance(value, str) and value
                    for value in (*call_ids, *result_ids)
                )
                or len(set(call_ids)) != len(calls)
                or len(set(result_ids)) != len(results)
                or set(call_ids) != set(result_ids)
            ):
                raise Reject("invalid_source_tool_linkage")
            by_id = dict(zip(result_ids, results, strict=True))
            ordered = [by_id[value] for value in call_ids]
            if echoes:
                by_id_echo = dict(zip(result_ids, echoes, strict=True))
                for i, call in enumerate(calls):
                    echo = by_id_echo[call_ids[i]]
                    assert echo is not None
                    if echo[0] != call["function"]["name"] or (
                        echo[1] is not None
                        and json_bytes(echo[1])
                        != json_bytes(state.parsed_arguments[index, i])
                    ):
                        raise Reject("invalid_source_tool_linkage")
        elif echoes:
            with_arguments = all(
                echo is not None and echo[1] is not None for echo in echoes
            )
            if not with_arguments and any(
                echo is not None and echo[1] is not None for echo in echoes
            ):
                raise Reject("invalid_source_tool_linkage")
            call_keys = [
                json_bytes(
                    [
                        call["function"]["name"],
                        state.parsed_arguments[index, i] if with_arguments else None,
                    ]
                )
                for i, call in enumerate(calls)
            ]
            if len(set(call_keys)) != len(calls):
                raise Reject("ambiguous_source_tool_linkage")
            echo_keys = [json_bytes(list(echo)) for echo in echoes if echo is not None]
            if len(set(echo_keys)) != len(results) or set(echo_keys) != set(call_keys):
                raise Reject("invalid_source_tool_linkage")
            by_echo = dict(zip(echo_keys, results, strict=True))
            ordered = [by_echo[key] for key in call_keys]
        else:
            names = [result.get("name") for result in results]
            call_names = [call["function"]["name"] for call in calls]
            if all(isinstance(name, str) and name for name in names) and len(
                set(call_names)
            ) == len(calls):
                if len(set(names)) != len(results) or set(names) != set(call_names):
                    raise Reject("invalid_source_tool_linkage")
                by_name = dict(zip(names, results, strict=True))
                ordered = [by_name[name] for name in call_names]
            elif positional or len(calls) == 1:
                ordered = results
            else:
                raise Reject("ambiguous_source_tool_linkage")
        for call, result in zip(calls, ordered, strict=True):
            if (
                result.get("name") is not None
                and result["name"] != call["function"]["name"]
            ):
                raise Reject("invalid_source_tool_linkage")
        if any(a is not b for a, b in zip(results, ordered, strict=True)):
            if not align_results:
                # Compatibility reason: this is a disabled transformation,
                # not missing correspondence or an unproven pairing.
                raise Reject("unproven_result_reordering")
            messages[index + 1 : end] = ordered
            state.transforms["result_batches_reordered"] += 1
        for call in calls:
            if "id" in call:
                del call["id"]
                state.transforms["source_linkage_fields_removed"] += 1
        for result in ordered:
            for key in ("tool_call_id", "name"):
                if key in result:
                    del result[key]
                    state.transforms["source_linkage_fields_removed"] += 1
        batches.append(BoundCallBatch(index, tuple(range(index + 1, end)), parallel))
        index = end
    state.batches = tuple(batches)


@dataclass(frozen=True, slots=True)
class CapabilityChange:
    """An adapter-parsed visible capability event at the current message."""

    tools: tuple[dict[str, Any], ...] = ()
    remove: tuple[str, ...] = ()
    replace: bool = False


Discovery = Callable[
    [RowState, int, Mapping[str, dict[str, Any]]], CapabilityChange | None
]
TargetValidator = Callable[[RowState, int, Mapping[str, dict[str, Any]]], None]


def validate_capabilities(
    state: RowState,
    *,
    check_arguments: bool = True,
    discovery: Discovery | None = None,
    target_validator: TargetValidator | None = None,
) -> None:
    """Walk visible capability epochs; adapters parse any discovery grammar.

    The callback runs after a visible message, so a result cannot authorize
    an earlier call or a sibling call in its preceding parallel batch.
    Later system capability changes use the same callback. A target validator
    must use only messages before its supplied index and active definitions.
    """
    active = {tool["function"]["name"]: tool for tool in state.sample.tools}
    compiled = {}

    def install(tool: dict[str, Any]) -> None:
        function = tool["function"]
        if check_arguments:
            # Omitted parameters are an omitted constraint, not invented {}.
            compiled[function["name"]] = compile_schema(function.get("parameters", {}))

    for tool in active.values():
        install(tool)
    for index, msg in enumerate(state.sample.messages):
        if msg["role"] == "assistant":
            for call_index, call in enumerate(msg.get("tool_calls", [])):
                name = call["function"]["name"]
                if name not in active:
                    raise Reject("undefined_function_calls")
                if check_arguments:
                    _validate_arguments(
                        state.parsed_arguments[index, call_index], compiled[name]
                    )
            if target_validator is not None:
                target_validator(state, index, MappingProxyType(active))
        if discovery is not None:
            change = discovery(state, index, MappingProxyType(active))
            if change is not None:
                for name in change.remove:
                    active.pop(name, None)
                    compiled.pop(name, None)
                additions, _ = reconcile_tools(list(change.tools))
                for tool in additions:
                    name = tool["function"]["name"]
                    if (
                        name in active
                        and json_bytes(active[name]) != json_bytes(tool)
                        and not change.replace
                    ):
                        raise Reject("conflicting_dynamic_tool_definition")
                    active[name] = tool
                    install(tool)


# Keep the public keyword readable without shadowing the schema function.
_validate_arguments = check_arguments


def override_system(state: RowState, *, override: str | None) -> None:
    if override is None:
        return
    if not isinstance(override, str):
        raise TypeError("system_message_override must be a string or None")
    state.sample = _override_system_messages(state.sample, override)
    # Indices may change. Final structure/linkage stages rebuild both caches.
    state.parsed_arguments.clear()
    state._argument_text.clear()
    state.transforms["system_message_override"] += 1


def _code_spans(content: str) -> list[tuple[int, int]]:
    """Conservatively shield Markdown code, including unfinished fences."""
    spans: list[tuple[int, int]] = []
    fence: tuple[int, str, int] | None = None
    offset = 0
    for line in content.splitlines(keepends=True):
        end = offset + len(line)
        if fence is not None:
            start, marker, width = fence
            if re.fullmatch(
                rf" {{0,3}}{re.escape(marker)}{{{width},}}[ \t]*(?:\r?\n)?", line
            ):
                spans.append((start, end))
                fence = None
        else:
            match = _FENCE_START.match(line)
            if match is not None:
                markers = match.group(1)
                fence = (offset, markers[0], len(markers))
            elif line.startswith(("    ", "\t")):
                spans.append((offset, end))
        offset = end
    if fence is not None:
        spans.append((fence[0], len(content)))

    fenced = tuple(spans)
    span_index = 0
    consumed = 0
    for match in _BACKTICKS.finditer(content):
        start = match.start()
        if start < consumed:
            continue
        while span_index < len(fenced) and fenced[span_index][1] <= start:
            span_index += 1
        limit = len(content)
        if span_index < len(fenced):
            boundary_start, boundary_end = fenced[span_index]
            if boundary_start <= start < boundary_end:
                continue
            limit = boundary_start
        width = len(match.group())
        close = re.compile(rf"(?<!`)`{{{width}}}(?!`)").search(
            content, match.end(), limit
        )
        if close is not None:
            consumed = close.end()
        else:
            # An unfinished inline code span is ambiguous. Preserve its line
            # rather than interpreting embedded documentation as private data.
            newline = content.find("\n", match.end(), limit)
            consumed = limit if newline < 0 else newline
        spans.append((start, consumed))
    return sorted(spans)


def _native_reasoning_tags(content: str) -> Iterator[re.Match[str]]:
    """Yield supported native control tokens outside protected literal code."""
    tags = list(_REASONING_TAG.finditer(content))
    code = _code_spans(content) if tags else []
    code_index = 0
    for match in tags:
        while code_index < len(code) and code[code_index][1] <= match.start():
            code_index += 1
        if (
            code_index < len(code)
            and code[code_index][0] <= match.start() < code[code_index][1]
        ):
            continue
        yield match


def native_reasoning_spans(content: str) -> tuple[tuple[int, int], ...]:
    """Locate exact balanced native spans while shielding literal code.

    Source adapters may reuse these boundaries for an audited prefix projection
    without removing other source text. Nested, mismatched or unmatched native
    tags quarantine; ordinary words and tool names have no special meaning.
    """
    spans = []
    opened = None
    for match in _native_reasoning_tags(content):
        tag = match.group()
        if not tag.startswith("</"):
            if opened is not None:
                raise Reject("ambiguous_reasoning")
            opened = (tag[1:-1], match.start())
        else:
            if opened is None or opened[0] != tag[2:-1]:
                raise Reject("ambiguous_reasoning")
            spans.append((opened[1], match.end()))
            opened = None
    if opened is not None:
        raise Reject("ambiguous_reasoning")
    return tuple(spans)


def validate_reasoning_boundaries(state: RowState) -> None:
    """Certify inline native boundaries without removing reasoning.

    Adapters may require this in both modes when malformed native export
    boundaries are source defects. The existing boundary parser shields code
    literals; structured reasoning remains an opaque string field.
    """
    for message in state.sample.messages:
        if message["role"] == "assistant" and isinstance(message.get("content"), str):
            native_reasoning_spans(message["content"])


def remove_reasoning(state: RowState, *, inline: bool = True) -> None:
    """Remove exact balanced native spans, preserving every outside character.

    Nested, mismatched or unmatched tags are inseparable and quarantined.
    Literal tags in fenced, inline, or indented code are preserved, including
    unfinished code spans; ordinary words and tool calls (including reasoning-
    like tools) are never interpreted as private tokens.

    ``inline=False`` removes only structured reasoning fields. Select it when
    an adapter has already decoded a proven source-native envelope and the
    remaining content is opaque protocol text that can contain literal tags.
    """
    for msg in state.sample.messages:
        if msg["role"] != "assistant":
            continue
        content = msg.get("content")
        if inline and isinstance(content, str):
            spans = native_reasoning_spans(content)
            if spans:
                parts = []
                cursor = 0
                for start, end in spans:
                    parts.append(content[cursor:start])
                    cursor = end
                parts.append(content[cursor:])
                msg["content"] = "".join(parts)
                state.transforms["reasoning_spans_removed"] += len(spans)
        if "reasoning_content" in msg:
            del msg["reasoning_content"]
            state.transforms["reasoning_fields_removed"] += 1


def _native_only_ambiguous_endpoint(content: str) -> bool:
    """Recognize two no-answer forms without interpreting ambiguous bodies."""
    tags = tuple(_native_reasoning_tags(content))
    if not tags:
        return False

    # Control tokens themselves are not an answer, even when unmatched.
    cursor = 0
    for tag in tags:
        if content[cursor : tag.start()].strip():
            break
        cursor = tag.end()
    else:
        if not content[cursor:].strip():
            return True

    # Skip only demonstrably balanced, non-nested prefixes. An open-only tail
    # cannot contain an outside answer; repeated openers do not close it.
    cursor = index = 0
    while index < len(tags):
        opened = tags[index]
        if content[cursor : opened.start()].strip() or opened.group().startswith("</"):
            return False
        if (
            index + 1 < len(tags)
            and tags[index + 1].group() == "</" + opened.group()[1:]
        ):
            cursor = tags[index + 1].end()
            index += 2
            continue
        return all(not tag.group().startswith("</") for tag in tags[index:])
    return False


def _has_answer_text(content: str) -> bool:
    """Inspect final answer presence without changing retained source content.

    Balanced native spans, native control tokens alone, and a native-only
    unclosed tail cannot complete a response. Other ambiguous boundaries remain
    the configured reasoning stage's responsibility.
    """
    try:
        spans = native_reasoning_spans(content)
    except Reject:
        return bool(content.strip()) and not _native_only_ambiguous_endpoint(content)
    cursor = 0
    for start, end in spans:
        if content[cursor:start].strip():
            return True
        cursor = end
    return bool(content[cursor:].strip())


def validate_termination(state: RowState, *, action_only: bool = False) -> None:
    messages = state.sample.messages
    if not messages or messages[-1]["role"] != "assistant":
        raise Reject("incomplete_termination")
    final = messages[-1]
    if final.get("tool_calls"):
        if not action_only:
            raise Reject("incomplete_termination")
    elif not _has_answer_text(final.get("content") or ""):
        raise Reject("empty_final_assistant")
    for msg in messages:
        if (
            msg["role"] == "assistant"
            and not msg.get("tool_calls")
            and not (
                (msg.get("content") or "").strip()
                or (msg.get("reasoning_content") or "").strip()
            )
        ):
            raise Reject("empty_assistant_target")


@dataclass(frozen=True, slots=True)
class JobEvent:
    """An exact adapter-decoded event; job IDs must be visibly grounded."""

    job: str
    phase: str  # started, pending, completed, fetched, abandoned, summarized
    payload: bool = False
    grounded: bool = True


def validate_lifecycle(
    events: Iterable[JobEvent],
    *,
    require_fetch: bool = False,
    require_completion_for: Literal["all", "started"] = "all",
) -> None:
    """Reusable lifecycle mechanics, independent of any runtime's field names.

    Poll-only rows need a visibly grounded identifier but need not include a
    start event. Adapters prove permission, identifiers and visible abandonment
    policy, and emit ``summarized`` for targets claiming completed results.
    The default requires completion for every observed job. An audited external
    status-query protocol may select ``started`` to permit pending/status-only
    external jobs while requiring completion for jobs started in this row.
    Grounding, transition checks and summary payload requirements still apply
    to every job. This option does not interpret arbitrary completion prose.
    """
    if require_completion_for not in ("all", "started"):
        raise ValueError("require_completion_for must be 'all' or 'started'")
    jobs: dict[str, tuple[str, bool]] = {}
    started: set[str] = set()
    for event in events:
        if not event.grounded:
            raise Reject("ungrounded_job_state")
        previous, payload = jobs.get(event.job, ("unknown", False))
        if event.phase == "summarized":
            allowed = ("fetched",) if require_fetch else ("completed", "fetched")
            if previous not in allowed or not payload:
                raise Reject("premature_job_summary")
        elif event.phase == "fetched":
            if previous not in ("completed", "fetched"):
                raise Reject("premature_job_fetch")
            jobs[event.job] = (event.phase, event.payload)
        elif event.phase in ("started", "pending", "completed", "abandoned"):
            jobs[event.job] = (event.phase, event.payload)
            if event.phase == "started":
                started.add(event.job)
        else:
            raise Reject("unknown_job_phase")
    for job, (phase, payload) in jobs.items():
        if require_completion_for == "started" and job not in started:
            continue
        if phase == "abandoned":
            continue
        if (
            phase not in (("fetched",) if require_fetch else ("completed", "fetched"))
            or not payload
        ):
            raise Reject("incomplete_async_job")
