"""Model-visible trajectory deduplication.

This module deliberately does not use the coarse first-user/tool-name keys in
``fcanalysis.overlap``.  Equality is established from the complete canonical
``messages`` and ``tools`` projection of each sample.

Level 1, conservative Level 1.5, and strict Level 2 are deletion rules.  Level
2 groups exact system/user context, complete unordered tools, ordered assistant
call batches and normalized arguments, and ordered normalized tool results
while excluding assistant prose and reasoning.  It retains the original member
with the fewest model-visible assistant-content characters, with stable input
order breaking ties.  Levels 3--5 are aggregate audit metadata only.
"""

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
import re
from typing import Any, Literal

import orjson

from .format import ConversationSample


type DeduplicationLevel = Literal["level_1", "level_1_5", "level_2"]

_CANONICAL_JSON_OPTIONS = orjson.OPT_SORT_KEYS
_TEXT_FIELDS = frozenset({"content", "description", "text", "title"})
_STRUCTURED_MARKERS = ("`", "~", "{", "}", "[", "]", "<", ">", "|", "\\")
_STRUCTURED_LINE_PREFIX = re.compile(r"(?:#{1,6}|>|[-+*]|\d+[.)])\s")
_CODE_LIKE_TEXT = re.compile(
    r"(?:"
    r"(?:^|\s)(?:class|const|def|enum|export|from|function|import|interface|let|"
    r"package|return|select|struct|type|using|var)\s"
    r"|[A-Za-z_]\w*\s*\("
    r"|=>|==|!=|:=|::|;|="
    r")",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class DatasetDeduplicationCounts:
    """Input, output, and removal counts for one source dataset."""

    input_samples: int
    output_samples: int
    removed_samples: int
    exact_removed_samples: int
    normalized_removed_samples: int


@dataclass(frozen=True, slots=True)
class DeduplicationReport:
    """Aggregate counts from one deterministic deduplication pass."""

    level: DeduplicationLevel
    input_samples: int
    output_samples: int
    removed_samples: int
    unique_trajectories: int
    duplicate_groups: int
    exact_removed_samples: int
    normalized_removed_samples: int
    by_dataset: dict[str, DatasetDeduplicationCounts]


@dataclass(frozen=True, slots=True)
class EquivalenceAuditCounts:
    """Aggregate group counts for one non-destructive equivalence audit."""

    eligible_samples: int
    unique_groups: int
    repeated_groups: int
    samples_in_repeated_groups: int
    extra_samples: int


@dataclass(frozen=True, slots=True)
class EquivalenceAuditReport:
    """Audit-only Level 3--5 metadata for an unchanged sample collection."""

    level_3: EquivalenceAuditCounts
    level_4: EquivalenceAuditCounts
    level_5: EquivalenceAuditCounts

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy suitable for a loader report."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SubsetCurationReport:
    """Cumulative per-subset deletion stages and final audit metadata."""

    input_samples: int
    output_samples: int
    removed_samples: int
    level_1: DeduplicationReport
    level_1_5: DeduplicationReport
    level_2: DeduplicationReport
    level_3: EquivalenceAuditCounts
    level_4: EquivalenceAuditCounts
    level_5: EquivalenceAuditCounts

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy suitable for a loader report."""
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    """Serialize a JSON-compatible value while ignoring object-key order."""
    return orjson.dumps(value, option=_CANONICAL_JSON_OPTIONS)


def _normalize_embedded_json(value: Any) -> Any:
    """Parse a JSON string when possible, preserving every non-JSON value."""
    if not isinstance(value, str):
        return value
    try:
        return orjson.loads(value)
    except orjson.JSONDecodeError:
        return value


def _ordered_call_batch(message: dict[str, Any]) -> list[Any]:
    """Project one assistant call batch without source linkage identifiers."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []

    projected: list[Any] = []
    for call in calls:
        if not isinstance(call, dict):
            projected.append(call)
            continue

        normalized_call: dict[str, Any] = {}
        if "type" in call:
            normalized_call["type"] = call["type"]

        function = call.get("function")
        if not isinstance(function, dict):
            normalized_call["function"] = function
            projected.append(normalized_call)
            continue

        normalized_function: dict[str, Any] = {}
        if "name" in function:
            normalized_function["name"] = function["name"]
        if "arguments" in function:
            normalized_function["arguments"] = _normalize_embedded_json(
                function["arguments"]
            )
        normalized_call["function"] = normalized_function
        projected.append(normalized_call)
    return projected


def _sorted_tool_environment(sample: ConversationSample) -> list[Any]:
    """Return a sorted comparison copy of the complete tool environment."""
    return sorted(sample.tools, key=_canonical_json)


def _level_2_key(sample: ConversationSample) -> bytes:
    """Build the strict context/action/result key with assistant prose omitted."""
    events: list[Any] = []
    for message in sample.messages:
        if not isinstance(message, dict):
            events.append(message)
            continue

        role = message.get("role")
        if role in {"system", "user"}:
            events.append({"role": role, "content": message.get("content")})
        elif role == "assistant":
            events.append(
                {"role": "assistant", "tool_calls": _ordered_call_batch(message)}
            )
        elif role == "tool":
            events.append(
                {
                    "role": "tool",
                    "content": _normalize_embedded_json(message.get("content")),
                }
            )
        else:
            # Unknown roles remain observable instead of being silently erased.
            events.append(message)

    return _canonical_json(
        {
            "grounded_event_trace": events,
            "tools": _sorted_tool_environment(sample),
        }
    )


def _assistant_content_characters(sample: ConversationSample) -> int:
    """Count model-visible assistant prose characters after loader filtering."""
    score = 0
    for message in sample.messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            msg = "assistant content must be a string or null before Level 2"
            raise TypeError(msg)
        score += len(content)
    return score


def _remove_blank_boundary_lines(text: str) -> str:
    """Remove wholly blank outer lines without changing nonblank-line spacing."""
    lines = text.split("\n")
    start = 0
    end = len(lines)

    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    return "\n".join(lines[start:end])


def _is_high_confidence_plain_prose(text: str) -> bool:
    """Whether a text value is safe for Level-1.5 whitespace normalization.

    This deliberately abstains on JSON values, Markdown/code delimiters,
    indented nonblank lines, list/heading/quote structure, and common code
    syntax.  False negatives only retain extra samples.
    """
    stripped = text.strip()
    if not stripped:
        return True
    try:
        orjson.loads(stripped)
    except orjson.JSONDecodeError:
        pass
    else:
        return False

    if any(marker in text for marker in _STRUCTURED_MARKERS):
        return False
    if _CODE_LIKE_TEXT.search(text):
        return False

    for line in text.split("\n"):
        if not line.strip():
            continue
        if line[0].isspace() or _STRUCTURED_LINE_PREFIX.match(line):
            return False
    return True


def _normalize_plain_prose(text: str) -> str:
    """Normalize approved whitespace, or return non-prose text unchanged."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not _is_high_confidence_plain_prose(normalized):
        return text
    return _remove_blank_boundary_lines(normalized)


def _normalize_level_1_5(value: Any, *, field: str | None = None) -> Any:
    """Build a conservative, context-aware copy without mutating ``value``.

    Only high-confidence prose in known text fields is normalized.  An
    ``arguments`` subtree is opaque: argument strings, arrays, object values,
    and their whitespace remain exact.  Object keys and array order are always
    preserved.
    """
    if field == "arguments":
        return value
    if isinstance(value, str):
        if field in _TEXT_FIELDS:
            return _normalize_plain_prose(value)
        return value
    if isinstance(value, list):
        return [_normalize_level_1_5(item, field=field) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_level_1_5(item, field=key) for key, item in value.items()
        }
    return value


def _normalize_messages_level_1_5(messages: list[dict[str, Any]]) -> list[Any]:
    """Normalize message prose while treating every tool result as opaque."""
    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        is_tool_result = message.get("role") == "tool"
        normalized.append(
            {
                key: (
                    item
                    if is_tool_result and key == "content"
                    else _normalize_level_1_5(item, field=key)
                )
                for key, item in message.items()
            }
        )
    return normalized


def _trajectory_key(
    sample: ConversationSample,
    *,
    level: Literal["level_1", "level_1_5"],
) -> bytes:
    if level == "level_1":
        messages: Any = sample.messages
        tools: list[Any] = sample.tools
    else:
        messages = _normalize_messages_level_1_5(sample.messages)
        tools = _normalize_level_1_5(sample.tools)

    # Tool definitions are an unordered multiset for comparison.  Sorting a
    # separate list preserves both duplicate multiplicity and sample.tools.
    sorted_tools = sorted(tools, key=_canonical_json)
    projection = {"messages": messages, "tools": sorted_tools}
    return _canonical_json(projection)


def _deduplicate_level_2(
    samples: list[ConversationSample],
) -> tuple[list[ConversationSample], DeduplicationReport]:
    """Select the stable shortest-assistant-content member of each strict group."""
    input_by_dataset: Counter[str] = Counter()
    output_by_dataset: Counter[str] = Counter()
    dataset_order: list[str] = []
    group_sizes: Counter[bytes] = Counter()
    selected_by_key: dict[bytes, tuple[int, int]] = {}

    for index, sample in enumerate(samples):
        dataset = sample.dataset
        if dataset not in input_by_dataset:
            dataset_order.append(dataset)
        input_by_dataset[dataset] += 1

        key = _level_2_key(sample)
        score = _assistant_content_characters(sample)
        group_sizes[key] += 1
        selected = selected_by_key.get(key)
        if selected is None or score < selected[1]:
            selected_by_key[key] = (index, score)

    selected_indices = {index for index, _ in selected_by_key.values()}
    kept = [sample for index, sample in enumerate(samples) if index in selected_indices]
    for sample in kept:
        output_by_dataset[sample.dataset] += 1

    by_dataset = {
        dataset: DatasetDeduplicationCounts(
            input_samples=input_by_dataset[dataset],
            output_samples=output_by_dataset[dataset],
            removed_samples=input_by_dataset[dataset] - output_by_dataset[dataset],
            exact_removed_samples=0,
            normalized_removed_samples=0,
        )
        for dataset in dataset_order
    }
    duplicate_groups = sum(count > 1 for count in group_sizes.values())
    removed_samples = len(samples) - len(kept)
    return kept, DeduplicationReport(
        level="level_2",
        input_samples=len(samples),
        output_samples=len(kept),
        removed_samples=removed_samples,
        unique_trajectories=len(group_sizes),
        duplicate_groups=duplicate_groups,
        exact_removed_samples=0,
        normalized_removed_samples=0,
        by_dataset=by_dataset,
    )


def _deduplicate_level_1_like(
    samples: list[ConversationSample],
    *,
    level: Literal["level_1", "level_1_5"],
    input_is_level_1_unique: bool = False,
) -> tuple[list[ConversationSample], DeduplicationReport]:
    input_by_dataset: Counter[str] = Counter()
    output_by_dataset: Counter[str] = Counter()
    exact_removed_by_dataset: Counter[str] = Counter()
    normalized_removed_by_dataset: Counter[str] = Counter()
    dataset_order: list[str] = []

    # Level 1.5 accounting follows two ordered conceptual passes: exact keys
    # are removed first, then the surviving exact variants are compared using
    # Level-1.5 keys.  Tracking all prior exact keys produces the same counts
    # without constructing and traversing an intermediate sample list.
    seen: set[bytes] = set()
    seen_exact: set[bytes] | None = set() if not input_is_level_1_unique else None
    duplicate_keys: set[bytes] = set()
    kept: list[ConversationSample] = []

    for sample in samples:
        dataset = sample.dataset
        if dataset not in input_by_dataset:
            dataset_order.append(dataset)
        input_by_dataset[dataset] += 1

        if level == "level_1":
            comparison_key = _trajectory_key(sample, level="level_1")
            exact_key = comparison_key
        else:
            comparison_key = _trajectory_key(sample, level="level_1_5")
            exact_key = (
                None
                if input_is_level_1_unique
                else _trajectory_key(sample, level="level_1")
            )
        is_exact_duplicate = seen_exact is not None and exact_key in seen_exact
        if seen_exact is not None:
            assert exact_key is not None
            seen_exact.add(exact_key)

        if comparison_key not in seen:
            seen.add(comparison_key)
            kept.append(sample)
            output_by_dataset[dataset] += 1
            continue

        duplicate_keys.add(comparison_key)
        if is_exact_duplicate:
            exact_removed_by_dataset[dataset] += 1
        else:
            normalized_removed_by_dataset[dataset] += 1

    by_dataset = {
        dataset: DatasetDeduplicationCounts(
            input_samples=input_by_dataset[dataset],
            output_samples=output_by_dataset[dataset],
            removed_samples=input_by_dataset[dataset] - output_by_dataset[dataset],
            exact_removed_samples=exact_removed_by_dataset[dataset],
            normalized_removed_samples=normalized_removed_by_dataset[dataset],
        )
        for dataset in dataset_order
    }
    exact_removed_samples = sum(exact_removed_by_dataset.values())
    normalized_removed_samples = sum(normalized_removed_by_dataset.values())

    return kept, DeduplicationReport(
        level=level,
        input_samples=len(samples),
        output_samples=len(kept),
        removed_samples=len(samples) - len(kept),
        unique_trajectories=len(seen),
        duplicate_groups=len(duplicate_keys),
        exact_removed_samples=exact_removed_samples,
        normalized_removed_samples=normalized_removed_samples,
        by_dataset=by_dataset,
    )


def deduplicate_samples(
    samples: list[ConversationSample],
    *,
    level: DeduplicationLevel = "level_1",
) -> tuple[list[ConversationSample], DeduplicationReport]:
    """Retain one sample for every selected model-visible equivalence key.

    ``level_1`` compares the complete canonical projection exactly, except for
    JSON object-key order and tool-definition list order.  ``level_1_5`` is an
    explicit opt-in that additionally normalizes line endings and removes only
    wholly blank boundary lines from high-confidence plain-prose fields.  It
    abstains on code, structured literals, arguments, and tool-result content.
    ``level_2`` fixes system/user turns, assistant-turn positions and ordered
    normalized call batches, normalized tool-result positions/content, and the
    complete unordered tool environment while excluding assistant prose and
    reasoning.  It retains the shortest assistant-content member, with stable
    first-seen selection on equal scores.

    The returned list contains the original retained ``ConversationSample``
    objects in input order.  Neither input samples nor any of their nested
    values are modified.  Dataset, sample ID, annotations, and raw source data
    are excluded from equality but are counted in the report.
    """
    if level not in ("level_1", "level_1_5", "level_2"):
        msg = f"unsupported deduplication level: {level!r}"
        raise ValueError(msg)
    if level == "level_2":
        return _deduplicate_level_2(samples)
    return _deduplicate_level_1_like(samples, level=level)


def _level_3_key(sample: ConversationSample) -> tuple[bytes, bool]:
    """Return the environment/action/result key and whether it contains a call."""
    system_messages: list[Any] = []
    behavior_trace: list[Any] = []
    has_calls = False

    for message in sample.messages:
        if not isinstance(message, dict):
            behavior_trace.append(message)
            continue

        role = message.get("role")
        if role == "system":
            system_messages.append(
                {"role": "system", "content": message.get("content")}
            )
        elif role == "user":
            # Preserve turn alignment, but omit user wording.
            behavior_trace.append({"role": "user"})
        elif role == "assistant":
            calls = _ordered_call_batch(message)
            if calls:
                has_calls = True
                behavior_trace.append({"role": "assistant", "tool_calls": calls})
        elif role == "tool":
            behavior_trace.append(
                {
                    "role": "tool",
                    "content": _normalize_embedded_json(message.get("content")),
                }
            )
        else:
            behavior_trace.append(message)

    key = _canonical_json(
        {
            "environment": {
                "system_messages": system_messages,
                "tools": _sorted_tool_environment(sample),
            },
            "behavior_trace": behavior_trace,
        }
    )
    return key, has_calls


def _level_4_key(sample: ConversationSample) -> bytes:
    """Return ordered call-name batches while preserving parallel boundaries."""
    batches: list[list[Any]] = []
    for message in sample.messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            continue
        names: list[Any] = []
        for call in calls:
            if not isinstance(call, dict):
                names.append(call)
                continue
            function = call.get("function")
            names.append(function.get("name") if isinstance(function, dict) else None)
        batches.append(names)
    return _canonical_json(batches)


def _level_5_key(sample: ConversationSample) -> bytes:
    """Return the complete unordered tool-environment key."""
    return _canonical_json(_sorted_tool_environment(sample))


def _audit_counts(keys: Iterable[bytes]) -> EquivalenceAuditCounts:
    groups = Counter(keys)
    repeated_sizes = [count for count in groups.values() if count > 1]
    return EquivalenceAuditCounts(
        eligible_samples=sum(groups.values()),
        unique_groups=len(groups),
        repeated_groups=len(repeated_sizes),
        samples_in_repeated_groups=sum(repeated_sizes),
        extra_samples=sum(count - 1 for count in repeated_sizes),
    )


def _iter_level_3_keys(samples: list[ConversationSample]) -> Iterator[bytes]:
    for sample in samples:
        key, has_calls = _level_3_key(sample)
        if has_calls:
            yield key


def audit_subset_samples(samples: list[ConversationSample]) -> EquivalenceAuditReport:
    """Compute aggregate Level 3--5 metadata without modifying any sample.

    Level 3 omits user wording and assistant prose, retains exact system/tool
    environment plus chronological user positions, call batches, normalized
    arguments, and normalized results, and excludes rows with no calls.  Level
    4 records ordered call-name batches with parallel boundaries.  Level 5
    records the complete unordered tool environment.  These levels never
    remove, annotate, reweight, or select samples.
    """
    # Count one level at a time. Complete keys remain the equality evidence,
    # but we avoid retaining three corpus-sized key collections concurrently.
    return EquivalenceAuditReport(
        level_3=_audit_counts(_iter_level_3_keys(samples)),
        level_4=_audit_counts(_level_4_key(sample) for sample in samples),
        level_5=_audit_counts(_level_5_key(sample) for sample in samples),
    )


def curate_subset_samples(
    samples: list[ConversationSample],
) -> tuple[list[ConversationSample], SubsetCurationReport]:
    """Apply cumulative Level 1, Level 1.5, and Level 2 within one subset.

    Audit metadata is computed over the final post-Level-2 collection.  The
    returned list contains only original sample objects in their original
    relative order; neither retained nor removed rows are modified.
    """
    after_level_1, level_1 = deduplicate_samples(samples, level="level_1")
    after_level_1_5, level_1_5 = _deduplicate_level_1_like(
        after_level_1,
        level="level_1_5",
        input_is_level_1_unique=True,
    )
    after_level_2, level_2 = deduplicate_samples(after_level_1_5, level="level_2")
    audit = audit_subset_samples(after_level_2)

    return after_level_2, SubsetCurationReport(
        input_samples=len(samples),
        output_samples=len(after_level_2),
        removed_samples=len(samples) - len(after_level_2),
        level_1=level_1,
        level_1_5=level_1_5,
        level_2=level_2,
        level_3=audit.level_3,
        level_4=audit.level_4,
        level_5=audit.level_5,
    )
