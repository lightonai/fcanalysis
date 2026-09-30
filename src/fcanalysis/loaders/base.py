import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from ..format import ConversationSample
from ..validation import has_invalid_arguments, has_undefined_function_calls

THINK_PATTERNS = (
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL),
)


@dataclass(slots=True)
class FilterConfig:
    strip_thinking: bool = False
    require_parseable_arguments: bool = False
    require_balanced_cardinality: bool = False
    require_defined_functions: bool = False
    require_valid_arguments: bool = False
    system_message_override: str | None = None
    # Migrated shared-stage loaders: align proven results to unchanged calls.
    align_results: bool = True


@dataclass(slots=True)
class LoadReport:
    dataset: str
    raw_count: int
    stage1_count: int
    stage1_drop_reasons: dict[str, int] = field(default_factory=dict)
    stage1_issue_counts: dict[str, int] = field(default_factory=dict)
    dataset_config_count: int | None = None
    dataset_config_drop_reasons: dict[str, int] = field(default_factory=dict)
    dataset_config_transform_counts: dict[str, Any] = field(default_factory=dict)
    filtered_count: int | None = None
    final_count: int | None = None
    filter_config: FilterConfig | None = None
    filter_drop_reasons: dict[str, int] = field(default_factory=dict)
    strip_thinking_applied: bool = False

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.dataset}",
            f"Raw samples: {self.raw_count}",
            f"Stage 1 (converted): {self.stage1_count}",
        ]
        lines.extend(_format_counts("Stage 1 drops", self.stage1_drop_reasons))
        lines.extend(_format_counts("Stage 1 issues", self.stage1_issue_counts))
        if self.dataset_config_count is not None:
            lines.append(f"After dataset-specific config: {self.dataset_config_count}")
        lines.extend(
            _format_counts("Dataset-specific drops", self.dataset_config_drop_reasons)
        )
        lines.extend(
            _format_counts(
                "Dataset-specific transforms", self.dataset_config_transform_counts
            )
        )
        if self.filtered_count is not None:
            lines.append(f"After universal filters: {self.filtered_count}")
            lines.extend(
                _format_counts("Universal filter drops", self.filter_drop_reasons)
            )
        if self.final_count is not None:
            lines.append(f"Final output: {self.final_count}")
        thinking_stripped = self.strip_thinking_applied or (
            self.filter_config is not None and self.filter_config.strip_thinking
        )
        lines.append(f"Thinking traces stripped: {thinking_stripped}")
        return "\n".join(lines)


def _format_counts(label: str, counts: dict[str, Any]) -> list[str]:
    if not counts:
        return []
    return [f"{label}:"] + [f"  {k}: {v}" for k, v in sorted(counts.items())]


def strip_thinking_from_sample(sample: ConversationSample) -> ConversationSample:
    for msg in sample.messages:
        if msg["role"] != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            for pattern in THINK_PATTERNS:
                content = pattern.sub("", content)
            msg["content"] = content.strip() or None
        if "reasoning_content" in msg:
            del msg["reasoning_content"]
    return sample


def _has_unparseable_arguments(sample: ConversationSample) -> bool:
    for msg in sample.messages:
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            args = tc.get("function", {}).get("arguments", "")
            if isinstance(args, str):
                try:
                    json.loads(args)
                except ValueError:
                    return True
    return False


def _has_unbalanced_cardinality(sample: ConversationSample) -> bool:
    messages = sample.messages
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            num_calls = len(msg["tool_calls"])
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                j += 1
            if num_calls != j - i - 1:
                return True
            i = j
        else:
            i += 1
    return False


def _override_system_messages(
    sample: ConversationSample, override: str
) -> ConversationSample:
    """Return a sample with caller-selected system context.

    The operation is deliberately post-validation and does not mutate the
    normalized input sample or its source ``raw`` value.  An empty override
    removes system messages; every other string replaces all source system
    messages with one exact value at the first source-system position, or at
    the beginning when the source has no system message.
    """
    first_system_index = next(
        (i for i, message in enumerate(sample.messages) if message["role"] == "system"),
        0,
    )
    messages = [message for message in sample.messages if message["role"] != "system"]
    if override:
        messages.insert(first_system_index, {"role": "system", "content": override})
    return replace(sample, messages=messages)


def apply_filters(
    samples: list[ConversationSample],
    config: FilterConfig,
) -> tuple[list[ConversationSample], dict[str, int]]:
    if config.strip_thinking:
        samples = [strip_thinking_from_sample(s) for s in samples]

    predicates = [
        (key, predicate)
        for key, predicate, enabled in (
            (
                "unparseable_arguments",
                _has_unparseable_arguments,
                config.require_parseable_arguments,
            ),
            (
                "unbalanced_cardinality",
                _has_unbalanced_cardinality,
                config.require_balanced_cardinality,
            ),
            (
                "undefined_function_calls",
                has_undefined_function_calls,
                config.require_defined_functions,
            ),
            (
                "invalid_arguments",
                has_invalid_arguments,
                config.require_valid_arguments,
            ),
        )
        if enabled
    ]

    drop_reasons: dict[str, int] = {}
    kept: list[ConversationSample] = []
    for s in samples:
        drop = False
        for key, predicate in predicates:
            if predicate(s):
                drop_reasons[key] = drop_reasons.get(key, 0) + 1
                drop = True
        if not drop:
            kept.append(s)

    if config.system_message_override is not None:
        if not isinstance(config.system_message_override, str):
            raise TypeError("system_message_override must be a string or None")
        kept = [
            _override_system_messages(s, config.system_message_override) for s in kept
        ]

    return kept, drop_reasons
