"""Qwen3.5 full-conversation rendering and additive token accounting.

This measurement view follows Qwen/Qwen3.5-4B's pinned text chat template,
including its tool instructions, JSON rendering, XML calls and end markers.
Unlike its inference template, it retains native reasoning on *every* assistant
message. Structured reasoning is verbatim; balanced inline native spans remain
in their original position. The shared code-aware scanner protects literal
tags. An unresolved native boundary preserves the entire content in an explicit
ambiguous category. Neither source samples nor loader training policy changes.

Content receives the stock outer whitespace trim. Structured reasoning does
not. Empty/whitespace-only reasoning wrappers count as formatting, not payload.
Assistant prompt headers are context; all assistant bodies and end delimiters
are trainable in this analysis view. Tool-call XML, names and arguments form one
category, including ordinary visible tools named ``think``.

The complete rendered string is tokenized once without padding, truncation or
extra special tokens. Categories partition character spans recorded *while*
rendering, never recovered by matching special-token strings. Tokens crossing
categories count as context if they touch context, otherwise as assistant
formatting. The crossing count is a diagnostic, not an additional category.
Boundary searches use the Qwen fast tokenizer's monotone source offsets; no
Python iteration over all token IDs or materialization of all offsets is needed.

Template source (Apache-2.0): https://huggingface.co/Qwen/Qwen3.5-4B/blob/
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja
See docs/third-party-notices.md and LICENSES/Apache-2.0.txt for attribution
and the incorporated material's license terms.
"""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Literal, Protocol

from .format import ConversationSample
from .loaders.normalization import Reject, parse_json
from .loaders.pipeline import native_reasoning_spans


QWEN35_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN35_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

TokenCategory = Literal[
    "reasoning_tokens",
    "tool_call_tokens",
    "assistant_prose_tokens",
    "assistant_formatting_tokens",
    "assistant_ambiguous_tokens",
    "context_tokens",
]


@dataclass(slots=True)
class TokenCounts:
    """Disjoint token totals and separate sample/boundary diagnostics."""

    reasoning_tokens: int = 0
    tool_call_tokens: int = 0
    assistant_prose_tokens: int = 0
    assistant_formatting_tokens: int = 0
    assistant_ambiguous_tokens: int = 0
    context_tokens: int = 0
    samples: int = 0
    ambiguous_assistant_messages: int = 0
    boundary_crossing_tokens: int = 0

    @property
    def trainable_tokens(self) -> int:
        return (
            self.reasoning_tokens
            + self.tool_call_tokens
            + self.assistant_prose_tokens
            + self.assistant_formatting_tokens
            + self.assistant_ambiguous_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.trainable_tokens + self.context_tokens

    def __add__(self, other: "TokenCounts") -> "TokenCounts":
        return TokenCounts(
            **{
                item.name: getattr(self, item.name) + getattr(other, item.name)
                for item in fields(self)
            }
        )

    def as_dict(self) -> dict[str, int]:
        return {
            **asdict(self),
            "total_tokens": self.total_tokens,
            "trainable_tokens": self.trainable_tokens,
        }


@dataclass(frozen=True, slots=True)
class RenderedSpan:
    start: int
    end: int
    category: TokenCategory


@dataclass(frozen=True, slots=True)
class RenderedConversation:
    text: str
    spans: tuple[RenderedSpan, ...]
    ambiguous_assistant_messages: int = 0


class _Encoding(Protocol):
    def __len__(self) -> int: ...

    def token_to_chars(self, token_index: int, /) -> tuple[int, int] | None: ...


class _Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[RenderedSpan] = []
        self.length = 0

    def add(self, text: str, category: TokenCategory) -> None:
        if not text:
            return
        end = self.length + len(text)
        if self.spans and self.spans[-1].category == category:
            self.spans[-1] = RenderedSpan(self.spans[-1].start, end, category)
        else:
            self.spans.append(RenderedSpan(self.length, end, category))
        self.parts.append(text)
        self.length = end


# Verbatim fixed text from the pinned template. Hugging Face overrides Jinja's
# tojson with json.dumps(ensure_ascii=False, sort_keys=False), without HTML
# escaping. Scalar call arguments instead use Jinja's Python-style str().
_TOOL_INSTRUCTIONS = """

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, allow_nan=False)


def _content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("Qwen35Counter supports canonical text content only")
    return content


def _inline_content(
    builder: _Builder, content: str, spans: tuple[tuple[int, int], ...]
) -> None:
    previous = 0
    for start, end in spans:
        before = content[previous:start]
        builder.add(
            before,
            "assistant_prose_tokens"
            if before.strip()
            else "assistant_formatting_tokens",
        )
        opening_end = content.index(">", start) + 1
        close_length = (
            len("</think>")
            if content.startswith("<think>", start)
            else len("</reasoning>")
        )
        closing_start = end - close_length
        builder.add(content[start:opening_end], "assistant_formatting_tokens")
        payload = content[opening_end:closing_start]
        builder.add(
            payload,
            "reasoning_tokens" if payload.strip() else "assistant_formatting_tokens",
        )
        builder.add(content[closing_start:end], "assistant_formatting_tokens")
        previous = end
    tail = content[previous:]
    builder.add(
        tail,
        "assistant_prose_tokens" if tail.strip() else "assistant_formatting_tokens",
    )


def _assistant(builder: _Builder, message: dict[str, Any]) -> int:
    original_content = _content(message)
    content = original_content.strip()
    # Scan before trim: stripping a first indented code line must not reinterpret
    # its literal markers as native reasoning.
    ambiguous = False
    try:
        spans = native_reasoning_spans(original_content)
    except Reject as exc:
        if exc.reason != "ambiguous_reasoning":
            raise
        spans = ()
        ambiguous = True
    left_trim = len(original_content) - len(original_content.lstrip())
    spans = tuple((start - left_trim, end - left_trim) for start, end in spans)
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("reasoning_content must be a string or null")
    builder.add("<|im_start|>assistant\n", "context_tokens")
    if isinstance(reasoning, str) or (not spans and not ambiguous):
        builder.add("<think>\n", "assistant_formatting_tokens")
        builder.add(
            reasoning or "",
            "reasoning_tokens"
            if reasoning and reasoning.strip()
            else "assistant_formatting_tokens",
        )
        builder.add("\n</think>\n\n", "assistant_formatting_tokens")
    if ambiguous:
        builder.add(content, "assistant_ambiguous_tokens")
    else:
        _inline_content(builder, content, spans)

    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be a list")
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError("tool call must be an object")
        function = call.get("function", call)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("tool call must have a string function name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = parse_json(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        # The source inline representation stays in place. In particular a
        # reasoning-only inline body is not extracted/reordered before calls.
        builder.add(
            "\n" if index else "\n\n" if content else "", "assistant_formatting_tokens"
        )
        pieces = ["<tool_call>\n<function=", function["name"], ">\n"]
        for name, value in arguments.items():
            if not isinstance(name, str):
                raise ValueError("tool call argument names must be strings")
            rendered = _json(value) if isinstance(value, (dict, list)) else str(value)
            pieces.extend(("<parameter=", name, ">\n", rendered, "\n</parameter>\n"))
        pieces.append("</function>\n</tool_call>")
        builder.add("".join(pieces), "tool_call_tokens")
    builder.add("<|im_end|>\n", "assistant_formatting_tokens")
    return int(ambiguous)


def _render(sample: ConversationSample) -> RenderedConversation:
    messages = sample.messages
    if not messages:
        raise ValueError("No messages provided")
    if not any(
        message.get("role") == "user"
        and not (
            _content(message).strip().startswith("<tool_response>")
            and _content(message).strip().endswith("</tool_response>")
        )
        for message in messages
    ):
        raise ValueError("No user query found in messages")
    builder = _Builder()
    first_system = messages[0].get("role") == "system"
    if sample.tools:
        builder.add(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
            "context_tokens",
        )
        for tool in sample.tools:
            builder.add("\n" + _json(tool), "context_tokens")
        builder.add("\n</tools>" + _TOOL_INSTRUCTIONS, "context_tokens")
        if first_system and (content := _content(messages[0]).strip()):
            builder.add("\n\n" + content, "context_tokens")
        builder.add("<|im_end|>\n", "context_tokens")
    elif first_system:
        builder.add(
            "<|im_start|>system\n" + _content(messages[0]).strip() + "<|im_end|>\n",
            "context_tokens",
        )
    ambiguous = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            if index:
                raise ValueError("System message must be at the beginning")
        elif role == "user":
            builder.add(
                "<|im_start|>user\n" + _content(message).strip() + "<|im_end|>\n",
                "context_tokens",
            )
        elif role == "assistant":
            ambiguous += _assistant(builder, message)
        elif role == "tool":
            if not index:
                raise ValueError("Tool result cannot be the first message")
            if messages[index - 1].get("role") != "tool":
                builder.add("<|im_start|>user", "context_tokens")
            builder.add(
                "\n<tool_response>\n"
                + _content(message).strip()
                + "\n</tool_response>",
                "context_tokens",
            )
            if index + 1 == len(messages) or messages[index + 1].get("role") != "tool":
                builder.add("<|im_end|>\n", "context_tokens")
        else:
            raise ValueError(f"Unexpected message role: {role!r}")
    return RenderedConversation("".join(builder.parts), tuple(builder.spans), ambiguous)


def _count(rendered: RenderedConversation, encoding: _Encoding) -> TokenCounts:
    spans = rendered.spans
    if not spans or spans[0].start != 0 or spans[-1].end != len(rendered.text):
        raise ValueError("Rendered spans must cover the complete text")
    if any(
        span.start >= span.end or (index and spans[index - 1].end != span.start)
        for index, span in enumerate(spans)
    ):
        raise ValueError("Rendered spans must form a nonempty contiguous partition")
    size = len(encoding)
    offsets: dict[int, tuple[int, int]] = {}

    def offset(index: int) -> tuple[int, int]:
        if index not in offsets:
            value = encoding.token_to_chars(index)
            if value is None or not 0 <= value[0] < value[1] <= len(rendered.text):
                raise ValueError("Tokenizer returned an unsupported source offset")
            offsets[index] = value
        return offsets[index]

    def lower_bound(position: int, *, end: bool = False) -> int:
        lo, hi = 0, size
        while lo < hi:
            mid = (lo + hi) // 2
            value = offset(mid)[int(end)]
            if value <= position if end else value < position:
                lo = mid + 1
            else:
                hi = mid
        return lo

    counts = TokenCounts(
        samples=1, ambiguous_assistant_messages=rendered.ambiguous_assistant_messages
    )
    crossings: set[int] = set()
    previous = 0
    for span in spans:
        stop = lower_bound(span.end) if span.end < len(rendered.text) else size
        setattr(counts, span.category, getattr(counts, span.category) + stop - previous)
        if span.end < len(rendered.text):
            crossings.update(range(lower_bound(span.end, end=True), stop))
        previous = stop
    starts = [span.start for span in spans]
    for index in crossings:
        start, end = offset(index)
        first = bisect_right(starts, start) - 1
        last = bisect_right(starts, end - 1) - 1
        original = spans[first].category
        categories = {span.category for span in spans[first : last + 1]}
        if len(categories) == 1:
            continue
        destination = (
            "context_tokens"
            if "context_tokens" in categories
            else "assistant_formatting_tokens"
        )
        setattr(counts, original, getattr(counts, original) - 1)
        setattr(counts, destination, getattr(counts, destination) + 1)
        counts.boundary_crossing_tokens += 1
    if counts.total_tokens != size:
        raise ValueError("Token categories do not conserve the complete token count")
    return counts


class Qwen35Counter:
    """Render the pinned Qwen3.5 text format with all historical reasoning.

    Pass the Hugging Face *fast* tokenizer loaded from the pinned assets, not
    a bare tokenizer.json: tokenizer_config.json adds further token definitions.
    The backend must retain its actual NFC normalizer. Existing backend padding
    or truncation is rejected instead of silently changing caller configuration.
    """

    def __init__(self, tokenizer: Any) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Qwen35Counter requires a Hugging Face fast tokenizer")
        self.tokenizer = tokenizer
        self.backend = tokenizer.backend_tokenizer
        if self.backend.padding is not None or self.backend.truncation is not None:
            raise ValueError("Tokenizer padding and truncation must be disabled")

    def render(self, sample: ConversationSample) -> RenderedConversation:
        return _render(sample)

    def count_rendered(
        self, rendered: Sequence[RenderedConversation]
    ) -> list[TokenCounts]:
        if not rendered:
            return []
        if self.backend.padding is not None or self.backend.truncation is not None:
            raise ValueError("Tokenizer padding and truncation must be disabled")
        encodings = self.backend.encode_batch(
            [item.text for item in rendered], add_special_tokens=False
        )
        return [
            _count(item, encoding)
            for item, encoding in zip(rendered, encodings, strict=True)
        ]

    def count_batch(self, samples: Iterable[ConversationSample]) -> list[TokenCounts]:
        return self.count_rendered([self.render(sample) for sample in samples])


def load_qwen35_counter(
    *, local_files_only: bool = True, tokenizer_path: str | Path | None = None
) -> Qwen35Counter:
    """Load tokenizer assets only; the default is a pinned, cache-only lookup."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path) if tokenizer_path is not None else QWEN35_MODEL_ID,
        revision=QWEN35_REVISION,
        local_files_only=local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    return Qwen35Counter(tokenizer)
