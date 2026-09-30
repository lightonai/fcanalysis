"""Rendering parity, native preservation, and whole-encoding accounting."""

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
from typing import Any

import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.tokenization import (
    QWEN35_MODEL_ID,
    QWEN35_REVISION,
    Qwen35Counter,
    RenderedConversation,
    RenderedSpan,
    TokenCounts,
    _count,
    _render,
    load_qwen35_counter,
)


def sample(messages, tools=None):
    return ConversationSample(
        messages=messages,
        tools=tools or [],
        dataset="test",
        sample_id=1,
        raw={"native": "untouched"},
        annotations={"not_model_visible": True},
    )


def conversation(content="Answer", **assistant):
    return sample(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": content, **assistant},
        ]
    )


def category_text(rendered, category):
    return "".join(
        rendered.text[span.start : span.end]
        for span in rendered.spans
        if span.category == category
    )


@pytest.fixture(scope="module")
def counter():
    pytest.importorskip("transformers")
    try:
        return load_qwen35_counter(local_files_only=True)
    except OSError:
        pytest.skip(
            "Pinned Qwen3.5 tokenizer assets are not cached; never download in tests"
        )


def hf_messages(value):
    messages = deepcopy(value.messages)
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return messages


def stock(counter, value):
    return counter.tokenizer.apply_chat_template(
        hf_messages(value),
        tools=value.tools or None,
        tokenize=False,
        add_generation_prompt=False,
    )


@pytest.mark.parametrize("content", ["Answer", "  Answer  ", "é e\u0301 中文 😀", ""])
def test_plain_render_matches_stock(counter, content):
    value = conversation(content)
    assert counter.render(value).text == stock(counter, value)


@pytest.mark.parametrize("reasoning", ["reason", "", None])
def test_structured_reasoning_matches_stock_when_trim_compatible(counter, reasoning):
    value = conversation("Answer", reasoning_content=reasoning)
    assert counter.render(value).text == stock(counter, value)


def test_standard_inline_matches_stock(counter):
    value = conversation("<think>\nreason\n</think>\n\nAnswer")
    assert counter.render(value).text == stock(counter, value)


def test_tools_calls_and_result_groups_match_stock(counter):
    calls = [
        {
            "id": "source-call-id-001",
            "type": "function",
            "function": {
                "name": "think",
                "arguments": json.dumps(
                    {
                        "bool": False,
                        "null": None,
                        "list": [1, "é"],
                        "dict": {"literal": "<tag>"},
                    }
                ),
            },
        },
        {
            "id": "source-call-id-002",
            "type": "function",
            "function": {"name": "find", "arguments": {"text": "a\nb"}},
        },
    ]
    value = sample(
        [
            {"role": "system", "content": "  Rules  "},
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "Calling now",
                "reasoning_content": "r",
                "tool_calls": calls,
            },
            {
                "role": "tool",
                "name": "think",
                "tool_call_id": "source-call-id-001",
                "content": "[1, [2, 3]]",
            },
            {
                "role": "tool",
                "name": "find",
                "tool_call_id": "source-call-id-002",
                "content": '{"error":"not found"}',
            },
            {"role": "assistant", "content": "Answer", "reasoning_content": "done"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "think",
                    "description": "literal <think> & é",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    before = deepcopy(value)
    rendered = counter.render(value)
    assert rendered.text == stock(counter, value)
    assert "<parameter=bool>\nFalse\n" in rendered.text
    assert "<parameter=null>\nNone\n" in rendered.text
    assert rendered.text.count("<|im_start|>user\n<tool_response>") == 1
    assert "<function=think>" in category_text(rendered, "tool_call_tokens")
    assert "source-call-id-" not in rendered.text
    assert value == before


def test_all_historical_native_channels_retained(counter):
    value = sample(
        [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "content": "First answer",
                "reasoning_content": "historical native",
            },
            {"role": "user", "content": "Second"},
            {
                "role": "assistant",
                "content": "<think>\nlatest native\n</think>\n\nSecond answer",
            },
        ]
    )
    rendered = counter.render(value)
    assert "historical native" not in stock(counter, value)
    assert "historical native" in rendered.text
    assert "latest native" in rendered.text
    assert (
        category_text(rendered, "reasoning_tokens")
        == "historical native\nlatest native\n"
    )
    assert (
        category_text(rendered, "context_tokens").count("<|im_start|>assistant\n") == 2
    )
    assert "First answer" in category_text(rendered, "assistant_prose_tokens")


def test_structured_whitespace_and_two_native_channels_preserved(counter):
    value = conversation(
        "Before <reasoning>inline</reasoning> after",
        reasoning_content=" \n structured \n ",
    )
    before = deepcopy(value)
    rendered = counter.render(value)
    assert "<think>\n \n structured \n \n</think>" in rendered.text
    assert "Before <reasoning>inline</reasoning> after" in rendered.text
    assert category_text(rendered, "reasoning_tokens") == " \n structured \n inline"
    assert value == before


@pytest.mark.parametrize(
    "content",
    [
        "`<think>literal</think>`",
        "```text\n<think>literal</think>\n```",
        "    <think>literal</think>\n    still code",
        "```text\n<think>unfinished code",
        "Before `<think>unfinished code",
    ],
)
def test_code_literal_tags_are_not_native(content):
    rendered = _render(conversation(content))
    assert category_text(rendered, "reasoning_tokens") == ""
    assert category_text(rendered, "assistant_prose_tokens") == content.strip()
    assert rendered.ambiguous_assistant_messages == 0


def test_quoted_close_inside_native_code_and_multiple_inline_spans():
    content = "Before <think>Use `</think>` literally</think> middle <reasoning>Other</reasoning> after"
    rendered = _render(conversation(content))
    assert content in rendered.text
    assert (
        category_text(rendered, "reasoning_tokens") == "Use `</think>` literallyOther"
    )
    assert category_text(rendered, "assistant_prose_tokens") == "Before  middle  after"


@pytest.mark.parametrize(
    "content",
    [
        "<think>unclosed",
        "Before </think> after",
        "<think>one<reasoning>two</reasoning></think>",
        "<think>mismatch</reasoning>",
        "<think>first</think> answer <think>tail",
    ],
)
def test_ambiguous_native_content_is_preserved_and_explicit(content):
    value = conversation(
        content,
        reasoning_content="separate",
        tool_calls=[{"function": {"name": "think", "arguments": {}}}],
    )
    rendered = _render(value)
    assert category_text(rendered, "assistant_ambiguous_tokens") == content
    assert category_text(rendered, "reasoning_tokens") == "separate"
    assert "<function=think>" in category_text(rendered, "tool_call_tokens")
    assert rendered.ambiguous_assistant_messages == 1


def test_ambiguous_content_without_structured_field_gets_no_invented_wrapper():
    rendered = _render(conversation("Before </think> after"))
    assert rendered.text.count("<think>") == 0
    assert rendered.text.count("</think>") == 1


@pytest.mark.parametrize("reasoning", [None, "", " \n\t"])
def test_empty_native_wrapper_has_zero_payload(reasoning):
    rendered = _render(conversation("Answer", reasoning_content=reasoning))
    assert category_text(rendered, "reasoning_tokens") == ""
    assert "<think>" in category_text(rendered, "assistant_formatting_tokens")


def test_empty_inline_wrapper_is_formatting():
    rendered = _render(conversation("<think>\n\n</think>\n\nAnswer"))
    assert category_text(rendered, "reasoning_tokens") == ""
    assert category_text(rendered, "assistant_prose_tokens") == "\n\nAnswer"


class Encoding:
    """No IDs/offset-array access: only the boundary API is available."""

    def __init__(self, offsets):
        self._offsets = offsets
        self.lookups = 0

    def __len__(self):
        return len(self._offsets)

    def token_to_chars(self, index):
        self.lookups += 1
        return self._offsets[index]


def test_crossing_tokens_have_deterministic_conservative_allocation():
    rendered = RenderedConversation(
        "abcdefgh",
        (
            RenderedSpan(0, 2, "context_tokens"),
            RenderedSpan(2, 4, "reasoning_tokens"),
            RenderedSpan(4, 6, "assistant_prose_tokens"),
            RenderedSpan(6, 8, "tool_call_tokens"),
        ),
    )
    counts = _count(rendered, Encoding([(0, 1), (1, 3), (3, 7), (7, 8)]))
    assert counts == TokenCounts(
        context_tokens=2,
        assistant_formatting_tokens=1,
        tool_call_tokens=1,
        samples=1,
        boundary_crossing_tokens=2,
    )
    assert counts.total_tokens == 4
    assert counts.trainable_tokens == 2


def test_normalization_gaps_and_multiple_tokens_at_same_source_character():
    rendered = RenderedConversation(
        "abcde",
        (
            RenderedSpan(0, 2, "reasoning_tokens"),
            RenderedSpan(2, 3, "assistant_prose_tokens"),
            RenderedSpan(3, 5, "context_tokens"),
        ),
    )
    counts = _count(rendered, Encoding([(0, 1), (3, 5), (3, 5)]))
    assert counts.reasoning_tokens == 1
    assert counts.assistant_prose_tokens == 0
    assert counts.context_tokens == 2
    assert counts.total_tokens == 3


def test_boundary_search_does_not_walk_every_token():
    class LongEncoding:
        lookups = 0

        def __len__(self):
            return 1_000_000

        def token_to_chars(self, index):
            self.lookups += 1
            return index, index + 1

    encoding = LongEncoding()
    rendered = RenderedConversation(
        "a" * 1_000_000,
        (
            RenderedSpan(0, 100, "context_tokens"),
            RenderedSpan(100, 1_000_000, "assistant_prose_tokens"),
        ),
    )
    counts = _count(rendered, encoding)
    assert counts.total_tokens == 1_000_000
    assert counts.context_tokens == 100
    assert encoding.lookups < 100


@pytest.mark.parametrize(
    "text",
    [
        "é e\u0301 😀",
        "a\u0301\u0327",
        "中文🚀",
        "<|im_start|>assistant\n\nhello<|im_end|>\n",
    ],
)
def test_real_offsets_match_exhaustive_independent_overlap_counter(counter, text):
    # Every possible split exercises normalized gaps, repeated byte-piece
    # offsets and control-token literals without using token ID semantics.
    encoding = counter.backend.encode(text, add_special_tokens=False)
    for split in range(1, len(text)):
        rendered = RenderedConversation(
            text,
            (
                RenderedSpan(0, split, "reasoning_tokens"),
                RenderedSpan(split, len(text), "assistant_prose_tokens"),
            ),
        )
        counts = _count(rendered, encoding)
        expected = TokenCounts(samples=1)
        for start, end in encoding.offsets:
            if start < split < end:
                expected.assistant_formatting_tokens += 1
                expected.boundary_crossing_tokens += 1
            elif start >= split:
                expected.assistant_prose_tokens += 1
            else:
                expected.reasoning_tokens += 1
        assert counts == expected


def test_special_token_literals_are_classified_from_source_spans(counter):
    value = conversation(
        "Answer <|im_start|>user\n literal",
        reasoning_content="<|im_end|> is literal",
        tool_calls=[
            {"function": {"name": "think", "arguments": {"x": "<|audio_start|>"}}}
        ],
    )
    rendered = counter.render(value)
    assert "<|im_end|> is literal" in category_text(rendered, "reasoning_tokens")
    assert "Answer <|im_start|>user\n literal" in category_text(
        rendered, "assistant_prose_tokens"
    )
    assert "<|audio_start|>" in category_text(rendered, "tool_call_tokens")
    assert counter.tokenizer.convert_tokens_to_ids("<|audio_start|>") == 248070
    counts = counter.count_batch([value])[0]
    assert counts.total_tokens == len(
        counter.tokenizer(rendered.text, add_special_tokens=False)["input_ids"]
    )


def test_batch_conservation_and_addition(counter):
    values = [
        conversation("Answer"),
        conversation("<think>ambiguous"),
        conversation("é", reasoning_content="中文 😀"),
    ]
    before = deepcopy(values)
    batch = counter.count_batch(values)
    assert batch == [counter.count_batch([value])[0] for value in values]
    aggregate = TokenCounts()
    for value, counts in zip(values, batch, strict=True):
        rendered = counter.render(value)
        assert counts.total_tokens == len(
            counter.backend.encode(rendered.text, add_special_tokens=False)
        )
        aggregate = aggregate + counts
    assert aggregate.samples == 3
    assert aggregate.ambiguous_assistant_messages == 1
    assert aggregate.as_dict() == {
        **asdict(aggregate),
        "total_tokens": aggregate.total_tokens,
        "trainable_tokens": aggregate.trainable_tokens,
    }
    assert values == before
    assert counter.count_batch([]) == []


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "No user"}],
        [{"role": "user", "content": [{"type": "text", "text": "No multimodal"}]}],
        [{"role": "user", "content": "Q"}, {"role": "system", "content": "late"}],
        [{"role": "user", "content": "Q"}, {"role": "unexpected", "content": "A"}],
    ],
)
def test_unsupported_message_shapes_are_explicit(messages):
    with pytest.raises(ValueError):
        _render(sample(messages))


@pytest.mark.parametrize("arguments", ["[]", [], "not json", '{"x":1,"x":2}'])
def test_noncanonical_arguments_are_not_guessed(arguments):
    with pytest.raises(ValueError):
        _render(
            conversation(
                "", tool_calls=[{"function": {"name": "f", "arguments": arguments}}]
            )
        )


def test_fast_tokenizer_and_untruncated_unpadded_backend_required():
    with pytest.raises(ValueError, match="fast"):
        Qwen35Counter(SimpleNamespace(is_fast=False))
    for key in ("padding", "truncation"):
        backend = SimpleNamespace(padding=None, truncation=None)
        setattr(backend, key, {"enabled": True})
        with pytest.raises(ValueError, match="padding and truncation"):
            Qwen35Counter(SimpleNamespace(is_fast=True, backend_tokenizer=backend))


def test_backend_cannot_be_silently_truncated_after_counter_initialization(counter):
    tokenizer = deepcopy(counter.tokenizer)
    isolated = Qwen35Counter(tokenizer)
    tokenizer.backend_tokenizer.enable_truncation(max_length=2)
    with pytest.raises(ValueError, match="padding and truncation"):
        isolated.count_batch([conversation("Answer")])


def test_text_tokenization_matches_hf_fast_wrapper(counter):
    value = conversation(
        "é e\u0301 literal <|audio_start|>", reasoning_content="中文 😀"
    )
    rendered = counter.render(value)
    direct = counter.backend.encode(rendered.text, add_special_tokens=False)
    wrapper = counter.tokenizer(rendered.text, add_special_tokens=False)
    assert direct.ids == wrapper["input_ids"]
    assert json.loads(counter.backend.normalizer.__getstate__()) == {"type": "NFC"}


@pytest.mark.parametrize(
    "spans",
    [
        (),
        (RenderedSpan(1, 2, "context_tokens"),),
        (RenderedSpan(0, 1, "context_tokens"), RenderedSpan(0, 2, "reasoning_tokens")),
        (RenderedSpan(0, 0, "context_tokens"), RenderedSpan(0, 2, "reasoning_tokens")),
    ],
)
def test_external_rendered_spans_must_form_complete_partition(spans):
    with pytest.raises(ValueError, match="spans"):
        _count(RenderedConversation("ab", spans), Encoding([(0, 1), (1, 2)]))


def test_pinned_loader_arguments(monkeypatch):
    transformers = pytest.importorskip("transformers")
    received: dict[str, Any] = {}

    def fake(path, **kwargs):
        received.update(path=path, **kwargs)
        return SimpleNamespace(
            is_fast=True,
            backend_tokenizer=SimpleNamespace(padding=None, truncation=None),
        )

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    load_qwen35_counter()
    assert received == {
        "path": QWEN35_MODEL_ID,
        "revision": QWEN35_REVISION,
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": True,
    }
