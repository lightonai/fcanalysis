"""Opt-in aggregate contract for the experimental pinned Turnstile source.

TURNSTILE_SNAPSHOT may point to an already downloaded snapshot directory. The
caller must supply the documented revision; no local snapshot identity is inferred.
"""

import os

import pytest

from fcanalysis.loaders.base import FilterConfig, apply_filters
from fcanalysis.loaders.turnstile import load


@pytest.mark.e2e
def test_turnstile_source_and_filter_contract() -> None:
    samples, report = load(path=os.environ.get("TURNSTILE_SNAPSHOT"))

    assert report.raw_count == report.stage1_count == len(samples) == 100_262
    assert report.stage1_drop_reasons == {}
    assert report.stage1_issue_counts == {}
    assert (
        sum(
            len(message.get("tool_calls", []))
            for sample in samples
            for message in sample.messages
        )
        == 277_784
    )
    assert (
        sum(
            message["role"] == "tool"
            for sample in samples
            for message in sample.messages
        )
        == 277_784
    )
    assert (
        sum(
            message["role"] == "user"
            for sample in samples
            for message in sample.messages
        )
        == 140_755
    )
    assert (
        sum(
            "reasoning_content" in message
            for sample in samples
            for message in sample.messages
        )
        == 140_755
    )

    no_call_ids = [
        sample.sample_id
        for sample in samples
        if not any(message.get("tool_calls") for message in sample.messages)
    ]
    assert no_call_ids == list(range(97_776, 100_262))
    for sample in samples:
        expected = sorted(set(sample.raw["api_names"] + sample.raw["distractors"]))
        assert [tool["function"]["name"] for tool in sample.tools] == expected

    kept, drops = apply_filters(
        samples,
        FilterConfig(
            strip_thinking=True,
            require_parseable_arguments=True,
            require_balanced_cardinality=True,
            require_defined_functions=True,
            require_valid_arguments=True,
        ),
    )
    assert len(kept) == 99_583
    assert drops == {"invalid_arguments": 679}
    assert all(
        "reasoning_content" not in message
        for sample in kept
        for message in sample.messages
    )
    assert (
        sum(
            not any(message.get("tool_calls") for message in sample.messages)
            for sample in kept
        )
        == 2_486
    )
