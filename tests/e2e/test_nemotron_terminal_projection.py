"""Fresh source projection and protocol check of every retained terminal sample.

Replays the loader and its fixture, then reads all pinned rows independently.
Does not call the converter, native-envelope decoder, source gates or curation
comparison implementation to establish the projected values. The attributed
upstream parser is reused for reference execution/completion interpretation;
this is not an independent proof of every rejection or deduplication decision.
"""

from dataclasses import asdict
import re

import pytest

from fcanalysis.loaders import nemotron_terminal as loader
from fcanalysis.loaders._terminus_json import TerminusJSONPlainParser
from fcanalysis.loaders.source import parquet_rows
from tests.matrix import NEMOTRON_TERMINAL_SPECS
from tests.tools.fixture_io import read_hash, read_report
from tests.tools.hash_jsonl import hash_samples

_REFERENCE = TerminusJSONPlainParser()
_CONFIRM = (
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)


def project(raw, strip):
    result = []
    for original in raw["conversations"]:
        message = dict(original)
        text = original["content"]
        if original["role"] == "assistant" and text.startswith("<think>"):
            positions = [m.start() for m in re.finditer(r"</think>(?=\n|$)", text)]
            assert len(positions) == 1
            end = positions[0]
            thinking = text[7:end]
            assert thinking
            tail = text[end + 8 :]
            assert not tail or tail.startswith("\n")
            message["content"] = tail[1:] if tail else ""
            if not strip:
                message["reasoning_content"] = thinking
        result.append(message)
    return result


def reference_protocol(messages):
    pending = False
    for index in range(1, len(messages), 2):
        response = _REFERENCE.parse_response(messages[index]["content"])
        feedback = "ERROR: " + response.error if response.error else ""
        if response.warning:
            feedback += "\nWARNINGS: " + response.warning
        rejected = "ERROR:" in feedback
        if index == len(messages) - 1:
            assert not rejected and pending and response.is_task_complete
            assert response.commands == []
            return
        observation = messages[index + 1]["content"]
        if rejected:
            assert observation.startswith(
                "Previous response had parsing errors:\nERROR: "
            )
            assert observation.endswith(
                "Please fix these issues and provide a proper JSON response."
            )
            continue
        if response.is_task_complete:
            assert not pending
            assert observation.startswith("Current terminal state:\n")
            assert observation.endswith("\n\n" + _CONFIRM)
            pending = True
        else:
            pending = False
            if response.warning:
                assert observation.startswith(
                    "Previous response had warnings:\nWARNINGS: "
                )
                observation = observation.split("\n\n", 1)[1]
            assert observation.startswith(
                ("New Terminal Output:\n", "Current Terminal Screen:\n")
            )
    raise AssertionError("No accepted source endpoint")


@pytest.mark.e2e
@pytest.mark.parametrize("spec", NEMOTRON_TERMINAL_SPECS, ids=lambda s: s.config_id)
def test_every_retained_terminal_sample_matches_raw_source_and_fixture(spec):
    stream, report = loader.iter_load(
        spec.dataset_config, spec.filter_config, **spec.extra_kwargs
    )

    def source():
        for config, files in loader._FILES.items():
            for raw in parquet_rows(
                loader.DATASET_ID, loader.DATASET_REVISION, files, batch_size=64
            ):
                yield config, raw

    raw_stream = source()
    physical = retained = 0

    def checked():
        nonlocal physical, retained
        for sample in stream:
            for config, raw in raw_stream:
                physical += 1
                source_id = raw["run_id"] + "/" + raw["trial_name"]
                if source_id == sample.sample_id:
                    break
            else:
                pytest.fail("Missing or out-of-order source sample")
            assert sample.dataset == loader.DATASET_ID + "/" + config
            assert sample.raw == raw
            assert sample.annotations == {}
            assert sample.tools == []
            messages = project(raw, spec.filter_config.strip_thinking)
            assert messages == sample.messages, sample.sample_id
            assert all(
                m["content"].strip() or m.get("reasoning_content", "").strip()
                for m in messages
                if m["role"] == "assistant"
            )
            reference_protocol(messages)
            retained += 1
            yield sample

    try:
        output_hash = hash_samples(checked())
        for _ in raw_stream:
            physical += 1
    finally:
        stream.close()
        raw_stream.close()
    assert physical == report.raw_count == 366154
    assert retained == report.final_count and retained > 0
    assert asdict(report) == read_report(spec.loader, spec.config_id)
    assert output_hash == read_hash(spec.loader, spec.config_id)
