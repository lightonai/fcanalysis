"""Offline controls for source selection and complete export accounting."""

import importlib.util
import json
from pathlib import Path

import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.base import LoadReport
from fcanalysis.serialization import iter_samples_jsonl


@pytest.fixture
def exporter():
    path = Path(__file__).resolve().parents[2] / "scripts" / "export_selected_source.py"
    spec = importlib.util.spec_from_file_location("export_selected_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_row(uuid, source, arguments='{"x": 1}'):
    return {
        "metadata": {"uuid": uuid, "source": source},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "f", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "f", "content": "1"},
            {
                "role": "assistant",
                "content": "<think>Check the result.</think>\nAnswer",
            },
        ],
    }


def test_export_applies_selection_validation_and_curation(
    exporter, monkeypatch, tmp_path
):
    retained = source_row("kept", "glaive")
    inputs = [
        source_row(source, source) for source in ("xlam", "xlam_tools", "when2call")
    ] + [
        retained,
        source_row("duplicate", "glaive"),
        source_row("invalid", "glaive", '{"x": "wrong type"}'),
    ]
    monkeypatch.setattr(
        exporter.nemotron_agentic_v2,
        "_source_lines",
        lambda split: iter(map(json.dumps, inputs if split == "tool_calling" else [])),
    )

    output = tmp_path / "export"
    exporter.export("nemotron_v2", output, tmp_path, min_free_gib=10)

    samples = list(iter_samples_jsonl(output / "canonical.jsonl.gz"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.raw == retained
    assert sample.sample_id == "tool_calling_kept"
    assert sample.messages[-1]["content"] == retained["messages"][-1]["content"]
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] is True
    assert report["raw_count"] == 6
    assert report["filtered_count"] == 2
    assert report["final_count"] == 1
    assert report["strip_thinking_applied"] is False
    assert report["conversations"] == {sample.dataset: 1}
    assert report["assistant_decisions"] == {sample.dataset: 2}


@pytest.mark.parametrize("failure", ["interrupted", "incomplete_report", "wrong_count"])
def test_failed_export_has_no_complete_report(exporter, monkeypatch, tmp_path, failure):
    sample = ConversationSample(
        dataset="example", sample_id="one", messages=[], tools=[]
    )
    report = LoadReport(dataset="example", raw_count=1, stage1_count=1)
    closed = []

    def rows():
        try:
            yield sample
            if failure == "interrupted":
                raise OSError("source interrupted after its last row")
            if failure == "wrong_count":
                report.final_count = 2
        finally:
            closed.append(True)

    monkeypatch.setattr(
        exporter, "selected_source", lambda *args, **kwargs: (rows(), report)
    )
    output = tmp_path / "export"
    error = OSError if failure == "interrupted" else ValueError
    with pytest.raises(error, match="source interrupted|conversation count differ"):
        exporter.export("dolci", output, tmp_path, min_free_gib=10)

    assert closed == [True]
    assert not (output / "report.json").exists()
    assert list(iter_samples_jsonl(output / "canonical.jsonl.gz")) == [sample]


def test_existing_output_is_preserved(exporter, monkeypatch, tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    existing = output / "canonical.jsonl.gz"
    existing.write_bytes(b"existing export")
    monkeypatch.setattr(
        exporter,
        "selected_source",
        lambda *args, **kwargs: pytest.fail("loader started"),
    )

    with pytest.raises(FileExistsError):
        exporter.export("dolci", output, tmp_path, min_free_gib=10)
    assert existing.read_bytes() == b"existing export"
