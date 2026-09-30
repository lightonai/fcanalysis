"""Canonical artifact reading preserves values and fails without silent loss."""

from dataclasses import asdict
import gzip
import io
import json
from pathlib import Path

import pytest

from fcanalysis.loaders.normalization import json_bytes
from fcanalysis.serialization import iter_samples_jsonl


def _payload():
    return {
        "messages": [
            {"role": "user", "content": "  café\nsecond line\t"},
            {
                "role": "assistant",
                "content": "<think>native</think>answer",
                "reasoning_content": "  exact reasoning\n",
                "source_extension": {"values": [1, 1.0, True, None, 10**80]},
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{ "id": 1, "other": [2, 1] }',
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"const": 10**80, "additionalProperties": False},
                },
                "extra": {"ordered": [2, 1]},
            }
        ],
        "dataset": "source/scope",
        "sample_id": 10**90,
    }


@pytest.mark.parametrize("compressed", [False, True])
def test_reader_preserves_all_serialized_fields_types_and_order(tmp_path, compressed):
    first = _payload()
    first["annotations"] = {"value": 1.0, "nested": {"keep": True}}
    first["raw"] = {"nested": [10**85, 1.0, 1, True], "text": " original\n"}
    second = {**_payload(), "sample_id": "same-source-id"}
    third = {**_payload(), "sample_id": "same-source-id"}
    path = tmp_path / ("samples.jsonl.gz" if compressed else "samples.jsonl")
    data = b"\n".join(json_bytes(row) for row in (first, second, third))
    path.write_bytes(gzip.compress(data) if compressed else data)

    samples = list(iter_samples_jsonl(str(path)))
    assert len(samples) == 3
    for actual, expected in zip(samples, (first, second, third), strict=True):
        assert json_bytes(asdict(actual)) == json_bytes(
            {"annotations": {}, "raw": {}, **expected}
        )
    assert samples[1].raw is not samples[2].raw
    assert samples[1].annotations is not samples[2].annotations
    samples[1].tools[0]["extra"]["ordered"].append(3)
    assert samples[2].tools[0]["extra"]["ordered"] == [2, 1]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("messages", {}, "messages must be an array of objects"),
        ("messages", [None], "messages must be an array of objects"),
        ("tools", None, "tools must be an array of objects"),
        ("tools", ["tool"], "tools must be an array of objects"),
        ("dataset", 5, "dataset must be a string"),
        ("sample_id", True, "sample_id must be a string or integer"),
        ("sample_id", 1.0, "sample_id must be a string or integer"),
        ("sample_id", None, "sample_id must be a string or integer"),
        ("annotations", [], "annotations must be an object"),
        ("raw", None, "raw must be an object"),
        ("unknown", "preserve me", "unknown sample fields: unknown"),
    ],
)
def test_reader_rejects_incompatible_root_shape(tmp_path, field, value, reason):
    path = tmp_path / "samples.jsonl"
    path.write_bytes(json_bytes({**_payload(), field: value}))
    with pytest.raises(ValueError, match=reason):
        list(iter_samples_jsonl(path))


@pytest.mark.parametrize("field", ["messages", "tools", "dataset", "sample_id"])
def test_reader_requires_canonical_root_fields(tmp_path, field):
    row = _payload()
    del row[field]
    path = tmp_path / "samples.jsonl"
    path.write_bytes(json_bytes(row))
    with pytest.raises(ValueError, match=f"missing sample fields: {field}"):
        list(iter_samples_jsonl(path))


@pytest.mark.parametrize(
    "bad_line",
    [
        b"\n",
        b" \t\r\n",
        b"[]\n",
        b"null\n",
        b'{"broken":\n',
        b'{"messages": [], "messages": []}\n',
        b'{"raw": {"nested": 1, "nested": 2}}\n',
        b'{"raw": NaN}\n',
        b'{"raw": Infinity}\n',
        b'{"raw": 1e999}\n',
        b'{"dataset": "\xff"}\n',
    ],
)
def test_reader_reports_bad_row_path_and_line(tmp_path, bad_line):
    path = tmp_path / "samples.jsonl"
    path.write_bytes(json_bytes(_payload()) + b"\n" + bad_line)
    rows = iter_samples_jsonl(path)
    assert next(rows).dataset == "source/scope"
    with pytest.raises(ValueError) as raised:
        next(rows)
    assert str(raised.value).startswith(f"{path}:2: ")


@pytest.mark.parametrize("compressed", [False, True])
def test_reader_supports_empty_files(tmp_path, compressed):
    path = tmp_path / ("empty.jsonl.gz" if compressed else "empty.jsonl")
    path.write_bytes(gzip.compress(b"") if compressed else b"")
    assert list(iter_samples_jsonl(path)) == []


@pytest.mark.parametrize("ending", ["exhaust", "error", "close"])
def test_reader_is_lazy_and_closes_owned_stream(monkeypatch, ending):
    opened = []
    stream = io.BytesIO(
        json_bytes(_payload()) + b"\n" + (b"bad\n" if ending == "error" else b"")
    )

    def open_stream(*args):
        opened.append(args)
        return stream

    monkeypatch.setattr("fcanalysis.serialization.gzip.open", open_stream)
    rows = iter_samples_jsonl(Path("example.jsonl.gz"))
    assert opened == []
    next(rows)
    assert not stream.closed
    if ending == "error":
        with pytest.raises(ValueError):
            next(rows)
    elif ending == "exhaust":
        with pytest.raises(StopIteration):
            next(rows)
    else:
        rows.close()
    assert stream.closed


def test_reader_does_not_claim_loader_validation(tmp_path):
    row = _payload()
    row["messages"] = [{"role": "unvalidated", "extra": [False, 0]}]
    row["tools"] = [{"unvalidated": True}]
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(row))
    sample = next(iter_samples_jsonl(path))
    assert sample.messages == row["messages"]
    assert sample.tools == row["tools"]
