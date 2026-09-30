"""Regression tests for fixture completion and forced-rewrite semantics."""

from pathlib import Path
from typing import Any

import pytest

from fcanalysis.loaders.base import LoadReport
from fcanalysis.loaders.normalization import Reject
from tests.tools import fixture_io


def test_fixture_serialization_preserves_large_integer_definitions(
    tmp_path, monkeypatch
):
    import gzip
    import json
    from tests.helpers import sample
    from tests.tools.hash_jsonl import hash_samples

    row = sample()
    row.tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"const": 51090942171709440000}},
                },
            },
        }
    ]
    report = LoadReport(dataset="test", raw_count=1, stage1_count=1, final_count=1)
    fixture_io.write_fixture(tmp_path, None, None, {}, report, iter([row]))
    with gzip.open(tmp_path / "output.jsonl.gz", "rb") as stream:
        payload = json.loads(stream.read())
    assert payload["tools"] == row.tools
    assert (tmp_path / "output.hash").read_text().strip() == hash_samples([row])
    monkeypatch.setattr(fixture_io, "fixture_dir", lambda *args: tmp_path)
    for restored in (
        fixture_io.read_sample_subset("test", "prod")[0],
        next(iter(fixture_io.iter_full_output("test", "prod"))),
    ):
        assert restored["tools"] == row.tools
        value = restored["tools"][0]["function"]["parameters"]["properties"]["n"][
            "const"
        ]
        assert type(value) is int


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_fixture_readers_reject_nonfinite_json(tmp_path, monkeypatch, value):
    import gzip

    line = ('{"value":' + value + "}\n").encode()
    (tmp_path / "sample.jsonl").write_bytes(line)
    with gzip.open(tmp_path / "output.jsonl.gz", "wb") as stream:
        stream.write(line)
    monkeypatch.setattr(fixture_io, "fixture_dir", lambda *args: tmp_path)
    with pytest.raises(Reject):
        fixture_io.read_sample_subset("test", "prod")
    with pytest.raises(Reject):
        next(iter(fixture_io.iter_full_output("test", "prod")))


def test_forced_rewrite_invalidates_old_hash_before_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "fixture"
    out_dir.mkdir()
    final_hash = out_dir / "output.hash"
    temporary_hash = out_dir / "output.hash.tmp"
    final_hash.write_text("old-complete-hash\n")
    temporary_hash.write_text("stale-temporary-hash\n")

    def fail_output_write(*_args: Any, **_kwargs: Any) -> None:
        assert not final_hash.exists()
        assert not temporary_hash.exists()
        raise RuntimeError("simulated interrupted forced rewrite")

    monkeypatch.setattr(fixture_io, "_write_full_output", fail_output_write)

    with pytest.raises(RuntimeError, match="simulated interrupted forced rewrite"):
        fixture_io.write_fixture(
            out_dir=out_dir,
            dataset_config=None,
            filter_config=None,
            extra_kwargs={},
            report=LoadReport(dataset="test", raw_count=0, stage1_count=0),
            samples=[],
        )

    assert not final_hash.exists()
    assert not temporary_hash.exists()


def test_streaming_fixture_finalizes_report_before_writing(tmp_path):
    import gzip
    import json
    from tests.helpers import sample

    report = LoadReport(dataset="test", raw_count=3, stage1_count=3)

    def rows():
        for index in range(3):
            yield sample(sample_id=str(index))
        report.final_count = 3

    fixture_io.write_fixture(tmp_path, None, None, {}, report, rows())
    assert json.loads((tmp_path / "report.json").read_text())["final_count"] == 3
    with gzip.open(tmp_path / "output.jsonl.gz", "rb") as f:
        assert f.read() == (tmp_path / "sample.jsonl").read_bytes()
    assert (tmp_path / "output.hash").exists()


def test_loader_failure_invalidates_old_completion_marker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.tools import generate_fixtures
    from tests.matrix import DOLCI_SPECS

    marker = tmp_path / "output.hash"
    marker.write_text("old")

    def fail_load(**kwargs):
        assert not marker.exists()
        raise RuntimeError("load interrupted")

    monkeypatch.setattr(generate_fixtures, "fixture_dir", lambda *a: tmp_path)
    monkeypatch.setattr(
        generate_fixtures.importlib,
        "import_module",
        lambda *a: SimpleNamespace(iter_load=fail_load),
    )
    with pytest.raises(RuntimeError, match="load interrupted"):
        generate_fixtures.run_one(DOLCI_SPECS[0], force=True)
    assert not marker.exists()
