"""Regression tests for fixture completion and forced-rewrite semantics."""

from pathlib import Path
from typing import Any

import pytest

from fcanalysis.loaders.base import LoadReport
from tests.tools import fixture_io


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
