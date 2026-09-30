"""Existing CLI dispatchers must use supported migrated loader contracts."""

import pytest
from datasets import Dataset

from fcanalysis.loaders import (
    apigen_mt,
    dolci,
    nemotron_agentic_v1,
    nemotron_agentic_v2,
    toolmind,
    txt360,
)
from fcanalysis.overlap import _load_dataset
from fcanalysis.semantic import _load_samples


DISPATCHED_MIGRATIONS = [
    "dolci",
    "apigen_mt",
    "nemotron_agentic_v1",
    "nemotron_agentic_v2",
    "toolmind",
    "txt360",
]


@pytest.fixture
def empty_migrated_sources(monkeypatch):
    monkeypatch.setattr(dolci, "parquet_rows", lambda *args, **kwargs: iter(()))
    empty = Dataset.from_dict({"system": [], "tools": [], "conversations": []})
    monkeypatch.setattr(apigen_mt, "load_dataset", lambda *args, **kwargs: empty)
    for module in (nemotron_agentic_v1, nemotron_agentic_v2, toolmind):
        monkeypatch.setattr(module, "jsonl_lines", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(txt360, "parquet_rows", lambda *args, **kwargs: iter(()))


@pytest.mark.parametrize("name", DISPATCHED_MIGRATIONS)
def test_overlap_dispatch_uses_supported_loader_policy(empty_migrated_sources, name):
    rows, report = _load_dataset(name)
    assert rows == []
    assert report.final_count == 0


@pytest.mark.parametrize("name", DISPATCHED_MIGRATIONS)
@pytest.mark.parametrize("mode", ["all", "dataset", "universal", "none"])
@pytest.mark.parametrize("strip", [True, False])
def test_semantic_dispatch_preserves_reasoning_choice_in_every_mode(
    empty_migrated_sources, name, mode, strip
):
    rows, report = _load_samples(name, "", mode, strip_thinking=strip)
    assert rows == []
    assert report.final_count == 0
    assert report.strip_thinking_applied is strip
