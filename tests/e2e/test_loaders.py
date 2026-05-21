"""End-to-end regression: every loader×config produces fixture-identical output.

For each FixtureSpec in tests.matrix:
1. Re-run the loader with the spec's config.
2. Assert sample count, LoadReport, and full-content SHA-256 hash match
   the on-disk fixture.
3. If the locally-generated `sample.jsonl` is present, also assert the
   deterministic 100-sample subset matches it (debugging aid).

Any mismatch is a behavioral regression. If a fixture is missing,
the test skips with instructions to run the generate_fixtures tool.

These tests are marked `e2e` (slow). Run with `pytest -m e2e` or target
a single loader with `pytest -k dolci`.
"""

import dataclasses
import importlib

import pytest

from tests.matrix import ALL_SPECS, FixtureSpec, LOADER_MODULES
from tests.tools.fixture_io import (
    SAMPLE_SUBSET_SIZE,
    deterministic_subset,
    fixture_dir,
    read_hash,
    read_report,
    read_sample_subset,
)
from tests.tools.hash_jsonl import hash_samples, sample_payload


@pytest.mark.e2e
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.fixture_id)
def test_loader_output_matches_fixture(spec: FixtureSpec) -> None:
    fdir = fixture_dir(spec.loader, spec.config_id)
    if not (fdir / "output.hash").exists():
        pytest.skip(
            f"Fixture missing for {spec.fixture_id}. Generate with: "
            f"python -m tests.tools.generate_fixtures --fixture {spec.fixture_id}"
        )

    module = importlib.import_module(LOADER_MODULES[spec.loader])
    samples, report = module.load(
        dataset_config=spec.dataset_config,
        filter_config=spec.filter_config,
        **spec.extra_kwargs,
    )

    stored_report = read_report(spec.loader, spec.config_id)
    stored_hash = read_hash(spec.loader, spec.config_id)

    expected_count = stored_report.get("filtered_count")
    if expected_count is None:
        expected_count = stored_report.get("dataset_config_count")
    if expected_count is None:
        expected_count = stored_report["stage1_count"]
    assert len(samples) == expected_count, (
        f"sample count diverged: got {len(samples)}, fixture {expected_count}"
    )

    assert dataclasses.asdict(report) == stored_report, (
        "LoadReport diverged from fixture"
    )

    assert hash_samples(samples) == stored_hash, (
        "full-output SHA-256 diverged from fixture"
    )

    if (fdir / "sample.jsonl").exists():
        stored_subset = read_sample_subset(spec.loader, spec.config_id)
        actual_subset_payloads = [
            sample_payload(s) for s in deterministic_subset(samples, SAMPLE_SUBSET_SIZE)
        ]
        assert actual_subset_payloads == stored_subset, (
            "deterministic 100-sample subset diverged from fixture"
        )
