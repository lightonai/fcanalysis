"""Check the public acceptance evidence without loading a source corpus."""

import pytest

from tests.matrix import ALL_SPECS
from tests.tools.fixture_io import (
    FIXTURES_ROOT,
    config_document,
    read_config,
    read_report,
)


def test_fixture_metadata_has_exactly_the_active_matrix_scopes():
    expected = {spec.fixture_id for spec in ALL_SPECS}
    assert len(expected) == len(ALL_SPECS)
    for filename in ("config.json", "report.json"):
        actual = {
            path.parent.relative_to(FIXTURES_ROOT).as_posix()
            for path in FIXTURES_ROOT.glob(f"*/*/{filename}")
        }
        assert actual == expected


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.fixture_id)
def test_active_fixture_configuration_and_population_accounting(spec):
    assert read_config(spec.loader, spec.config_id) == config_document(
        spec.dataset_config, spec.filter_config, spec.extra_kwargs
    )
    report = read_report(spec.loader, spec.config_id)
    assert (
        report["filter_config"]
        == read_config(spec.loader, spec.config_id)["filter_config"]
    )
    counts = [
        report[key]
        for key in ("raw_count", "stage1_count", "filtered_count", "final_count")
    ]
    assert all(type(count) is int and count >= 0 for count in counts)
    assert counts == sorted(counts, reverse=True)
    assert report["raw_count"] - report["stage1_count"] == sum(
        report["stage1_drop_reasons"].values()
    )

    details = report["dataset_config_transform_counts"]
    scopes = details.get("curation", details.get("curation_by_subset"))
    if isinstance(scopes, dict):
        scopes = [scope for subset in scopes.values() for scope in subset]
    assert isinstance(scopes, list) and scopes
    assert sum(scope["output_samples"] for scope in scopes) == report["final_count"]
    for scope in scopes:
        population = scope["input_samples"]
        for level in ("level_1", "level_1_5", "level_2"):
            stage = scope["stages"][level]
            assert stage["input_samples"] == population
            assert (
                stage["input_samples"] - stage["removed_samples"]
                == stage["output_samples"]
            )
            population = stage["output_samples"]
        assert population == scope["output_samples"]
        assert scope["input_samples"] - population == scope["removed_samples"]
        for audit in scope["audit"].values():
            assert audit["eligible_samples"] <= population
            assert (
                audit["eligible_samples"] - audit["unique_groups"]
                == audit["extra_samples"]
            )
            assert (
                audit["samples_in_repeated_groups"] - audit["repeated_groups"]
                == (audit["extra_samples"])
            )

    reconstruction = details.get("reconstruction")
    if reconstruction is not None:
        records = (
            [reconstruction]
            if "input_rows" in reconstruction
            else list(reconstruction.values())
        )
        assert sum(r["validated_rows"] for r in records) == report["filtered_count"]
        assert sum(r["output_rows"] for r in records) == sum(
            scope["input_samples"] for scope in scopes
        )
        for record in records:
            assert record["input_rows"] == sum(
                record[key]
                for key in (
                    "missing_donor_rows",
                    "ambiguous_donor_rows",
                    "validation_dropped_rows",
                    "validated_rows",
                )
            )
            assert record["validated_rows"] == (
                record["removed_prefix_rows"] + record["output_rows"]
            )
            assert record["restored_rows"] <= record["validated_rows"]
            assert record["restored_rows"] <= record["restored_messages"]
