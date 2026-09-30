"""Unit tests for fcanalysis.statistics.

P1 coverage of the most-used aggregator functions: compute_basic_stats,
compute_percentiles, compute_dataset_overview, compute_token_length_statistics.
The higher-level compute_* functions (calling_patterns, parallel_diversity,
turn_structure, etc.) compose these primitives and are best exercised
through the loader E2E fixtures and Phase 4 integration tests.
"""

import math

from fcanalysis.statistics import (
    compute_basic_stats,
    compute_dataset_overview,
    compute_percentiles,
    compute_token_length_statistics,
)


class TestComputeBasicStats:
    def test_empty_returns_none(self) -> None:
        result = compute_basic_stats([])
        assert result == {"mean": None, "median": None, "min": None, "max": None}

    def test_empty_with_std(self) -> None:
        result = compute_basic_stats([], include_std=True)
        assert result["std_dev"] is None

    def test_single_value(self) -> None:
        result = compute_basic_stats([5.0])
        assert result["mean"] == 5.0
        assert result["median"] == 5.0
        assert result["min"] == 5.0
        assert result["max"] == 5.0

    def test_single_value_std_is_zero(self) -> None:
        result = compute_basic_stats([5.0], include_std=True)
        assert result["std_dev"] == 0.0

    def test_multiple_values(self) -> None:
        result = compute_basic_stats([1, 2, 3, 4, 5])
        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_std_dev_population(self) -> None:
        # numpy's np.std defaults to population std (ddof=0).
        # For [1,2,3,4,5], mean=3, sum of squared deviations = 4+1+0+1+4 = 10.
        result = compute_basic_stats([1, 2, 3, 4, 5], include_std=True)
        expected = math.sqrt(10 / 5)
        std_dev = result["std_dev"]
        assert std_dev is not None
        assert math.isclose(std_dev, expected, rel_tol=1e-9)

    def test_no_std_key_when_disabled(self) -> None:
        result = compute_basic_stats([1, 2, 3])
        assert "std_dev" not in result


class TestComputePercentiles:
    def test_empty_returns_none(self) -> None:
        result = compute_percentiles([], [50, 90])
        assert result == {"percentile_50": None, "percentile_90": None}

    def test_single_element_returns_none(self) -> None:
        # Code path requires >= 2 elements.
        result = compute_percentiles([42], [50])
        assert result == {"percentile_50": None}

    def test_two_elements_median(self) -> None:
        result = compute_percentiles([1, 9], [50])
        assert result["percentile_50"] == 5.0

    def test_multiple_percentiles(self) -> None:
        result = compute_percentiles(list(range(101)), [25, 50, 75])
        assert result["percentile_25"] == 25.0
        assert result["percentile_50"] == 50.0
        assert result["percentile_75"] == 75.0

    def test_p99(self) -> None:
        result = compute_percentiles(list(range(101)), [99])
        assert result["percentile_99"] == 99.0


class TestComputeDatasetOverview:
    def test_no_filtering(self) -> None:
        result = compute_dataset_overview(100, 100, "no filter")
        assert result["total_unfiltered"] == 100
        assert result["total_filtered"] == 100
        assert result["samples_filtered_out"] == 0
        assert result["filter_percentage"] == 0
        assert result["filter_description"] == "no filter"

    def test_full_filtering(self) -> None:
        result = compute_dataset_overview(100, 0, "all")
        assert result["samples_filtered_out"] == 100
        assert result["filter_percentage"] == 100.0

    def test_partial_filtering(self) -> None:
        result = compute_dataset_overview(200, 150, "some")
        assert result["samples_filtered_out"] == 50
        assert result["filter_percentage"] == 25.0

    def test_zero_unfiltered_avoids_divide_by_zero(self) -> None:
        result = compute_dataset_overview(0, 0, "n/a")
        assert result["filter_percentage"] == 0


class TestComputeTokenLengthStatistics:
    def test_empty_returns_empty(self) -> None:
        assert compute_token_length_statistics([]) == {}

    def test_includes_basic_stats(self) -> None:
        result = compute_token_length_statistics([1, 2, 3, 4, 5])
        assert "mean" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result
        assert "std_dev" in result

    def test_includes_percentiles(self) -> None:
        result = compute_token_length_statistics([1, 2, 3, 4, 5])
        for p in (25, 75, 90, 95, 99):
            assert f"percentile_{p}" in result

    def test_includes_distribution(self) -> None:
        result = compute_token_length_statistics([1, 1, 2, 3])
        assert result["distribution"] == {1: 2, 2: 1, 3: 1}

    def test_single_token_length(self) -> None:
        # Single value: basic stats yield value=5, std=0, percentiles None.
        result = compute_token_length_statistics([5])
        assert result["mean"] == 5.0
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["std_dev"] == 0.0
        # Percentiles require >= 2 elements; single value returns None for each.
        assert result["percentile_25"] is None
        assert result["percentile_99"] is None
