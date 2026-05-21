"""Unit tests for fcanalysis.semantic_filter.

Covers filter_by_categories and the filter_ams wrapper. The category
filter is the join point between loader output and the LLM-classified
no-FC turn categories; getting its semantics right matters because it
determines the final post-AMS sample count.
"""

from fcanalysis.cross_tabulation import SemanticResults
from fcanalysis.format import ConversationSample
from fcanalysis.semantic_filter import (
    SemanticFilterResult,
    filter_ams,
    filter_by_categories,
)

from tests.helpers import assistant, call, sample, tool_response, user


def _semantic_for(sample_id: str | int, turns: dict[int, str]) -> SemanticResults:
    """Build a {sample_id -> {turn_index -> {category: str}}} mapping."""
    return {sample_id: {idx: {"category": cat} for idx, cat in turns.items()}}


def _fc_sample(sid: str | int) -> ConversationSample:
    """Sample with only FC turns (no no-FC)."""
    return sample(
        messages=[user("hi"), assistant(tool_calls=[call()]), tool_response()],
        sample_id=sid,
    )


def _nofc_sample(sid: str | int) -> ConversationSample:
    """Sample with at least one no-FC turn (text-only assistant)."""
    return sample(
        messages=[user("hi"), assistant(content="text only reply")],
        sample_id=sid,
    )


class TestFilterByCategories:
    def test_empty_input(self) -> None:
        kept, result = filter_by_categories([], {}, {"ANTI_MANUAL_SOLVE"})
        assert kept == []
        assert result.input_samples == 0
        assert result.output_samples == 0
        assert result.removed_samples == 0
        assert result.removal_rate == 0.0

    def test_keeps_unclassified_with_nofc(self) -> None:
        s = _nofc_sample("u1")
        kept, result = filter_by_categories([s], {}, {"ANTI_MANUAL_SOLVE"})
        assert kept == [s]
        assert result.unclassified_samples == 1
        assert result.fc_only_samples == 0
        assert result.classified_samples == 0
        assert result.removed_samples == 0

    def test_keeps_unclassified_fc_only(self) -> None:
        s = _fc_sample("f1")
        kept, result = filter_by_categories([s], {}, {"ANTI_MANUAL_SOLVE"})
        assert kept == [s]
        assert result.fc_only_samples == 1
        assert result.unclassified_samples == 0
        assert result.classified_samples == 0

    def test_keeps_empty_semantic_entry_as_fc_only(self) -> None:
        # Semantic dict has an entry but no turns: treated as fc-only.
        s = _fc_sample("e1")
        sem: SemanticResults = {"e1": {}}
        kept, result = filter_by_categories([s], sem, {"ANTI_MANUAL_SOLVE"})
        assert kept == [s]
        assert result.fc_only_samples == 1
        assert result.classified_samples == 0

    def test_drops_sample_with_excluded_category(self) -> None:
        s = _nofc_sample("d1")
        sem = _semantic_for("d1", {1: "ANTI_MANUAL_SOLVE"})
        kept, result = filter_by_categories([s], sem, {"ANTI_MANUAL_SOLVE"})
        assert kept == []
        assert result.removed_samples == 1
        assert result.classified_samples == 1
        assert result.samples_flagged_by_category == {"ANTI_MANUAL_SOLVE": 1}

    def test_keeps_classified_sample_without_excluded_category(self) -> None:
        s = _nofc_sample("k1")
        sem = _semantic_for("k1", {1: "BENIGN"})
        kept, result = filter_by_categories([s], sem, {"ANTI_MANUAL_SOLVE"})
        assert kept == [s]
        assert result.classified_samples == 1
        assert result.samples_flagged_by_category == {}
        assert result.removed_samples == 0

    def test_drops_when_any_turn_matches(self) -> None:
        s = _nofc_sample("a1")
        sem = _semantic_for("a1", {1: "BENIGN", 2: "ANTI_MANUAL_SOLVE"})
        kept, result = filter_by_categories([s], sem, {"ANTI_MANUAL_SOLVE"})
        assert kept == []
        assert result.samples_flagged_by_category == {"ANTI_MANUAL_SOLVE": 1}

    def test_multiple_exclude_categories(self) -> None:
        # Sample with category X is removed when X is in exclude set.
        s1 = _nofc_sample("a")
        s2 = _nofc_sample("b")
        s3 = _nofc_sample("c")
        sem: SemanticResults = {
            "a": {1: {"category": "BAD_A"}},
            "b": {1: {"category": "BAD_B"}},
            "c": {1: {"category": "OK"}},
        }
        kept, result = filter_by_categories([s1, s2, s3], sem, {"BAD_A", "BAD_B"})
        assert [s.sample_id for s in kept] == ["c"]
        assert result.samples_flagged_by_category == {"BAD_A": 1, "BAD_B": 1}

    def test_removal_rate_property(self) -> None:
        samples = [_nofc_sample(i) for i in range(4)]
        sem: SemanticResults = {
            0: {1: {"category": "ANTI_MANUAL_SOLVE"}},
            1: {1: {"category": "ANTI_MANUAL_SOLVE"}},
            2: {1: {"category": "OK"}},
            3: {1: {"category": "OK"}},
        }
        _, result = filter_by_categories(samples, sem, {"ANTI_MANUAL_SOLVE"})
        assert result.removed_samples == 2
        assert result.removal_rate == 0.5

    def test_result_is_dataclass(self) -> None:
        kept, result = filter_by_categories([], {}, set())
        assert isinstance(result, SemanticFilterResult)


class TestFilterAms:
    def test_uses_ams_category_only(self) -> None:
        s = _nofc_sample("a")
        sem = _semantic_for("a", {1: "ANTI_MANUAL_SOLVE"})
        kept, result = filter_ams([s], sem)
        assert kept == []
        assert result.exclude_categories == {"ANTI_MANUAL_SOLVE"}

    def test_other_categories_not_excluded(self) -> None:
        s = _nofc_sample("b")
        sem = _semantic_for("b", {1: "SOME_OTHER"})
        kept, _ = filter_ams([s], sem)
        assert kept == [s]
