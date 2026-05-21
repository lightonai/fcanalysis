import json
from pathlib import Path

from fcanalysis.behavioral import analyze_dataset_behavior
from fcanalysis.cross_tabulation import (
    SemanticResults,
    cross_tabulate,
    load_semantic_results,
    print_cross_tab_report,
)
from fcanalysis.format import ConversationSample


def _sample(sid: str, messages: list[dict]) -> ConversationSample:
    return ConversationSample(
        messages=messages, tools=[], dataset="t", sample_id=sid, raw={}
    )


def _tc(name: str = "f") -> dict:
    return {"type": "function", "function": {"name": name, "arguments": "{}"}}


def _no_fc(sid: str) -> ConversationSample:
    return _sample(
        sid,
        [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "I won't"},
        ],
    )


def _fc(sid: str) -> ConversationSample:
    return _sample(
        sid,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [_tc()]},
            {"role": "tool", "content": "r"},
        ],
    )


class TestLoadSemanticResults:
    def test_loads_classifications(self, tmp_path: Path) -> None:
        path = tmp_path / "sem.jsonl"
        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "sample_id": "a",
                        "classifications": [
                            {
                                "turn_index": 0,
                                "category": "S4_DIRECT_ANSWER",
                                "justified": True,
                                "reasoning": "x",
                            }
                        ],
                    }
                )
                + "\n"
            )
        results = load_semantic_results(path)
        assert "a" in results
        assert results["a"][0]["category"] == "S4_DIRECT_ANSWER"

    def test_skips_error_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "sem.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"sample_id": "a", "error": "x"}) + "\n")
            f.write(
                json.dumps(
                    {
                        "sample_id": "b",
                        "classifications": [
                            {
                                "turn_index": 0,
                                "category": "S4_DIRECT_ANSWER",
                                "justified": True,
                                "reasoning": "x",
                            }
                        ],
                    }
                )
                + "\n"
            )
        results = load_semantic_results(path)
        assert "a" not in results
        assert "b" in results


class TestCrossTabulate:
    def test_returns_full_report_structure(self) -> None:
        samples = [_no_fc("a"), _fc("b")]
        analyses = analyze_dataset_behavior(samples)
        semantic: SemanticResults = {
            "a": {
                0: {
                    "turn_index": 0,
                    "category": "ANTI_MANUAL_SOLVE",
                    "justified": False,
                    "reasoning": "x",
                }
            }
        }
        report = cross_tabulate(analyses, semantic)
        for key in (
            "coverage",
            "category_counts",
            "justified_vs_unjustified",
            "length_by_category",
            "length_justified_vs_unjustified",
            "position_by_category",
            "position_justified_vs_unjustified",
            "tool_count_by_category",
            "tool_count_justified_vs_unjustified",
            "system_prompt_by_category",
            "unjustified_rate_by_length_bin",
        ):
            assert key in report
        assert report["coverage"]["matched_no_fc_turns"] == 1
        assert report["coverage"]["unmatched_no_fc_turns"] == 0
        assert report["category_counts"] == {"ANTI_MANUAL_SOLVE": 1}
        jvu = report["justified_vs_unjustified"]
        assert jvu["total_classified"] == 1
        assert jvu["unjustified"] == 1
        assert jvu["justified"] == 0

    def test_unmatched_no_fc_counted(self) -> None:
        analyses = analyze_dataset_behavior([_no_fc("a")])
        report = cross_tabulate(analyses, {})
        assert report["coverage"]["matched_no_fc_turns"] == 0
        assert report["coverage"]["unmatched_no_fc_turns"] == 1

    def test_length_bins_present(self) -> None:
        analyses = analyze_dataset_behavior([_no_fc("a")])
        semantic: SemanticResults = {
            "a": {
                0: {
                    "turn_index": 0,
                    "category": "S4_DIRECT_ANSWER",
                    "justified": True,
                    "reasoning": "x",
                }
            }
        }
        report = cross_tabulate(analyses, semantic)
        bins = report["unjustified_rate_by_length_bin"]
        # All six bins should be present.
        assert set(bins) == {"0-50", "50-100", "100-150", "150-250", "250-500", "500+"}


class TestPrintCrossTabReport:
    def test_returns_string_with_sections(self) -> None:
        analyses = analyze_dataset_behavior([_no_fc("a")])
        semantic: SemanticResults = {
            "a": {
                0: {
                    "turn_index": 0,
                    "category": "S4_DIRECT_ANSWER",
                    "justified": True,
                    "reasoning": "x",
                }
            }
        }
        report = cross_tabulate(analyses, semantic)
        out = print_cross_tab_report(report)
        for section in (
            "Coverage",
            "Semantic Category Distribution",
            "Justified vs Unjustified",
            "Unjustified Rate by Length Bin",
        ):
            assert section in out
