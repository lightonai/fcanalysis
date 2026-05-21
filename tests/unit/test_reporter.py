import io
import json
from pathlib import Path

from fcanalysis.behavioral import analyze_dataset_behavior
from fcanalysis.core import analyze_sample
from fcanalysis.format import ConversationSample
from fcanalysis.reporter import (
    print_abstention_supervision,
    print_conversation_turn_structure,
    print_dataset_overview,
    print_full_report,
    print_function_calling_patterns,
    print_function_diversity,
    print_parallel_function_diversity,
    print_termination_supervision,
    print_token_length_distribution,
    print_tool_call_validation,
    render_metrics_markdown,
    save_json_report,
    save_text_report,
)
from fcanalysis.statistics import aggregate_statistics


def _sample(
    messages: list[dict], tools: list[dict] | None = None
) -> ConversationSample:
    return ConversationSample(
        messages=messages, tools=tools or [], dataset="test", sample_id="s1", raw={}
    )


def _tc(name: str = "f") -> dict:
    return {"type": "function", "function": {"name": name, "arguments": "{}"}}


def _tool(name: str = "f") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _build_stats(samples: list[ConversationSample]) -> dict:
    messages_list = [s.messages for s in samples]
    tools_list = [s.tools for s in samples]
    analyses = [analyze_sample(m, extract_function_names=True) for m in messages_list]
    return aggregate_statistics(
        analyses=analyses,
        messages_list=messages_list,
        tools_list=tools_list,
        token_lengths=[10, 20, 30],
        dataset_overview={
            "total_unfiltered": 10,
            "samples_filtered_out": 5,
            "filter_percentage": 50.0,
            "total_filtered": 5,
            "filter_description": "test filter",
        },
    )


class TestPrintFunctions:
    def setup_method(self) -> None:
        self.samples = [
            _sample(
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "tool_calls": [_tc()]},
                    {"role": "tool", "content": "r"},
                    {"role": "assistant", "content": "done"},
                ],
                tools=[_tool()],
            )
        ]
        self.stats = _build_stats(self.samples)

    def test_print_dataset_overview(self) -> None:
        buf = io.StringIO()
        print_dataset_overview(self.stats, file=buf)
        out = buf.getvalue()
        assert "DATASET OVERVIEW" in out
        assert "test filter" in out

    def test_print_token_length_distribution(self) -> None:
        buf = io.StringIO()
        print_token_length_distribution(self.stats, file=buf)
        assert "TOKEN LENGTH DISTRIBUTION" in buf.getvalue()

    def test_print_function_diversity(self) -> None:
        buf = io.StringIO()
        print_function_diversity(self.stats, file=buf)
        out = buf.getvalue()
        assert "FUNCTION DIVERSITY" in out
        assert "Top 20" in out

    def test_print_function_calling_patterns(self) -> None:
        buf = io.StringIO()
        print_function_calling_patterns(self.stats, file=buf)
        assert "FUNCTION CALLING PATTERNS" in buf.getvalue()

    def test_print_parallel_function_diversity(self) -> None:
        buf = io.StringIO()
        print_parallel_function_diversity(self.stats, file=buf)
        assert "PARALLEL" in buf.getvalue().upper()

    def test_print_conversation_turn_structure(self) -> None:
        buf = io.StringIO()
        print_conversation_turn_structure(self.stats, file=buf)
        assert "CONVERSATION" in buf.getvalue().upper()

    def test_print_termination_supervision(self) -> None:
        buf = io.StringIO()
        print_termination_supervision(self.stats, file=buf)
        assert "TERMINATION" in buf.getvalue().upper()

    def test_print_abstention_supervision(self) -> None:
        buf = io.StringIO()
        print_abstention_supervision(self.stats, file=buf)
        assert "ABSTENTION" in buf.getvalue().upper()

    def test_print_tool_call_validation(self) -> None:
        buf = io.StringIO()
        print_tool_call_validation(self.stats, file=buf)
        assert "TOOL CALL" in buf.getvalue().upper()

    def test_print_full_report(self) -> None:
        buf = io.StringIO()
        print_full_report(self.stats, file=buf)
        out = buf.getvalue()
        # Full report includes all sections.
        for section in (
            "DATASET OVERVIEW",
            "TOKEN LENGTH DISTRIBUTION",
            "FUNCTION DIVERSITY",
            "FUNCTION CALLING PATTERNS",
        ):
            assert section in out


class TestRenderMetricsMarkdown:
    def test_returns_string_with_table_structure(self) -> None:
        samples = [
            _sample(
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "tool_calls": [_tc()]},
                    {"role": "tool", "content": "r"},
                    {"role": "assistant", "content": "done"},
                ],
                tools=[_tool()],
            )
        ]
        from fcanalysis.loaders.base import LoadReport

        report = LoadReport(dataset="test", raw_count=1, stage1_count=1)
        analyses = [
            analyze_sample(s.messages, extract_function_names=True) for s in samples
        ]
        beh_analyses = analyze_dataset_behavior(samples, analyses)
        from fcanalysis.behavioral import compute_bias_report

        bias = compute_bias_report(beh_analyses)
        md = render_metrics_markdown(
            samples=samples,
            report=report,
            analyses=analyses,
            beh_analyses=beh_analyses,
            bias=bias,
        )
        assert isinstance(md, str)
        assert "|" in md


class TestSaveReports:
    def test_save_json(self, tmp_path: Path) -> None:
        stats = {"total_samples": 5, "foo": [1, 2, 3]}
        path = tmp_path / "report.json"
        save_json_report(stats, str(path))
        with open(path) as f:
            assert json.load(f) == stats

    def test_save_text(self, tmp_path: Path) -> None:
        stats = _build_stats(
            [
                _sample(
                    [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                )
            ]
        )
        path = tmp_path / "report.txt"
        save_text_report(stats, str(path))
        assert path.exists()
        with open(path) as f:
            content = f.read()
        assert "DATASET OVERVIEW" in content
