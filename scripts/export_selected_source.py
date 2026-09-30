"""Export one selected source as retained canonical conversations.

Run from a repository checkout after `uv sync --locked`. Each invocation writes
canonical.jsonl.gz and report.json to a new output directory. Reasoning is kept;
source selection, validation, reconstruction, and curation run in the loader.
A report is written only after exhaustion and agreement with the emitted count.

See docs/loading.md for the selected sources and output fields. The report counts
conversations and assistant messages; it does not count supervised tokens or
apply model-specific training admission or cross-source mixture curation.
"""

import argparse
from collections import Counter
from collections.abc import Generator
from contextlib import closing
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shutil
import time

from fcanalysis.loaders import (
    dolci,
    nemotron_agentic_v1,
    nemotron_agentic_v2,
    nemotron_terminal,
    toolmind,
    toolmind_web,
    toucan,
    txt360,
    ultradata_tool_use,
)
from fcanalysis.format import ConversationSample
from fcanalysis.loaders.base import FilterConfig, LoadReport
from fcanalysis.loaders.curation import CurationConfig


SOURCES = (
    "dolci",
    "nemotron_v1",
    "nemotron_v2",
    "terminal",
    "toolmind",
    "toolmind_web",
    "toucan",
    "txt360_high",
    "txt360_medium",
    "txt360_low",
    "ultradata",
)


def selected_source(
    name: str, *, temporary_directory: Path
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    filters = FilterConfig(
        strip_thinking=False,
        require_parseable_arguments=True,
        require_balanced_cardinality=True,
        require_defined_functions=True,
        require_valid_arguments=True,
    )
    curation = CurationConfig(
        max_level="level_2", audit=True, temporary_directory=temporary_directory
    )
    if name == "dolci":
        return dolci.iter_load(filter_config=filters, curation_config=curation)
    if name == "nemotron_v1":
        return nemotron_agentic_v1.iter_load(
            filter_config=filters, curation_config=curation
        )
    if name == "nemotron_v2":
        return nemotron_agentic_v2.iter_load(
            nemotron_agentic_v2.NemotronAgenticV2Config(
                exclude_tool_calling_sources=("xlam", "xlam_tools", "when2call")
            ),
            filter_config=filters,
            curation_config=curation,
        )
    if name == "terminal":
        return nemotron_terminal.iter_load(
            filter_config=filters, curation_config=curation
        )
    if name == "toolmind":
        return toolmind.iter_load(
            toolmind.ToolMindConfig(
                sources=[
                    "open_datasets/BUTTONInstruct-query.jsonl",
                    "open_datasets/ToolACE-query.jsonl",
                    "open_datasets/glaive-function-calling-v2-query.jsonl",
                    "open_datasets/tau-train-query.jsonl",
                ]
            ),
            filter_config=filters,
            curation_config=curation,
        )
    if name == "toolmind_web":
        return toolmind_web.iter_load(filter_config=filters, curation_config=curation)
    if name == "toucan":
        return toucan.iter_load(filter_config=filters, curation_config=curation)
    if name in ("txt360_high", "txt360_medium", "txt360_low"):
        return txt360.iter_load(
            split=name.removeprefix("txt360_"),
            filter_config=filters,
            curation_config=curation,
        )
    if name == "ultradata":
        return ultradata_tool_use.iter_load(
            ultradata_tool_use.UltraDataToolUseConfig(
                sources=tuple(
                    source
                    for source in ultradata_tool_use.SOURCES
                    if source != "toolmind_graphsyn"
                )
            ),
            filter_config=filters,
            curation_config=curation,
        )
    raise ValueError(f"unknown selected source: {name}")


def export(
    name: str, output: Path, temporary_directory: Path, min_free_gib: int
) -> None:
    if name not in SOURCES or min_free_gib < 10:
        raise ValueError("unknown source or free-space reserve below 10 GiB")
    if not temporary_directory.is_dir():
        raise ValueError("curation temporary directory must already exist")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, loader_report = selected_source(name, temporary_directory=temporary_directory)
    conversations: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    with (
        closing(rows),
        gzip.open(
            output / "canonical.jsonl.gz", "xt", encoding="utf-8", compresslevel=1
        ) as stream,
    ):
        for index, sample in enumerate(rows, 1):
            if name == "nemotron_v2" and sample.dataset.endswith("/tool_calling"):
                metadata = sample.raw.get("metadata")
                if isinstance(metadata, dict) and metadata.get("source") in {
                    "xlam",
                    "xlam_tools",
                    "when2call",
                }:
                    raise ValueError("excluded Nemotron v2 raw source survived")
            if name == "ultradata" and sample.raw.get("source") == "toolmind_graphsyn":
                raise ValueError("excluded UltraData raw source survived")
            conversations[sample.dataset] += 1
            decisions[sample.dataset] += sum(
                message.get("role") == "assistant" for message in sample.messages
            )
            stream.write(
                json.dumps(asdict(sample), ensure_ascii=False, allow_nan=False) + "\n"
            )
            if index % 10000 == 0:
                free_gib = shutil.disk_usage(output).free / 2**30
                print(
                    json.dumps(
                        {
                            "source": name,
                            "conversations": index,
                            "free_gib": round(free_gib, 1),
                        }
                    ),
                    flush=True,
                )
                if free_gib < min_free_gib:
                    raise OSError("free-space reserve reached before export completed")
    if loader_report.final_count != sum(conversations.values()):
        raise ValueError("loader report and exported conversation count differ")
    report = {
        **asdict(loader_report),
        "complete": True,
        "source": name,
        "conversations": dict(conversations),
        "assistant_decisions": dict(decisions),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "complete": True,
                "source": name,
                "conversations": sum(conversations.values()),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory for JSONL and report"
    )
    parser.add_argument(
        "--temporary-directory",
        type=Path,
        required=True,
        help="existing scratch directory for curation",
    )
    parser.add_argument(
        "--min-free-gib",
        type=int,
        default=500,
        help="output disk reserve, checked every 10,000 exports (default: 500; minimum: 10)",
    )
    args = parser.parse_args()
    export(args.source, args.output, args.temporary_directory, args.min_free_gib)


if __name__ == "__main__":
    main()
