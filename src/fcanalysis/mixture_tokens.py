"""Command-line analysis of an ordered union of retained canonical JSONL files.

Example::

    python -m fcanalysis.mixture_tokens \
        --input dolci=/path/dolci-native.jsonl.gz \
        --input apigen_mt=/path/apigen-native.jsonl.gz \
        --output /path/token-counts.json --workers 4

Inputs must be complete validated canonical outputs in the desired reasoning
mode. Input names and order are explicit; no datasets are downloaded, selected,
or silently mixed by this command. Only tokenizer assets may be downloaded.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

from huggingface_hub import snapshot_download

from .loaders.curation import CurationConfig, MixtureScope
from .mixture import MixtureSource, deduplicate_mixture
from .serialization import iter_samples_jsonl
from .token_analysis import TokenAnalysisConfig, TokenAnalysisReport, analyze_tokens
from .tokenization import QWEN35_MODEL_ID, QWEN35_REVISION


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="retained canonical JSONL or JSONL.gz, in deterministic tie order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="retained-native-mixture")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-characters", type=int, default=1_000_000)
    parser.add_argument("--batch-samples", type=int, default=128)
    parser.add_argument("--pending-batches", type=int)
    parser.add_argument("--temporary-directory", type=Path)
    parser.add_argument("--partition-mib", type=int, default=64)
    parser.add_argument("--minimum-free-gib", type=float, default=30)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    paths: list[tuple[str, Path]] = []
    for source in args.input:
        name, separator, path = source.partition("=")
        if not separator or not name.strip() or not path:
            parser.error("each --input must have the form NAME=PATH")
        selected = Path(path).resolve()
        if not selected.is_file():
            parser.error(f"input does not exist: {selected}")
        paths.append((name, selected))
    if len({name for name, _ in paths}) != len(paths):
        parser.error("input names must be unique")
    if args.minimum_free_gib < 0:
        parser.error("--minimum-free-gib must be nonnegative")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        parser.error("temporary output already exists; choose a new path")
    if args.temporary_directory is not None:
        args.temporary_directory.mkdir(parents=True, exist_ok=True)

    tokenizer_path = (
        str(args.tokenizer_path.resolve())
        if args.tokenizer_path is not None
        else snapshot_download(
            QWEN35_MODEL_ID,
            revision=QWEN35_REVISION,
            local_files_only=args.local_files_only,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    )
    token_config = TokenAnalysisConfig(
        tokenizer_path=tokenizer_path,
        workers=args.workers,
        batch_characters=args.batch_characters,
        batch_samples=args.batch_samples,
        pending_batches=args.pending_batches,
    )
    started = time.monotonic()
    last_progress = started
    last_phase = ""
    scratch = args.temporary_directory or Path(tempfile.gettempdir())

    def mixture_progress(phase: str, count: int) -> None:
        nonlocal last_progress, last_phase
        now = time.monotonic()
        if phase != last_phase or now - last_progress >= 10:
            free = shutil.disk_usage(scratch).free
            if free < args.minimum_free_gib * 1024**3:
                raise OSError("temporary filesystem is below the free-space reserve")
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "samples": count,
                        "seconds": now - started,
                        "free_gib": free / 1024**3,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            last_progress = now
            last_phase = phase

    def token_progress(report: TokenAnalysisReport) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress >= 10:
            print(
                json.dumps(
                    {
                        "phase": "tokens",
                        "samples": report.samples,
                        "tokens": report.tokens.total_tokens,
                        "seconds": now - started,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            last_progress = now

    sources = [MixtureSource(name, iter_samples_jsonl(path)) for name, path in paths]
    with deduplicate_mixture(
        sources,
        scope=MixtureScope(args.name, args.split),
        config=CurationConfig(temporary_directory=args.temporary_directory),
        partition_size_bytes=args.partition_mib * 1024**2,
        progress=mixture_progress,
    ) as mixture:
        tokens = analyze_tokens(mixture, token_config, progress=token_progress)
        if not mixture.report.complete:
            raise RuntimeError("mixture analysis ended without complete curation")
        if (
            mixture.report.curation is None
            or tokens.samples != mixture.report.curation.output_samples
        ):
            raise RuntimeError("tokenized sample count differs from mixture output")
        report = {
            "mixture": asdict(mixture.report),
            "tokenization": {
                "model": QWEN35_MODEL_ID if args.tokenizer_path is None else None,
                "revision": QWEN35_REVISION if args.tokenizer_path is None else None,
                "tokenizer_path": tokenizer_path,
                "rendering": "qwen35_all_history_source_reasoning",
                "trainable": "all_assistant_bodies_and_end_markers",
                "padding": False,
                "truncation": False,
                "generation_prompt": False,
            },
            "execution": {
                "workers": args.workers,
                "batch_characters": args.batch_characters,
                "batch_samples": args.batch_samples,
                "partition_mib": args.partition_mib,
                "minimum_free_gib": args.minimum_free_gib,
                "seconds": time.monotonic() - started,
            },
            **tokens.as_dict(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary.rename(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "samples": tokens.samples,
                "total_tokens": tokens.tokens.total_tokens,
                "trainable_tokens": tokens.tokens.trainable_tokens,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
