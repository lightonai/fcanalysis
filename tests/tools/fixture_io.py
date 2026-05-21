"""Read and write the on-disk fixture format.

Layout per fixture (under tests/fixtures/loaders/{loader}/{config_id}/):
    config.json        committed   canonical config used to produce the output
    report.json        committed   LoadReport (dataclasses.asdict)
    output.hash        committed   SHA-256 of uncompressed JSONL bytes
    sample.jsonl       gitignored  deterministic 100-sample subset (debug aid)
    output.jsonl.gz    gitignored  full output (gzip-compressed)

The three committed files form the regression contract. sample.jsonl and
output.jsonl.gz are regenerated locally by generate_fixtures.py and are
used for human-readable diff debugging when the hash check fails.
"""

import dataclasses
import gzip
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.base import LoadReport

from .hash_jsonl import sample_to_jsonl_line


FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "loaders"

SAMPLE_SUBSET_SIZE = 100


def fixture_dir(loader: str, config_id: str) -> Path:
    return FIXTURES_ROOT / loader / config_id


def _serialize_config(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return {"_repr": repr(config)}


def write_fixture(
    out_dir: Path,
    dataset_config: Any,
    filter_config: Any,
    extra_kwargs: dict[str, Any],
    report: LoadReport,
    samples: list[ConversationSample],
) -> dict[str, Any]:
    """Write all five fixture files for one (loader, config) run.

    Atomicity contract: `output.hash` is written LAST via temp+rename,
    so its presence on disk strictly implies the other four files are
    complete. The fixture-skip check (`output.hash` exists) is therefore
    safe even across crashes or SIGINT mid-write.

    Returns a summary dict with sample_count and hash for the caller to
    log.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    config_doc = {
        "dataset_config": _serialize_config(dataset_config),
        "filter_config": _serialize_config(filter_config),
        "extra_kwargs": extra_kwargs,
    }
    (out_dir / "config.json").write_bytes(
        orjson.dumps(config_doc, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
        + b"\n"
    )

    report_doc = dataclasses.asdict(report)
    (out_dir / "report.json").write_bytes(
        orjson.dumps(report_doc, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
        + b"\n"
    )

    full_hash, total = _write_full_output(out_dir / "output.jsonl.gz", samples)

    subset = deterministic_subset(samples, SAMPLE_SUBSET_SIZE)
    with (out_dir / "sample.jsonl").open("wb") as fh:
        for sample in subset:
            fh.write(sample_to_jsonl_line(sample))

    hash_tmp = out_dir / "output.hash.tmp"
    hash_tmp.write_text(full_hash + "\n")
    hash_tmp.rename(out_dir / "output.hash")

    return {"sample_count": total, "hash": full_hash}


def _write_full_output(
    path: Path, samples: list[ConversationSample]
) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with gzip.open(path, "wb", compresslevel=6) as fh:
        for sample in samples:
            line = sample_to_jsonl_line(sample)
            fh.write(line)
            h.update(line)
            total += 1
    return h.hexdigest(), total


def deterministic_subset(
    samples: list[ConversationSample], size: int
) -> list[ConversationSample]:
    """Stable sub-sample of `size` samples by uniform stepping.

    The same input list always produces the same subset, so the subset
    in a fixture is exactly reproducible from any re-run that produces
    the same overall sample list.
    """
    if len(samples) <= size:
        return list(samples)
    step = len(samples) / size
    return [samples[int(i * step)] for i in range(size)]


def read_hash(loader: str, config_id: str) -> str:
    return (fixture_dir(loader, config_id) / "output.hash").read_text().strip()


def read_report(loader: str, config_id: str) -> dict[str, Any]:
    return json.loads((fixture_dir(loader, config_id) / "report.json").read_bytes())


def read_sample_subset(loader: str, config_id: str) -> list[dict[str, Any]]:
    text = (fixture_dir(loader, config_id) / "sample.jsonl").read_bytes()
    return [orjson.loads(line) for line in text.splitlines() if line]


def iter_full_output(loader: str, config_id: str) -> Iterable[dict[str, Any]]:
    path = fixture_dir(loader, config_id) / "output.jsonl.gz"
    with gzip.open(path, "rb") as fh:
        for line in fh:
            yield orjson.loads(line)
