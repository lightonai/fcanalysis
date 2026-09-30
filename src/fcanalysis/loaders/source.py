"""Bounded raw-file iteration for adapters with an explicit pinned file list."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
import pyarrow.parquet as pq


def pinned_file(dataset: str, revision: str, filename: str) -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=dataset,
                revision=revision,
                filename=filename,
                repo_type="dataset",
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        return Path(
            hf_hub_download(
                repo_id=dataset,
                revision=revision,
                filename=filename,
                repo_type="dataset",
            )
        )


def parquet_rows(
    dataset: str,
    revision: str,
    files: Sequence[str],
    *,
    batch_size: int = 256,
    local_root: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Read published file order; an optional local root bypasses Hub lookup.

    A caller using local files owns their revision selection. This function
    performs no content repair, grouping or per-file identity bookkeeping.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for filename in files:
        path = (
            Path(local_root) / filename
            if local_root is not None
            else pinned_file(dataset, revision, filename)
        )
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()


def jsonl_lines(dataset: str, revision: str, files: Sequence[str]) -> Iterator[str]:
    """Yield physical UTF-8 lines in pinned file order without rewriting them.

    Adapters count each line and parse it inside their conversion boundary, so
    a malformed JSON row can be quarantined without ending file iteration.
    Blank lines are deliberately included; validity is not a source-reader
    policy. Newline translation is disabled to preserve released text.
    """
    for filename in files:
        with pinned_file(dataset, revision, filename).open(
            encoding="utf-8", newline=""
        ) as source:
            yield from source
