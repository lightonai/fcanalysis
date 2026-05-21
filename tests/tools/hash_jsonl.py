"""Canonical serialization and hashing for ConversationSample lists.

A loader's output is the ordered list of samples it produces. The
regression contract is: every implementation produces the same samples
in the same order, where "same sample" means messages, tools, dataset,
sample_id are byte-identical after canonical JSON serialization. The
`raw` field is excluded; it carries the source HF row for auditability
and is not part of the loader's behavioral contract.

Canonical form: each sample is one JSONL line of orjson.dumps with
OPT_SORT_KEYS. The hash is SHA-256 over the concatenation of every
line in produced order.
"""

import hashlib
from collections.abc import Iterable
from typing import Any

import orjson

from fcanalysis.format import ConversationSample


_ORJSON_OPTS = orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE


def sample_payload(sample: ConversationSample) -> dict[str, Any]:
    return {
        "messages": sample.messages,
        "tools": sample.tools,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
    }


def sample_to_jsonl_line(sample: ConversationSample) -> bytes:
    return orjson.dumps(sample_payload(sample), option=_ORJSON_OPTS)


def hash_samples(samples: Iterable[ConversationSample]) -> str:
    h = hashlib.sha256()
    for sample in samples:
        h.update(sample_to_jsonl_line(sample))
    return h.hexdigest()


def write_samples_jsonl(samples: Iterable[ConversationSample], fh) -> str:
    """Write samples to a binary file handle and return the SHA-256 hex digest.

    Single pass: writes and hashes in the same loop, so the file and the
    hash both reflect exactly the same byte stream.
    """
    h = hashlib.sha256()
    for sample in samples:
        line = sample_to_jsonl_line(sample)
        fh.write(line)
        h.update(line)
    return h.hexdigest()
