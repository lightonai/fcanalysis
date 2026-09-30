"""Independent physical and raw schema census for the migrated Dolci pin."""

from collections import Counter

import pytest
import pyarrow.parquet as pq

from fcanalysis.loaders.dolci import DATASET_ID, DATASET_REVISION, DATA_FILES
from fcanalysis.loaders.source import pinned_file


@pytest.mark.e2e
def test_pinned_dolci_raw_census():
    assert DATASET_REVISION == "dc042846f0f2de0f15eedae3d6ced04223ed47eb"
    subsets = Counter()
    roles = Counter()
    null_contents = 0
    total = 0
    for index, filename in enumerate(DATA_FILES):
        with pq.ParquetFile(
            pinned_file(DATASET_ID, DATASET_REVISION, filename)
        ) as file:
            assert file.metadata.num_rows == (37929 if index == 5 else 37930)
            for batch in file.iter_batches(batch_size=256):
                for raw in batch.to_pylist():
                    total += 1
                    assert set(raw) == {"id", "messages", "dataset_source"}
                    subsets[raw["dataset_source"]] += 1
                    assert raw["messages"][0]["role"] == "system"
                    assert sum(m["role"] == "system" for m in raw["messages"]) == 1
                    for message_index, message in enumerate(raw["messages"]):
                        assert set(message) == {
                            "role",
                            "content",
                            "function_calls",
                            "functions",
                        }
                        roles[message["role"]] += 1
                        assert isinstance(message["content"], (str, type(None)))
                        null_contents += message["content"] is None
                        if message_index:
                            assert message["functions"] is None
    assert total == 227579
    assert null_contents == 626290
    assert roles == {
        "system": 227579,
        "user": 557251,
        "assistant": 1171693,
        "environment": 615063,
    }
    assert subsets == {
        "allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated": 200000,
        "allenai/olmo-toolu-s2-sft-m3": 8074,
        "allenai/olmo-toolu-s2-sft-m4v2": 9085,
        "allenai/olmo-toolu-s2-sft-m5v2": 5417,
        "allenai/olmo-toolu_deepresearch_no_thinking_DRv4": 5003,
    }
