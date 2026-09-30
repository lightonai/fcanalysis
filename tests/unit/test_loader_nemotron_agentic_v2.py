"""V2-specific ID-bound ordered batches and legacy search response slots."""

from copy import deepcopy
import json

import pytest

from fcanalysis.loaders import nemotron_agentic_v2 as loader
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.normalization import Reject


def _row():
    return {
        "metadata": {"uuid": "row"},
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
            for name in ("f", "g")
        ],
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for name in ("f", "g")
                ],
            },
            {"role": "tool", "tool_call_id": "g", "name": "g", "content": "G"},
            {"role": "tool", "tool_call_id": "f", "name": "f", "content": "F"},
            {"role": "assistant", "content": "Answer"},
        ],
    }


def test_aligned_explicit_ids_preserve_atomic_result_order_and_raw():
    raw = _row()
    raw["messages"][2:4] = list(reversed(raw["messages"][2:4]))
    original = deepcopy(raw)
    stages = loader._pipeline(FilterConfig(), "tool_calling")
    state = stages.process(loader._convert_row(raw, "tool_calling"))
    assert state is not None
    assert [m["content"] for m in state.sample.messages[2:4]] == ["F", "G"]
    assert raw == original
    assert not state.batches[0].parallel
    assert stages.transforms["result_batches_reordered"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["messages"][1]["tool_calls"][1].update(id="f"),
        lambda r: r["messages"][2].update(tool_call_id="f"),
        lambda r: r["messages"][2].pop("tool_call_id"),
        lambda r: r["messages"][1]["tool_calls"][0].pop("id"),
        lambda r: r["messages"][2].update(name="f"),
    ],
)
def test_parallel_contradictions_and_partial_ids_fail(mutation):
    raw = _row()
    mutation(raw)
    stages = loader._pipeline(FilterConfig(), "tool_calling")
    assert stages.process(loader._convert_row(raw, "tool_calling")) is None
    assert stages.drops == {"invalid_source_tool_linkage": 1}


@pytest.mark.parametrize("split", ["interactive_agent", "search", "tool_calling"])
@pytest.mark.parametrize("align_results", [False, True])
def test_result_alignment_policy_is_independent_of_parallel_contract(
    split, align_results
):
    raw = _row()
    original = deepcopy(raw)
    stages = loader._pipeline(FilterConfig(align_results=align_results), split)
    state = stages.process(loader._convert_row(raw, split))
    if align_results:
        assert state is not None
        assert [m["content"] for m in state.sample.messages[2:4]] == ["F", "G"]
        assert not state.batches[0].parallel
        assert stages.transforms["result_batches_reordered"] == 1
    else:
        assert state is None
        assert stages.drops == {"unproven_result_reordering": 1}
    assert raw == original


def test_search_null_legacy_slot_is_classified_but_nonnull_never_discarded():
    raw = _row()
    raw["messages"][1]["function_call"] = None
    assert "function_call" not in loader._convert_row(raw, "search").messages[1]
    raw["messages"][1]["function_call"] = {"name": "f", "arguments": "{}"}
    with pytest.raises(Reject, match="unresolved_legacy_function_call"):
        loader._convert_row(raw, "search")
    raw["messages"][1]["function_call"] = None
    with pytest.raises(Reject, match="unresolved_legacy_function_call"):
        loader._convert_row(raw, "tool_calling")


@pytest.mark.parametrize("reverse_calls", [False, True])
def test_alignment_deduplication_preserves_call_order_significance(
    monkeypatch, tmp_path, reverse_calls
):
    first = _row()
    second = deepcopy(first)
    # The first row needs alignment. Equal call order must deduplicate after
    # alignment; different call order must remain significant for curation.
    if reverse_calls:
        second["messages"][1]["tool_calls"].reverse()
    else:
        second["messages"][2:4] = list(reversed(second["messages"][2:4]))
    second["metadata"]["uuid"] = "duplicate"
    monkeypatch.setattr(
        loader,
        "_source_lines",
        lambda split: iter([json.dumps(first), json.dumps(second)]),
    )
    values, report = loader.load(
        loader.NemotronAgenticV2Config(splits=("tool_calling",)),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert len(values) == report.final_count == (2 if reverse_calls else 1)
    assert values[0].sample_id == "tool_calling_row"
    assert values[0].raw == first
    stats = report.dataset_config_transform_counts["curation_by_subset"][
        "tool_calling"
    ][0]
    assert stats["stages"]["level_1"]["removed_samples"] == (0 if reverse_calls else 1)
    assert set(stats["audit"]) == {"level_3", "level_4", "level_5"}
    assert not values[0].annotations


def test_raw_source_exclusion_precedes_conversion_and_within_split_curation(
    monkeypatch, tmp_path
):
    rows = []
    for source in ("xlam", "xlam_tools", "when2call", "glaive"):
        raw = _row()
        raw["metadata"] = {"uuid": source, "source": source}
        rows.append(raw)
    original = deepcopy(rows)
    monkeypatch.setattr(
        loader, "_source_lines", lambda split: iter(map(json.dumps, rows))
    )
    selected, report = loader.load(
        loader.NemotronAgenticV2Config(
            splits=("tool_calling",),
            exclude_tool_calling_sources=("xlam", "xlam_tools", "when2call"),
        ),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert [sample.sample_id for sample in selected] == ["tool_calling_glaive"]
    assert selected[0].raw == rows[-1]
    assert rows == original
    assert report.raw_count == 4
    assert report.stage1_count == report.filtered_count == report.final_count == 1
    assert report.dataset_config_drop_reasons == {
        "excluded_raw_source/xlam": 1,
        "excluded_raw_source/xlam_tools": 1,
        "excluded_raw_source/when2call": 1,
    }
    assert report.dataset_config_transform_counts["source"]["subsets"][
        "tool_calling"
    ] == {
        "physical": 4,
        "parseable": 4,
        "source_excluded": 3,
        "converted": 1,
        "validated": 1,
    }
    # Without the source rule, the first identical row is the curation winner.
    all_rows, unfiltered = loader.load(
        loader.NemotronAgenticV2Config(splits=("tool_calling",)),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert [sample.sample_id for sample in all_rows] == ["tool_calling_xlam"]
    assert unfiltered.final_count == 1


@pytest.mark.parametrize("sources", [("xlam", "xlam"), ("",), (1,), ["xlam"]])
def test_raw_source_exclusion_requires_distinct_source_names(sources):
    with pytest.raises(ValueError, match="distinct nonempty strings"):
        loader.iter_load(
            loader.NemotronAgenticV2Config(exclude_tool_calling_sources=sources)
        )


@pytest.mark.parametrize("split", ["interactive_agent", "search"])
def test_raw_source_exclusion_does_not_apply_to_other_splits(
    monkeypatch, tmp_path, split
):
    raw = _row()
    raw["metadata"]["source"] = "xlam"
    monkeypatch.setattr(loader, "_source_lines", lambda split: iter([json.dumps(raw)]))
    rows, report = loader.load(
        loader.NemotronAgenticV2Config(
            splits=(split,), exclude_tool_calling_sources=("xlam",)
        ),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert len(rows) == report.final_count == 1
    assert report.dataset_config_drop_reasons == {}


@pytest.mark.parametrize("strip_thinking", [False, True])
def test_reused_source_uuid_preserves_distinct_calls_and_curates_exact_content(
    monkeypatch, tmp_path, strip_thinking
):
    first = _row()
    first["messages"][2:4] = list(reversed(first["messages"][2:4]))
    first["messages"][1]["reasoning_content"] = "Source reasoning."
    second = deepcopy(first)
    second["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
        '{"query":"different"}'
    )
    rows = [first, second, deepcopy(first)]
    original = deepcopy(rows)
    monkeypatch.setattr(
        loader, "_source_lines", lambda split: iter(map(json.dumps, rows))
    )
    values, report = loader.load(
        loader.NemotronAgenticV2Config(splits=("interactive_agent",)),
        filter_config=FilterConfig(strip_thinking=strip_thinking),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert report.final_count == len(values) == 2
    assert [sample.sample_id for sample in values] == ["interactive_agent_row"] * 2
    assert [
        sample.messages[1]["tool_calls"][0]["function"]["arguments"]
        for sample in values
    ] == ["{}", '{"query":"different"}']
    assert [sample.raw for sample in values] == [first, second]
    assert rows == original
    stats = report.dataset_config_transform_counts["curation_by_subset"][
        "interactive_agent"
    ][0]
    assert stats["stages"]["level_1"]["removed_samples"] == 1
    assert stats["stages"]["level_1_5"]["removed_samples"] == 0
    assert stats["stages"]["level_2"]["removed_samples"] == 0


def _download_row(*, start=True, matching=True):
    names = ("datagovsg_initiate_download", "datagovsg_poll_download")
    raw = _row()
    raw["tools"] = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in names
    ]
    raw["messages"] = [{"role": "user", "content": "Download dataset d_123."}]
    for i, name in enumerate(names):
        if i == 0 and not start:
            continue
        args = {"datasetId": "d_123", "columnNames": ["year"]}
        if i == 1 and not matching:
            args["columnNames"] = ["other"]
        result = (
            {"message": "Download successfully initiated. Proceed to poll download"}
            if i == 0
            else {"url": "https://example.org/download.csv"}
        )
        raw["messages"] += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": name, "content": json.dumps(result)},
        ]
    raw["messages"].append(
        {"role": "assistant", "content": "Here is the download URL."}
    )
    return raw


@pytest.mark.parametrize("start", [False, True])
def test_download_lifecycle_retains_visible_url_and_direct_non_csv_poll(start):
    raw = _download_row(start=start)
    stages = loader._pipeline(FilterConfig(), "tool_calling")
    state = stages.process(loader._convert_row(raw, "tool_calling"))
    assert state is not None
    assert json.loads(state.sample.messages[-2]["content"])["url"] == (
        "https://example.org/download.csv"
    )


def test_started_download_requires_identical_poll_arguments_and_resolved_result():
    raw = _download_row(matching=False)
    stages = loader._pipeline(FilterConfig(), "tool_calling")
    assert stages.process(loader._convert_row(raw, "tool_calling")) is None
    assert stages.drops == {"mismatched_download_request": 1}
    raw = _download_row()
    raw["messages"][-2]["content"] = '{"message":"undocumented pending response"}'
    stages = loader._pipeline(FilterConfig(), "tool_calling")
    assert stages.process(loader._convert_row(raw, "tool_calling")) is None
    assert stages.drops == {"unresolved_async_state": 1}
