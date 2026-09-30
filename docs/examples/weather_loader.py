"""A runnable adapter for the synthetic source in docs/adding-loaders.md.

Input is an iterable of JSON strings covering one complete example/train scope.
Rows contain exactly id, dialogue, and functions. Dialogue uses canonical role
names and permits at most one call per assistant message. Functions are bare
definitions, available throughout the conversation; tool results are string/null
content. Removing native reasoning validates its shared inline grammar; retained
reasoning has no separate boundary-validation stage beyond endpoint checks.

This example owns no source files. Callers keep their input stream open until
loading finishes and close it themselves. It downloads data from no service and
executes none of the recorded calls.
"""

from collections import Counter
from collections.abc import Generator, Iterable, Iterator
from contextlib import closing
from copy import deepcopy
from functools import partial
import json
from typing import Any

from fcanalysis import ConversationSample
from fcanalysis.loaders.base import FilterConfig, LoadReport
from fcanalysis.loaders.curation import (
    CurationConfig,
    CurationInput,
    CurationScope,
    curate,
)
from fcanalysis.loaders.normalization import Reject, normalize_tools, parse_json
from fcanalysis.loaders.pipeline import (
    Pipeline,
    RowState,
    Stage,
    link_calls,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_structure,
    validate_termination,
)

DATASET_ID = "example/weather"


def convert_row(raw: Any) -> ConversationSample:
    if not isinstance(raw, dict) or set(raw) != {"id", "dialogue", "functions"}:
        raise Reject("unsupported_source_fields")
    if type(raw["id"]) not in (str, int):
        raise Reject("invalid_source_id")
    if not isinstance(raw["dialogue"], list) or not isinstance(raw["functions"], list):
        raise Reject("unsupported_source_shape")
    return ConversationSample(
        dataset=DATASET_ID,
        sample_id=raw["id"],
        messages=deepcopy(raw["dialogue"]),
        tools=normalize_tools(raw["functions"], bare=True),
        raw=raw,
    )


def validate_source_protocol(state: RowState) -> None:
    # This source defines sequential singleton actions, with no discovery,
    # background jobs, prefix omissions, or source-specific state prerequisites.
    for message in state.sample.messages:
        if len(message.get("tool_calls", [])) > 1:
            raise Reject("unsupported_call_batch")


def build_pipeline(filters: FilterConfig) -> Pipeline:
    linkage = partial(link_calls, align_results=filters.align_results)
    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("structure", validate_structure),
        ("source_protocol", validate_source_protocol),
        ("linkage", linkage),
        ("capabilities", validate_capabilities),
    ]
    if filters.strip_thinking:
        stages.append(("reasoning", remove_reasoning))
    if filters.system_message_override is not None:
        stages.append(
            (
                "system_override",
                partial(override_system, override=filters.system_message_override),
            )
        )
    stages.extend(
        [
            ("final_structure", validate_structure),
            # Singleton adjacency independently establishes pairing again after
            # IDs are removed, including when a system override shifts indices.
            ("final_linkage", linkage),
            ("final_capabilities", validate_capabilities),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def iter_load(
    lines: Iterable[str],
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    filters = filter_config or FilterConfig(strip_thinking=True)
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=0,
        stage1_count=0,
        filter_config=filters,
        strip_thinking_applied=filters.strip_thinking,
    )
    pipeline = build_pipeline(filters)

    def inputs() -> Iterator[CurationInput]:
        conversion_drops: Counter[str] = Counter()
        valid = 0
        for line in lines:
            report.raw_count += 1
            try:
                sample = convert_row(parse_json(line))
            except Reject as exc:
                conversion_drops[exc.reason] += 1
                continue
            report.stage1_count += 1
            state = pipeline.process(sample)
            if state is not None:
                valid += 1
                yield CurationInput(state.sample, state.batches, state.parsed_arguments)
        report.stage1_drop_reasons = dict(conversion_drops)
        report.dataset_config_count = report.stage1_count
        report.filtered_count = valid
        report.filter_drop_reasons = dict(pipeline.drops)
        report.dataset_config_transform_counts.update(
            {
                "pipeline_passed": dict(pipeline.passed),
                "pipeline_drops": dict(pipeline.stage_drops),
                "transformations": dict(pipeline.transforms),
            }
        )

    result = curate(
        inputs(),
        scope=CurationScope(DATASET_ID, "example", "train"),
        config=curation_config,
    )

    def output() -> Generator[ConversationSample, None, None]:
        try:
            count = 0
            for sample in result:
                count += 1
                yield sample
            report.final_count = count
            report.dataset_config_transform_counts["curation"] = [
                counts.as_dict() for counts in result.reports.values()
            ]
        finally:
            result.close()

    return output(), report


def load(
    lines: Iterable[str],
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    rows, report = iter_load(lines, filter_config, curation_config=curation_config)
    with closing(rows):
        return list(rows), report


def example_row() -> dict[str, Any]:
    return {
        "id": "weather-1",
        "functions": [
            {
                "name": "get_weather",
                "description": "Return the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        ],
        "dialogue": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Sunny."},
            {
                "role": "assistant",
                "content": "<think>Read the weather result.</think>Sunny.",
            },
        ],
    }


def main() -> None:
    original = example_row()
    duplicate = deepcopy(original)
    duplicate["id"] = "weather-2"
    invalid = deepcopy(original)
    invalid["id"] = "weather-3"
    invalid["dialogue"][1]["tool_calls"][0]["function"]["arguments"]["city"] = 7
    samples, report = load(json.dumps(row) for row in (original, duplicate, invalid))
    print(
        {
            "read": report.raw_count,
            "converted": report.stage1_count,
            "valid": report.filtered_count,
            "retained": report.final_count,
        }
    )
    print(report.filter_drop_reasons)
    print(samples[0].messages[-1])


if __name__ == "__main__":
    main()
