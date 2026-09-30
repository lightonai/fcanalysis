"""Data Turnstile synthetic domains: experimental, unmigrated conversion.

Origin, generation and license
------------------------------
Amazon's upstream release is
https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains , pinned to
70bf9e9a6234e694efa36d1ec6c207c65a695f5b. The authoritative card is
https://huggingface.co/datasets/amazon/Turnstile-Synthetic-Domains/blob/70bf9e9a6234e694efa36d1ec6c207c65a695f5b/README.md .
Ramakrishnan and Sharma describe the method in Data Turnstile: A Scalable Open
Framework for Function-Calling Data Generation (https://arxiv.org/abs/2607.29250).
The card links https://github.com/amazon-science/data-turnstile as the generator;
it does not identify the exact generation commit/command for every released row.
The pinned card declares CC BY-NC 4.0 for the dataset, distinct from upstream
code terms. Per-row inherited rights are not independently established here.

The card attributes English synthetic interactions to Qwen2.5-32B-Instruct:
typed-role DAG decomposition, per-role validation before continuation, and
feedback retries. API definitions and observations are synthesized, not evidence
of real execution. Its 100,262 interactions cover 1,025 APIs and 17 templates,
including irrelevance, one-to-three user turns and serial/nominally parallel
patterns. These generation claims do not certify each released target's quality.

Source schema and verified conversion boundaries
------------------------------------------------
One default data.jsonl is joined with the same pinned api_definitions.json.
Rows expose interaction_template_name, api_names, distractors and interaction.
The last is an ordered list of single-key SYSTEM/USER/THINKING/API_CALL/API_OBS/
ASST objects. API_CALL is a Python expression; observations are strings or JSON
values. Source labels, intended-call names and template metadata stay in raw.
The separate definition file is not duplicated into raw. Local file overrides
are supported but are not certified as the pinned release.

All called and distractor definitions are joined, deduplicated by name and
sorted by exact name so source target/distractor position does not reveal the
answer. Definitions expose name/description/parameters; unrelated API metadata
is not promoted into model context. Exact dict/float aliases map to object/number
only at schema-valued positions; constraints and instance values remain intact.
Malformed nested required values are preserved, not repaired or certified.

The card explicitly describes storing nominally parallel calls as serial
API_CALL/API_OBS pairs. This adapter preserves that order and emits one call
per assistant message with one immediate named result. It does not regroup
calls based on a template label, split result arrays or infer extra outputs.
Only a bare function name with literal keyword arguments is accepted. Positional
arguments, unpacking, duplicate keywords, evaluated expressions and non-JSON
values are rejected; no source code is executed.

Reasoning and remaining uncertainties
-------------------------------------
The explicit THINKING role is native reasoning, attached verbatim as
reasoning_content to the immediately following API_CALL or ASST. Duplicate,
misplaced or dangling THINKING entries are rejected. No historical-prefix
reconstruction is inferred or performed. Reasoning is retained by default;
FilterConfig.strip_thinking=True invokes the legacy remover after conversion.
That remover drops reasoning_content, regex-removes balanced think/reasoning
spans from assistant prose and trims outer whitespace; it is not the shared
literal-aware native-boundary validator. A complete corpus inventory of inline
native/literal tags or dedicated reasoning tools is not certified here.
Ordinary calls, their definitions/arguments/results and other explanatory prose
are not removed merely for expressing reasoning; THINKING is a source role,
not a rule to delete tools named think.

Structural representability is not full supported JSON Schema validation,
model-visible grounding, policy compliance, observation truth or complete-answer
acceptance. Undefined calls are reported as issues unless their legacy optional
filter is enabled. There is no source-context gate, final-assistant requirement,
post-override context validation, or shared Levels 1/1.5/2/3-5 curation. Existing
fixtures do not confer the migrated loaders' stronger supervision guarantees.

Exact ordered experimental pipeline
-----------------------------------
1. Resolve the pinned data and definition files, or explicit local inputs.
   Parse and prepare definitions once, recording unusable definitions by name.
2. Read JSONL in physical order, assigning the existing zero-based sample_id.
   Reject malformed/non-JSON row values. Join all selected API/distractor names;
   reject rows needing absent or unusable definitions and copy canonical tools.
3. Convert source roles in order, associate THINKING with its next assistant,
   parse literal keyword calls, and require each call's immediate observation.
   Preserve string observations; serialize other JSON values. Reject orphaned,
   interrupted or unresolved calls/reasoning. Keep the complete original row.
4. Count undefined-call issues without dropping solely for that issue.
5. If FilterConfig is supplied, optionally strip native reasoning, then apply
   enabled legacy parseability/cardinality/defined-function/partial-argument
   checks, then system override. No final-view source validation follows.
6. Return a materialized sample list and legacy report. No deduplication,
   bounded-corpus-memory iterator, semantic acceptance or new source audit is
   implied by this documentation update.
"""

import ast
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ..format import ConversationSample
from ..validation import has_undefined_function_calls
from .base import FilterConfig, LoadReport, apply_filters

DATASET_ID = "amazon/Turnstile-Synthetic-Domains"
DATASET_REVISION = "70bf9e9a6234e694efa36d1ec6c207c65a695f5b"

_TYPE_ALIASES = {"dict": "object", "float": "number"}
_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions"}
_SCHEMA_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_CHILDREN = {
    "additionalProperties",
    "unevaluatedProperties",
    "additionalItems",
    "unevaluatedItems",
    "propertyNames",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
}


class _ConversionError(ValueError):
    """A known source row or definition defect, suitable for aggregate reporting."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _is_json_value(value: Any) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_is_json_value(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _is_json_value(item) for key, item in value.items()
        )
    return False


def _json_text(value: Any, reason: str) -> str:
    if not _is_json_value(value):
        raise _ConversionError(reason)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _normalize_schema(schema: Any) -> Any:
    """Translate known type aliases only at JSON Schema positions.

    Property names, enums, defaults and unknown keyword payloads are data, not
    schema nodes. Preserve them verbatim instead of recursively rewriting keys.
    This does not validate all schema constraints.
    """
    if type(schema) is bool:
        return schema
    if not isinstance(schema, dict):
        raise _ConversionError("malformed_tool_schema")
    result = deepcopy(schema)
    if "type" in schema:
        source_types = schema["type"]
        values = source_types if isinstance(source_types, list) else [source_types]
        if not values or any(not isinstance(value, str) for value in values):
            raise _ConversionError("unsupported_schema_type")
        normalized = [_TYPE_ALIASES.get(value, value) for value in values]
        if any(value not in _JSON_TYPES for value in normalized):
            raise _ConversionError("unsupported_schema_type")
        result["type"] = normalized if isinstance(source_types, list) else normalized[0]
    for key in _SCHEMA_MAPS & schema.keys():
        children = schema[key]
        if not isinstance(children, dict):
            raise _ConversionError("malformed_tool_schema")
        result[key] = {
            name: _normalize_schema(child) for name, child in children.items()
        }
    for key in _SCHEMA_LISTS & schema.keys():
        children = schema[key]
        if not isinstance(children, list):
            raise _ConversionError("malformed_tool_schema")
        result[key] = [_normalize_schema(child) for child in children]
    for key in _SCHEMA_CHILDREN & schema.keys():
        result[key] = _normalize_schema(schema[key])
    if "items" in schema:
        items = schema["items"]
        result["items"] = (
            [_normalize_schema(child) for child in items]
            if isinstance(items, list)
            else _normalize_schema(items)
        )
    # Draft-7 schema dependencies can be schemas or arrays of property names.
    if "dependencies" in schema:
        dependencies = schema["dependencies"]
        if not isinstance(dependencies, dict):
            raise _ConversionError("malformed_tool_schema")
        result["dependencies"] = {
            name: deepcopy(child)
            if isinstance(child, list)
            else _normalize_schema(child)
            for name, child in dependencies.items()
        }
    if "dependentSchemas" in schema:
        children = schema["dependentSchemas"]
        if not isinstance(children, dict):
            raise _ConversionError("malformed_tool_schema")
        result["dependentSchemas"] = {
            name: _normalize_schema(child) for name, child in children.items()
        }
    return result


def _prepare_definitions(
    definitions: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Prepare tools once; reject a defective definition only when a row uses it."""
    if not isinstance(definitions, dict):
        raise ValueError(
            "Turnstile API definitions must be a name-to-definition object"
        )
    tools: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for name, definition in definitions.items():
        try:
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(definition, dict)
                or definition.get("name") != name
                or not isinstance(definition.get("description"), str)
                or not isinstance(definition.get("parameters"), dict)
                or not _is_json_value(definition)
            ):
                raise _ConversionError("malformed_tool_definition")
            tools[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": definition["description"],
                    "parameters": _normalize_schema(definition["parameters"]),
                },
            }
        except _ConversionError as error:
            errors[name] = error.reason
    return tools, errors


def _parse_call(expression: Any) -> dict[str, Any]:
    if not isinstance(expression, str):
        raise _ConversionError("invalid_call_syntax")
    try:
        node = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as error:
        raise _ConversionError("invalid_call_syntax") from error
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.args
    ):
        raise _ConversionError("unsupported_call_expression")
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise _ConversionError("unsupported_call_expression")
        if keyword.arg in arguments:
            raise _ConversionError("duplicate_call_keyword")
        try:
            value = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError, RecursionError) as error:
            raise _ConversionError("nonliteral_argument") from error
        if not _is_json_value(value):
            raise _ConversionError("non_json_argument")
        arguments[keyword.arg] = value
    return {
        "type": "function",
        "function": {
            "name": node.func.id,
            "arguments": _json_text(arguments, "non_json_argument"),
        },
    }


def _convert_row(
    row: Any,
    sample_id: int,
    tools_by_name: dict[str, dict[str, Any]],
    definition_errors: dict[str, str],
) -> ConversationSample:
    if not isinstance(row, dict):
        raise _ConversionError("malformed_row")
    names: list[str] = []
    for field in ("api_names", "distractors"):
        value = row.get(field)
        if not isinstance(value, list) or any(
            not isinstance(name, str) or not name for name in value
        ):
            raise _ConversionError("malformed_tool_list")
        names.extend(value)
    tools = []
    for name in sorted(set(names)):
        if name in definition_errors:
            raise _ConversionError(definition_errors[name])
        if name not in tools_by_name:
            raise _ConversionError("missing_tool_definition")
        tools.append(deepcopy(tools_by_name[name]))

    interaction = row.get("interaction")
    if not isinstance(interaction, list) or not interaction:
        raise _ConversionError("malformed_interaction")
    messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None
    pending_call: str | None = None
    for entry in interaction:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise _ConversionError("malformed_role")
        role, value = next(iter(entry.items()))
        if not isinstance(role, str) or role not in {
            "SYSTEM",
            "USER",
            "THINKING",
            "API_CALL",
            "API_OBS",
            "ASST",
        }:
            raise _ConversionError("unknown_role")
        if role == "THINKING":
            if pending_reasoning is not None:
                raise _ConversionError("duplicate_reasoning")
            if pending_call is not None:
                raise _ConversionError("ambiguous_reasoning")
            if not isinstance(value, str):
                raise _ConversionError("invalid_message_content")
            pending_reasoning = value
            continue
        if pending_reasoning is not None and role not in {"API_CALL", "ASST"}:
            raise _ConversionError("ambiguous_reasoning")
        if pending_call is not None and role != "API_OBS":
            raise _ConversionError("unbalanced_call_sequence")
        if role == "API_OBS":
            if pending_call is None:
                raise _ConversionError("orphan_observation")
            messages.append(
                {
                    "role": "tool",
                    "name": pending_call,
                    "content": value
                    if isinstance(value, str)
                    else _json_text(value, "invalid_observation"),
                }
            )
            pending_call = None
            continue
        if role == "API_CALL":
            call = _parse_call(value)
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [call],
            }
            pending_call = call["function"]["name"]
        else:
            if not isinstance(value, str):
                raise _ConversionError("invalid_message_content")
            message = {
                "role": {"SYSTEM": "system", "USER": "user", "ASST": "assistant"}[role],
                "content": value,
            }
        if pending_reasoning is not None:
            message["reasoning_content"] = pending_reasoning
            pending_reasoning = None
        messages.append(message)
    if pending_reasoning is not None:
        raise _ConversionError("dangling_reasoning")
    if pending_call is not None:
        raise _ConversionError("unbalanced_call_sequence")
    return ConversationSample(
        messages=messages,
        tools=tools,
        dataset=DATASET_ID,
        sample_id=sample_id,
        raw=row,
    )


def load(
    filter_config: FilterConfig | None = None,
    *,
    path: str | Path | None = None,
    api_definitions_path: str | Path | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    """Load the pinned release or local JSONL and sibling API definitions.

    ``path`` may be a snapshot directory or a data JSONL file. A local input uses
    its sibling ``api_definitions.json`` unless overridden explicitly. Conversion
    defects drop rows with a first-failure reason; no source-quality policy or
    deduplication is applied. Reasoning is preserved unless requested otherwise
    through ``FilterConfig``. Source API metadata remains in the definitions
    file and is not added to model tools or repeated in each sample's raw row.
    """
    if path is None:
        data_path = Path(
            hf_hub_download(
                DATASET_ID,
                "data.jsonl",
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
        definitions_path = (
            Path(api_definitions_path)
            if api_definitions_path is not None
            else Path(
                hf_hub_download(
                    DATASET_ID,
                    "api_definitions.json",
                    repo_type="dataset",
                    revision=DATASET_REVISION,
                )
            )
        )
    else:
        data_path = Path(path)
        if data_path.is_dir():
            data_path = data_path / "data.jsonl"
        definitions_path = (
            Path(api_definitions_path)
            if api_definitions_path is not None
            else data_path.with_name("api_definitions.json")
        )
    definitions = json.loads(definitions_path.read_text(encoding="utf-8"))
    tools_by_name, definition_errors = _prepare_definitions(definitions)
    samples: list[ConversationSample] = []
    drops: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    raw_count = 0
    with data_path.open(encoding="utf-8") as source:
        for sample_id, line in enumerate(source):
            raw_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                drops["invalid_row_json"] += 1
                continue
            try:
                if not _is_json_value(row):
                    raise _ConversionError("non_json_row_value")
                sample = _convert_row(row, sample_id, tools_by_name, definition_errors)
            except _ConversionError as error:
                drops[error.reason] += 1
                continue
            if has_undefined_function_calls(sample):
                issues["undefined_function_calls"] += 1
            samples.append(sample)
    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=raw_count,
        stage1_count=len(samples),
        stage1_drop_reasons=dict(drops),
        stage1_issue_counts=dict(issues),
    )
    if filter_config is not None:
        samples, filter_drops = apply_filters(samples, filter_config)
        report.filtered_count = len(samples)
        report.filter_config = filter_config
        report.filter_drop_reasons = filter_drops
    return samples, report
