"""Bounded TOUCAN producer/runtime contracts, checked before and after transforms.

Producer a1976dea7688c0f36533aa6e3a3fa4c27e017b4b emits [ERROR: and
[UNEXPECTED_ERROR: sentinels; these are failures rather than response targets.
Exact Qwen/Kimi tool-protocol delimiters in nonliteral assistant text are
unresolved serialization, never a reason to remove a valid tool episode.

Exa MCP source 52fa16aaf0e86b4070a65f395de663ea2fba3ae6 (2025-08-09),
src/tools/deepResearchStart.ts and deepResearchCheck.ts, exposes taskId,
status running/completed/failed and a report field. This source corroborates
pinned TOUCAN results but is not asserted to be every row's exact server build.
The check call is both polling and fetch: completed status alone is insufficient.
Its literal 'No report generated' fallback is not a report. Transport errors
remain visible and do not invent success or erase a previously started job.

Exact Coinranking/Figma API-key calls, AI prompt-management user_login
(username/password), TusClases login (email/password), and EVE Online SSO
(code exchange) calls are grounded in prior user/system
literals or actual linked result JSON string values, never descriptions,
assistant reasoning, future results, target labels or raw server metadata.
The test establishes literal availability, not arbitrary entity/policy truth.
Rednote's login and the bare login alias accept no arguments and launch browser
login; their visible Playwright/interactive-auth responses are preserved.
The one azure-mcp-login candidate is undefined, not an inferred auth capability.
This classifies observed call contracts, not every server's hidden auth state.
The inspected 12306 search departure-date schema has a demonstrated relative-date
omission (OSS source d42968e4-6b57-523a-8fa0-da0e66e9cb99): its only clock was
invented in assistant reasoning. Before any tool results, this exact consuming
slot with an English today/tomorrow/yesterday or this/next/last-weekday request
requires a visible user/system target date or labeled current date. Prior
unclassified tool results cause bounded abstention, not an invented clock.
The exact 12306-mcp-server-search alias is the pinned MCPManager line197
registration of server_name + '-' + tool.name. Its full teacher census finds
479 Kimi and393 Qwen candidates, of which only13 Qwen rows pass earlier gates;
three have this demonstrated missing-clock defect. Unrelated date-bearing tool
contracts and historical facts are outside this gate.
Qwen's optional hidden-hint flags are not released consistently. A present true
flag is a known omitted-context defect; absence is unknown, never inferred true.
MCP resources and instruction/unlock operations are preserved. No generic
resource text is interpreted as granting new callable names: the pinned wrapper
registers its callable map before inference. Unsupported discovery stays an
undefined capability until an audited runtime grammar is added.
"""

import re
from typing import Any

from .context import calendar_dates, collect_string_values, contains_literal_token
from .normalization import Reject, parse_json
from .pipeline import JobEvent, RowState, validate_lifecycle, _code_spans

_START = frozenset({"exa-search-deep_researcher_start", "deep_researcher_start"})
_CHECK = frozenset({"exa-search-deep_researcher_check", "deep_researcher_check"})
_CREDENTIALS = {
    "coinranking-configure_api_key": ("api_key",),
    "configure_api_key": ("api_key",),
    "figma-api-integration-set_api_key": ("api_key",),
    "set_api_key": ("api_key",),
    "ai咖提示词管理mcp-user_login": ("username", "password"),
    "user_login": ("username", "password"),
    "tusclasesparticulares-automation-server-tusclasesparticulares_login": (
        "email",
        "password",
    ),
    "tusclasesparticulares_login": ("email", "password"),
    "eve-online-market-data-server-authenticate": ("code",),
}
_RELATIVE_TRAVEL_DATE = re.compile(
    r"\b(?:today|tomorrow|yesterday|"
    r"(?:this|next|last)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b",
    re.IGNORECASE,
)
_CURRENT_DATE = re.compile(
    r"\b(?:today(?:'s date)?|(?:the )?current date)(?:\s+is\b|\s*[:=])\s*([^\n!?]+)",
    re.IGNORECASE,
)


def _train_calendar_definition(function: dict[str, Any]) -> bool:
    """The inspected 12306 departure-date contract, not a generic search name."""
    schema = function.get("parameters")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    return (
        function.get("name") in {"search", "12306-mcp-server-search"}
        and function.get("description") == "查询12306火车票"
        and isinstance(properties, dict)
        and properties.get("date")
        == {
            "type": "string",
            "format": "date",
            "description": "出发日期 格式：YYYY-MM-DD",
        }
        and properties.get("fromCity") == {"type": "string", "description": "出发城市"}
        and properties.get("toCity") == {"type": "string", "description": "到达城市"}
    )


def _validate_train_date(
    arguments: dict[str, Any],
    request: str,
    visible_text: list[str],
    *,
    prior_result: bool,
) -> None:
    # Unknown prior tool results could establish clock state. Abstain in that
    # case without treating arbitrary result dates as current-clock evidence.
    # This gate detects a demonstrated omission before any results; it does not
    # certify date arithmetic, all languages, or later unclassified protocols.
    if prior_result or not _RELATIVE_TRAVEL_DATE.search(request):
        return
    target = arguments.get("date")
    if not isinstance(target, str) or not calendar_dates(target):
        return  # Complete argument/schema validation is a preceding stage.
    for text in visible_text:
        if target in calendar_dates(text):
            return
        for match in _CURRENT_DATE.finditer(text):
            # A period in an English month abbreviation is not a sentence end.
            clause = re.sub(
                r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.(?=\s)",
                r"\1",
                match[1],
                flags=re.IGNORECASE,
            )
            if calendar_dates(re.split(r"[.;]", clause, maxsplit=1)[0]):
                return
    raise Reject("ungrounded_relative_date")


_CONTROL = re.compile(
    r"<\|(?:reserved_token_\d+|tool_call_(?:begin|argument_begin|end)|tool_calls_section_(?:begin|end)|im_[a-z0-9_]+|system|user|assistant|channel|end_of_response|endoftext)\|>"
)
_PROTOCOL = re.compile(r"</?function_calls?>|<invoke\s+name\s*=|</?tool_calls?>")
_ERROR = re.compile(r"^\s*\[(?:ERROR|UNEXPECTED_ERROR):")


def validate_producer_artifacts(state: RowState) -> None:
    for message in state.sample.messages:
        if message["role"] != "assistant":
            continue
        content = message.get("content") or ""
        # The shared literal scanner preserves quoted inline/fenced code. A
        # technical answer demonstrating tool-call syntax is ordinary content.
        text = content
        for start, end in reversed(_code_spans(content)):
            text = text[:start] + " " * (end - start) + text[end:]
        if _ERROR.match(text):
            raise Reject("producer_error_target")
        if _CONTROL.search(text) or _PROTOCOL.search(text):
            raise Reject("assistant_protocol_residue")


def _object(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        value = parse_json(content)
    except Reject:
        return None
    return value if isinstance(value, dict) else None


def _known_omitted_hint(raw: dict[str, Any]) -> bool:
    metadata = raw.get("metadata")
    if isinstance(metadata, str):
        metadata = _object(metadata)
    if not isinstance(metadata, dict):
        return False
    configs = metadata.get("synthetic_data_gen_configs", [])
    if not isinstance(configs, list):
        return False
    for config in configs:
        if not isinstance(config, dict):
            continue
        params = config.get("generation_params")
        if not isinstance(params, dict) or params.get("agent") != "qwen_agent":
            continue
        if (
            params.get("enable_tool_hint") is True
            or params.get("enable_irrelevant_warning") is True
        ):
            return True
    return False


def validate_context(state: RowState) -> None:
    if _known_omitted_hint(state.sample.raw):
        raise Reject("omitted_teacher_hint")
    static = {tool["function"]["name"]: tool["function"] for tool in state.sample.tools}
    train_calendars = {
        name
        for name, function in static.items()
        if _train_calendar_definition(function)
    }
    if not (static.keys() & (_START | _CHECK | _CREDENTIALS.keys()) or train_calendars):
        return
    messages = state.sample.messages
    visible_text: list[str] = []
    visible_values: set[str] = set()
    events: list[JobEvent] = []
    latest_request = ""
    prior_result = False
    by_result = {
        result_index: (batch.assistant_index, offset)
        for batch in state.batches
        for offset, result_index in enumerate(batch.result_indices)
    }
    pending: dict[tuple[int, int], tuple[str, str | None]] = {}
    for index, message in enumerate(messages):
        role = message["role"]
        if role in ("system", "user"):
            if message.get("content"):
                visible_text.append(message["content"])
            if role == "user":
                latest_request = message.get("content") or ""
            continue
        if role == "assistant":
            for offset, call in enumerate(message.get("tool_calls", [])):
                name = call["function"]["name"]
                args = state.parsed_arguments[index, offset]
                if name in train_calendars:
                    _validate_train_date(
                        args, latest_request, visible_text, prior_result=prior_result
                    )
                for credential in _CREDENTIALS.get(name, ()):
                    if credential in args:
                        value = args[credential]
                        if not isinstance(value, str) or not (
                            value in visible_values
                            or contains_literal_token(value, visible_text)
                        ):
                            raise Reject("ungrounded_credential")
                if name in _CHECK:
                    job = args.get("taskId")
                    if (
                        not isinstance(job, str)
                        or not job
                        or not (
                            job in visible_values
                            or contains_literal_token(job, visible_text)
                        )
                    ):
                        raise Reject("ungrounded_job_state")
                    pending[index, offset] = ("check", job)
                elif name in _START:
                    pending[index, offset] = ("start", None)
            continue
        if role != "tool":
            continue
        prior_result = True
        payload = _object(message.get("content"))
        if index in by_result:
            command = pending.get(by_result[index])
            if command is not None:
                operation, requested = command
                text = message.get("content") or ""
                if payload is None:
                    error_prefix = (
                        "Research start error"
                        if operation == "start"
                        else "Research check error"
                    )
                    exact_failure = (
                        "Failed to start research task. Please try again."
                        if operation == "start"
                        else "Failed to check research task status. Please try again."
                    )
                    if not (text.startswith(error_prefix) or text == exact_failure):
                        raise Reject("unrecognized_async_result")
                elif operation == "start":
                    job = payload.get("taskId")
                    if (
                        payload.get("success") is not True
                        or not isinstance(job, str)
                        or not job
                    ):
                        raise Reject("unrecognized_async_result")
                    events.append(JobEvent(job, "started"))
                else:
                    if requested is None:
                        raise Reject("ungrounded_job_state")
                    if payload.get("taskId") != requested:
                        raise Reject("invalid_async_result_linkage")
                    status = payload.get("status")
                    if status == "completed" and payload.get("success") is True:
                        report = payload.get("report")
                        present = (
                            isinstance(report, str)
                            and bool(report.strip())
                            and report != "No report generated"
                        )
                        events.append(JobEvent(requested, "completed", payload=present))
                    elif status == "running" and payload.get("success") is True:
                        events.append(JobEvent(requested, "pending"))
                    elif status == "failed" and payload.get("success") is False:
                        # Runtime-declared terminal failure ends this attempt;
                        # actual error observation and any retry are preserved.
                        events.append(JobEvent(requested, "abandoned"))
                    elif (
                        payload.get("success") is False
                        and payload.get("error") == "Task not found"
                    ):
                        pass  # Failure supplies no new lifecycle state.
                    else:
                        raise Reject("unrecognized_async_result")
        if payload is not None:
            collect_string_values(payload, visible_values)
    validate_lifecycle(events)
