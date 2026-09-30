"""Exact TOUCAN context boundaries; no semantic keyword filtering."""

import json

import pytest

from tests.unit.test_loader_toucan import assistant, call, result, row, run, tool


START = "exa-search-deep_researcher_start"
CHECK = "exa-search-deep_researcher_check"


def job_tools():
    return [
        tool(
            START,
            {
                "type": "object",
                "properties": {"instructions": {"type": "string"}},
                "required": ["instructions"],
            },
        ),
        tool(
            CHECK,
            {
                "type": "object",
                "properties": {"taskId": {"type": "string"}},
                "required": ["taskId"],
            },
        ),
    ]


def start_pair(job="job-42"):
    return [
        call(START, {"instructions": "Research this topic."}),
        result(START, json.dumps({"success": True, "taskId": job})),
    ]


def check_pair(job="job-42", **payload):
    return [
        call(CHECK, {"taskId": job}),
        result(
            CHECK,
            json.dumps(
                {
                    "success": True,
                    "taskId": job,
                    "status": "completed",
                    "report": "Full research report.",
                    **payload,
                }
            ),
        ),
    ]


def job_row(messages):
    return row(
        [
            {"role": "user", "content": "Research this topic."},
            *messages,
            assistant("Here are the findings."),
        ],
        job_tools(),
    )


def test_complete_job_keeps_start_poll_payload_and_answer():
    raw = job_row(
        [*start_pair(), *check_pair(status="running", report=None), *check_pair()]
    )
    state, _, _ = run(raw, strip=True)
    assert state
    assert len(state.batches) == 3
    assert "Full research report." in state.sample.messages[-2]["content"]


@pytest.mark.parametrize(
    "report", [None, "", "   ", "No report generated", {}, ["summary"]]
)
def test_completed_status_without_source_report_is_not_completion(report):
    state, pipeline, _ = run(job_row([*start_pair(), *check_pair(report=report)]))
    assert state is None and pipeline.drops == {"incomplete_async_job": 1}


@pytest.mark.parametrize(
    "messages",
    [[*start_pair()], [*start_pair(), *check_pair(status="running", report=None)]],
)
def test_started_pending_job_is_incomplete(messages):
    state, pipeline, _ = run(job_row(messages))
    assert state is None and pipeline.drops == {"incomplete_async_job": 1}


def test_unobserved_id_cannot_come_from_future_result_or_assistant_claim():
    raw = job_row([assistant("Use job-42."), *check_pair(), *start_pair()])
    state, pipeline, _ = run(raw)
    assert state is None and pipeline.drops == {"ungrounded_job_state": 1}


def test_poll_only_row_can_use_visible_user_identifier():
    raw = row(
        [
            {
                "role": "user",
                "content": "Check research job-42 and give me its report.",
            },
            *check_pair(),
            assistant(),
        ],
        job_tools(),
    )
    assert run(raw)[0]


def test_poll_sibling_cannot_use_start_result_from_same_batch():
    start_call, start_result = start_pair()
    check_call, check_result = check_pair()
    state, pipeline, _ = run(
        job_row([start_call, check_call, start_result, check_result])
    )
    assert state is None and pipeline.drops == {"ungrounded_job_state": 1}


def test_poll_id_and_linked_response_id_must_agree():
    check_call, _ = check_pair()
    check_result = result(
        CHECK,
        json.dumps(
            {
                "success": True,
                "taskId": "wrong",
                "status": "completed",
                "report": "report",
            }
        ),
    )
    state, pipeline, _ = run(job_row([*start_pair(), check_call, check_result]))
    assert state is None and pipeline.drops == {"invalid_async_result_linkage": 1}


def test_linked_terminal_failure_then_retry_is_preserved():
    raw = job_row(
        [
            *start_pair(),
            *check_pair(success=False, status="failed", report=None),
            *start_pair("retry-job"),
            *check_pair("retry-job"),
        ]
    )
    state, _, _ = run(raw)
    assert state and len(state.batches) == 4


def test_linked_transport_error_does_not_fabricate_completion():
    raw = job_row(
        [
            *start_pair(),
            call(CHECK, {"taskId": "job-42"}),
            result(CHECK, "Research check error (504): timed out"),
        ]
    )
    state, pipeline, _ = run(raw)
    assert state is None and pipeline.drops == {"incomplete_async_job": 1}


def test_final_context_rechecks_removed_source_system_job_id():
    raw = row(
        [
            {
                "role": "system",
                "content": "The user authorized polling research job-42.",
            },
            {"role": "user", "content": "Check it."},
            *check_pair(),
            assistant(),
        ],
        job_tools(),
    )
    assert run(raw)[0]
    state, pipeline, _ = run(raw, override="")
    assert state is None and pipeline.drops == {"ungrounded_job_state": 1}
    assert pipeline.stage_drops == {"final_context": 1}


def test_override_cannot_rescue_original_missing_job_context():
    raw = job_row(check_pair())
    state, pipeline, _ = run(raw, override="The user authorized job-42.")
    assert state is None and pipeline.stage_drops == {"source_policy": 1}


def credential_tools():
    return [
        tool(
            "coinranking-configure_api_key",
            {
                "type": "object",
                "properties": {"api_key": {"type": "string"}},
                "required": ["api_key"],
            },
        )
    ]


def credential_row(prefix):
    return row(
        [
            *prefix,
            {"role": "user", "content": "Configure the API."},
            call("coinranking-configure_api_key", {"api_key": "literal-test-key"}),
            result("coinranking-configure_api_key", "configured"),
            assistant(),
        ],
        credential_tools(),
    )


def test_credential_from_visible_user_or_system_is_preserved():
    for role in ("user", "system"):
        assert run(
            credential_row([{"role": role, "content": "Use literal-test-key."}])
        )[0]


def test_credential_from_audit_metadata_or_assistant_is_not_context():
    raw = credential_row([assistant("The key is literal-test-key.")])
    raw["metadata"] = json.dumps({"api_key": "literal-test-key"})
    state, pipeline, _ = run(raw)
    assert state is None and pipeline.drops == {"ungrounded_credential": 1}


def test_credential_from_schema_description_or_default_is_not_context():
    raw = credential_row([])
    raw["available_tools"] = credential_tools()
    raw["available_tools"][0]["function"]["description"] = (
        "API key example literal-test-key"
    )
    raw["available_tools"][0]["function"]["parameters"]["properties"]["api_key"][
        "default"
    ] = "literal-test-key"
    assert run(raw)[1].drops == {"ungrounded_credential": 1}


def test_final_context_rechecks_removed_credential_system():
    raw = credential_row([{"role": "system", "content": "Use literal-test-key"}])
    assert run(raw)[0]
    state, pipeline, _ = run(raw, override="")
    assert state is None and pipeline.stage_drops == {"final_context": 1}


def test_linked_result_json_value_can_ground_credential_but_map_key_cannot():
    for content, accepted in [
        (' {"key": "literal-test-key"}', True),
        ('{"literal-test-key":"unrelated"}', False),
    ]:
        raw = credential_row(
            [
                {"role": "user", "content": "Get configuration."},
                call("get_config", {}),
                result("get_config", content),
            ]
        )
        raw["available_tools"] = credential_tools() + [
            tool("get_config", {"type": "object", "properties": {}})
        ]
        assert bool(run(raw)[0]) is accepted


def test_visible_resources_and_unlocks_never_disappear():
    names = [
        "blockscout-mcp-server-__unlock_blockchain_analysis__",
        "server-list_resources",
        "server-read_resource",
        "think",
    ]
    tools = [tool(n) for n in names]
    messages = [{"role": "user", "content": "Use the visible workflow."}]
    for name in names:
        messages.extend([call(name), result(name, "visible state")])
    messages.append(assistant("done"))
    state, _, _ = run(row(messages, tools), strip=True)
    assert state and len(state.batches) == 4 and len(state.sample.tools) == 4


def test_result_tool_mentions_do_not_grant_unsupported_dynamic_capability():
    # Qwen-Agent's pinned MCP wrapper registers the callable map before inference;
    # a tool-list-shaped resource is not an inspected dynamic-call protocol.
    raw = row(
        [
            {"role": "user", "content": "Discover"},
            call("list_tools", {}),
            result("list_tools", json.dumps({"tools": [tool("new_tool")]})),
            call("new_tool"),
            result("new_tool"),
            assistant(),
        ],
        [tool("list_tools", {"type": "object"})],
    )
    state, pipeline, _ = run(raw)
    assert state is None and pipeline.drops == {"undefined_function_calls": 1}


def test_proven_omitted_teacher_hint_is_quarantined_without_inventing_it():
    raw = row()
    raw["metadata"] = json.dumps(
        {
            "synthetic_data_gen_configs": [
                {"generation_params": {"agent": "qwen_agent", "enable_tool_hint": True}}
            ]
        }
    )
    state, pipeline, _ = run(raw)
    assert state is None and pipeline.drops == {"omitted_teacher_hint": 1}
    raw["metadata"] = "{}"
    assert run(raw)[0]


def test_recoverable_poll_error_does_not_erase_previously_observed_report():
    raw = job_row(
        [
            *start_pair(),
            *check_pair(),
            call(CHECK, {"taskId": "job-42"}),
            result(CHECK, "Research check error (504): timed out"),
        ]
    )
    state, _, _ = run(raw)
    assert state and len(state.batches) == 3


@pytest.mark.parametrize(
    "name,arguments",
    [
        (
            "ai咖提示词管理mcp-user_login",
            {"username": "test-user", "password": "literal-test-password"},
        ),
        ("user_login", {"username": "test-user", "password": "literal-test-password"}),
        (
            "tusclasesparticulares-automation-server-tusclasesparticulares_login",
            {"email": "user@example.test", "password": "literal-test-password"},
        ),
        (
            "tusclasesparticulares_login",
            {"email": "user@example.test", "password": "literal-test-password"},
        ),
        ("eve-online-market-data-server-authenticate", {"code": "literal-test-code"}),
    ],
)
def test_observed_auth_arguments_require_released_credentials(name, arguments):
    definition = tool(
        name,
        {
            "type": "object",
            "properties": {key: {"type": "string"} for key in arguments},
            "required": list(arguments),
        },
    )
    messages = [
        {"role": "user", "content": "Log in."},
        call(name, arguments),
        result(name, "Login failed."),
        assistant("Authentication failed."),
    ]
    state, pipeline, _ = run(row(messages, [definition]))
    assert state is None and pipeline.drops == {"ungrounded_credential": 1}
    messages[0]["content"] += " " + " ".join(arguments.values())
    state, _, _ = run(row(messages, [definition]))
    assert state and state.sample.messages[-2]["content"] == "Login failed."


@pytest.mark.parametrize("name", ["rednote-content-access-server-login", "login"])
def test_visible_argument_free_browser_auth_preserves_failure(name):
    definition = tool(
        name, {"type": "object", "properties": {}, "additionalProperties": False}
    )
    state, _, _ = run(
        row(
            [
                {"role": "user", "content": "Log in."},
                call(name, {}),
                result(name, "browserType.launch: Executable does not exist"),
                assistant("The browser could not start."),
            ],
            [definition],
        )
    )
    assert state


def train_tool(name="search"):
    return tool(
        name,
        {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "出发日期 格式：YYYY-MM-DD",
                },
                "fromCity": {"type": "string", "description": "出发城市"},
                "toCity": {"type": "string", "description": "到达城市"},
            },
            "required": ["date", "fromCity", "toCity"],
            "additionalProperties": False,
        },
        description="查询12306火车票",
    )


def train_row(prefix=(), name="search"):
    return row(
        [
            *prefix,
            {
                "role": "user",
                "content": "I need to travel from Beijing to Shanghai next Monday.",
            },
            assistant("", reasoning_content="Today is 2025-08-15."),
            call(name, {"date": "2025-08-18", "fromCity": "北京", "toCity": "上海"}),
            result(name, "No tickets found."),
            assistant("No tickets were found."),
        ],
        [train_tool(name)],
    )


@pytest.mark.parametrize("strip", [False, True])
@pytest.mark.parametrize("name", ["search", "12306-mcp-server-search"])
def test_train_relative_departure_cannot_use_assistant_invented_clock(strip, name):
    state, pipeline, _ = run(train_row(name=name), strip=strip)
    assert state is None and pipeline.drops == {"ungrounded_relative_date": 1}


@pytest.mark.parametrize(
    "context",
    [
        "Today's date: 2025-08-15.",
        "Current date is August 15, 2025.",
        "Today's date: Aug. 15, 2025.",
        "Travel date is 2025-08-18.",
        "Travel date is 18th August 2025.",
    ],
)
def test_train_visible_current_clock_or_requested_date_supplies_anchor(context):
    assert run(train_row([{"role": "system", "content": context}]))[0]


def test_train_unrelated_historical_date_and_schema_examples_are_not_a_clock():
    raw = train_row([{"role": "user", "content": "I was born on 1990-08-18."}])
    definitions = json.loads(raw["available_tools"])
    definitions[0]["function"]["parameters"]["examples"] = [{"date": "2025-08-18"}]
    raw["available_tools"] = json.dumps(definitions)
    raw["metadata"] = json.dumps({"current_date": "2025-08-15"})
    state, pipeline, _ = run(raw)
    assert state is None and pipeline.drops == {"ungrounded_relative_date": 1}


def test_train_context_rechecks_system_clock_removal_and_cannot_be_rescued():
    raw = train_row([{"role": "system", "content": "Today is 2025-08-15."}])
    state, pipeline, _ = run(raw, override="")
    assert state is None and pipeline.stage_drops == {"final_context": 1}
    state, pipeline, _ = run(train_row(), override="Today is 2025-08-15.")
    assert state is None and pipeline.stage_drops == {"source_policy": 1}


def test_train_unknown_prior_result_causes_bounded_abstention():
    raw = train_row(
        [
            {"role": "user", "content": "Read the trip context first."},
            call("read_trip", {}),
            result("read_trip", "Unclassified runtime observation."),
        ]
    )
    raw["available_tools"] = json.dumps(
        [train_tool(), tool("read_trip", {"type": "object"})]
    )
    assert run(raw)[0]


def test_unrelated_search_date_contract_is_not_train_calendar_policy():
    raw = train_row()
    definitions = json.loads(raw["available_tools"])
    definitions[0]["function"]["description"] = "Find historical events on a date."
    raw["available_tools"] = json.dumps(definitions)
    assert run(raw)[0]


def test_future_clock_result_cannot_ground_earlier_train_search():
    raw = train_row()
    messages = json.loads(raw["messages"])
    messages[-2]["content"] = '{"current_date":"2025-08-15"}'
    raw["messages"] = json.dumps(messages)
    assert run(raw)[1].drops == {"ungrounded_relative_date": 1}


def test_unknown_current_date_does_not_promote_later_historical_date():
    raw = train_row(
        [
            {
                "role": "system",
                "content": "Today is unknown. Historic event: August 15, 1990.",
            }
        ]
    )
    assert run(raw)[1].drops == {"ungrounded_relative_date": 1}
