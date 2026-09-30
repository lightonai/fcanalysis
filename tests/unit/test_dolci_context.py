"""Adversarial visible-prefix tests for the audited Dolci SimFC contracts."""

from copy import deepcopy
import json

import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.dolci_context import SIMFC_SOURCE, validate_simfc_credentials
from fcanalysis.loaders.normalization import Reject
from fcanalysis.loaders.pipeline import RowState, link_calls, validate_structure


def _tool(
    name="loginuser", argument="password", description="The password for logging in."
):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    argument: {
                        "type": "string",
                        "description": description,
                        "default": "",
                    }
                },
            },
        },
    }


def _call(name="loginuser", arguments=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments or {"password": "secret123"}),
                },
            }
        ],
    }


def _state(messages=None, tools=None, source=SIMFC_SOURCE):
    sample = ConversationSample(
        messages=deepcopy(
            messages
            or [
                {"role": "user", "content": "Please log me in."},
                _call(),
                {"role": "tool", "content": '{"ok":true}'},
                {"role": "assistant", "content": "Done"},
            ]
        ),
        tools=deepcopy(tools or [_tool()]),
        dataset="Dolci",
        sample_id="test",
        raw={"dataset_source": source},
    )
    state = RowState(sample)
    validate_structure(state)
    link_calls(state, positional=True, parallel=True)
    return state


def _reject(state):
    with pytest.raises(Reject, match="dolci_unresolved_credential_context"):
        validate_simfc_credentials(state)


@pytest.mark.parametrize(
    "name,argument,description,value,user",
    [
        (
            "loginuser",
            "password",
            "The password for logging in.",
            "mypassword",
            "I want to log in to my Pet Store account.",
        ),
        (
            "dashboard",
            "api_key",
            "The API key associated with the user account.",
            "abcd1234",
            "Can you fetch the dashboard for the survey with the code 'XYZ123'?",
        ),
        (
            "fetchOauthAuthorizedApplications",
            "access_token",
            "The access token for the OAuth application",
            "your_access_token",
            "Fetch my authorized OAuth applications. I have my credentials ready.",
        ),
    ],
)
def test_pinned_source_hidden_credential_patterns_quarantine(
    name, argument, description, value, user
):
    # Reduced from pinned shard/row 0/203, 1/2700 and 2/23067 respectively.
    state = _state(
        [
            {"role": "user", "content": user},
            _call(name, {argument: value}),
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Done"},
        ],
        [_tool(name, argument, description)],
    )
    _reject(state)


@pytest.mark.parametrize("role", ["user", "system"])
def test_supplied_credential_and_source_contents_survive(role):
    state = _state(
        [
            {"role": role, "content": 'Use password: "secret123".'},
            {"role": "user", "content": "Please log me in."},
            _call(),
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Done"},
        ]
    )
    before = deepcopy(state.sample)
    validate_simfc_credentials(state)
    assert state.sample == before


@pytest.mark.parametrize(
    "prefix",
    [
        {"role": "assistant", "content": "I will use secret123."},
        {"role": "assistant", "content": "<think>The password is secret123.</think>"},
        {"role": "user", "content": "The other password is secret1234."},
    ],
)
def test_assistant_invention_and_substring_are_not_evidence(prefix):
    state = _state(
        [
            {"role": "user", "content": "Log me in"},
            prefix,
            _call(),
            {"role": "tool", "content": '{"password":"secret123"}'},
            {"role": "assistant", "content": "Done"},
        ]
    )
    _reject(state)


def _result_then_login(result):
    return _state(
        [
            {"role": "user", "content": "Obtain my credential then log me in."},
            _call("credential_store", {"account": "me"}),
            {"role": "tool", "content": result},
            _call(),
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Done"},
        ],
        [_tool(), _tool("credential_store", "account", "Account to retrieve")],
    )


def test_earlier_linked_result_string_value_is_visible_evidence():
    validate_simfc_credentials(_result_then_login('{"data":{"password":"secret123"}}'))


@pytest.mark.parametrize(
    "result",
    [
        '{"secret123":true}',
        '{"message":"The password is secret123."}',
        '{"password":"secret1234"}',
        "secret123",
        '{"password":"secret123","password":"something_else"}',
    ],
)
def test_result_key_prose_substring_and_ambiguous_json_do_not_establish_value(result):
    _reject(_result_then_login(result))


def test_same_parallel_batch_cannot_use_its_future_result():
    calls = _call("credential_store", {"account": "me"})
    calls["tool_calls"].extend(_call()["tool_calls"])
    state = _state(
        [
            {"role": "user", "content": "Obtain my password then log in"},
            calls,
            {"role": "tool", "content": '{"password":"secret123"}'},
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Done"},
        ],
        [_tool(), _tool("credential_store", "account", "Account to retrieve")],
    )
    _reject(state)


def test_deliberately_wrong_password_without_specified_test_literal_is_uncertain():
    state = _state(
        [
            {"role": "user", "content": "Try logging in with a wrong password."},
            _call(),
            {"role": "tool", "content": '{"ok":false}'},
            {"role": "assistant", "content": "That password failed"},
        ]
    )
    _reject(state)
    state.sample.messages[0]["content"] = "Try the wrong password secret123."
    validate_simfc_credentials(state)


@pytest.mark.parametrize(
    "name,argument,description",
    [
        ("newAddress", "password", "The password for the new Ethereum address"),
        (
            "generate_password_strength_report",
            "password",
            "The password to be analyzed",
        ),
        ("get_balance", "token", "The token for which the balance is to be retrieved."),
        (
            "playlist",
            "token",
            "Pagination token for retrieving additional results. Default is None.",
        ),
    ],
)
def test_unrelated_source_contracts_are_not_keyword_filtered(
    name, argument, description
):
    state = _state(
        [
            {"role": "user", "content": "Please perform this task."},
            _call(name, {argument: "chosen_value"}),
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "Done"},
        ],
        [_tool(name, argument, description)],
    )
    validate_simfc_credentials(state)


def test_source_scope_and_exact_function_name_are_required():
    validate_simfc_credentials(_state(source="science-source"))
    state = _state()
    state.sample.tools[0]["function"]["name"] = "not_audited.loginuser"
    state.sample.messages[1]["tool_calls"][0]["function"]["name"] = (
        "not_audited.loginuser"
    )
    validate_simfc_credentials(state)


@pytest.mark.parametrize("value", [None, ""])
def test_explicit_absence_does_not_invent_a_credential(value):
    state = _state()
    state.parsed_arguments[(1, 0)]["password"] = value
    validate_simfc_credentials(state)


@pytest.mark.parametrize("parameter", [True, False])
def test_boolean_parameter_schemas_do_not_crash_contract_inspection(parameter):
    state = _state()
    state.sample.tools[0]["function"]["parameters"]["properties"]["password"] = (
        parameter
    )
    validate_simfc_credentials(state)


def _chart_state(user, value="2022-05-07"):
    tool = _tool(
        "bolivia_songs",
        "date",
        "The date for which the chart information is required, in the format YYYY-MM-DD.",
    )
    tool["function"]["description"] = (
        "Fetches the Bolivia Songs chart information for a given range and date."
    )
    tool["function"]["parameters"]["properties"]["date"]["default"] = "2022-05-07"
    return _state(
        [
            {"role": "user", "content": user},
            _call("bolivia_songs", {"date": value}),
            {"role": "tool", "content": '{"songs":[]}'},
            {"role": "assistant", "content": "Done"},
        ],
        [tool],
    )


@pytest.mark.parametrize(
    "user",
    [
        "Get the chart for 2022-05-07.",
        "Get the chart for May 7, 2022.",
        "Get the chart for May 7th 2022.",
        "Get the chart for 7 May 2022.",
        "Get the chart for 7th May, 2022.",
    ],
)
def test_explicit_calendar_date_normalization_is_nondestructive(user):
    state = _chart_state(user)
    before = deepcopy(state.sample)
    validate_simfc_credentials(state)
    assert state.sample == before


@pytest.mark.parametrize(
    "user",
    [
        "Get the chart for today.",
        "Get the chart for May 7.",
        "Get the chart for 05/07/2022.",
        "Get the chart for May 7st, 2022.",
        "Get the chart for any date you choose.",
    ],
)
def test_relative_ambiguous_or_unspecified_dates_are_quarantined(user):
    with pytest.raises(Reject, match="dolci_unresolved_date_context"):
        validate_simfc_credentials(_chart_state(user))


def test_raw_bolivia_songs_today_pattern_without_clock_quarantines():
    # Pinned shard0 row108 msg20: earlier results give release/concert dates,
    # but no current clock; the source synthesizes an unsupported June10 date.
    state = _chart_state(
        "What is the release date of MAÑANA SERÁ BONITO and can you list the top 3 Bolivia Songs for today?",
        "2024-06-10",
    )
    with pytest.raises(Reject, match="dolci_unresolved_date_context"):
        validate_simfc_credentials(state)


def test_earlier_linked_clock_value_can_ground_chart_date():
    state = _chart_state("Get the clock and then today's chart.", "2024-06-10")
    state.sample.messages[1:1] = [
        _call("clock", {}),
        {"role": "tool", "content": '{"today":"2024-06-10"}'},
    ]
    state = _state(state.sample.messages, state.sample.tools + [_tool("clock")])
    validate_simfc_credentials(state)


def test_later_clock_and_assistant_date_invention_are_not_evidence():
    state = _chart_state("Get today's chart.", "2024-06-10")
    state.sample.messages[1:1] = [
        {"role": "assistant", "content": "Today is June 10, 2024."}
    ]
    state.sample.messages[-2]["content"] = '{"date":"2024-06-10"}'
    state = _state(state.sample.messages, state.sample.tools)
    with pytest.raises(Reject, match="dolci_unresolved_date_context"):
        validate_simfc_credentials(state)


def test_invalid_calendar_date_is_not_normalized_into_validity():
    with pytest.raises(Reject, match="dolci_unresolved_date_context"):
        validate_simfc_credentials(_chart_state("Get February 30, 2024.", "2024-02-30"))


def _instagram_tool():
    tool = _tool(
        "instagram.followers",
        "user_id",
        "The ID of the Instagram user whose followers are to be retrieved.",
    )
    tool["function"]["description"] = (
        "Retrieves the list of followers for a given Instagram user."
    )
    tool["function"]["parameters"]["properties"]["user_id"]["default"] = "25025320"
    return tool


def _upload_tool():
    return {
        "type": "function",
        "function": {
            "name": "temp_upload_urls",
            "description": "Generates temporary upload URLs for a given video on a specific account using the provided source name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "default": "",
                        "description": "The unique identifier for the video.",
                    },
                    "account_id": {
                        "type": "string",
                        "default": "",
                        "description": "The unique identifier for the account.",
                    },
                },
            },
        },
    }


def test_tiktok_handle_does_not_establish_instagram_identifier():
    # Pinned shard0 row9200: TikTok metadata has no Instagram identifier.
    state = _state(
        [
            {"role": "user", "content": "Get TikTok metadata for 'charlidamelio'."},
            _call("user.metadata_information", {"username": "charlidamelio"}),
            {
                "role": "tool",
                "content": '{"followers":15000000,"followings":100,"description":"TikTok dancer"}',
            },
            {
                "role": "user",
                "content": "How many followers does she have on Instagram?",
            },
            _call("instagram.followers", {"user_id": "charlidamelio_instagram_id"}),
            {"role": "tool", "content": '{"followers":30000000}'},
            {"role": "assistant", "content": "She has thirty million followers"},
        ],
        [_instagram_tool()],
    )
    with pytest.raises(Reject, match="dolci_unresolved_identifier_context"):
        validate_simfc_credentials(state)


@pytest.mark.parametrize(
    "result",
    [
        '{"email":"john.doe@example.com"}',
        '{"email":"john.doe@example.com","video_id":"video789"}',
        '{"email":"john.doe@example.com","account_id":"account456"}',
    ],
)
def test_email_or_only_one_id_cannot_establish_video_and_account_identifiers(result):
    # Pinned shard0 row31477 provides only valid:true and an email before upload.
    state = _state(
        [
            {
                "role": "user",
                "content": "Get an email and temporary video upload URLs.",
            },
            _call("fetch_email_of_a_person", {"person": "John"}),
            {"role": "tool", "content": result},
            _call(
                "temp_upload_urls", {"video_id": "video789", "account_id": "account456"}
            ),
            {"role": "tool", "content": '{"url":"https://example.com/upload"}'},
            {"role": "assistant", "content": "Done"},
        ],
        [_upload_tool()],
    )
    with pytest.raises(Reject, match="dolci_unresolved_identifier_context"):
        validate_simfc_credentials(state)


@pytest.mark.parametrize("user_id", ["", "12345"])
def test_empty_or_assistant_invented_identifier_is_not_grounded(user_id):
    state = _state(
        [
            {"role": "user", "content": "Get Instagram followers."},
            {"role": "assistant", "content": f"The ID is {user_id}."},
            _call("instagram.followers", {"user_id": user_id}),
            {"role": "tool", "content": json.dumps({"user_id": user_id})},
            {"role": "assistant", "content": "Done"},
        ],
        [_instagram_tool()],
    )
    with pytest.raises(Reject, match="dolci_unresolved_identifier_context"):
        validate_simfc_credentials(state)


@pytest.mark.parametrize(
    "result",
    [
        '{"instagram_id":"12345"}',
        '{"data":{"instagram_id":"12345"}}',
    ],
)
def test_linked_identifier_result_can_ground_later_call_without_content_changes(result):
    state = _state(
        [
            {
                "role": "user",
                "content": "Look up the Instagram identifier and fetch followers.",
            },
            _call("lookup_user", {"handle": "user"}),
            {"role": "tool", "content": result},
            _call("instagram.followers", {"user_id": "12345"}),
            {"role": "tool", "content": '{"followers":42}'},
            {"role": "assistant", "content": "Done"},
        ],
        [_instagram_tool()],
    )
    before = deepcopy(state.sample)
    validate_simfc_credentials(state)
    assert state.sample == before


def test_user_supplied_upload_identifiers_are_preserved():
    state = _state(
        [
            {
                "role": "user",
                "content": "Get upload URLs for video video789 in account account456.",
            },
            _call(
                "temp_upload_urls", {"video_id": "video789", "account_id": "account456"}
            ),
            {"role": "tool", "content": '{"url":"https://example.com/upload"}'},
            {"role": "assistant", "content": "Done"},
        ],
        [_upload_tool()],
    )
    validate_simfc_credentials(state)
