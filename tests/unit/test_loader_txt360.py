"""TxT360's exact source contract, restoration, and final-view gates."""

from copy import deepcopy
from typing import Any
import json

import pytest

from fcanalysis.loaders import txt360
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.normalization import Reject
from fcanalysis.loaders.pipeline import (
    RowState,
    reconcile_definitions,
    validate_structure,
    validate_capabilities,
)


NO_CURATION = CurationConfig(max_level=None, audit=False)


def definition(name="lookup"):
    return {
        "name": name,
        "description": "A visible operation",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }


def row(messages):
    return {"messages": json.dumps(messages)}


def family(field="think"):
    initial = [
        {"role": "system", "tools": [definition()]},
        {"role": "user", "content": "Find one"},
    ]
    call = {
        "role": "assistant",
        "tool_calls": [{"name": "lookup", "arguments": '{"x":1}'}],
    }
    return [
        row(initial + [dict(call, **{field: "native call"})]),
        row(
            initial
            + [
                call,
                {"role": "tool", "content": "found", "tool_calls": []},
                {"role": "assistant", field: "native answer", "content": "Done"},
            ]
        ),
    ]


def run(monkeypatch, rows, split="high", filters=None, config=None):
    monkeypatch.setattr(txt360, "parquet_rows", lambda *a, **kw: iter(deepcopy(rows)))
    return txt360.load(
        split, config, filters or FilterConfig(), curation_config=NO_CURATION
    )


@pytest.mark.parametrize(
    ("split", "field"),
    [("high", "think"), ("medium", "think_fast"), ("low", "think_faster")],
)
def test_exact_restoration_before_native_removal(monkeypatch, split, field):
    rows = family(field)
    samples, report = run(monkeypatch, rows, split)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == 1
    assert sample.raw == rows[1]
    assert sample.messages[2]["reasoning_content"] == "native call"
    assert sample.messages[-1]["reasoning_content"] == "native answer"
    assert "content" not in sample.messages[0]
    assert report.raw_count == report.stage1_count == 2
    assert report.final_count == 1
    assert (
        report.dataset_config_transform_counts["reconstruction"]["restored_messages"]
        == 1
    )
    stripped, _ = run(monkeypatch, rows, split, FilterConfig(strip_thinking=True))
    assert all("reasoning_content" not in message for message in stripped[0].messages)
    assert stripped[0].raw == rows[1]


def test_empty_donor_is_positive_evidence(monkeypatch):
    rows = family()
    msgs = json.loads(rows[0]["messages"])
    msgs[-1]["think"] = ""
    rows[0] = row(msgs)
    samples, _ = run(monkeypatch, rows)
    assert samples[0].messages[2]["reasoning_content"] == ""


def test_missing_donor_quarantines_even_when_stripping(monkeypatch):
    samples, report = run(
        monkeypatch, family()[1:], filters=FilterConfig(strip_thinking=True)
    )
    assert not samples
    assert (
        report.dataset_config_transform_counts["reconstruction"]["missing_donor_rows"]
        == 1
    )


def test_conflicting_donors_are_not_chosen_by_adjacency(monkeypatch):
    first, last = family()
    alt = json.loads(first["messages"])
    alt[-1]["think"] = "conflicting"
    samples, report = run(monkeypatch, [first, last, row(alt)])
    assert not samples
    assert (
        report.dataset_config_transform_counts["reconstruction"]["ambiguous_donor_rows"]
        == 1
    )


def test_nonadjacent_matching_donor_and_source_order(monkeypatch):
    first, last = family()
    other = row(
        [
            {"role": "user", "content": "Other"},
            {"role": "assistant", "content": "Yes", "think": ""},
        ]
    )
    samples, _ = run(monkeypatch, [last, other, first])
    assert [s.sample_id for s in samples] == [0, 1]
    assert samples[0].messages[2]["reasoning_content"] == "native call"


@pytest.mark.parametrize("difference", ["system", "tools", "arguments", "result"])
def test_prefix_proof_keeps_exact_context(monkeypatch, difference):
    first, last = family()
    msgs = json.loads(last["messages"])
    if difference == "system":
        msgs[0]["content"] = "Changed"
    elif difference == "tools":
        msgs[0]["tools"][0]["description"] = "Changed"
    elif difference == "arguments":
        msgs[2]["tool_calls"][0]["arguments"] = '{"x":2}'
    else:
        # Results after a donor boundary do not distinguish the donor's action.
        msgs[3]["content"] = "changed result"
    samples, _ = run(monkeypatch, [first, row(msgs)])
    assert bool(samples) is (difference == "result")
    if samples:
        assert samples[0].messages[3]["content"] == "changed result"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m[-1].update(think_fast="wrong effort"),
        lambda m: m[-1].pop("think"),
        lambda m: m[-1].update(think=None),
        lambda m: m[1].update(tool_calls=[{"name": "lookup", "arguments": "{}"}]),
        lambda m: m[-1].update(unknown="visible"),
        lambda m: m[0].update(role="developer"),
    ],
)
def test_source_schema_violations_are_explicit(mutate):
    msgs = json.loads(family()[0]["messages"])
    mutate(msgs)
    with pytest.raises(Reject):
        txt360._prefix_record(row(msgs), 0, "high")


def test_missing_and_empty_system_content_remain_distinct():
    a = txt360._convert_row(
        row(
            [
                {"role": "system", "tools": []},
                {"role": "user", "content": "u"},
                {"role": "assistant", "think": "r", "content": "a"},
            ]
        ),
        0,
        "high",
    )
    b = txt360._convert_row(
        row(
            [
                {"role": "system", "tools": [], "content": ""},
                {"role": "user", "content": "u"},
                {"role": "assistant", "think": "r", "content": "a"},
            ]
        ),
        0,
        "high",
    )
    assert "content" not in a.messages[0]
    assert b.messages[0]["content"] == ""


def test_later_capability_epoch_is_not_frontloaded():
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "think": "", "content": "a"},
        {"role": "system", "tools": [definition()]},
        {"role": "assistant", "think": "", "content": "b"},
    ]
    with pytest.raises(Reject, match="unresolved_later_tool_environment"):
        txt360._convert_row(row([]), 0, "high", msgs)


def test_nested_raw_isolation():
    raw = family()[0]
    original = deepcopy(raw)
    sample = txt360._convert_row(raw, 0, "high")
    sample.tools[0]["function"]["parameters"]["properties"]["x"]["type"] = "string"
    sample.messages[-1]["tool_calls"][0]["function"]["arguments"] = "{}"
    assert raw == original


def test_exact_definitions_deduplicate_and_collisions_reject():
    raw = row(
        [
            {"role": "system", "tools": [definition(), definition()]},
            {"role": "user", "content": "u"},
            {"role": "assistant", "think": "", "content": "a"},
        ]
    )
    sample = txt360._convert_row(raw, 0, "high")
    state = RowState(sample)
    reconcile_definitions(state)
    assert len(sample.tools) == 1
    sample.tools.append(deepcopy(sample.tools[0]))
    sample.tools[-1]["function"]["description"] = "collision"
    with pytest.raises(Reject, match="conflicting_duplicate_tool_names"):
        reconcile_definitions(state)


@pytest.mark.parametrize("arguments", [None, [], 2, "null"])
def test_no_invented_arguments(arguments):
    msgs = json.loads(family()[0]["messages"])
    msgs[-1]["tool_calls"][0]["arguments"] = arguments
    with pytest.raises(Reject, match="non_object_arguments"):
        txt360._convert_row(row(msgs), 0, "high")


def test_null_parameters_not_invented():
    msgs = json.loads(family()[0]["messages"])
    msgs[0]["tools"][0]["parameters"] = None
    state = RowState(txt360._convert_row(row(msgs), 0, "high"))
    validate_structure(state)
    with pytest.raises(Reject):
        validate_capabilities(state)
    assert state.sample.tools[0]["function"]["parameters"] is None


@pytest.mark.parametrize("alter", ["missing", "orphan", "parallel"])
def test_unproven_call_result_linkage_is_quarantined(monkeypatch, alter):
    first, last = family()
    f = json.loads(first["messages"])
    m = json.loads(last["messages"])
    if alter == "missing":
        m.pop(3)
    elif alter == "orphan":
        m.insert(4, {"role": "tool", "content": "extra"})
    else:
        for messages in (f, m):
            messages[2]["tool_calls"].append({"name": "lookup", "arguments": '{"x":2}'})
        m.insert(4, {"role": "tool", "content": "second"})
    samples, report = run(monkeypatch, [row(f), row(m)])
    assert not samples
    assert report.filter_drop_reasons


@pytest.mark.parametrize("strip", [True, False])
def test_reasoning_only_final_is_not_a_complete_answer(monkeypatch, strip):
    raw = row(
        [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "think": "secret",
                "content": "<think>private</think>",
            },
        ]
    )
    samples, report = run(
        monkeypatch, [raw], filters=FilterConfig(strip_thinking=strip)
    )
    assert not samples
    assert report.filter_drop_reasons == {"empty_final_assistant": 1}


@pytest.mark.parametrize("strip", [True, False])
@pytest.mark.parametrize("native", ["", "Separate native reasoning"])
@pytest.mark.parametrize(
    "content",
    [
        "<think>\nLet me determine which tickets",
        "<think>Let me determine <think>the assistant can't use",
        "<think>First pass.</think> <reasoning>Another unfinished pass",
        "</think>",
        " \n<reasoning>First attempt<think>Another unfinished thought",
        "<think>A<think>B<think>C<think>D",
        "<think>Earlier thought</think><reasoning>Unfinished<think>Again",
        "<think>Literal close `</think>` followed by unfinished thought",
        " \n</think></reasoning> ",
        "<think></reasoning>",
    ],
)
def test_low_native_only_ambiguous_endpoint_is_incomplete(
    monkeypatch, strip, native, content
):
    raw = row(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Book tickets for the show."},
            {
                "role": "assistant",
                "think_faster": native,
                "content": content,
            },
        ]
    )
    before = deepcopy(raw)
    samples, report = run(
        monkeypatch,
        [raw],
        split="low",
        filters=FilterConfig(strip_thinking=strip),
    )
    assert not samples
    reason = "ambiguous_reasoning" if strip else "empty_final_assistant"
    assert report.filter_drop_reasons == {reason: 1}
    assert raw == before


@pytest.mark.parametrize(
    "content",
    [
        "A visible answer.</think>",
        "A visible answer.<think>Unresolved",
        "```text\n<think>\n```",
        "<think>Earlier thought</think>A visible answer.<think>Unresolved",
        "<think>Earlier thought</think>`<think>`",
        "<think>Unresolved mismatched boundary</reasoning>",
        "</think>A visible answer.<think>Unresolved",
    ],
)
def test_low_keep_preserves_other_unresolved_or_literal_native_text(
    monkeypatch, content
):
    raw = row(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "think_faster": "", "content": content},
        ]
    )
    samples, _ = run(
        monkeypatch,
        [raw],
        split="low",
        filters=FilterConfig(strip_thinking=False),
    )
    assert len(samples) == 1
    assert samples[0].messages[-1] == {
        "role": "assistant",
        "content": content,
        "reasoning_content": "",
    }
    assert samples[0].raw == raw


def test_visible_think_tool_is_preserved(monkeypatch):
    rows = family()
    for i, raw in enumerate(rows):
        msgs = json.loads(raw["messages"])
        msgs[0]["tools"][0]["name"] = "think"
        msgs[2]["tool_calls"][0]["name"] = "think"
        rows[i] = row(msgs)
    samples, _ = run(monkeypatch, rows, filters=FilterConfig(strip_thinking=True))
    assert samples[0].tools[0]["function"]["name"] == "think"
    assert samples[0].messages[2]["tool_calls"][0]["function"]["name"] == "think"


def test_final_system_override_rechecks_credential_evidence(monkeypatch):
    rows = family()
    for i, raw in enumerate(rows):
        msgs = json.loads(raw["messages"])
        msgs[0]["content"] = "The API key is abc-secret-123."
        msgs[0]["tools"][0]["name"] = "get_amazon_product_details"
        msgs[2]["tool_calls"][0]["name"] = "get_amazon_product_details"
        msgs[0]["tools"][0]["parameters"] = {
            "type": "object",
            "properties": {"api_key": {"type": "string"}},
        }
        msgs[2]["tool_calls"][0]["arguments"] = '{"api_key":"abc-secret-123"}'
        rows[i] = row(msgs)
    kept, _ = run(monkeypatch, rows)
    assert len(kept) == 1
    dropped, report = run(
        monkeypatch, rows, filters=FilterConfig(system_message_override="")
    )
    assert not dropped
    assert report.filter_drop_reasons["ungrounded_credential_argument"] == 1


@pytest.mark.parametrize("content", [None, "", "  ", [], 1])
def test_has_qualifying_user_requires_nonempty_string(content):
    assert not txt360._has_qualifying_user([{"role": "user", "content": content}])


def test_flattened_function_envelope_and_required_are_named_repairs():
    function = definition()
    function["type"] = "function"
    function["required"] = ["x"]
    converted = txt360._convert_tools([function])[0]
    assert converted == {"type": "function", "function": definition()}
    assert function["type"] == "function"
    function["required"] = []
    with pytest.raises(Reject, match="conflicting_legacy_required"):
        txt360._convert_tools([function])


def test_new_secret_generation_is_not_existing_credential_use(monkeypatch):
    rows = family()
    for i, raw in enumerate(rows):
        msgs = json.loads(raw["messages"])
        msgs[0]["tools"][0]["name"] = "create_api_key"
        msgs[0]["tools"][0]["parameters"] = {
            "type": "object",
            "properties": {"api_key": {"type": "string"}},
        }
        msgs[2]["tool_calls"][0] = {
            "name": "create_api_key",
            "arguments": '{"api_key":"newly-generated-value"}',
        }
        rows[i] = row(msgs)
    samples, _ = run(monkeypatch, rows)
    assert len(samples) == 1


def test_explicit_empty_reasoning_preserves_context_only_assistant(monkeypatch):
    first = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "context", "think": ""},
    ]
    second = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "context"},
        {"role": "assistant", "content": "answer", "think": "second"},
    ]
    samples, report = run(monkeypatch, [row(first), row(second)])
    assert len(samples) == 1
    assert [m["content"] for m in samples[0].messages] == ["u", "context", "answer"]
    assert samples[0].messages[1]["reasoning_content"] == ""
    assert (
        report.dataset_config_transform_counts["reconstruction"]["removed_prefix_rows"]
        == 1
    )


def test_ordinary_inline_tags_remain_exact_in_source_comparison():
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "Example `<think>`", "think": "private"},
    ]
    record = txt360._prefix_record(row(messages), 0, "high")
    assert record.projected_messages[-1] == {
        "role": "assistant",
        "content": "Example `<think>`",
    }


def test_default_strips_native_reasoning(monkeypatch):
    monkeypatch.setattr(txt360, "parquet_rows", lambda *a, **kw: iter(family()))
    samples, _ = txt360.load(curation_config=NO_CURATION)
    assert all("reasoning_content" not in m for m in samples[0].messages)


def test_invalid_flattened_envelope_type_and_empty_name_quarantine():
    raw = definition()
    raw["type"] = "other"
    with pytest.raises(Reject, match="invalid_flattened_function_type"):
        txt360._convert_tools([raw])
    raw["type"] = "function"
    raw["name"] = ""
    with pytest.raises(Reject, match="malformed_tool_definitions"):
        txt360._convert_tools([raw])


def credential_state(
    name, field, description, value, *, before=(), after=(), default=None
):
    """A canonical linked view isolates source policy from reconstruction tests."""
    from fcanalysis.format import ConversationSample

    schema = {"description": description}
    if default is not None:
        schema["default"] = default
    messages: list[dict[str, Any]] = [
        *before,
        {"role": "assistant", "content": "invented only here"},
        *after,
    ]
    state = RowState(
        ConversationSample(
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "parameters": {"type": "object", "properties": {field: schema}},
                    },
                }
            ],
            dataset=txt360.DATASET_ID,
            sample_id=0,
            raw={},
        )
    )
    messages[len(before)]["tool_calls"] = [
        {"function": {"name": name, "arguments": json.dumps({field: value})}}
    ]
    state.parsed_arguments[len(before), 0] = {field: value}
    return state


@pytest.mark.parametrize(
    ("field", "description"),
    [
        ("apikey", "The API key for authentication"),
        ("authorization", "Authorization token for the API request"),
        ("password", "The password for login in clear text"),
        ("authentication_token", "The authentication token for API access."),
        ("credentials", "Credentials required to access the asset."),
    ],
)
def test_audited_existing_descriptions_require_prior_evidence(field, description):
    state = credential_state(
        "source_consumer", field, description, "invented only here"
    )
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(state)
    state.sample.messages.insert(
        0, {"role": "user", "content": "Use invented only here"}
    )
    state.parsed_arguments = {(1, 0): {field: "invented only here"}}
    txt360._validate_visible_context(state)


@pytest.mark.parametrize(
    "result",
    ['{"code":123456}', '{"code":"123456"}', '"123456"'],
)
def test_numeric_otp_uses_exact_prior_result_scalar_values(result):
    description = (
        "Numeric verification code received from Telegram. Must be a positive integer "
        "(typically 5-6 digits). Note: Codes sent via Telegram's official application "
        "may expire immediately after being used by the client."
    )
    state = credential_state(
        "submit_code_telegram_submitcode_get",
        "otp",
        description,
        123456,
        before=[{"role": "tool", "content": result}],
    )
    txt360._validate_visible_context(state)


@pytest.mark.parametrize(
    ("before", "after", "value"),
    [
        ([{"role": "tool", "content": '{"123456":null}'}], [], 123456),
        ([{"role": "tool", "content": '{"code":true}'}], [], 1),
        ([{"role": "tool", "content": '{"code":null}'}], [], 123456),
        ([{"role": "tool", "content": '{"code":1234567}'}], [], 123456),
        ([{"role": "user", "content": "value1234567"}], [], 123456),
        (
            [{"role": "assistant", "content": "123456", "reasoning_content": "123456"}],
            [],
            123456,
        ),
        ([], [{"role": "tool", "content": '{"code":123456}'}], 123456),
    ],
)
def test_numeric_otp_rejects_ineligible_evidence(before, after, value):
    state = credential_state(
        "Verify Signup Code",
        "code",
        "The verification code received by the user",
        value,
        before=before,
        after=after,
    )
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(state)


def test_structured_existing_credentials_require_each_scalar():
    state = credential_state(
        "establish_secure_connection",
        "credentials",
        "Credentials required to access the asset.",
        {"username": "admin", "password": "not-given"},
        before=[{"role": "user", "content": "The username is admin"}],
    )
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(state)


@pytest.mark.parametrize(
    ("name", "field", "description", "value"),
    [
        (
            "sending_voice_otp_custom_otp",
            "otp",
            "4-digit numeric OTP code to be sent via voice call. Must be an integer between 0 and 9999, representing the verification code to be delivered to the recipient.",
            1234,
        ),
        (
            "create_account",
            "password",
            "The password for the new account",
            "new-secret",
        ),
        (
            "generate_password_hash",
            "password",
            "The password to be hashed",
            "new-secret",
        ),
        ("validatePassword", "password", "The password to be validated", "new-secret"),
        ("encrypt_data", "key", "The encryption key", "new-key"),
        (
            "get_token_price",
            "token",
            "The contract address of the token. Defaults to the address of the BUSD token.",
            "0x123",
        ),
        (
            "generate_basic_auth_header",
            "password",
            "The password for Basic Auth.",
            "new-secret",
        ),
        ("configureSSH", "password", "The password for SSH access.", "new-secret"),
    ],
)
def test_creation_and_ordinary_field_meanings_are_not_credentials(
    name, field, description, value
):
    txt360._validate_visible_context(credential_state(name, field, description, value))


def test_only_exact_released_public_constant_is_eligible():
    description = "API authentication key. Use 'test' for limited access (rate-limited) or obtain a premium key from https://ipdata.co/ for production use."
    txt360._validate_visible_context(
        credential_state("multi_language_support", "api_key", description, "test")
    )
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(
            credential_state(
                "multi_language_support", "api_key", description, "invented"
            )
        )
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(
            credential_state(
                "consumer",
                "apikey",
                "The API key for authentication",
                "example",
                default="example",
            )
        )


@pytest.mark.parametrize(
    "text", ["123456.0", "+123456", "-123456", "1,123456", "0.123456"]
)
def test_numeric_otp_rejects_numeric_fragments_in_user_text(text):
    state = credential_state(
        "Verify Signup Code",
        "code",
        "The verification code received by the user",
        123456,
        before=[{"role": "user", "content": text}],
    )
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(state)


def test_numeric_otp_allows_sentence_punctuation_and_preserves_leading_zeroes():
    state = credential_state(
        "Verify Signup Code",
        "code",
        "The verification code received by the user",
        "012345",
        before=[{"role": "user", "content": "The code is 012345."}],
    )
    txt360._validate_visible_context(state)
    state.parsed_arguments[1, 0] = {"code": "12345"}
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(state)


def low_bundle_family(*, two_calls=True):
    rows = family("think_faster")
    for i, raw in enumerate(rows):
        messages = json.loads(raw["messages"])
        if two_calls:
            messages[0]["tools"].append(definition("other"))
            messages[2]["tool_calls"].append({"name": "other", "arguments": '{"x":2}'})
        if i == 1:
            messages[3]["content"] = json.dumps(
                [
                    {
                        "name": call["name"],
                        "results": {"found": json.loads(call["arguments"])["x"]},
                    }
                    for call in messages[2]["tool_calls"]
                ]
            )
        rows[i] = row(messages)
    return rows


@pytest.mark.parametrize("strip", [False, True])
def test_low_bundle_requires_exact_context_and_preserves_full_arrays(
    monkeypatch, strip
):
    rows = low_bundle_family()
    samples, report = run(monkeypatch, rows, "low", FilterConfig(strip_thinking=strip))
    assert len(samples) == 1
    assert samples[0].raw == rows[-1]
    assert [m["role"] for m in samples[0].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [json.loads(m["content"]) for m in samples[0].messages[3:5]] == [
        [{"name": "lookup", "results": {"found": 1}}],
        [{"name": "other", "results": {"found": 2}}],
    ]
    assert all(m.keys() == {"role", "content"} for m in samples[0].messages[3:5])
    assert (
        report.dataset_config_transform_counts["pipeline_passed"]["final_linkage"] == 1
    )


@pytest.mark.parametrize(
    "issue", ["unrelated_name", "repeated_name", "reordered", "missing_entry"]
)
def test_low_unproved_multi_call_array_is_not_split(monkeypatch, issue):
    rows = low_bundle_family()
    messages = json.loads(rows[-1]["messages"])
    entries = json.loads(messages[3]["content"])
    if issue == "unrelated_name":
        entries[0]["name"] = "person name"
    elif issue == "repeated_name":
        entries[1]["name"] = entries[0]["name"]
    elif issue == "reordered":
        entries.reverse()
    else:
        entries.pop()
    messages[3]["content"] = json.dumps(entries)
    rows[-1] = row(messages)
    samples, report = run(monkeypatch, rows, "low")
    assert not samples
    assert report.filter_drop_reasons["unbalanced_cardinality"] == 2


def test_low_unmatched_singleton_array_stays_an_ordinary_result(monkeypatch):
    rows = low_bundle_family(two_calls=False)
    messages = json.loads(rows[-1]["messages"])
    entries = [{"name": "ordinary entity", "results": {"found": 1}}]
    messages[3]["content"] = json.dumps(entries)
    rows[-1] = row(messages)
    samples, report = run(monkeypatch, rows, "low")
    assert len(samples) == 1
    assert samples[0].messages[3]["content"] == json.dumps(entries)
    assert (
        "source_result_bundles_expanded"
        not in report.dataset_config_transform_counts["transformations"]
    )


def test_low_batch_proof_does_not_apply_to_other_efforts(monkeypatch):
    rows = [
        row(json.loads(raw["messages"].replace('"think_faster"', '"think"')))
        for raw in low_bundle_family()
    ]
    samples, report = run(monkeypatch, rows, "high")
    assert not samples
    assert report.filter_drop_reasons["unbalanced_cardinality"] == 2


def test_low_result_evidence_uses_contextual_proof_not_shape():
    from fcanalysis.loaders._txt360_results import proved_results, result_values

    tool_message = {
        "role": "tool",
        "content": json.dumps([{"name": "a", "results": "data"}]),
    }
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "a", "arguments": "{}"}}],
        },
        tool_message,
    ]
    proof = proved_results(messages, split="low")
    assert list(
        result_values(json.loads(tool_message["content"]), tool_message, proved=proof)
    ) == ["data"]
    messages[0]["tool_calls"][0]["function"]["name"] = "ordinary_lookup"
    proof = proved_results(messages, split="low")
    assert not proof
    assert list(
        result_values(json.loads(tool_message["content"]), tool_message, proved=proof)
    ) == ["a", "data"]


def test_known_low_cross_call_serialization_damage_stays_quarantined(monkeypatch):
    wordle = "Get Today's Word"
    champion = "Get League of Legends Champion Meta Data"
    rows = family("think_faster")
    for index, raw in enumerate(rows):
        messages = json.loads(raw["messages"])
        messages[0]["tools"][0]["name"] = wordle
        messages[0]["tools"][0]["parameters"] = {
            "type": "object",
            "properties": {"difficulty": {"type": "string"}},
            "required": ["difficulty"],
        }
        messages[2]["tool_calls"][0] = {
            "name": wordle,
            "arguments": json.dumps(
                {
                    "difficulty": 'difficult\\"), ' + champion + '(rankname=\\"gold',
                    "name": "Zed",
                }
            ),
        }
        if index == 1:
            messages[3]["content"] = json.dumps(
                [
                    {"name": wordle, "results": {"todayWord": "xylophone"}},
                    {"name": champion, "results": {"roles": ["Assassin"]}},
                ]
            )
        rows[index] = row(messages)
    samples, report = run(monkeypatch, rows, "low")
    assert not samples
    assert (
        report.dataset_config_drop_reasons["corrupted_source_call_serialization"] == 1
    )


def test_code_looking_arguments_do_not_generically_imply_corruption():
    from fcanalysis.loaders._txt360_results import expanded_results

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "code_example",
                    "arguments": '{"text":"Get League of Legends Champion Meta Data(rankname=gold)"}',
                }
            ],
        },
        {"role": "tool", "content": '[{"name":"unrelated","results":"literal code"}]'},
    ]
    assert expanded_results(messages, split="low") == {}


def test_low_primitive_schema_aliases_preserve_constraints_and_instance_data(
    monkeypatch,
):
    from collections import Counter

    rows = family("think_faster")
    parameter_schema = {
        "type": "dict",
        "properties": {
            "x": {"type": "int", "minimum": 1, "maximum": 3},
            "y": {"type": "float", "default": {"type": "int"}},
        },
        "required": ["x"],
        "additionalProperties": False,
    }
    for index, raw in enumerate(rows):
        messages = json.loads(raw["messages"])
        messages[0]["tools"][0]["parameters"] = parameter_schema
        rows[index] = row(messages)
    untouched = deepcopy(rows)
    samples, _ = run(monkeypatch, rows, "low")
    assert len(samples) == 1
    schema = samples[0].tools[0]["function"]["parameters"]
    assert schema["type"] == "object"
    assert schema["properties"]["x"] == {"type": "integer", "minimum": 1, "maximum": 3}
    assert schema["properties"]["y"] == {"type": "number", "default": {"type": "int"}}
    assert schema["additionalProperties"] is False
    assert rows == untouched
    assert samples[0].raw == untouched[-1]
    counts = Counter()
    txt360._convert_tools(
        [dict(definition(), parameters=parameter_schema)], counts, split="low"
    )
    assert counts["source_schema_type_aliases"] == 3
    for index, raw in enumerate(rows):
        messages = json.loads(raw["messages"])
        messages[2]["tool_calls"][0]["arguments"] = '{"x":4}'
        rows[index] = row(messages)
    samples, report = run(monkeypatch, rows, "low")
    assert not samples
    assert report.filter_drop_reasons["invalid_arguments"] == 1


@pytest.mark.parametrize("kind", ["any", "List[int]", {}, ["int", "null"]])
def test_unknown_schema_types_remain_unsupported(kind):
    function = definition()
    function["parameters"]["properties"]["x"]["type"] = kind
    converted = txt360._convert_tools([function], split="low")
    assert converted[0]["function"]["parameters"]["properties"]["x"]["type"] == kind
    from fcanalysis.format import ConversationSample

    state = RowState(
        ConversationSample(dataset="test", sample_id=0, messages=[], tools=converted)
    )
    with pytest.raises(Reject):
        validate_capabilities(state)


@pytest.mark.parametrize("split", ["high", "medium"])
def test_low_alias_policy_does_not_change_other_efforts(split):
    function = definition()
    function["parameters"]["properties"]["x"]["type"] = "int"
    converted = txt360._convert_tools([function], split=split)
    assert converted[0]["function"]["parameters"]["properties"]["x"]["type"] == "int"


def calendar_family(
    user="Schedule an Uber ride next Monday at 8 AM.", *, system="", split="low"
):
    field = {"low": "think_faster", "medium": "think_fast", "high": "think"}[split]
    rows = family(field)
    changed = []
    for raw in rows:
        messages = json.loads(raw["messages"])
        function = messages[0]["tools"][0]
        function.update(
            name="request_uber_ride",
            description="Request an Uber ride to be scheduled for a specific time and location.",
            parameters={
                "type": "object",
                "properties": {
                    "pickup_time": {
                        "type": "string",
                        "description": "The date and time for the pickup, in ISO 8601 format.",
                        "examples": ["2023-04-09"],
                    }
                },
                "required": ["pickup_time"],
            },
        )
        messages[0]["content"] = system
        messages[1]["content"] = user
        messages[2]["tool_calls"][0].update(
            name=function["name"],
            arguments='{"pickup_time":"2023-04-10T08:00:00-04:00"}',
        )
        if field in messages[2]:
            messages[2][field] = "Today is 2023-04-09, so next Monday is April 10."
        changed.append(row(messages))
    return changed


@pytest.mark.parametrize("strip", [True, False])
def test_calendar_rejects_invented_ride_date_before_results(monkeypatch, strip):
    rows = calendar_family()
    before = deepcopy(rows)
    samples, report = run(monkeypatch, rows, "low", FilterConfig(strip_thinking=strip))
    assert not samples
    assert report.filter_drop_reasons["ungrounded_temporal_reference"] == 1
    assert rows == before


@pytest.mark.parametrize(
    ("user", "system"),
    [
        ("Schedule an Uber next Monday at 8 AM.", "Current date: 2023-04-09"),
        ("Schedule an Uber next Monday, April 10, 2023, at 8 AM.", ""),
        ("Schedule an Uber next Monday, April 10, at 8 AM.", ""),
        ("Schedule an Uber next month, April 2023, at 8 AM.", ""),
        ("Schedule an Uber next Monday (10/04/2023) at 8 AM.", ""),
    ],
)
def test_calendar_keeps_explicit_clock_and_unsupported_date_forms(
    monkeypatch, user, system
):
    rows = calendar_family(user, system=system)
    samples, _ = run(monkeypatch, rows, "low")
    assert len(samples) == 1
    assert samples[0].raw == rows[-1]


def test_calendar_rechecks_visible_clock_after_system_override(monkeypatch):
    rows = calendar_family(system="Current date: 2023-04-09")
    samples, report = run(
        monkeypatch,
        rows,
        "low",
        FilterConfig(system_message_override="You are helpful."),
    )
    assert not samples
    assert report.filter_drop_reasons["ungrounded_temporal_reference"] == 1


def test_calendar_policy_requires_exact_low_definition_contract(monkeypatch):
    for split in ("high", "medium"):
        samples, _ = run(monkeypatch, calendar_family(split=split), split)
        assert len(samples) == 1
    changed = []
    for raw in calendar_family():
        messages = json.loads(raw["messages"])
        messages[0]["tools"][0]["description"] = "Generate fictional ride examples."
        changed.append(row(messages))
    samples, _ = run(monkeypatch, changed, "low")
    assert len(samples) == 1


def high_calendar_family(*, split="high", schema_default=None, system=""):
    rows = calendar_family(
        "Check the Philadelphia Eagles live score today.", system=system, split=split
    )
    changed = []
    for raw in rows:
        messages = json.loads(raw["messages"])
        function = messages[0]["tools"][0]
        function.update(
            name="american_football_livescores",
            description="Retrieves live scores, game status updates, and match statistics for ongoing American football games at professional (NFL) and college (NCAA) levels. Use this function to get real-time sports data including current scores, quarter/time progress, and game highlights.",
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Filter matches by date (format: YYYY-MM-DD). If not provided, defaults to current date.",
                        "default": schema_default,
                    }
                },
                "required": ["date"],
            },
        )
        messages[2]["tool_calls"][0].update(
            name=function["name"], arguments='{"date":"2023-12-25"}'
        )
        changed.append(row(messages))
    return changed


@pytest.mark.parametrize("strip", [True, False])
def test_high_calendar_rejects_missing_clock_and_preserves_raw(monkeypatch, strip):
    rows = high_calendar_family()
    original = deepcopy(rows)
    samples, report = run(monkeypatch, rows, "high", FilterConfig(strip_thinking=strip))
    assert not samples
    assert report.filter_drop_reasons["ungrounded_temporal_reference"] == 1
    assert rows == original


def test_high_calendar_requires_exact_effort_and_parameter_schema(monkeypatch):
    for split in ("low", "medium"):
        samples, _ = run(monkeypatch, high_calendar_family(split=split), split)
        assert len(samples) == 1
    # A declared static date changes the source contract: the date may simply
    # have been copied from that default, which is outside the clock gate.
    samples, _ = run(
        monkeypatch, high_calendar_family(schema_default="2023-12-25"), "high"
    )
    assert len(samples) == 1


def test_high_calendar_rechecks_system_clock_after_override(monkeypatch):
    rows = high_calendar_family(system="Current date: 2023-12-25")
    samples, _ = run(monkeypatch, rows, "high")
    assert len(samples) == 1
    samples, report = run(
        monkeypatch,
        rows,
        "high",
        FilterConfig(system_message_override="You are helpful."),
    )
    assert not samples
    assert report.filter_drop_reasons["ungrounded_temporal_reference"] == 1


@pytest.mark.parametrize(
    ("name", "field", "description", "value"),
    [
        (
            "mutual_funds",
            "x_mashape_key",
            "API authentication key for accessing financial data services",
            "YOUR_API_KEY",
        ),
        (
            "receivenotification",
            "apitokeninstance",
            "Authentication token for API access. This secure token must be generated through the Green API dashboard and grants authorized access to receive notifications for the specified WhatsApp instance.",
            "invented-token",
        ),
        (
            "verify_sms_otp_input",
            "otp_input",
            "User-entered one-time password (e.g., '123456'). Must match the format and length of the sent OTP.",
            123456,
        ),
        (
            "account",
            "api_secret",
            "The ConvertKit account's API secret used for authentication. This sensitive value is typically found in your ConvertKit account settings under API credentials.",
            "invented-secret",
        ),
        (
            "GetQRcode",
            "apiTokenInstance",
            "The API token for the user's account",
            "invented-token",
        ),
        (
            "healthData.updateMedicalHistory",
            "authToken",
            "The authentication token to access the healthcare database. This should be a string of alphanumeric characters.",
            "invented-token",
        ),
    ],
)
def test_additional_declared_credential_spellings_require_visible_evidence(
    name, field, description, value
):
    state = credential_state(name, field, description, value)
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(state)
    grounded = credential_state(
        name,
        field,
        description,
        value,
        before=[{"role": "user", "content": f"Use {value}."}],
    )
    txt360._validate_visible_context(grounded)


def test_additional_credential_demo_constant_is_exact():
    description = (
        "The RapidAPI key for accessing the `data_visualisation_` API. "
        "Defaults to 'demo'."
    )
    txt360._validate_visible_context(
        credential_state("getting_data", "x_rapidapi_key", description, "demo")
    )
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(
            credential_state("getting_data", "x_rapidapi_key", description, "DEMO")
        )


@pytest.mark.parametrize(
    ("name", "field", "description", "value"),
    [
        (
            "new88",
            "user_credentials",
            "Authentication details required for account-related actions",
            {"username": "new-user", "password": "new-password"},
        ),
        (
            "calling_api_method",
            "oauth_signature",
            "HMAC-SHA1 signature for request authentication",
            "computed-signature",
        ),
        (
            "airlines_marketing_names",
            "md5apikey",
            "MD5-hashed API key for authenticating with the Airhex API. Users must register at airhex.com to obtain an API key, then convert it to an MD5 hash for this parameter. Example format: '5f4dcc3b5aa765d61d8327deb882cf99'",
            "derived-from-supplied-key",
        ),
    ],
)
def test_mixed_creation_and_computed_authentication_values_are_not_literal_tokens(
    name, field, description, value
):
    txt360._validate_visible_context(credential_state(name, field, description, value))


def test_additional_structured_credentials_require_all_supplied_values():
    arguments = {
        "salesforce": {"client_id": "client123", "client_secret": "secret456"},
        "pega": {"api_key": "pega789"},
    }
    description = "Authentication details for both Salesforce and Pega Platform."
    with pytest.raises(Reject, match="ungrounded_credential_argument"):
        txt360._validate_visible_context(
            credential_state(
                "sync_salesforce_data",
                "authentication_details",
                description,
                arguments,
                before=[{"role": "user", "content": "Use client123 and pega789."}],
            )
        )
    txt360._validate_visible_context(
        credential_state(
            "sync_salesforce_data",
            "authentication_details",
            description,
            arguments,
            before=[
                {"role": "user", "content": "Use client123, secret456 and pega789."}
            ],
        )
    )


@pytest.mark.parametrize(
    ("name", "field", "description"),
    [
        ("register_device", "authToken", "The authentication token for the device."),
        ("generate_basic_auth_header", "username", "The username for Basic Auth."),
        (
            "subscribeToWebhookEvents",
            "secret",
            "A shared secret key for securing webhook communication.",
        ),
    ],
)
def test_newly_assigned_or_encoded_credentials_do_not_require_prior_state(
    name, field, description
):
    txt360._validate_visible_context(
        credential_state(name, field, description, "newly-chosen-value")
    )


def test_unsupported_static_default_contracts_are_held_without_public_bypasses():
    from fcanalysis.loaders._txt360_credentials import (
        _EXISTING_DESCRIPTIONS,
        _PUBLIC_ACCESS_CONSTANTS,
    )

    description = (
        "The API key for authenticating with the Financial Modeling Prep API. "
        "Default is 'rapidapi'."
    )
    for value in ("rapidapi", "a-different-token"):
        txt360._validate_visible_context(
            credential_state("stock_quote_price", "apikey", description, value)
        )
    assert description not in _EXISTING_DESCRIPTIONS["apikey"]
    assert not any(
        "RandomWordGenerator.com" in text
        for texts in _EXISTING_DESCRIPTIONS.values()
        for text in texts
    )
    assert {value for _, _, value in _PUBLIC_ACCESS_CONSTANTS} == {
        "demo",
        "test",
        "TokenDemoRapidapi",
    }
