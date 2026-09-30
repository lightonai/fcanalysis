from copy import deepcopy

import pytest

from fcanalysis.loaders.context import (
    calendar_dates,
    collect_string_values,
    contains_literal_token,
    iter_call_evidence,
    json_scalar_values,
    validate_argument_evidence,
)
from fcanalysis.loaders.normalization import Reject
from fcanalysis.loaders.pipeline import RowState
from fcanalysis.format import ConversationSample


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2024-02-29T10:20:30Z; 2023-02-29; 2025-13-01", {"2024-02-29"}),
        ("September 1st, 2025 and 1 Sept. 2025", {"2025-09-01"}),
        ("On 15 August 2025; AUG 16, 2025.", {"2025-08-15", "2025-08-16"}),
        ("tomorrow, next Monday, August 15, 08/09/2025, 9.8.2025", set()),
        ("202500-01-01; 2025-01-010; 0000-01-01; February 30, 2025", set()),
    ],
)
def test_calendar_evidence_needs_unambiguous_full_valid_dates(text, expected):
    assert calendar_dates(text) == expected


def _evidence_state(prefix, *, secret=123456):
    messages = [
        *prefix,
        {
            "role": "assistant",
            "content": "I know 123456",
            "reasoning_content": "123456",
            "tool_calls": [
                {"type": "function", "function": {"name": "verify", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": '{"otp":123456}'},
    ]
    sample = ConversationSample(
        dataset="test",
        sample_id="1",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "verify",
                    "description": "Example secret: 123456",
                    "parameters": {"type": "object"},
                },
            }
        ],
        raw={"otp": 123456},
    )
    state = RowState(sample)
    state.parsed_arguments[len(prefix), 0] = {
        "otp": secret,
        "new_password": "create this",
    }
    return state


def test_call_evidence_respects_parallel_and_sequential_frontiers():
    state = _evidence_state(
        [
            {"role": "system", "content": "Use the verification tool."},
            {"role": "user", "content": "Start verification."},
        ]
    )
    first = state.sample.messages[2]
    first["tool_calls"].append(deepcopy(first["tool_calls"][0]))
    state.parsed_arguments[2, 1] = {"otp": "second"}
    state.sample.messages.extend(
        [
            {"role": "tool", "content": "Your next code is 654321."},
            deepcopy(first),
            {"role": "tool", "content": '{"otp":"future-secret"}'},
            {"role": "assistant", "content": "Done."},
        ]
    )
    state.sample.messages[5]["tool_calls"] = [deepcopy(first["tool_calls"][0])]
    state.parsed_arguments[5, 0] = {"otp": 123456}
    before = deepcopy(state.sample)
    projected = []

    def result_values(parsed, message):
        projected.append(message["content"])
        return json_scalar_values(parsed)

    observed = [
        (name, arguments, tuple(texts), frozenset(results))
        for name, definition, arguments, texts, results in iter_call_evidence(
            state, result_values=result_values
        )
    ]
    prior = ("Use the verification tool.", "Start verification.")
    assert observed == [
        ("verify", state.parsed_arguments[2, 0], prior, frozenset()),
        ("verify", {"otp": "second"}, prior, frozenset()),
        (
            "verify",
            {"otp": 123456},
            (*prior, "Your next code is 654321."),
            frozenset({"123456"}),
        ),
    ]
    assert projected == ['{"otp":123456}', '{"otp":"future-secret"}']
    assert state.sample == before


@pytest.mark.parametrize(
    "prefix, accepted",
    [
        ([], False),
        ([{"role": "user", "content": "My code is 123456."}], True),
        ([{"role": "system", "content": "Code: 1234567"}], False),
        ([{"role": "tool", "content": '{"otp":123456}'}], True),
        ([{"role": "tool", "content": '{"123456":true}'}], False),
        ([{"role": "tool", "content": '{"example":"maybe 123456"}'}], False),
        ([{"role": "tool", "content": "Your code: 123456."}], True),
    ],
)
def test_selected_argument_evidence_excludes_assistant_future_raw_and_definition_values(
    prefix, accepted
):
    state = _evidence_state(prefix)
    before = deepcopy(state.sample)

    def run():
        validate_argument_evidence(
            state,
            lambda name, field, definition, value: name == "verify" and field == "otp",
            values=json_scalar_values,
        )

    if accepted:
        run()
    else:
        with pytest.raises(Reject, match="ungrounded_credential_argument"):
            run()
    assert state.sample == before


@pytest.mark.parametrize("result, accepted", [("missing", False), (123456, True)])
def test_result_projection_excludes_echoed_inputs_without_changing_visible_content(
    result, accepted
):
    from fcanalysis.loaders.normalization import json_bytes

    state = _evidence_state(
        [
            {
                "role": "tool",
                "content": json_bytes(
                    {"arguments": {"otp": 123456}, "results": result}
                ).decode(),
            }
        ]
    )
    before = deepcopy(state.sample)

    def run():
        validate_argument_evidence(
            state,
            lambda name, field, definition, value: field == "otp",
            values=json_scalar_values,
            result_values=lambda value, message: json_scalar_values(
                value.get("results")
            ),
        )

    if accepted:
        run()
    else:
        with pytest.raises(Reject, match="ungrounded_credential_argument"):
            run()
    assert state.sample == before


def test_literal_tokens_do_not_accept_substrings_or_regex_syntax():
    assert contains_literal_token("user_12", ["ID: user_12."])
    assert not contains_literal_token("user_12", ["user_123"])
    assert not contains_literal_token("a.b", ["axb"])
    assert contains_literal_token("a.b", ["(a.b)"])
    assert not contains_literal_token("", ["anything"])


def test_case_insensitive_matching_requires_explicit_runtime_permission():
    assert not contains_literal_token("JANE", ["Jane"])
    assert contains_literal_token("JANE", ["Jane"], case_insensitive=True)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Code: 123456.", True),
        ("Code: 123456, then submit.", True),
        ("Code: 123456.0", False),
        ("Code: -123456", False),
        ("Code: +123456", False),
        ("Code: 0.123456", False),
        ("Code: 123456,000", False),
        ("Code: 1,123456", False),
        ("Code: 123456e2", False),
        ("Code: 1234567", False),
    ],
)
def test_numeric_literal_comparison_does_not_accept_number_fragments(text, expected):
    assert (
        contains_literal_token("123456", [text], strict_numeric_tokens=True) is expected
    )


def test_numeric_boundary_option_preserves_exact_spelling_and_non_numeric_literals():
    assert contains_literal_token("0012", ["OTP: 0012."], strict_numeric_tokens=True)
    assert not contains_literal_token("12", ["OTP: 0012"], strict_numeric_tokens=True)
    assert contains_literal_token("1.0", ["Value: 1.0."], strict_numeric_tokens=True)
    assert not contains_literal_token(
        "1.0", ["Value: 1.00"], strict_numeric_tokens=True
    )
    assert contains_literal_token("a.b", ["(a.b)"], strict_numeric_tokens=True)


def test_structured_evidence_is_exact_string_leaves_not_prose_substrings():
    evidence = set()
    collect_string_values(
        {
            "user_id": "actual_123",
            "nested": [{"comment": "maybe hidden_456"}],
            "number": 123,
        },
        evidence,
    )
    assert evidence == {"actual_123", "maybe hidden_456"}
    assert "hidden_456" not in evidence
    assert "user_id" not in evidence
    assert "123" not in evidence


def test_only_explicitly_approved_dictionary_keys_become_evidence():
    evidence = set()
    collect_string_values(
        {"cards": {"card_123": {"balance": 5}}, "unapproved_456": "value"},
        evidence,
        key_filter=lambda key: key.startswith("card_"),
    )
    assert evidence == {"card_123", "value"}


def test_numeric_credential_evidence_is_explicit_and_preserves_scalar_spelling():
    raw = {"otp": 123456, "label": " 0012 ", "nested": [0, 1.0, True, False, None, ""]}
    before = deepcopy(raw)
    assert list(json_scalar_values(raw)) == ["123456", " 0012 ", "0", "1.0"]
    assert raw == before
    assert not contains_literal_token("123456", ["code 1234567"])


def test_numeric_evidence_excludes_method_selectors_without_guessing_other_keys():
    value = {
        "method": {"name": "sms", "value": 123},
        "credentials": [{"method": "email", "value": 456}],
        "other": '{"otp":789}',
    }
    assert list(json_scalar_values(value, skip_keys={"method"})) == [
        "456",
        '{"otp":789}',
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_cannot_become_json_evidence(value):
    with pytest.raises(Reject, match="nonfinite_json_number"):
        list(json_scalar_values(value))
