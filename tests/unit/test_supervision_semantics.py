from copy import deepcopy
import json

import pytest

from fcanalysis.supervision_semantics import (
    FEATURES,
    SemanticSummary,
    annotation_request,
    unknown_assessment,
    validate_assessment,
)
from tests.helpers import assistant, call, func, sample, system, tool_response, user


def fixture():
    return sample(
        [
            system("Ask for confirmation before purchasing."),
            user("Buy it."),
            assistant("Please confirm the purchase."),
            user("Yes. FUTURE_SECRET"),
            assistant(tool_calls=[call("buy")]),
            tool_response("Purchase failed: unavailable."),
            assistant("The purchase completed."),
        ],
        tools=[func("buy")],
        dataset="SOURCE_NOT_FOR_JUDGE",
        sample_id="NATIVE_ID_NOT_FOR_JUDGE",
    )


def assessment(index=2):
    value = fixture()
    record = unknown_assessment(value, index)
    validate_assessment(value, record)
    return value, record


def test_prefix_request_excludes_future_and_source_shortcuts():
    value = fixture()
    before = deepcopy(value)
    value.annotations = {"unvalidated_success": True}
    request = annotation_request(value, 2, max_characters=20000)
    body = json.loads(request[1]["content"])
    assert body["messages"] == before.messages[:3]
    assert body["tools"] == value.tools and body["target_index"] == 2
    assert "FUTURE_SECRET" not in json.dumps(request)
    assert "SOURCE_NOT_FOR_JUDGE" not in json.dumps(request)
    assert "NATIVE_ID_NOT_FOR_JUDGE" not in json.dumps(request)
    assert "unvalidated_success" not in json.dumps(request)
    assert value.messages == before.messages
    with pytest.raises(ValueError, match="do not truncate"):
        annotation_request(value, 2, max_characters=1)


def test_unknown_defaults_do_not_pretend_semantic_success_or_failure():
    value, record = assessment()
    assert record["features"]["tool_abstention"]["presence"] == "present"
    assert record["features"]["tool_abstention"]["correctness"] == "unknown"
    assert record["features"]["clarification"]["presence"] == "unknown"
    summary = SemanticSummary()
    summary.add(value, record)
    output = summary.as_dict()
    assert output["targets"] == 1
    for name in FEATURES:
        assert output["features"][name]["known_correctness"] == 0
        assert sum(c["count"] for c in output["features"][name]["counts"]) == 1


def test_evidenced_clarification_record_and_incorrect_completion_claim():
    value, record = assessment()
    record["features"]["clarification"] = {
        "presence": "present",
        "correctness": "correct",
        "rationale": "The explicit system policy requires confirmation, which is not yet given.",
        "evidence": [
            {"message_index": 0, "field": "content", "quote": "Ask for confirmation"},
            {
                "message_index": 2,
                "field": "content",
                "quote": "Please confirm the purchase.",
            },
        ],
    }
    validate_assessment(value, record)
    summary = SemanticSummary()
    summary.add(value, record)
    other = unknown_assessment(value, 6)
    other["features"]["completion_claim"] = {
        "presence": "present",
        "correctness": "incorrect",
        "rationale": "The explicit purchase result reports failure, contradicting this assertion.",
        "evidence": [
            {
                "message_index": 5,
                "field": "content",
                "quote": "Purchase failed: unavailable.",
            },
            {
                "message_index": 6,
                "field": "content",
                "quote": "The purchase completed.",
            },
        ],
    }
    summary.add(value, other)
    assert summary.as_dict()["features"]["completion_claim"]["known_correctness"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing_feature",
        "extra_feature",
        "invalid_state",
        "boolean_state",
        "false_call",
        "absent_correct",
        "unknown_correct",
        "blank_rationale",
        "no_evidence",
        "future_evidence",
        "invented_quote",
        "missing_field",
        "bool_index",
    ],
)
def test_invalid_and_unsupported_assessments_fail(change):
    value, record = assessment()
    item = record["features"]["clarification"]
    if change == "missing_feature":
        record["features"].pop("clarification")
    elif change == "extra_feature":
        record["features"]["quality"] = item
    elif change == "invalid_state":
        item["presence"] = "likely"
    elif change == "boolean_state":
        item["presence"] = True
    elif change == "false_call":
        record["features"]["tool_selection"]["presence"] = "present"
    elif change == "absent_correct":
        item.update(presence="absent", correctness="correct")
    elif change == "unknown_correct":
        item["correctness"] = "correct"
    elif change == "blank_rationale":
        item["rationale"] = " "
    elif change == "no_evidence":
        item.update(presence="present", correctness="incorrect")
    elif change == "future_evidence":
        item["evidence"] = [
            {"message_index": 3, "field": "content", "quote": "FUTURE_SECRET"}
        ]
    elif change == "invented_quote":
        item["evidence"] = [
            {"message_index": 2, "field": "content", "quote": "fiction"}
        ]
    elif change == "missing_field":
        item["evidence"] = [
            {"message_index": 2, "field": "reasoning_content", "quote": "thinking"}
        ]
    elif change == "bool_index":
        record["target_index"] = True
    with pytest.raises(ValueError):
        validate_assessment(value, record)


def test_references_into_tool_calls_are_literal_not_execution():
    value, record = assessment(4)
    record["features"]["tool_selection"].update(
        correctness="unknown",
        evidence=[
            {"message_index": 4, "field": "tool_calls", "quote": '"name": "buy"'}
        ],
    )
    validate_assessment(value, record)
    assert record["features"]["error_recovery"]["correctness"] == "unknown"
