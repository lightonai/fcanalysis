"""Unit tests for toucan loader-specific conversion, predicates, and transforms.

Synthetic inputs only (no dataset access). The full-dataset behavioral
contract is covered by the e2e fixtures; these tests pin the per-function
semantics that the conversion and Stage 2 config depend on.
"""

import orjson
import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.toucan import (
    ToucanConfig,
    _apply_dataset_config,
    _convert_messages,
    _convert_sample,
    _ends_incomplete,
    _has_conflicting_duplicate_tools,
    _is_reasoning_tool,
    _is_scaffold_tool,
    _passes_quality,
    _stage1_issues,
    _strip_tools,
    load,
)

from tests.helpers import assistant, call, func, system, tool_response, user


# --------------------------------------------------------------------------
# raw-message and row builders (Toucan's pre-conversion schema)
# --------------------------------------------------------------------------


def raw_asst(content=None, function_call=None, reasoning_content=None):
    m: dict = {"role": "assistant"}
    if content is not None:
        m["content"] = content
    if function_call is not None:
        m["function_call"] = function_call
    if reasoning_content is not None:
        m["reasoning_content"] = reasoning_content
    return m


def raw_fn(content):
    return {"role": "function", "content": content}


def fcall(name, arguments=""):
    return {"name": name, "arguments": arguments}


def conv(messages=None, tools=None, raw=None):
    return ConversationSample(
        messages=messages or [],
        tools=tools or [],
        dataset="toucan",
        sample_id="0",
        raw=raw or {},
    )


def qa(question_quality=5, scenario_realism=5):
    return orjson.dumps(
        {
            "question_quality": {"score": question_quality},
            "scenario_realism": {"score": scenario_realism},
        }
    ).decode()


def ra(completeness=4, conciseness=4, desired=1.0):
    return orjson.dumps(
        {
            "completeness": {"score": completeness},
            "conciseness": {"score": conciseness},
            "desired_tools_used_percentage": desired,
        }
    ).decode()


def qrow(subset="single-turn-original", question=(5, 5), response=(4, 4, 1.0)):
    row = {
        "subset_name": subset,
        "question_quality_assessment": qa(*question),
        "response_quality_assessment": "" if subset == "irrelevant" else ra(*response),
    }
    return row


# --------------------------------------------------------------------------
# _is_reasoning_tool / _is_scaffold_tool
# --------------------------------------------------------------------------


class TestIsReasoningTool:
    def test_bare_and_suffixed_think(self) -> None:
        assert _is_reasoning_tool("think") is True
        assert _is_reasoning_tool("THINK") is True
        assert _is_reasoning_tool("server-think") is True

    def test_core_families(self) -> None:
        assert _is_reasoning_tool("mcp-sequentialthinking-tools") is True
        assert _is_reasoning_tool("clear_thought") is True
        assert _is_reasoning_tool("clear-thought") is True
        assert _is_reasoning_tool("think-tool") is True

    def test_reasoning_method_tools(self) -> None:
        assert _is_reasoning_tool("mentalmodel") is True
        assert _is_reasoning_tool("analogicalReasoning") is True
        assert _is_reasoning_tool("collaborativeReasoning") is True
        assert _is_reasoning_tool("visualReasoning") is True
        assert _is_reasoning_tool("get_thoughts") is True
        assert _is_reasoning_tool("get_thought_stats") is True

    def test_full_clear_thought_op_set(self) -> None:
        # the whole reasoning-server op vocabulary, incl. standalone prefixes
        # and bare names that the upstream loader missed.
        for n in (
            "decisionFramework",
            "decision-framework-server-decisionFramework",
            "scientificMethod",
            "socraticmethod",
            "metacognitiveMonitoring",
            "structuredArgumentation",
            "debuggingapproach",
            "designpattern",
            "chain-of-draft-server-chain-of-draft",
            "lotus-wisdom-lotuswisdom",
        ):
            assert _is_reasoning_tool(n) is True, n

    def test_thinking_scaffolds_and_domain_branded(self) -> None:
        assert _is_reasoning_tool("systemsthinking") is True
        assert _is_reasoning_tool("creativethinking") is True
        assert _is_reasoning_tool("pentestthinkingMCP") is True
        # domain tools branded as "structured thinking" count as reasoning.
        assert _is_reasoning_tool("gamedesignthinking") is True
        assert _is_reasoning_tool("skiaanimationthinking") is True

    def test_not_reasoning(self) -> None:
        assert _is_reasoning_tool("get_weather") is False
        assert _is_reasoning_tool("rethink") is False  # no -think suffix/bare/thinking
        # the think-tank server's real ops, despite "think" in the server name
        assert _is_reasoning_tool("think-tank-exa_search") is False
        assert _is_reasoning_tool("think-tank-create_relations") is False
        assert _is_reasoning_tool("think-tank-list_tasks") is False
        # model-listing utility, not a reasoning scaffold
        assert _is_reasoning_tool("listReasoningModels") is False
        assert _is_reasoning_tool("mindbridge-listReasoningModels") is False

    def test_scaffold_tools_are_not_reasoning(self) -> None:
        # reasoning and scaffold token sets are disjoint
        assert _is_reasoning_tool("fs-list_resources") is False
        assert _is_reasoning_tool("exa-search-deep_researcher_check") is False


class TestIsScaffoldTool:
    def test_handshakes(self) -> None:
        assert _is_scaffold_tool("server__unlock_abc__") is True
        assert _is_scaffold_tool("foo__get_instructions") is True

    def test_resource_primitives(self) -> None:
        assert _is_scaffold_tool("fs-list_resources") is True
        assert _is_scaffold_tool("fs-read_resource") is True
        assert _is_scaffold_tool("srv-get_resource") is True

    def test_deep_researcher_poller(self) -> None:
        assert _is_scaffold_tool("exa-search-deep_researcher_start") is True
        assert _is_scaffold_tool("exa-search-deep_researcher_check") is True

    def test_reasoning_and_domain_tools_are_not_scaffold(self) -> None:
        assert _is_scaffold_tool("sequentialthinking") is False
        assert _is_scaffold_tool("mentalmodel") is False
        assert _is_scaffold_tool("get_weather") is False
        assert _is_scaffold_tool("think-tank-exa_search") is False


# --------------------------------------------------------------------------
# _convert_messages  (collapse Toucan's split messages into OpenAI format)
# --------------------------------------------------------------------------


class TestConvertMessages:
    def test_returns_empty_issues(self) -> None:
        # _convert_messages never drops; structural issues come from
        # _stage1_issues. Its issue dict is always empty.
        _, issues = _convert_messages([{"role": "user", "content": "hi"}])
        assert issues == {}

    def test_system_and_user_passthrough(self) -> None:
        out, _ = _convert_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
            ]
        )
        assert out == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ]

    def test_none_content_becomes_empty_string_for_system_user(self) -> None:
        out, _ = _convert_messages([{"role": "user", "content": None}])
        assert out == [{"role": "user", "content": ""}]

    def test_single_text_assistant(self) -> None:
        out, _ = _convert_messages([raw_asst(content="hello")])
        assert out == [{"role": "assistant", "content": "hello"}]

    def test_consecutive_text_and_call_collapse_into_one(self) -> None:
        out, _ = _convert_messages(
            [
                raw_asst(content="let me check"),
                raw_asst(function_call=fcall("get_weather", '{"city":"x"}')),
            ]
        )
        assert out == [
            {
                "role": "assistant",
                "content": "let me check",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"x"}',
                        },
                    }
                ],
            }
        ]

    def test_parallel_calls_preserve_order(self) -> None:
        out, _ = _convert_messages(
            [
                raw_asst(function_call=fcall("a", "{}")),
                raw_asst(function_call=fcall("b", "{}")),
            ]
        )
        names = [tc["function"]["name"] for tc in out[0]["tool_calls"]]
        assert names == ["a", "b"]
        assert out[0]["content"] is None

    def test_function_message_becomes_tool_and_flushes(self) -> None:
        out, _ = _convert_messages(
            [
                raw_asst(function_call=fcall("a", "{}")),
                raw_fn("result-a"),
            ]
        )
        assert out == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"type": "function", "function": {"name": "a", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "result-a"},
        ]

    def test_missing_arguments_default_empty_string(self) -> None:
        out, _ = _convert_messages([raw_asst(function_call={"name": "a"})])
        assert out[0]["tool_calls"][0]["function"]["arguments"] == ""

    def test_reasoning_content_preserved(self) -> None:
        out, _ = _convert_messages(
            [raw_asst(content="answer", reasoning_content="because")]
        )
        assert out[0]["reasoning_content"] == "because"
        assert out[0]["content"] == "answer"

    def test_multiple_text_parts_joined_with_blank_line(self) -> None:
        out, _ = _convert_messages(
            [raw_asst(content="part1"), raw_asst(content="part2")]
        )
        assert out[0]["content"] == "part1\n\npart2"

    def test_empty_assistant_with_nothing_is_dropped(self) -> None:
        # An assistant message with empty content and no call/reasoning
        # contributes nothing and flush() emits no message.
        out, _ = _convert_messages([raw_asst(content="")])
        assert out == []

    def test_tool_message_none_content_becomes_empty(self) -> None:
        out, _ = _convert_messages(
            [raw_asst(function_call=fcall("a")), {"role": "function", "content": None}]
        )
        assert out[-1] == {"role": "tool", "content": ""}

    def test_user_breaks_assistant_run(self) -> None:
        out, _ = _convert_messages(
            [
                raw_asst(content="a1"),
                {"role": "user", "content": "u"},
                raw_asst(content="a2"),
            ]
        )
        assert [m["role"] for m in out] == ["assistant", "user", "assistant"]
        assert out[0]["content"] == "a1"
        assert out[2]["content"] == "a2"


# --------------------------------------------------------------------------
# _stage1_issues  (per-row structural flags on converted messages)
# --------------------------------------------------------------------------


class TestStage1Issues:
    def test_clean_conversation_no_issues(self) -> None:
        msgs = [system("s"), user("u"), assistant(content="a")]
        assert _stage1_issues(msgs) == {}

    def test_no_system_message(self) -> None:
        msgs = [user("u"), assistant(content="a")]
        assert _stage1_issues(msgs).get("no_system_message") == 1

    def test_ends_on_tool_response(self) -> None:
        msgs = [
            system("s"),
            assistant(tool_calls=[call(name="a")]),
            {"role": "tool", "content": "r"},
        ]
        issues = _stage1_issues(msgs)
        assert issues.get("ends_on_tool_response") == 1

    def test_ends_on_empty_assistant(self) -> None:
        msgs = [system("s"), user("u"), {"role": "assistant", "content": None}]
        assert _stage1_issues(msgs).get("ends_on_empty_assistant") == 1

    def test_unbalanced_call_response(self) -> None:
        msgs = [
            system("s"),
            assistant(tool_calls=[call(name="a"), call(name="b")]),
            {"role": "tool", "content": "only-one"},
        ]
        assert _stage1_issues(msgs).get("unbalanced_call_response") == 1

    def test_balanced_calls_no_unbalanced_flag(self) -> None:
        msgs = [
            system("s"),
            assistant(tool_calls=[call(name="a"), call(name="b")]),
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
            assistant(content="done"),
        ]
        issues = _stage1_issues(msgs)
        assert "unbalanced_call_response" not in issues

    def test_unparseable_tool_call_arguments(self) -> None:
        msgs = [
            system("s"),
            assistant(tool_calls=[call(name="a", arguments="{not json")]),
            {"role": "tool", "content": "r"},
            assistant(content="ok"),
        ]
        assert _stage1_issues(msgs).get("unparseable_tool_call_arguments") == 1

    def test_empty_argument_string_not_flagged(self) -> None:
        # Empty/whitespace args are skipped, not flagged as unparseable.
        msgs = [
            system("s"),
            assistant(tool_calls=[call(name="a", arguments="")]),
            {"role": "tool", "content": "r"},
            assistant(content="ok"),
        ]
        assert "unparseable_tool_call_arguments" not in _stage1_issues(msgs)


# --------------------------------------------------------------------------
# _convert_sample  (row dict -> ConversationSample)
# --------------------------------------------------------------------------


class TestConvertSample:
    def test_parses_messages_and_tools(self) -> None:
        row = {
            "uuid": "abc",
            "messages": orjson.dumps(
                [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            ).decode(),
            "available_tools": orjson.dumps([func("get_weather")]).decode(),
        }
        sample_obj, issues = _convert_sample(row, "Kimi-K2")
        assert sample_obj.dataset == "Agent-Ark/Toucan-1.5M:Kimi-K2"
        assert sample_obj.sample_id == "abc"
        assert sample_obj.tools == [func("get_weather")]
        assert [m["role"] for m in sample_obj.messages] == [
            "system",
            "user",
            "assistant",
        ]
        assert sample_obj.raw is row

    def test_missing_uuid_becomes_empty_string(self) -> None:
        row = {"messages": orjson.dumps([{"role": "user", "content": "u"}]).decode()}
        sample_obj, _ = _convert_sample(row, "OSS")
        assert sample_obj.sample_id == ""

    def test_unparseable_available_tools_falls_back_to_empty(self) -> None:
        row = {
            "uuid": "x",
            "messages": orjson.dumps([{"role": "user", "content": "u"}]).decode(),
            "available_tools": "{not valid json",
        }
        sample_obj, _ = _convert_sample(row, "Qwen3")
        assert sample_obj.tools == []

    def test_no_system_message_issue_surfaces(self) -> None:
        row = {
            "uuid": "x",
            "messages": orjson.dumps([{"role": "user", "content": "u"}]).decode(),
            "available_tools": "[]",
        }
        _, issues = _convert_sample(row, "Kimi-K2")
        assert issues.get("no_system_message") == 1


# --------------------------------------------------------------------------
# _passes_quality
# --------------------------------------------------------------------------


class TestPassesQuality:
    cfg = ToucanConfig(drop_low_quality=True)

    def test_all_scores_meet_defaults(self) -> None:
        row = qrow(question=(5, 5), response=(4, 4, 1.0))
        assert _passes_quality(row, self.cfg) is True

    def test_low_question_quality_fails(self) -> None:
        row = qrow(question=(4, 5))
        assert _passes_quality(row, self.cfg) is False

    def test_low_scenario_realism_fails(self) -> None:
        row = qrow(question=(5, 4))
        assert _passes_quality(row, self.cfg) is False

    def test_low_completeness_fails(self) -> None:
        row = qrow(response=(3, 4, 1.0))
        assert _passes_quality(row, self.cfg) is False

    def test_low_conciseness_fails(self) -> None:
        row = qrow(response=(4, 3, 1.0))
        assert _passes_quality(row, self.cfg) is False

    def test_partial_tool_use_fails_when_required(self) -> None:
        row = qrow(response=(4, 4, 0.5))
        assert _passes_quality(row, self.cfg) is False

    def test_partial_tool_use_ok_when_not_required(self) -> None:
        cfg = ToucanConfig(drop_low_quality=True, require_full_tool_use=False)
        row = qrow(response=(4, 4, 0.5))
        assert _passes_quality(row, cfg) is True

    def test_irrelevant_subset_gated_on_question_only(self) -> None:
        # Irrelevant rows carry an empty response assessment and pass on the
        # question side alone.
        row = qrow(subset="irrelevant", question=(5, 5))
        assert _passes_quality(row, self.cfg) is True

    def test_irrelevant_subset_still_fails_low_question(self) -> None:
        row = qrow(subset="irrelevant", question=(4, 5))
        assert _passes_quality(row, self.cfg) is False

    def test_none_question_assessment_fails(self) -> None:
        row = {
            "subset_name": "single-turn-original",
            "question_quality_assessment": None,
            "response_quality_assessment": ra(),
        }
        assert _passes_quality(row, self.cfg) is False

    def test_empty_response_assessment_fails_non_irrelevant(self) -> None:
        row = {
            "subset_name": "multi-turn",
            "question_quality_assessment": qa(5, 5),
            "response_quality_assessment": "",
        }
        assert _passes_quality(row, self.cfg) is False

    def test_custom_thresholds(self) -> None:
        # Comparison logic only (synthetic scores); real data is on a 1-5 scale.
        cfg = ToucanConfig(
            drop_low_quality=True,
            min_question_quality=8,
            min_completeness=5,
        )
        assert _passes_quality(qrow(question=(7, 9)), cfg) is False
        assert _passes_quality(qrow(question=(8, 9), response=(5, 4, 1.0)), cfg) is True

    def test_float_quality_score_is_rejected(self) -> None:
        # Scores are gated with isinstance(x, int); a float 5.0 fails even though
        # 5.0 >= 5 numerically. Pins the int-only contract for question scores.
        row = {
            "subset_name": "single-turn-original",
            "question_quality_assessment": orjson.dumps(
                {"question_quality": {"score": 5.0}, "scenario_realism": {"score": 5}}
            ).decode(),
            "response_quality_assessment": ra(),
        }
        assert _passes_quality(row, self.cfg) is False

    def test_desired_tools_used_percentage_accepts_int_one(self) -> None:
        # desired_tools_used_percentage is checked with isinstance(x, (int, float))
        # and == 1.0, so an integer 1 passes -- asymmetric with the int-only
        # score checks above. Pins that asymmetry.
        row = {
            "subset_name": "multi-turn",
            "question_quality_assessment": qa(5, 5),
            "response_quality_assessment": orjson.dumps(
                {
                    "completeness": {"score": 4},
                    "conciseness": {"score": 4},
                    "desired_tools_used_percentage": 1,
                }
            ).decode(),
        }
        assert _passes_quality(row, self.cfg) is True


# --------------------------------------------------------------------------
# _ends_incomplete
# --------------------------------------------------------------------------


class TestEndsIncomplete:
    def test_empty_messages(self) -> None:
        assert _ends_incomplete(conv(messages=[])) is True

    def test_ends_on_tool(self) -> None:
        s = conv(messages=[assistant(tool_calls=[call()]), tool_response()])
        assert _ends_incomplete(s) is True

    def test_ends_on_empty_assistant(self) -> None:
        s = conv(messages=[user("u"), {"role": "assistant", "content": None}])
        assert _ends_incomplete(s) is True

    def test_ends_on_text_assistant_is_complete(self) -> None:
        s = conv(messages=[user("u"), assistant(content="final answer")])
        assert _ends_incomplete(s) is False

    def test_ends_on_assistant_with_only_tool_calls_is_complete(self) -> None:
        # A final assistant with tool_calls (even no content) is not "empty".
        s = conv(messages=[user("u"), assistant(tool_calls=[call()])])
        assert _ends_incomplete(s) is False


# --------------------------------------------------------------------------
# _has_conflicting_duplicate_tools
# --------------------------------------------------------------------------


class TestHasConflictingDuplicateTools:
    def test_no_tools(self) -> None:
        assert _has_conflicting_duplicate_tools({"available_tools": "[]"}) is False

    def test_unique_names(self) -> None:
        raw = {"available_tools": orjson.dumps([func("a"), func("b")]).decode()}
        assert _has_conflicting_duplicate_tools(raw) is False

    def test_same_name_same_definition(self) -> None:
        raw = {"available_tools": orjson.dumps([func("a"), func("a")]).decode()}
        assert _has_conflicting_duplicate_tools(raw) is False

    def test_same_name_different_description(self) -> None:
        raw = {
            "available_tools": orjson.dumps(
                [func("a", description="x"), func("a", description="y")]
            ).decode()
        }
        assert _has_conflicting_duplicate_tools(raw) is True

    def test_same_name_different_parameters(self) -> None:
        raw = {
            "available_tools": orjson.dumps(
                [
                    func("a", parameters={"type": "object", "properties": {"x": {}}}),
                    func("a", parameters={"type": "object", "properties": {"y": {}}}),
                ]
            ).decode()
        }
        assert _has_conflicting_duplicate_tools(raw) is True

    def test_unparseable_tools_returns_false(self) -> None:
        assert _has_conflicting_duplicate_tools({"available_tools": "{bad"}) is False

    def test_missing_available_tools_key(self) -> None:
        assert _has_conflicting_duplicate_tools({}) is False

    def test_skips_non_dict_tool_entry(self) -> None:
        # A non-dict entry in the tool list is skipped (not crashed on); the
        # real conflict among the dict entries is still detected.
        raw = {
            "available_tools": orjson.dumps(
                [func("a", description="x"), "not-a-dict", func("a", description="y")]
            ).decode()
        }
        assert _has_conflicting_duplicate_tools(raw) is True

    def test_skips_non_str_function_name(self) -> None:
        # A tool whose function.name is not a string cannot key the by-name map,
        # so it is skipped and never registers as a conflict.
        raw = {
            "available_tools": orjson.dumps(
                [{"function": {"name": 123, "description": "x"}}]
            ).decode()
        }
        assert _has_conflicting_duplicate_tools(raw) is False


# --------------------------------------------------------------------------
# _strip_tools  (mutating transform; returns (changed, removed_reasoning, removed_scaffold))
# --------------------------------------------------------------------------


class TestStripTools:
    def test_no_reasoning_tools_unchanged(self) -> None:
        s = conv(
            tools=[func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name="get_weather")]),
                tool_response(),
            ],
        )
        assert _strip_tools(s, True, False) == (False, False, False)

    def test_strips_reasoning_call_and_its_response(self) -> None:
        s = conv(
            tools=[func("think"), func("get_weather")],
            messages=[
                assistant(
                    content="reasoning",
                    tool_calls=[call(name="think"), call(name="get_weather")],
                ),
                tool_response(content="", tool_call_id="t"),
                tool_response(content="sunny", tool_call_id="w"),
            ],
        )
        changed, rem_r, rem_s = _strip_tools(s, True, False)
        assert (changed, rem_r, rem_s) == (True, True, False)
        asst = s.messages[0]
        assert [tc["function"]["name"] for tc in asst["tool_calls"]] == ["get_weather"]
        tool_msgs = [m for m in s.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "sunny"
        # the reasoning tool is pruned from the tool list
        assert [t["function"]["name"] for t in s.tools] == ["get_weather"]

    def test_reasoning_only_turn_with_content_keeps_text_drops_calls(self) -> None:
        s = conv(
            tools=[func("think")],
            messages=[
                assistant(content="visible", tool_calls=[call(name="think")]),
                tool_response(content=""),
            ],
        )
        assert _strip_tools(s, True, False)[0] is True
        assert s.messages == [{"role": "assistant", "content": "visible"}]

    def test_reasoning_only_turn_without_content_is_dropped(self) -> None:
        s = conv(
            tools=[func("think")],
            messages=[
                assistant(tool_calls=[call(name="think")]),
                tool_response(content=""),
            ],
        )
        assert _strip_tools(s, True, False)[0] is True
        assert s.messages == []

    def test_unbalanced_turn_left_untouched(self) -> None:
        # 2 calls, 1 response -> not the loader's normal output; leave alone.
        s = conv(
            tools=[func("think"), func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name="think"), call(name="get_weather")]),
                tool_response(),
            ],
        )
        before = [dict(m) for m in s.messages]
        changed, rem_r, _ = _strip_tools(s, True, False)
        # messages unchanged, but the reasoning tool is still pruned from the list
        assert [m for m in s.messages] == before
        assert [t["function"]["name"] for t in s.tools] == ["get_weather"]
        assert changed is True and rem_r is True  # tools_changed

    def test_prunes_uncalled_reasoning_tool_from_list(self) -> None:
        # A reasoning tool defined but never called is still pruned.
        s = conv(
            tools=[func("think"), func("get_weather")],
            messages=[user("u"), assistant(content="hi")],
        )
        assert _strip_tools(s, True, False)[0] is True
        assert [t["function"]["name"] for t in s.tools] == ["get_weather"]

    def test_scaffold_kept_when_flag_off(self) -> None:
        # With strip_scaffold=False, a resource-primitive call is NOT stripped.
        s = conv(
            tools=[func("fs-read_resource")],
            messages=[
                assistant(tool_calls=[call(name="fs-read_resource")]),
                tool_response(content="file contents"),
            ],
        )
        assert _strip_tools(s, True, False) == (False, False, False)
        assert len(s.messages) == 2

    def test_scaffold_stripped_when_flag_on(self) -> None:
        s = conv(
            tools=[func("fs-read_resource"), func("get_weather")],
            messages=[
                assistant(
                    tool_calls=[call(name="fs-read_resource"), call(name="get_weather")]
                ),
                tool_response(content="file", tool_call_id="r"),
                tool_response(content="sunny", tool_call_id="w"),
            ],
        )
        changed, rem_r, rem_s = _strip_tools(s, True, True)
        assert (changed, rem_r, rem_s) == (True, False, True)
        assert [t["function"]["name"] for t in s.tools] == ["get_weather"]

    def test_both_families_reported_separately(self) -> None:
        s = conv(
            tools=[func("think"), func("fs-read_resource"), func("get_weather")],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="think"),
                        call(name="fs-read_resource"),
                        call(name="get_weather"),
                    ]
                ),
                tool_response(content="", tool_call_id="t"),
                tool_response(content="file", tool_call_id="r"),
                tool_response(content="sunny", tool_call_id="w"),
            ],
        )
        changed, rem_r, rem_s = _strip_tools(s, True, True)
        assert (changed, rem_r, rem_s) == (True, True, True)
        assert [t["function"]["name"] for t in s.tools] == ["get_weather"]

    def test_uncalled_scaffold_tool_attributed_via_list_prune(self) -> None:
        # A scaffold tool DEFINED but never called is still pruned, and the
        # scaffold family is attributed via the tool-list-prune branch alone
        # (not the message loop). Pins removed_scaffold from list pruning.
        s = conv(
            tools=[func("srv-read_resource"), func("get_weather")],
            messages=[user("u"), assistant(content="hi")],
        )
        changed, rem_r, rem_s = _strip_tools(s, True, True)
        assert (changed, rem_r, rem_s) == (True, False, True)
        assert [t["function"]["name"] for t in s.tools] == ["get_weather"]


# --------------------------------------------------------------------------
# _apply_dataset_config  (Stage 2 orchestration: drops then transform)
# --------------------------------------------------------------------------


class TestApplyDatasetConfig:
    def _sample(self, subset, **raw_extra):
        raw = qrow(subset=subset)
        raw.update(raw_extra)
        return conv(messages=[system("s"), user("u"), assistant(content="a")], raw=raw)

    def test_default_config_no_tools_is_noop(self) -> None:
        # Samples with no tools/tool_calls: default config drops nothing and the
        # (default-ON) reasoning transform finds nothing to strip.
        samples = [self._sample("single-turn-original"), self._sample("multi-turn")]
        kept, drops, transforms = _apply_dataset_config(samples, ToucanConfig())
        assert len(kept) == 2
        assert drops == {}
        assert transforms == {}

    def test_default_config_strips_reasoning(self) -> None:
        # strip_reasoning_tools defaults True: ToucanConfig() with NO args strips a
        # reasoning call + its response and prunes the tool, while keeping scaffold
        # (strip_scaffold_tools default False). Pins the default-ON contract that a
        # weaker no-tools test would miss.
        s = conv(
            tools=[func("think"), func("srv-read_resource"), func("get_weather")],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="think"),
                        call(name="srv-read_resource"),
                        call(name="get_weather"),
                    ]
                ),
                tool_response(content="", tool_call_id="t"),
                tool_response(content="file", tool_call_id="r"),
                tool_response(content="sunny", tool_call_id="w"),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        kept, _, transforms = _apply_dataset_config([s], ToucanConfig())
        # reasoning stripped (think), scaffold kept (read_resource survives)
        assert transforms == {"stripped_reasoning_tools": 1}
        kept_names = [t["function"]["name"] for t in kept[0].tools]
        assert kept_names == ["srv-read_resource", "get_weather"]

    def test_subset_filter(self) -> None:
        samples = [
            self._sample("single-turn-original"),
            self._sample("multi-turn"),
            self._sample("irrelevant"),
        ]
        cfg = ToucanConfig(subsets=("multi-turn",))
        kept, drops, _ = _apply_dataset_config(samples, cfg)
        assert len(kept) == 1
        assert kept[0].raw["subset_name"] == "multi-turn"
        assert drops["subset_not_selected"] == 2

    def test_low_quality_drop(self) -> None:
        good = conv(
            messages=[system("s"), user("u"), assistant(content="a")],
            raw=qrow(question=(5, 5), response=(4, 4, 1.0)),
        )
        bad = conv(
            messages=[system("s"), user("u"), assistant(content="a")],
            raw=qrow(question=(2, 2), response=(1, 1, 0.0)),
        )
        cfg = ToucanConfig(drop_low_quality=True)
        kept, drops, _ = _apply_dataset_config([good, bad], cfg)
        assert len(kept) == 1
        assert drops["low_quality"] == 1

    def test_incomplete_termination_drop(self) -> None:
        complete = conv(
            messages=[user("u"), assistant(content="done")],
            raw=qrow(),
        )
        incomplete = conv(
            messages=[assistant(tool_calls=[call()]), tool_response()],
            raw=qrow(),
        )
        cfg = ToucanConfig(drop_incomplete_termination=True)
        kept, drops, _ = _apply_dataset_config([complete, incomplete], cfg)
        assert len(kept) == 1
        assert drops["incomplete_termination"] == 1

    def test_conflicting_duplicate_tools_drop(self) -> None:
        clean = conv(messages=[user("u")], raw=qrow())
        conflicting_raw = qrow()
        conflicting_raw["available_tools"] = orjson.dumps(
            [func("a", description="x"), func("a", description="y")]
        ).decode()
        conflicting = conv(messages=[user("u")], raw=conflicting_raw)
        cfg = ToucanConfig(drop_conflicting_duplicate_tools=True)
        kept, drops, _ = _apply_dataset_config([clean, conflicting], cfg)
        assert len(kept) == 1
        assert drops["conflicting_duplicate_tools"] == 1

    def test_strip_reasoning_tools_is_a_transform_on_survivors(self) -> None:
        s = conv(
            tools=[func("think"), func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name="think"), call(name="get_weather")]),
                tool_response(content="", tool_call_id="t"),
                tool_response(content="sunny", tool_call_id="w"),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        cfg = ToucanConfig(strip_reasoning_tools=True)  # the default
        kept, drops, transforms = _apply_dataset_config([s], cfg)
        assert len(kept) == 1
        assert transforms["stripped_reasoning_tools"] == 1
        assert "stripped_scaffold_tools" not in transforms
        assert [t["function"]["name"] for t in kept[0].tools] == ["get_weather"]

    def test_scaffold_strip_reported_separately(self) -> None:
        # strip_scaffold_tools (off by default) is reported under its own key.
        s = conv(
            tools=[func("srv-read_resource"), func("get_weather")],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="srv-read_resource"),
                        call(name="get_weather"),
                    ]
                ),
                tool_response(content="file", tool_call_id="r"),
                tool_response(content="sunny", tool_call_id="w"),
            ],
            raw=qrow(),
        )
        cfg = ToucanConfig(strip_reasoning_tools=False, strip_scaffold_tools=True)
        kept, _, transforms = _apply_dataset_config([s], cfg)
        assert transforms == {"stripped_scaffold_tools": 1}
        assert [t["function"]["name"] for t in kept[0].tools] == ["get_weather"]

    def test_dropped_rows_are_not_transformed(self) -> None:
        # A row dropped by subset selection must not be counted as stripped,
        # even if it contains reasoning tools (transform runs on survivors only).
        dropped = conv(
            tools=[func("think")],
            messages=[
                assistant(tool_calls=[call(name="think")]),
                tool_response(content=""),
            ],
            raw=qrow(subset="irrelevant"),
        )
        cfg = ToucanConfig(
            subsets=("multi-turn",)
        )  # strip_reasoning_tools default True
        kept, drops, transforms = _apply_dataset_config([dropped], cfg)
        assert kept == []
        assert drops["subset_not_selected"] == 1
        assert transforms == {}

    def test_drops_are_orthogonal_and_counted_per_criterion(self) -> None:
        # The drop criteria are evaluated independently (no short-circuit): a
        # single row failing ALL of them is counted under EACH reason, so
        # per-criterion counts can sum to more than the rows actually removed.
        bad_raw = qrow(subset="irrelevant", question=(1, 1))
        bad_raw["available_tools"] = orjson.dumps(
            [func("a", description="x"), func("a", description="y")]
        ).decode()
        s = conv(
            messages=[assistant(tool_calls=[call()]), tool_response()],
            raw=bad_raw,
        )
        cfg = ToucanConfig(
            subsets=("multi-turn",),
            drop_low_quality=True,
            drop_incomplete_termination=True,
            drop_conflicting_duplicate_tools=True,
        )
        kept, drops, _ = _apply_dataset_config([s], cfg)
        assert kept == []
        assert drops == {
            "subset_not_selected": 1,
            "low_quality": 1,
            "conflicting_duplicate_tools": 1,
            "incomplete_termination": 1,
        }

    def test_incomplete_termination_evaluated_before_strip(self) -> None:
        # Drop decisions use PRE-transform messages: a row ending in a
        # reasoning-only tool turn is judged incomplete (ends on a tool response)
        # and dropped, so strip_reasoning_tools never runs to "rescue" it. Pins
        # the drop/transform ordering.
        s = conv(
            tools=[func("think")],
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="think")]),
                tool_response(content=""),
            ],
            raw=qrow(),
        )
        cfg = ToucanConfig(drop_incomplete_termination=True)  # strip_reasoning default
        kept, drops, transforms = _apply_dataset_config([s], cfg)
        assert kept == []
        assert drops["incomplete_termination"] == 1
        assert transforms == {}


# --------------------------------------------------------------------------
# load() config validation (rejects unsupported configs before any I/O)
# --------------------------------------------------------------------------


class TestLoadConfigValidation:
    def test_sft_config_rejected(self) -> None:
        # SFT is a Kimi-K2 derivative, not a teacher config; load() rejects it
        # up front (before any shard resolution / download).
        with pytest.raises(ValueError, match="SFT"):
            load(configs=("SFT",))

    def test_mixed_teacher_and_sft_rejected(self) -> None:
        with pytest.raises(ValueError, match="SFT"):
            load(configs=("Kimi-K2", "SFT"))

    def test_unknown_config_rejected(self) -> None:
        with pytest.raises(ValueError):
            load(configs=("Nonexistent",))
