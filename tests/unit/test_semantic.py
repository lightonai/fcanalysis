import json
from typing import Any

from fcanalysis.behavioral import SemanticLayerInput
from fcanalysis.semantic import (
    CATEGORIES,
    CATEGORIES_SET,
    SYSTEM_PROMPT,
    _is_justified,
    _majority_vote,
    _parse_response,
    _validate_response,
    build_prompt,
)


def _input(
    sample_id: str | int = "s1",
    no_fc_turn_indices: list[int] | None = None,
    no_fc_user_message_indices: list[int] | None = None,
) -> SemanticLayerInput:
    return SemanticLayerInput(
        sample_id=sample_id,
        messages=[
            {"role": "user", "content": "hello", "reasoning_content": "drop me"},
            {"role": "assistant", "content": "<think>hidden</think>visible"},
        ],
        tools=[{"type": "function", "function": {"name": "f"}}],
        no_fc_turn_indices=no_fc_turn_indices or [0],
        no_fc_user_message_indices=no_fc_user_message_indices or [0],
    )


class TestCategories:
    def test_categories_set_matches_list(self) -> None:
        assert CATEGORIES_SET == frozenset(CATEGORIES)

    def test_anti_categories_present(self) -> None:
        for anti in (
            "ANTI_MANUAL_SOLVE",
            "ANTI_UNJUSTIFIED_REFUSAL",
            "ANTI_PRESSURE_CAVE",
        ):
            assert anti in CATEGORIES_SET

    def test_system_prompt_mentions_all_categories(self) -> None:
        # Every classification name appears at least once in the prompt.
        for cat in CATEGORIES:
            assert cat in SYSTEM_PROMPT, f"missing in prompt: {cat}"


class TestBuildPrompt:
    def test_returns_two_messages(self) -> None:
        out = build_prompt(_input())
        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"

    def test_system_message_is_system_prompt(self) -> None:
        out = build_prompt(_input())
        assert out[0]["content"] == SYSTEM_PROMPT

    def test_strips_think_tags_from_messages(self) -> None:
        out = build_prompt(_input())
        # The assistant's "<think>hidden</think>visible" should become "visible".
        assert "<think>hidden</think>" not in out[1]["content"]
        assert "visible" in out[1]["content"]

    def test_strips_reasoning_content(self) -> None:
        out = build_prompt(_input())
        assert "drop me" not in out[1]["content"]

    def test_includes_turn_map(self) -> None:
        out = build_prompt(
            _input(no_fc_turn_indices=[2, 5], no_fc_user_message_indices=[3, 7])
        )
        # Turn map is rendered as a JSON list of dicts.
        assert '"turn_index": 2' in out[1]["content"]
        assert '"user_message_index": 7' in out[1]["content"]

    def test_includes_indexed_messages(self) -> None:
        out = build_prompt(_input())
        # Each message in the payload should carry a positional "index" field.
        assert '"index": 0' in out[1]["content"]
        assert '"index": 1' in out[1]["content"]


class TestValidateResponse:
    def _good(self, turn_index: int = 0) -> dict[str, Any]:
        return {
            "classifications": [
                {
                    "turn_index": turn_index,
                    "category": "S4_DIRECT_ANSWER",
                    "justified": True,
                    "reasoning": "model answered from knowledge",
                }
            ]
        }

    def test_valid_response_returns_none(self) -> None:
        assert _validate_response(self._good(), [0]) is None

    def test_not_a_dict(self) -> None:
        assert _validate_response([], [0]) is not None

    def test_missing_classifications(self) -> None:
        assert _validate_response({}, [0]) is not None

    def test_non_list_classifications(self) -> None:
        assert _validate_response({"classifications": "nope"}, [0]) is not None

    def test_missing_field(self) -> None:
        r = self._good()
        del r["classifications"][0]["reasoning"]
        assert _validate_response(r, [0]) is not None

    def test_invalid_category(self) -> None:
        r = self._good()
        r["classifications"][0]["category"] = "NOT_A_CATEGORY"
        assert _validate_response(r, [0]) is not None

    def test_non_bool_justified(self) -> None:
        r = self._good()
        r["classifications"][0]["justified"] = "yes"
        assert _validate_response(r, [0]) is not None

    def test_turn_mismatch(self) -> None:
        # Expected turns [0, 1], response has only [0].
        assert _validate_response(self._good(), [0, 1]) is not None


class TestParseResponse:
    def test_plain_json(self) -> None:
        assert _parse_response('{"k": 1}') == {"k": 1}

    def test_strips_markdown_fences(self) -> None:
        assert _parse_response('```json\n{"k": 2}\n```') == {"k": 2}

    def test_strips_fenced_without_language(self) -> None:
        assert _parse_response('```\n{"k": 3}\n```') == {"k": 3}

    def test_raises_on_invalid_json(self) -> None:
        import pytest

        with pytest.raises(json.JSONDecodeError):
            _parse_response("not json")


class TestIsJustified:
    def test_anti_categories_not_justified(self) -> None:
        for c in (
            "ANTI_MANUAL_SOLVE",
            "ANTI_UNJUSTIFIED_REFUSAL",
            "ANTI_PRESSURE_CAVE",
        ):
            assert _is_justified(c) is False

    def test_other_unjustified_not_justified(self) -> None:
        assert _is_justified("OTHER_UNJUSTIFIED") is False

    def test_justified_categories(self) -> None:
        for c in ("S4_DIRECT_ANSWER", "S3_CLARIFICATION", "OTHER_JUSTIFIED"):
            assert _is_justified(c) is True


class TestMajorityVote:
    def _gen(self, turn_index: int, category: str, reasoning: str = "r") -> dict:
        return {
            "classifications": [
                {
                    "turn_index": turn_index,
                    "category": category,
                    "justified": _is_justified(category),
                    "reasoning": reasoning,
                }
            ]
        }

    def test_unanimous(self) -> None:
        results = [self._gen(0, "S4_DIRECT_ANSWER") for _ in range(5)]
        out = _majority_vote(results, [0])
        c = out["classifications"][0]
        assert c["turn_index"] == 0
        assert c["category"] == "S4_DIRECT_ANSWER"
        assert c["justified"] is True
        assert c["vote_counts"] == {"S4_DIRECT_ANSWER": 5}
        assert "tied" not in c

    def test_majority_wins(self) -> None:
        results = [
            self._gen(0, "S4_DIRECT_ANSWER"),
            self._gen(0, "S4_DIRECT_ANSWER"),
            self._gen(0, "S4_DIRECT_ANSWER"),
            self._gen(0, "ANTI_MANUAL_SOLVE"),
            self._gen(0, "ANTI_MANUAL_SOLVE"),
        ]
        out = _majority_vote(results, [0])
        c = out["classifications"][0]
        assert c["category"] == "S4_DIRECT_ANSWER"
        assert c["vote_counts"]["S4_DIRECT_ANSWER"] == 3
        assert c["vote_counts"]["ANTI_MANUAL_SOLVE"] == 2

    def test_tie_alphabetical_when_global_counts_also_tied(self) -> None:
        # Both turn-local and global counts tie at 2-2; tiebreaker falls
        # through to category-name ascending → S3_CLARIFICATION wins.
        results = [
            self._gen(0, "S3_CLARIFICATION"),
            self._gen(0, "S3_CLARIFICATION"),
            self._gen(0, "S4_DIRECT_ANSWER"),
            self._gen(0, "S4_DIRECT_ANSWER"),
        ]
        out = _majority_vote(results, [0])
        c = out["classifications"][0]
        assert c["category"] == "S3_CLARIFICATION"
        assert c.get("tied") is True

    def test_tie_broken_by_global_frequency(self) -> None:
        # Turn 0 ties at S3=2, S4=2. Turn 1 votes S4 three times → globally
        # S4=5, S3=2. Global frequency tiebreaker picks S4 for turn 0.
        def gen_two_turns(t0: str, t1: str) -> dict:
            return {
                "classifications": [
                    {
                        "turn_index": 0,
                        "category": t0,
                        "justified": True,
                        "reasoning": "r",
                    },
                    {
                        "turn_index": 1,
                        "category": t1,
                        "justified": True,
                        "reasoning": "r",
                    },
                ]
            }

        results = [
            gen_two_turns("S3_CLARIFICATION", "S4_DIRECT_ANSWER"),
            gen_two_turns("S3_CLARIFICATION", "S4_DIRECT_ANSWER"),
            gen_two_turns("S4_DIRECT_ANSWER", "S4_DIRECT_ANSWER"),
            gen_two_turns("S4_DIRECT_ANSWER", "S5_DECLINE"),
        ]
        out = _majority_vote(results, [0, 1])
        # Turn 0: local 2-2 tie; global S3=2, S4=3 → S4 wins.
        assert out["classifications"][0]["category"] == "S4_DIRECT_ANSWER"
        assert out["classifications"][0].get("tied") is True
