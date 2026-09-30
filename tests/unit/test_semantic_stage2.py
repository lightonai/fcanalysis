"""Unit tests for fcanalysis.semantic: v3 prompt + stage-2 verify-and-correct.

Covers the canonical v3 prompt, the stage-2 verify-and-correct pass
(validation, merge, and the end-to-end two-stage flow via a fake client), the
usage/cost accounting, and the dry-run estimator. The networked client itself
is exercised only through a fake — no real API calls.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from fcanalysis.behavioral import SemanticLayerInput
from fcanalysis.cross_tabulation import SemanticResults, load_semantic_results
from fcanalysis.semantic import (
    ANTI_PATTERN_CATEGORIES,
    CATEGORIES,
    CATEGORIES_SET,
    PRICING,
    STAGE2_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V3,
    UsageTracker,
    _apply_stage2,
    _build_request_kwargs,
    _default_concurrency,
    _is_justified,
    _usage_cost,
    _validate_stage2_response,
    build_prompt,
    build_stage2_prompt,
    classify_sample,
    estimate_cost,
    run_semantic_layer,
)
from fcanalysis.semantic_filter import _UNJUSTIFIED_CATEGORIES, filter_by_categories

from tests.helpers import assistant, sample, user


def _input(
    sample_id: str | int = "s1",
    no_fc_turn_indices: list[int] | None = None,
    no_fc_user_message_indices: list[int] | None = None,
) -> SemanticLayerInput:
    return SemanticLayerInput(
        sample_id=sample_id,
        messages=[
            {"role": "system", "content": "You must verify identity before any tool."},
            {"role": "user", "content": "Cancel my order 123"},
            {"role": "assistant", "content": "First, I need to verify your identity."},
        ],
        tools=[{"type": "function", "function": {"name": "cancel_order"}}],
        no_fc_turn_indices=no_fc_turn_indices or [0],
        no_fc_user_message_indices=no_fc_user_message_indices or [1],
    )


# --- Canonical v3 prompt ----------------------------------------------------


class TestPrereqCategory:
    def test_in_categories_and_set(self) -> None:
        assert "S_PREREQUISITE" in CATEGORIES
        assert "S_PREREQUISITE" in CATEGORIES_SET

    def test_is_justified(self) -> None:
        assert _is_justified("S_PREREQUISITE") is True

    def test_not_an_anti_pattern(self) -> None:
        assert "S_PREREQUISITE" not in ANTI_PATTERN_CATEGORIES


class TestCanonicalPrompt:
    def test_system_prompt_is_v3(self) -> None:
        assert SYSTEM_PROMPT == SYSTEM_PROMPT_V3

    def test_v3_mentions_all_categories(self) -> None:
        for cat in CATEGORIES:
            assert cat in SYSTEM_PROMPT_V3, f"missing in v3: {cat}"

    def test_v3_adds_fabrication_and_param_guards(self) -> None:
        assert "S_PREREQUISITE" in SYSTEM_PROMPT_V3
        assert "UNLESS the agent's policy requires an incomplete prerequisite" in (
            SYSTEM_PROMPT_V3
        )
        assert "Honor the agent's own system policy." in SYSTEM_PROMPT_V3
        assert "Fabricated / hallucinated tool use" in SYSTEM_PROMPT_V3
        assert "available tools list" in SYSTEM_PROMPT_V3
        assert "itself the quantity being requested" in SYSTEM_PROMPT_V3

    def test_build_prompt_uses_v3(self) -> None:
        assert build_prompt(_input())[0]["content"] == SYSTEM_PROMPT_V3


# --- Change 3: stage-2 prompt + validation ----------------------------------


class TestBuildStage2Prompt:
    def test_two_messages_system_is_stage2(self) -> None:
        out = build_stage2_prompt(
            _input(), turn_index=0, stage1_category="ANTI_MANUAL_SOLVE"
        )
        assert len(out) == 2
        assert out[0]["content"] == STAGE2_SYSTEM_PROMPT
        assert out[1]["role"] == "user"

    def test_includes_turn_and_label(self) -> None:
        out = build_stage2_prompt(
            _input(), turn_index=0, stage1_category="ANTI_MANUAL_SOLVE"
        )
        user = out[1]["content"]
        assert "turn_index 0" in user
        assert "message index 1" in user  # the no-FC user message index
        assert "ANTI_MANUAL_SOLVE" in user

    def test_strips_thinking_like_stage1(self) -> None:
        si = SemanticLayerInput(
            sample_id="x",
            messages=[
                {"role": "user", "content": "hi", "reasoning_content": "drop me"},
                {"role": "assistant", "content": "<think>secret</think>ok"},
            ],
            tools=[],
            no_fc_turn_indices=[0],
            no_fc_user_message_indices=[0],
        )
        user = build_stage2_prompt(si, 0, "ANTI_MANUAL_SOLVE")[1]["content"]
        assert "drop me" not in user
        assert "<think>secret</think>" not in user


class TestValidateStage2:
    def _confirm(self) -> dict[str, Any]:
        return {
            "policy_prerequisite_incomplete": False,
            "required_param_missing": False,
            "tool_cannot_satisfy": False,
            "info_already_provided": False,
            "category": "ANTI_MANUAL_SOLVE",
            "justified": False,
            "reasoning": "tool matched and params present",
            "correction": {
                "tool_calls": [{"name": "cancel_order", "arguments": {"id": 123}}],
                "content": None,
                "explanation": "should have called cancel_order",
            },
        }

    def _overturn(self) -> dict[str, Any]:
        return {
            "policy_prerequisite_incomplete": True,
            "category": "S_PREREQUISITE",
            "justified": True,
            "reasoning": "policy mandates identity verification first",
            "correction": None,
        }

    def test_valid_confirm(self) -> None:
        assert _validate_stage2_response(self._confirm()) is None

    def test_valid_confirm_with_content_only(self) -> None:
        r = self._confirm()
        r["correction"] = {
            "tool_calls": [],
            "content": "I will not provide that.",
            "explanation": "hold firm under pressure",
        }
        assert _validate_stage2_response(r) is None

    def test_valid_overturn(self) -> None:
        assert _validate_stage2_response(self._overturn()) is None

    def test_overturn_optional_checks_absent_ok(self) -> None:
        r = self._overturn()
        del r["policy_prerequisite_incomplete"]
        assert _validate_stage2_response(r) is None

    def test_confirm_requires_correction(self) -> None:
        r = self._confirm()
        r["correction"] = None
        assert _validate_stage2_response(r) is not None

    def test_confirm_correction_must_be_non_empty(self) -> None:
        r = self._confirm()
        r["correction"] = {"tool_calls": [], "content": "", "explanation": "x"}
        assert _validate_stage2_response(r) is not None

    def test_confirm_correction_needs_explanation(self) -> None:
        r = self._confirm()
        r["correction"] = {"tool_calls": [{"name": "f", "arguments": {}}]}
        assert _validate_stage2_response(r) is not None

    def test_overturn_must_not_have_correction(self) -> None:
        r = self._overturn()
        r["correction"] = {"tool_calls": [], "content": "x", "explanation": "y"}
        assert _validate_stage2_response(r) is not None

    def test_invalid_category(self) -> None:
        r = self._overturn()
        r["category"] = "NOPE"
        assert _validate_stage2_response(r) is not None

    def test_non_bool_justified(self) -> None:
        r = self._overturn()
        r["justified"] = "yes"
        assert _validate_stage2_response(r) is not None

    def test_non_bool_check_field(self) -> None:
        r = self._overturn()
        r["policy_prerequisite_incomplete"] = "true"
        assert _validate_stage2_response(r) is not None

    def test_not_a_dict(self) -> None:
        assert _validate_stage2_response([]) is not None


class TestApplyStage2:
    def _cls(self) -> dict[str, Any]:
        return {
            "turn_index": 0,
            "category": "ANTI_MANUAL_SOLVE",
            "justified": False,
            "reasoning": "stage-1 reason",
            "vote_counts": {"ANTI_MANUAL_SOLVE": 5},
        }

    def test_overturn_updates_final_keeps_stage1(self) -> None:
        cls = self._cls()
        stage2 = {
            "category": "S_PREREQUISITE",
            "justified": True,
            "reasoning": "policy mandates verification",
            "correction": None,
        }
        _apply_stage2(cls, stage2)
        assert cls["stage1_category"] == "ANTI_MANUAL_SOLVE"
        assert cls["stage1_reasoning"] == "stage-1 reason"
        assert cls["category"] == "S_PREREQUISITE"
        assert cls["justified"] is True  # derived from final category
        assert cls["reasoning"] == "policy mandates verification"
        assert cls["verification"] is stage2
        # Load-bearing stage-1 field preserved.
        assert cls["vote_counts"] == {"ANTI_MANUAL_SOLVE": 5}

    def test_confirm_keeps_anti_pattern_and_derives_justified(self) -> None:
        cls = self._cls()
        stage2 = {
            "category": "ANTI_MANUAL_SOLVE",
            "justified": False,
            "reasoning": "confirmed",
            "correction": {
                "tool_calls": [{"name": "cancel_order", "arguments": {}}],
                "content": None,
                "explanation": "call the tool",
            },
        }
        _apply_stage2(cls, stage2)
        assert cls["category"] == "ANTI_MANUAL_SOLVE"
        assert cls["stage1_category"] == "ANTI_MANUAL_SOLVE"
        assert cls["justified"] is False
        assert cls["verification"]["correction"]["tool_calls"][0]["name"] == (
            "cancel_order"
        )

    def test_justified_derived_not_trusted_from_model(self) -> None:
        # Even if the model returns an inconsistent justified flag, the merge
        # derives it from the final category so it can never contradict.
        cls = self._cls()
        stage2 = {
            "category": "ANTI_MANUAL_SOLVE",
            "justified": True,  # inconsistent with an anti-pattern category
            "reasoning": "confirmed",
            "correction": {
                "tool_calls": [{"name": "f", "arguments": {}}],
                "explanation": "x",
            },
        }
        _apply_stage2(cls, stage2)
        assert cls["justified"] is False


# --- Two-stage flow via a fake client ---------------------------------------


class FakeClient:
    """Stand-in for JudgeClient.complete used by classify_sample.

    Returns canned stage-1 / stage-2 JSON, dispatched by inspecting whether the
    system message is the stage-2 prompt. Records call counts for assertions.
    """

    def __init__(self, stage1: dict[str, Any], stage2: dict[str, Any] | None) -> None:
        self._stage1 = stage1
        self._stage2 = stage2
        self.stage1_calls = 0
        self.stage2_calls = 0
        self.stage1_thinking: bool | None = None
        self.stage2_thinking: bool | None = None

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        **_: Any,
    ) -> str:
        if messages[0]["content"] == STAGE2_SYSTEM_PROMPT:
            self.stage2_calls += 1
            self.stage2_thinking = thinking
            return json.dumps(self._stage2)
        self.stage1_calls += 1
        self.stage1_thinking = thinking
        return json.dumps(self._stage1)


class SeqStage2Client(FakeClient):
    """FakeClient whose stage-2 verdicts vary per call (exercises the agreement gate)."""

    def __init__(
        self, stage1: dict[str, Any], stage2_seq: list[dict[str, Any]]
    ) -> None:
        super().__init__(stage1, stage2=None)
        self._seq = stage2_seq

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        **_: Any,
    ) -> str:
        if messages[0]["content"] == STAGE2_SYSTEM_PROMPT:
            self.stage2_calls += 1
            return json.dumps(self._seq[min(self.stage2_calls - 1, len(self._seq) - 1)])
        self.stage1_calls += 1
        return json.dumps(self._stage1)


_ANTI_VERDICT: dict[str, Any] = {
    "category": "ANTI_MANUAL_SOLVE",
    "justified": False,
    "reasoning": "fabricated claims; still an anti-pattern",
    "correction": {
        "tool_calls": [{"name": "cancel_order", "arguments": {}}],
        "content": None,
        "explanation": "should have called the tool",
    },
}
_JUST_VERDICT: dict[str, Any] = {
    "category": "S_PREREQUISITE",
    "justified": True,
    "reasoning": "policy mandates verification first",
    "correction": None,
}


def _stage1_payload(category: str) -> dict[str, Any]:
    return {
        "classifications": [
            {
                "turn_index": 0,
                "category": category,
                "justified": _is_justified(category),
                "reasoning": f"stage-1 said {category}",
            }
        ]
    }


class TestClassifySampleFlow:
    def test_stage1_only_no_verify(self) -> None:
        client = FakeClient(_stage1_payload("ANTI_MANUAL_SOLVE"), stage2=None)
        out = asyncio.run(
            classify_sample(
                client,
                _input(),
                num_generations=3,
                cache_warmup=False,
                verify_and_correct=False,
            )
        )
        assert out["sample_id"] == "s1"
        assert out["num_valid_generations"] == 3
        c = out["classifications"][0]
        assert c["category"] == "ANTI_MANUAL_SOLVE"
        assert c["justified"] is False
        assert c["vote_counts"] == {"ANTI_MANUAL_SOLVE": 3}
        assert "stage1_category" not in c  # stage 2 never ran
        assert client.stage2_calls == 0

    def test_stage2_overturns_false_positive(self) -> None:
        stage2 = {
            "policy_prerequisite_incomplete": True,
            "category": "S_PREREQUISITE",
            "justified": True,
            "reasoning": "policy mandates verification first",
            "correction": None,
        }
        client = FakeClient(_stage1_payload("ANTI_MANUAL_SOLVE"), stage2=stage2)
        out = asyncio.run(
            classify_sample(
                client,
                _input(),
                num_generations=5,
                cache_warmup=True,
                verify_and_correct=True,
            )
        )
        c = out["classifications"][0]
        assert c["category"] == "S_PREREQUISITE"  # overturned
        assert c["justified"] is True
        assert c["stage1_category"] == "ANTI_MANUAL_SOLVE"
        assert c["verification"]["category"] == "S_PREREQUISITE"
        # default verify_votes=2: two unanimous justified verdicts -> clean overturn
        assert client.stage2_calls == 2
        assert "contested" not in c

    def test_stage2_prompt_contains_fabricated_claims_rule(self) -> None:
        # The strengthened rule (validated by the stage-2' re-verification runs)
        # is folded into the ONE canonical stage-2 prompt.
        assert "Fabricated-claims rule" in STAGE2_SYSTEM_PROMPT

    def test_verify_votes_split_is_contested_justified(self) -> None:
        client = SeqStage2Client(
            _stage1_payload("ANTI_MANUAL_SOLVE"), [_ANTI_VERDICT, _JUST_VERDICT]
        )
        out = asyncio.run(
            classify_sample(
                client,
                _input(),
                num_generations=3,
                cache_warmup=False,
                verify_and_correct=True,
            )
        )
        c = out["classifications"][0]
        # split verdict -> precision-first default: justified + contested flag
        assert client.stage2_calls == 2
        assert c["contested"] is True
        assert c["justified"] is True
        assert c["category"] == "S_PREREQUISITE"
        assert c["stage1_category"] == "ANTI_MANUAL_SOLVE"

    def test_verify_votes_unanimous_anti_confirms(self) -> None:
        client = SeqStage2Client(
            _stage1_payload("ANTI_MANUAL_SOLVE"), [_ANTI_VERDICT, _ANTI_VERDICT]
        )
        out = asyncio.run(
            classify_sample(
                client,
                _input(),
                num_generations=3,
                cache_warmup=False,
                verify_and_correct=True,
            )
        )
        c = out["classifications"][0]
        assert client.stage2_calls == 2
        assert c["category"] == "ANTI_MANUAL_SOLVE"
        assert c["justified"] is False
        assert "contested" not in c
        assert (
            c["verification"]["correction"]["explanation"]
            == "should have called the tool"
        )

    def test_verify_votes_1_disables_gate(self) -> None:
        client = SeqStage2Client(_stage1_payload("ANTI_MANUAL_SOLVE"), [_JUST_VERDICT])
        out = asyncio.run(
            classify_sample(
                client,
                _input(),
                num_generations=3,
                cache_warmup=False,
                verify_and_correct=True,
                verify_votes=1,
            )
        )
        c = out["classifications"][0]
        assert client.stage2_calls == 1
        assert c["category"] == "S_PREREQUISITE"
        assert "contested" not in c

    def test_verify_thinking_is_plumbed_to_stage2_only(self) -> None:
        # Stage 1 must stay non-thinking (preserves vote diversity); stage 2
        # gets thinking=True when --verify-thinking is on.
        stage2 = {
            "category": "S_PREREQUISITE",
            "justified": True,
            "reasoning": "policy mandates verification first",
            "correction": None,
        }
        client = FakeClient(_stage1_payload("ANTI_MANUAL_SOLVE"), stage2=stage2)
        asyncio.run(
            classify_sample(
                client,
                _input(),
                num_generations=3,
                verify_and_correct=True,
                verify_thinking=True,
            )
        )
        assert client.stage2_thinking is True
        assert client.stage1_thinking is None  # stage 1 used the client default

    def test_stage2_confirms_and_corrects(self) -> None:
        stage2 = {
            "category": "ANTI_MANUAL_SOLVE",
            "justified": False,
            "reasoning": "no policy gate; tool should have been called",
            "correction": {
                "tool_calls": [{"name": "cancel_order", "arguments": {"id": 123}}],
                "content": None,
                "explanation": "call cancel_order with the given id",
            },
        }
        client = FakeClient(_stage1_payload("ANTI_MANUAL_SOLVE"), stage2=stage2)
        out = asyncio.run(
            classify_sample(
                client, _input(), num_generations=3, verify_and_correct=True
            )
        )
        c = out["classifications"][0]
        assert c["category"] == "ANTI_MANUAL_SOLVE"
        assert c["stage1_category"] == "ANTI_MANUAL_SOLVE"
        assert c["justified"] is False
        assert c["verification"]["correction"]["tool_calls"][0]["name"] == (
            "cancel_order"
        )

    def test_verify_skips_justified_turns(self) -> None:
        client = FakeClient(_stage1_payload("S4_DIRECT_ANSWER"), stage2=None)
        out = asyncio.run(
            classify_sample(
                client, _input(), num_generations=3, verify_and_correct=True
            )
        )
        c = out["classifications"][0]
        assert c["category"] == "S4_DIRECT_ANSWER"
        assert "stage1_category" not in c
        assert client.stage2_calls == 0  # justified turn, no re-check

    def test_insufficient_generations_emits_error_row(self) -> None:
        class BadClient:
            async def complete(self, messages: Any, **kw: Any) -> str:
                return "not json at all"

        out = asyncio.run(classify_sample(BadClient(), _input(), num_generations=3))
        assert "error" in out
        assert out["sample_id"] == "s1"
        assert out["num_valid_generations"] == 0
        assert "classifications" not in out


# --- Request body construction (thinking / temperature / JSON matrix) -------


class TestBuildRequestKwargs:
    def _kw(self, **over: Any) -> dict[str, Any]:
        base: dict[str, Any] = dict(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2048,
            temperature=0.7,
            thinking=False,
            reasoning_effort=None,
            json_mode=True,
        )
        base.update(over)
        return _build_request_kwargs(**base)

    def test_non_thinking_sends_temperature_and_disabled(self) -> None:
        k = self._kw(thinking=False)
        assert k["temperature"] == 0.7
        assert k["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in k["extra_body"]
        assert k["response_format"] == {"type": "json_object"}

    def test_thinking_omits_temperature_and_enables(self) -> None:
        k = self._kw(thinking=True, reasoning_effort="high")
        assert "temperature" not in k  # thinking ignores it; don't send it
        assert k["extra_body"]["thinking"] == {"type": "enabled"}
        assert k["extra_body"]["reasoning_effort"] == "high"

    def test_thinking_without_effort_omits_effort(self) -> None:
        k = self._kw(thinking=True, reasoning_effort=None)
        assert k["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in k["extra_body"]

    def test_no_json_mode_omits_response_format(self) -> None:
        assert "response_format" not in self._kw(json_mode=False)

    def test_non_v4_model_omits_thinking_toggle(self) -> None:
        # Non-V4 models don't accept the thinking extra_body; we must not send it.
        k = self._kw(model="some-other-model", thinking=False)
        assert "thinking" not in k.get("extra_body", {})
        assert k["temperature"] == 0.7

    def test_sampling_params_routing_for_vllm(self) -> None:
        # Local vLLM / Qwen: standard params top-level, extended in extra_body.
        k = self._kw(
            model="Qwen/Qwen3.5-122B-A10B-FP8",
            thinking=False,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )
        assert k["top_p"] == 0.8
        assert k["presence_penalty"] == 0.0
        assert k["extra_body"]["top_k"] == 20
        assert k["extra_body"]["min_p"] == 0.0
        assert k["extra_body"]["repetition_penalty"] == 1.0

    def test_extended_sampling_gated_off_for_deepseek(self) -> None:
        # DeepSeek supports top_p/presence_penalty but NOT top_k/min_p/rep —
        # those must never be sent to it.
        k = self._kw(
            model="deepseek-v4-pro",
            thinking=False,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )
        assert k["top_p"] == 0.8
        assert k["presence_penalty"] == 0.0
        eb = k.get("extra_body", {})
        assert "top_k" not in eb
        assert "min_p" not in eb
        assert "repetition_penalty" not in eb

    def test_thinking_omits_all_sampling(self) -> None:
        # DeepSeek thinking ignores sampling params, so none are sent.
        k = self._kw(
            model="deepseek-v4-pro",
            thinking=True,
            reasoning_effort="high",
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
        )
        assert "temperature" not in k
        assert "top_p" not in k
        assert "presence_penalty" not in k
        assert "top_k" not in k.get("extra_body", {})
        assert k["extra_body"]["thinking"] == {"type": "enabled"}


class TestDefaultConcurrency:
    def test_v4_pro_just_under_cap(self) -> None:
        assert _default_concurrency("deepseek-v4-pro") == 480  # 500 - 20

    def test_high_cap_model_is_bounded(self) -> None:
        # flash's cap is 2500, but the auto default is bounded to avoid opening
        # thousands of sockets from one process.
        assert _default_concurrency("deepseek-v4-flash") == 512

    def test_unknown_model_conservative(self) -> None:
        assert _default_concurrency("mystery") == 256


# --- Usage accounting + pricing ---------------------------------------------


class TestUsageTracker:
    def test_add_with_cache_breakdown(self) -> None:
        t = UsageTracker()
        t.add(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_cache_hit_tokens": 600,
                "prompt_cache_miss_tokens": 400,
            }
        )
        d = t.as_dict()
        assert d["calls"] == 1
        assert d["cache_hit_tokens"] == 600
        assert d["cache_miss_tokens"] == 400
        assert d["completion_tokens"] == 200

    def test_add_without_cache_fields_all_miss(self) -> None:
        t = UsageTracker()
        t.add({"prompt_tokens": 500, "completion_tokens": 100})
        d = t.as_dict()
        assert d["cache_hit_tokens"] == 0
        assert d["cache_miss_tokens"] == 500

    def test_add_none_is_noop(self) -> None:
        t = UsageTracker()
        t.add(None)
        assert t.as_dict()["calls"] == 0

    def test_add_pydantic_like_object(self) -> None:
        class U:
            def model_dump(self) -> dict[str, int]:
                return {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_cache_hit_tokens": 4,
                    "prompt_cache_miss_tokens": 6,
                }

        t = UsageTracker()
        t.add(U())
        assert t.as_dict()["cache_hit_tokens"] == 4

    def test_cost_matches_pricing(self) -> None:
        d = {
            "cache_hit_tokens": 1_000_000,
            "cache_miss_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        }
        p = PRICING["deepseek-v4-pro"]
        expected = p["cache_hit"] + p["cache_miss"] + p["output"]
        cost = _usage_cost(d, "deepseek-v4-pro")
        assert cost is not None
        assert abs(cost - expected) < 1e-9

    def test_cost_unknown_model_is_none(self) -> None:
        assert _usage_cost({"cache_miss_tokens": 100}, "mystery-model") is None


# --- Dry-run estimator ------------------------------------------------------


class TestEstimateCost:
    def test_basic_counts(self) -> None:
        est = estimate_cost(
            [_input()], model="deepseek-v4-pro", num_generations=5, cache_warmup=True
        )
        assert est["samples"] == 1
        assert est["no_fc_turns"] == 1
        assert est["stage1_requests"] == 5
        assert est["cost_usd"] is not None and est["cost_usd"] > 0

    def test_cost_scales_with_generations(self) -> None:
        one = estimate_cost([_input()], model="deepseek-v4-pro", num_generations=1)
        five = estimate_cost([_input()], model="deepseek-v4-pro", num_generations=5)
        assert five["cost_usd"] > one["cost_usd"]

    def test_cache_warmup_shifts_miss_to_hit(self) -> None:
        warm = estimate_cost(
            [_input()], model="deepseek-v4-pro", num_generations=5, cache_warmup=True
        )
        cold = estimate_cost(
            [_input()], model="deepseek-v4-pro", num_generations=5, cache_warmup=False
        )
        # Warmup makes generations 2..N hit on the user portion.
        assert warm["cache_miss_tokens"] < cold["cache_miss_tokens"]
        assert warm["cache_hit_tokens"] > cold["cache_hit_tokens"]
        assert warm["cost_usd"] < cold["cost_usd"]

    def test_verify_adds_stage2_requests(self) -> None:
        off = estimate_cost(
            [_input()], model="deepseek-v4-pro", verify_and_correct=False
        )
        on = estimate_cost(
            [_input()],
            model="deepseek-v4-pro",
            verify_and_correct=True,
            stage2_flag_rate=1.0,
        )
        assert off["stage2_requests_est"] == 0
        assert on["stage2_requests_est"] >= 1
        assert on["cost_usd"] > off["cost_usd"]

    def test_unknown_model_cost_none(self) -> None:
        est = estimate_cost([_input()], model="mystery")
        assert est["cost_usd"] is None


# --- run_semantic_layer: on-disk schema (HARD INVARIANT 1) + resume ---------


class FakeRunClient:
    """FakeClient enriched with the attributes run_semantic_layer touches."""

    def __init__(self, stage1: dict[str, Any], stage2: dict[str, Any] | None) -> None:
        self.model = "deepseek-v4-pro"
        self.usage = UsageTracker()
        self._stage1 = stage1
        self._stage2 = stage2

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        if messages[0]["content"] == STAGE2_SYSTEM_PROMPT:
            return json.dumps(self._stage2)
        return json.dumps(self._stage1)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestRunSemanticLayer:
    def test_output_schema_and_load_roundtrip(self, tmp_path: Path) -> None:
        inputs = [_input(sample_id=f"s{i}") for i in range(3)]
        client = FakeRunClient(_stage1_payload("S4_DIRECT_ANSWER"), stage2=None)
        out = tmp_path / "nested" / "semantic_results_dolci.jsonl"

        _run(
            run_semantic_layer(
                inputs, client, out, num_generations=3, cache_warmup=False
            )
        )

        lines = out.read_text().splitlines()
        assert len(lines) == 3
        for line in lines:
            row = json.loads(line)
            # Exact top-level schema required by downstream consumers.
            assert set(row) >= {"sample_id", "classifications", "num_valid_generations"}
            assert "error" not in row
            assert row["num_valid_generations"] == 3
            for c in row["classifications"]:
                assert set(c) >= {
                    "turn_index",
                    "category",
                    "justified",
                    "reasoning",
                    "vote_counts",
                }
                assert c["category"] in CATEGORIES_SET

        # cross_tabulation reader round-trips it.
        results = load_semantic_results(out)
        assert set(results) == {"s0", "s1", "s2"}
        assert results["s0"][0]["category"] == "S4_DIRECT_ANSWER"

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        inputs = [_input(sample_id=f"s{i}") for i in range(3)]
        client = FakeRunClient(_stage1_payload("S4_DIRECT_ANSWER"), stage2=None)
        out = tmp_path / "semantic_results_dolci.jsonl"

        _run(
            run_semantic_layer(
                inputs, client, out, num_generations=2, cache_warmup=False
            )
        )
        first = out.read_text().splitlines()
        assert len(first) == 3

        # Second run with the same inputs: all already done, nothing appended.
        _run(
            run_semantic_layer(
                inputs, client, out, num_generations=2, cache_warmup=False
            )
        )
        second = out.read_text().splitlines()
        assert len(second) == 3

    def test_stage2_overturn_persisted(self, tmp_path: Path) -> None:
        stage2 = {
            "category": "S_PREREQUISITE",
            "justified": True,
            "reasoning": "policy mandates verification",
            "correction": None,
        }
        client = FakeRunClient(_stage1_payload("ANTI_MANUAL_SOLVE"), stage2=stage2)
        out = tmp_path / "semantic_results_dolci.jsonl"
        _run(
            run_semantic_layer(
                [_input(sample_id="s0")],
                client,
                out,
                num_generations=3,
                cache_warmup=False,
                verify_and_correct=True,
            )
        )
        row = json.loads(out.read_text().splitlines()[0])
        c = row["classifications"][0]
        assert c["category"] == "S_PREREQUISITE"  # final (post-stage-2) label
        assert c["stage1_category"] == "ANTI_MANUAL_SOLVE"  # auditable overturn
        assert c["justified"] is True
        assert "verification" in c


class TestDownstreamFilterInteraction:
    """An overturned S_PREREQUISITE must survive the AMS filter; a confirmed
    AMS must still be dropped. This is the whole point of the verify pass."""

    def test_prereq_not_in_unjustified_set(self) -> None:
        assert "S_PREREQUISITE" not in _UNJUSTIFIED_CATEGORIES

    def test_ams_filter_keeps_overturned_prereq(self) -> None:
        s_keep = sample(
            messages=[user("cancel order"), assistant(content="verify identity first")],
            sample_id="keep",
        )
        s_drop = sample(
            messages=[user("what is 2+2"), assistant(content="it is 4")],
            sample_id="drop",
        )
        semantic: SemanticResults = {
            "keep": {0: {"category": "S_PREREQUISITE", "justified": True}},
            "drop": {0: {"category": "ANTI_MANUAL_SOLVE", "justified": False}},
        }
        kept, result = filter_by_categories(
            [s_keep, s_drop], semantic, {"ANTI_MANUAL_SOLVE"}
        )
        kept_ids = {s.sample_id for s in kept}
        assert kept_ids == {"keep"}
        assert result.removed_samples == 1
