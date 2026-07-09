"""Semantic layer: LLM classification of no-FC turns over an OpenAI-compatible
Chat Completions endpoint.

In production the judge runs LOCALLY and free on Qwen3.5-397B-A17B-FP8 served by
vLLM (`http://localhost:8003/v1`; see scripts/run_judge_server.sh + run_judge.sh).
The same client also supports DeepSeek's hosted API (`https://api.deepseek.com`,
model `deepseek-v4-pro`) as an optional paid backend — select it with
`--base-url`/`--model`/`--api-key-env`. Concurrency is bounded by an
`asyncio.Semaphore` (DeepSeek imposes an account-level *concurrent-connection*
cap — 500 for v4-pro — returning HTTP 429 when exceeded; local vLLM has no such
cap but the semaphore still paces the fan-out), and transient 429/5xx errors are
retried with the OpenAI SDK's exponential backoff.

Two coupled correctness changes shaped the current prompt + flow:

* **Canonical stage-1 prompt.** The semantic judge is v3-only. The prompt has a
  justified slot for policy-mandated prerequisites (auth / consent /
  verification), a sharper ``ANTI_MANUAL_SOLVE`` vs ``S6_NEAR_MISS`` boundary,
  and explicit fabricated/hallucinated tool-use routing to
  ``OTHER_UNJUSTIFIED``.

* **Two-stage classify → verify-and-correct.** Optionally
  (``--verify-and-correct-flags``), every turn whose stage-1 label is an
  anti-pattern is re-examined by a targeted second pass that can overturn the
  flag to a justified category and, when it confirms the flag, emit a corrected
  version of that assistant turn. Both the stage-1 and final labels are
  recorded.

DeepSeek specifics that shape the design:

* **Thinking mode ignores ``temperature``.** Multi-generation + majority vote
  relies on temperature-driven diversity, so stage-1 classification defaults to
  *non-thinking* mode. ``--thinking`` is available for ablation but collapses
  vote diversity (use with ``--num-generations 1``).
* **JSON output is ``{"type": "json_object"}`` only** — no JSON-Schema / enum
  guided decoding. Category validity is therefore enforced by
  ``_validate_response`` + retry rather than by the decoder.
* **Context caching is automatic and prefix-based** (``prompt_cache_hit_tokens``
  / ``prompt_cache_miss_tokens`` in ``usage``). The static system prompt is the
  shared prefix across all requests. Within a sample the N generations share an
  identical prompt, so the cache-warmup path fires one generation first (to
  populate the cache) then fans out the rest as full-prompt cache hits.

Output schema (one JSONL line per sample) is unchanged and load-bearing for
downstream consumers (`cross_tabulation`, `semantic_filter`, and the external
fc-curation-mechanistic analysis)::

    {"sample_id", "classifications": [{"turn_index", "category", "justified",
     "reasoning", "vote_counts"}], "num_valid_generations"}

Stage 2, when enabled, adds extra keys to each classification entry
(``stage1_category``, ``stage1_reasoning``, ``verification``) without changing
the load-bearing ones.
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

from .behavioral import (
    SemanticLayerInput,
    analyze_dataset_behavior,
    prepare_semantic_layer_inputs,
)
from .format import ConversationSample
from .loaders import LOADER_MODULES
from .loaders.base import (
    THINK_PATTERNS,
    FilterConfig,
    LoadReport,
    strip_thinking_from_sample,
)


LOADER_NAMES: tuple[str, ...] = tuple(LOADER_MODULES)


CATEGORIES = [
    # Justified no-FC (spec scenarios where not calling is correct)
    "S3_CLARIFICATION",
    "S_PREREQUISITE",
    "S4_DIRECT_ANSWER",
    "S5_DECLINE",
    "S6_NEAR_MISS",
    "S10_NO_TOOLS",
    "M3_IRRELEVANT",
    "M5_PARTIAL_PARAMS",
    "M8_JUSTIFIED_PERSISTENCE",
    "E1_EXPLANATION",
    # Anti-patterns (unjustified no-FC)
    "ANTI_MANUAL_SOLVE",
    "ANTI_UNJUSTIFIED_REFUSAL",
    "ANTI_PRESSURE_CAVE",
    # Catch-all
    "OTHER_JUSTIFIED",
    "OTHER_UNJUSTIFIED",
]

CATEGORIES_SET = frozenset(CATEGORIES)

# Stage-1 labels that trigger the optional stage-2 verify-and-correct pass.
# Mirrors semantic_filter._UNJUSTIFIED_CATEGORIES: the set of "model was WRONG
# not to call" labels. Any category here is also exactly `not _is_justified`.
ANTI_PATTERN_CATEGORIES = frozenset(
    {
        "ANTI_MANUAL_SOLVE",
        "ANTI_UNJUSTIFIED_REFUSAL",
        "ANTI_PRESSURE_CAVE",
        "OTHER_UNJUSTIFIED",
    }
)


# --- Stage-1 system prompt --------------------------------------------------
#
# v3 is the only supported semantic-judge prompt. Earlier prompt drafts were
# archived outside this repo and intentionally removed from the runtime surface.

SYSTEM_PROMPT_V3 = """\
You classify no-FC (no function call) turns in function calling training data.

You will see: (1) the available tools, (2) the full conversation with message indices,
and (3) which user turns had NO tool calls. For each marked turn, determine:

1. **Category** -- pick exactly one from the list below.
2. **Justified** -- was the model correct not to call any tool?
3. **Reasoning** -- one sentence explaining why.

### Justified categories (model was RIGHT not to call)

- **S3_CLARIFICATION**: A tool matches the query but required parameters are missing. \
The model correctly asks the user for the specific missing info.
- **S_PREREQUISITE**: The agent's own policy requires a prerequisite step before calling \
the tool -- authenticating / verifying the user's identity, obtaining required \
confirmation or consent, or collecting a mandatory security field -- and that step has \
not yet been completed in the conversation. The model correctly performing that \
prerequisite (e.g. "I need to verify your identity first") is JUSTIFIED, even if a tool \
nominally matches and the functional parameters appear present.
- **S4_DIRECT_ANSWER**: No available tool matches the user's query. \
The model correctly answers from its own knowledge.
- **S5_DECLINE**: No tool matches and the model cannot answer. \
It correctly explains what tools are available and declines.
- **S6_NEAR_MISS**: A tool looks superficially related but does not actually fit \
(e.g., tool computes median but user asks for mean; wrong unit / entity / operation; or a \
REQUIRED parameter cannot be supplied because it is itself the value the user is asking \
for). The model correctly explains the mismatch or proceeds without it. A nominally- \
matching tool that cannot actually be called or cannot satisfy the request is \
S6_NEAR_MISS, NOT ANTI_MANUAL_SOLVE.
- **S10_NO_TOOLS**: No tools are defined at all. Model answers or declines appropriately.
- **M3_IRRELEVANT**: In a multi-turn conversation, this particular query doesn't match \
any tool even though earlier/later turns did. Model correctly doesn't force-fit.
- **M5_PARTIAL_PARAMS**: User provided some but not all required params. \
Model correctly asks for the remaining ones (not a first-contact clarification; \
this is a follow-up after the user already provided partial info).
- **M8_JUSTIFIED_PERSISTENCE**: User pushes back on a correct refusal. \
Model correctly maintains its position.
- **E1_EXPLANATION**: User asks for explanation/knowledge, not execution. \
No relevant knowledge/retrieval tool is available. Model answers directly.

### Unjustified categories (model was WRONG not to call; anti-patterns)

- **ANTI_MANUAL_SOLVE**: A tool matches the query AND required params are available, \
but the model answers directly instead of calling it. \
This includes: computing math, looking up facts, solving problems that a tool handles. \
Even for short queries. Even if the model CAN answer correctly on its own. \
NOTE: completing a policy-mandated prerequisite first (see S_PREREQUISITE) is NOT this. \
NOTE: this applies ONLY when a listed tool genuinely fits AND its required parameters are \
actually available from the conversation. If a required parameter is itself the quantity \
being requested, or the task is simple arithmetic / manipulation of values already \
provided or already retrieved that NO available tool performs, it is NOT this \
(see S6_NEAR_MISS / S4_DIRECT_ANSWER).
- **ANTI_UNJUSTIFIED_REFUSAL**: A tool matches and params are sufficient, \
but the model refuses, asks unnecessary questions, or says it can't help. \
Includes: asking for optional params, re-asking for info already provided, \
claiming it doesn't have access when the tool is right there.
- **ANTI_PRESSURE_CAVE**: Model previously refused correctly, but now caves to \
user pressure and provides an inappropriate direct answer or fabricated explanation.
- **Fabricated / hallucinated tool use** -- classify as **OTHER_UNJUSTIFIED**: the \
assistant "calls" or names a tool that is NOT in the available tools list, invents a tool \
that does not exist, or fabricates a tool result / claims an action, computation, or \
verification succeeded without actually issuing a valid tool call. Inventing tools or \
results is never justified; the correct behavior is to call a real listed tool or to state \
that none can do this.

### Catch-all

- **OTHER_JUSTIFIED**: Justified no-FC that doesn't fit the categories above.
- **OTHER_UNJUSTIFIED**: Unjustified no-FC that doesn't fit the categories above -- \
including fabricated or hallucinated tool calls and invented tool results.

### Key principles

- If a matching tool exists and required params are available, the model SHOULD call it \
-- UNLESS the agent's policy requires an incomplete prerequisite step (authentication, \
consent, mandatory verification), or the tool cannot actually satisfy the request. \
Honor the agent's own system policy.
- Tools take priority over direct answers, even if the model could answer from knowledge.
- Short queries deserve tool calls just as much as long ones.
- Only the tools listed under the available tools are real. If the assistant references, \
"calls", or relies on a tool that is not in the available tools list, or invents results \
/ outcomes, the turn is OTHER_UNJUSTIFIED (fabrication) -- never a justified category.
- Before labeling ANTI_MANUAL_SOLVE, confirm a listed tool actually fits AND its required \
parameters are available. Simple arithmetic on values already known, or a tool whose \
required parameter is itself the quantity being requested, is S6_NEAR_MISS or a justified \
direct answer, not ANTI_MANUAL_SOLVE.
- Each turn is evaluated in full conversation context; look at what happened before.

### Output format

Respond with exactly this JSON structure (no extra text outside the JSON):

```json
{"classifications": [{"turn_index": <int>, "category": "<CATEGORY>", \
"justified": <bool>, "reasoning": "<one sentence>"}]}
```

One entry per turn listed in "No-FC Turns to Classify". \
`turn_index` must match the values provided."""


# Back-compat alias: several callers and tests reference SYSTEM_PROMPT directly.
SYSTEM_PROMPT = SYSTEM_PROMPT_V3


# --- Stage-2 verify-and-correct prompt --------------------------------------

STAGE2_SYSTEM_PROMPT = """\
You are auditing a SINGLE no-FC (no function call) turn that a first-pass classifier
flagged as an anti-pattern -- it judged that the model should have called a tool but
instead answered directly, refused, or caved to pressure. Re-examine that one turn with
the full conversation and the agent's own policy, decide whether the flag is correct, and
-- only when it is correct -- produce the corrected assistant turn.

You will see: (1) the available tools, (2) the full conversation with message indices
(message index 0 is usually the agent's system / policy message -- READ IT), and (3) the
specific flagged turn and its stage-1 category.

Re-examine the flagged turn by answering, in order:

(a) Does the agent's own policy require a prerequisite step that has NOT yet been completed
    -- authenticating / verifying the user's identity, obtaining confirmation or consent,
    or collecting a mandatory security field -- before the tool may be called?
(b) Is a required parameter genuinely missing (not optional, and not already supplied
    earlier in the conversation)?
(c) Can a nominally-matching tool actually satisfy this request, or does it only look
    related (wrong unit, wrong entity, wrong operation)?
(d) Was the information the model "should have used" already provided or the request
    already served earlier in the conversation, making a tool call unnecessary?
(e) Did the assistant FABRICATE -- "call" or name a tool that is NOT in the available
    tools list, invent a non-existent tool, or claim a result / action / verification
    succeeded without actually issuing a valid tool call?

If (e) holds, the flag is CORRECT: fabrication is never justified. Do NOT overturn it via
(c) -- a fabricated or non-existent tool is not a near-miss. Keep OTHER_UNJUSTIFIED (or the
stage-1 anti category) and supply the correction.
Otherwise, if ANY of (a)-(c) hold (or (d) makes the call unnecessary), the flag is a FALSE
POSITIVE. Reclassify to the appropriate JUSTIFIED category:
- S_PREREQUISITE  -- a policy-mandated prerequisite (auth / consent / verification) is incomplete (a).
- S3_CLARIFICATION or M5_PARTIAL_PARAMS -- a required parameter is genuinely missing (b).
- S6_NEAR_MISS    -- the tool only superficially matches and cannot satisfy the request (c).
- M3_IRRELEVANT   -- no available tool matches this particular turn.
- S4_DIRECT_ANSWER or E1_EXPLANATION -- no tool matches; answering from knowledge is correct.
- OTHER_JUSTIFIED -- justified for some other reason (e.g. (d)).

Otherwise CONFIRM the anti-pattern. Keep the original category (ANTI_MANUAL_SOLVE,
ANTI_UNJUSTIFIED_REFUSAL, ANTI_PRESSURE_CAVE) or OTHER_UNJUSTIFIED, and provide the
corrected turn: what the assistant SHOULD have produced here -- the correct tool call(s)
with concrete arguments drawn from the conversation, and/or the correct text (e.g. a
refusal that holds firm under pressure).

### Fabricated-claims rule (extends check (e); takes precedence over (c) and (d))

Fabrication is not limited to calling a non-existent tool. If the assistant ASSERTS that
an action was completed, that a result / availability / status was retrieved, confirmed
or verified, or presents specific retrieved-looking facts, numbers or links, WITHOUT a
valid tool call (in this turn or earlier in the conversation) actually producing that
information -- that is FABRICATION (e). Do NOT overturn such a turn via (c) "tool cannot
satisfy" or (d) "info already provided" unless the asserted information genuinely appears
earlier in the conversation. Fabricated claims are never justified: CONFIRM the
anti-pattern (keep the stage-1 category or use OTHER_UNJUSTIFIED, and set
fabricated_tool_use=true) and provide the correction -- the honest text admitting the
limitation and/or the correct real tool call.

### Output format

Respond with exactly this JSON object and nothing outside it:

```json
{"policy_prerequisite_incomplete": <bool>, "required_param_missing": <bool>, \
"tool_cannot_satisfy": <bool>, "info_already_provided": <bool>, "fabricated_tool_use": <bool>, \
"category": "<CATEGORY>", "justified": <bool>, "reasoning": "<one or two sentences citing (a)-(e)>", \
"correction": {"tool_calls": [{"name": "<tool name>", "arguments": {}}], \
"content": "<assistant text, or null>", "explanation": "<why this is correct here>"}}
```

Rules:
- `category` must be exactly one of the allowed category names.
- When the flag is a FALSE POSITIVE (justified == true), set `correction` to null -- there
  is nothing to correct.
- When you CONFIRM the anti-pattern (justified == false), `correction` is REQUIRED and must
  contain at least one tool call OR non-empty `content`, plus an `explanation`. Use
  `tool_calls: []` if the correct turn is purely text."""


def _clean_indexed_messages(sem_input: SemanticLayerInput) -> list[dict[str, Any]]:
    """Strip thinking traces / reasoning_content and attach positional indices.

    Shared by `build_prompt` and `build_stage2_prompt` so both stages see the
    exact same rendering of the conversation.
    """
    cleaned_messages = []
    for msg in sem_input.messages:
        cleaned = {**msg}
        content = cleaned.get("content")
        if isinstance(content, str):
            for pattern in THINK_PATTERNS:
                content = pattern.sub("", content)
            cleaned["content"] = content.strip() or None
        cleaned.pop("reasoning_content", None)
        cleaned_messages.append(cleaned)

    return [{**msg, "index": i} for i, msg in enumerate(cleaned_messages)]


def build_prompt(sem_input: SemanticLayerInput) -> list[dict[str, str]]:
    turn_map = [
        {"turn_index": ti, "user_message_index": mi}
        for ti, mi in zip(
            sem_input.no_fc_turn_indices,
            sem_input.no_fc_user_message_indices,
            strict=True,
        )
    ]

    indexed_messages = _clean_indexed_messages(sem_input)

    user_content = f"""## Sample Data

```json
{json.dumps({"tools": sem_input.tools, "messages": indexed_messages}, separators=(",", ":"))}
```

## No-FC Turns to Classify

```json
{json.dumps(turn_map, separators=(",", ":"))}
```

Respond with JSON in the format specified above."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT_V3},
        {"role": "user", "content": user_content},
    ]


def build_stage2_prompt(
    sem_input: SemanticLayerInput,
    turn_index: int,
    stage1_category: str,
) -> list[dict[str, str]]:
    """Build the stage-2 verify-and-correct prompt for one flagged turn."""
    indexed_messages = _clean_indexed_messages(sem_input)
    umi_by_turn = dict(
        zip(
            sem_input.no_fc_turn_indices,
            sem_input.no_fc_user_message_indices,
            strict=True,
        )
    )
    user_message_index = umi_by_turn[turn_index]

    user_content = f"""## Sample Data

```json
{json.dumps({"tools": sem_input.tools, "messages": indexed_messages}, separators=(",", ":"))}
```

## Flagged Turn

Stage 1 classified turn_index {turn_index} (the user message at message index \
{user_message_index}, plus the assistant reply that followed it) as `{stage1_category}` \
-- an anti-pattern. Re-examine ONLY this turn using the full conversation above and the \
agent's policy (message index 0), answer checks (a)-(d), then either reclassify to a \
justified category or confirm `{stage1_category}` and supply the corrected turn.

Respond with JSON in the format specified above."""

    return [
        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _validate_response(result: Any, expected_turns: list[int]) -> str | None:
    # Returns None on success; an error string otherwise.
    if not isinstance(result, dict):
        return "response is not a dict"

    classifications = result.get("classifications")
    if not isinstance(classifications, list):
        return "missing or non-list 'classifications'"

    for c in classifications:
        if not isinstance(c, dict):
            return f"classification entry is not a dict: {c!r}"
        for fld in ("turn_index", "category", "justified", "reasoning"):
            if fld not in c:
                return f"missing field '{fld}' in {c!r}"
        if not isinstance(c["turn_index"], int):
            return f"turn_index is not int: {c['turn_index']!r}"
        if c["category"] not in CATEGORIES_SET:
            return f"invalid category: {c['category']!r}"
        if not isinstance(c["justified"], bool):
            return f"justified is not bool: {c['justified']!r}"
        if not isinstance(c["reasoning"], str):
            return f"reasoning is not str: {c['reasoning']!r}"

    returned_turns = sorted(c["turn_index"] for c in classifications)
    if returned_turns != sorted(expected_turns):
        return f"turn mismatch: expected {sorted(expected_turns)}, got {returned_turns}"

    return None


def _validate_stage2_response(result: Any) -> str | None:
    """Validate a stage-2 verify-and-correct response. None on success.

    Lenient on the (a)-(d) booleans (optional, but type-checked if present) to
    keep parse-retry pressure low under JSON mode, strict on the load-bearing
    fields. A correction object is required iff the final category is an
    anti-pattern (the flag was confirmed); when overturned to a justified
    category there is nothing to correct.
    """
    if not isinstance(result, dict):
        return "response is not a dict"

    category = result.get("category")
    if category not in CATEGORIES_SET:
        return f"invalid category: {category!r}"
    if not isinstance(result.get("justified"), bool):
        return f"justified is not bool: {result.get('justified')!r}"
    if not isinstance(result.get("reasoning"), str):
        return f"reasoning is not str: {result.get('reasoning')!r}"

    for fld in (
        "policy_prerequisite_incomplete",
        "required_param_missing",
        "tool_cannot_satisfy",
        "info_already_provided",
    ):
        if fld in result and not isinstance(result[fld], bool):
            return f"{fld} is not bool: {result[fld]!r}"

    correction = result.get("correction")
    if category in ANTI_PATTERN_CATEGORIES:
        # Confirmed anti-pattern: a corrected turn is mandatory.
        if not isinstance(correction, dict):
            return "correction required (dict) when anti-pattern confirmed"
        if not isinstance(correction.get("explanation"), str):
            return "correction.explanation missing or not str"
        tool_calls = correction.get("tool_calls")
        content = correction.get("content")
        has_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
        has_content = isinstance(content, str) and content.strip() != ""
        if not (has_calls or has_content):
            return "correction needs at least one tool_call or non-empty content"
    else:
        # Overturned to a justified category: correction must be absent/null.
        if correction not in (None, {}):
            return "correction must be null when reclassified as justified"

    return None


# Majority-vote ensemble size. 3 gives real majorities (survives one outlier)
# at lower output cost than 5; with cache-warmup, N barely affects input cost.
NUM_GENERATIONS = 3
# Per-generation slot retries: a slot re-rolls on empty / unparseable /
# schema-invalid content AND on transient API errors that survived the SDK's
# own backoff. Sibling slots are independent, so retrying one never discards a
# slot that already validated. 5 is comfortably enough for deepseek-v4-pro.
MAX_RETRIES_PER_GEN = 5


def _parse_response(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    return json.loads(stripped)


def _is_justified(category: str) -> bool:
    return not (category.startswith("ANTI_") or category == "OTHER_UNJUSTIFIED")


def _majority_vote(
    results: list[dict[str, Any]],
    expected_turns: list[int],
) -> dict[str, Any]:
    # Ties are broken by global category frequency (descending), then by
    # category name (ascending). This is the deterministic tiebreaker
    # locked in by current consumers.
    global_counts: Counter[str] = Counter()
    for r in results:
        for c in r["classifications"]:
            global_counts[c["category"]] += 1

    classifications = []
    for ti in expected_turns:
        votes: list[dict[str, Any]] = []
        for r in results:
            for c in r["classifications"]:
                if c["turn_index"] == ti:
                    votes.append(c)
                    break

        category_counts: Counter[str] = Counter(v["category"] for v in votes)
        top_two = category_counts.most_common(2)
        top_count = top_two[0][1]
        tied = len(top_two) > 1 and top_two[0][1] == top_two[1][1]
        if tied:
            tied_cats = [
                cat for cat, cnt in category_counts.items() if cnt == top_count
            ]
            tied_cats.sort(key=lambda c: (-global_counts[c], c))
            winner_category = tied_cats[0]
        else:
            winner_category = top_two[0][0]

        representative = next(v for v in votes if v["category"] == winner_category)

        entry: dict[str, Any] = {
            "turn_index": ti,
            "category": winner_category,
            "justified": _is_justified(winner_category),
            "reasoning": representative["reasoning"],
            "vote_counts": dict(category_counts),
        }
        if tied:
            entry["tied"] = True

        classifications.append(entry)

    return {"classifications": classifications}


def _apply_stage2(
    classification: dict[str, Any],
    stage2: dict[str, Any],
) -> dict[str, Any]:
    """Merge a stage-2 verdict into a stage-1 classification entry, in place.

    Records the original stage-1 label/reasoning (auditability of overturns)
    and overwrites the load-bearing `category`/`justified`/`reasoning` with the
    final (post-stage-2) decision. `justified` is derived from the final
    category so it can never contradict it. The full stage-2 object (the
    (a)-(d) checks and the correction) is stored under `verification`.
    """
    final_category = stage2["category"]
    classification["stage1_category"] = classification["category"]
    classification["stage1_reasoning"] = classification.get("reasoning", "")
    classification["category"] = final_category
    classification["justified"] = _is_justified(final_category)
    classification["reasoning"] = stage2.get(
        "reasoning", classification.get("reasoning", "")
    )
    classification["verification"] = stage2
    return classification


# --- Judge client (OpenAI-compatible: local vLLM by default, DeepSeek optional) ---

# Defaults target the local vLLM judge; pass --base-url/--model/--api-key-env to
# use DeepSeek's hosted API (deepseek-v4-pro) or any other OpenAI-compatible backend.
DEFAULT_BASE_URL = "http://localhost:8003/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_API_KEY_ENV = "VLLM_KEY"
# DeepSeek account-level concurrent-connection caps (HTTP 429 when exceeded).
CONCURRENCY_CAP = {"deepseek-v4-pro": 500, "deepseek-v4-flash": 2500}
# Cap the *auto* default even when the account cap is far higher (flash's 2500):
# opening thousands of sockets from one event loop costs file descriptors and
# memory for little throughput gain. Override explicitly with --concurrency.
_MAX_AUTO_CONCURRENCY = 512


def _default_concurrency(model: str) -> int:
    """Default semaphore size: just under the model's concurrent-connection cap.

    Staying ~20 below the cap absorbs connection churn (a finishing connection
    overlapping a newly opened one) so we don't trip an occasional HTTP 429 by
    sitting exactly at the limit, while still using ~96% of capacity. Unknown
    models fall back to a conservative 256.
    """
    cap = CONCURRENCY_CAP.get(model)
    if cap is None:
        return 256
    return max(1, min(cap - 20, _MAX_AUTO_CONCURRENCY))


# USD per 1M tokens (input cache-hit, input cache-miss, output).
PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
    "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
}


class UsageTracker:
    """Accumulates token usage across calls and computes cost.

    DeepSeek reports `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` in
    the `usage` object; when absent (non-DeepSeek backend), all prompt tokens
    are charged at the cache-miss rate.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        if hasattr(usage, "model_dump"):
            d = usage.model_dump()
        elif isinstance(usage, dict):
            d = usage
        else:
            return
        self.calls += 1
        self.prompt_tokens += int(d.get("prompt_tokens") or 0)
        self.completion_tokens += int(d.get("completion_tokens") or 0)
        hit = d.get("prompt_cache_hit_tokens")
        miss = d.get("prompt_cache_miss_tokens")
        if hit is None and miss is None:
            self.cache_miss_tokens += int(d.get("prompt_tokens") or 0)
        else:
            self.cache_hit_tokens += int(hit or 0)
            self.cache_miss_tokens += int(miss or 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
        }


def _usage_cost(d: dict[str, int], model: str) -> float | None:
    price = PRICING.get(model)
    if price is None:
        return None
    return (
        d.get("cache_hit_tokens", 0) * price["cache_hit"]
        + d.get("cache_miss_tokens", 0) * price["cache_miss"]
        + d.get("completion_tokens", 0) * price["output"]
    ) / 1_000_000


def _format_usage(d: dict[str, int], model: str, label: str = "") -> str:
    prompt = d.get("cache_hit_tokens", 0) + d.get("cache_miss_tokens", 0)
    hit_pct = (d.get("cache_hit_tokens", 0) / prompt * 100) if prompt else 0.0
    cost = _usage_cost(d, model)
    cost_str = f"${cost:,.2f}" if cost is not None else "n/a (unknown model price)"
    head = f"Usage{f' [{label}]' if label else ''}:"
    return (
        f"{head} {d.get('calls', 0):,} calls | "
        f"prompt {prompt:,} tok (cache hit {hit_pct:.1f}%) | "
        f"completion {d.get('completion_tokens', 0):,} tok | "
        f"est. cost {cost_str}"
    )


class CompletionClient(Protocol):
    """The minimal surface the generation/classification helpers need.

    `JudgeClient` satisfies this structurally; tests pass in lightweight
    fakes. Keeping the orchestration typed against the protocol (not the
    concrete client) decouples it from the HTTP/openai machinery.
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = ...,
        thinking: bool | None = ...,
        reasoning_effort: str | None = ...,
    ) -> str | None: ...


class ManagedClient(CompletionClient, Protocol):
    """A completion client that also exposes usage accounting and its model.

    `run_semantic_layer` reports per-dataset cost, so it needs these in
    addition to `complete`.
    """

    model: str
    usage: "UsageTracker"


def _build_request_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    thinking: bool,
    reasoning_effort: str | None,
    json_mode: bool,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    """Construct the chat-completions request body.

    Pure (no I/O) so the sampling / thinking / JSON matrix is unit-testable.
    Sampling params are sent only in non-thinking mode (thinking ignores them on
    DeepSeek; the judge runs non-thinking anyway). ``top_p`` / ``presence_penalty``
    are OpenAI-standard and go to any backend; ``top_k`` / ``min_p`` /
    ``repetition_penalty`` are NOT OpenAI-standard — vLLM reads them from
    ``extra_body``, but DeepSeek's API does not support them, so they are sent
    only to non-DeepSeek (e.g. local vLLM / Qwen) backends.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    extra_body: dict[str, Any] = {}
    if model.startswith("deepseek-v4"):
        extra_body["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if thinking:
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if not model.startswith("deepseek"):
            if top_k is not None:
                extra_body["top_k"] = top_k
            if min_p is not None:
                extra_body["min_p"] = min_p
            if repetition_penalty is not None:
                extra_body["repetition_penalty"] = repetition_penalty
    if extra_body:
        kwargs["extra_body"] = extra_body
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


class JudgeClient:
    """Single OpenAI-compatible client (local vLLM or DeepSeek) with a global concurrency cap.

    `openai` and `httpx` are imported lazily so this module can be imported in
    environments without the LLM dependencies (tests, dry-run).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str,
        concurrency: int = 256,
        timeout: int = 600,
        max_retries: int = 8,
        thinking: bool = False,
        reasoning_effort: str | None = None,
        temperature: float = 0.7,
        top_p: float | None = 0.8,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
        presence_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        json_mode: bool = True,
    ) -> None:
        import httpx
        import openai

        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.json_mode = json_mode
        self._sem = asyncio.Semaphore(concurrency)
        # httpx connection pool sized a touch above the semaphore so the
        # semaphore (not the pool) is the binding concurrency limit.
        pool = concurrency + 16
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=pool,
                    max_keepalive_connections=pool,
                ),
                timeout=httpx.Timeout(timeout, connect=30.0),
            ),
        )
        self.usage = UsageTracker()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str | None:
        # Per-call overrides fall back to the client defaults. The stage-2 pass
        # uses `thinking=True` to reason carefully without touching stage-1
        # voting (which must stay non-thinking to keep temperature diversity).
        kwargs = _build_request_kwargs(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            thinking=self.thinking if thinking is None else thinking,
            reasoning_effort=(
                self.reasoning_effort if reasoning_effort is None else reasoning_effort
            ),
            json_mode=self.json_mode,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
        )
        async with self._sem:
            response = await self._client.chat.completions.create(**kwargs)

        self.usage.add(getattr(response, "usage", None))
        if not response.choices:
            return None
        return response.choices[0].message.content

    async def warmup(self) -> None:
        """Probe the endpoint (also validates the API key); raises on failure."""
        await self._client.models.list()

    async def aclose(self) -> None:
        await self._client.close()


async def _single_generation(
    client: CompletionClient,
    messages: list[dict[str, str]],
    expected_turns: list[int],
    *,
    max_tokens: int,
) -> dict[str, Any] | None:
    """One stage-1 generation slot, retried up to MAX_RETRIES_PER_GEN times.

    Each attempt re-rolls the call, recovering from empty / unparseable /
    schema-invalid content AND from transient API errors that survived the SDK's
    own backoff. Returns a validated result, or None if every attempt produced
    bad content. If *all* attempts failed at the HTTP layer, re-raises the last
    error so the caller can surface it (the sample becomes an error row).
    Independent of sibling slots: a slot that validates is never discarded
    because another slot failed.
    """
    last_exc: Exception | None = None
    for _attempt in range(MAX_RETRIES_PER_GEN):
        try:
            content = await client.complete(messages, max_tokens=max_tokens)
        except Exception as e:  # transient error past SDK backoff; re-roll
            last_exc = e
            continue
        if not content:
            continue
        try:
            result = _parse_response(content)
        except ValueError:
            continue
        if _validate_response(result, expected_turns) is None:
            return result
    if last_exc is not None:
        raise last_exc
    return None


async def _run_stage1(
    client: CompletionClient,
    messages: list[dict[str, str]],
    expected_turns: list[int],
    num_generations: int,
    cache_warmup: bool,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[BaseException]]:
    """Run the N stage-1 generations, optionally warming the prefix cache.

    With cache_warmup, the first generation runs alone so DeepSeek caches the
    (system + user) prefix; the remaining N-1 then fan out and hit that cache,
    cutting input cost ~ (N-1)/N for this sample. Temperature still varies the
    *output* (caching only affects input-prefix compute), so vote diversity is
    preserved.
    """

    async def one() -> dict[str, Any] | None:
        return await _single_generation(
            client, messages, expected_turns, max_tokens=max_tokens
        )

    if cache_warmup and num_generations > 1:
        try:
            first: dict[str, Any] | None | BaseException = await one()
        except Exception as e:  # warmup call failed; still fan out (cache cold)
            first = e
        rest = await asyncio.gather(
            *[one() for _ in range(num_generations - 1)], return_exceptions=True
        )
        results: list[Any] = [first, *rest]
    else:
        results = await asyncio.gather(
            *[one() for _ in range(num_generations)], return_exceptions=True
        )

    valid = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, BaseException)]
    return valid, errors


async def _verify_and_correct_turn(
    client: CompletionClient,
    sem_input: SemanticLayerInput,
    turn_index: int,
    stage1_category: str,
    *,
    max_tokens: int,
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any] | None:
    """Stage-2 re-check of one flagged turn, retried up to MAX_RETRIES_PER_GEN
    times. Returns None if no valid verdict could be obtained (the caller then
    keeps the stage-1 label). Stage-2 failure is non-fatal, so transient errors
    are swallowed and retried rather than propagated. ``thinking`` enables
    DeepSeek reasoning for this pass independently of stage 1."""
    messages = build_stage2_prompt(sem_input, turn_index, stage1_category)
    for _attempt in range(MAX_RETRIES_PER_GEN):
        try:
            content = await client.complete(
                messages,
                max_tokens=max_tokens,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )
        except Exception:  # non-fatal; re-roll, then fall back to stage-1 label
            continue
        if not content:
            continue
        try:
            result = _parse_response(content)
        except ValueError:
            continue
        if _validate_stage2_response(result) is None:
            return result
    return None


async def classify_sample(
    client: CompletionClient,
    sem_input: SemanticLayerInput,
    *,
    num_generations: int = NUM_GENERATIONS,
    cache_warmup: bool = True,
    verify_and_correct: bool = False,
    verify_thinking: bool = False,
    verify_reasoning_effort: str | None = None,
    max_tokens: int = 2048,
    stage2_max_tokens: int = 3072,
    verify_votes: int = 2,
) -> dict[str, Any]:
    messages = build_prompt(sem_input)
    expected_turns = sem_input.no_fc_turn_indices

    valid, server_errors = await _run_stage1(
        client, messages, expected_turns, num_generations, cache_warmup, max_tokens
    )

    if len(valid) < num_generations:
        error_msg = (
            str(server_errors[0]) if server_errors else "insufficient_valid_generations"
        )
        return {
            "sample_id": sem_input.sample_id,
            "error": error_msg,
            "num_valid_generations": len(valid),
        }

    voted = _majority_vote(valid, expected_turns)

    if verify_and_correct:
        for cls in voted["classifications"]:
            if cls["category"] not in ANTI_PATTERN_CATEGORIES:
                continue
            # Stage 2 runs `verify_votes` independent samples per flagged turn and
            # applies an AGREEMENT GATE (validated on the full v3 run, where two
            # independent stage-2 passes agreed on only 87.3% of overturns):
            #   - all valid votes anti      -> confirm (anti + correction)
            #   - all valid votes justified -> overturn (justified)
            #   - split                     -> the verifier is self-inconsistent on
            #     this turn; precision-first default: justified, contested=True.
            # Vote failures are non-fatal: act on the valid votes; if none are
            # valid, keep the stage-1 label (sample still emitted, unverified).
            results = await asyncio.gather(
                *[
                    _verify_and_correct_turn(
                        client,
                        sem_input,
                        cls["turn_index"],
                        cls["category"],
                        max_tokens=stage2_max_tokens,
                        thinking=verify_thinking,
                        reasoning_effort=verify_reasoning_effort,
                    )
                    for _ in range(max(1, verify_votes))
                ],
                return_exceptions=True,
            )
            verdicts = [r for r in results if isinstance(r, dict)]
            if not verdicts:
                continue
            anti = [v for v in verdicts if v["category"] in ANTI_PATTERN_CATEGORIES]
            just = [v for v in verdicts if v["category"] not in ANTI_PATTERN_CATEGORIES]
            if anti and just:
                _apply_stage2(cls, just[0])
                cls["contested"] = True
            elif anti:
                _apply_stage2(cls, anti[0])
            else:
                _apply_stage2(cls, just[0])
            if len(verdicts) < max(1, verify_votes):
                cls["verify_votes_valid"] = len(verdicts)

    voted["sample_id"] = sem_input.sample_id
    voted["num_valid_generations"] = len(valid)
    return voted


async def run_semantic_layer(
    sem_inputs: list[SemanticLayerInput],
    client: ManagedClient,
    output_path: str | Path,
    *,
    num_generations: int = NUM_GENERATIONS,
    cache_warmup: bool = True,
    verify_and_correct: bool = False,
    verify_thinking: bool = False,
    verify_reasoning_effort: str | None = None,
    max_tokens: int = 2048,
    stage2_max_tokens: int = 3072,
    verify_votes: int = 2,
    num_workers: int = 256,
    label: str = "",
) -> dict[str, int]:
    """Classify all samples with no-FC turns, appending one JSONL line each.

    Returns the usage delta (this call's token counts) so the caller can report
    per-dataset cost.

    Resume behavior: if ``output_path`` already exists, every row without an
    ``"error"`` key is treated as already classified and its ``sample_id`` is
    excluded. Resume is keyed purely by ``sample_id`` and does not inspect which
    loader config produced those IDs, so always point distinct loader
    configurations at distinct output paths.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str | int] = set()
    if output_path.exists():
        keep_lines: list[str] = []
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in row:
                    completed.add(row["sample_id"])
                    keep_lines.append(line.rstrip("\n"))
        tmp_path = output_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w") as f:
            for kl in keep_lines:
                f.write(kl + "\n")
        tmp_path.replace(output_path)

    remaining = [s for s in sem_inputs if s.sample_id not in completed]
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}Total: {len(sem_inputs)}, Already done: {len(completed)}, "
        f"Remaining: {len(remaining)}"
    )

    usage_before = client.usage.as_dict()
    if not remaining:
        print(f"{prefix}Nothing to do.")
        return {k: 0 for k in usage_before}

    workers = max(1, min(num_workers, len(remaining)))
    print(
        f"{prefix}Starting {workers} workers "
        f"(concurrency cap arbitrated by the client semaphore)"
    )

    done = 0
    errors = 0
    start_time = time.monotonic()

    write_lock = asyncio.Lock()

    queue: asyncio.Queue[SemanticLayerInput | None] = asyncio.Queue()
    for sem_input in remaining:
        queue.put_nowait(sem_input)
    for _ in range(workers):
        queue.put_nowait(None)

    async def worker(fh: Any) -> None:
        nonlocal done, errors
        while True:
            item = await queue.get()
            if item is None:
                return

            try:
                result = await classify_sample(
                    client,
                    item,
                    num_generations=num_generations,
                    cache_warmup=cache_warmup,
                    verify_and_correct=verify_and_correct,
                    verify_thinking=verify_thinking,
                    verify_reasoning_effort=verify_reasoning_effort,
                    max_tokens=max_tokens,
                    stage2_max_tokens=stage2_max_tokens,
                    verify_votes=verify_votes,
                )
            except Exception as e:
                result = {"sample_id": item.sample_id, "error": f"unexpected: {e}"}

            try:
                async with write_lock:
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()
            except Exception:
                errors += 1
                continue

            if "error" in result:
                errors += 1
            else:
                done += 1

            total_processed = done + errors
            if total_processed % 100 == 0:
                elapsed = time.monotonic() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - total_processed) / rate if rate > 0 else 0
                cost = _usage_cost(client.usage.as_dict(), client.model)
                cost_str = f", ${cost:,.2f} so far" if cost is not None else ""
                print(
                    f"{prefix}Progress: {total_processed}/{len(remaining)} "
                    f"({done} ok, {errors} errors) "
                    f"[{rate:.1f} samples/s, ETA {eta / 60:.0f}m{cost_str}]"
                )

    output_fh = open(output_path, "a")
    try:
        await asyncio.gather(*[worker(output_fh) for _ in range(workers)])
    finally:
        output_fh.close()

    elapsed = time.monotonic() - start_time
    usage_after = client.usage.as_dict()
    delta = {k: usage_after[k] - usage_before.get(k, 0) for k in usage_after}
    print(
        f"{prefix}Done. {done} classified, {errors} errors "
        f"in {elapsed / 60:.1f}m. Results in {output_path}"
    )
    print(f"{prefix}{_format_usage(delta, client.model, label=label)}")
    return delta


# --- Dry-run cost estimation ------------------------------------------------


def estimate_cost(
    sem_inputs: list[SemanticLayerInput],
    *,
    model: str = DEFAULT_MODEL,
    num_generations: int = NUM_GENERATIONS,
    cache_warmup: bool = True,
    verify_and_correct: bool = False,
    thinking: bool = False,
    verify_thinking: bool = False,
    verify_votes: int = 2,
    tokens_per_char: float = 0.3,
    output_tokens_per_turn: int = 45,
    thinking_tokens_per_call: int = 800,
    stage2_flag_rate: float = 0.3,
    stage2_output_tokens: int = 250,
) -> dict[str, Any]:
    """Approximate request / token / cost counts for a run, without calling the API.

    Token counts are estimated from character length (`tokens_per_char`, ~0.3
    for English per DeepSeek's guidance) — they are rough, not exact. Cache
    modeling: the static system prompt is treated as a cache hit on every
    request; with `cache_warmup`, generations 2..N of a sample also hit on the
    user portion. Stage-2 volume is estimated as `stage2_flag_rate` of all no-FC
    turns (the true rate is unknown until stage 1 runs).
    """
    sys_tokens = len(SYSTEM_PROMPT_V3) * tokens_per_char
    stage2_sys_tokens = len(STAGE2_SYSTEM_PROMPT) * tokens_per_char

    n_samples = len(sem_inputs)
    total_user_tokens = 0.0
    total_turns = 0
    for si in sem_inputs:
        msgs = build_prompt(si)
        total_user_tokens += len(msgs[1]["content"]) * tokens_per_char
        total_turns += len(si.no_fc_turn_indices)

    # Stage 1.
    stage1_requests = n_samples * num_generations
    sys_hit = sys_tokens * stage1_requests
    if cache_warmup and num_generations > 1:
        user_miss = total_user_tokens  # one warmup gen per sample
        user_hit = total_user_tokens * (num_generations - 1)
    else:
        user_miss = total_user_tokens * num_generations
        user_hit = 0.0
    stage1_hit = sys_hit + user_hit
    stage1_miss = user_miss
    stage1_out = num_generations * total_turns * output_tokens_per_turn
    if thinking:
        stage1_out += stage1_requests * thinking_tokens_per_call

    # Stage 2 (estimated volume).
    stage2_requests = 0.0
    stage2_hit = stage2_miss = stage2_out = 0.0
    if verify_and_correct:
        stage2_requests = total_turns * stage2_flag_rate * max(1, verify_votes)
        avg_user = (total_user_tokens / n_samples) if n_samples else 0.0
        stage2_hit = stage2_sys_tokens * stage2_requests
        stage2_miss = avg_user * stage2_requests
        stage2_out = stage2_output_tokens * stage2_requests
        if verify_thinking:
            stage2_out += stage2_requests * thinking_tokens_per_call

    hit = stage1_hit + stage2_hit
    miss = stage1_miss + stage2_miss
    out = stage1_out + stage2_out
    totals = {
        "calls": int(round(stage1_requests + stage2_requests)),
        "cache_hit_tokens": int(round(hit)),
        "cache_miss_tokens": int(round(miss)),
        "completion_tokens": int(round(out)),
    }
    return {
        "model": model,
        "samples": n_samples,
        "no_fc_turns": total_turns,
        "stage1_requests": int(stage1_requests),
        "stage2_requests_est": int(round(stage2_requests)),
        "cache_hit_tokens": totals["cache_hit_tokens"],
        "cache_miss_tokens": totals["cache_miss_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "cost_usd": _usage_cost(totals, model),
    }


def _format_estimate(est: dict[str, Any], label: str = "") -> str:
    cost = est["cost_usd"]
    cost_str = f"${cost:,.2f}" if cost is not None else "n/a (unknown model price)"
    head = f"DRY RUN estimate{f' [{label}]' if label else ''} (model={est['model']}):"
    return (
        f"{head}\n"
        f"  samples={est['samples']:,}, no-FC turns={est['no_fc_turns']:,}\n"
        f"  stage-1 requests={est['stage1_requests']:,}, "
        f"stage-2 requests (est)={est['stage2_requests_est']:,}\n"
        f"  input cache-hit tokens={est['cache_hit_tokens']:,}, "
        f"cache-miss tokens={est['cache_miss_tokens']:,}\n"
        f"  output tokens={est['completion_tokens']:,}\n"
        f"  ESTIMATED COST: {cost_str}  (approximate; token counts heuristic)"
    )


# --- Dataset loading + CLI --------------------------------------------------

TXT360_SPLITS = ["high", "medium", "low"]

TxT360SeedGroup = Literal["last", "last_clean", "latest_clean_prefix"]
TXT360_SEED_GROUPS: tuple[TxT360SeedGroup, ...] = (
    "last",
    "last_clean",
    "latest_clean_prefix",
)


# strip_thinking is handled separately by _load_samples so it can be toggled
# independently of filter mode.
_UNIVERSAL_DROP_FILTERS = FilterConfig(
    strip_thinking=False,
    require_parseable_arguments=True,
    require_balanced_cardinality=True,
    require_defined_functions=True,
    require_valid_arguments=True,
)


FILTER_MODES = ("all", "dataset", "universal", "none")


def _load_samples(
    loader_name: str,
    split: str,
    filter_mode: str = "all",
    strip_thinking: bool = True,
    txt360_seed_group: TxT360SeedGroup | None = None,
) -> tuple[list[ConversationSample], LoadReport]:
    apply_dataset = filter_mode in ("all", "dataset")
    apply_universal = filter_mode in ("all", "universal")
    filters = _UNIVERSAL_DROP_FILTERS if apply_universal else None

    match loader_name:
        case "apigen_mt":
            from .loaders import apigen_mt

            ds_cfg = (
                apigen_mt.APIGenMTConfig(
                    strip_think_tool=True,
                    drop_undefined_function_calls=True,
                    drop_repeated_tool_call_streaks=True,
                    drop_error_recovery_loops=True,
                    drop_consecutive_assistant=True,
                )
                if apply_dataset
                else None
            )
            samples, report = apigen_mt.load(
                dataset_config=ds_cfg, filter_config=filters
            )

        case "dolci":
            from .loaders import dolci

            ds_cfg = (
                dolci.DolciConfig(
                    drop_consecutive_text_text_assistant=True,
                    merge_text_fc_assistant=True,
                )
                if apply_dataset
                else None
            )
            samples, report = dolci.load(dataset_config=ds_cfg, filter_config=filters)

        case "nemotron_agentic_v1":
            from .loaders import nemotron_agentic_v1

            # Always pass splits to avoid loading interactive_agent
            # (excluded from training). Drop flags depend on filter_mode.
            ds_cfg = nemotron_agentic_v1.NemotronAgenticV1Config(
                splits=("tool_calling",),
                drop_orphan_samples=apply_dataset,
                drop_conflicting_duplicate_tools=apply_dataset,
                drop_empty_system=apply_dataset,
            )
            samples, report = nemotron_agentic_v1.load(
                dataset_config=ds_cfg, filter_config=filters
            )

        case "nemotron_agentic_v2":
            from .loaders import nemotron_agentic_v2

            # This loader has no dataset-specific drop/transform flags; its
            # default split selection is the training config.
            samples, report = nemotron_agentic_v2.load(filter_config=filters)

        case "nemotron_terminal":
            from .loaders import nemotron_terminal

            ds_cfg = (
                nemotron_terminal.NemotronTerminalConfig(
                    strip_malformed=True,
                    drop_orphans=True,
                    drop_incomplete=True,
                )
                if apply_dataset
                else None
            )
            samples, report = nemotron_terminal.load(
                dataset_config=ds_cfg, filter_config=filters
            )

        case "toolmind":
            from .loaders import toolmind

            ds_cfg = (
                toolmind.ToolMindConfig(
                    sources=["graph_syn_datasets/graphsyn.jsonl"],
                    seed_group_filter="longest_clean",
                    drop_non_object_arguments=True,
                    drop_consecutive_text_assistant=True,
                    merge_split_assistant=True,
                    strip_think_tool=True,
                )
                if apply_dataset
                else None
            )
            samples, report = toolmind.load(
                dataset_config=ds_cfg, filter_config=filters
            )

        case "txt360":
            from .loaders import txt360

            if txt360_seed_group is not None:
                ds_cfg = txt360.TxT360Config(
                    seed_group_filter=txt360_seed_group,
                    require_non_empty_user=apply_dataset,
                    drop_user_tool_call_samples=apply_dataset,
                )
            elif apply_dataset:
                ds_cfg = txt360.TxT360Config(
                    seed_group_filter="latest_clean_prefix",
                    require_non_empty_user=True,
                    drop_user_tool_call_samples=True,
                )
            else:
                ds_cfg = None
            samples, report = txt360.load(
                split=split,
                dataset_config=ds_cfg,
                filter_config=filters,
            )

        case _:
            raise ValueError(f"Unknown loader: {loader_name!r}")

    if strip_thinking:
        samples = [strip_thinking_from_sample(s) for s in samples]
        report.strip_thinking_applied = True

    return samples, report


def _output_path_for(output_dir: Path, dataset: str, split: str) -> Path:
    # TxT360 has distinct splits with disjoint populations; keep them in
    # separate files so resume-by-sample-id never collides across splits.
    if dataset == "txt360":
        return output_dir / f"semantic_results_txt360_{split}.jsonl"
    return output_dir / f"semantic_results_{dataset}.jsonl"


def _prepare_inputs(
    dataset: str,
    args: argparse.Namespace,
) -> list[SemanticLayerInput]:
    do_strip = not args.no_strip_thinking
    seed_group_note = (
        f", txt360_seed_group={args.txt360_seed_group!r}"
        if args.txt360_seed_group is not None and dataset == "txt360"
        else ""
    )
    print(
        f"[{dataset}] Loading (filter_mode={args.filter_mode!r}, "
        f"strip_thinking={do_strip}{seed_group_note})..."
    )
    samples, report = _load_samples(
        dataset,
        args.split,
        args.filter_mode,
        do_strip,
        args.txt360_seed_group,
    )
    print(report.summary())
    print(f"[{dataset}] Kept {len(samples)} samples after filtering.")

    analyses = analyze_dataset_behavior(samples)
    sem_inputs = prepare_semantic_layer_inputs(samples, analyses)
    print(f"[{dataset}] Samples with no-FC turns: {len(sem_inputs)}")

    if args.limit is not None:
        sem_inputs = sem_inputs[: args.limit]
        print(f"[{dataset}] Limited to first {len(sem_inputs)} samples (--limit).")
    return sem_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Semantic layer: classify no-FC turns over an OpenAI-compatible "
            "endpoint (default: the local vLLM Qwen judge; DeepSeek's API also "
            "supported via --base-url/--model). Loops over --datasets, writing one "
            "resume-friendly JSONL per dataset under --output-dir. Use --dry-run "
            "to estimate cost without calling the API."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=LOADER_NAMES,
        help="One or more dataset loaders to run, in order.",
    )
    parser.add_argument(
        "--split",
        default="high",
        choices=TXT360_SPLITS,
        help="TxT360 split (ignored for other loaders).",
    )
    parser.add_argument(
        "--filter-mode",
        default="all",
        choices=FILTER_MODES,
        help=(
            "Which Stage 2 drop filters to apply. "
            "'all' (default): dataset-specific + universal. "
            "'dataset': dataset-specific only. "
            "'universal': universal only. "
            "'none': raw Stage 1 output."
        ),
    )
    parser.add_argument(
        "--no-strip-thinking",
        action="store_true",
        help=(
            "Keep thinking traces (<think>/<reasoning> tags and "
            "reasoning_content fields) in assistant messages. "
            "By default, thinking is stripped regardless of --filter-mode."
        ),
    )
    parser.add_argument(
        "--txt360-seed-group",
        default=None,
        choices=TXT360_SEED_GROUPS,
        help=(
            "Override TxT360's seed-group selection mode regardless of "
            "--filter-mode. Ignored for non-TxT360 loaders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="semantic_results",
        help=(
            "Directory for per-dataset JSONL outputs "
            "(semantic_results_<dataset>.jsonl)."
        ),
    )

    # --- Model / API ---
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model name served by the endpoint."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible API base URL (local vLLM or DeepSeek).",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable holding the API key.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Max concurrent in-flight API calls. ONE asyncio semaphore bounds "
            "ALL calls — every generation, every retry, every stage-2 request — "
            "so this equals the peak DeepSeek concurrent-connection count. "
            "Default: just under the model's account cap (480 for "
            "deepseek-v4-pro, cap 500). Setting it above the cap only buys 429 "
            "backoff, not throughput."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="OpenAI SDK retries for transient 429/5xx errors (exp. backoff).",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-request timeout (seconds)."
    )

    # --- Classification behavior ---
    parser.add_argument(
        "--num-generations",
        type=int,
        default=NUM_GENERATIONS,
        help="Generations per sample for majority vote.",
    )
    parser.add_argument(
        "--verify-and-correct-flags",
        action="store_true",
        help="Enable the stage-2 verify-and-correct pass on anti-pattern turns.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Enable DeepSeek thinking mode. NOTE: thinking ignores temperature, "
            "so multi-generation votes will not diverge — use with "
            "--num-generations 1."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high", "max"],
        help="Reasoning effort, applied to whichever stage runs in thinking mode.",
    )
    parser.add_argument(
        "--verify-thinking",
        action="store_true",
        help=(
            "Run the stage-2 verify-and-correct pass in DeepSeek thinking mode "
            "(recommended: stage 2 is N=1 and reasoning-heavy). Independent of "
            "--thinking (stage 1). Requires --verify-and-correct-flags."
        ),
    )
    parser.add_argument(
        "--verify-votes",
        type=int,
        default=2,
        help=(
            "Independent stage-2 samples per flagged turn. With 2 (default) an "
            "agreement gate applies: all anti -> confirmed; all justified -> "
            "overturned; split -> justified + contested flag (the verifier is "
            "self-inconsistent on that turn). 1 disables the gate."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for non-thinking mode (vote diversity).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus top_p (non-thinking). Qwen instruct default 0.8.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="top_k (non-thinking; vLLM/Qwen only, via extra_body). Qwen default 20.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="min_p (non-thinking; vLLM/Qwen only, via extra_body).",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help=(
            "presence_penalty (non-thinking). Kept at 0.0 for this task on "
            "purpose: the judge emits structured JSON with one classification "
            "PER no-FC turn in a single response, so Qwen's recommended 1.5-2.0 "
            "would penalize repeating JSON keys and the same category across "
            "turns. Raise only to experiment."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="repetition_penalty (non-thinking; vLLM/Qwen only, via extra_body).",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable response_format json_object (rely on prompt + parsing).",
    )
    parser.add_argument(
        "--no-cache-warmup",
        action="store_true",
        help=(
            "Disable the warmup-then-fan-out cache pattern (fire all N "
            "generations concurrently; forgoes the intra-sample cache discount)."
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Max output tokens (stage 1)."
    )
    parser.add_argument(
        "--stage2-max-tokens",
        type=int,
        default=None,
        help=(
            "Max output tokens for stage 2 (default 3072, or 8192 with "
            "--verify-thinking to fit reasoning plus the corrected turn)."
        ),
    )

    # --- Cost / scale guardrails ---
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples (per dataset) with no-FC turns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate request/token/cost counts and exit without calling the API.",
    )
    parser.add_argument(
        "--stage2-flag-rate",
        type=float,
        default=0.3,
        help="Assumed fraction of no-FC turns flagged (for --dry-run stage-2 estimate).",
    )

    args = parser.parse_args()

    if args.thinking and args.num_generations > 1:
        print(
            "WARNING: --thinking ignores temperature, so the "
            f"{args.num_generations} generations will be (near-)identical and "
            "majority vote is degenerate. Consider --num-generations 1."
        )
    if args.verify_thinking and not args.verify_and_correct_flags:
        print(
            "WARNING: --verify-thinking has no effect without "
            "--verify-and-correct-flags (stage 2 is not running)."
        )

    json_mode = not args.no_json_mode
    cache_warmup = not args.no_cache_warmup
    stage2_max_tokens = (
        args.stage2_max_tokens
        if args.stage2_max_tokens is not None
        else (8192 if args.verify_thinking else 3072)
    )

    # ---- Dry run: no client, no API key required. ----
    if args.dry_run:
        grand: dict[str, Any] | None = None
        for dataset in args.datasets:
            sem_inputs = _prepare_inputs(dataset, args)
            est = estimate_cost(
                sem_inputs,
                model=args.model,
                num_generations=args.num_generations,
                cache_warmup=cache_warmup,
                verify_and_correct=args.verify_and_correct_flags,
                thinking=args.thinking,
                verify_thinking=args.verify_thinking,
                verify_votes=args.verify_votes,
                stage2_flag_rate=args.stage2_flag_rate,
            )
            print(_format_estimate(est, label=dataset))
            out_path = _output_path_for(Path(args.output_dir), dataset, args.split)
            print(f"  would write: {out_path}")
            if grand is None:
                grand = {
                    k: v
                    for k, v in est.items()
                    if k
                    in (
                        "cache_hit_tokens",
                        "cache_miss_tokens",
                        "completion_tokens",
                        "stage1_requests",
                        "stage2_requests_est",
                    )
                }
            else:
                for k in grand:
                    grand[k] += est[k]
        if grand is not None and len(args.datasets) > 1:
            total_cost = _usage_cost(grand, args.model)
            cost_str = f"${total_cost:,.2f}" if total_cost is not None else "n/a"
            print(
                f"\nDRY RUN grand total across {len(args.datasets)} datasets: "
                f"{grand['stage1_requests']:,} stage-1 + "
                f"{grand['stage2_requests_est']:,} stage-2 (est) requests, "
                f"ESTIMATED COST {cost_str}"
            )
        print("\nDry run complete. No API calls were made.")
        return

    # ---- Live run. ----
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(
            f"API key not found in ${args.api_key_env}. "
            f"Export it or pass --api-key-env, or use --dry-run."
        )

    concurrency = (
        args.concurrency
        if args.concurrency is not None
        else _default_concurrency(args.model)
    )
    cap = CONCURRENCY_CAP.get(args.model)
    if cap is not None and concurrency > cap:
        print(
            f"WARNING: --concurrency {concurrency} exceeds the {args.model} "
            f"account cap of {cap}; expect HTTP 429s (the SDK will back off)."
        )
    print(
        f"Concurrency: up to {concurrency} simultaneous in-flight API calls"
        + (f" (account cap {cap})." if cap is not None else ".")
    )

    client = JudgeClient(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        concurrency=concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        json_mode=json_mode,
    )

    async def _run_all() -> None:
        print(f"Warming up endpoint {args.base_url} (model={args.model})...")
        await client.warmup()
        try:
            for dataset in args.datasets:
                sem_inputs = _prepare_inputs(dataset, args)
                if not sem_inputs:
                    print(f"[{dataset}] No samples with no-FC turns; skipping.")
                    continue
                out_path = _output_path_for(Path(args.output_dir), dataset, args.split)
                await run_semantic_layer(
                    sem_inputs,
                    client,
                    out_path,
                    num_generations=args.num_generations,
                    cache_warmup=cache_warmup,
                    verify_and_correct=args.verify_and_correct_flags,
                    verify_thinking=args.verify_thinking,
                    verify_reasoning_effort=args.reasoning_effort,
                    max_tokens=args.max_tokens,
                    stage2_max_tokens=stage2_max_tokens,
                    verify_votes=args.verify_votes,
                    num_workers=concurrency,
                    label=dataset,
                )
        finally:
            await client.aclose()

        print(f"\n{_format_usage(client.usage.as_dict(), client.model, label='total')}")

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
