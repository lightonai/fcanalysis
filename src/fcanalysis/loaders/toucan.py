"""Loader for Agent-Ark/Toucan-1.5M.

Toucan is a synthetic tool-agent SFT dataset: 1,527,259 trajectories generated
against 495+ real MCP servers (4,868 distinct tools) by three teacher models,
each locked to one agent framework (verified, see toucan/ census c01-c09):

    config    teacher          framework      rows     reasoning   parallel
    Kimi-K2   Kimi-K2          Qwen-Agent     518,516  no          11.5% turns
    OSS       GPT-OSS-120B     OpenAI-Agent   457,130  yes (field) 0%
    Qwen3     Qwen3-32B        Qwen-Agent     551,613  no           8.4% turns

A fourth published config, ``SFT`` (119,287 rows), is a reformatted (ms-swift
roles), quality-filtered, re-balanced subset of the Kimi-K2 config (uuid-contained
100%, trajectory-identical 99.99%; toucan census c06). This loader handles the
three teacher configs only; ``SFT`` is a derived artifact and is reproduced by
loading Kimi-K2 with the quality gate, not loaded directly.

Raw teacher schema (9 string columns, revision 0df3cf37): ``uuid``,
``subset_name`` in {single-turn-original, single-turn-diversify, irrelevant,
multi-turn}, ``messages`` (JSON string), ``question``, ``available_tools`` (JSON
string, OpenAI tool schema), ``target_tools`` (comma-separated short names),
``question_quality_assessment``, ``response_quality_assessment`` (JSON strings;
the irrelevant subset carries an empty response assessment), ``metadata`` (JSON
string).

Raw message format (verified over all 1.53M rows): roles system / user /
assistant / function. Assistant turns are split: reasoning/answer text and each
tool call are separate consecutive messages. A tool call is an assistant message
with empty ``content`` and a legacy ``function_call: {name, arguments}`` (one call
per message; no ``tool_calls`` array exists). Parallel calls are a run of K such
messages followed by K ``function`` responses in the same order
(``function.name`` matches the call name, 100% over balanced runs). OSS assistant
messages additionally carry a ``reasoning_content`` field (the GPT-OSS harmony
channel, 1.9M messages); Kimi-K2/Qwen3 do not.

Mapping to OpenAI format (collapses the split messages; validated by the
round-trip and system-context censuses):
    - The two exact teacher-framework tool templates are parsed from raw system
      messages. Their complete definitions are authoritative because that is the
      context serialized to the teacher model. The verified template span is
      removed and the target chat template renders the extracted definitions once
      from ``sample.tools``.
    - ``available_tools`` is generation metadata, not always the model-visible
      contract. In the pinned snapshot, 1,017,937 rows have non-empty but different
      system/metadata definition multisets; 488,369 match exactly. The 20,953 rows
      without a recognized system template fall back to ``available_tools``.
      Sources are never unioned, because that would expose tools the model did not
      see and make unsupported calls appear defined.
    - All 1,506,306 captured system messages are exactly one valid known template
      and contain no independent instructions. The recognizer nevertheless handles
      future templates embedded at the beginning, middle, or end: it removes only
      exact structurally valid tool spans, retains all other bytes, and fails
      closed on malformed or unfamiliar candidates. The original row remains in
      ``sample.raw`` for audit reconstruction.
    - A maximal run of consecutive assistant messages becomes ONE OpenAI
      assistant message: ``content`` = the joined non-empty text contents;
      ``reasoning_content`` = the joined reasoning fields (OSS only, so
      strip_thinking can remove it); ``tool_calls`` = one entry per
      ``function_call`` in run order, ``{type: function, function: {name,
      arguments}}`` with no generated id (positional linkage per the checklist).
    - Each ``function`` message becomes a ``tool`` message (content = the raw tool
      output). Positional linkage: the Nth tool message after an assistant turn
      answers the Nth tool_call. Name-order match makes this exact (census c04).

Lossiness: the original split boundaries (which text went with which call message,
the per-message empty contents, the ``function.name`` echo) are not reconstructable
from the OpenAI projection alone, but ``sample.raw`` holds the unmodified row, so
reconstruction is 100% exact from raw. Re-serialization round-trip (rebuild the
raw ``messages`` list from the projection) is measured in c10; the residual is the
text/call grouping ambiguity, never silent data loss. Unparseable ``function_call``
arguments (~0.4% of qwen-agent calls, 0 for OSS) are preserved as-is and counted
as Stage 1 issues.

Dataset-specific config (Stage 2, before universal filters):
    - ``subsets``: keep only these ``subset_name`` values (default: all four).
      Excluding ``irrelevant`` removes the no-tool-call decline trajectories;
      keeping only it isolates them.
    - ``drop_low_quality``: drop rows whose assessments fail the gate
      (defaults reproduce the paper's SFT thresholds: question_quality >= 5,
      scenario_realism >= 5, completeness >= 4, conciseness >= 4,
      desired_tools_used_percentage == 1.0). The ``irrelevant`` subset has no
      response assessment and is gated on the question side only (question_quality
      and scenario_realism), never dropped for missing response scores.
    - ``drop_incomplete_termination``: drop rows whose last message is a tool
      response or an empty assistant message (no final answer; <1% of rows).
    - ``drop_conflicting_duplicate_tools``: drop rows whose canonical
      model-visible ``sample.tools`` defines the same exact visible name with more
      than one distinct complete tool definition (name-based lookup would be
      order-dependent).
      Object-key order is canonicalized, but no schema or description is
      heuristically normalized. Same-name variation across different rows is
      allowed; byte-equivalent duplicates within one row are not conflicts.
    - ``strip_scaffold_tools`` (default off): an explicit counterfactual
      transform, run on survivors after the drops, that removes selected
      framework-plumbing calls and their positionally linked responses. It can
      remove server-unlock handshakes, MCP resource primitives, and the
      ``deep_researcher`` async protocol. These calls are real and often
      information-bearing, so production keeps them. The counterfactual can hide
      malformed calls or delete observations used by later assistant messages;
      it must not be interpreted as a quality improvement.
    - Reasoning tools are never rewritten. Twenty-two exact audited action
      families are preserved intact and annotated outside model-visible content
      under ``reasoning_tool_families`` when actually called. Uncalled
      definitions remain available but unmarked. Six ordinary Clear Thought
      session-state identities and two information-bearing resource identities
      are also retained without reasoning-family markers. Fuzzy, misspelled,
      case-shifted, undefined, conflicting, and unbalanced calls remain visible
      to the ordinary validators rather than being deleted by a name heuristic.

      NOTE: reasoning-tool preservation, the opt-in scaffold counterfactual, and
      canonical tool-context normalization are deliberate divergences from the
      upstream loader.

Universal filters then apply (strip_thinking removes the OSS reasoning_content;
the four drop filters compose orthogonally per the base class).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import pyarrow.parquet as pq

from ..format import ConversationSample
from .base import FilterConfig, LoadReport, apply_filters

DATASET_ID = "Agent-Ark/Toucan-1.5M"
REVISION = "0df3cf37f2abefb380370cfb02eabea2a35ae782"
TEACHER_CONFIGS = ("Kimi-K2", "OSS", "Qwen3")
SUBSETS = (
    "single-turn-original",
    "single-turn-diversify",
    "irrelevant",
    "multi-turn",
)


@dataclass
class ToucanConfig:
    """Dataset-specific configuration for Toucan-1.5M (teacher configs)."""

    subsets: tuple[str, ...] = SUBSETS
    drop_low_quality: bool = False
    min_question_quality: int = 5
    min_scenario_realism: int = 5
    min_completeness: int = 4
    min_conciseness: int = 4
    require_full_tool_use: bool = True
    drop_incomplete_termination: bool = False
    drop_conflicting_duplicate_tools: bool = False
    # Framework scaffold plumbing is kept by default because it is real, often
    # information-bearing tool use. This opt-in counterfactual can hide malformed
    # calls and remove observations used by the final answer.
    strip_scaffold_tools: bool = False


# Retained solely to reproduce and census the retired name-based policy. This
# heuristic never authorizes production transformation. Exact audited reasoning
# identities and ordinary state/resource false positives are excluded so the
# helper reports only unresolved legacy candidates.
_LEGACY_REASONING_SUBSTRINGS = (
    "thinking",
    "clear_thought",
    "clear-thought",
    "think-tool",
    "mentalmodel",
    "analogicalreasoning",
    "collaborativereasoning",
    "visualreasoning",
    "decisionframework",
    "scientificmethod",
    "socraticmethod",
    "metacognit",
    "structuredargumentation",
    "debuggingapproach",
    "designpattern",
    "chain-of-draft",
    "chain_of_draft",
    "lotuswisdom",
)


# These Clear Thought operations are ordinary stateful tools, not reasoning
# traces. Their observations expose session state that is unavailable in the
# call arguments, and clear_thoughts mutates that state. The pinned snapshot
# contains exactly three audited naming families. Do not infer family membership
# from an arbitrary ``*-get_thoughts`` suffix: the snapshot contains an undefined
# ``mcpollinations-get_thoughts`` hallucination that is not a Clear Thought tool.
_THOUGHT_STATE_OPERATIONS = (
    "get_thoughts",
    "get_thought_stats",
    "clear_thoughts",
)

_THOUGHT_STATE_NAMESPACES = (
    "",  # bare: think / get_thoughts / get_thought_stats / clear_thoughts
    "think-tool",
    "think-tool-server",
)

_THOUGHT_STATE_NAMESPACE_BY_NAME = {
    (operation if not namespace else f"{namespace}-{operation}"): namespace
    for namespace in _THOUGHT_STATE_NAMESPACES
    for operation in _THOUGHT_STATE_OPERATIONS
}

_THOUGHT_WRITE_NAMESPACE_BY_NAME = {
    ("think" if not namespace else f"{namespace}-think"): namespace
    for namespace in _THOUGHT_STATE_NAMESPACES
}


# Exact call-name vocabularies established by the full pinned-snapshot audits.
# Preservation is deliberately name-exact: case variants, misspellings, and
# names with serialized arguments appended do not inherit an audited identity.
# Such undefined calls remain visible so universal defined-function validation
# can reject them rather than being bypassed.
REASONING_TOOL_FAMILIES_ANNOTATION = "reasoning_tool_families"
THINK_TOOL_FAMILY = "think_tool"
SEQUENTIAL_THINKING_TOOL_FAMILY = "sequential_thinking"
PENTEST_THINKING_TOOL_FAMILY = "pentest_thinking"
GAME_DESIGN_THINKING_TOOL_FAMILY = "game_design_thinking"
SKIA_ANIMATION_THINKING_TOOL_FAMILY = "skia_animation_thinking"
LOTUS_WISDOM_TOOL_FAMILY = "lotus_wisdom"
STRUCTURED_ARGUMENTATION_TOOL_FAMILY = "structured_argumentation"
ANALOGICAL_REASONING_TOOL_FAMILY = "analogical_reasoning"
MENTAL_MODEL_TOOL_FAMILY = "mental_model"
DECISION_FRAMEWORK_TOOL_FAMILY = "decision_framework"
SCIENTIFIC_METHOD_TOOL_FAMILY = "scientific_method"
DEBUGGING_APPROACH_TOOL_FAMILY = "debugging_approach"
DESIGN_PATTERN_TOOL_FAMILY = "design_pattern"
COLLABORATIVE_REASONING_TOOL_FAMILY = "collaborative_reasoning"
VISUAL_REASONING_TOOL_FAMILY = "visual_reasoning"
METACOGNITIVE_MONITORING_TOOL_FAMILY = "metacognitive_monitoring"
PROGRAMMING_PARADIGM_TOOL_FAMILY = "programming_paradigm"
CHAIN_OF_DRAFT_TOOL_FAMILY = "chain_of_draft"
CREATIVE_THINKING_TOOL_FAMILY = "creative_thinking"
SYSTEMS_THINKING_TOOL_FAMILY = "systems_thinking"
SOCRATIC_METHOD_TOOL_FAMILY = "socratic_method"
CLEAR_THOUGHT_DISPATCHER_TOOL_FAMILY = "clear_thought_dispatcher"

_THINK_TOOL_NAMES = (
    frozenset(_THOUGHT_STATE_NAMESPACE_BY_NAME)
    | frozenset(_THOUGHT_WRITE_NAMESPACE_BY_NAME)
    | frozenset(
        {
            # Independent audited deployments. Marcopesani returns one of three
            # random encouragements; Think Tank returns structured step metadata.
            # Neither is the state namespace used by the twelve names above.
            "think-mcp-server-think",
            "think-tank-think",
        }
    )
)

_SEQUENTIAL_THINKING_TOOL_NAMES = frozenset(
    {
        "clear-thought-sequentialthinking",
        "clear-thought-server-sequentialthinking",
        "model-context-protocol-server-sequentialthinking",
        "reference-servers-sequentialthinking",
        "sequential-thinking-sequentialthinking",
        "sequential-thinking-tools-sequentialthinking_tools",
        "sequentialthinking",
        "sequentialthinking_tools",
    }
)

# PentestThinking validates and scores model-authored attack steps, selects a
# search strategy, and returns server-created scores/node IDs. The pinned MCTS
# results include runtime-derived identifiers, so this is not an echo contract.
_PENTEST_THINKING_TOOL_NAMES = frozenset(
    {
        "pentestthinkingMCP",
        "pentestthinking-pentestthinkingMCP",
    }
)

# Game-design calls return cumulative title/component/library/branch/history
# state. The advertised summary/export companions are broken in the pinned
# deployment, but that does not make the write observations episode-independent.
_GAME_DESIGN_THINKING_TOOL_NAMES = frozenset(
    {
        "gamedesignthinking",
        "game-engine-server-gamedesignthinking",
    }
)

# Skia-animation calls return turn/session-local history and branch state plus
# sequence metadata. Domain claims remain model-authored, but the explicit tool
# episode is stateful and is retained for later row-level selection.
_SKIA_ANIMATION_THINKING_TOOL_NAMES = frozenset(
    {
        "skiaanimationthinking",
        "react-native-skia-animation-thinking-tool-skiaanimationthinking",
    }
)

# Lotus Wisdom is a stateful contemplative protocol, not a call-local echo. The
# write operation appends validated steps and returns cumulative tag/domain
# journeys. The summary operation exposes the accumulated steps, including
# truncated content that is absent from the summary call arguments. The pinned
# snapshot uses one bare OSS spelling and one Qwen-Agent-qualified spelling for
# each operation; their definitions are otherwise identical. State is observed
# to reset at each user turn in Toucan's generated trajectories. Preserve only
# these four exact, definition-backed names: malformed and fuzzy variants do not
# inherit the audited identity.
_LOTUS_WISDOM_TOOL_NAMES = frozenset(
    {
        "lotuswisdom",
        "lotuswisdom_summary",
        "lotus-wisdom-lotuswisdom",
        "lotus-wisdom-lotuswisdom_summary",
    }
)

# Structured Argumentation is a shared protocol label, not one implementation.
# The pinned snapshot has five exact, definition-backed visible identities across
# three audited server lineages:
#   * Waldzell's standalone server keeps a user-turn-local argument history and
#     relationship graph and returns cumulative history/graph counts plus IDs;
#   * the Chirag/ThinkFar Clear Thought lineage produces runtime timestamp IDs
#     when the model omits argumentId, and later calls reuse those observations;
#   * Waldzell's compact Clear Thought lineage mutates a session store and returns
#     a server-created session UUID and aggregate session/store statistics.
# All successful definition-backed raw results replay against those exact
# contracts. Error and malformed episodes stay visible to the ordinary
# validators. Preserve only these case-sensitive identities:
# underscore/case/composite/appended-JSON lookalikes in the raw snapshot are
# undefined and do not inherit protection.
_STRUCTURED_ARGUMENTATION_TOOL_NAMES = frozenset(
    {
        "structured-argumentation-server-structuredArgumentation",
        "structuredArgumentation",
        "clear-thought-server-structuredargumentation",
        "structuredargumentation",
        "clear-thought-structuredargumentation",
    }
)

# Analogical Reasoning is one standalone Waldzell MCP lineage rendered under a
# qualified Kimi-K2/Qwen3 name and a bare OSS name. The model supplies the
# substantive domains, mappings, scores, justifications, inferences, limits,
# and continuation state. The server validates/filters them, assigns missing
# element IDs, mutates an internal history/domain registry, writes a mapping
# visualization to stderr, and returns a compact current-call projection. All
# 7,805 captured successes replay exactly without a samplingSummary, although a
# capture-compatible source revision can request an MCP client-model summary.
# Errors and malformed calls explain retries and must remain visible to the
# ordinary validators. Preserve only these two case-sensitive identities; the
# raw misspelled and cross-server lookalikes are undefined attempt evidence.
_ANALOGICAL_REASONING_TOOL_NAMES = frozenset(
    {
        "analogical-reasoning-server-analogicalReasoning",
        "analogicalReasoning",
    }
)

# Clear Thought mental-model calls contain the model-authored problem analysis,
# but the deployed servers are explicit tools rather than native hidden thought.
# The Chirag/ThinkFar lineage validates/coerces the call and returns deterministic
# status metadata. Waldzell's lineage additionally mutates session state and
# returns a runtime-supplied session UUID, cumulative model count, and the three
# most recent model/problem pairs. The pinned snapshot contains three exact
# visible names and four complete definition fingerprints: the bare OSS name is
# intentionally shared across the two lineages in different rows. All 11,680
# parseable definition-backed observations replay exactly against those source
# contracts; 17 malformed-JSON episodes retain their harness errors. Preserve
# only these case-sensitive names. Underscore/cross-server/severely misspelled
# lookalikes remain undefined attempt evidence and do not inherit audited-family
# membership.
_MENTAL_MODEL_TOOL_NAMES = frozenset(
    {
        "mentalmodel",
        "clear-thought-server-mentalmodel",
        "clear-thought-mentalmodel",
    }
)

# Decision Framework is a shared protocol label across three audited MCP
# lineages in the pinned snapshot. Chirag/ThinkFar shallowly validates and then
# returns the model-authored decision object, with formatter errors remaining
# visible. Waldzell's standalone server assigns missing element IDs, records
# decision state, calculates expected-value or multi-criteria scores, and
# returns a compact projection. Waldzell Clear Thought stores decisions and
# exposes session/store context; that exact spelling is definition-only in the
# snapshot. The five exact visible names span six complete definition
# fingerprints because bare ``decisionframework`` is shared by two lineages in
# different rows. All 13,263 definition-backed captured calls replay against
# the two called source contracts, including 187 argument-decode/request errors
# and 951 runtime/formatter failures. Preserve only these case-sensitive names;
# the observed underscore spelling and serialized Sequential Thinking names are
# undefined anomaly evidence and do not inherit this audited identity.
_DECISION_FRAMEWORK_TOOL_NAMES = frozenset(
    {
        "decisionFramework",
        "decision-framework-server-decisionFramework",
        "decisionframework",
        "clear-thought-server-decisionframework",
        "clear-thought-decisionframework",
    }
)

# Scientific Method is a shared protocol label across three audited MCP
# lineages in the pinned snapshot. Chirag/ThinkFar shallowly validates the
# model-authored inquiry and returns deterministic stage/status metadata.
# Waldzell's standalone server performs stage-specific validation, stores
# inquiry/hypothesis/experiment state, and returns a compact current-call
# projection. Waldzell Clear Thought stores inquiries and is designed to expose
# hidden session context; both of its captured definition fingerprints are
# definition-only here. The five exact visible names span six complete
# definition fingerprints because bare ``scientificmethod`` is shared by two
# lineages in different rows. The full replay accounts for all 16,626 exact
# definition-backed calls: 14,375 successes, 1,235 handler errors, 1,009
# argument-decode/request errors, and seven transport failures whose handler
# result is unobservable. Preserve only these case-sensitive names. The nine
# observed underscore, case, truncated, suffixed, and cross-server attempts are
# undefined anomaly evidence and do not inherit this audited identity.
_SCIENTIFIC_METHOD_TOOL_NAMES = frozenset(
    {
        "scientificMethod",
        "scientific-method-server-scientificMethod",
        "scientificmethod",
        "clear-thought-server-scientificmethod",
        "clear-thought-scientificmethod",
    }
)

# Debugging Approach and Design Pattern are explicit Clear Thought operations,
# not native hidden chain-of-thought. The Chirag/ThinkFar lineage validates and
# normalizes model-authored debugging/design structures and returns deterministic
# status metadata; all 16,890 handler-produced observations in the pinned
# snapshot replay exactly against those handlers. Its remaining 57 calls expose
# argument-decode failures and are retained for ordinary parseability checks.
# Waldzell's distinct Debugging Approach contract mutates session state and is
# designed to return a runtime session UUID plus cumulative/recent debugging
# context; both Waldzell definition fingerprints are definition-only in this
# snapshot. The five exact visible names span six complete definition
# fingerprints because bare ``debuggingapproach`` is shared across the two
# lineages in different rows. Preserve only these case-sensitive identities.
# The observed underscore spelling, unsupported Clear Thought Design alias,
# integration-pattern name, and malformed composite names are undefined and do
# not inherit audited-family membership.
_DEBUGGING_APPROACH_TOOL_NAMES = frozenset(
    {
        "debuggingapproach",
        "clear-thought-server-debuggingapproach",
        "clear-thought-debuggingapproach",
    }
)

_DESIGN_PATTERN_TOOL_NAMES = frozenset(
    {
        "designpattern",
        "clear-thought-server-designpattern",
    }
)

# Collaborative Reasoning is a shared protocol label across three audited MCP
# lineages in the pinned snapshot. Chirag/ThinkFar shallowly validates the
# model-authored collaboration object, executes a formatting pass whose nested
# field failures are externally visible, and otherwise echoes the object. The
# standalone Waldzell server validates/coerces personas, contributions, and
# disagreements, mutates registries/history, can select the next persona, and
# returns a compact projection. Waldzell Clear Thought stores sessions and
# returns a runtime session UUID plus recent-session context. The five exact
# visible names span six complete definition fingerprints because bare
# ``collaborativereasoning`` is shared by the legacy and compact lineages in
# separate rows. Preserve only these case-sensitive identities; observed
# underscore, misspelled, and composite names do not inherit membership.
_COLLABORATIVE_REASONING_TOOL_NAMES = frozenset(
    {
        "collaborativeReasoning",
        "collaborative-reasoning-server-collaborativeReasoning",
        "collaborativereasoning",
        "clear-thought-server-collaborativereasoning",
        "clear-thought-collaborativereasoning",
    }
)

# Visual Reasoning likewise spans three audited MCP lineages and five exact
# visible names/six definition fingerprints. The legacy handler logs every
# element before returning deterministic diagram metadata, so missing nested
# properties can produce an embedded handler error. The standalone server is
# explicitly stateful: at Toucan's observed user-turn scope it accumulates
# diagram elements and operation history, assigns missing element IDs, and
# returns cumulative counts. Its implementation also requires a truthy
# transformationType even though the advertised schema marks it optional; the
# resulting structured failures remain valuable visible evidence. Waldzell
# Clear Thought stores operations and is designed to return a runtime session
# UUID, aggregate store statistics, and recent operations; its two definitions
# are uncalled in this snapshot. Preserve only these case-sensitive identities.
_VISUAL_REASONING_TOOL_NAMES = frozenset(
    {
        "visualReasoning",
        "visual-reasoning-server-visualReasoning",
        "visualreasoning",
        "clear-thought-server-visualreasoning",
        "clear-thought-visualreasoning",
    }
)

# Metacognitive Monitoring is a shared protocol label across three audited MCP
# lineages and five exact visible identities/six definition fingerprints in the
# pinned Toucan snapshot. The Chirag/ThinkFar Clear Thought handler shallowly
# validates the model-authored assessment, can expose formatter failures, and
# otherwise returns deterministic status metadata. Waldzell's standalone
# server performs deeper validation/coercion, mutates monitoring/knowledge/claim
# state, and returns a compact current-call projection. Waldzell Clear Thought
# stores assessments and is designed to expose runtime session/store context;
# both of its exact definitions are uncalled in this snapshot. The complete
# census found 5,739 exact calls: 5,732 are definition-backed and seven bare
# lowercase calls are undefined. Every one of the 5,658 parseable defined
# call/result pairs replays exactly against its pinned lineage contract; the
# other 74 expose argument-decode failures. Preserve only these case-sensitive
# identities. The four observed fuzzy, misspelled, and appended-argument names
# are undefined attempt evidence and do not inherit audited-family membership.
_METACOGNITIVE_MONITORING_TOOL_NAMES = frozenset(
    {
        "metacognitiveMonitoring",
        "metacognitive-monitoring-server-metacognitiveMonitoring",
        "metacognitivemonitoring",
        "clear-thought-server-metacognitivemonitoring",
        "clear-thought-metacognitivemonitoring",
    }
)

# Programming Paradigm is an explicit Clear Thought operation. The
# Chirag/ThinkFar request handler validates truthy string paradigm/problem
# fields, normalizes optional arrays, and returns deterministic current-call
# status metadata. The complete snapshot contains 1,591 definition-backed calls
# across the qualified and bare aliases, plus one separate undefined bare call.
# Preservation keeps successful, malformed, and undefined evidence visible to
# the ordinary validators.
_PROGRAMMING_PARADIGM_TOOL_NAMES = frozenset(
    {
        "clear-thought-server-programmingparadigm",
        "programmingparadigm",
    }
)

# Chain of Draft is a stateful MCP protocol. The server retains per-session
# thought history and branches; results expose cumulative branch identifiers
# and history length that are not a function of the current call alone. The
# pinned snapshot has one exact qualified identity and nine rows. Preserve its
# successful, schema-error, and argument-decode episodes intact.
_CHAIN_OF_DRAFT_TOOL_NAMES = frozenset({"chain-of-draft-server-chain-of-draft"})

# These three compact Waldzell Clear Thought methods are definition-only in the
# pinned snapshot, but their captured implementations are genuine stateful
# operations: they mutate session stores and can return hidden cumulative
# context. Exact classification prevents the legacy broad name predicate from
# deleting a future call. Definitions alone do not create action markers.
_CREATIVE_THINKING_TOOL_NAMES = frozenset(
    {"clear-thought-creativethinking", "creativethinking"}
)

_SYSTEMS_THINKING_TOOL_NAMES = frozenset(
    {"clear-thought-systemsthinking", "systemsthinking"}
)

_SOCRATIC_METHOD_TOOL_NAMES = frozenset(
    {"clear-thought-socraticmethod", "socraticmethod"}
)

# Clear Thought 1.5 consolidates reasoning, state, research, code execution,
# notebook, and orchestration operations behind one explicit dispatcher. Its
# observations can contain server-generated IDs, cumulative session state,
# stored notebook state, code output, and auto-seeded sequential-thinking
# results. The two names are the exact bare and server-shuffled identities
# declared in Toucan's model-visible system context. Because one dispatcher
# mixes reasoning and ordinary state/I/O operations, it is protected as a
# whole. An actual call receives a dispatcher marker; mere availability does
# not.
_CLEAR_THOUGHT_DISPATCHER_TOOL_NAMES = frozenset(
    {"clear-thought-clear_thought", "clear_thought"}
)

_CLEAR_THOUGHT_DISPATCHER_OPERATION_FAMILIES = {
    "sequential_thinking": SEQUENTIAL_THINKING_TOOL_FAMILY,
    "mental_model": MENTAL_MODEL_TOOL_FAMILY,
    "debugging_approach": DEBUGGING_APPROACH_TOOL_FAMILY,
    "creative_thinking": CREATIVE_THINKING_TOOL_FAMILY,
    "visual_reasoning": VISUAL_REASONING_TOOL_FAMILY,
    "metacognitive_monitoring": METACOGNITIVE_MONITORING_TOOL_FAMILY,
    "scientific_method": SCIENTIFIC_METHOD_TOOL_FAMILY,
    "collaborative_reasoning": COLLABORATIVE_REASONING_TOOL_FAMILY,
    "decision_framework": DECISION_FRAMEWORK_TOOL_FAMILY,
    "socratic_method": SOCRATIC_METHOD_TOOL_FAMILY,
    "structured_argumentation": STRUCTURED_ARGUMENTATION_TOOL_FAMILY,
    "systems_thinking": SYSTEMS_THINKING_TOOL_FAMILY,
    "analogical_reasoning": ANALOGICAL_REASONING_TOOL_FAMILY,
}

# Clear Thought session operations are ordinary state tools, not reasoning
# traces. session_info/export read hidden server state; session_import mutates
# it and can clear existing data first. They are exact non-strippable identities
# but intentionally do not belong to a reasoning action-marker family.
_CLEAR_THOUGHT_SESSION_TOOL_NAMES = frozenset(
    {
        "clear-thought-session_export",
        "session_export",
        "clear-thought-session_import",
        "session_import",
        "clear-thought-session_info",
        "session_info",
    }
)

# These two exact names are MCP resource primitives exposed by the unified
# Clear Thought server. They are caught by both the broad ``clear-thought``
# reasoning substring and the ordinary resource-scaffold classifier. The full
# snapshot audit found 66 teacher-visible, positionally paired calls whose
# results can contain server resources not present in the model's arguments.
# Exempt them from reasoning stripping; the separately configured scaffold
# transform may still remove their complete call/result episodes.
_CLEAR_THOUGHT_RESOURCE_TOOL_NAMES = frozenset(
    {
        "clear-thought-list_resources",
        "clear-thought-read_resource",
    }
)

_PRESERVED_REASONING_TOOL_NAMES = (
    _THINK_TOOL_NAMES
    | _SEQUENTIAL_THINKING_TOOL_NAMES
    | _PENTEST_THINKING_TOOL_NAMES
    | _GAME_DESIGN_THINKING_TOOL_NAMES
    | _SKIA_ANIMATION_THINKING_TOOL_NAMES
    | _LOTUS_WISDOM_TOOL_NAMES
    | _STRUCTURED_ARGUMENTATION_TOOL_NAMES
    | _ANALOGICAL_REASONING_TOOL_NAMES
    | _MENTAL_MODEL_TOOL_NAMES
    | _DECISION_FRAMEWORK_TOOL_NAMES
    | _SCIENTIFIC_METHOD_TOOL_NAMES
    | _DEBUGGING_APPROACH_TOOL_NAMES
    | _DESIGN_PATTERN_TOOL_NAMES
    | _COLLABORATIVE_REASONING_TOOL_NAMES
    | _VISUAL_REASONING_TOOL_NAMES
    | _METACOGNITIVE_MONITORING_TOOL_NAMES
    | _PROGRAMMING_PARADIGM_TOOL_NAMES
    | _CHAIN_OF_DRAFT_TOOL_NAMES
    | _CREATIVE_THINKING_TOOL_NAMES
    | _SYSTEMS_THINKING_TOOL_NAMES
    | _SOCRATIC_METHOD_TOOL_NAMES
    | _CLEAR_THOUGHT_DISPATCHER_TOOL_NAMES
)

_LEGACY_REASONING_EXCLUSIONS = (
    _PRESERVED_REASONING_TOOL_NAMES
    | _CLEAR_THOUGHT_SESSION_TOOL_NAMES
    | _CLEAR_THOUGHT_RESOURCE_TOOL_NAMES
)


def _reasoning_tool_families(messages: list[dict[str, Any]]) -> list[str]:
    """Return exact audited reasoning-tool families actually called in a row.

    Definitions alone do not mark a sample: an unused tool does not teach an
    action. The result is stable and ordered for deterministic serialization.
    """

    called_names: set[str] = set()
    dispatcher_operations: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if isinstance(name, str):
                called_names.add(name)
            if name not in _CLEAR_THOUGHT_DISPATCHER_TOOL_NAMES:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = orjson.loads(arguments)
                except orjson.JSONDecodeError:
                    continue
            if isinstance(arguments, dict):
                operation = arguments.get("operation")
                if isinstance(operation, str):
                    dispatcher_operations.add(operation)

    dispatched_families = {
        family
        for operation in dispatcher_operations
        if (family := _CLEAR_THOUGHT_DISPATCHER_OPERATION_FAMILIES.get(operation))
    }

    def called_or_dispatched(names: frozenset[str], family: str) -> bool:
        return bool(called_names & names or family in dispatched_families)

    families = []
    if called_names & _THINK_TOOL_NAMES:
        families.append(THINK_TOOL_FAMILY)
    if called_or_dispatched(
        _SEQUENTIAL_THINKING_TOOL_NAMES, SEQUENTIAL_THINKING_TOOL_FAMILY
    ):
        families.append(SEQUENTIAL_THINKING_TOOL_FAMILY)
    if called_names & _PENTEST_THINKING_TOOL_NAMES:
        families.append(PENTEST_THINKING_TOOL_FAMILY)
    if called_names & _GAME_DESIGN_THINKING_TOOL_NAMES:
        families.append(GAME_DESIGN_THINKING_TOOL_FAMILY)
    if called_names & _SKIA_ANIMATION_THINKING_TOOL_NAMES:
        families.append(SKIA_ANIMATION_THINKING_TOOL_FAMILY)
    if called_names & _LOTUS_WISDOM_TOOL_NAMES:
        families.append(LOTUS_WISDOM_TOOL_FAMILY)
    if called_or_dispatched(
        _STRUCTURED_ARGUMENTATION_TOOL_NAMES,
        STRUCTURED_ARGUMENTATION_TOOL_FAMILY,
    ):
        families.append(STRUCTURED_ARGUMENTATION_TOOL_FAMILY)
    if called_or_dispatched(
        _ANALOGICAL_REASONING_TOOL_NAMES, ANALOGICAL_REASONING_TOOL_FAMILY
    ):
        families.append(ANALOGICAL_REASONING_TOOL_FAMILY)
    if called_or_dispatched(_MENTAL_MODEL_TOOL_NAMES, MENTAL_MODEL_TOOL_FAMILY):
        families.append(MENTAL_MODEL_TOOL_FAMILY)
    if called_or_dispatched(
        _DECISION_FRAMEWORK_TOOL_NAMES, DECISION_FRAMEWORK_TOOL_FAMILY
    ):
        families.append(DECISION_FRAMEWORK_TOOL_FAMILY)
    if called_or_dispatched(
        _SCIENTIFIC_METHOD_TOOL_NAMES, SCIENTIFIC_METHOD_TOOL_FAMILY
    ):
        families.append(SCIENTIFIC_METHOD_TOOL_FAMILY)
    if called_or_dispatched(
        _DEBUGGING_APPROACH_TOOL_NAMES, DEBUGGING_APPROACH_TOOL_FAMILY
    ):
        families.append(DEBUGGING_APPROACH_TOOL_FAMILY)
    if called_names & _DESIGN_PATTERN_TOOL_NAMES:
        families.append(DESIGN_PATTERN_TOOL_FAMILY)
    if called_or_dispatched(
        _COLLABORATIVE_REASONING_TOOL_NAMES,
        COLLABORATIVE_REASONING_TOOL_FAMILY,
    ):
        families.append(COLLABORATIVE_REASONING_TOOL_FAMILY)
    if called_or_dispatched(_VISUAL_REASONING_TOOL_NAMES, VISUAL_REASONING_TOOL_FAMILY):
        families.append(VISUAL_REASONING_TOOL_FAMILY)
    if called_or_dispatched(
        _METACOGNITIVE_MONITORING_TOOL_NAMES,
        METACOGNITIVE_MONITORING_TOOL_FAMILY,
    ):
        families.append(METACOGNITIVE_MONITORING_TOOL_FAMILY)
    if called_names & _PROGRAMMING_PARADIGM_TOOL_NAMES:
        families.append(PROGRAMMING_PARADIGM_TOOL_FAMILY)
    if called_names & _CHAIN_OF_DRAFT_TOOL_NAMES:
        families.append(CHAIN_OF_DRAFT_TOOL_FAMILY)
    if called_or_dispatched(
        _CREATIVE_THINKING_TOOL_NAMES, CREATIVE_THINKING_TOOL_FAMILY
    ):
        families.append(CREATIVE_THINKING_TOOL_FAMILY)
    if called_or_dispatched(_SYSTEMS_THINKING_TOOL_NAMES, SYSTEMS_THINKING_TOOL_FAMILY):
        families.append(SYSTEMS_THINKING_TOOL_FAMILY)
    if called_or_dispatched(_SOCRATIC_METHOD_TOOL_NAMES, SOCRATIC_METHOD_TOOL_FAMILY):
        families.append(SOCRATIC_METHOD_TOOL_FAMILY)
    if called_names & _CLEAR_THOUGHT_DISPATCHER_TOOL_NAMES:
        families.append(CLEAR_THOUGHT_DISPATCHER_TOOL_FAMILY)
    return families


# Non-reasoning framework SCAFFOLD/plumbing: generic MCP protocol machinery and
# the degenerate async poller. Real (often information-bearing) tool use, not
# reasoning -- KEPT by default; strippable via strip_scaffold_tools.
_SCAFFOLD_TOOL_SUBSTRINGS = (
    "__unlock",
    "__get_instructions",
    "list_resources",
    "read_resource",
    "get_resource",
    "deep_researcher",  # exa-search async poller; in Toucan it rarely completes
)


def _legacy_reasoning_candidate(name: str) -> bool:
    """Return whether the retired broad heuristic would target this name.

    This is an audit/census helper only. Production never deletes a tool call
    because this predicate returns true.
    """

    if name in _LEGACY_REASONING_EXCLUSIONS:
        return False
    low = name.lower()
    return (
        low.endswith("-think")
        or low == "think"
        or any(s in low for s in _LEGACY_REASONING_SUBSTRINGS)
    )


def _is_scaffold_tool(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in _SCAFFOLD_TOOL_SUBSTRINGS)


_KIMI_TOOL_TEMPLATE_PREFIX = "<|im_system|>tool_declare<|im_middle|>"
_KIMI_TOOL_TEMPLATE_SUFFIX = "<|im_end|>"

_XML_TOOL_TEMPLATE_PREFIX = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)
_XML_TOOL_TEMPLATE_SUFFIX = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def _valid_embedded_tool_definition(tool: Any) -> bool:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return False
    function = tool.get("function")
    return (
        isinstance(function, dict)
        and isinstance(function.get("name"), str)
        and bool(function["name"])
        and isinstance(function.get("parameters"), dict)
    )


def _parse_kimi_tool_body(body: str) -> list[dict[str, Any]] | None:
    try:
        tools = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(tools, list) or not all(
        _valid_embedded_tool_definition(tool) for tool in tools
    ):
        return None
    return tools


def _parse_xml_tool_body(body: str) -> list[dict[str, Any]] | None:
    """Parse the observed one-JSON-object-per-line XML template body."""

    lines = [line for line in body.splitlines() if line.strip()]
    try:
        tools = [orjson.loads(line) for line in lines]
    except orjson.JSONDecodeError:
        return None
    if not all(_valid_embedded_tool_definition(tool) for tool in tools):
        return None
    return tools


def _find_valid_template(
    content: str,
    start: int,
    prefix: str,
    suffix: str,
    body_parser: Any,
) -> tuple[int, list[dict[str, Any]]] | None:
    """Return the end and tools of a verified template at ``start``."""

    body_start = start + len(prefix)
    suffix_start = content.find(suffix, body_start)
    while suffix_start >= 0:
        tools = body_parser(content[body_start:suffix_start])
        if tools is not None:
            return suffix_start + len(suffix), tools
        suffix_start = content.find(suffix, suffix_start + 1)
    return None


def _extract_embedded_tool_system_content(
    content: str,
) -> tuple[str, list[dict[str, Any]], bool, int]:
    """Extract verified tool-template spans and preserve surrounding text.

    Both framework templates may occur at the beginning, middle, or end of a
    future system message. A candidate is extracted only when its exact fixed
    framing and serialized tool body validate. Malformed and unfamiliar
    candidates remain byte-for-byte rather than being heuristically truncated.
    Returns ``(remaining_content, extracted_tools, removed, invalid_candidates)``.
    """

    templates = (
        (
            _KIMI_TOOL_TEMPLATE_PREFIX,
            _KIMI_TOOL_TEMPLATE_SUFFIX,
            _parse_kimi_tool_body,
        ),
        (_XML_TOOL_TEMPLATE_PREFIX, _XML_TOOL_TEMPLATE_SUFFIX, _parse_xml_tool_body),
    )
    kept: list[str] = []
    extracted_tools: list[dict[str, Any]] = []
    cursor = 0
    removed = False
    invalid_candidates = 0

    while cursor < len(content):
        candidates = [
            (position, prefix, suffix, parser)
            for prefix, suffix, parser in templates
            if (position := content.find(prefix, cursor)) >= 0
        ]
        if not candidates:
            kept.append(content[cursor:])
            break

        start, prefix, suffix, parser = min(candidates, key=lambda item: item[0])
        match = _find_valid_template(content, start, prefix, suffix, parser)
        if match is None:
            # Keep the unmatched prefix and continue looking after it. This
            # preserves the content exactly while still allowing a later valid
            # template in the same message to be normalized.
            invalid_candidates += 1
            prefix_end = start + len(prefix)
            kept.append(content[cursor:prefix_end])
            cursor = prefix_end
            continue

        end, tools = match
        kept.append(content[cursor:start])
        extracted_tools.extend(tools)
        cursor = end
        removed = True
    else:
        # The final valid template ended exactly at the end of the message.
        pass

    if not removed:
        return content, [], False, invalid_candidates
    remaining = "".join(kept)
    if not remaining.strip():
        remaining = ""
    return (
        remaining,
        extracted_tools,
        True,
        invalid_candidates,
    )


def _strip_embedded_tool_system_content(content: str) -> tuple[str, bool]:
    """Compatibility wrapper returning only retained text and removal status."""

    remaining, _tools, removed, _invalid = _extract_embedded_tool_system_content(
        content
    )
    return remaining, removed


def _is_embedded_tool_system_message(content: str) -> bool:
    """Return whether ``content`` consists only of verified tool templates."""

    remaining, removed = _strip_embedded_tool_system_content(content)
    return removed and not remaining


# ---------------------------------------------------------------------------
# Stage 1 conversion
# ---------------------------------------------------------------------------


def _convert_messages_with_tool_context(
    raw_msgs: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    list[dict[str, Any]],
    bool,
]:
    """Collapse messages and extract the exact model-visible tool context."""

    out: list[dict[str, Any]] = []
    issues: dict[str, int] = {}
    system_tools: list[dict[str, Any]] = []
    has_valid_tool_template = False
    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    calls: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal content_buf, reasoning_buf, calls
        if not (content_buf or reasoning_buf or calls):
            return
        msg: dict[str, Any] = {"role": "assistant"}
        text = "\n\n".join(p for p in content_buf if p).strip()
        msg["content"] = text or None
        reasoning = "\n\n".join(p for p in reasoning_buf if p).strip()
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        out.append(msg)
        content_buf, reasoning_buf, calls = [], [], []

    for m in raw_msgs:
        role = m.get("role")
        if role == "assistant":
            c = m.get("content")
            if c:
                content_buf.append(c)
            rc = m.get("reasoning_content")
            if rc:
                reasoning_buf.append(rc)
            fc = m.get("function_call")
            if fc is not None:
                calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": fc.get("name"),
                            "arguments": fc.get("arguments", ""),
                        },
                    }
                )
        elif role == "function":
            flush()
            out.append({"role": "tool", "content": m.get("content") or ""})
        elif role == "system":
            flush()
            original_content = m.get("content") or ""
            (
                normalized_content,
                extracted_tools,
                removed,
                invalid_candidates,
            ) = _extract_embedded_tool_system_content(original_content)
            if invalid_candidates:
                issues["malformed_embedded_tool_template"] = 1
            if removed:
                has_valid_tool_template = True
                system_tools.extend(extracted_tools)
            # The extracted definitions are the contract the teacher model
            # actually saw. The target chat template will render that same
            # contract from sample.tools. Preserve every independent system
            # instruction surrounding a verified template span.
            if normalized_content or not removed:
                out.append({"role": role, "content": normalized_content})
        else:  # user or any unexpected role: preserve
            flush()
            out.append({"role": role, "content": m.get("content") or ""})
    flush()
    return out, issues, system_tools, has_valid_tool_template


def _convert_messages(
    raw_msgs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Compatibility wrapper for message-only conversion callers."""

    messages, issues, _system_tools, _has_template = (
        _convert_messages_with_tool_context(raw_msgs)
    )
    return messages, issues


def _stage1_issues(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Structural Stage 1 issues on the converted messages (per-row flags)."""
    issues: dict[str, int] = {}
    if messages and messages[-1]["role"] == "tool":
        issues["ends_on_tool_response"] = 1
    if (
        messages
        and messages[-1]["role"] == "assistant"
        and not messages[-1].get("content")
        and not messages[-1].get("tool_calls")
    ):
        issues["ends_on_empty_assistant"] = 1
    # cardinality + unparseable args
    i = 0
    n = len(messages)
    unbalanced = False
    unparseable = False
    while i < n:
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            ncalls = len(m["tool_calls"])
            for tc in m["tool_calls"]:
                args = tc["function"].get("arguments", "")
                if isinstance(args, str) and args.strip():
                    try:
                        orjson.loads(args)
                    except orjson.JSONDecodeError:
                        unparseable = True
            j = i + 1
            nresp = 0
            while j < n and messages[j]["role"] == "tool":
                nresp += 1
                j += 1
            if ncalls != nresp:
                unbalanced = True
            i = j
        else:
            i += 1
    if unbalanced:
        issues["unbalanced_call_response"] = 1
    if unparseable:
        issues["unparseable_tool_call_arguments"] = 1
    return issues


def _convert_sample(
    row: dict[str, Any], cfg_name: str
) -> tuple[ConversationSample, dict[str, int]]:
    raw_msgs = orjson.loads(row["messages"])
    try:
        available_tools = (
            orjson.loads(row["available_tools"]) if row.get("available_tools") else []
        )
    except orjson.JSONDecodeError:
        available_tools = []
    messages, issues, system_tools, has_tool_template = (
        _convert_messages_with_tool_context(raw_msgs)
    )
    if has_tool_template:
        tools = system_tools
        available_fingerprints = sorted(
            orjson.dumps(tool, option=orjson.OPT_SORT_KEYS) for tool in available_tools
        )
        system_fingerprints = sorted(
            orjson.dumps(tool, option=orjson.OPT_SORT_KEYS) for tool in system_tools
        )
        if available_fingerprints != system_fingerprints:
            issues["system_available_tool_context_mismatch"] = 1
    else:
        tools = available_tools
    # This remains a source-data census flag.  Normalized output intentionally
    # omits the framework tool-system templates, so checking converted messages
    # would incorrectly flag every row.
    if not any(m.get("role") == "system" for m in raw_msgs):
        issues["no_system_message"] = 1
    issues.update(_stage1_issues(messages))
    families = _reasoning_tool_families(messages)
    annotations = {REASONING_TOOL_FAMILIES_ANNOTATION: families} if families else {}
    sample = ConversationSample(
        messages=messages,
        tools=tools,
        dataset=f"{DATASET_ID}:{cfg_name}",
        sample_id=row.get("uuid") or "",
        annotations=annotations,
        raw=row,
    )
    return sample, issues


# ---------------------------------------------------------------------------
# Stage 2: dataset-specific config
# ---------------------------------------------------------------------------


def _passes_quality(row: dict[str, Any], config: ToucanConfig) -> bool:
    """Quality gate using the assessment columns. Irrelevant rows are gated on
    the question side only (they carry no response assessment)."""
    try:
        q = orjson.loads(row["question_quality_assessment"])
    except orjson.JSONDecodeError, TypeError:
        return False
    qq = (q.get("question_quality") or {}).get("score")
    sr = (q.get("scenario_realism") or {}).get("score")
    if not (isinstance(qq, int) and qq >= config.min_question_quality):
        return False
    if not (isinstance(sr, int) and sr >= config.min_scenario_realism):
        return False
    if row.get("subset_name") == "irrelevant":
        return True
    ra_raw = row.get("response_quality_assessment") or ""
    try:
        r = orjson.loads(ra_raw)
    except orjson.JSONDecodeError:
        return False
    comp = (r.get("completeness") or {}).get("score")
    conc = (r.get("conciseness") or {}).get("score")
    dt = r.get("desired_tools_used_percentage")
    if not (isinstance(comp, int) and comp >= config.min_completeness):
        return False
    if not (isinstance(conc, int) and conc >= config.min_conciseness):
        return False
    if config.require_full_tool_use and not (
        isinstance(dt, (int, float)) and dt == 1.0
    ):
        return False
    return True


def _ends_incomplete(sample: ConversationSample) -> bool:
    msgs = sample.messages
    if not msgs:
        return True
    last = msgs[-1]
    if last["role"] == "tool":
        return True
    if (
        last["role"] == "assistant"
        and not last.get("content")
        and not last.get("tool_calls")
    ):
        return True
    return False


def _conflicting_duplicate_tool_names(tools: Any) -> set[str]:
    """Exact visible names mapped to multiple complete definitions in one row.

    Canonical JSON serialization removes only irrelevant object-key ordering.
    Every actual field and every array order remains part of the definition; no
    schema equivalence, description normalization, case folding, or cross-row
    comparison is attempted.
    """

    if not isinstance(tools, list):
        return set()
    by_name: dict[str, set[bytes]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        by_name.setdefault(name, set()).add(
            orjson.dumps(tool, option=orjson.OPT_SORT_KEYS)
        )
    return {name for name, definitions in by_name.items() if len(definitions) > 1}


def _has_conflicting_duplicate_tools(value: Any) -> bool:
    """Whether the canonical model-visible context conflicts by exact name.

    Production passes a ``ConversationSample`` so this check uses the same
    authoritative definitions rendered during training. Raw-row support remains
    for focused backward-compatible unit tests and reads ``available_tools``.
    """

    if isinstance(value, ConversationSample):
        tools = value.tools
    elif isinstance(value, dict):
        try:
            tools = orjson.loads(value.get("available_tools") or "[]")
        except orjson.JSONDecodeError:
            return False
    else:
        tools = value
    return bool(_conflicting_duplicate_tool_names(tools))


def _legacy_tool_projection(
    sample: ConversationSample, strip_reasoning: bool, strip_scaffold: bool
) -> tuple[bool, bool, bool]:
    """Reproduce retired reasoning and optional scaffold projections for audit.

    Production calls this helper only for the separately registered scaffold
    counterfactual, always with ``strip_reasoning=False``. The reasoning branch
    remains available solely to reproduce historical artifacts and tests; no
    ``ToucanConfig`` field can enable it.

    A definition is pruned only when at least one exact-name call was removed
    and no exact-name call remains in the source trajectory. Definitions that
    were merely available but never called are always preserved.

    Only strips an assistant turn whose tool_calls are balanced with the
    following run of tool messages. Undefined calls and every call/definition
    under a conflicting or unbalanced visible name remain untouched. This
    prevents the counterfactual from hiding the very defect that a downstream
    validator must see.
    """

    defined_names = {
        (tool.get("function") or {}).get("name")
        for tool in sample.tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    conflicting_names = _conflicting_duplicate_tool_names(sample.tools)

    # If any episode for a name is structurally unbalanced, preserve all events
    # and definitions under that name. Partial deletion would make the remaining
    # row even less faithful and could manufacture an undefined call.
    unbalanced_names: set[str] = set()
    i = 0
    while i < len(sample.messages):
        message = sample.messages[i]
        calls = (
            message.get("tool_calls") if message.get("role") == "assistant" else None
        )
        if not calls:
            i += 1
            continue
        j = i + 1
        while j < len(sample.messages) and sample.messages[j].get("role") == "tool":
            j += 1
        if len(calls) != j - i - 1:
            unbalanced_names.update(
                (call.get("function") or {}).get("name") or "" for call in calls
            )
        i = j

    def classify(name: str, *, is_call: bool = False) -> tuple[bool, bool]:
        # Returns (should_strip, is_reasoning).
        if name in conflicting_names or name in unbalanced_names:
            return False, False
        if is_call and name not in defined_names:
            return False, False
        if strip_reasoning and _legacy_reasoning_candidate(name):
            return True, True
        if strip_scaffold and _is_scaffold_tool(name):
            return True, False
        return False, False

    msgs = sample.messages
    changed = False
    removed_reasoning = False
    removed_scaffold = False
    stripped_call_names: set[str] = set()
    retained_call_names: set[str] = set()
    out: list[dict[str, Any]] = []
    i = 0
    n = len(msgs)
    while i < n:
        m = msgs[i]
        calls = m.get("tool_calls") if m["role"] == "assistant" else None
        if not calls:
            out.append(m)
            i += 1
            continue
        # collect the following run of tool messages
        j = i + 1
        resp = []
        while j < n and msgs[j]["role"] == "tool":
            resp.append(msgs[j])
            j += 1
        if len(resp) != len(calls):
            out.append(m)  # unbalanced: leave untouched
            retained_call_names.update(
                call.get("function", {}).get("name") or "" for call in calls
            )
            i += 1
            continue
        keep_calls = []
        keep_resp = []
        for call, r in zip(calls, resp, strict=True):
            name = call.get("function", {}).get("name") or ""
            strip, is_reasoning = classify(name, is_call=True)
            if strip:
                changed = True
                stripped_call_names.add(name)
                if is_reasoning:
                    removed_reasoning = True
                else:
                    removed_scaffold = True
            else:
                retained_call_names.add(name)
                keep_calls.append(call)
                keep_resp.append(r)
        if keep_calls:
            nm = dict(m)
            nm["tool_calls"] = keep_calls
            out.append(nm)
            out.extend(keep_resp)
        elif (m.get("content") or "").strip():
            nm = {k: v for k, v in m.items() if k != "tool_calls"}
            out.append(nm)
            out.extend(keep_resp)  # empty
        # else: assistant turn was only stripped calls with no content -> drop it
        i = j
    # A definition is destructive context, not an event. Prune it only when the
    # source trajectory contained an exact-name call, every such call was
    # deliberately stripped, and none was retained. An uncalled definition is
    # therefore always preserved, regardless of broad name classification.
    pruned_tools = []
    for t in sample.tools:
        name = (t.get("function") or {}).get("name") or ""
        if name not in stripped_call_names or name in retained_call_names:
            pruned_tools.append(t)
    tools_changed = len(pruned_tools) != len(sample.tools)
    if changed:
        sample.messages = out
    if changed or tools_changed:
        sample.tools = pruned_tools
    return changed or tools_changed, removed_reasoning, removed_scaffold


def _apply_dataset_config(
    samples: list[ConversationSample], config: ToucanConfig
) -> tuple[list[ConversationSample], dict[str, int], dict[str, int]]:
    drops: dict[str, int] = {}
    transforms: dict[str, int] = {}
    keep_subsets = set(config.subsets)

    kept: list[ConversationSample] = []
    d_subset = d_quality = d_term = d_confdup = 0
    for s in samples:
        drop = False
        if s.raw.get("subset_name") not in keep_subsets:
            d_subset += 1
            drop = True
        if config.drop_low_quality and not _passes_quality(s.raw, config):
            d_quality += 1
            drop = True
        if config.drop_incomplete_termination and _ends_incomplete(s):
            d_term += 1
            drop = True
        if config.drop_conflicting_duplicate_tools and _has_conflicting_duplicate_tools(
            s
        ):
            d_confdup += 1
            drop = True
        if not drop:
            kept.append(s)
    if d_subset:
        drops["subset_not_selected"] = d_subset
    if d_quality:
        drops["low_quality"] = d_quality
    if d_confdup:
        drops["conflicting_duplicate_tools"] = d_confdup
    # transform on survivors, after drops (loader-checklist ordering)
    if config.strip_scaffold_tools:
        n_scaffold = 0
        for s in kept:
            _, _, removed_scaffold = _legacy_tool_projection(
                s, strip_reasoning=False, strip_scaffold=True
            )
            n_scaffold += removed_scaffold
        if n_scaffold:
            transforms["stripped_scaffold_tools"] = n_scaffold
    if d_term:
        drops["incomplete_termination"] = d_term
    return kept, drops, transforms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_shards(configs: tuple[str, ...], path: str | Path | None) -> list[Path]:
    if path is not None:
        root = Path(path)
        shards: list[Path] = []
        for c in configs:
            shards.extend(sorted((root / c).glob("*.parquet")))
        if not shards:
            raise FileNotFoundError(f"No parquet shards under {root} for {configs}")
        return shards
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"{c}/*.parquet" for c in configs],
    )
    root = Path(local)
    return [p for c in configs for p in sorted((root / c).glob("*.parquet"))]


def load(
    dataset_config: ToucanConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    path: str | Path | None = None,
    configs: tuple[str, ...] = TEACHER_CONFIGS,
) -> tuple[list[ConversationSample], LoadReport]:
    """Load Toucan-1.5M teacher configs and convert all rows to OpenAI format.

    ``configs`` selects which teacher configs to load (default: all three).
    ``path`` is the snapshot root (``.../Toucan-1.5M`` with ``Kimi-K2/`` etc.
    subdirectories); None downloads the pinned revision from HuggingFace.
    The ``SFT`` config is not supported (it is a Kimi-K2 derivative; reproduce it
    by loading Kimi-K2 with ``drop_low_quality=True``).
    """
    bad = set(configs) - set(TEACHER_CONFIGS)
    if bad:
        raise ValueError(
            f"Unsupported config(s) {sorted(bad)}; this loader handles the teacher "
            f"configs {TEACHER_CONFIGS}. SFT is a Kimi-K2 derivative (see docstring)."
        )
    shards = _resolve_shards(configs, path)

    samples: list[ConversationSample] = []
    issue_counts: dict[str, int] = {}
    raw_count = 0

    gc_was_enabled = gc.isenabled()
    gc.disable()
    for shard in shards:
        cfg_name = shard.parent.name
        table = pq.read_table(shard)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        n = len(cols["uuid"])
        raw_count += n
        for idx in range(n):
            row = {name: cols[name][idx] for name in cols}
            sample, issues = _convert_sample(row, cfg_name)
            samples.append(sample)
            for reason, count in issues.items():
                issue_counts[reason] = issue_counts.get(reason, 0) + count
    if gc_was_enabled:
        gc.enable()
    gc.collect()

    report = LoadReport(
        dataset=f"{DATASET_ID} [{'+'.join(configs)}]",
        raw_count=raw_count,
        stage1_count=len(samples),
        stage1_issue_counts=issue_counts,
    )

    if dataset_config is not None:
        samples, ds_drops, ds_transforms = _apply_dataset_config(
            samples, dataset_config
        )
        report.dataset_config_count = len(samples)
        report.dataset_config_drop_reasons = ds_drops
        report.dataset_config_transform_counts = ds_transforms

    if filter_config is not None:
        samples, drop_reasons = apply_filters(samples, filter_config)
        report.filtered_count = len(samples)
        report.filter_config = filter_config
        report.filter_drop_reasons = drop_reasons

    return samples, report
