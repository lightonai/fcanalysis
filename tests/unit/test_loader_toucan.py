"""Unit tests for toucan loader-specific conversion, predicates, and transforms.

Synthetic inputs only (no dataset access). The full-dataset behavioral
contract is covered by the e2e fixtures; these tests pin the per-function
semantics that the conversion and Stage 2 config depend on.
"""

import copy

import orjson
import pytest

from fcanalysis.format import ConversationSample
from fcanalysis.validation import has_invalid_arguments, has_undefined_function_calls
from fcanalysis.loaders.toucan import (
    ANALOGICAL_REASONING_TOOL_FAMILY,
    COLLABORATIVE_REASONING_TOOL_FAMILY,
    DECISION_FRAMEWORK_TOOL_FAMILY,
    DEBUGGING_APPROACH_TOOL_FAMILY,
    DESIGN_PATTERN_TOOL_FAMILY,
    GAME_DESIGN_THINKING_TOOL_FAMILY,
    LOTUS_WISDOM_TOOL_FAMILY,
    MENTAL_MODEL_TOOL_FAMILY,
    PENTEST_THINKING_TOOL_FAMILY,
    REASONING_TOOL_FAMILIES_ANNOTATION,
    SCIENTIFIC_METHOD_TOOL_FAMILY,
    SEQUENTIAL_THINKING_TOOL_FAMILY,
    SKIA_ANIMATION_THINKING_TOOL_FAMILY,
    STRUCTURED_ARGUMENTATION_TOOL_FAMILY,
    THINK_TOOL_FAMILY,
    VISUAL_REASONING_TOOL_FAMILY,
    ToucanConfig,
    _apply_dataset_config,
    _ANALOGICAL_REASONING_TOOL_NAMES,
    _COLLABORATIVE_REASONING_TOOL_NAMES,
    _DECISION_FRAMEWORK_TOOL_NAMES,
    _DEBUGGING_APPROACH_TOOL_NAMES,
    _DESIGN_PATTERN_TOOL_NAMES,
    _convert_messages,
    _convert_sample,
    _ends_incomplete,
    _has_conflicting_duplicate_tools,
    _is_embedded_tool_system_message,
    _is_reasoning_tool,
    _is_scaffold_tool,
    _GAME_DESIGN_THINKING_TOOL_NAMES,
    _LOTUS_WISDOM_TOOL_NAMES,
    _MENTAL_MODEL_TOOL_NAMES,
    _PENTEST_THINKING_TOOL_NAMES,
    _PRESERVED_REASONING_TOOL_NAMES,
    _SCIENTIFIC_METHOD_TOOL_NAMES,
    _SEQUENTIAL_THINKING_TOOL_NAMES,
    _SKIA_ANIMATION_THINKING_TOOL_NAMES,
    _STRUCTURED_ARGUMENTATION_TOOL_NAMES,
    _strip_embedded_tool_system_content,
    _passes_quality,
    _THINK_TOOL_NAMES,
    _VISUAL_REASONING_TOOL_NAMES,
    _stage1_issues,
    _strip_tools,
    load,
)

from tests.helpers import assistant, call, func, system, tool_response, user


EXPECTED_THINK_TOOL_NAMES = (
    "think",
    "get_thoughts",
    "get_thought_stats",
    "clear_thoughts",
    "think-tool-think",
    "think-tool-get_thoughts",
    "think-tool-get_thought_stats",
    "think-tool-clear_thoughts",
    "think-tool-server-think",
    "think-tool-server-get_thoughts",
    "think-tool-server-get_thought_stats",
    "think-tool-server-clear_thoughts",
)

EXPECTED_SEQUENTIAL_THINKING_TOOL_NAMES = (
    "clear-thought-sequentialthinking",
    "clear-thought-server-sequentialthinking",
    "model-context-protocol-server-sequentialthinking",
    "reference-servers-sequentialthinking",
    "sequential-thinking-sequentialthinking",
    "sequential-thinking-tools-sequentialthinking_tools",
    "sequentialthinking",
    "sequentialthinking_tools",
)

EXPECTED_PENTEST_THINKING_TOOL_NAMES = (
    "pentestthinkingMCP",
    "pentestthinking-pentestthinkingMCP",
)

EXPECTED_GAME_DESIGN_THINKING_TOOL_NAMES = (
    "gamedesignthinking",
    "game-engine-server-gamedesignthinking",
)

EXPECTED_SKIA_ANIMATION_THINKING_TOOL_NAMES = (
    "skiaanimationthinking",
    "react-native-skia-animation-thinking-tool-skiaanimationthinking",
)

EXPECTED_LOTUS_WISDOM_TOOL_NAMES = (
    "lotuswisdom",
    "lotuswisdom_summary",
    "lotus-wisdom-lotuswisdom",
    "lotus-wisdom-lotuswisdom_summary",
)

EXPECTED_STRUCTURED_ARGUMENTATION_TOOL_NAMES = (
    "structured-argumentation-server-structuredArgumentation",
    "structuredArgumentation",
    "clear-thought-server-structuredargumentation",
    "structuredargumentation",
    "clear-thought-structuredargumentation",
)

EXPECTED_ANALOGICAL_REASONING_TOOL_NAMES = (
    "analogical-reasoning-server-analogicalReasoning",
    "analogicalReasoning",
)

EXPECTED_MENTAL_MODEL_TOOL_NAMES = (
    "mentalmodel",
    "clear-thought-server-mentalmodel",
    "clear-thought-mentalmodel",
)

EXPECTED_DECISION_FRAMEWORK_TOOL_NAMES = (
    "decisionFramework",
    "decision-framework-server-decisionFramework",
    "decisionframework",
    "clear-thought-server-decisionframework",
    "clear-thought-decisionframework",
)

EXPECTED_SCIENTIFIC_METHOD_TOOL_NAMES = (
    "scientificMethod",
    "scientific-method-server-scientificMethod",
    "scientificmethod",
    "clear-thought-server-scientificmethod",
    "clear-thought-scientificmethod",
)

EXPECTED_DEBUGGING_APPROACH_TOOL_NAMES = (
    "debuggingapproach",
    "clear-thought-server-debuggingapproach",
    "clear-thought-debuggingapproach",
)

EXPECTED_DESIGN_PATTERN_TOOL_NAMES = (
    "designpattern",
    "clear-thought-server-designpattern",
)

EXPECTED_COLLABORATIVE_REASONING_TOOL_NAMES = (
    "collaborativeReasoning",
    "collaborative-reasoning-server-collaborativeReasoning",
    "collaborativereasoning",
    "clear-thought-server-collaborativereasoning",
    "clear-thought-collaborativereasoning",
)

EXPECTED_VISUAL_REASONING_TOOL_NAMES = (
    "visualReasoning",
    "visual-reasoning-server-visualReasoning",
    "visualreasoning",
    "clear-thought-server-visualreasoning",
    "clear-thought-visualreasoning",
)

EXPECTED_PRESERVED_REASONING_TOOL_NAMES = (
    EXPECTED_THINK_TOOL_NAMES
    + EXPECTED_SEQUENTIAL_THINKING_TOOL_NAMES
    + EXPECTED_PENTEST_THINKING_TOOL_NAMES
    + EXPECTED_GAME_DESIGN_THINKING_TOOL_NAMES
    + EXPECTED_SKIA_ANIMATION_THINKING_TOOL_NAMES
    + EXPECTED_LOTUS_WISDOM_TOOL_NAMES
    + EXPECTED_STRUCTURED_ARGUMENTATION_TOOL_NAMES
    + EXPECTED_ANALOGICAL_REASONING_TOOL_NAMES
    + EXPECTED_MENTAL_MODEL_TOOL_NAMES
    + EXPECTED_DECISION_FRAMEWORK_TOOL_NAMES
    + EXPECTED_SCIENTIFIC_METHOD_TOOL_NAMES
    + EXPECTED_DEBUGGING_APPROACH_TOOL_NAMES
    + EXPECTED_DESIGN_PATTERN_TOOL_NAMES
    + EXPECTED_COLLABORATIVE_REASONING_TOOL_NAMES
    + EXPECTED_VISUAL_REASONING_TOOL_NAMES
)

# One non-protected witness for every broad legacy reasoning-name rule. These
# names deliberately exercise the destructive classifier without inheriting an
# exact audited-family exemption.
UNCALLED_LEGACY_REASONING_DEFINITION_NAMES = (
    "unaudited-thinking-probe",
    "unaudited-clear_thought-probe",
    "unaudited-clear-thought-probe",
    "unaudited-think-tool-probe",
    "unaudited-mentalmodel-probe",
    "unaudited-analogicalreasoning-probe",
    "unaudited-collaborativereasoning-probe",
    "unaudited-visualreasoning-probe",
    "unaudited-decisionframework-probe",
    "unaudited-scientificmethod-probe",
    "unaudited-socraticmethod-probe",
    "unaudited-metacognitive-probe",
    "unaudited-structuredargumentation-probe",
    "unaudited-debuggingapproach-probe",
    "unaudited-designpattern-probe",
    "unaudited-chain-of-draft-probe",
    "unaudited-chain_of_draft-probe",
    "unaudited-lotuswisdom-probe",
    "unaudited-think",
)

UNCALLED_SCAFFOLD_DEFINITION_NAMES = (
    "server__unlock_probe__",
    "server__get_instructions",
    "server-list_resources",
    "server-read_resource",
    "server-get_resource",
    "server-deep_researcher_check",
)

AUDITED_REASONING_TOOL_CASES = (
    *(
        pytest.param(name, THINK_TOOL_FAMILY, id=name)
        for name in EXPECTED_THINK_TOOL_NAMES
    ),
    *(
        pytest.param(name, SEQUENTIAL_THINKING_TOOL_FAMILY, id=name)
        for name in EXPECTED_SEQUENTIAL_THINKING_TOOL_NAMES
    ),
    *(
        pytest.param(name, PENTEST_THINKING_TOOL_FAMILY, id=name)
        for name in EXPECTED_PENTEST_THINKING_TOOL_NAMES
    ),
    *(
        pytest.param(name, GAME_DESIGN_THINKING_TOOL_FAMILY, id=name)
        for name in EXPECTED_GAME_DESIGN_THINKING_TOOL_NAMES
    ),
    *(
        pytest.param(name, SKIA_ANIMATION_THINKING_TOOL_FAMILY, id=name)
        for name in EXPECTED_SKIA_ANIMATION_THINKING_TOOL_NAMES
    ),
    *(
        pytest.param(name, LOTUS_WISDOM_TOOL_FAMILY, id=name)
        for name in EXPECTED_LOTUS_WISDOM_TOOL_NAMES
    ),
    *(
        pytest.param(name, STRUCTURED_ARGUMENTATION_TOOL_FAMILY, id=name)
        for name in EXPECTED_STRUCTURED_ARGUMENTATION_TOOL_NAMES
    ),
    *(
        pytest.param(name, ANALOGICAL_REASONING_TOOL_FAMILY, id=name)
        for name in EXPECTED_ANALOGICAL_REASONING_TOOL_NAMES
    ),
    *(
        pytest.param(name, MENTAL_MODEL_TOOL_FAMILY, id=name)
        for name in EXPECTED_MENTAL_MODEL_TOOL_NAMES
    ),
    *(
        pytest.param(name, DECISION_FRAMEWORK_TOOL_FAMILY, id=name)
        for name in EXPECTED_DECISION_FRAMEWORK_TOOL_NAMES
    ),
    *(
        pytest.param(name, SCIENTIFIC_METHOD_TOOL_FAMILY, id=name)
        for name in EXPECTED_SCIENTIFIC_METHOD_TOOL_NAMES
    ),
    *(
        pytest.param(name, DEBUGGING_APPROACH_TOOL_FAMILY, id=name)
        for name in EXPECTED_DEBUGGING_APPROACH_TOOL_NAMES
    ),
    *(
        pytest.param(name, DESIGN_PATTERN_TOOL_FAMILY, id=name)
        for name in EXPECTED_DESIGN_PATTERN_TOOL_NAMES
    ),
    *(
        pytest.param(name, COLLABORATIVE_REASONING_TOOL_FAMILY, id=name)
        for name in EXPECTED_COLLABORATIVE_REASONING_TOOL_NAMES
    ),
    *(
        pytest.param(name, VISUAL_REASONING_TOOL_FAMILY, id=name)
        for name in EXPECTED_VISUAL_REASONING_TOOL_NAMES
    ),
)


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


def kimi_tool_system(tools=None):
    serialized = orjson.dumps(tools or [{"type": "function"}]).decode()
    return f"<|im_system|>tool_declare<|im_middle|>{serialized}<|im_end|>"


def xml_tool_system(tools=None):
    serialized = "\n".join(
        orjson.dumps(tool).decode() for tool in (tools or [{"type": "function"}])
    )
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{serialized}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
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


def conversion_row(called_names=(), defined_names=None):
    """Minimal raw Toucan row for Stage-1 marker/conversion tests."""

    definitions = list(defined_names if defined_names is not None else called_names)
    raw_messages = [{"role": "user", "content": "u"}]
    for name in called_names:
        raw_messages.extend(
            [
                raw_asst(function_call=fcall(name, "{}")),
                raw_fn("ok"),
            ]
        )
    raw_messages.append(raw_asst(content="done"))
    row = qrow()
    row.update(
        {
            "uuid": "marker-row",
            "messages": orjson.dumps(raw_messages).decode(),
            "available_tools": orjson.dumps(
                [func(name) for name in definitions]
            ).decode(),
        }
    )
    return row


# --------------------------------------------------------------------------
# _is_reasoning_tool / _is_scaffold_tool
# --------------------------------------------------------------------------


class TestIsReasoningTool:
    def test_exact_preservation_vocabularies_are_complete_and_disjoint(self) -> None:
        """Pin the curation contract independently of loader construction."""

        assert (
            len(EXPECTED_THINK_TOOL_NAMES) == len(set(EXPECTED_THINK_TOOL_NAMES)) == 12
        )
        assert (
            len(EXPECTED_SEQUENTIAL_THINKING_TOOL_NAMES)
            == len(set(EXPECTED_SEQUENTIAL_THINKING_TOOL_NAMES))
            == 8
        )
        exact_families = (
            EXPECTED_THINK_TOOL_NAMES,
            EXPECTED_SEQUENTIAL_THINKING_TOOL_NAMES,
            EXPECTED_PENTEST_THINKING_TOOL_NAMES,
            EXPECTED_GAME_DESIGN_THINKING_TOOL_NAMES,
            EXPECTED_SKIA_ANIMATION_THINKING_TOOL_NAMES,
            EXPECTED_LOTUS_WISDOM_TOOL_NAMES,
            EXPECTED_STRUCTURED_ARGUMENTATION_TOOL_NAMES,
            EXPECTED_ANALOGICAL_REASONING_TOOL_NAMES,
            EXPECTED_MENTAL_MODEL_TOOL_NAMES,
            EXPECTED_DECISION_FRAMEWORK_TOOL_NAMES,
            EXPECTED_SCIENTIFIC_METHOD_TOOL_NAMES,
            EXPECTED_DEBUGGING_APPROACH_TOOL_NAMES,
            EXPECTED_DESIGN_PATTERN_TOOL_NAMES,
            EXPECTED_COLLABORATIVE_REASONING_TOOL_NAMES,
            EXPECTED_VISUAL_REASONING_TOOL_NAMES,
        )
        assert [len(names) for names in exact_families] == [
            12,
            8,
            2,
            2,
            2,
            4,
            5,
            2,
            3,
            5,
            5,
            3,
            2,
            5,
            5,
        ]
        assert sum(len(names) for names in exact_families) == len(
            set().union(*map(set, exact_families))
        )
        assert _THINK_TOOL_NAMES == frozenset(EXPECTED_THINK_TOOL_NAMES)
        assert _SEQUENTIAL_THINKING_TOOL_NAMES == frozenset(
            EXPECTED_SEQUENTIAL_THINKING_TOOL_NAMES
        )
        assert _PENTEST_THINKING_TOOL_NAMES == frozenset(
            EXPECTED_PENTEST_THINKING_TOOL_NAMES
        )
        assert _GAME_DESIGN_THINKING_TOOL_NAMES == frozenset(
            EXPECTED_GAME_DESIGN_THINKING_TOOL_NAMES
        )
        assert _SKIA_ANIMATION_THINKING_TOOL_NAMES == frozenset(
            EXPECTED_SKIA_ANIMATION_THINKING_TOOL_NAMES
        )
        assert _LOTUS_WISDOM_TOOL_NAMES == frozenset(EXPECTED_LOTUS_WISDOM_TOOL_NAMES)
        assert _STRUCTURED_ARGUMENTATION_TOOL_NAMES == frozenset(
            EXPECTED_STRUCTURED_ARGUMENTATION_TOOL_NAMES
        )
        assert _ANALOGICAL_REASONING_TOOL_NAMES == frozenset(
            EXPECTED_ANALOGICAL_REASONING_TOOL_NAMES
        )
        assert _MENTAL_MODEL_TOOL_NAMES == frozenset(EXPECTED_MENTAL_MODEL_TOOL_NAMES)
        assert _DECISION_FRAMEWORK_TOOL_NAMES == frozenset(
            EXPECTED_DECISION_FRAMEWORK_TOOL_NAMES
        )
        assert _SCIENTIFIC_METHOD_TOOL_NAMES == frozenset(
            EXPECTED_SCIENTIFIC_METHOD_TOOL_NAMES
        )
        assert _DEBUGGING_APPROACH_TOOL_NAMES == frozenset(
            EXPECTED_DEBUGGING_APPROACH_TOOL_NAMES
        )
        assert _DESIGN_PATTERN_TOOL_NAMES == frozenset(
            EXPECTED_DESIGN_PATTERN_TOOL_NAMES
        )
        assert _COLLABORATIVE_REASONING_TOOL_NAMES == frozenset(
            EXPECTED_COLLABORATIVE_REASONING_TOOL_NAMES
        )
        assert _VISUAL_REASONING_TOOL_NAMES == frozenset(
            EXPECTED_VISUAL_REASONING_TOOL_NAMES
        )
        assert _PRESERVED_REASONING_TOOL_NAMES == frozenset(
            EXPECTED_PRESERVED_REASONING_TOOL_NAMES
        )

    @pytest.mark.parametrize(
        ("name", "family"),
        AUDITED_REASONING_TOOL_CASES,
    )
    def test_every_exact_audited_name_is_preserved(
        self, name: str, family: str
    ) -> None:
        del family
        assert _is_reasoning_tool(name) is False

    def test_audited_think_names_are_preserved_but_lookalikes_are_not(self) -> None:
        assert _is_reasoning_tool("think") is False
        assert _is_reasoning_tool("think-tool-think") is False
        assert _is_reasoning_tool("think-tool-server-think") is False
        # Tool names are case-sensitive; an unaudited variant does not inherit
        # an audited identity merely through case folding.
        assert _is_reasoning_tool("THINK") is True
        assert _is_reasoning_tool("server-think") is True

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-sequentialthinking",
            "clear-thought-server-sequentialthinking",
            "model-context-protocol-server-sequentialthinking",
            "reference-servers-sequentialthinking",
            "sequential-thinking-sequentialthinking",
            "sequential-thinking-tools-sequentialthinking_tools",
            "sequentialthinking",
            "sequentialthinking_tools",
        ],
    )
    def test_eight_exact_sequential_names_are_preserved(self, name: str) -> None:
        assert _is_reasoning_tool(name) is False

    def test_core_families(self) -> None:
        assert _is_reasoning_tool("mcp-sequentialthinking-tools") is True
        assert _is_reasoning_tool("clear_thought") is True
        assert _is_reasoning_tool("clear-thought") is True
        assert _is_reasoning_tool("think-tool") is True

    def test_reasoning_method_tools(self) -> None:
        assert _is_reasoning_tool("mentalmodel") is False
        assert _is_reasoning_tool("collaborativeReasoning") is False
        assert _is_reasoning_tool("visualReasoning") is False

    @pytest.mark.parametrize("name", EXPECTED_COLLABORATIVE_REASONING_TOOL_NAMES)
    def test_five_exact_collaborative_reasoning_names_are_preserved(
        self, name: str
    ) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize("name", EXPECTED_VISUAL_REASONING_TOOL_NAMES)
    def test_five_exact_visual_reasoning_names_are_preserved(self, name: str) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-collaborative_reasoning",
            "clear-thought-server-collaboratereasoning",
            "collaborative-reasoning-server-collaboraborReasoning",
            "virtual-reasoning-server-visualReasoning",
            "CollaborativeReasoning",
            "VisualReasoning",
            'visualReasoning", {"diagramId": "d-1"',
        ],
    )
    def test_collaborative_visual_lookalikes_do_not_inherit_audited_identity(
        self, name: str
    ) -> None:
        assert name not in _COLLABORATIVE_REASONING_TOOL_NAMES
        assert name not in _VISUAL_REASONING_TOOL_NAMES

    @pytest.mark.parametrize("name", EXPECTED_ANALOGICAL_REASONING_TOOL_NAMES)
    def test_two_exact_analogical_reasoning_names_are_preserved(
        self, name: str
    ) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "analogical-reasoning-server-analogialReasoning",
            "hotel-booking-server-analogicalReasoning",
            "AnalogicalReasoning",
            "analogical_reasoning",
            'analogicalReasoning", {"analogyId": "a-1',
        ],
    )
    def test_fuzzy_analogical_variants_do_not_inherit_audited_identity(
        self, name: str
    ) -> None:
        assert name not in _ANALOGICAL_REASONING_TOOL_NAMES
        # The typo and invented underscore spelling are not caught by the legacy
        # lexical predicate, while the other lookalikes are. Neither heuristic
        # outcome grants exact-family status.
        if name in {
            "analogical-reasoning-server-analogialReasoning",
            "analogical_reasoning",
        }:
            assert _is_reasoning_tool(name) is False
        else:
            assert _is_reasoning_tool(name) is True

    @pytest.mark.parametrize("name", EXPECTED_MENTAL_MODEL_TOOL_NAMES)
    def test_three_exact_mental_model_names_are_preserved(self, name: str) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-mental_model",
            "clear_thought-server-mentalmodel",
            "clear-thought-server-meuqalModel",
            "clear-thought-server-clearmodel",
            "Clear-Thought-MentalModel",
            "hotel-booking-server-mentalmodel",
            'clear-thought-mentalmodel", {"modelName": "first_principles"',
        ],
    )
    def test_fuzzy_mental_model_variants_do_not_inherit_audited_identity(
        self, name: str
    ) -> None:
        assert name not in _MENTAL_MODEL_TOOL_NAMES
        assert _is_reasoning_tool(name) is True

    @pytest.mark.parametrize("name", EXPECTED_DECISION_FRAMEWORK_TOOL_NAMES)
    def test_five_exact_decision_framework_names_are_preserved(self, name: str) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-decision_framework",
            "DecisionFramework",
            "decision-framework-server-decisionframework",
            "hotel-booking-server-decisionFramework",
            'decisionFramework", {"decisionId": "d-1"',
            'model-context-protocol-server-sequentialthinking\\", '
            '{\\"thought\\": \\"apply a decision framework\\"',
        ],
    )
    def test_fuzzy_decision_framework_variants_do_not_inherit_identity(
        self, name: str
    ) -> None:
        assert name not in _DECISION_FRAMEWORK_TOOL_NAMES
        assert _is_reasoning_tool(name) is True

    @pytest.mark.parametrize("name", EXPECTED_SCIENTIFIC_METHOD_TOOL_NAMES)
    def test_five_exact_scientific_method_names_are_preserved(self, name: str) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-scientific_method",
            "clear-thought-server-meticScientificMethod",
            "clear-thought-server-scientificmethod@8751",
            "clear-thought-server-se scientificmethod",
            "scientific-method-server-scientificmethod",
        ],
    )
    def test_observed_scientific_method_lookalikes_do_not_inherit_identity(
        self, name: str
    ) -> None:
        assert name not in _SCIENTIFIC_METHOD_TOOL_NAMES
        assert _is_reasoning_tool(name) is True

    def test_severely_misspelled_scientific_name_is_neither_promoted_nor_classified(
        self,
    ) -> None:
        name = "scientific-method-server-scienticMethod"
        assert name not in _SCIENTIFIC_METHOD_TOOL_NAMES
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize("name", EXPECTED_DEBUGGING_APPROACH_TOOL_NAMES)
    def test_three_exact_debugging_approach_names_are_preserved(
        self, name: str
    ) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize("name", EXPECTED_DESIGN_PATTERN_TOOL_NAMES)
    def test_two_exact_design_pattern_names_are_preserved(self, name: str) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-debugging_approach",
            "clear-thought-designpattern",
            "clear-thought-server-integrationpattern",
        ],
    )
    def test_observed_clear_thought_debugging_design_lookalikes_do_not_inherit_identity(
        self, name: str
    ) -> None:
        assert name not in _DEBUGGING_APPROACH_TOOL_NAMES
        assert name not in _DESIGN_PATTERN_TOOL_NAMES
        assert _is_reasoning_tool(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "search_files_by_pattern",
            'flux-imagegen-server-generateImage", {"prompt": "professional design"',
        ],
    )
    def test_lexical_pattern_false_positives_are_not_reasoning_tools(
        self, name: str
    ) -> None:
        assert name not in _DEBUGGING_APPROACH_TOOL_NAMES
        assert name not in _DESIGN_PATTERN_TOOL_NAMES
        assert _is_reasoning_tool(name) is False

    def test_thought_state_operations_are_not_reasoning(self) -> None:
        for prefix in ("", "think-tool-", "think-tool-server-"):
            assert _is_reasoning_tool(f"{prefix}get_thoughts") is False
            assert _is_reasoning_tool(f"{prefix}get_thought_stats") is False
            assert _is_reasoning_tool(f"{prefix}clear_thoughts") is False

    def test_full_clear_thought_op_set(self) -> None:
        # the whole reasoning-server op vocabulary, incl. standalone prefixes
        # and bare names that the upstream loader missed.
        for n in (
            "socraticmethod",
            "metacognitiveMonitoring",
            "chain-of-draft-server-chain-of-draft",
        ):
            assert _is_reasoning_tool(n) is True, n

    @pytest.mark.parametrize("name", EXPECTED_LOTUS_WISDOM_TOOL_NAMES)
    def test_four_exact_lotus_wisdom_names_are_preserved(self, name: str) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "LotusWisdom",
            "lotuswisdom-framework",
            "lotus-wisdom-lotuswisdom_summary ",
            'lotus-wisdom-lotuswisdom", {"tag": "embody',
        ],
    )
    def test_fuzzy_lotus_variants_do_not_inherit_audited_identity(
        self, name: str
    ) -> None:
        assert name not in _LOTUS_WISDOM_TOOL_NAMES
        assert _is_reasoning_tool(name) is True

    def test_undefined_lotus_wisdom_misspelling_is_not_an_audited_name(
        self,
    ) -> None:
        # The pinned snapshot contains this one undefined call. Its spelling does
        # not match either exact operation, so it is retained as an ordinary call
        # for defined-function validation rather than being granted protection.
        name = "lotus-wisdom-wisdom"
        assert name not in _LOTUS_WISDOM_TOOL_NAMES
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize("name", EXPECTED_STRUCTURED_ARGUMENTATION_TOOL_NAMES)
    def test_five_exact_structured_argumentation_names_are_preserved(
        self, name: str
    ) -> None:
        assert _is_reasoning_tool(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-structuredArgumentation",
            "clear-thought-structured_argumentation",
            "ai-research-assistant---structured-argumentation-server-"
            "structuredArgumentation",
            'structured-argumentation-server-structuredArgumentation", '
            '{"argumentId": "arg-14',
        ],
    )
    def test_fuzzy_structured_argumentation_variants_do_not_inherit_identity(
        self, name: str
    ) -> None:
        assert name not in _STRUCTURED_ARGUMENTATION_TOOL_NAMES
        assert _is_reasoning_tool(name) is True

    def test_other_operation_on_argumentation_server_is_not_in_family(self) -> None:
        name = "structured-argumentation-server-visualizeArgumentMap"
        assert name not in _STRUCTURED_ARGUMENTATION_TOOL_NAMES
        assert _is_reasoning_tool(name) is False

    def test_unaudited_thinking_scaffolds(self) -> None:
        assert _is_reasoning_tool("systemsthinking") is True
        assert _is_reasoning_tool("creativethinking") is True

    @pytest.mark.parametrize(
        "name",
        [
            "pentestthinking-pentesttestingMCP",
            "pentestthinking-pentestthinkingMcp",
            "react-native-skia-animationthinkingtool-skiaanimationthinking",
            'game-engine-server-gamedesignthinking", {"name": "other',
        ],
    )
    def test_fuzzy_domain_variants_do_not_inherit_audited_identity(
        self, name: str
    ) -> None:
        assert _is_reasoning_tool(name) is True

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
        # Structural issues come from _stage1_issues. Conversion reports no
        # issue merely because a known redundant system template was omitted.
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

    @pytest.mark.parametrize(
        "content",
        [kimi_tool_system(), xml_tool_system()],
    )
    def test_known_embedded_tool_system_is_omitted(self, content: str) -> None:
        assert _is_embedded_tool_system_message(content) is True
        out, _ = _convert_messages(
            [
                {"role": "system", "content": content},
                {"role": "user", "content": "u"},
            ]
        )
        assert out == [{"role": "user", "content": "u"}]

    @pytest.mark.parametrize("template", [kimi_tool_system(), xml_tool_system()])
    @pytest.mark.parametrize(
        ("content_factory", "expected"),
        [
            (lambda template: f"{template}\n\nFollow policy P.", "Follow policy P."),
            (lambda template: f"Follow policy P.\n\n{template}", "Follow policy P."),
            (
                lambda template: f"Instruction A.\n\n{template}\n\nInstruction B.",
                "Instruction A.\n\nInstruction B.",
            ),
        ],
    )
    def test_preserves_custom_system_text_around_verified_template(
        self, template: str, content_factory, expected: str
    ) -> None:
        content = content_factory(template)
        remaining, removed = _strip_embedded_tool_system_content(content)
        assert removed is True
        assert remaining == expected
        assert _is_embedded_tool_system_message(content) is False
        out, _ = _convert_messages([{"role": "system", "content": content}])
        assert out == [{"role": "system", "content": expected}]

    def test_malformed_xml_template_is_preserved_whole(self) -> None:
        content = xml_tool_system().replace('{"type":"function"}', "custom instruction")
        remaining, removed = _strip_embedded_tool_system_content(content)
        assert (remaining, removed) == (content, False)
        out, _ = _convert_messages([{"role": "system", "content": content}])
        assert out == [{"role": "system", "content": content}]

    def test_unexpected_system_is_not_heuristically_edited(self) -> None:
        content = "Follow the user's instructions. The word <tools> is incidental."
        assert _is_embedded_tool_system_message(content) is False
        out, _ = _convert_messages([{"role": "system", "content": content}])
        assert out == [{"role": "system", "content": content}]

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


class TestReasoningToolFamilyAnnotations:
    @pytest.mark.parametrize(
        ("name", "family"),
        AUDITED_REASONING_TOOL_CASES,
    )
    def test_every_exact_name_is_marked_and_preserved_by_default(
        self, name: str, family: str
    ) -> None:
        sample, _ = _convert_sample(conversion_row(called_names=(name,)), "OSS")
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.annotations == {REASONING_TOOL_FAMILIES_ANNOTATION: [family]}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert [
            tool_call["function"]["name"]
            for message in sample.messages
            for tool_call in message.get("tool_calls") or []
        ] == [name]
        assert [tool["function"]["name"] for tool in sample.tools] == [name]

    def test_marks_only_families_with_actual_exact_calls(self) -> None:
        sample, _ = _convert_sample(
            conversion_row(
                called_names=(
                    "think",
                    "sequentialthinking",
                    "pentestthinkingMCP",
                    "gamedesignthinking",
                    "skiaanimationthinking",
                    "lotuswisdom",
                    "structuredArgumentation",
                    "analogicalReasoning",
                    "mentalmodel",
                    "decisionFramework",
                    "scientificMethod",
                    "debuggingapproach",
                    "designpattern",
                    "collaborativeReasoning",
                    "visualReasoning",
                ),
                defined_names=(
                    "think",
                    "sequentialthinking",
                    "pentestthinkingMCP",
                    "gamedesignthinking",
                    "skiaanimationthinking",
                    "lotuswisdom",
                    "structuredArgumentation",
                    "analogicalReasoning",
                    "mentalmodel",
                    "decisionFramework",
                    "scientificMethod",
                    "debuggingapproach",
                    "designpattern",
                    "collaborativeReasoning",
                    "visualReasoning",
                    "get_weather",
                ),
            ),
            "OSS",
        )

        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [
                THINK_TOOL_FAMILY,
                SEQUENTIAL_THINKING_TOOL_FAMILY,
                PENTEST_THINKING_TOOL_FAMILY,
                GAME_DESIGN_THINKING_TOOL_FAMILY,
                SKIA_ANIMATION_THINKING_TOOL_FAMILY,
                LOTUS_WISDOM_TOOL_FAMILY,
                STRUCTURED_ARGUMENTATION_TOOL_FAMILY,
                ANALOGICAL_REASONING_TOOL_FAMILY,
                MENTAL_MODEL_TOOL_FAMILY,
                DECISION_FRAMEWORK_TOOL_FAMILY,
                SCIENTIFIC_METHOD_TOOL_FAMILY,
                DEBUGGING_APPROACH_TOOL_FAMILY,
                DESIGN_PATTERN_TOOL_FAMILY,
                COLLABORATIVE_REASONING_TOOL_FAMILY,
                VISUAL_REASONING_TOOL_FAMILY,
            ]
        }

    def test_unused_definitions_do_not_mark_the_row(self) -> None:
        sample, _ = _convert_sample(
            conversion_row(
                called_names=("get_weather",),
                defined_names=(
                    "get_weather",
                    "think",
                    "sequentialthinking",
                    "pentestthinkingMCP",
                    "gamedesignthinking",
                    "skiaanimationthinking",
                    "lotuswisdom",
                    "structuredArgumentation",
                    "analogicalReasoning",
                    "mentalmodel",
                    "decisionFramework",
                    "scientificMethod",
                    "clear-thought-debuggingapproach",
                    "clear-thought-server-designpattern",
                    "clear-thought-collaborativereasoning",
                    "clear-thought-visualreasoning",
                ),
            ),
            "Kimi-K2",
        )

        assert sample.annotations == {}

    def test_uncalled_legacy_definitions_are_preserved_without_marking(self) -> None:
        defined_names = (
            "get_weather",
            *UNCALLED_LEGACY_REASONING_DEFINITION_NAMES,
            *UNCALLED_SCAFFOLD_DEFINITION_NAMES,
        )
        sample, _ = _convert_sample(
            conversion_row(
                called_names=("get_weather",),
                defined_names=defined_names,
            ),
            "Kimi-K2",
        )
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config(
            [sample],
            ToucanConfig(strip_reasoning_tools=True, strip_scaffold_tools=True),
        )

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.annotations == {}
        assert sample.tools == original_tools

    def test_fuzzy_case_variant_does_not_receive_an_audited_marker(self) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=("SequentialThinking",)), "Qwen3"
        )

        assert sample.annotations == {}

    def test_fuzzy_lotus_variant_does_not_receive_an_audited_marker(self) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=("LotusWisdom",)), "Qwen3"
        )

        assert sample.annotations == {}

    def test_fuzzy_structured_variant_does_not_receive_a_marker(self) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=("clear-thought-structured_argumentation",)),
            "Qwen3",
        )

        assert sample.annotations == {}

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-collaborative_reasoning",
            "clear-thought-server-collaboratereasoning",
            "collaborative-reasoning-server-collaboraborReasoning",
            "virtual-reasoning-server-visualReasoning",
            "CollaborativeReasoning",
            "VisualReasoning",
        ],
    )
    def test_collaborative_visual_lookalike_does_not_receive_a_marker(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,), defined_names=(name,)), "Qwen3"
        )

        assert sample.annotations == {}

    @pytest.mark.parametrize(
        "name",
        [
            "analogical-reasoning-server-analogialReasoning",
            "hotel-booking-server-analogicalReasoning",
            "AnalogicalReasoning",
            "analogical_reasoning",
        ],
    )
    def test_fuzzy_analogical_variant_does_not_receive_a_marker(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,), defined_names=(name,)), "Qwen3"
        )

        assert sample.annotations == {}

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-mental_model",
            "clear_thought-server-mentalmodel",
            "clear-thought-server-meuqalModel",
            "clear-thought-server-clearmodel",
            "Clear-Thought-MentalModel",
            "hotel-booking-server-mentalmodel",
        ],
    )
    def test_fuzzy_mental_model_variant_does_not_receive_a_marker(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,)),
            "Qwen3",
        )

        assert sample.annotations == {}

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-decision_framework",
            "DecisionFramework",
            "decision-framework-server-decisionframework",
            "hotel-booking-server-decisionFramework",
        ],
    )
    def test_fuzzy_decision_framework_variant_does_not_receive_a_marker(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,)),
            "Qwen3",
        )

        assert sample.annotations == {}

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-debugging_approach",
            "clear-thought-designpattern",
            "clear-thought-server-integrationpattern",
            "search_files_by_pattern",
        ],
    )
    def test_observed_debugging_design_nearby_name_does_not_receive_a_marker(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,), defined_names=(name,)), "Qwen3"
        )

        assert sample.annotations == {}

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-scientific_method",
            "clear-thought-server-meticScientificMethod",
            "clear-thought-server-scientificmethod@8751",
            "clear-thought-server-se scientificmethod",
            "scientific-method-server-scienticMethod",
            "scientific-method-server-scientificmethod",
        ],
    )
    def test_observed_scientific_method_lookalike_is_unmarked(self, name: str) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,)),
            "Qwen3",
        )

        assert sample.annotations == {}


class TestStage1Issues:
    def test_clean_conversation_no_issues(self) -> None:
        msgs = [system("s"), user("u"), assistant(content="a")]
        assert _stage1_issues(msgs) == {}

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

    def test_same_name_difference_outside_description_and_schema_conflicts(
        self,
    ) -> None:
        first = func("a")
        second = copy.deepcopy(first)
        second["function"]["strict"] = True
        raw = {"available_tools": orjson.dumps([first, second]).decode()}

        assert _has_conflicting_duplicate_tools(raw) is True

    def test_object_key_order_alone_is_not_a_conflict(self) -> None:
        first = func("a")
        function = first["function"]
        second = {
            "function": {
                "parameters": function["parameters"],
                "description": function["description"],
                "name": function["name"],
            },
            "type": "function",
        }
        raw = {"available_tools": orjson.dumps([first, second]).decode()}

        assert _has_conflicting_duplicate_tools(raw) is False

    def test_nested_object_key_order_alone_is_not_a_conflict(self) -> None:
        first = func(
            "a",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City"}},
                "required": ["location"],
            },
        )
        second = {
            "function": {
                "parameters": {
                    "required": ["location"],
                    "properties": {
                        "location": {"description": "City", "type": "string"}
                    },
                    "type": "object",
                },
                "description": "",
                "name": "a",
            },
            "type": "function",
        }
        raw = {"available_tools": orjson.dumps([first, second]).decode()}

        assert _has_conflicting_duplicate_tools(raw) is False

    def test_array_order_remains_part_of_the_complete_definition(self) -> None:
        first = func(
            "a",
            parameters={
                "type": "object",
                "properties": {"x": {}, "y": {}},
                "required": ["x", "y"],
            },
        )
        second = copy.deepcopy(first)
        second["function"]["parameters"]["required"] = ["y", "x"]
        raw = {"available_tools": orjson.dumps([first, second]).decode()}

        # The comparator is structural equality modulo object-key order, not
        # full JSON-Schema semantic equivalence. Arrays are never reordered.
        assert _has_conflicting_duplicate_tools(raw) is True

    def test_case_distinct_visible_names_are_not_merged(self) -> None:
        raw = {
            "available_tools": orjson.dumps(
                [func("sequentialthinking"), func("SequentialThinking")]
            ).decode()
        }

        assert _has_conflicting_duplicate_tools(raw) is False

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

    def test_malformed_function_object_is_ignored_without_crashing(self) -> None:
        raw = {
            "available_tools": orjson.dumps(
                [{"type": "function", "function": "bad"}, func("a")]
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

    @pytest.mark.parametrize("name", EXPECTED_PRESERVED_REASONING_TOOL_NAMES)
    def test_audited_reasoning_tool_calls_are_always_preserved(self, name: str) -> None:
        s = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="observation"),
                assistant(content="done"),
            ],
        )
        original_messages = copy.deepcopy(s.messages)
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.messages == original_messages
        assert s.tools == original_tools

    @pytest.mark.parametrize("name", EXPECTED_PRESERVED_REASONING_TOOL_NAMES)
    def test_audited_call_keeps_its_positional_result_and_definition_when_mixed(
        self, name: str
    ) -> None:
        s = conv(
            tools=[
                func(name, description="protected"),
                func("chain-of-draft", description="strippable"),
                func("get_weather", description="ordinary"),
            ],
            messages=[
                assistant(
                    content="visible assistant content",
                    tool_calls=[
                        call(name=name, call_id="protected"),
                        call(name="chain-of-draft", call_id="reasoning"),
                        call(name="get_weather", call_id="ordinary"),
                    ],
                ),
                tool_response(content="protected result", tool_call_id="protected"),
                tool_response(content="reasoning result", tool_call_id="reasoning"),
                tool_response(content="weather result", tool_call_id="ordinary"),
                assistant(content="done"),
            ],
        )

        assert _strip_tools(s, True, False) == (True, True, False)
        assert s.messages[0]["content"] == "visible assistant content"
        assert [
            tool_call["function"]["name"] for tool_call in s.messages[0]["tool_calls"]
        ] == [name, "get_weather"]
        assert [
            message["tool_call_id"]
            for message in s.messages
            if message["role"] == "tool"
        ] == ["protected", "ordinary"]
        assert [
            message["content"] for message in s.messages if message["role"] == "tool"
        ] == ["protected result", "weather result"]
        assert [tool["function"]["name"] for tool in s.tools] == [
            name,
            "get_weather",
        ]

    def test_all_uncalled_audited_definitions_are_preserved(self) -> None:
        s = conv(
            tools=[func(name) for name in EXPECTED_PRESERVED_REASONING_TOOL_NAMES],
            messages=[user("u"), assistant(content="done")],
        )
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.tools == original_tools

    def test_undefined_fuzzy_reasoning_call_is_left_for_validation(self) -> None:
        s = conv(
            tools=[func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name="SequentialThinking")]),
                tool_response(content="Tool does not exist"),
            ],
        )
        original_messages = copy.deepcopy(s.messages)

        # The broad legacy name predicate recognizes this spelling, but an
        # undefined call must not be deleted before defined-function validation.
        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.messages == original_messages

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-structuredArgumentation",
            "clear-thought-structured_argumentation",
            "ai-research-assistant---structured-argumentation-server-"
            "structuredArgumentation",
            'structured-argumentation-server-structuredArgumentation", '
            '{"argumentId": "arg-14',
        ],
    )
    def test_undefined_structured_lookalikes_are_left_for_validation(
        self, name: str
    ) -> None:
        sample = conv(
            tools=[func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="Tool does not exist"),
            ],
        )
        original_messages = copy.deepcopy(sample.messages)

        assert _strip_tools(sample, True, False) == (False, False, False)
        assert sample.messages == original_messages

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-debugging_approach",
            "clear-thought-designpattern",
            "clear-thought-server-integrationpattern",
        ],
    )
    def test_defined_balanced_debugging_design_lookalikes_remain_legacy_strippable(
        self, name: str
    ) -> None:
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="lookalike observation"),
            ],
        )

        assert name not in _DEBUGGING_APPROACH_TOOL_NAMES
        assert name not in _DESIGN_PATTERN_TOOL_NAMES
        assert _strip_tools(sample, True, False) == (True, True, False)
        assert sample.messages == []
        assert sample.tools == []

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-debugging_approach",
            "clear-thought-designpattern",
            "clear-thought-server-integrationpattern",
        ],
    )
    def test_undefined_debugging_design_lookalikes_are_left_for_validation(
        self, name: str
    ) -> None:
        sample = conv(
            tools=[func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="Tool does not exist"),
            ],
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        assert _strip_tools(sample, True, False) == (False, False, False)
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-mental_model",
            "clear_thought-server-mentalmodel",
            "clear-thought-server-meuqalModel",
            "clear-thought-server-clearmodel",
            "Clear-Thought-MentalModel",
            "hotel-booking-server-mentalmodel",
            'clear-thought-mentalmodel", {"modelName": "first_principles"',
        ],
    )
    def test_defined_balanced_mental_model_lookalikes_remain_legacy_strippable(
        self, name: str
    ) -> None:
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="lookalike observation"),
            ],
        )

        assert name not in _MENTAL_MODEL_TOOL_NAMES
        assert _strip_tools(sample, True, False) == (True, True, False)
        assert sample.messages == []
        assert sample.tools == []

    def test_defined_severely_misspelled_scientific_name_fails_closed(self) -> None:
        name = "scientific-method-server-scienticMethod"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="lookalike observation"),
            ],
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        assert name not in _SCIENTIFIC_METHOD_TOOL_NAMES
        assert _is_reasoning_tool(name) is False
        assert _strip_tools(sample, True, False) == (False, False, False)
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-decision_framework",
            "DecisionFramework",
            "decision-framework-server-decisionframework",
            "hotel-booking-server-decisionFramework",
            'decisionFramework", {"decisionId": "d-1"',
        ],
    )
    def test_defined_balanced_decision_lookalikes_remain_legacy_strippable(
        self, name: str
    ) -> None:
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="lookalike observation"),
            ],
        )

        assert name not in _DECISION_FRAMEWORK_TOOL_NAMES
        assert _strip_tools(sample, True, False) == (True, True, False)
        assert sample.messages == []
        assert sample.tools == []

    def test_undefined_observed_decision_underscore_call_is_left_for_validation(
        self,
    ) -> None:
        name = "clear-thought-decision_framework"
        sample = conv(
            tools=[func("clear-thought-decisionframework")],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content=f"Tool {name} does not exists."),
            ],
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        assert _strip_tools(sample, True, False) == (False, False, False)
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-scientific_method",
            "clear-thought-server-meticScientificMethod",
            "clear-thought-server-scientificmethod@8751",
            "clear-thought-server-se scientificmethod",
            "scientific-method-server-scientificmethod",
        ],
    )
    def test_defined_balanced_scientific_lookalikes_remain_legacy_strippable(
        self, name: str
    ) -> None:
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="lookalike observation"),
            ],
        )

        assert name not in _SCIENTIFIC_METHOD_TOOL_NAMES
        assert _strip_tools(sample, True, False) == (True, True, False)
        assert sample.messages == []
        assert sample.tools == []

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-scientific_method",
            "clear-thought-server-meticScientificMethod",
            "clear-thought-server-scientificmethod@8751",
            "clear-thought-server-se scientificmethod",
            "scientific-method-server-scienticMethod",
            "scientific-method-server-scientificmethod",
        ],
    )
    def test_undefined_observed_scientific_lookalikes_remain_visible(
        self, name: str
    ) -> None:
        sample = conv(
            tools=[func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content=f"Tool {name} does not exists."),
            ],
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        assert _strip_tools(sample, True, False) == (False, False, False)
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    @pytest.mark.parametrize(
        "name",
        [
            'lotus-wisdom-lotuswisdom", {"tag": "embody',
            "lotus-wisdom-wisdom",
        ],
    )
    def test_undefined_lotus_variants_are_left_for_validation(self, name: str) -> None:
        sample = conv(
            tools=[func("get_weather")],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="Tool does not exist"),
            ],
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        assert _strip_tools(sample, True, False) == (False, False, False)
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_conflicting_name_is_never_partially_stripped(self) -> None:
        first = func("chain-of-draft", description="contract A")
        second = func("chain-of-draft", description="contract B")
        s = conv(
            tools=[first, second],
            messages=[
                assistant(tool_calls=[call(name="chain-of-draft")]),
                tool_response(content="ambiguous observation"),
            ],
        )
        original_messages = copy.deepcopy(s.messages)
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.messages == original_messages
        assert s.tools == original_tools

    def test_strips_reasoning_call_and_its_response(self) -> None:
        s = conv(
            tools=[func("chain-of-draft"), func("get_weather")],
            messages=[
                assistant(
                    content="reasoning",
                    tool_calls=[
                        call(name="chain-of-draft"),
                        call(name="get_weather"),
                    ],
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
            tools=[func("chain-of-draft")],
            messages=[
                assistant(content="visible", tool_calls=[call(name="chain-of-draft")]),
                tool_response(content=""),
            ],
        )
        assert _strip_tools(s, True, False)[0] is True
        assert s.messages == [{"role": "assistant", "content": "visible"}]

    def test_reasoning_only_turn_without_content_is_dropped(self) -> None:
        s = conv(
            tools=[func("chain-of-draft")],
            messages=[
                assistant(tool_calls=[call(name="chain-of-draft")]),
                tool_response(content=""),
            ],
        )
        assert _strip_tools(s, True, False)[0] is True
        assert s.messages == []

    def test_unbalanced_turn_left_untouched(self) -> None:
        # 2 calls, 1 response -> not the loader's normal output; leave alone.
        s = conv(
            tools=[func("chain-of-draft"), func("get_weather")],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="chain-of-draft"),
                        call(name="get_weather"),
                    ]
                ),
                tool_response(),
            ],
        )
        before = [dict(m) for m in s.messages]
        changed, rem_r, _ = _strip_tools(s, True, False)
        # The whole unbalanced name is protected, including its definition.
        assert [m for m in s.messages] == before
        assert [t["function"]["name"] for t in s.tools] == [
            "chain-of-draft",
            "get_weather",
        ]
        assert changed is False and rem_r is False

    def test_preserves_uncalled_reasoning_tool_definition(self) -> None:
        s = conv(
            tools=[func("chain-of-draft"), func("get_weather")],
            messages=[user("u"), assistant(content="hi")],
        )
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.tools == original_tools

    @pytest.mark.parametrize("name", UNCALLED_LEGACY_REASONING_DEFINITION_NAMES)
    def test_preserves_every_uncalled_legacy_reasoning_definition_rule(
        self, name: str
    ) -> None:
        assert _is_reasoning_tool(name) is True
        s = conv(
            tools=[func(name), func("get_weather")],
            messages=[user("u"), assistant(content="hi")],
        )
        original_messages = copy.deepcopy(s.messages)
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.messages == original_messages
        assert s.tools == original_tools

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
            tools=[
                func("chain-of-draft"),
                func("fs-read_resource"),
                func("get_weather"),
            ],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="chain-of-draft"),
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

    @pytest.mark.parametrize("name", UNCALLED_SCAFFOLD_DEFINITION_NAMES)
    def test_preserves_every_uncalled_scaffold_definition_rule(self, name: str) -> None:
        assert _is_scaffold_tool(name) is True
        s = conv(
            tools=[func(name), func("get_weather")],
            messages=[user("u"), assistant(content="hi")],
        )
        original_messages = copy.deepcopy(s.messages)
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, True) == (False, False, False)
        assert s.messages == original_messages
        assert s.tools == original_tools

    def test_prunes_only_definitions_whose_calls_were_actually_stripped(
        self,
    ) -> None:
        s = conv(
            tools=[
                func("chain-of-draft"),
                func("fs-read_resource"),
                func("mentalmodel"),
                func("srv-get_resource"),
                func("get_weather"),
            ],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="chain-of-draft", call_id="reasoning"),
                        call(name="fs-read_resource", call_id="scaffold"),
                        call(name="get_weather", call_id="ordinary"),
                    ]
                ),
                tool_response(content="draft", tool_call_id="reasoning"),
                tool_response(content="file", tool_call_id="scaffold"),
                tool_response(content="sunny", tool_call_id="ordinary"),
            ],
        )

        assert _strip_tools(s, True, True) == (True, True, True)
        assert [
            tool_call["function"]["name"] for tool_call in s.messages[0]["tool_calls"]
        ] == ["get_weather"]
        assert [tool["function"]["name"] for tool in s.tools] == [
            "mentalmodel",
            "srv-get_resource",
            "get_weather",
        ]

    def test_unbalanced_episode_protects_same_name_balanced_episode_and_definition(
        self,
    ) -> None:
        s = conv(
            tools=[func("chain-of-draft")],
            messages=[
                assistant(tool_calls=[call(name="chain-of-draft", call_id="first")]),
                tool_response(content="first", tool_call_id="first"),
                assistant(
                    tool_calls=[
                        call(name="chain-of-draft", call_id="second"),
                        call(name="chain-of-draft", call_id="third"),
                    ]
                ),
                tool_response(content="second", tool_call_id="second"),
            ],
        )
        original_messages = copy.deepcopy(s.messages)
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.messages == original_messages
        assert s.tools == original_tools

    def test_called_stripped_name_prunes_all_identical_duplicate_definitions(
        self,
    ) -> None:
        definition = func("chain-of-draft", description="same contract")
        s = conv(
            tools=[copy.deepcopy(definition), copy.deepcopy(definition)],
            messages=[
                assistant(tool_calls=[call(name="chain-of-draft")]),
                tool_response(content="draft"),
            ],
        )

        assert _strip_tools(s, True, False) == (True, True, False)
        assert s.messages == []
        assert s.tools == []

    def test_uncalled_conflicting_duplicate_definitions_remain_byte_for_byte(
        self,
    ) -> None:
        s = conv(
            tools=[
                func("chain-of-draft", description="contract A"),
                func("chain-of-draft", description="contract B"),
            ],
            messages=[user("u"), assistant(content="done")],
        )
        original_tools = copy.deepcopy(s.tools)

        assert _strip_tools(s, True, False) == (False, False, False)
        assert s.tools == original_tools


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
            tools=[
                func("chain-of-draft"),
                func("srv-read_resource"),
                func("get_weather"),
            ],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="chain-of-draft"),
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
        # Legacy reasoning stripped; scaffold kept (read_resource survives).
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

    def test_same_name_definition_variation_across_rows_is_allowed(self) -> None:
        first_raw = qrow()
        first_raw["available_tools"] = orjson.dumps(
            [func("sequentialthinking", description="reference contract")]
        ).decode()
        second_raw = qrow()
        second_raw["available_tools"] = orjson.dumps(
            [func("sequentialthinking", description="session contract")]
        ).decode()
        samples = [
            conv(messages=[user("u")], raw=first_raw),
            conv(messages=[user("v")], raw=second_raw),
        ]

        kept, drops, _ = _apply_dataset_config(
            samples, ToucanConfig(drop_conflicting_duplicate_tools=True)
        )

        assert kept == samples
        assert drops == {}

    def test_identical_same_row_duplicates_are_allowed_and_not_deduplicated(
        self,
    ) -> None:
        duplicate = func("get_weather", description="same contract")
        raw = qrow()
        raw["available_tools"] = orjson.dumps([duplicate, duplicate]).decode()
        sample = conv(
            messages=[user("u"), assistant(content="done")],
            tools=[copy.deepcopy(duplicate), copy.deepcopy(duplicate)],
            raw=raw,
        )
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config(
            [sample], ToucanConfig(drop_conflicting_duplicate_tools=True)
        )

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.tools == original_tools
        assert len(sample.tools) == 2

    def test_real_oss_sequential_name_collision_is_dropped_before_transform(
        self,
    ) -> None:
        # Production UUID 7e196aff-06a9-55d3-b316-b038a029ec5b contains the
        # reference and Waldzell contracts under this one visible name. The
        # reduced definitions pin the ambiguity and drop/transform ordering.
        raw = qrow()
        raw["available_tools"] = orjson.dumps(
            [
                func("sequentialthinking", description="reference contract"),
                func("sequentialthinking", description="Waldzell session contract"),
            ]
        ).decode()
        sample = conv(
            tools=[
                func("sequentialthinking", description="reference contract"),
                func("sequentialthinking", description="Waldzell session contract"),
            ],
            messages=[
                assistant(tool_calls=[call(name="sequentialthinking")]),
                tool_response(content='{"thoughtHistoryLength":1}'),
            ],
            raw=raw,
        )
        sample.sample_id = "7e196aff-06a9-55d3-b316-b038a029ec5b"

        kept, drops, transforms = _apply_dataset_config(
            [sample], ToucanConfig(drop_conflicting_duplicate_tools=True)
        )

        assert kept == []
        assert drops == {"conflicting_duplicate_tools": 1}
        assert transforms == {}

    def test_strip_reasoning_tools_is_a_transform_on_survivors(self) -> None:
        s = conv(
            tools=[func("chain-of-draft"), func("get_weather")],
            messages=[
                assistant(
                    tool_calls=[
                        call(name="chain-of-draft"),
                        call(name="get_weather"),
                    ]
                ),
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

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "events"),
        [
            (
                "e5ba7254-e907-5f46-8918-270e6f2cb087",
                "Kimi-K2",
                [
                    (
                        "think-tool-get_thought_stats",
                        "No thoughts have been recorded yet.",
                    ),
                    (
                        "think-tool-get_thoughts",
                        "No thoughts have been recorded yet.",
                    ),
                    ("think-tool-clear_thoughts", "Cleared 0 recorded thoughts."),
                ],
            ),
            (
                "400a283d-1ae2-5623-85d9-d39eecbfbe5a",
                "Qwen3",
                [
                    ("think-tool-server-think", "OUTLINE AGILE AND WATERFALL"),
                    ("think-tool-server-think", "COMPARE PROS AND CONS"),
                    ("think-tool-server-think", "ANALYZE PROJECT CONTEXT"),
                    ("think-tool-server-think", "CREATE A DECISION MATRIX"),
                    (
                        "think-tool-server-get_thoughts",
                        "Thought #1: OUTLINE ... Thought #4: DECISION MATRIX",
                    ),
                ],
            ),
            (
                "783ba926-364b-51af-bde8-34c655e88ebf",
                "OSS",
                [
                    (
                        "get_thought_stats",
                        '{"type":"text","text":"No thoughts have been recorded yet."}',
                    ),
                    (
                        "get_thoughts",
                        '{"type":"text","text":"No thoughts have been recorded yet."}',
                    ),
                ],
            ),
            (
                "1c8500a5-35bd-5cbc-853d-020b10e11706",
                "OSS",
                [
                    (
                        "think",
                        '{"type":"text","text":"Thought recorded: solution requires A and B"}',
                    ),
                    (
                        "get_thought_stats",
                        '{"average_length":25,"total_thoughts":1}',
                    ),
                ],
            ),
            (
                "78ace17d-45dc-5eab-97bc-42d6a953f49f",
                "Qwen3",
                [("think-tool-clear_thoughts", "Cleared 0 recorded thoughts.")],
            ),
        ],
    )
    def test_real_thought_state_episodes_are_preserved(
        self, sample_id: str, teacher: str, events: list[tuple[str, str]]
    ) -> None:
        """Exact call order and representative observations from five raw rows."""

        names = [name for name, _ in events]
        messages = []
        for i, (name, observation) in enumerate(events):
            messages.extend(
                [
                    assistant(tool_calls=[call(name=name)]),
                    tool_response(content=observation, tool_call_id=str(i)),
                ]
            )
        messages.append(assistant(content="final answer grounded in the observations"))

        s = conv(
            tools=[func(name) for name in dict.fromkeys(names)],
            messages=messages,
            raw=qrow(),
        )
        s.sample_id = sample_id
        s.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        original_messages = copy.deepcopy(s.messages)

        kept, _, transforms = _apply_dataset_config([s], ToucanConfig())

        assert transforms == {}
        assert kept[0].sample_id == sample_id
        assert kept[0].messages == original_messages
        assert [t["function"]["name"] for t in kept[0].tools] == list(
            dict.fromkeys(names)
        )

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "events"),
        [
            (
                "15581dd7-8b4b-53a5-b3aa-128fee7e3e97",
                "Kimi-K2",
                [
                    (
                        "pentestthinking-pentestthinkingMCP",
                        '{"nodeId":"root","score":1,"strategyUsed":"beam_search",'
                        '"stats":{"totalNodes":0}}',
                    )
                ],
            ),
            (
                "aed80ff3-6634-5d28-b5fc-23067798dc4c",
                "Kimi-K2",
                [
                    (
                        "pentestthinking-pentestthinkingMCP",
                        '{"nodeId":"root-1754716755174",'
                        '"score":0.6727749999999996,"strategyUsed":"mcts",'
                        '"stats":{"totalNodes":0}}',
                    )
                ],
            ),
            (
                "89f8b724-009b-5740-a2d9-293f05ee9486",
                "OSS",
                [
                    (
                        "pentestthinkingMCP",
                        '{"nodeId":"root","score":0.8,"strategyUsed":"beam_search"}',
                    )
                ],
            ),
            (
                "1b6538ad-ffd4-52a2-a464-91509274bffa",
                "Kimi-K2",
                [
                    (
                        "game-engine-server-gamedesignthinking",
                        '{"error":"m.toLowerCase is not a function","status":"failed"}',
                    ),
                    (
                        "game-engine-server-gamedesignthinking",
                        '{"components":[],"libraries":[],'
                        '"gameTitle":"Gravity Flip",'
                        '"thoughtHistoryLength":1,"isComplete":false}',
                    ),
                ],
            ),
            (
                "03beff5b-f4d9-5622-a8bc-12d2fa2d400b",
                "OSS",
                [
                    (
                        "gamedesignthinking",
                        '{"components":["overview"],'
                        '"libraries":["threejs, cannonjs"],'
                        '"thoughtHistoryLength":1,"isComplete":true}',
                    )
                ],
            ),
            (
                "b70a6243-1bc3-519e-a3fc-b18c3dc17e79",
                "OSS",
                [
                    (
                        "skiaanimationthinking",
                        '{"branches":[],"thoughtHistoryLength":1,'
                        '"codeSnippetProvided":false}',
                    ),
                    (
                        "skiaanimationthinking",
                        '{"branches":[],"thoughtHistoryLength":2,'
                        '"codeSnippetProvided":true}',
                    ),
                ],
            ),
            (
                "e56466ce-7363-55dc-9389-e45cf35cc5af",
                "Qwen3",
                [
                    (
                        "react-native-skia-animation-thinking-tool-"
                        "skiaanimationthinking",
                        '{"branches":[],"thoughtHistoryLength":1,'
                        '"codeSnippetProvided":false}',
                    )
                ],
            ),
        ],
    )
    def test_real_domain_thinking_episodes_are_preserved(
        self, sample_id: str, teacher: str, events: list[tuple[str, str]]
    ) -> None:
        """Production-derived observation reductions from seven accepted rows."""

        names = [name for name, _ in events]
        messages = []
        for index, (name, observation) in enumerate(events):
            messages.extend(
                [
                    assistant(tool_calls=[call(name=name)]),
                    tool_response(content=observation, tool_call_id=str(index)),
                ]
            )
        messages.append(assistant(content="final answer after the explicit protocol"))

        sample = conv(
            tools=[func(name) for name in dict.fromkeys(names)],
            messages=messages,
            raw=qrow(),
        )
        sample.sample_id = sample_id
        sample.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert [tool["function"]["name"] for tool in sample.tools] == list(
            dict.fromkeys(names)
        )

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "events"),
        [
            (
                "1a30f26f-c05c-5517-a26b-6af0c1611cdd",
                "Kimi-K2",
                [
                    (
                        "lotus-wisdom-lotuswisdom",
                        '{"status":"processing","currentStep":"open",'
                        '"journey":"open","processLength":1}',
                    ),
                    (
                        "lotus-wisdom-lotuswisdom",
                        '{"status":"MEDITATION_COMPLETE","duration":3,'
                        '"prompt":"What insights emerged during the pause?",'
                        '"journey":"open → examine → meditate"}',
                    ),
                    (
                        "lotus-wisdom-lotuswisdom_summary",
                        '{"journeyLength":5,"domainJourney":"process_flow → '
                        'meta_cognitive → meditation → non_dual_recognition",'
                        '"steps":[{"tag":"open","stepNumber":1}]}',
                    ),
                ],
            ),
            (
                "38321102-d59c-50a9-a91c-3ff3e96b63a9",
                "OSS",
                [
                    (
                        "lotuswisdom",
                        '{"status":"WISDOM_READY","processComplete":true,'
                        '"finalStep":"express","journeyLength":5,'
                        '"finalJourney":"open → examine → recognize → integrate → '
                        'express"}',
                    ),
                    (
                        "lotuswisdom_summary",
                        '{"journeyLength":5,"steps":['
                        '{"tag":"open","domain":"process_flow","stepNumber":1}]}',
                    ),
                ],
            ),
            (
                "43caabce-276f-5efc-9ab9-b6fec5357a65",
                "Qwen3",
                [
                    (
                        "lotus-wisdom-lotuswisdom",
                        '{"status":"WISDOM_READY","processComplete":true,'
                        '"finalStep":"complete","journeyLength":5,'
                        '"instruction":"PROCESS_COMPLETE_SPEAK_WISDOM"}',
                    )
                ],
            ),
            (
                "6b790412-5e05-5de8-8726-fbd7e545a4af",
                "Qwen3",
                [
                    (
                        "lotus-wisdom-lotuswisdom",
                        '{"error":"Invalid tag: skillful_means. Must be one of: '
                        "upaya, expedient, direct, gradual, sudden, recognize, "
                        "transform, integrate, transcend, embody, examine, reflect, "
                        'verify, refine, complete, open, engage, express, meditate",'
                        '"status":"failed"}',
                    ),
                    (
                        "lotus-wisdom-lotuswisdom",
                        '{"status":"processing","currentStep":"upaya",'
                        '"journey":"open → examine → upaya","processLength":3}',
                    ),
                ],
            ),
        ],
    )
    def test_real_lotus_wisdom_episodes_are_preserved(
        self, sample_id: str, teacher: str, events: list[tuple[str, str]]
    ) -> None:
        """Production-derived reductions cover both namespaces and all results."""

        names = [name for name, _ in events]
        messages = []
        for index, (name, observation) in enumerate(events):
            messages.extend(
                [
                    assistant(tool_calls=[call(name=name)]),
                    tool_response(content=observation, tool_call_id=str(index)),
                ]
            )
        messages.append(assistant(content="final answer after the Lotus protocol"))
        sample = conv(
            tools=[func(name) for name in dict.fromkeys(names)],
            messages=messages,
            raw=qrow(),
        )
        sample.sample_id = sample_id
        sample.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_lotus_turn_local_state_reset_observations_are_preserved(self) -> None:
        # Production UUID bc3ea0e4-6ad0-57b0-98f4-ee6ccb2278d7 exposes the
        # observed harness boundary: the second user turn's corrected `engage`
        # call returns a one-step journey instead of inheriting turn 1 state.
        name = "lotus-wisdom-lotuswisdom"
        sample = conv(
            tools=[func(name)],
            messages=[
                user("first inquiry"),
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"status":"processing","journey":"open → examine → '
                    'recognize → meditate → integrate","processLength":5}'
                ),
                assistant(content="first answer"),
                user("follow-up inquiry"),
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"error":"Invalid tag: engaging. Must be one of: '
                    "upaya, expedient, direct, gradual, sudden, recognize, transform, "
                    "integrate, transcend, embody, examine, reflect, verify, refine, "
                    'complete, open, engage, express, meditate","status":"failed"}'
                ),
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"status":"processing","currentStep":"engage",'
                    '"journey":"engage","processLength":1}'
                ),
                assistant(content="second answer"),
            ],
            raw=qrow(subset="multi-turn"),
        )
        sample.sample_id = "bc3ea0e4-6ad0-57b0-98f4-ee6ccb2278d7"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "name", "events"),
        [
            (
                "bd8f707c-8bd8-5f3d-bb30-4e343c9269b6",
                "Kimi-K2",
                "structured-argumentation-server-structuredArgumentation",
                [
                    (
                        {
                            "claim": "Automated crisis allocation is faster",
                            "premises": ["Delay may cost lives"],
                            "conclusion": "Use automated support",
                            "argumentType": "thesis",
                            "confidence": 0.8,
                            "nextArgumentNeeded": True,
                            "suggestedNextTypes": ["antithesis"],
                        },
                        '{"argumentId":"arg-1","argumentType":"thesis",'
                        '"nextArgumentNeeded":true,"suggestedNextTypes":'
                        '["antithesis"],"argumentHistoryLength":1,'
                        '"relationshipCount":1}',
                    ),
                    (
                        {
                            "claim": "Automation lacks human judgment",
                            "premises": ["Context affects ethical choices"],
                            "conclusion": "Do not automate final authority",
                            "argumentType": "antithesis",
                            "confidence": 0.75,
                            "respondsTo": "arg-1",
                            "contradicts": ["arg-1"],
                            "nextArgumentNeeded": True,
                            "suggestedNextTypes": ["objection", "synthesis"],
                        },
                        '{"argumentId":"arg-2","argumentType":"antithesis",'
                        '"nextArgumentNeeded":true,"suggestedNextTypes":'
                        '["objection","synthesis"],"argumentHistoryLength":2,'
                        '"relationshipCount":2}',
                    ),
                ],
            ),
            (
                "0856dfe8-eb92-5c60-af55-9e60b27e9888",
                "OSS",
                "structuredArgumentation",
                [
                    (
                        {
                            "claim": "Use daily stand-ups",
                            "premises": ["They surface blockers"],
                            "conclusion": "Alignment improves",
                            "argumentId": "A1",
                            "argumentType": "thesis",
                            "confidence": 0.9,
                            "nextArgumentNeeded": True,
                            "suggestedNextTypes": ["antithesis"],
                        },
                        '{"argumentId":"A1","argumentType":"thesis",'
                        '"nextArgumentNeeded":true,"suggestedNextTypes":'
                        '["antithesis"],"argumentHistoryLength":1,'
                        '"relationshipCount":1}',
                    ),
                    (
                        {
                            "claim": "Stand-ups can waste time",
                            "premises": ["They can repeat task-board data"],
                            "conclusion": "Strict focus is required",
                            "argumentId": "A2",
                            "argumentType": "antithesis",
                            "confidence": 0.85,
                            "respondsTo": "A1",
                            "nextArgumentNeeded": True,
                            "suggestedNextTypes": ["synthesis"],
                        },
                        '{"argumentId":"A2","argumentType":"antithesis",'
                        '"nextArgumentNeeded":true,"suggestedNextTypes":'
                        '["synthesis"],"argumentHistoryLength":2,'
                        '"relationshipCount":2}',
                    ),
                ],
            ),
            (
                "c65c17c5-65f7-55e5-a06a-a390c8c275b6",
                "Qwen3",
                "clear-thought-server-structuredargumentation",
                [
                    (
                        {
                            "claim": "Centralized logs improve correlation",
                            "premises": ["Timestamps need synchronization"],
                            "conclusion": "Centralize and synchronize logs",
                            "argumentType": "thesis",
                            "confidence": 0.95,
                            "nextArgumentNeeded": True,
                        },
                        '{"argumentType":"thesis","claim":"Centralized logs '
                        'improve correlation","confidence":0.95,'
                        '"nextArgumentNeeded":true,'
                        '"argumentId":"arg-1756620618128","status":"success"}',
                    ),
                    (
                        {
                            "claim": "Time synchronization is critical",
                            "premises": ["Unsynchronized logs mislead"],
                            "conclusion": "Use a shared time reference",
                            "argumentType": "objection",
                            "confidence": 0.9,
                            "respondsTo": "arg-1756620618128",
                            "nextArgumentNeeded": True,
                        },
                        '{"argumentType":"objection","claim":"Time '
                        'synchronization is critical","confidence":0.9,'
                        '"nextArgumentNeeded":true,'
                        '"argumentId":"arg-1756620623244","status":"success"}',
                    ),
                ],
            ),
            (
                "a5662ed5-c9ac-5ed0-adc9-123902c84138",
                "OSS",
                "structuredargumentation",
                [
                    (
                        {
                            "claim": "Green zoning improves resilience",
                            "premises": ["Green infrastructure reduces runoff"],
                            "conclusion": "Adopt the ordinance",
                            "argumentId": "arg1",
                            "argumentType": "thesis",
                            "confidence": 0.92,
                            "nextArgumentNeeded": True,
                        },
                        '{"argumentType":"thesis","claim":"Green zoning '
                        'improves resilience","confidence":0.92,'
                        '"nextArgumentNeeded":true,"argumentId":"arg1",'
                        '"status":"success"}',
                    ),
                    (
                        {
                            "claim": "Green zoning can raise costs",
                            "premises": ["Requirements reduce buildable area"],
                            "conclusion": "Affordability may worsen",
                            "argumentId": "arg2",
                            "argumentType": "antithesis",
                            "respondsTo": "arg1",
                            "confidence": 0.85,
                            "nextArgumentNeeded": True,
                        },
                        '{"argumentType":"antithesis","claim":"Green zoning '
                        'can raise costs","confidence":0.85,'
                        '"nextArgumentNeeded":true,"argumentId":"arg2",'
                        '"status":"success"}',
                    ),
                ],
            ),
            (
                "5804fe49-69d3-5b6f-83e0-ee146b99dd86",
                "Qwen3",
                "clear-thought-structuredargumentation",
                [
                    (
                        {
                            "claim": "The proposed project is most effective",
                            "premises": ["It addresses the core issue"] * 4,
                            "conclusion": "Approve and fund it",
                            "argumentType": "deductive",
                            "confidence": 0.9,
                            "nextArgumentNeeded": True,
                        },
                        '{"claim":"The proposed project is most effective",'
                        '"premisesCount":4,"argumentType":"deductive",'
                        '"confidence":0.9,"nextArgumentNeeded":true,'
                        '"status":"success","sessionContext":{'
                        '"sessionId":"25a01a8e-cac5-4b9f-ba93-24ba28689c69",'
                        '"totalOperations":1,"creativeStoreStats":{'
                        '"totalSessions":1,"activeSessions":1,"totalIdeas":4,'
                        '"totalInsights":1,"totalConnections":0,'
                        '"averageIdeasPerSession":4,"uniqueTechniques":1,'
                        '"techniqueUsage":{"socratic-method":1},'
                        '"topTechniques":[{"technique":"socratic-method",'
                        '"count":1}]}}}',
                    ),
                ],
            ),
        ],
    )
    def test_real_structured_argumentation_episodes_are_preserved(
        self,
        sample_id: str,
        teacher: str,
        name: str,
        events: list[tuple[dict, str]],
    ) -> None:
        """Production reductions cover all five exact identities and lineages."""

        messages = []
        for index, (arguments, observation) in enumerate(events):
            messages.extend(
                [
                    assistant(
                        tool_calls=[
                            call(
                                name=name,
                                arguments=orjson.dumps(arguments).decode(),
                            )
                        ]
                    ),
                    tool_response(content=observation, tool_call_id=str(index)),
                ]
            )
        messages.append(assistant(content="final answer after argumentation"))
        sample = conv(tools=[func(name)], messages=messages, raw=qrow())
        sample.sample_id = sample_id
        sample.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_structured_argumentation_user_turn_state_reset_is_preserved(self) -> None:
        # Production UUID bd8f707c-8bd8-5f3d-bb30-4e343c9269b6 ends its first
        # user turn at history length 8, then starts the next turn at 1. A row-
        # global replay would incorrectly predict 9; the Toucan harness boundary
        # is therefore part of the observed episode semantics.
        name = "structured-argumentation-server-structuredArgumentation"
        sample = conv(
            tools=[func(name)],
            messages=[
                user("first issue"),
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"argumentId":"arg-8",'
                    '"argumentHistoryLength":8,"relationshipCount":8}'
                ),
                assistant(content="first answer"),
                user("follow-up issue"),
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"argumentId":"arg-9",'
                    '"argumentHistoryLength":1,"relationshipCount":1}'
                ),
                assistant(content="follow-up answer"),
            ],
            raw=qrow(subset="multi-turn"),
        )
        sample.sample_id = "bd8f707c-8bd8-5f3d-bb30-4e343c9269b6"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_structured_argumentation_error_and_retry_are_preserved(self) -> None:
        # Production UUID 634e88ab-84bd-5cd4-9120-4c6668e165a4 returns this
        # validation error when nextArgumentNeeded is absent, then continues the
        # argumentation episode. The dataset-specific transform must not erase the
        # evidence; universal argument validation can reject the row later.
        name = "structured-argumentation-server-structuredArgumentation"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments='{"claim":"bounded analogy",'
                            '"premises":["formal similarity"],'
                            '"conclusion":"use cautiously",'
                            '"argumentType":"synthesis","confidence":0.88}',
                        )
                    ]
                ),
                tool_response(
                    content='{"error":"Invalid nextArgumentNeeded: must be a '
                    'boolean","status":"failed"}'
                ),
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments='{"claim":"bounded analogy",'
                            '"premises":["formal similarity"],'
                            '"conclusion":"use cautiously",'
                            '"argumentType":"synthesis","confidence":0.88,'
                            '"nextArgumentNeeded":false}',
                        )
                    ]
                ),
                tool_response(
                    content='{"argumentId":"arg-3",'
                    '"argumentHistoryLength":3,"relationshipCount":3,'
                    '"nextArgumentNeeded":false}'
                ),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        sample.sample_id = "634e88ab-84bd-5cd4-9120-4c6668e165a4"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_exact_structured_name_with_unparseable_arguments_is_not_hidden(
        self,
    ) -> None:
        name = "structured-argumentation-server-structuredArgumentation"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name, arguments='{"claim":')]),
                tool_response(content="Malformed tool call"),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_terminal_next_argument_needed_true_episode_is_not_stripped(self) -> None:
        # Production UUID bbfdfe34-252a-579d-a226-52d6d61e5449 ends on this tool
        # result. Production's incomplete-termination drop handles that separate
        # quality issue; reasoning stripping must not rewrite the source episode.
        name = "structured-argumentation-server-structuredArgumentation"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"argumentId":"angels_objection_01",'
                    '"argumentType":"objection","nextArgumentNeeded":true,'
                    '"suggestedNextTypes":["rebuttal"],'
                    '"argumentHistoryLength":4,"relationshipCount":4}'
                ),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

        dropped, drop_reasons, _ = _apply_dataset_config(
            [copy.deepcopy(sample)],
            ToucanConfig(drop_incomplete_termination=True),
        )
        assert dropped == []
        assert drop_reasons == {"incomplete_termination": 1}

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "name", "events"),
        [
            (
                "68074f1d-535b-5fac-a2ae-7191d5a1b5e9",
                "Kimi-K2",
                "analogical-reasoning-server-analogicalReasoning",
                [
                    (
                        {
                            "sourceDomain": {
                                "name": "Ecosystem Sustainability",
                                "elements": [],
                            },
                            "targetDomain": {
                                "name": "Urban Economic Sustainability",
                                "elements": [],
                            },
                            "mappings": [{}] * 7,
                            "analogyId": "ecosystem-urban-economics-043",
                            "purpose": "creative-generation",
                            "confidence": 0.85,
                            "iteration": 1,
                            "nextOperationNeeded": True,
                            "suggestedOperations": [
                                "draw-inference",
                                "evaluate-limitation",
                            ],
                        },
                        '{"analogyId":"ecosystem-urban-economics-043",'
                        '"purpose":"creative-generation","iteration":1,'
                        '"sourceDomain":"Ecosystem Sustainability",'
                        '"targetDomain":"Urban Economic Sustainability",'
                        '"mappingCount":7,"inferenceCount":0,'
                        '"nextOperationNeeded":true,"suggestedOperations":'
                        '["draw-inference","evaluate-limitation"]}',
                    ),
                    (
                        {
                            "sourceDomain": {
                                "name": "Ecosystem Sustainability",
                                "elements": [],
                            },
                            "targetDomain": {
                                "name": "Urban Economic Sustainability",
                                "elements": [],
                            },
                            "inferences": [{}] * 5,
                            "analogyId": "ecosystem-urban-economics-043",
                            "purpose": "creative-generation",
                            "confidence": 0.9,
                            "iteration": 2,
                            "nextOperationNeeded": True,
                            "suggestedOperations": ["evaluate-limitation"],
                        },
                        '{"analogyId":"ecosystem-urban-economics-043",'
                        '"purpose":"creative-generation","iteration":2,'
                        '"sourceDomain":"Ecosystem Sustainability",'
                        '"targetDomain":"Urban Economic Sustainability",'
                        '"mappingCount":0,"inferenceCount":5,'
                        '"nextOperationNeeded":true,"suggestedOperations":'
                        '["evaluate-limitation"]}',
                    ),
                ],
            ),
            (
                "b2201cd0-2e19-5d94-bc97-2d5c8d70b701",
                "OSS",
                "analogicalReasoning",
                [
                    (
                        {
                            "sourceDomain": {
                                "name": "Library Committee",
                                "elements": [],
                            },
                            "targetDomain": {
                                "name": "Blockchain Network",
                                "elements": [],
                            },
                            "mappings": [{}] * 5,
                            "inferences": [{}],
                            "analogyId": "AN001",
                            "purpose": "explanation",
                            "confidence": 0.9,
                            "iteration": 1,
                            "nextOperationNeeded": False,
                        },
                        '{"analogyId":"AN001","purpose":"explanation",'
                        '"iteration":1,"sourceDomain":"Library Committee",'
                        '"targetDomain":"Blockchain Network",'
                        '"mappingCount":5,"inferenceCount":1,'
                        '"nextOperationNeeded":false}',
                    )
                ],
            ),
            (
                "ee728302-569f-51f2-a716-cc6a19047d8c",
                "Qwen3",
                "analogical-reasoning-server-analogicalReasoning",
                [
                    (
                        {
                            "sourceDomain": {
                                "name": "Telecommunication Network",
                                "elements": [],
                            },
                            "targetDomain": {
                                "name": "Quantum Entanglement",
                                "elements": [],
                            },
                            "mappings": [{}] * 3,
                            "inferences": [{}, {}],
                            "analogyId": "QE-Telecom",
                            "purpose": "explanation",
                            "confidence": 0.8,
                            "iteration": 1,
                            "nextOperationNeeded": True,
                            "suggestedOperations": [
                                "add-mapping",
                                "revise-mapping",
                                "draw-inference",
                            ],
                        },
                        '{"analogyId":"QE-Telecom","purpose":"explanation",'
                        '"iteration":1,'
                        '"sourceDomain":"Telecommunication Network",'
                        '"targetDomain":"Quantum Entanglement",'
                        '"mappingCount":3,"inferenceCount":2,'
                        '"nextOperationNeeded":true,"suggestedOperations":'
                        '["add-mapping","revise-mapping","draw-inference"]}',
                    )
                ],
            ),
        ],
    )
    def test_real_analogical_reasoning_episodes_are_preserved(
        self,
        sample_id: str,
        teacher: str,
        name: str,
        events: list[tuple[dict, str]],
    ) -> None:
        """Production reductions cover both exact identities and all teachers."""

        messages = []
        for index, (arguments, observation) in enumerate(events):
            call_id = f"analogical-{index}"
            messages.extend(
                [
                    assistant(
                        tool_calls=[
                            call(
                                name=name,
                                arguments=orjson.dumps(arguments).decode(),
                                call_id=call_id,
                            )
                        ]
                    ),
                    tool_response(content=observation, tool_call_id=call_id),
                ]
            )
        messages.append(assistant(content="final answer after the analogy"))
        sample = conv(tools=[func(name)], messages=messages, raw=qrow())
        sample.sample_id = sample_id
        sample.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_analogical_runtime_generated_element_ids_are_not_hidden(self) -> None:
        # Production UUID 4cf95823-27f1-5365-bcd8-e38ec0ca867c omitted some
        # element IDs. The server mutates those elements by assigning IDs before
        # returning this iteration-2 summary, so the call is not a pure echo.
        name = "analogical-reasoning-server-analogicalReasoning"
        arguments = {
            "sourceDomain": {
                "name": "Supply Chain Logistics",
                "elements": [
                    {
                        "name": "Distribution hub",
                        "type": "entity",
                        "description": "Routes material through the network",
                    }
                ],
            },
            "targetDomain": {
                "name": "Neural Networks",
                "elements": [
                    {
                        "id": "neuron",
                        "name": "Neuron",
                        "type": "entity",
                        "description": "Routes signals through a network",
                    }
                ],
            },
            "mappings": [{}] * 10,
            "inferences": [{}, {}],
            "analogyId": "enhanced-supply-chain-neural-analogy",
            "purpose": "explanation",
            "confidence": 0.9,
            "iteration": 2,
            "nextOperationNeeded": False,
        }
        observation = (
            '{"analogyId":"enhanced-supply-chain-neural-analogy",'
            '"purpose":"explanation","iteration":2,'
            '"sourceDomain":"Supply Chain Logistics",'
            '"targetDomain":"Neural Networks","mappingCount":10,'
            '"inferenceCount":2,"nextOperationNeeded":false}'
        )
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(
                    tool_calls=[
                        call(name=name, arguments=orjson.dumps(arguments).decode())
                    ]
                ),
                tool_response(content=observation),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        sample.sample_id = "4cf95823-27f1-5365-bcd8-e38ec0ca867c"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_analogical_server_validation_error_and_retry_are_preserved(self) -> None:
        # Production UUID 78f8abdf-d849-593d-915d-549cc4f32f4d first omitted
        # required element descriptions, then retried. Both the error evidence
        # and the later successful call must remain visible to validators.
        name = "analogical-reasoning-server-analogicalReasoning"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments=orjson.dumps(
                                {
                                    "sourceDomain": {
                                        "name": "Airbnb",
                                        "elements": [
                                            {
                                                "id": "supply",
                                                "name": "Host supply",
                                                "type": "entity",
                                            }
                                        ],
                                    },
                                    "targetDomain": {
                                        "name": "Videographer Marketplace",
                                        "elements": [],
                                    },
                                    "analogyId": "EventVid-Airbnb-001",
                                    "purpose": "explanation",
                                    "confidence": 0.8,
                                    "iteration": 1,
                                    "nextOperationNeeded": True,
                                }
                            ).decode(),
                        )
                    ]
                ),
                tool_response(
                    content='{"error":"Invalid element description for '
                    'element supply: must be a string","status":"failed"}'
                ),
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments=orjson.dumps(
                                {
                                    "sourceDomain": {
                                        "name": "Airbnb",
                                        "elements": [],
                                    },
                                    "targetDomain": {
                                        "name": "Videographer Marketplace",
                                        "elements": [],
                                    },
                                    "analogyId": "EventVid-Airbnb-001",
                                    "purpose": "explanation",
                                    "confidence": 0.8,
                                    "iteration": 1,
                                    "nextOperationNeeded": True,
                                    "suggestedOperations": ["draw-inference"],
                                }
                            ).decode(),
                        )
                    ]
                ),
                tool_response(
                    content='{"analogyId":"EventVid-Airbnb-001",'
                    '"purpose":"explanation","iteration":1,'
                    '"sourceDomain":"Airbnb",'
                    '"targetDomain":"Videographer Marketplace",'
                    '"mappingCount":0,"inferenceCount":0,'
                    '"nextOperationNeeded":true,'
                    '"suggestedOperations":["draw-inference"]}'
                ),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        sample.sample_id = "78f8abdf-d849-593d-915d-549cc4f32f4d"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_analogical_harness_decode_error_and_retry_are_preserved(self) -> None:
        # Production UUID bbcb911b-e576-5d3a-9562-32c5ed106baa contains an
        # invalid JSON call followed by a successful corrected call.
        name = "analogical-reasoning-server-analogicalReasoning"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments='{"sourceDomain":{"name":"Water",',
                        )
                    ]
                ),
                tool_response(
                    content="An error occurred when calling tool "
                    "`analogical-reasoning-server-analogicalReasoning`: "
                    "JSONDecodeError: Expecting value"
                ),
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments=orjson.dumps(
                                {
                                    "sourceDomain": {
                                        "name": "Water Wave Phenomena",
                                        "elements": [],
                                    },
                                    "targetDomain": {
                                        "name": "Quantum Superposition",
                                        "elements": [],
                                    },
                                    "mappings": [{}] * 7,
                                    "inferences": [{}, {}, {}],
                                    "analogyId": (
                                        "quantum_superposition_water_waves_v1"
                                    ),
                                    "purpose": "explanation",
                                    "confidence": 0.75,
                                    "iteration": 1,
                                    "nextOperationNeeded": False,
                                }
                            ).decode(),
                        )
                    ]
                ),
                tool_response(
                    content='{"analogyId":'
                    '"quantum_superposition_water_waves_v1",'
                    '"purpose":"explanation","iteration":1,'
                    '"sourceDomain":"Water Wave Phenomena",'
                    '"targetDomain":"Quantum Superposition",'
                    '"mappingCount":7,"inferenceCount":3,'
                    '"nextOperationNeeded":false}'
                ),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        sample.sample_id = "bbcb911b-e576-5d3a-9562-32c5ed106baa"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_analogical_nested_string_argument_error_and_retry_are_preserved(
        self,
    ) -> None:
        # Production UUID d673405f-d069-5a06-b1a5-a602f71da332 passes a JSON
        # string where the MCP client requires a dictionary, then retries with a
        # dictionary. This harness-level failure must not be transformed away.
        name = "analogical-reasoning-server-analogicalReasoning"
        valid_arguments = {
            "sourceDomain": {"name": "Marathon Running", "elements": []},
            "targetDomain": {"name": "Company Profitability", "elements": []},
            "mappings": [{}, {}, {}],
            "inferences": [{}],
            "analogyId": "profitability-marathon",
            "purpose": "explanation",
            "confidence": 0.75,
            "iteration": 0,
            "nextOperationNeeded": True,
            "suggestedOperations": [
                "add-mapping",
                "revise-mapping",
                "draw-inference",
            ],
        }
        nested_string = orjson.dumps(orjson.dumps(valid_arguments).decode()).decode()
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name, arguments=nested_string)]),
                tool_response(
                    content="ValidationError: arguments Input should be a valid "
                    "dictionary"
                ),
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments=orjson.dumps(valid_arguments).decode(),
                        )
                    ]
                ),
                tool_response(
                    content='{"analogyId":"profitability-marathon",'
                    '"purpose":"explanation","iteration":0,'
                    '"sourceDomain":"Marathon Running",'
                    '"targetDomain":"Company Profitability",'
                    '"mappingCount":3,"inferenceCount":1,'
                    '"nextOperationNeeded":true,"suggestedOperations":'
                    '["add-mapping","revise-mapping","draw-inference"]}'
                ),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        sample.sample_id = "d673405f-d069-5a06-b1a5-a602f71da332"
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_undefined_exact_analogical_call_is_marked_but_not_hidden(self) -> None:
        # Production UUID 01be4095-2d51-52e2-a87a-8e55d2ccf82f calls the exact
        # qualified identity without advertising that definition. The marker
        # records the explicit trace; it does not waive definedness validation.
        name = "analogical-reasoning-server-analogicalReasoning"
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,), defined_names=("get_weather",)),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [ANALOGICAL_REASONING_TOOL_FAMILY]
        }

    @pytest.mark.parametrize("name", EXPECTED_ANALOGICAL_REASONING_TOOL_NAMES)
    def test_unused_exact_analogical_definition_is_retained_without_marker(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(
                called_names=("get_weather",),
                defined_names=(name, "get_weather"),
            ),
            "OSS",
        )
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.tools == original_tools
        assert sample.annotations == {}

    @pytest.mark.parametrize(
        "name",
        [
            "analogical-reasoning-server-analogialReasoning",
            "hotel-booking-server-analogicalReasoning",
        ],
    )
    def test_undefined_fuzzy_analogical_calls_are_not_family_members(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(
                called_names=(name,),
                defined_names=("analogical-reasoning-server-analogicalReasoning",),
            ),
            "Qwen3",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {}

    def test_mixed_parallel_analogical_and_ordinary_calls_remain_coherent(
        self,
    ) -> None:
        analogical = "analogicalReasoning"
        sample = conv(
            tools=[func(analogical), func("get_weather")],
            messages=[
                assistant(
                    tool_calls=[
                        call(name=analogical, call_id="analogy"),
                        call(name="get_weather", call_id="weather"),
                    ]
                ),
                tool_response(
                    content='{"analogyId":"a","nextOperationNeeded":false}',
                    tool_call_id="analogy",
                ),
                tool_response(content="sunny", tool_call_id="weather"),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_unbalanced_exact_analogical_call_is_not_hidden(self) -> None:
        name = "analogicalReasoning"
        sample = conv(
            tools=[func(name)],
            messages=[assistant(tool_calls=[call(name=name)])],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_conflicting_analogical_definitions_drop_before_transformation(
        self,
    ) -> None:
        name = "analogicalReasoning"
        definitions = [
            func(name, description="definition A"),
            func(name, description="definition B"),
        ]
        raw = qrow()
        raw["available_tools"] = orjson.dumps(definitions).decode()
        sample = conv(
            tools=copy.deepcopy(definitions),
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="observation"),
                assistant(content="final answer"),
            ],
            raw=raw,
        )

        kept, drops, transforms = _apply_dataset_config(
            [sample], ToucanConfig(drop_conflicting_duplicate_tools=True)
        )

        assert kept == []
        assert drops == {"conflicting_duplicate_tools": 1}
        assert transforms == {}

    def test_terminal_analogical_continuation_true_is_not_stripped(self) -> None:
        name = "analogical-reasoning-server-analogicalReasoning"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"analogyId":"a-1",'
                    '"nextOperationNeeded":true,'
                    '"suggestedOperations":["draw-inference"]}'
                ),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

        dropped, drop_reasons, _ = _apply_dataset_config(
            [copy.deepcopy(sample)],
            ToucanConfig(drop_incomplete_termination=True),
        )
        assert dropped == []
        assert drop_reasons == {"incomplete_termination": 1}

    @pytest.mark.parametrize(
        "observation",
        [
            '{"analogyId":"a-1","purpose":"evaluation",'
            '"iteration":0.5,"sourceDomain":"A","targetDomain":"B",'
            '"mappingCount":0,"inferenceCount":0,'
            '"nextOperationNeeded":false}',
            '{"analogyId":"a-1","purpose":"explanation",'
            '"iteration":1,"sourceDomain":"A","targetDomain":"B",'
            '"mappingCount":0,"inferenceCount":0,'
            '"nextOperationNeeded":false,'
            '"samplingSummary":"future server-generated summary"}',
        ],
        ids=["runtime-schema-divergence", "future-sampling-summary"],
    )
    def test_analogical_success_variants_are_preserved_without_result_heuristics(
        self, observation: str
    ) -> None:
        # The first case is accepted by the audited runtime despite violating the
        # advertised purpose enum/integer intent. The second is compatible with
        # the later upstream sampling implementation but absent from this raw
        # snapshot. Neither uncertainty authorizes destructive rewriting.
        name = "analogicalReasoning"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content=observation),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "name", "arguments", "observation"),
        [
            (
                "419bd1d6-f450-5104-af40-b47d2f8b6a97",
                "Kimi-K2",
                "clear-thought-server-mentalmodel",
                {
                    "modelName": "first_principles",
                    "problem": "Evaluate a startup architecture decision",
                    "steps": ["Identify constraints", "Compare trade-offs"],
                    "reasoning": "Start from operational requirements.",
                    "conclusion": "Use the smallest reversible architecture.",
                },
                '{"modelName":"first_principles","status":"success",'
                '"hasSteps":true,"hasConclusion":true}',
            ),
            (
                "494f133f-ceac-529e-bc3d-bfad6e2feaea",
                "Kimi-K2",
                "clear-thought-server-mentalmodel",
                {
                    "modelName": "first_principles",
                    "problem": "Scale a collaborative whiteboard",
                    "steps": ["Measure WebSocket limits"],
                },
                '{"modelName":"first_principles","status":"success",'
                '"hasSteps":true,"hasConclusion":false}',
            ),
            (
                "74bbcbca-f35b-5f79-ade8-0ffeb60c35da",
                "OSS",
                "mentalmodel",
                {
                    "modelName": "first_principles",
                    "problem": "Choose an MVP architecture",
                    "steps": ["Identify goals", "Evaluate constraints"],
                    "reasoning": "Compare actual need against engineering cost.",
                    "conclusion": "Favor maintainability with scaling hooks.",
                },
                '{"modelName":"first_principles","status":"success",'
                '"hasSteps":true,"hasConclusion":true}',
            ),
            (
                "e7178321-e93c-5b11-9fcf-a929777ff425",
                "Qwen3",
                "clear-thought-mentalmodel",
                {
                    "modelName": "first_principles",
                    "problem": "Recurring project failures in the team",
                    "steps": ["Define the issue", "Identify gaps"],
                    "reasoning": "Break the problem into fundamental elements.",
                    "conclusion": "Address the foundational process gaps.",
                },
                '{"modelName":"first_principles","status":"success",'
                '"hasSteps":true,"hasConclusion":true,'
                '"sessionContext":{"sessionId":'
                '"4c4b897b-67cd-4b95-b8e0-939e4b0b6302",'
                '"totalMentalModels":1,"recentModels":['
                '{"modelName":"first_principles",'
                '"problem":"Recurring project failures in the team"}]}}',
            ),
        ],
    )
    def test_real_mental_model_success_contracts_are_preserved(
        self,
        sample_id: str,
        teacher: str,
        name: str,
        arguments: dict,
        observation: str,
    ) -> None:
        """Pin both source lineages, all exact names, and all teachers."""

        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(
                    tool_calls=[
                        call(name=name, arguments=orjson.dumps(arguments).decode())
                    ]
                ),
                tool_response(content=observation),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        sample.sample_id = sample_id
        sample.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_mental_model_harness_decode_error_and_retry_are_preserved(self) -> None:
        # Production UUID d65f0f37-b90a-54bf-afdb-2084cc1a5fd2 contains two
        # malformed calls before a later successful exact-name retry.
        name = "clear-thought-server-mentalmodel"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments='{"modelName":"pareto_principle",',
                        )
                    ]
                ),
                tool_response(
                    content="An error occurred when calling tool "
                    "`clear-thought-server-mentalmodel`: JSONDecodeError"
                ),
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments=orjson.dumps(
                                {
                                    "modelName": "pareto_principle",
                                    "problem": "Prioritize the critical work",
                                    "steps": ["Rank impact", "Check feasibility"],
                                    "conclusion": "Select the highest-impact set.",
                                }
                            ).decode(),
                        )
                    ]
                ),
                tool_response(
                    content='{"modelName":"pareto_principle",'
                    '"status":"success","hasSteps":true,'
                    '"hasConclusion":true}'
                ),
                assistant(content="final answer"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_mental_model_failed_status_without_retry_is_preserved(self) -> None:
        name = "mentalmodel"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments=orjson.dumps(
                                {"modelName": "", "problem": "Choose an option"}
                            ).decode(),
                        )
                    ]
                ),
                tool_response(
                    content='{"error":"Invalid modelName: must be a string",'
                    '"status":"failed"}'
                ),
                assistant(content="I could not apply the requested model."),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

    def test_undefined_exact_mental_model_call_is_marked_but_not_hidden(
        self,
    ) -> None:
        # Three pinned rows use this exact audited spelling without advertising
        # it. The marker records the attempted family call; defined-function
        # validation remains responsible for rejecting the episode.
        name = "clear-thought-mentalmodel"
        sample, _ = _convert_sample(
            conversion_row(
                called_names=(name,),
                defined_names=("clear-thought-server-mentalmodel",),
            ),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [MENTAL_MODEL_TOOL_FAMILY]
        }

    @pytest.mark.parametrize(
        "name",
        [
            "clear-thought-mental_model",
            "clear_thought-server-mentalmodel",
            "clear-thought-server-meuqalModel",
            "clear-thought-server-clearmodel",
        ],
    )
    def test_undefined_fuzzy_mental_model_calls_are_not_family_members(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(
                called_names=(name,),
                defined_names=("clear-thought-server-mentalmodel",),
            ),
            "Qwen3",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {}

    def test_mental_model_definition_variation_across_rows_is_allowed(self) -> None:
        name = "mentalmodel"
        samples = []
        for description in (
            "Chirag/ThinkFar contract",
            "Waldzell stateful contract",
        ):
            sample = conv(
                tools=[func(name, description=description)],
                messages=[
                    assistant(tool_calls=[call(name=name)]),
                    tool_response(content="observation"),
                    assistant(content="done"),
                ],
                raw=qrow(),
            )
            raw_tools = copy.deepcopy(sample.tools)
            sample.raw["available_tools"] = orjson.dumps(raw_tools).decode()
            samples.append(sample)

        kept, drops, transforms = _apply_dataset_config(
            samples,
            ToucanConfig(drop_conflicting_duplicate_tools=True),
        )

        assert kept == samples
        assert drops == {}
        assert transforms == {}
        assert [sample.tools[0]["function"]["description"] for sample in kept] == [
            "Chirag/ThinkFar contract",
            "Waldzell stateful contract",
        ]

    def test_terminal_mental_model_call_is_preserved_then_incomplete_gate_drops(
        self,
    ) -> None:
        name = "clear-thought-server-mentalmodel"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(
                    content='{"modelName":"first_principles",'
                    '"status":"success","hasSteps":true,'
                    '"hasConclusion":true}'
                ),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert drops == {}
        assert transforms == {}
        assert kept == [sample]
        assert sample.messages == original_messages

        dropped, drop_reasons, _ = _apply_dataset_config(
            [copy.deepcopy(sample)],
            ToucanConfig(drop_incomplete_termination=True),
        )
        assert dropped == []
        assert drop_reasons == {"incomplete_termination": 1}

    def test_decision_framework_called_lineages_are_preserved_together(self) -> None:
        names = (
            "decisionFramework",
            "decision-framework-server-decisionFramework",
            "decisionframework",
            "clear-thought-server-decisionframework",
        )
        sample, _ = _convert_sample(
            conversion_row(called_names=names),
            "Qwen3",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [DECISION_FRAMEWORK_TOOL_FAMILY]
        }

    def test_decision_framework_success_and_error_observations_are_preserved(
        self,
    ) -> None:
        chirag = "clear-thought-server-decisionframework"
        standalone = "decision-framework-server-decisionFramework"
        sample = conv(
            tools=[func(chirag), func(standalone)],
            messages=[
                assistant(tool_calls=[call(name=chirag, call_id="c-success")]),
                tool_response(
                    content='{"decisionId":"d-1","stage":"decision"}',
                    tool_call_id="c-success",
                ),
                assistant(tool_calls=[call(name=chirag, call_id="c-format")]),
                tool_response(
                    content="McpError: criterion.weight.toFixed is not a function",
                    tool_call_id="c-format",
                ),
                assistant(tool_calls=[call(name=standalone, call_id="s-success")]),
                tool_response(
                    content='{"decisionId":"d-2","optionCount":2}',
                    tool_call_id="s-success",
                ),
                assistant(tool_calls=[call(name=standalone, call_id="s-error")]),
                tool_response(
                    content='{"error":"Invalid riskTolerance: must be a string",'
                    '"status":"failed"}',
                    tool_call_id="s-error",
                ),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_uncalled_clear_thought_decision_definition_is_unmarked_and_kept(
        self,
    ) -> None:
        name = "clear-thought-decisionframework"
        sample, _ = _convert_sample(
            conversion_row(
                called_names=("get_weather",),
                defined_names=("get_weather", name),
            ),
            "Qwen3",
        )
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.tools == original_tools
        assert sample.annotations == {}

    def test_undefined_exact_decision_call_is_marked_but_not_hidden(self) -> None:
        name = "decisionframework"
        sample, _ = _convert_sample(
            conversion_row(
                called_names=(name,),
                defined_names=("clear-thought-server-decisionframework",),
            ),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [DECISION_FRAMEWORK_TOOL_FAMILY]
        }

    def test_undefined_fuzzy_decision_call_is_unmarked_and_not_hidden(self) -> None:
        name = "clear-thought-decision_framework"
        sample, _ = _convert_sample(
            conversion_row(
                called_names=(name,),
                defined_names=("clear-thought-decisionframework",),
            ),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {}

    def test_decision_framework_definition_variation_across_rows_is_allowed(
        self,
    ) -> None:
        name = "decisionframework"
        samples = []
        for description in (
            "Chirag/ThinkFar echo contract",
            "Waldzell Clear Thought session-store contract",
        ):
            sample = conv(
                tools=[func(name, description=description)],
                messages=[user("u"), assistant(content="done")],
                raw=qrow(),
            )
            sample.raw["available_tools"] = orjson.dumps(sample.tools).decode()
            samples.append(sample)

        kept, drops, transforms = _apply_dataset_config(
            samples,
            ToucanConfig(drop_conflicting_duplicate_tools=True),
        )

        assert kept == samples
        assert drops == {}
        assert transforms == {}
        assert [sample.tools[0]["function"]["description"] for sample in kept] == [
            "Chirag/ThinkFar echo contract",
            "Waldzell Clear Thought session-store contract",
        ]

    def test_terminal_decision_call_is_preserved_then_incomplete_gate_drops(
        self,
    ) -> None:
        name = "clear-thought-server-decisionframework"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content='{"decisionId":"d-1","stage":"decision"}'),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages

        dropped, drop_reasons, _ = _apply_dataset_config(
            [copy.deepcopy(sample)],
            ToucanConfig(drop_incomplete_termination=True),
        )
        assert dropped == []
        assert drop_reasons == {"incomplete_termination": 1}

    def test_serialized_sequential_name_is_not_a_decision_family_call(self) -> None:
        name = (
            'model-context-protocol-server-sequentialthinking\\", '
            '{\\"thought\\": \\"apply a decision framework\\"'
        )
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,), defined_names=("get_weather",)),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.annotations == {}

    def test_scientific_method_called_lineages_are_preserved_together(self) -> None:
        names = (
            "scientificMethod",
            "scientific-method-server-scientificMethod",
            "scientificmethod",
            "clear-thought-server-scientificmethod",
        )
        sample, _ = _convert_sample(
            conversion_row(called_names=names),
            "Qwen3",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_scientific_method_success_error_decode_and_transport_results_are_kept(
        self,
    ) -> None:
        chirag = "clear-thought-server-scientificmethod"
        standalone = "scientific-method-server-scientificMethod"
        sample = conv(
            tools=[func(chirag), func(standalone)],
            messages=[
                assistant(tool_calls=[call(name=chirag, call_id="c-success")]),
                tool_response(
                    content=(
                        '{"content":[{"type":"text","text":'
                        '"{\\n  \\"stage\\": \\"question\\"\\n}"}]}'
                    ),
                    tool_call_id="c-success",
                ),
                assistant(tool_calls=[call(name=chirag, call_id="c-error")]),
                tool_response(
                    content=(
                        '{"content":[{"type":"text","text":'
                        '"{\\n  \\"status\\": \\"failed\\"\\n}"}],'
                        '"isError":true}'
                    ),
                    tool_call_id="c-error",
                ),
                assistant(tool_calls=[call(name=standalone, call_id="s-success")]),
                tool_response(
                    content=(
                        '{"inquiryId":"i-1","stage":"question","iteration":1,'
                        '"hasObservation":false,"hasQuestion":true,'
                        '"hasHypothesis":false,"hasExperiment":false,'
                        '"hasAnalysis":false,"hasConclusion":false,'
                        '"nextStageNeeded":true}'
                    ),
                    tool_call_id="s-success",
                ),
                assistant(tool_calls=[call(name=standalone, call_id="s-error")]),
                tool_response(
                    content='{"error":"Invalid inquiryId: must be a string",'
                    '"status":"failed"}',
                    tool_call_id="s-error",
                ),
                assistant(tool_calls=[call(name=standalone, call_id="decode")]),
                tool_response(
                    content="JSONDecodeError: Expecting ',' delimiter",
                    tool_call_id="decode",
                ),
                assistant(tool_calls=[call(name=standalone, call_id="transport")]),
                tool_response(
                    content="Session reconnect (client creation) exception: Session terminated",
                    tool_call_id="transport",
                ),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_uncalled_clear_thought_scientific_definition_is_unmarked_and_kept(
        self,
    ) -> None:
        name = "clear-thought-scientificmethod"
        sample, _ = _convert_sample(
            conversion_row(
                called_names=("get_weather",),
                defined_names=("get_weather", name),
            ),
            "Qwen3",
        )
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.tools == original_tools
        assert sample.annotations == {}

    @pytest.mark.parametrize("name", ["scientificMethod", "scientificmethod"])
    def test_undefined_exact_scientific_call_is_marked_but_not_hidden(
        self, name: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(
                called_names=(name,),
                defined_names=("clear-thought-server-scientificmethod",),
            ),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [SCIENTIFIC_METHOD_TOOL_FAMILY]
        }

    def test_scientific_method_definition_variation_across_rows_is_allowed(
        self,
    ) -> None:
        name = "scientificmethod"
        samples = []
        for description in (
            "Chirag/ThinkFar deterministic-status contract",
            "Waldzell Clear Thought session-store contract",
        ):
            sample = conv(
                tools=[func(name, description=description)],
                messages=[user("u"), assistant(content="done")],
                raw=qrow(),
            )
            sample.raw["available_tools"] = orjson.dumps(sample.tools).decode()
            samples.append(sample)

        kept, drops, transforms = _apply_dataset_config(
            samples,
            ToucanConfig(drop_conflicting_duplicate_tools=True),
        )

        assert kept == samples
        assert drops == {}
        assert transforms == {}
        assert [sample.tools[0]["function"]["description"] for sample in kept] == [
            "Chirag/ThinkFar deterministic-status contract",
            "Waldzell Clear Thought session-store contract",
        ]

    def test_terminal_scientific_call_is_preserved_then_incomplete_gate_drops(
        self,
    ) -> None:
        name = "clear-thought-server-scientificmethod"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content='{"content":[{"type":"text","text":"success"}]}'),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages

        dropped, drop_reasons, _ = _apply_dataset_config(
            [copy.deepcopy(sample)],
            ToucanConfig(drop_incomplete_termination=True),
        )
        assert dropped == []
        assert drop_reasons == {"incomplete_termination": 1}

    def test_debugging_and_design_lineages_are_preserved_together(self) -> None:
        names = (
            "debuggingapproach",
            "clear-thought-server-debuggingapproach",
            "clear-thought-debuggingapproach",
            "designpattern",
            "clear-thought-server-designpattern",
        )
        sample, _ = _convert_sample(conversion_row(called_names=names), "Qwen3")
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [
                DEBUGGING_APPROACH_TOOL_FAMILY,
                DESIGN_PATTERN_TOOL_FAMILY,
            ]
        }

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "name", "observation", "family"),
        [
            (
                "cd6aabab-ace0-53cf-84db-0766ab781830",
                "Kimi-K2",
                "clear-thought-server-debuggingapproach",
                '{"content":[{"type":"text","text":"{\\n  \\"approachName\\": '
                '\\"divide_conquer\\",\\n  \\"status\\": \\"success\\",\\n  '
                '\\"hasSteps\\": true,\\n  \\"hasResolution\\": false\\n}"}]}',
                DEBUGGING_APPROACH_TOOL_FAMILY,
            ),
            (
                "d8045c3f-c607-55d7-83df-eee7117cd769",
                "OSS",
                "debuggingapproach",
                '{"content":[{"type":"text","text":"debug status"}]}',
                DEBUGGING_APPROACH_TOOL_FAMILY,
            ),
            (
                "cd6aabab-ace0-53cf-84db-0766ab781830",
                "Kimi-K2",
                "clear-thought-server-designpattern",
                '{"content":[{"type":"text","text":"design status"}]}',
                DESIGN_PATTERN_TOOL_FAMILY,
            ),
            (
                "3be63bfe-5a78-56e6-8f90-a3be5fe548a4",
                "OSS",
                "designpattern",
                '{"content":[{"type":"text","text":"design status"}]}',
                DESIGN_PATTERN_TOOL_FAMILY,
            ),
        ],
    )
    def test_real_debugging_design_success_witnesses_are_preserved(
        self,
        sample_id: str,
        teacher: str,
        name: str,
        observation: str,
        family: str,
    ) -> None:
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content=observation),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        sample.sample_id = sample_id
        sample.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        sample.annotations = {REASONING_TOOL_FAMILIES_ANNOTATION: [family]}
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_debugging_design_handler_and_decode_failures_remain_visible(self) -> None:
        debugging = "clear-thought-server-debuggingapproach"
        design = "clear-thought-server-designpattern"
        sample = conv(
            tools=[func(debugging), func(design)],
            messages=[
                assistant(tool_calls=[call(name=debugging, call_id="debug-error")]),
                tool_response(
                    content=(
                        '{"content":[{"type":"text","text":'
                        '"{\\n  \\"error\\": \\"Invalid approachName: must be a string\\",'
                        '\\n  \\"status\\": \\"failed\\"\\n}"}],"isError":true}'
                    ),
                    tool_call_id="debug-error",
                ),
                assistant(
                    tool_calls=[
                        call(name=design, arguments="{broken", call_id="design-decode")
                    ]
                ),
                tool_response(
                    content="JSONDecodeError: Unterminated string",
                    tool_call_id="design-decode",
                ),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert _stage1_issues(sample.messages) == {"unparseable_tool_call_arguments": 1}

    def test_preservation_exposes_invalid_debugging_design_arguments(self) -> None:
        name = "designpattern"
        parameters = {
            "type": "object",
            "properties": {
                "patternName": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["patternName", "context"],
        }
        sample = conv(
            tools=[func(name, parameters=parameters)],
            messages=[
                assistant(
                    tool_calls=[
                        call(
                            name=name,
                            arguments={"patternName": "scalability", "extra": True},
                        )
                    ]
                ),
                tool_response(content="handler status"),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert has_invalid_arguments(sample) is True

    def test_uncalled_waldzell_debugging_definitions_are_unmarked_and_kept(
        self,
    ) -> None:
        samples = []
        for name in ("debuggingapproach", "clear-thought-debuggingapproach"):
            sample, _ = _convert_sample(
                conversion_row(
                    called_names=("get_weather",),
                    defined_names=("get_weather", name),
                ),
                "OSS",
            )
            samples.append(sample)
        original_tools = [copy.deepcopy(sample.tools) for sample in samples]

        kept, drops, transforms = _apply_dataset_config(samples, ToucanConfig())

        assert kept == samples
        assert drops == {}
        assert transforms == {}
        assert [sample.tools for sample in samples] == original_tools
        assert [sample.annotations for sample in samples] == [{}, {}]

    @pytest.mark.parametrize(
        ("name", "defined_name", "family"),
        [
            (
                "debuggingapproach",
                "clear-thought-server-debuggingapproach",
                DEBUGGING_APPROACH_TOOL_FAMILY,
            ),
            (
                "designpattern",
                "clear-thought-server-designpattern",
                DESIGN_PATTERN_TOOL_FAMILY,
            ),
        ],
    )
    def test_undefined_exact_debugging_design_call_is_marked_but_not_hidden(
        self, name: str, defined_name: str, family: str
    ) -> None:
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,), defined_names=(defined_name,)),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {REASONING_TOOL_FAMILIES_ANNOTATION: [family]}
        assert has_undefined_function_calls(sample) is True

    def test_debugging_definition_variation_across_rows_is_allowed(self) -> None:
        name = "debuggingapproach"
        samples = []
        for description in (
            "Chirag/ThinkFar deterministic-status contract",
            "Waldzell session-store contract",
        ):
            sample = conv(
                tools=[func(name, description=description)],
                messages=[user("u"), assistant(content="done")],
                raw=qrow(),
            )
            sample.raw["available_tools"] = orjson.dumps(sample.tools).decode()
            samples.append(sample)

        kept, drops, transforms = _apply_dataset_config(
            samples, ToucanConfig(drop_conflicting_duplicate_tools=True)
        )

        assert kept == samples
        assert drops == {}
        assert transforms == {}

    def test_terminal_design_call_is_preserved_then_incomplete_gate_drops(
        self,
    ) -> None:
        name = "clear-thought-server-designpattern"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content="design status"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages

        dropped, drop_reasons, _ = _apply_dataset_config(
            [copy.deepcopy(sample)],
            ToucanConfig(drop_incomplete_termination=True),
        )
        assert dropped == []
        assert drop_reasons == {"incomplete_termination": 1}

    def test_collaborative_and_visual_lineages_are_preserved_together(self) -> None:
        names = (
            "collaborativeReasoning",
            "collaborative-reasoning-server-collaborativeReasoning",
            "collaborativereasoning",
            "clear-thought-server-collaborativereasoning",
            "clear-thought-collaborativereasoning",
            "visualReasoning",
            "visual-reasoning-server-visualReasoning",
            "visualreasoning",
            "clear-thought-server-visualreasoning",
            "clear-thought-visualreasoning",
        )
        sample, _ = _convert_sample(conversion_row(called_names=names), "Qwen3")
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [
                COLLABORATIVE_REASONING_TOOL_FAMILY,
                VISUAL_REASONING_TOOL_FAMILY,
            ]
        }

    @pytest.mark.parametrize(
        ("sample_id", "teacher", "name", "observation", "family"),
        [
            (
                "cd6aabab-ace0-53cf-84db-0766ab781830",
                "Kimi-K2",
                "clear-thought-server-collaborativereasoning",
                '{"topic":"root cause analysis","nextContributionNeeded":true}',
                COLLABORATIVE_REASONING_TOOL_FAMILY,
            ),
            (
                "43948be7-bb3d-53db-a288-1de3f8f35dfa",
                "Kimi-K2",
                "collaborative-reasoning-server-collaborativeReasoning",
                '{"sessionId":"session-1","nextContributionNeeded":true}',
                COLLABORATIVE_REASONING_TOOL_FAMILY,
            ),
            (
                "79f0967b-812c-52d5-98c3-ee64a40ffe28",
                "Qwen3",
                "clear-thought-collaborativereasoning",
                '{"status":"success","sessionContext":{"sessionId":"runtime-uuid"}}',
                COLLABORATIVE_REASONING_TOOL_FAMILY,
            ),
            (
                "8694a637-351f-5f21-af4b-7a927ce4bd68",
                "Kimi-K2",
                "clear-thought-server-visualreasoning",
                '{"content":[{"type":"text","text":"{\\"status\\":\\"success\\"}"}]}',
                VISUAL_REASONING_TOOL_FAMILY,
            ),
            (
                "a18dc779-c8a9-5094-a59d-19d0ef5f2b33",
                "OSS",
                "visualreasoning",
                '{"content":[{"type":"text","text":"{\\"elementCount\\":1}"}]}',
                VISUAL_REASONING_TOOL_FAMILY,
            ),
            (
                "86984058-6e05-5912-800e-86edecd5fc09",
                "Qwen3",
                "visual-reasoning-server-visualReasoning",
                '{"diagramId":"d-1","elementCount":2,"historyLength":3}',
                VISUAL_REASONING_TOOL_FAMILY,
            ),
        ],
    )
    def test_real_collaborative_visual_success_witnesses_are_preserved(
        self,
        sample_id: str,
        teacher: str,
        name: str,
        observation: str,
        family: str,
    ) -> None:
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content=observation),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        sample.sample_id = sample_id
        sample.dataset = f"Agent-Ark/Toucan-1.5M:{teacher}"
        sample.annotations = {REASONING_TOOL_FAMILIES_ANNOTATION: [family]}
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools

    def test_collaborative_visual_failures_and_retries_remain_visible(self) -> None:
        collaborative = "clear-thought-server-collaborativereasoning"
        visual = "visual-reasoning-server-visualReasoning"
        sample = conv(
            tools=[func(collaborative), func(visual)],
            messages=[
                assistant(
                    tool_calls=[call(name=collaborative, call_id="collab-error")]
                ),
                tool_response(
                    content="McpError: Cannot read properties of undefined (reading 'join')",
                    tool_call_id="collab-error",
                ),
                assistant(
                    tool_calls=[call(name=collaborative, call_id="collab-retry")]
                ),
                tool_response(
                    content='{"topic":"retry succeeded"}', tool_call_id="collab-retry"
                ),
                assistant(tool_calls=[call(name=visual, call_id="visual-error")]),
                tool_response(
                    content='{"error":"Missing required property: transformationType","status":"failed"}',
                    tool_call_id="visual-error",
                ),
                assistant(tool_calls=[call(name=visual, call_id="visual-retry")]),
                tool_response(
                    content='{"diagramId":"d-1","historyLength":1}',
                    tool_call_id="visual-retry",
                ),
                assistant(content="done"),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages

    def test_standalone_visual_cumulative_state_observations_are_preserved(
        self,
    ) -> None:
        name = "visualReasoning"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name, call_id="create")]),
                tool_response(
                    content='{"diagramId":"d","elementCount":1,"historyLength":1}',
                    tool_call_id="create",
                ),
                assistant(tool_calls=[call(name=name, call_id="update")]),
                tool_response(
                    content='{"diagramId":"d","elementCount":1,"historyLength":2}',
                    tool_call_id="update",
                ),
                assistant(content="The second result depends on the first call."),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages

    def test_uncalled_compact_collaborative_visual_definitions_are_unmarked_and_kept(
        self,
    ) -> None:
        names = (
            "clear-thought-collaborativereasoning",
            "clear-thought-visualreasoning",
        )
        sample, _ = _convert_sample(
            conversion_row(
                called_names=("get_weather",),
                defined_names=("get_weather", *names),
            ),
            "Kimi-K2",
        )
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.tools == original_tools
        assert sample.annotations == {}

    def test_undefined_exact_visual_call_is_marked_but_not_hidden(self) -> None:
        name = "visualreasoning"
        sample, _ = _convert_sample(
            conversion_row(called_names=(name,), defined_names=("get_weather",)),
            "Kimi-K2",
        )
        original_messages = copy.deepcopy(sample.messages)
        original_tools = copy.deepcopy(sample.tools)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages
        assert sample.tools == original_tools
        assert sample.annotations == {
            REASONING_TOOL_FAMILIES_ANNOTATION: [VISUAL_REASONING_TOOL_FAMILY]
        }
        assert has_undefined_function_calls(sample) is True

    @pytest.mark.parametrize("name", ["collaborativereasoning", "visualreasoning"])
    def test_collaborative_visual_definition_variation_across_rows_is_allowed(
        self, name: str
    ) -> None:
        samples = []
        for description in (
            "Chirag/ThinkFar legacy contract",
            "Waldzell Clear Thought state-store contract",
        ):
            sample = conv(
                tools=[func(name, description=description)],
                messages=[user("u"), assistant(content="done")],
                raw=qrow(),
            )
            sample.raw["available_tools"] = orjson.dumps(sample.tools).decode()
            samples.append(sample)

        kept, drops, transforms = _apply_dataset_config(
            samples, ToucanConfig(drop_conflicting_duplicate_tools=True)
        )

        assert kept == samples
        assert drops == {}
        assert transforms == {}

    def test_terminal_visual_call_is_preserved_then_incomplete_gate_drops(self) -> None:
        name = "visual-reasoning-server-visualReasoning"
        sample = conv(
            tools=[func(name)],
            messages=[
                assistant(tool_calls=[call(name=name)]),
                tool_response(content='{"nextOperationNeeded":true}'),
            ],
            raw=qrow(),
        )
        original_messages = copy.deepcopy(sample.messages)

        kept, drops, transforms = _apply_dataset_config([sample], ToucanConfig())

        assert kept == [sample]
        assert drops == {}
        assert transforms == {}
        assert sample.messages == original_messages

        dropped, drop_reasons, _ = _apply_dataset_config(
            [copy.deepcopy(sample)],
            ToucanConfig(drop_incomplete_termination=True),
        )
        assert dropped == []
        assert drop_reasons == {"incomplete_termination": 1}

    def test_all_audited_think_writes_are_preserved_without_state_reads(self) -> None:
        s = conv(
            tools=[
                func("think-tool-server-think"),
                func("think-tool-server-get_thoughts"),
                func("think"),
            ],
            messages=[
                assistant(tool_calls=[call(name="think-tool-server-think")]),
                tool_response(content="recorded"),
                assistant(tool_calls=[call(name="think")]),
                tool_response(content="unrelated bare echo"),
                assistant(tool_calls=[call(name="think-tool-server-get_thoughts")]),
                tool_response(content="Thought #1: recorded"),
            ],
            raw=qrow(),
        )

        kept, _, transforms = _apply_dataset_config([s], ToucanConfig())

        assert transforms == {}
        assert [
            c["function"]["name"]
            for m in kept[0].messages
            for c in m.get("tool_calls") or []
        ] == [
            "think-tool-server-think",
            "think",
            "think-tool-server-get_thoughts",
        ]

    def test_arbitrary_matching_suffix_does_not_create_a_state_family(self) -> None:
        # The raw snapshot contains an undefined mcpollinations-get_thoughts
        # hallucination. A shared arbitrary prefix is not evidence that it and a
        # hypothetical *-think call share the audited Clear Thought state.
        s = conv(
            tools=[
                func("mcpollinations-think"),
                func("mcpollinations-get_thoughts"),
            ],
            messages=[
                assistant(tool_calls=[call(name="mcpollinations-think")]),
                tool_response(content="echo"),
                assistant(tool_calls=[call(name="mcpollinations-get_thoughts")]),
                tool_response(content="Tool does not exist"),
            ],
            raw=qrow(),
        )

        kept, _, transforms = _apply_dataset_config([s], ToucanConfig())

        assert transforms == {"stripped_reasoning_tools": 1}
        assert [t["function"]["name"] for t in kept[0].tools] == [
            "mcpollinations-get_thoughts"
        ]
        assert [
            c["function"]["name"]
            for m in kept[0].messages
            for c in m.get("tool_calls") or []
        ] == ["mcpollinations-get_thoughts"]

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
            tools=[func("chain-of-draft")],
            messages=[
                assistant(tool_calls=[call(name="chain-of-draft")]),
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
            tools=[func("chain-of-draft")],
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="chain-of-draft")]),
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
