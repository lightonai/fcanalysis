"""Fixture matrix for loader regression tests.

For every supported loader, enumerate one production config (matching
training_mix.md) plus one variant per boolean flag flipped (dataset-
specific + universal). For loaders with multi-option fields
(seed_group_filter, splits), enumerate one variant per alternative
value.

Each entry produces one frozen fixture under
tests/fixtures/loaders/{loader}/{config_id}/.
"""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol

from fcanalysis.loaders import LOADER_MODULES as LOADER_MODULES
from fcanalysis.loaders.apigen_mt import APIGenMTConfig
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.dolci import DolciConfig
from fcanalysis.loaders.nemotron_agentic_v1 import NemotronAgenticV1Config
from fcanalysis.loaders.nemotron_agentic_v2 import NemotronAgenticV2Config
from fcanalysis.loaders.nemotron_terminal import NemotronTerminalConfig
from fcanalysis.loaders.toolmind import ToolMindConfig
from fcanalysis.loaders.txt360 import TxT360Config


class _DataclassLike(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


UNIVERSAL_FIELDS: tuple[str, ...] = (
    "strip_thinking",
    "require_parseable_arguments",
    "require_balanced_cardinality",
    "require_defined_functions",
    "require_valid_arguments",
)


PROD_FILTER: FilterConfig = FilterConfig(
    strip_thinking=True,
    require_parseable_arguments=True,
    require_balanced_cardinality=True,
    require_defined_functions=True,
    require_valid_arguments=True,
)


@dataclass(frozen=True)
class FixtureSpec:
    """One (loader, config) tuple to capture as a regression fixture."""

    loader: str
    config_id: str
    dataset_config: Any
    filter_config: FilterConfig
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def fixture_id(self) -> str:
        return f"{self.loader}/{self.config_id}"


def _universal_variants(prod_filter: FilterConfig) -> list[tuple[str, FilterConfig]]:
    return [
        (f"no-{flag}", replace(prod_filter, **{flag: False}))
        for flag in UNIVERSAL_FIELDS
    ]


def _build_specs[T: _DataclassLike](
    loader: str,
    prod_config: T,
    dataset_flags: list[str],
    extra_variants: list[tuple[str, T]] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> list[FixtureSpec]:
    base_kwargs = extra_kwargs or {}
    specs: list[FixtureSpec] = [
        FixtureSpec(loader, "prod", prod_config, PROD_FILTER, base_kwargs),
    ]
    for flag in dataset_flags:
        flipped = replace(prod_config, **{flag: False})
        specs.append(
            FixtureSpec(loader, f"no-{flag}", flipped, PROD_FILTER, base_kwargs)
        )
    for variant_id, filter_cfg in _universal_variants(PROD_FILTER):
        specs.append(
            FixtureSpec(loader, variant_id, prod_config, filter_cfg, base_kwargs)
        )
    if extra_variants:
        for variant_id, dataset_cfg in extra_variants:
            specs.append(
                FixtureSpec(loader, variant_id, dataset_cfg, PROD_FILTER, base_kwargs)
            )
    return specs


_APIGEN_MT_PROD = APIGenMTConfig(
    strip_think_tool=True,
    drop_undefined_function_calls=True,
    drop_repeated_tool_call_streaks=True,
    drop_consecutive_assistant=True,
)
APIGEN_MT_SPECS: list[FixtureSpec] = _build_specs(
    "apigen_mt",
    _APIGEN_MT_PROD,
    dataset_flags=[
        "strip_think_tool",
        "drop_undefined_function_calls",
        "drop_repeated_tool_call_streaks",
        "drop_consecutive_assistant",
    ],
)


_DOLCI_PROD = DolciConfig(
    drop_consecutive_text_text_assistant=True,
    merge_text_fc_assistant=True,
    drop_conflicting_duplicate_tools=True,
)
DOLCI_SPECS: list[FixtureSpec] = _build_specs(
    "dolci",
    _DOLCI_PROD,
    dataset_flags=[
        "drop_consecutive_text_text_assistant",
        "merge_text_fc_assistant",
        "drop_conflicting_duplicate_tools",
    ],
)


_NEM_V1_PROD = NemotronAgenticV1Config(
    splits=("tool_calling",),
    drop_orphan_samples=True,
    drop_empty_system=True,
    drop_conflicting_duplicate_tools=True,
)
NEMOTRON_V1_SPECS: list[FixtureSpec] = _build_specs(
    "nemotron_agentic_v1",
    _NEM_V1_PROD,
    dataset_flags=[
        "drop_orphan_samples",
        "drop_empty_system",
        "drop_conflicting_duplicate_tools",
    ],
    extra_variants=[
        (
            "splits-interactive-only",
            replace(_NEM_V1_PROD, splits=("interactive_agent",)),
        ),
    ],
)


_NEM_V2_PROD = NemotronAgenticV2Config(
    splits=("interactive_agent", "search"),
)
NEMOTRON_V2_SPECS: list[FixtureSpec] = _build_specs(
    "nemotron_agentic_v2",
    _NEM_V2_PROD,
    dataset_flags=[],
    extra_variants=[
        (
            "splits-interactive-only",
            NemotronAgenticV2Config(splits=("interactive_agent",)),
        ),
        ("splits-search-only", NemotronAgenticV2Config(splits=("search",))),
    ],
)


_NEM_TERM_PROD = NemotronTerminalConfig(
    strip_malformed=True,
    drop_orphans=True,
    drop_incomplete=True,
)
NEMOTRON_TERMINAL_SPECS: list[FixtureSpec] = _build_specs(
    "nemotron_terminal",
    _NEM_TERM_PROD,
    dataset_flags=["strip_malformed", "drop_orphans", "drop_incomplete"],
)


_TOOLMIND_PROD = ToolMindConfig(
    sources=["graph_syn_datasets/graphsyn.jsonl"],
    seed_group_filter="longest_clean",
    drop_non_object_arguments=True,
    drop_consecutive_text_assistant=True,
    merge_split_assistant=True,
    strip_think_tool=True,
)
TOOLMIND_SPECS: list[FixtureSpec] = _build_specs(
    "toolmind",
    _TOOLMIND_PROD,
    dataset_flags=[
        "drop_non_object_arguments",
        "drop_consecutive_text_assistant",
        "merge_split_assistant",
        "strip_think_tool",
    ],
    extra_variants=[
        (
            "seed_group-longest",
            replace(_TOOLMIND_PROD, seed_group_filter="longest"),
        ),
        (
            "seed_group-none",
            replace(_TOOLMIND_PROD, seed_group_filter="none"),
        ),
    ],
)


_TXT360_PROD = TxT360Config(
    seed_group_filter="latest_clean_prefix",
    require_non_empty_user=True,
    drop_user_tool_call_samples=True,
)
TXT360_SPECS: list[FixtureSpec] = _build_specs(
    "txt360",
    _TXT360_PROD,
    dataset_flags=["require_non_empty_user", "drop_user_tool_call_samples"],
    extra_variants=[
        ("seed_group-last", replace(_TXT360_PROD, seed_group_filter="last")),
        (
            "seed_group-last_clean",
            replace(_TXT360_PROD, seed_group_filter="last_clean"),
        ),
        ("seed_group-none", replace(_TXT360_PROD, seed_group_filter="none")),
    ],
    extra_kwargs={"split": "high"},
)


ALL_SPECS: list[FixtureSpec] = [
    *APIGEN_MT_SPECS,
    *DOLCI_SPECS,
    *NEMOTRON_V1_SPECS,
    *NEMOTRON_V2_SPECS,
    *NEMOTRON_TERMINAL_SPECS,
    *TOOLMIND_SPECS,
    *TXT360_SPECS,
]


SPECS_BY_ID: dict[str, FixtureSpec] = {s.fixture_id: s for s in ALL_SPECS}


def specs_for_loader(loader: str) -> list[FixtureSpec]:
    return [s for s in ALL_SPECS if s.loader == loader]
