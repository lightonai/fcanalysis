"""Fixture matrix for loader regression tests.

For legacy loaders, enumerate production plus one variant per boolean flag
and multi-option field. Migrated loaders enforce mandatory production
invariants. Corpus fixtures cover retained/stripped native reasoning and every
selected source scope; direct adversarial tests exercise system overrides.

Each entry produces one frozen fixture under
tests/fixtures/loaders/{loader}/{config_id}/.
"""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol

from fcanalysis.loaders import LOADER_MODULES as LOADER_MODULES
from fcanalysis.loaders.apigen_mt import APIGenMTConfig
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.dolci import DolciConfig
from fcanalysis.loaders.nemotron_agentic_v1 import NemotronAgenticV1Config
from fcanalysis.loaders.nemotron_agentic_v2 import NemotronAgenticV2Config
from fcanalysis.loaders.nemotron_terminal import NemotronTerminalConfig
from fcanalysis.loaders.toolmind import ToolMindConfig
from fcanalysis.loaders.toolmind_web import ToolMindWebConfig
from fcanalysis.loaders.toucan import ToucanConfig
from fcanalysis.loaders.txt360 import TxT360Config
from fcanalysis.loaders.ultradata_tool_use import UltraDataToolUseConfig


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


# Migrated adapters enforce the accepted structural contract. Obsolete flags
# that removed visible tools or guessed assistant serialization are no longer
# valid fixture variations. Keep production and supported final-view controls.
def _migrated_specs(loader: str, config: Any) -> list[FixtureSpec]:
    kwargs = {"curation_config": CurationConfig(), "batch_size": 256}
    return [
        FixtureSpec(loader, "prod", config, PROD_FILTER, kwargs),
        FixtureSpec(
            loader,
            "no-strip_thinking",
            config,
            replace(PROD_FILTER, strip_thinking=False),
            kwargs,
        ),
        FixtureSpec(
            loader,
            "system-override-empty",
            config,
            replace(PROD_FILTER, system_message_override=""),
            kwargs,
        ),
        FixtureSpec(
            loader,
            "system-override-custom",
            config,
            replace(
                PROD_FILTER,
                system_message_override="Use the available tools carefully.",
            ),
            kwargs,
        ),
    ]


APIGEN_MT_SPECS = _migrated_specs("apigen_mt", APIGenMTConfig())
DOLCI_SPECS = _migrated_specs("dolci", DolciConfig())


NEMOTRON_V1_SPECS = _migrated_specs("nemotron_agentic_v1", NemotronAgenticV1Config())[
    :2
]
NEMOTRON_V2_SPECS = _migrated_specs("nemotron_agentic_v2", NemotronAgenticV2Config())[
    :2
]


NEMOTRON_TERMINAL_SPECS = [
    replace(spec, extra_kwargs={**spec.extra_kwargs, "batch_size": 64})
    for spec in _migrated_specs("nemotron_terminal", NemotronTerminalConfig())[:2]
]

TOOLMIND_SPECS = _migrated_specs("toolmind", ToolMindConfig())[:2]
TOOLMIND_WEB_SPECS = _migrated_specs("toolmind_web", ToolMindWebConfig())[:2]
ULTRADATA_SPECS = _migrated_specs("ultradata_tool_use", UltraDataToolUseConfig())[:2]
TOUCAN_SPECS = [
    replace(spec, extra_kwargs={**spec.extra_kwargs, "batch_size": 32})
    for spec in _migrated_specs("toucan", ToucanConfig())[:2]
]

# All three effort partitions have independent reconstruction and curation.
# Keep existing high fixture names while making medium/low coverage explicit.
TXT360_SPECS = [
    replace(
        spec,
        config_id=spec.config_id if effort == "high" else f"{effort}-{spec.config_id}",
        extra_kwargs={**spec.extra_kwargs, "split": effort},
    )
    for effort in ("high", "medium", "low")
    for spec in _migrated_specs("txt360", TxT360Config())[:2]
]


ALL_SPECS: list[FixtureSpec] = [
    *APIGEN_MT_SPECS,
    *DOLCI_SPECS,
    *NEMOTRON_V1_SPECS,
    *NEMOTRON_V2_SPECS,
    *NEMOTRON_TERMINAL_SPECS,
    *TOOLMIND_SPECS,
    *TOOLMIND_WEB_SPECS,
    *TOUCAN_SPECS,
    *TXT360_SPECS,
    *ULTRADATA_SPECS,
]


SPECS_BY_ID: dict[str, FixtureSpec] = {s.fixture_id: s for s in ALL_SPECS}


def specs_for_loader(loader: str) -> list[FixtureSpec]:
    return [s for s in ALL_SPECS if s.loader == loader]
