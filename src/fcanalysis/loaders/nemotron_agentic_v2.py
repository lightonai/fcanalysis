"""Load complete Nemotron SFT Agentic v2 conversations with shared stages.

Origin and license
------------------
NVIDIA's pinned card is
https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2/blob/7c804833427f633ccd53b582dbf02525fd680f78/README.md .
The Super model report describes related generation in section 3.1.1:
https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf .
The upstream developer hub is https://github.com/NVIDIA-NeMo/Nemotron . These
sources do not identify the exact generator commit/command for each row. The
card declares CC BY 4.0 plus additional Apache 2.0 and MIT terms; it does not
map all inherited records to licenses. Named upstream tool/trajectory sources
are UltraTool, ToolEyes, AutoTools, API-Bank, Glaive-Function-Calling-v2 and a
TOUCAN subsample. Their restrictions and exact transformation depths remain
source-specific and unknown where unverified; code licenses are not inherited
data licenses. The report also names xLAM/custom tools and different aggregate
counts/models, so those claims are not silently promoted to pinned row facts.

Generation, source schema and omissions
--------------------------------------
The card describes customer-service user/assistant/environment simulations;
Wikidata-derived search questions solved with search tools; and general tool
simulations/judging using DeepSeek-V3.2 and GLM-4.6. The related Super report
describes six-stage policy/scenario generation and multi-rollout judging;
its search pipeline uses NeMo Data Designer and Tavily. Actual pinned files
contain interactive_agent 278,880, search 5,968, and tool_calling 707,052 rows.
The card's prose counts differ from its table/raw files; raw counts govern.

Released UUIDs are not unique identities: the complete census has 991,898
distinct IDs among 991,900 rows. Two interactive UUIDs each occur twice with
different exact call arguments. One pair fails the existing unsupported
date-time schema gate; both occurrences of the other pair survive in both
reasoning modes. Preserve the released IDs and distinct conversations. Source
audits must compare complete content and physical order, not pair UUID ordinals
or deduplicate solely by ID.

messages and tools are model-visible. metadata, model, domain, temperature,
parallel_tool_calls, chat_template_kwargs, filter_reason, processing_info,
match_contexts, matched_categories, uuid and used_in are audit/generation
fields; none supplies simulator state or creates a loss mask. All released
calls/results have source IDs; search also echoes result names. Its 70,210
assistant function_call slots are null alongside the tool_calls field and
are explicitly discarded only in that known subset. Non-null legacy calls
quarantine. Arguments are JSON strings. Systems are preserved exactly: 3,208
tool-calling rows have none. Complete definitions preserve strict/schema fields;
23 legacy function-level required fields receive the named constraint-union
repair. Unknown model-visible fields and unsupported schema constraints fail
closed. Call IDs prove pairing, but neither the pinned card nor the general-tool
generation report establishes a batch permutation contract. The raw
parallel_tool_calls flag is False on all 278,880 interactive rows, which have
only single-call batches; it is absent throughout search and tool_calling.
ID-bound multi-call batches preserve call order and align their complete result
messages to that order by default. FilterConfig.align_results=False instead
rejects a candidate requiring this transformation. Paired call/result units
remain atomic and ordered for curation; alignment grants no permutation
equivalence. Missing/extra/orphan/ambiguous results quarantine; the
raw census found 137,741 unbalanced tool-calling batches. The report's explicit
parallel-tools construction concerns its separate Agentic CLI data and does
not establish this subset's protocol.

The repaired tool_calling metadata.source explicitly labels released source
families, including Agent-Ark/Toucan-1.5M; metadata.sdg_model labels the released
teacher. Full pinned counts are TOUCAN 277,032; tool_eyes 245,321; xlam 62,826;
ultra_tools 30,561; api_bank 28,892; when2call 13,963; missing_func 12,555;
glaive 10,308; auto_tools 5,976; xlam_tools 2,391; and ten custom_* families
totaling 17,227. Released teacher labels count DeepSeek-v3.2 371,850,
GPT-OSS-120B 277,032, and GLM-4.6 58,170. Those labels do not establish exact
generation-code ancestry or a verified row-by-row transformation from the
upstream source.

Reasoning, context and uncertainty
---------------------------------
Native reasoning_content occurs in all interactive assistants (1,324,660
nonempty; 99,550 empty), in 70,208 search assistants, and in 2,726,293
of 2,850,682 tool-calling assistants. A full content census finds four unmatched
closing think tags in interactive, 15 in search, and one opening reasoning tag
in tool_calling; literal tags can be ordinary code or prose and are not
identified by reasoning keywords. FilterConfig.strip_thinking removes
structured reasoning and exact safely closed assistant think/reasoning spans,
with code shielding and unmatched-boundary quarantine when stripping. No-argument loading
strips native reasoning; explicit strip_thinking=False retains it. Declared think,
reasoning, resource and orchestration tools remain observable calls. Full
conversations are returned; trainer-specific prefix splitting, history
reasoning visibility, loss masks and weights remain trainer responsibilities.
No verified prefix family or omission rule is inferred from UUIDs/adjacency.
Final replies containing only recognized native reasoning are incomplete in
either mode; keeping reasoning does not turn it into an answer. Native control
tokens alone also provide no answer. Neither does an open-only native tail
preceded solely by balanced native spans and whitespace. Code literals and
other ambiguous boundaries retain their configured handling.

The source helper validates audited credential argument slots using prior
user/system values or linked result leaves. Assistant reasoning/prose, tool
schema examples, future results and metadata cannot supply missing codes or
state. A full description/name census supplies finite exact additional consuming
signatures, including nonstandard API-key fields. Documented Bearer wrappers and
observed hyphen-separated individual OTP digits can use prior visible constituent
values without changing source text or arguments. Generated passwords,
composite/DOB credentials, derived signatures, composed cookies and ambiguous
demo/default contracts remain outside the extension; explicit placeholder
defaults do not supply an API key.
The check is bounded: arbitrary natural-language policy compliance,
confirmation meaning, calendar arithmetic and simulated result truth are not
certified. Dynamic capability use needs a verified visible runtime grant,
never a future definition union or a name-only guess. Undefined downstream
calls quarantine.

The temporal census finds initial facility-tour, conference-room, API-Bank
conflict-check and TOUCAN-derived 12306 train calls converting next-weekday
requests into absolute dates with no visible clock. Reviewed consuming calendar
slots require a visible explicit target date or labeled current date for that
conversion. Historical dates do not become clocks. This bounded gate abstains
after prior tool results, whose temporal meaning needs a separate source
contract. Explicit user date ranges and partial dates needing year association
also cause abstention rather than missing-date proof. The shared schema contract
intentionally excludes date-time and other
unsupported formats, even in unused definitions; optional format packages must
not silently change acceptance.

Ordered pipeline
----------------
1. Read selected pinned JSONL files in physical order, count raw/parseable rows.
   Optional exact metadata.source exclusions for tool_calling run before
   conversion, validation and within-subset curation.
2. Copy canonical source roles/fields and result values with raw isolation;
   explicitly classify null legacy search function_call and call indexes.
3. Repair legacy required and reconcile complete definitions; reject collisions.
4. Validate structure/object arguments and source ID/name linkage; align results
   to unchanged call order under FilterConfig.align_results (default True).
   Validate active capabilities and supported JSON Schema.
5. Apply audited credential/job/calendar grounding and asynchronous lifecycle
   gates. Pinned data.gov.sg starts require matching poll arguments and a
   visible download URL. Grounded external status-only polling is retained.
   Upstream judging and assistant context-only masks do not reject otherwise
   valid targets.
6. Apply optional native reasoning removal and the post-validation system override.
7. Recheck final structure, linkage/cardinality, capabilities/schema, source
   context, asynchronous lifecycle and nonempty complete assistant termination.
   The transforms affect only assistant content and system positions, allowing
   proven source pairing to be reused while final batch coordinates are rebuilt.
8. Curate Levels 1, 1.5 and 2 cumulatively within each released subset when
   configured; retain original winners and publish aggregate-only Levels 3--5.
   Never curate across subsets/versions or cap domain populations. Reports
   record source, drops, transformations, filtered count and actual final count
   only after successful exhaustion; closing early does not claim completion.
"""

from dataclasses import dataclass

from .base import FilterConfig
from .curation import CurationConfig
from .nemotron import convert_row, iter_dataset, pipeline
from .source import jsonl_lines

DATASET_ID = "nvidia/Nemotron-SFT-Agentic-v2"
DATASET_REVISION = "7c804833427f633ccd53b582dbf02525fd680f78"
_FILES = {
    "interactive_agent": "data/interactive_agent.jsonl",
    "search": "data/search.jsonl",
    "tool_calling": "data/tool_calling.jsonl",
}
_ALL_SPLITS = tuple(_FILES)


@dataclass(slots=True)
class NemotronAgenticV2Config:
    splits: tuple[str, ...] = _ALL_SPLITS
    exclude_tool_calling_sources: tuple[str, ...] = ()
    interactive_agent_domain_cap: int | None = None
    drop_conflicting_duplicate_tools: bool = False
    drop_invalid_source_tool_linkage: bool = False
    deduplicate_within_subset: bool = True
    publish_equivalence_metadata: bool = True


def _source_lines(split: str):
    return jsonl_lines(DATASET_ID, DATASET_REVISION, [_FILES[split]])


def _convert_row(raw, split: str, line: int = 1):
    return convert_row(raw, dataset=DATASET_ID, split=split, line=line, version=2)


def _pipeline(filters: FilterConfig, split: str = "tool_calling"):
    return pipeline(filters, anonymous=False, parallel=False)


def iter_load(
    dataset_config: NemotronAgenticV2Config | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
):
    config = dataset_config or NemotronAgenticV2Config()
    excluded = config.exclude_tool_calling_sources
    if (
        not isinstance(excluded, tuple)
        or any(not isinstance(source, str) or not source for source in excluded)
        or len(set(excluded)) != len(excluded)
    ):
        raise ValueError(
            "excluded tool-calling sources must be distinct nonempty strings"
        )
    if config.interactive_agent_domain_cap is not None:
        raise ValueError("Domain caps are not part of complete-trajectory curation")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    curation = curation_config or CurationConfig(
        max_level="level_2" if config.deduplicate_within_subset else None,
        audit=config.publish_equivalence_metadata or config.deduplicate_within_subset,
    )

    def raw_select(split: str, raw: object) -> str | None:
        if split != "tool_calling" or not isinstance(raw, dict):
            return None
        metadata = raw.get("metadata")
        source = metadata.get("source") if isinstance(metadata, dict) else None
        if source in excluded:
            return f"excluded_raw_source/{source}"
        return None

    return iter_dataset(
        dataset=DATASET_ID,
        revision=DATASET_REVISION,
        files=_FILES,
        splits=config.splits,
        version=2,
        lines=_source_lines,
        filters=filter_config or FilterConfig(strip_thinking=True),
        curation=curation,
        raw_select=raw_select if excluded else None,
    )


def load(
    dataset_config: NemotronAgenticV2Config | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
):
    rows, report = iter_load(
        dataset_config,
        filter_config,
        curation_config=curation_config,
        batch_size=batch_size,
    )
    try:
        return list(rows), report
    finally:
        rows.close()
