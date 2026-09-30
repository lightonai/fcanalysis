"""Load complete Nemotron Agentic v1 conversations through shared stages.

Origin and license
------------------
NVIDIA's pinned dataset card is
https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/blob/650d590978ca35c8f1ecea2faf136e5fac421b62/README.md .
The corresponding Nano model report (https://arxiv.org/abs/2512.20848,
section 3.1.2) describes synthetic conversational tool use. The developer hub
is https://github.com/NVIDIA-NeMo/Nemotron ; neither the pinned card nor the
report identifies a generation-code commit or a call/result serialization
implementation for these exact files. The card declares CC BY 4.0 and cites
Apache 2.0 for inherited Glaive-Function-Calling-v2. Other reused public tool
sets and their row-level license mappings are unknown; the developer hub's
code license does not license those inherited data automatically.

Generation, schema and uncertainty
----------------------------------
The card describes separate LLM user, assistant and tool simulators with LLM
judging. Interactive generation uses Qwen3 thinking/instruct, Qwen3-32B and
GPT-OSS-120B; general tool calling uses Qwen3-235B variants and USA personas.
The pinned JSONLs contain interactive_agent (19,028 rows) and tool_calling
(316,094). These are complete released conversations, not a verified prefix
family release. No prefix collapse or reasoning restoration is inferred.
Top-level uuid, license, used_in and interactive reasoning are audit metadata.
messages and tools supply the training context; systems are retained verbatim,
including absent/empty/multiple systems. Observed roles are system/user/
assistant/tool. Complete function definitions include name, description,
parameters, optional strict and 83 legacy function-level required fields; the
last is explicitly moved into parameters.required without discarding either
constraint. Unknown model-visible source fields quarantine.

Interactive calls use JSON-string arguments with complete call/result IDs.
Tool-calling arguments are objects (668,144) or lists (7; non-object calls
quarantine); results are structured JSON or strings, with call IDs but no
result IDs/names. Singletons pair unambiguously. A full source census found
60,466 multi-call batches and 6,846 unbalanced batches; no inspected producer
contract proves the anonymous multi-call pairing, so those unresolved batches
quarantine. This does not infer pairing from equal counts. Source IDs are
removed only after validation. Tool results are serialized without rewriting
string values; no hidden state is invented from UUIDs, judging or metadata.

Reasoning and bounded context audit
----------------------------------
Native reasoning_content is present on all 70,794 interactive assistants
(45,680 nonempty) and 1,127,098 tool-calling assistants (all nonempty; four
assistants omit it). The pinned census found one literal '<reasoning>' opening
in tool-calling content and no closed native tag spans in interactive content.
FilterConfig.strip_thinking removes structured native reasoning and exact
safely closed think/reasoning spans via the shared representation-aware stage;
unmatched boundaries quarantine when stripping. No-argument loading strips
native reasoning;
explicit strip_thinking=False retains it. Ordinary prose, code and declared
reasoning-like tools remain model-visible behavior. Trainer-specific history reasoning,
prefix splitting, loss masks and weighting are outside the loader.
Final replies containing only recognized native reasoning are incomplete in
either mode; keeping reasoning does not turn it into an answer. Native control
tokens alone also provide no answer. Neither does an open-only native tail
preceded solely by balanced native spans and whitespace. Code literals and
other ambiguous boundaries retain their configured handling.

Raw interactive row 0 invents authenticate_user.verification_code='123456'
before any visible code; that row fails the credential gate. The source helper
lists audited credential slots and authentication functions, checking exact
prior user/system values or linked result leaves, never assistant inventions,
definition examples or future state. This is a bounded literal-state check,
extended by a full description/name census to finite exact consuming signatures
(including nonstandard API-key fields). Documented Bearer wrappers and observed
hyphen-separated individual OTP digits may use prior visible constituent values;
source text and arguments stay unchanged. Generated passwords, composite/DOB
credentials, derived signatures, composed cookies and ambiguous demo/default
contracts are held outside this extension. Explicit placeholder defaults do not
provide a missing API key.
This remains a bounded context check,
not general semantic certification of arbitrary policies, confirmations or
time calculations. No discovery grant is assumed merely from a function name;
unknown downstream calls quarantine. The asynchronous source grammar grounds
job IDs and checks
SERP/SimilarWeb start-to-completed-result lifecycles, preserving grounded
external status-only queries and retries. These checks use exact listed
operations and structured statuses, not a generic poll/async keyword rule.

The temporal audit also finds first hotel-availability calls converting tonight
or tomorrow into absolute dates with no visible clock. Reviewed consuming
booking/scheduling/travel slots require a prior explicit target date or labeled
current date for that conversion. Historical facts do not become clocks. This
bounded check abstains after a prior tool result, whose temporal meaning needs
a separate source contract. Explicit user date ranges and partial dates that
need year association also cause abstention; they are not missing-date proof.
The gate does not certify arbitrary date arithmetic.
The shared schema contract deliberately rejects date-time and other unsupported
formats even in unused definitions, rather than relying on optional packages.

Ordered pipeline
----------------
1. Read the pinned subset JSONL in physical order; count physical/parseable rows.
2. Copy source-visible fields, serialize structured results and preserve raw.
3. Apply the named legacy required repair and reconcile full definitions.
4. Validate structure/object arguments and source linkage; align proven results
   to unchanged call order by default (FilterConfig.align_results). Validate
   active static tools and supported JSON Schema; distinguish unsupported schemas.
5. Apply source credential/job/calendar gates and asynchronous lifecycle checks;
   upstream scores/masks are not gates.
6. Apply optional native reasoning removal and post-validation system override.
7. Recheck final structure, linkage/cardinality, capabilities/arguments, source
   context, asynchronous lifecycle and complete nonempty assistant termination.
   Only content and system positions changed, permitting reuse of source
   pairing for final coordinates.
8. Apply cumulative Levels 1, 1.5 and 2 within each selected released subset
   when configured. Keep original winners; Levels 3--5 are aggregate audit only.
   Versions/subsets are never combined. Publish source, stage, transformation,
   curation and actual final counts after successful iterator exhaustion.
"""

from dataclasses import dataclass

from .base import FilterConfig
from .curation import CurationConfig
from .nemotron import convert_row, iter_dataset, pipeline
from .source import jsonl_lines

DATASET_ID = "nvidia/Nemotron-Agentic-v1"
DATASET_REVISION = "650d590978ca35c8f1ecea2faf136e5fac421b62"
_FILES = {
    "interactive_agent": "data/interactive_agent.jsonl",
    "tool_calling": "data/tool_calling.jsonl",
}
_ALL_SPLITS = tuple(_FILES)


@dataclass(slots=True)
class NemotronAgenticV1Config:
    splits: tuple[str, ...] = _ALL_SPLITS
    # Compatibility flags cannot disable the production invariants.
    drop_orphan_samples: bool = False
    drop_empty_system: bool = False
    drop_conflicting_duplicate_tools: bool = False
    deduplicate_within_subset: bool = True
    publish_equivalence_metadata: bool = True


def _source_lines(split: str):
    return jsonl_lines(DATASET_ID, DATASET_REVISION, [_FILES[split]])


def _convert_row(raw, split: str, line: int = 1):
    return convert_row(raw, dataset=DATASET_ID, split=split, line=line, version=1)


def _pipeline(filters: FilterConfig, split: str = "interactive_agent"):
    return pipeline(filters, anonymous=split == "tool_calling", parallel=False)


def iter_load(
    dataset_config: NemotronAgenticV1Config | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
):
    config = dataset_config or NemotronAgenticV1Config()
    if config.drop_empty_system:
        raise ValueError("Empty source systems are valid and must be preserved")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    curation = curation_config or CurationConfig(
        max_level="level_2" if config.deduplicate_within_subset else None,
        audit=config.publish_equivalence_metadata or config.deduplicate_within_subset,
    )
    return iter_dataset(
        dataset=DATASET_ID,
        revision=DATASET_REVISION,
        files=_FILES,
        splits=config.splits,
        version=1,
        lines=_source_lines,
        filters=filter_config or FilterConfig(strip_thinking=True),
        curation=curation,
    )


def load(
    dataset_config: NemotronAgenticV1Config | None = None,
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
