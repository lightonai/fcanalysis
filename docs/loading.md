# Using loaders

[Documentation](README.md) · Prerequisite: [How conversations are loaded](conversations.md)

Choose the adapter for your released source format. Start with
[loading](#load-and-iterate), [exporting](#export-retained-conversations),
[reasoning settings](#reasoning-and-system-settings),
[curation](#curation-settings), and [reports](#read-the-report).
[Prefix reconstruction](#assistant-target-prefixes) and
[Terminal](#terminal-conversations) apply only to those sources.

## Available loaders

Adapter modules document source revisions, files, conversion rules, defaults,
terms, and limitations. All except Turnstile provide `load()` and `iter_load()`.

| Module | Source selection | Processing |
| --- | --- | --- |
| [apigen_mt](../src/fcanalysis/loaders/apigen_mt.py) | APIGen-MT-5k, one train file | Native calls |
| [dolci](../src/fcanalysis/loaders/dolci.py) | Dolci-Instruct-SFT-Tool-Use; curation by `dataset_source` | Native calls |
| [nemotron_agentic_v1](../src/fcanalysis/loaders/nemotron_agentic_v1.py) | `NemotronAgenticV1Config.splits` | Native calls |
| [nemotron_agentic_v2](../src/fcanalysis/loaders/nemotron_agentic_v2.py) | `NemotronAgenticV2Config.splits`; optional `exclude_tool_calling_sources` | Native calls; exclusions precede curation |
| [nemotron_terminal](../src/fcanalysis/loaders/nemotron_terminal.py) | `NemotronTerminalConfig.configs`; optional local `path` | [Textual terminal](#terminal-conversations) |
| [toolmind](../src/fcanalysis/loaders/toolmind.py) | `ToolMindConfig.sources` selects published files | [Reconstruction](#assistant-target-prefixes), then native calls |
| [toolmind_web](../src/fcanalysis/loaders/toolmind_web.py) | ToolMind-Web-QA's `open-wiki-traj.jsonl` | Source textual calls converted to native calls |
| [toucan](../src/fcanalysis/loaders/toucan.py) | `configs` selects teachers; `ToucanConfig.subsets` selects subsets; optional local `path` | Native calls |
| [txt360](../src/fcanalysis/loaders/txt360.py) | `split="high"`, `"medium"`, or `"low"` | [Reconstruction](#assistant-target-prefixes), then native calls |
| [ultradata_tool_use](../src/fcanalysis/loaders/ultradata_tool_use.py) | `UltraDataToolUseConfig.sources` | Native calls |
| [turnstile](../src/fcanalysis/loaders/turnstile.py) | Experimental; optional local source/definition paths | Legacy `load()` only |

Turnstile lacks the staged schema, availability, ending, final-context, curation,
and iteration checks described here. Its legacy reasoning removal also trims
outside whitespace. Use the [weather example](adding-loaders.md) for a new adapter.

## Load and iterate

`load(...)` returns `(list[ConversationSample], LoadReport)`.
`iter_load(...)` returns an iterator and the report it updates. Use keyword
arguments: signatures vary by source, especially TxT360's leading `split`.

This loads Nemotron v2's complete tool-calling split with reasoning retained.
It can download source files:

```python
from contextlib import closing

from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.nemotron_agentic_v2 import (
    NemotronAgenticV2Config,
    iter_load,
)

rows, report = iter_load(
    dataset_config=NemotronAgenticV2Config(splits=("tool_calling",)),
    filter_config=FilterConfig(strip_thinking=False),
)
with closing(rows):
    for sample in rows:
        print(sample.sample_id, len(sample.messages))

print(report.summary())
```

Use adapter source selectors when excluded data must not influence
reconstruction or curation. Discarding returned rows cannot undo those effects.

### Memory, disk, and completion

`load()` materializes the output; `iter_load()` avoids keeping that entire list.
Selection still needs the complete comparison scope, so the first `next()` can
do substantial work. `batch_size` controls source reads, not the number of
returned samples; some JSONL adapters accept it while reading physical lines.

Reconstruction and curation use temporary disk storage with bounded caches.
Disk needs grow with input size and enabled comparisons; large individual rows
and dependencies still consume memory. This is not a fixed process-memory limit.

Use `closing(rows)` when stopping early. Only successful exhaustion sets
`report.final_count`; a break, close, or error leaves it incomplete. Receiving
the last sample is not exhaustion until the iterator advances to its end.

### Local data

TOUCAN and Nemotron-Terminal accept local `path` values with their expected
source layout; Turnstile also accepts source and definition paths. See the
adapter for filenames and configuration resolution. Local callers choose the
revision; other loaders resolve their pinned upstream files.

For normalized JSONL, use [`iter_samples_jsonl()`](analysis.md#save-and-read-canonical-conversations).
It checks serialized containers without rerunning loader validation.
For your own raw format, follow [Adding a loader](adding-loaders.md).

## Export retained conversations

[`scripts/export_selected_source.py`](../scripts/export_selected_source.py)
exports one of the 11 selected source configurations below. Run it from a
checkout after `uv sync --locked`, with output and scratch paths outside Git:

```sh
mkdir -p /path/to/scratch
uv run python scripts/export_selected_source.py \
  --source nemotron_v2 \
  --output /path/to/exports/nemotron_v2 \
  --temporary-directory /path/to/scratch
```

The output directory must not already exist. The script preserves reasoning,
selects sources before reconstruction/curation, and runs loader validation and
Levels 1, 1.5, and 2 with aggregate audit enabled. Adapters pin upstream revisions.
This recipe covers nine upstream datasets; TxT360 contributes three exports.

| `--source` | Selected input |
| --- | --- |
| `dolci` | All five Dolci Tool-Use partitions |
| `nemotron_v1` | Both Agentic v1 splits |
| `nemotron_v2` | All three Agentic v2 splits; exclude raw `metadata.source` values `xlam`, `xlam_tools`, and `when2call` from `tool_calling` |
| `terminal` | All four Nemotron-Terminal configurations |
| `toolmind` | Only the BUTTONInstruct, ToolACE, Glaive, and tau-train files |
| `toolmind_web` | ToolMind-Web-QA's `open-wiki-traj.jsonl` |
| `toucan` | All three teachers and four subsets; the separate SFT derivative is not added |
| `txt360_high`, `txt360_medium`, `txt360_low` | The corresponding TxT360 agent effort split |
| `ultradata` | UltraData Tool-Use, excluding `toolmind_graphsyn` |

The script writes:

- `canonical.jsonl.gz`: full `ConversationSample` records, including `raw` and
  annotations. Use `messages` and `tools` for model input.
- `report.json`: the loader report, conversation and assistant-message counts
  by dataset, elapsed time, and `complete: true`. It is written only after the
  iterator finishes and its final count matches the export. An interrupted run
  can leave a partial JSONL file; use a fresh directory for a rerun.

The default `--min-free-gib 500` checks free space on the output filesystem every
10,000 exported conversations; its minimum is 10 GiB. Configure it for your disk,
and allow separate space for the source cache and curation scratch files.

These are counts within each source, before mixture curation or model-specific
training admission. [Mixture curation](analysis.md#curate-a-mixture-of-retained-streams)
can combine the ten native exports; keep Terminal separate. Assistant-message
counts are candidates for supervision, not supervised-token counts. See
[token accounting](analysis.md#count-tokens).

This selection recipe does not establish training or redistribution rights.
Dataset terms still apply; mixed Dolci/TxT360 inputs and third-party tool results
require their own review.

## Reasoning and system settings

[`FilterConfig`](../src/fcanalysis/loaders/base.py) controls supported transforms:

| Setting | Behavior |
| --- | --- |
| `strip_thinking=True` | Remove supported native assistant reasoning. Preserve visible tools, arguments/results, and ordinary explanations. |
| `strip_thinking=False` | Retain native reasoning; source boundary and ending checks still apply. |
| `system_message_override=None` | Preserve normalized systems, including positions and empty values. |
| `system_message_override=""` | Remove systems, subject to final source checks. |
| `system_message_override="..."` | Replace all systems with this exact text at the first prior system position, or prepend if none existed; recheck the final view. |
| `align_results=True` | Align proven results to call order; the default. |
| `align_results=False` | Reject batches needing result movement. |

**Choose reasoning explicitly when constructing `FilterConfig`:**

```python
from fcanalysis.loaders.base import FilterConfig

preserve_reasoning = FilterConfig(strip_thinking=False)
remove_reasoning = FilterConfig(strip_thinking=True)
replace_system_and_remove_reasoning = FilterConfig(
    strip_thinking=True,
    system_message_override="Answer concisely.",
)
```

Omitting `filter_config` enables removal in native staged adapters. Passing
`FilterConfig()` retains reasoning because its field defaults to `False`,
even if you created it only to change another setting. Terminal defaults to
retention and allows only its [structured-reasoning removal](#terminal-conversations).

Shared removal recognizes balanced, nonnested `<think>`/`<reasoning>` spans
outside fenced, inline, and indented code, preserving every outside character.
Malformed boundaries fail removal; retain-mode boundary checks depend on the
adapter. User/tool text and visible tools named `think` remain intact.

The legacy `require_parseable_arguments`, `require_balanced_cardinality`,
`require_defined_functions`, and `require_valid_arguments` flags cannot disable
required staged checks. Unsupported lossy options, such as retry removal or
assistant merging, can raise `ValueError`.

## Curation settings

Curation selects original winners within each source scope.
[`CurationConfig`](../src/fcanalysis/loaders/curation.py) defaults to Levels 1,
1.5, and 2, plus aggregate audit:

```python
from fcanalysis.loaders.curation import CurationConfig

default_curation = CurationConfig()
exact_only = CurationConfig(max_level="level_1", audit=False)
keep_duplicates = CurationConfig(max_level=None, audit=False)
```

Pass it as `curation_config=...` to the loader. `max_level` accepts
`"level_1"`, `"level_1_5"`, `"level_2"`, or `None`; `audit` controls
group reporting independently. `temporary_directory` selects an existing
scratch directory for temporary storage.

For Nemotron v1/v2, an explicit `curation_config` takes precedence over legacy
`deduplicate_within_subset` and `publish_equivalence_metadata` settings.

| Level | Native comparison and winner |
| --- | --- |
| 1 | Complete messages and definitions, ignoring object-key and definition-list order. Arguments compare as parsed objects; results remain exact strings. Keep the first original. |
| 1.5 | Like Level 1, allowing line-ending and outer-blank-line differences in conservatively recognized plain non-tool content. Keep the first survivor. Spaces on retained lines, code, structured literals, arguments, results, and structured reasoning remain exact. |
| 2 | Omit assistant `content` and `reasoning_content` from comparison; keep positions, calls, results, users, systems, and definitions. Select the least total assistant-content characters, excluding structured reasoning; source order breaks ties. |

Levels run in order on survivors; earlier losers are never reconsidered.
**Each level compares the original values: Level 2 does not inherit Level 1.5's
whitespace normalization.** It applies to no-call conversations too, and can
choose between different answers. It is not a quality ranking.

IDs, annotations, and `raw` do not determine equality. Field absence, null,
scalar types, array order, and retained strings remain distinct. Source-proven
parallel batches may compare whole call/result pairs without order, preserving
multiplicity. Winners keep their original values and source order.
[Terminal uses different comparisons](#terminal-curation-and-counts).

With `audit=True`, the final retained population is also grouped:

| Level | Native grouping |
| --- | --- |
| 3 | Rows with calls: retain definitions, systems, results, user positions, and calls; omit user content and assistant prose/reasoning, and remove assistants without calls. |
| 4 | Ordered call-name batches with multiplicity and permitted parallel sorting. No-call rows share an empty trace. |
| 5 | Complete unordered definitions, including empty tool lists. |

These groups never delete, cap, weight, or annotate rows. Disabling general
deletion does not disable validation, source selection, or maximal-prefix
selection. Keep complete scopes: independently curating arbitrary chunks can
lose the winner a whole-scope run would select.

## Read the report

`LoadReport.summary()` formats these fields; some names are historical:

| Field | Meaning |
| --- | --- |
| `raw_count` | Physical input rows read |
| `stage1_count` | Converted or preliminary source-shape-accepted rows, depending on adapter |
| `stage1_drop_reasons` | Exclusive failures at that first boundary |
| `stage1_issue_counts` | Diagnostics that may overlap |
| `dataset_config_count`, `dataset_config_drop_reasons` | Source-specific selection accounting; see the adapter |
| `filtered_count` | Final-view-valid candidates before prefix selection and curation |
| `filter_drop_reasons` | Later validation/exclusion reasons |
| `final_count` | Returned conversations, final only after successful exhaustion |
| `dataset_config_transform_counts` | Nested transforms, pipeline, reconstruction, and curation reports |
| `filter_config`, `strip_thinking_applied` | Recorded settings and reasoning-removal status, not a count of affected rows |

A rejected candidate contributes its first failing-stage reason. Nested stage
and reason totals can describe the same loss; do not add them together.
Issue counts can overlap, and transform counts can include later exclusions.

The [offline example](getting-started.md#run-a-loader-without-downloading-data)
reads 3 rows, converts 3, validates 2, and returns 1: one invalid argument and
one duplicate explain the losses. These are conversation counts, not training
targets. Prefix and Terminal reports add the source-specific counts below.

## Diagnose unexpected exclusions

Inspect the first reason and the original source row. Common failures are
unsupported fields/schemas, invalid arguments, unproven matches, missing prior
state, and incomplete or empty assistants. Unsupported schema means validity
cannot be established, not that the argument is necessarily wrong. See
[schema support](adding-loaders.md#schemas-availability-and-prior-state).

Do not remove constraints or source text just to pass validation. A legitimate
alternative interpretation belongs in the adapter with source evidence and tests.

## Assistant-target prefixes

ToolMind and TxT360 publish overlapping assistant-target prefixes. The exporter
retains the current target's reasoning but omits it from historical assistants.
The loader restores those historical messages before validating the candidate:

```text
row 1: user → assistant 1+ calls
row 2: user → assistant 1- calls → results 1 → assistant 2+ calls
row 3: user → assistant 1- calls → results 1 → assistant 2- calls → results 2 → assistant 3+ answer

+ complete target; - known historical omission
```

Rows 1 and 2 can donate assistants to row 3, although their own call endings
cannot be retained. Row 3 needs its own observations and proven matches.
Restoration never supplies results.

A donor must match the entire projected prefix and external context within
the same source scope. Only object-key order and the proven omitted field are
ignored; text, field presence, types, tools, and message/array order stay exact.
Shared IDs, similar questions, and adjacent rows do not prove a match.

| Adapter | Separate donor scopes | Projection |
| --- | --- | --- |
| ToolMind | Each of eight published files | Remove exactly one leading balanced `<think>...</think>` span; preserve all following text and complete external raw tools. |
| TxT360 | `high`, `medium`, `low` | Remove only the selected effort's string `think`, `think_fast`, or `think_faster` field; keep system-embedded tools in history. |

Both require a donor for every historical assistant. Exactly one distinct
complete message must resolve each boundary; identical copies are fine.
Missing/conflicting donors exclude at the first unresolved boundary. Donors
may occur later or fail their own endpoint check. Explicit empty reasoning is
supported; absent, null, non-string, or incompatible fields cannot become guessed
emptiness. ToolMind also rejects missing, inseparable, or extra native spans.

The full donor scope is gathered before validation. Restoration copies whole
messages and keeps the candidate's ID and `raw`; it runs even when reasoning
will be stripped. Different continuations after a shared assistant can reuse
its donor, but different earlier results prevent a later match.

After final-view validation, remove only valid strict whole-message prefixes
of valid longer candidates in the same context. This comparison uses restored
source histories captured before conversion or optional transforms. Invalid
longer rows cannot remove valid shorter rows; branches survive, equal-length
duplicates go to curation, and output stays in source order.

### Prefix counts

| Reconstruction field | Population |
| --- | --- |
| `input_rows` | Source-shape-accepted physical candidates |
| `missing_donor_rows`, `ambiguous_donor_rows` | Candidates failing their first unresolved boundary |
| `validation_dropped_rows` | Restored candidates failing conversion/validation |
| `validated_rows` | Valid candidates before prefix selection |
| `restored_rows`, `restored_messages` | Restored occurrences among valid candidates before selection |
| `removed_prefix_rows` | Valid strict prefixes removed |
| `output_rows` | Selected rows before general curation |

Input equals donor failures plus validation failures plus validated rows.
Validated equals removed prefixes plus output. Restoration counts overlap these
populations and may differ by reasoning mode. They do not count distinct donors,
recovered tasks, or training targets.
[Implementing reconstruction](adding-loaders.md#implementing-prefix-reconstruction)
explains the callback and anchor boundaries.

## Terminal conversations

Nemotron-Terminal records assistant JSON actions and user messages carrying
harness feedback. `tools=[]`; there are no synthetic native calls. The initial
user message retains the task, screen, and interface instructions. Later user
messages retain observations; source metadata and private tests stay outside
model input.

Responses contain `analysis`, `plan`, ordered `commands`, and optional
`task_complete`. Commands send keystrokes with waits in a persistent terminal;
empty keystrokes poll and control characters can interrupt input. Each batch
has one shared observation, even when the command list is empty. A screen does
not prove a process finished. Preserve source quoting, timing, and truncated
screens; runtime wait caps do not rewrite explicit durations.

### Parsing, reasoning, and completion

The [parser](../src/fcanalysis/loaders/_terminus_json.py) and
[interpreter](../src/fcanalysis/loaders/_nemotron_terminal_protocol.py) interpret
recorded text without executing it. Supported repairs, defaults, coercions,
and warnings require matching released feedback. Parser rejection differs from
an executed command returning an error; discarded batches never count as executed.
Supported failures and retries remain in the conversation.

Native reasoning uses an exactly leading `<think>` envelope with a nonempty
interior and one unambiguous `</think>` followed by newline or end. The loader
moves the verbatim interior into `reasoning_content`, removing only wrappers
and one joining newline. Unfinished code fences inside it do not hide the
delimiter. Ambiguous envelopes fail. Default loading retains this field;
`strip_thinking=True` removes only it. Visible JSON `analysis`/`plan` and
literal tags remain unchanged; native drafts never become executed actions.

Completion requires two accepted declarations:

1. The first receives the harness confirmation prompt and screen, leaving
   confirmation pending.
2. Parser rejection leaves it pending; accepted false/absent completion clears it.
3. A second accepted declaration while pending closes the episode. Later
   interaction is invalid.

The final consumed object needs literal `task_complete: true`, literal
`commands: []`, string `analysis`/`plan`, no duplicate prose keys, and no
automatic repair. Supported surrounding text/fences and nonrepair warnings
can remain. A single declaration at file end, coerced completion, or unobserved
final commands/polls cannot close the episode.

After transforms, every assistant needs non-whitespace content or structured
reasoning. Removing a historical reasoning-only draft can therefore exclude
the whole candidate. Nonempty system overrides are unsupported; empty override
is a no-op. Settings cannot bypass source schema, producer, role order, initial
interface, episode length, feedback, or confirmation checks.

### Terminal curation and counts

Curate across all files of each published configuration, returning unchanged
winners in source order:

| Level | Terminal comparison |
| --- | --- |
| 1 and 1.5 | Exact complete content, including native reasoning, JSON spelling, and screen whitespace |
| 2 | Only for strict, complete, warning-free JSON: omit native reasoning and `analysis`/`plan`; preserve all other fields, command order/multiplicity/durations, completion, and user feedback. Other responses keep full content/reasoning. Shortest original assistant content wins, excluding structured reasoning; source order breaks ties. |
| 3 | Audit rows with interpreted commands using the conservative Level 2 view; replace only the initial task/screen with the fixed interface prefix, keeping later observations. |
| 4 | Audit ordered batches of `send_keys` labels plus each accepted `completion_request`, including final confirmation. Keep empty accepted batches; skip parser-rejected responses. |
| 5 | Audit the fixed terminal interface. |

Commands are never interchangeable by permutation. Audit groups never delete
or weight rows. Native-call analyses and mixture comparisons do not interpret
these textual actions.

Per-configuration reports count physical, converted, validated, and retained
rows. `validated_protocol_issues` counts occurrences before curation;
`completion_requests` counts confirmation prompts, excluding the final
confirming assistant, unlike Level 4's trace. Transform counts may include later
exclusions. Reasoning modes produce different populations.

Acceptance establishes this recorded protocol, not task success or an exact
historical runtime. Known resets, handoffs, and incompatible interfaces are
unsupported. See [extending textual protocols](adding-loaders.md#extending-textual-protocols).
