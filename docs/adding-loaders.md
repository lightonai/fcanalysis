# Adding a loader

[Documentation](README.md) · Prerequisite: [How conversations are loaded](conversations.md)

An adapter decodes one source format and supplies its validation rules around
shared functions. There is no required loader base class.

Follow steps 1–6 for the ordinary path. Then consult
[call/result matching](#matching-calls-and-results),
[schemas and state](#schemas-availability-and-prior-state),
[prefix reconstruction](#implementing-prefix-reconstruction), or
[textual protocols](#extending-textual-protocols) if your source needs them.
The [component map](#shared-components) links to the implementation.

The complete [weather adapter](examples/weather_loader.py) reads synthetic
JSON lines and uses the real pipeline and curator:

```sh
uv run python docs/examples/weather_loader.py
```

## 1. Describe the source

Put the supported source contract in the adapter's module docstring:

- Origin, release revision, files, splits, traversal order, and declared terms.
- What one row represents and which fields are visible to the model.
- Roles, tool definitions, calls, results, and native reasoning encodings.
- Evidence for call/result associations, available tools, and required prior state.
- Permitted conversions, repairs, exclusions, transforms, and final ending.
- Scope of reconstruction and curation, defaults, and limits of verification.

Unsupported model-visible fields need explicit exclusions, not silent removal.
Source scores and generator metadata stay outside model input. Optional quality
selection may narrow the input, but a score does not prove validity or correctness.

The tutorial's source has exactly these root fields:

| Field | Meaning |
| --- | --- |
| `id` | Existing string/integer source ID |
| `dialogue` | Canonical-role messages; at most one call per assistant, followed by its result |
| `functions` | Bare function definitions, all initially available |

Arguments may be objects or JSON object strings; result content may be absent,
text, or null.
An assistant answer ends each conversation. The caller supplies one complete
`example/train` scope in source order. This source has no dynamic discovery,
background jobs, or omitted history. Retain mode checks reasoning only at the
endpoint; removal validates the shared inline boundaries.

## 2. Convert into independently owned data

The example's `convert_row()` validates root fields, copies messages, and wraps
bare definitions:

```python
def convert_row(raw: Any) -> ConversationSample:
    if not isinstance(raw, dict) or set(raw) != {"id", "dialogue", "functions"}:
        raise Reject("unsupported_source_fields")
    if type(raw["id"]) not in (str, int):
        raise Reject("invalid_source_id")
    if not isinstance(raw["dialogue"], list) or not isinstance(raw["functions"], list):
        raise Reject("unsupported_source_shape")
    return ConversationSample(
        dataset=DATASET_ID,
        sample_id=raw["id"],
        messages=deepcopy(raw["dialogue"]),
        tools=normalize_tools(raw["functions"], bare=True),
        raw=raw,
    )
```

This excerpt uses imports and `DATASET_ID` from the complete example.
The pipeline checks message details. Normalized containers must not alias
`raw`; preserve the original row, field absence, null, types, order, and text.

Use the [normalization helpers](../src/fcanalysis/loaders/normalization.py):

| Helper | Behavior |
| --- | --- |
| `parse_json()` | Reject malformed JSON, duplicate keys, and nonfinite/non-JSON values; preserve large integers. |
| `normalize_arguments()` | Serialize structured objects; preserve existing object strings. |
| `serialize_result()` | Preserve result strings; serialize structured JSON values. |
| `normalize_tools()` | Copy a known bare or enveloped definition list. |
| `reconcile_tools()` | Keep first identical same-name definitions and first-seen distinct names; reject conflicting complete definitions. |

Keep unused definitions and every supplied envelope/schema field. Extract
embedded definitions only from a proven template, preserving other system text.
Non-JSON call syntax and bundled results need source-specific parsers; arbitrary
expressions must never execute. A source-authorized schema repair must preserve
constraints and instance values. Known single-type aliases can be normalized;
unknown aliases and union entries are not guessed.

Raise `Reject("reason")` for recognized data exclusions. Let unexpected
exceptions propagate so programming bugs cannot silently look like fewer rows.

## 3. Assemble source and final checks

A [`Pipeline`](../src/fcanalysis/loaders/pipeline.py) runs named stages in order.
Each receives `RowState` and either succeeds or raises `Reject`.
The weather example uses:

| Stage | Purpose |
| --- | --- |
| `reconcile_definitions` | Resolve identical/conflicting definitions. |
| `validate_structure` | Check shapes and parse/serialize arguments. |
| `validate_source_protocol` | Enforce this source's one-call-per-assistant limit. |
| `link_calls` | Prove matches, align results, remove temporary linkage fields. |
| `validate_capabilities` | Check schemas, tool availability, and arguments. |
| Configured reasoning removal and system override | Transform the validated source. |
| Final structure, linkage, and capability checks | Recheck the affected output and rebuild coordinates/caches. |
| `validate_termination` | Require a visible final answer and nonempty assistant decisions. |

`Pipeline.process(sample)` wraps the sample without copying it. It returns
a successful `RowState` or `None`, recording passed stages, first failures,
and transforms. Source checks precede transforms; an override cannot hide a
source failure. Recheck every property a transform can affect.

### State after transformations

`state.batches` holds proven call/result coordinates and any parallel permission.
`state.parsed_arguments` caches arguments by message/call coordinates. Neither
belongs in the output sample.

Once IDs are removed, rerunning `link_calls()` needs independent evidence.
Weather's single adjacent call/result pair supplies it, even after a system
override shifts indices. For other sources, either retain sufficient evidence
or justify reuse of bindings across transforms that preserve the associations,
updating coordinates when needed. System override clears argument caches.
Setting `positional=True` merely to make a second pass succeed is incorrect.

An alternative ending also needs a documented source rule. The helper
`validate_termination(action_only=True)` exists, but no registered adapter uses
it. Producer-specific control markers need adapter checks beyond shared native
reasoning checks. Keep valid retries and errors; never truncate a failed row
or invent an answer to make it complete.

## 4. Curate a complete scope

Pass accepted states as
`CurationInput(state.sample, state.batches, state.parsed_arguments)`.
The example calls `curate()` with
`CurationScope(DATASET_ID, "example", "train")`. It returns original winners
in source order; [`CurationConfig`](loading.md#curation-settings) controls selection.

A scope names dataset, subset, and split within the adapter's fixed release.
Arbitrary chunks cannot be curated independently and then merged or re-curated:
earlier levels can discard a candidate that would win in the full population.
Reconstruction likewise needs the full donor scope before validation and
prefix selection before curation.

Exact complete tool environments are safe native deletion partitions because
every deletion comparison includes them. Level 4 audit groups can cross those
partitions; combine matching groups instead of adding distinct-group counts.

`curate(comparison_factory=...)` allows a protocol-specific comparison;
`canonical_comparison()` supplies the native view. Comparison values never
replace the stored winner. To combine already retained datasets, use the
[mixture utility](analysis.md#curate-a-mixture-of-retained-streams).

## 5. Expose iteration and reporting

Follow the example's `iter_load()` and `load()`:

1. Validate options and create the report and pipeline.
2. Count physical inputs and known conversion failures.
3. Feed final-view-valid states into the complete-scope curator.
4. Yield retained samples; set `final_count` only after successful exhaustion.
5. Close owned resources in `finally`; implement `load()` by consuming this iterator.

The tutorial leaves file ownership with its caller. Put a copy of
`weather_loader.py` beside your script and read your source like this:

```python
from contextlib import closing

from weather_loader import iter_load

with open("/path/to/weather.jsonl", encoding="utf-8") as source:
    rows, report = iter_load(source)
    with closing(rows):
        for sample in rows:
            print(sample.sample_id, len(sample.messages))

print(report.summary())
```

An adapter that opens files must close them on exhaustion, error, and early
close. [Source helpers](../src/fcanalysis/loaders/source.py) read pinned Parquet
batches and physical JSONL lines. Keep the
[reporting boundaries](loading.md#read-the-report) distinct: physical, converted,
valid, prefix-selected, and curated rows; first failures; overlapping issues;
and transforms that may include later exclusions.

## 6. Verify and integrate

Test small synthetic rows at the boundaries:

| Case | Expected result |
| --- | --- |
| Ordinary complete row | Correct roles, values, definitions, and source ID |
| Mutate output | `raw` and original inputs unchanged |
| Contradictory IDs or missing results | Exclusion without weaker-evidence fallback |
| Unsupported schema / invalid arguments | Distinct failures; no dropped constraints or inserted defaults |
| Reasoning modes and system override | Exact preserved text and affected final checks |
| Tool-ending or empty final assistant | Exclusion under the ordinary ending |
| Valid duplicates | Deterministic unchanged winners and reconciled counts |
| Empty input, early close, programming error | Correct completion state, cleanup, and error propagation |

See [pipeline](../tests/unit/test_loader_pipeline.py),
[APIGen-MT](../tests/unit/test_loader_apigen_mt.py), and
[curation tests](../tests/unit/test_loader_curation.py). Add adversarial tests for
your source grammar, discovery timing, reconstruction, or textual protocol.

Place the adapter in `src/fcanalysis/loaders/` and tests in `tests/unit/`.
For command-line dispatch, add it to the existing
[`LOADER_MODULES` mapping](../src/fcanalysis/loaders/__init__.py) and check callers'
argument handling. Update the [catalogue](loading.md#available-loaders).

[tests/matrix.py](../tests/matrix.py), [fixture tooling](../tests/tools/generate_fixtures.py),
and [E2E tests](../tests/e2e) cover pinned corpus regression. State whether
verification used synthetic cases, a source sample, or an exhausted release.
A unit-test pass does not establish full-corpus acceptance. Keep source data
and bulky generated outputs outside Git.

## Matching calls and results

`link_calls()` requires one uninterrupted result block immediately after each
assistant batch, with exactly one result per call. Extra tool messages are
orphans. Evidence is checked in this order:

| Evidence | Required match |
| --- | --- |
| Any non-null call/result ID | Complete nonempty string IDs, unique on each side, with equal sets. No fallback after contradiction. |
| No IDs; source-decoded echoes | Every result echoes either name plus all supplied object arguments, or name only, uniformly within the batch. Both sides need unique matching signatures. |
| No IDs/echoes; complete outer result names | Distinct call names and result names with equal sets. |
| No independent name match | Singleton adjacency, or a documented source positional protocol. Counts alone do not prove a multi-call match. |

An echo repeats call information in result content. Enable its interpretation
through an adapter-owned `result_echo` decoder; ordinary payload keys are not
evidence. If any echo is recognized, all results need valid echoes even with
IDs. Supplied names and arguments must agree with the bound calls. Complete
IDs can disambiguate repeated calls or mixed echo forms; without them,
repeated signatures cannot fall back to position. Argument equality preserves
scalar types, array order, and strings; only object-key order is ignored.

Partial outer names may supplement independently proven positions; each
non-null name must agree, and empty strings are invalid. All-absent/all-null
IDs mean no ID evidence. A source adapter may remove IDs proven to be producer
annotations, but this is not a repair for broken references.

Matching aligns whole results to unchanged call order and removes temporary
IDs and outer names. With `align_results=False`, required movement fails as
`unproven_result_reordering`. Contents, including echoes, remain intact.

`positional=True` asserts source-proven result order. `parallel=True` separately
permits unordered whole-pair comparisons in curation; matching IDs or a wide
batch alone cannot establish it. Multiplicity, argument array order, and
boundaries between successive batches remain significant.

Current diagnostic precedence is in
[`link_calls()`](../src/fcanalysis/loaders/pipeline.py): repeated call signatures
report ambiguity; repeated/mismatching result signatures report invalid linkage.
Without independent pairing, ambiguity can precede validation of a partial
malformed outer name.

## Schemas, availability, and prior state

`validate_capabilities()` checks initial definitions, including unused ones,
and validates availability and arguments at each call. Omitted `parameters`
adds no schema-specific constraint; arguments still must be an object.

The [schema checker](../src/fcanalysis/loaders/schema.py) uses a supported declared
`$schema`, or Draft 2020-12 by default. Its support check accepts the selected
validator's keywords plus an explicit annotation allowlist, and only the formats
`date`, `ipv4`, `ipv6`, and `uuid`. Unknown drafts, keywords, formats, and
external references are rejected. Local references are resolved when arguments
are checked; resolution failures report `unresolved_tool_schema`, separately
from `invalid_arguments`. Schema defaults are never inserted.

Use `discovery=...` to decode visible capability grants, replacements, or
revocations. They apply after the message that establishes them; a discovery
result cannot authorize earlier or sibling calls. Conflicting definitions need
explicit source permission to replace. Use `target_validator` for checks
against prior evidence.

[Context helpers](../src/fcanalysis/loaders/context.py) expose borrowed preceding
evidence for immediate use without mutation or retention; they do not implement
discovery. Source rules identify eligible evidence channels. Schema examples,
assistant guesses/native reasoning, future results, audit metadata, and echoed
call arguments cannot supply missing external state.
[Dolci's gates](../src/fcanalysis/loaders/dolci_context.py) show concrete checks.

For background work, `validate_lifecycle()` consumes source-decoded job events
with previously grounded IDs. Started jobs need completion or supported visible
abandonment; pending status is not abandonment. The helper requires a
payload-bearing successful closure, including a fetch when required, even
without a summary. Summaries also need completion and payload. Its default
covers every observed job; `require_completion_for="started"` permits an
explicit external-status exception without relaxing grounding or summaries.

Document the source scope, evidence, and tests for any rule replacing shared
behavior. Configuration switches cannot silently disable required checks.

## Implementing prefix reconstruction

Establish the exporter's exact historical omissions first.
[ToolMind and TxT360](loading.md#assistant-target-prefixes) provide examples;
shared labels or similar rows cannot establish the same contract.

[`PrefixRecord`](../src/fcanalysis/loaders/reconstruction.py) contains:

| Field | Adapter supplies |
| --- | --- |
| `value` | Original payload passed to validation |
| `scope`, `context` | Complete donor scope and exact external context |
| `messages`, `projected_messages` | Full source history, and the same history omitting only proven lost fields for comparison |
| `required` | Ordered, unique historical indices needing donors |
| `anchor` | Final message index if it supplies a complete assistant target, otherwise `None`; never also required |

`reconstruct_prefixes(records, validate=..., config=...)` gathers all donors,
restores owned message copies, and calls `validate(original_payload, messages)`.
Convert and validate the whole candidate; return the accepted value or `None`.
Do not mutate originals; let programming exceptions propagate.

The engine captures restored histories before the callback can transform them,
selects valid maximal prefixes, and returns callback values in source order.
Consume the result into curation inside its context manager or close it in
`finally`. Its [counts](loading.md#prefix-counts) precede general curation.

There is no separate donor-only record type. A record ending
`user → assistant call → results` cannot use its earlier assistant as `anchor`.
Supporting it requires distinguishing donor boundary, candidate endpoint, and
accounting. A truncated proof cannot justify a longer returned history. Never
move observations, borrow future results, or split a call batch. User corrections
are tool observations only under a source rule establishing that interpretation.

## Extending textual protocols

[Terminal's behavior](loading.md#terminal-conversations) replaces native call
interpretation with its released text interface. Its
[adapter](../src/fcanalysis/loaders/nemotron_terminal.py) checks the transcript
before transforms, then final structure, interface, and nonempty assistants.
Permitted transforms preserve visible protocol text and order, allowing proof
reuse. `Pipeline(state_factory=...)` caches parsed responses;
`remove_reasoning(inline=False)` removes only decoded native reasoning.

The [interpreter](../src/fcanalysis/loaders/_nemotron_terminal_protocol.py) guards
the attributed parser against duplicate execution/completion keys, nonfinite
values, unsupported shapes/Unicode, and negative/nonfinite durations. Historical
duplicate `analysis`/`plan` keys use the parser's last value but retain all
source text; they cannot qualify for final submission or prose-omitting comparison.

Feedback establishes interpretation. Historical `ERROR:` in combined parser
feedback rejects the whole batch; completion can also discard faulty commands.
Supported feedback variations are limited to unknown-field ordering and a known
older Python closing-delimiter diagnostic. Preserve source text even when the
parser consumes a repaired view. These exceptions do not loosen native-call JSON.

The current adapter excludes resets and handoffs. Supporting them requires
actual per-request context or exact exporter-backed replacement records; a
flattened transcript plus summary cannot establish what the model received.
Distinguish tasks, attempts, participants, segments, and generations. Retain the
actual replacement context and visible compression triggers/feedback; do not
splice discarded history into it. Missing numeric context budgets alone do not
invalidate recorded compression. Earlier segments need a supported ending;
summarizer supervision needs its own released input/output.

Each participant receives only its actual observations. Hidden worker reasoning
cannot become coordinator context, nor automatic spawning a fabricated
delegation call. Context resets do not imply environment resets. Distinguish
action feedback, process completion, segment closure, submission, attempt end,
and verifier success. Test these boundaries and retained source projections
before claiming support; private state or tests cannot fill missing context.

## Shared components

| Component | Responsibility |
| --- | --- |
| [base.py](../src/fcanalysis/loaders/base.py) | Configuration, reports, system overrides |
| [source.py](../src/fcanalysis/loaders/source.py) | Pinned source reads |
| [normalization.py](../src/fcanalysis/loaders/normalization.py), [legacy_tools.py](../src/fcanalysis/loaders/legacy_tools.py) | Strict values, serialization, explicit legacy conversion |
| [schema.py](../src/fcanalysis/loaders/schema.py) | Schema support and argument checks |
| [pipeline.py](../src/fcanalysis/loaders/pipeline.py) | Row state and validation/transform stages |
| [context.py](../src/fcanalysis/loaders/context.py) | Prior-evidence traversal and calendar checks |
| [reconstruction.py](../src/fcanalysis/loaders/reconstruction.py) | Donor restoration and valid prefix selection |
| [curation.py](../src/fcanalysis/loaders/curation.py) | Temporary storage, selection, aggregate groups |

Population operations use temporary disk storage with bounded caches. They
do not provide distributed loading or a fixed process-memory ceiling. See
[iteration and cleanup](loading.md#memory-disk-and-completion).
