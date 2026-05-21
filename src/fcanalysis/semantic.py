import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

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


SYSTEM_PROMPT = """\
You classify no-FC (no function call) turns in function calling training data.

You will see: (1) the available tools, (2) the full conversation with message indices,
and (3) which user turns had NO tool calls. For each marked turn, determine:

1. **Category** -- pick exactly one from the list below.
2. **Justified** -- was the model correct not to call any tool?
3. **Reasoning** -- one sentence explaining why.

### Justified categories (model was RIGHT not to call)

- **S3_CLARIFICATION**: A tool matches the query but required parameters are missing. \
The model correctly asks the user for the specific missing info.
- **S4_DIRECT_ANSWER**: No available tool matches the user's query. \
The model correctly answers from its own knowledge.
- **S5_DECLINE**: No tool matches and the model cannot answer. \
It correctly explains what tools are available and declines.
- **S6_NEAR_MISS**: A tool looks superficially related but does not actually fit \
(e.g., tool computes median but user asks for mean). Model correctly explains the mismatch.
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
Even for short queries. Even if the model CAN answer correctly on its own.
- **ANTI_UNJUSTIFIED_REFUSAL**: A tool matches and params are sufficient, \
but the model refuses, asks unnecessary questions, or says it can't help. \
Includes: asking for optional params, re-asking for info already provided, \
claiming it doesn't have access when the tool is right there.
- **ANTI_PRESSURE_CAVE**: Model previously refused correctly, but now caves to \
user pressure and provides an inappropriate direct answer or fabricated explanation.

### Catch-all

- **OTHER_JUSTIFIED**: Justified no-FC that doesn't fit the categories above.
- **OTHER_UNJUSTIFIED**: Unjustified no-FC that doesn't fit the categories above.

### Key principles

- If a matching tool exists and required params are available, model SHOULD call it.
- Tools take priority over direct answers, even if the model could answer from knowledge.
- Short queries deserve tool calls just as much as long ones.
- Each turn is evaluated in full conversation context; look at what happened before.

### Output format

Respond with exactly this JSON structure (no extra text outside the JSON):

```json
{"classifications": [{"turn_index": <int>, "category": "<CATEGORY>", \
"justified": <bool>, "reasoning": "<one sentence>"}]}
```

One entry per turn listed in "No-FC Turns to Classify". \
`turn_index` must match the values provided."""


def build_prompt(sem_input: SemanticLayerInput) -> list[dict[str, str]]:
    turn_map = [
        {"turn_index": ti, "user_message_index": mi}
        for ti, mi in zip(
            sem_input.no_fc_turn_indices,
            sem_input.no_fc_user_message_indices,
            strict=True,
        )
    ]

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

    indexed_messages = [{**msg, "index": i} for i, msg in enumerate(cleaned_messages)]

    user_content = f"""## Sample Data

```json
{json.dumps({"tools": sem_input.tools, "messages": indexed_messages}, indent=2)}
```

## No-FC Turns to Classify

```json
{json.dumps(turn_map)}
```

Respond with JSON in the format specified above."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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


NUM_GENERATIONS = 5
MAX_RETRIES_PER_GEN = 3


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


class LeastConnectionsPool:
    """Least-connections load balancer across multiple OpenAI async clients.

    `openai` and `httpx` are imported lazily so this module can be imported
    in environments without the LLM dependencies (e.g. test harnesses).
    """

    def __init__(
        self,
        base_urls: list[str],
        timeout: int = 600,
        max_connections: int = 200,
    ) -> None:
        import httpx
        import openai

        self._clients = [
            openai.AsyncOpenAI(
                base_url=url,
                api_key="EMPTY",
                timeout=timeout,
                max_retries=0,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=max_connections,
                    ),
                    timeout=httpx.Timeout(timeout, connect=30.0),
                ),
            )
            for url in base_urls
        ]
        self._in_flight = [0] * len(self._clients)
        self._lock = asyncio.Lock()

    async def acquire(self) -> tuple[Any, int]:
        async with self._lock:
            idx = min(range(len(self._clients)), key=lambda i: self._in_flight[i])
            self._in_flight[idx] += 1
            return self._clients[idx], idx

    async def release(self, idx: int) -> None:
        async with self._lock:
            self._in_flight[idx] -= 1

    async def warmup(self) -> None:
        """Probe every backend; raises if any client cannot reach its endpoint."""
        await asyncio.gather(*[c.models.list() for c in self._clients])

    async def aclose(self) -> None:
        for client in self._clients:
            await client.close()


async def _single_generation(
    pool: LeastConnectionsPool,
    model: str,
    messages: list[dict[str, str]],
    expected_turns: list[int],
    json_mode: bool = False,
) -> dict[str, Any] | None:
    for _retry in range(MAX_RETRIES_PER_GEN):
        client, idx = await pool.acquire()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 0.0,
                "extra_body": {
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                },
            }
            if json_mode:
                num_turns = len(expected_turns)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "classifications": {
                                    "type": "array",
                                    "minItems": num_turns,
                                    "maxItems": num_turns,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "turn_index": {
                                                "type": "integer",
                                                "enum": expected_turns,
                                            },
                                            "category": {
                                                "type": "string",
                                                "enum": CATEGORIES,
                                            },
                                            "justified": {"type": "boolean"},
                                            "reasoning": {"type": "string"},
                                        },
                                        "required": [
                                            "turn_index",
                                            "category",
                                            "justified",
                                            "reasoning",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["classifications"],
                            "additionalProperties": False,
                        },
                    },
                }
            response = await client.chat.completions.create(**kwargs)
        finally:
            await pool.release(idx)

        if not response.choices:
            continue
        content = response.choices[0].message.content
        if not content:
            continue

        try:
            result = _parse_response(content)
        except ValueError:
            continue

        if _validate_response(result, expected_turns) is None:
            return result

    return None


async def classify_sample(
    pool: LeastConnectionsPool,
    model: str,
    sem_input: SemanticLayerInput,
    json_mode: bool = False,
) -> dict[str, Any]:
    messages = build_prompt(sem_input)
    expected_turns = sem_input.no_fc_turn_indices

    tasks = [
        _single_generation(pool, model, messages, expected_turns, json_mode)
        for _ in range(NUM_GENERATIONS)
    ]
    gen_results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [r for r in gen_results if isinstance(r, dict)]
    server_errors = [r for r in gen_results if isinstance(r, Exception)]

    if len(valid) < NUM_GENERATIONS:
        error_msg = (
            str(server_errors[0]) if server_errors else "insufficient_valid_generations"
        )
        return {
            "sample_id": sem_input.sample_id,
            "error": error_msg,
            "num_valid_generations": len(valid),
        }

    voted = _majority_vote(valid, expected_turns)
    voted["sample_id"] = sem_input.sample_id
    voted["num_valid_generations"] = len(valid)
    return voted


async def run_semantic_layer(
    sem_inputs: list[SemanticLayerInput],
    model: str,
    base_urls: list[str] | None = None,
    output_path: str | Path = "semantic_results.jsonl",
    concurrency: int = 768,
    json_mode: bool = False,
) -> None:
    """Run semantic classification on all samples with no-FC turns.

    Resume behavior: if ``output_path`` already exists, every row in it
    that does not contain ``"error"`` is treated as already classified
    and its ``sample_id`` is excluded from this run. Resume is keyed
    purely by ``sample_id`` and does not inspect which loader config
    produced those IDs. This means that writing two different sample
    populations (e.g. one run with ``--filter-mode all`` and another
    with ``--filter-mode none --txt360-seed-group last``) to the same
    output file will silently skip any ``sample_id``s already present
    from the prior run, even though those rows describe samples from a
    different population. Always point distinct loader configurations
    at distinct output paths.
    """
    if base_urls is None:
        base_urls = ["http://localhost:8000/v1"]

    output_path = Path(output_path)
    pool = LeastConnectionsPool(base_urls)

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
    print(
        f"Total: {len(sem_inputs)}, Already done: {len(completed)}, "
        f"Remaining: {len(remaining)}"
    )

    if not remaining:
        print("Nothing to do.")
        return

    print(f"Warming up {len(base_urls)} backends...")
    await pool.warmup()

    num_workers = max(1, concurrency // NUM_GENERATIONS)
    print(
        f"Starting {num_workers} workers "
        f"({num_workers * NUM_GENERATIONS} max concurrent API calls)"
    )

    done = 0
    errors = 0
    start_time = time.monotonic()

    write_lock = asyncio.Lock()

    queue: asyncio.Queue[SemanticLayerInput | None] = asyncio.Queue()
    for sem_input in remaining:
        queue.put_nowait(sem_input)
    for _ in range(num_workers):
        queue.put_nowait(None)

    async def worker(fh: Any) -> None:
        nonlocal done, errors
        while True:
            item = await queue.get()
            if item is None:
                return

            try:
                result = await classify_sample(pool, model, item, json_mode)
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
                print(
                    f"Progress: {total_processed}/{len(remaining)} "
                    f"({done} ok, {errors} errors) "
                    f"[{rate:.1f} samples/s, ETA {eta / 60:.0f}m]"
                )

    output_fh = open(output_path, "a")
    try:
        await asyncio.gather(*[worker(output_fh) for _ in range(num_workers)])
    finally:
        output_fh.close()
        await pool.aclose()

    elapsed = time.monotonic() - start_time
    print(
        f"\nDone. {done} classified, {errors} errors "
        f"in {elapsed / 60:.1f}m. Results in {output_path}"
    )


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

            # v2 has no dataset-specific drop/transform flags; splits
            # default to ("interactive_agent", "search") which is the
            # training config. Always pass a config for split selection.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Semantic layer: classify no-FC turns with a vLLM-served LLM. "
            "By default, runs the selected loader with every dataset-specific "
            "flag and every universal filter turned on. Use --filter-mode to "
            "control filtering."
        ),
    )
    parser.add_argument(
        "--loader",
        required=True,
        choices=LOADER_NAMES,
        help="Dataset loader to run.",
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
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Name of the vLLM-served model.",
    )
    parser.add_argument(
        "--base-urls",
        nargs="+",
        default=[f"http://localhost:{8000 + i}/v1" for i in range(8)],
        help="Base URLs of vLLM instances (default: 8 instances on ports 8000-8007).",
    )
    parser.add_argument(
        "--output",
        default="semantic_results.jsonl",
        help="JSONL file to append classifications to (resume-friendly).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=768,
        help="Max in-flight API calls (workers = concurrency // NUM_GENERATIONS).",
    )
    parser.add_argument(
        "--txt360-seed-group",
        default=None,
        choices=TXT360_SEED_GROUPS,
        help=(
            "Override TxT360's seed-group selection mode regardless of "
            "--filter-mode. When set, builds a TxT360Config with the "
            "given seed_group_filter; the other two TxT360 dataset flags "
            "(require_non_empty_user, drop_user_tool_call_samples) follow "
            "--filter-mode. Ignored for non-TxT360 loaders. Default: not "
            "set (TxT360 dataset config follows --filter-mode entirely)."
        ),
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help=("Force JSON output via response_format (slower but fewer parse errors)."),
    )
    args = parser.parse_args()

    do_strip = not args.no_strip_thinking
    seed_group_note = (
        f", txt360_seed_group={args.txt360_seed_group!r}"
        if args.txt360_seed_group is not None
        else ""
    )
    print(
        f"Loading dataset via '{args.loader}' loader "
        f"(filter_mode={args.filter_mode!r}, strip_thinking={do_strip}"
        f"{seed_group_note})..."
    )
    samples, report = _load_samples(
        args.loader,
        args.split,
        args.filter_mode,
        do_strip,
        args.txt360_seed_group,
    )
    print(report.summary())
    print(f"Kept {len(samples)} samples after filtering.")

    print("Running programmatic (behavioral) layer...")
    analyses = analyze_dataset_behavior(samples)

    print("Preparing semantic inputs...")
    sem_inputs = prepare_semantic_layer_inputs(samples, analyses)
    print(f"Samples with no-FC turns: {len(sem_inputs)}")
    print(f"Using {len(args.base_urls)} vLLM instances: {args.base_urls}")

    asyncio.run(
        run_semantic_layer(
            sem_inputs=sem_inputs,
            model=args.model,
            base_urls=args.base_urls,
            output_path=args.output,
            concurrency=args.concurrency,
            json_mode=args.json_mode,
        )
    )


if __name__ == "__main__":
    main()
