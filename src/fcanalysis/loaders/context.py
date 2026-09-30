"""Literal evidence primitives for source-specific visible-context policies.

These helpers establish only exact occurrence. Adapters decide which prior
roles/results are evidence, which argument fields require it, how the runtime
interprets evidence, and whether a visible action was authorized. Definition
examples, arbitrary prose mentions, future events and assistant inventions do
not become valid state merely by calling a string-matching helper.
"""

from collections.abc import Callable, Collection, Iterable, Iterator
from datetime import date
import re
from typing import TYPE_CHECKING, Any

from .normalization import Reject, json_bytes, parse_json

if TYPE_CHECKING:
    from .pipeline import RowState

# English spellings are explicit: process locale must not change evidence.
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_PATTERN = "(?:" + "|".join(_MONTHS) + ")"
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_MONTH_FIRST = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)
_DAY_FIRST = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\.?\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)


def calendar_dates(text: str) -> set[str]:
    """Extract valid full ISO or English written dates as ISO comparison values.

    Requires an explicit four-digit year. Invalid calendar dates, relative
    phrases and locale-dependent numeric slash/dot dates supply no evidence.
    The adapter decides whether a found date is a requested date, a current
    clock, or irrelevant historical context. This helper never consults the
    machine clock, infers a year, computes a relative date or changes source text.
    """
    found: set[str] = set()

    def add(year: str, month: str | int, day: str) -> None:
        try:
            found.add(date(int(year), int(month), int(day)).isoformat())
        except ValueError:
            pass

    for year, month, day in _ISO_DATE.findall(text):
        add(year, month, day)
    for month, day, year in _MONTH_FIRST.findall(text):
        add(year, _MONTHS[month.lower()], day)
    for day, month, year in _DAY_FIRST.findall(text):
        add(year, _MONTHS[month.lower()], day)
    return found


def contains_literal_token(
    value: str,
    texts: Iterable[str],
    *,
    case_insensitive: bool = False,
    strict_numeric_tokens: bool = False,
) -> bool:
    """Match an exact nonempty literal with word-character boundaries.

    Punctuation inside the value is literal, not regular-expression syntax.
    Source-specific equivalence projections belong in the adapter. A caller
    may enable case-insensitive matching only for an audited runtime contract.
    Numeric-token checking is also explicit: numeric-looking values cannot
    match part of a signed, decimal or grouped number, but sentence-ending
    punctuation remains valid. No numerical equivalence is inferred.
    """
    if not value:
        return False
    if strict_numeric_tokens and re.fullmatch(
        r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value
    ):
        pattern = r"(?<![\w.,+-])" + re.escape(value) + r"(?!\w|[.,]\d)"
    else:
        pattern = r"(?<![\w])" + re.escape(value) + r"(?![\w])"
    flags = re.IGNORECASE if case_insensitive else 0
    return any(re.search(pattern, content, flags=flags) for content in texts)


def collect_string_values(
    value: Any,
    evidence: set[str],
    *,
    key_filter: Callable[[str], bool] | None = None,
) -> None:
    """Accumulate exact JSON string leaves, plus explicitly approved map keys.

    Keys are often schema labels, so none are evidence by default. A source
    policy can admit known maps keyed by item/payment identifiers. String
    contents are never scanned for embedded tokens or recursively parsed.
    Numeric/string coercion is not an automatic grounding equivalence.
    """
    if isinstance(value, dict):
        if key_filter is not None:
            evidence.update(key for key in value if key_filter(key))
        for item in value.values():
            collect_string_values(item, evidence, key_filter=key_filter)
    elif isinstance(value, list):
        for item in value:
            collect_string_values(item, evidence, key_filter=key_filter)
    elif isinstance(value, str):
        evidence.add(value)


def json_scalar_values(value: Any, *, skip_keys: Collection[str] = ()) -> Iterator[str]:
    """Yield nonempty string leaves and exact JSON spellings of numeric leaves.

    A source policy must explicitly opt into numeric evidence, for example an
    integer OTP returned by a tool. Booleans, nulls and map keys never become
    credentials. ``skip_keys`` excludes source-defined method selectors or
    other ineligible fields and their descendants. Values are not parsed again,
    rounded, case-folded, or modified; malformed numeric JSON still rejects.
    """
    if isinstance(value, str):
        if value:
            yield value
    elif type(value) in (int, float):
        yield json_bytes(value).decode()
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in skip_keys:
                yield from json_scalar_values(child, skip_keys=skip_keys)
    elif isinstance(value, list):
        for child in value:
            yield from json_scalar_values(child, skip_keys=skip_keys)


def iter_call_evidence(
    state: RowState,
    *,
    result_values: Callable[[Any, dict[str, Any]], Iterable[str]],
) -> Iterator[tuple[str, dict[str, Any], dict[str, Any], list[str], set[str]]]:
    """Yield each call with its static definition and strictly earlier evidence.

    Run after schema and linkage validation. Each item contains the function
    name, definition, parsed arguments, prior user/system/non-JSON result texts,
    and projected JSON result values. Assistant content, definitions, raw and
    future results never enter the evidence channels. Parallel sibling calls
    therefore see the same preceding evidence. Adapters own result projection
    and the interpretation of this evidence; occurrence is not authorization.

    The yielded containers are borrowed read-only views valid until the next
    advance; consume them immediately rather than retaining or mutating them.
    This keeps traversal bounded to one conversation without copying its growing
    history for each call. Visible dynamic capability protocols need their own
    definition resolution; this helper only looks up the initial tool set.
    """
    definitions = {
        tool["function"]["name"]: tool["function"] for tool in state.sample.tools
    }
    texts: list[str] = []
    results: set[str] = set()
    for index, message in enumerate(state.sample.messages):
        content = message.get("content")
        if message["role"] in {"system", "user"} and isinstance(content, str):
            texts.append(content)
        elif message["role"] == "tool" and isinstance(content, str):
            try:
                parsed = parse_json(content)
            except Reject:
                texts.append(content)
            else:
                results.update(result_values(parsed, message))
        elif message["role"] == "assistant":
            for call_index, call in enumerate(message.get("tool_calls", [])):
                name = call["function"]["name"]
                yield (
                    name,
                    definitions.get(name, {}),
                    state.parsed_arguments[index, call_index],
                    texts,
                    results,
                )


def validate_argument_evidence(
    state: RowState,
    requires_evidence: Callable[[str, str, dict[str, Any], Any], bool],
    *,
    values: Callable[[Any], Iterable[str]],
    result_values: Callable[[Any, dict[str, Any]], Iterable[str]] | None = None,
    reason: str = "ungrounded_credential_argument",
    strict_numeric_tokens: bool = False,
) -> None:
    """Require prior literal evidence for adapter-selected argument values.

    Run after schema and call/result linkage validation. The adapter supplies
    the exact consuming-slot policy and explicitly chooses its scalar-value
    projection. Prior user/system text and linked tool results are eligible;
    assistant text/reasoning, future results, raw metadata and definitions are
    never evidence. JSON results supply exact projected values; non-JSON result
    text uses token boundaries. This establishes literal occurrence only, not
    arbitrary natural-language authorization or the truth of simulated results.
    ``result_values`` may use a different source projection, for example to
    exclude echoed input arguments from an otherwise preserved result envelope.
    It receives the parsed payload and original message, allowing an adapter
    to recognize only independently proved result envelopes. It must not
    mutate either argument or use future values as evidence.
    """
    projection = result_values or (lambda parsed, message: values(parsed))
    for name, definition, arguments, texts, results in iter_call_evidence(
        state, result_values=projection
    ):
        for field, value in arguments.items():
            if requires_evidence(name, field, definition, value):
                for leaf in values(value):
                    if leaf not in results and not contains_literal_token(
                        leaf, texts, strict_numeric_tokens=strict_numeric_tokens
                    ):
                        raise Reject(reason)


_DEICTIC = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|next\s+(?:week|month|year|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b",
    re.IGNORECASE,
)
_CLOCK_ASSERTION = re.compile(
    r"\b(?:today(?:['’]s\s+date)?|current\s+(?:date(?:\s+and\s+time)?|time))"
    r"\s*(?:is\b|[:=(])\s*([^\n]+)",
    re.IGNORECASE,
)
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_WRITTEN_DATE = re.compile(
    rf"\b(?:{_MONTH}\s+{_DAY}|{_DAY}\s+{_MONTH})"
    rf"(?:\s*[–—-]\s*(?:{_MONTH}\s+)?{_DAY})?(?:,?\s+\d{{4}})?\b",
    re.IGNORECASE,
)
_NUMERIC_DATE = re.compile(
    r"\b(?:\d{1,4}/\d{1,2}(?:/\d{1,4})?|\d{1,4}\.\d{1,2}\.\d{1,4})\b"
)
_MONTH_YEAR = re.compile(rf"\b{_MONTH}\s+\d{{4}}\b", re.IGNORECASE)


def _has_clock(text: str) -> bool:
    for match in _CLOCK_ASSERTION.finditer(text):
        # Keep abbreviated month periods while stopping unrelated sentences.
        clause = re.sub(
            r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.(?=\s)",
            r"\1",
            match.group(1),
            flags=re.IGNORECASE,
        )
        if calendar_dates(re.split(r"[.;!?]", clause, maxsplit=1)[0]):
            return True
    return False


def _unresolved_explicit_date(text: str) -> bool:
    """Abstain from unsupported explicit dates instead of assuming omission."""
    if _NUMERIC_DATE.search(text) or _MONTH_YEAR.search(text):
        return True
    for match in _WRITTEN_DATE.finditer(text):
        value = match.group()
        if re.search(r"[–—-]", value) or not calendar_dates(value):
            return True
    return False


def _calendar_values(
    value: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield ".".join(path), value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _calendar_values(child, (*path, key))
    elif isinstance(value, list):
        for child in value:
            yield from _calendar_values(child, (*path, "*"))


def validate_calendar_evidence(
    state: RowState,
    *,
    slots: Callable[[str, dict[str, Any]], Collection[str]],
    reason: str = "ungrounded_temporal_reference",
) -> None:
    """Check a bounded missing-clock condition for adapter-selected date slots.

    ``slots`` selects exact consuming paths from each source tool definition;
    dotted paths use ``*`` for array elements. Run after schema and linkage
    checks. A relative user request plus an emitted full calendar date needs
    either that explicit date in prior user/system text or a labeled current
    clock. This does not verify date arithmetic. Assistant reasoning, definition
    examples, raw metadata and future messages supply no evidence.

    Stop at the first tool result: unclassified backend content may establish
    temporal context. Also abstain for user dates needing unsupported calendar
    interpretation, including partial month/day or month/year forms and ranges.
    No machine clock or missing year is ever substituted. Adapters choose
    whether this deliberately bounded policy fits
    their audited source contracts and rerun it after context transformations.
    """
    definitions = {
        tool["function"]["name"]: tool["function"] for tool in state.sample.tools
    }
    selected_slots = {
        name: slots(name, definition) for name, definition in definitions.items()
    }
    if not any(selected_slots.values()):
        return
    dates: set[str] = set()
    relative = clock = unresolved_date = False
    for mi, message in enumerate(state.sample.messages):
        role, content = message["role"], message.get("content")
        if role in ("user", "system") and isinstance(content, str):
            dates.update(calendar_dates(content))
            clock |= _has_clock(content)
            if role == "user":
                relative |= bool(_DEICTIC.search(content))
                unresolved_date |= _unresolved_explicit_date(content)
        elif role == "tool":
            # An unclassified visible backend result may establish event dates
            # or clock state. Do not turn its absence from a narrow parser into
            # proof that later calendar arithmetic is ungrounded.
            return
        elif role == "assistant" and relative and not clock and not unresolved_date:
            for ci, call in enumerate(message.get("tool_calls", [])):
                selected = selected_slots.get(call["function"]["name"], frozenset())
                for path, value in _calendar_values(state.parsed_arguments[mi, ci]):
                    if path in selected and calendar_dates(value) - dates:
                        raise Reject(reason)
