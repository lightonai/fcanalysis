"""Pinned Nemotron-Terminal textual protocol; never executes source commands.

The fixed initial instructions are from nvidia/Nemotron-Terminal-Corpus,
revision a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9 (CC BY 4.0). Parser behavior
is interpreted using the attributed Harbor parser in _terminus_json.py.
All parsed/defaulted/repaired values are ephemeral validation/comparison data.
Canonical assistant text, commands and feedback are never rewritten here.
"""

from dataclasses import dataclass
import json
import math
import re
from typing import Any

from ._terminus_json import ParseResult, TerminusJSONPlainParser
from .normalization import Reject, parse_json

PROTOCOL_PREFIX = r"""You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {
      "keystrokes": "ls -la\n",
      "duration": 0.1
    },
    {
      "keystrokes": "cd project\n",
      "duration": 0.1
    }
  ],
  "task_complete": true
}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- Most bash commands should end with a newline (\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {"keystrokes": "", "duration": 10.0} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action

Task Description:
"""
SCREEN_PREFIXES = ("New Terminal Output:\n", "Current Terminal Screen:\n")
ERROR_PREFIX = "Previous response had parsing errors:\n"
WARNING_PREFIX = "Previous response had warnings:\n"
CONFIRMATION = (
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)
CONFIRM_PREFIX = "Current terminal state:\n"
ERROR_SUFFIX = "\n\nPlease fix these issues and provide a proper JSON response."
RESET_NOTICE = "Performed context summarization and handoff to continue task."
HANDOFF_PREFIX = "Here are the answers the other agent provided."


def split_native(content: str) -> tuple[str | None, str]:
    """Decode the exporter's leading reasoning field and one join newline.

    Reasoning is opaque: an unfinished fence cannot hide its export delimiter.
    More than one possible closing delimiter is ambiguous, including when a
    literal delimiter inside reasoning could also satisfy the export grammar.
    Tags inside the remaining JSON payload are ordinary literal text.
    """
    if not content.startswith("<think>"):
        return None, content
    ends = [m.start() for m in re.finditer(r"</think>(?=\n|$)", content)]
    if len(ends) != 1:
        raise Reject("ambiguous_source_reasoning_boundary")
    end = ends[0]
    reasoning = content[len("<think>") : end]
    if not reasoning:
        raise Reject("unsupported_empty_exported_reasoning")
    remainder = content[end + len("</think>") :]
    return reasoning, remainder[1:] if remainder else ""


class _FastParser(TerminusJSONPlainParser):
    """Use C JSON decoding for the common complete-object case.

    A valid complete object has exactly the boundaries the upstream scanner
    finds. Nonstandard constants and ambiguous duplicate keys are guarded
    separately, with the adapter's documented opaque-prose exception.
    Everything else uses the original scanner and ordered repair attempts.
    """

    def _extract_json_content(self, response: str) -> tuple[str, list[str]]:
        text = response.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                value = json.loads(text)
            except ValueError, RecursionError:
                pass
            else:
                if isinstance(value, dict):
                    return text, []
        return super()._extract_json_content(response)


_PARSER = _FastParser()


@dataclass(slots=True)
class Response:
    result: ParseResult
    # Strict complete visible object, when available. This is not repaired text.
    original_object: dict[str, Any] | None
    consumed_text: str
    consumed_object: dict[str, Any] | None
    duplicate_prose_fields: int = 0


def _finite(value: Any) -> bool:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise Reject("unsupported_protocol_unicode") from exc
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(k) and _finite(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def interpret(content: str) -> Response:
    """Interpret the visible response with conservative ambiguity guards."""
    # Check the exact object consumed by the parser, including any repair.
    # Never silently accept duplicate keys or nonfinite execution parameters.
    original = None
    decoded = None
    duplicate_prose_fields = 0
    try:
        parsed = parse_json(content)
    except Reject:
        pass
    else:
        if isinstance(parsed, dict):
            original = parsed
    try:
        result = _PARSER.parse_response(content)
        consumed, _ = _PARSER._extract_json_content(content)
        if result.warning.startswith("- AUTO-CORRECTED:"):
            initial = _PARSER._try_parse_response(content)
            for _, fix in _PARSER._get_auto_fixes():
                candidate, changed = fix(content, initial.error)
                if changed and not _PARSER._try_parse_response(candidate).error:
                    consumed, _ = _PARSER._extract_json_content(candidate)
                    break
        if consumed:
            try:
                decoded = parse_json(consumed)
            except Reject as exc:
                # A genuine JSON syntax error may be an observed recovery step.
                # Duplicate-key/nonfinite ambiguity is not an accepted mapping.
                duplicates = []

                def source_object(pairs):
                    obj = {}
                    for key, value in pairs:
                        if not _finite(value):
                            raise Reject("nonfinite_protocol_value")
                        if key in obj:
                            duplicates.append(key)
                        obj[key] = value
                    return obj

                try:
                    decoded = json.loads(consumed, object_pairs_hook=source_object)
                except Reject:
                    raise
                except ValueError, RecursionError:
                    pass
                else:
                    if (
                        exc.reason == "duplicate_json_key"
                        and duplicates
                        and set(duplicates) <= {"analysis", "plan"}
                        and _finite(decoded)
                    ):
                        # These fields never determine executed commands or
                        # completion. Upstream json.loads keeps the last value;
                        # preserve every occurrence in source text, and abstain
                        # from Level 2 prose removal for this non-strict object.
                        duplicate_prose_fields = len(duplicates)
                    else:
                        raise Reject("ambiguous_protocol_json") from exc
            else:
                if not _finite(decoded):
                    raise Reject("nonfinite_protocol_value")
        if any(
            not math.isfinite(c.duration) or c.duration < 0 for c in result.commands
        ):
            raise Reject("unsupported_command_duration")
    except Reject:
        raise
    except (
        ValueError,
        RecursionError,
        OverflowError,
        TypeError,
        AttributeError,
    ) as exc:
        raise Reject("unsupported_parser_shape") from exc
    return Response(result, original, consumed, decoded, duplicate_prose_fields)


def parser_feedback(result: ParseResult) -> str:
    if result.error:
        value = "ERROR: " + result.error
        if result.warning:
            value += "\nWARNINGS: " + result.warning
        return value
    return "WARNINGS: " + result.warning if result.warning else ""


def screen(text: str) -> bool:
    return text.startswith(SCREEN_PREFIXES)


def confirmation(text: str) -> bool:
    return (
        text.startswith(CONFIRM_PREFIX)
        and text.endswith("\n\n" + CONFIRMATION)
        and screen(text[len(CONFIRM_PREFIX) : -len("\n\n" + CONFIRMATION)])
    )


def expected_error(result: ParseResult) -> str:
    return ERROR_PREFIX + parser_feedback(result) + ERROR_SUFFIX


def expected_warning_prefix(result: ParseResult) -> str:
    return WARNING_PREFIX + parser_feedback(result) + "\n\n"


def _warning_order(text: str) -> str:
    """Ignore only the order of names emitted from an upstream Python set."""
    return re.sub(
        r"(?m)^((?:WARNINGS: )?- Command [0-9]+: Unknown fields: )([^\n]+)$",
        lambda match: match[1] + ", ".join(sorted(match[2].split(", "))),
        text,
    )


def error_feedback_matches(response: Response, observation: str) -> bool:
    expected = expected_error(response.result)
    if _warning_order(expected) == _warning_order(observation):
        return True
    # CPython 3.14 points at a trailing comma; earlier decoders point at the
    # following closing bracket/brace. Check those exact characters/locations
    # in the consumed JSON rather than accepting an arbitrary error string.
    error = response.result.error
    match = re.match(
        r"Invalid JSON: Illegal trailing comma before end of (object|array): "
        r"line [0-9]+ column [0-9]+ \(char ([0-9]+)\)",
        error,
    )
    if not match:
        return False
    consumed = response.consumed_text
    comma = int(match[2])
    closing = comma + 1
    while closing < len(consumed) and consumed[closing] in " \r\n\t":
        closing += 1
    bracket = "}" if match[1] == "object" else "]"
    if consumed[comma : comma + 1] != "," or consumed[closing : closing + 1] != bracket:
        return False
    message = (
        "Expecting property name enclosed in double quotes"
        if bracket == "}"
        else "Expecting value"
    )
    legacy = "Invalid JSON: " + str(json.JSONDecodeError(message, consumed, closing))
    alternative = expected.replace(error, legacy + error[match.end() :], 1)
    return _warning_order(alternative) == _warning_order(observation)


def warning_feedback_matches(result: ParseResult, observation: str) -> bool:
    if not observation.startswith(WARNING_PREFIX):
        return False
    header, separator, terminal = observation[len(WARNING_PREFIX) :].partition("\n\n")
    return (
        bool(separator)
        and _warning_order(header) == _warning_order(parser_feedback(result))
        and screen(terminal)
    )
