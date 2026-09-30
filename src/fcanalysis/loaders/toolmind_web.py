"""ToolMind-Web-QA: strict conversion of released long-search trajectories.

Origin and license
------------------
The pinned release is https://huggingface.co/datasets/Nanbeige/ToolMind-Web-QA
at revision ``2690dcdfdd82ab147aad4a65ab231d4e344f5cd0``; its pinned card is
https://huggingface.co/datasets/Nanbeige/ToolMind-Web-QA/blob/2690dcdfdd82ab147aad4a65ab231d4e344f5cd0/README.md
and declares Apache-2.0.  The Nanbeige4.1 paper
(https://arxiv.org/abs/2602.13367v1, section 2.2) describes temporally
selected Wikipedia entities, graph random walks, multi-framework trajectory
synthesis, and turn-level critics for logical soundness, call accuracy, and
information gain.  It names Serper, Jina, and E2B/MiroThinker as services or
frameworks; the release does not provide their generation revisions or terms.
The upstream data repository is the pinned Hugging Face repository above;
its card does not supply an executable, revision-pinned generation repository.

The repository has 6,801 QA-only rows in ``syn_wikiqa.jsonl`` and 5,624
complete trajectory rows in ``open-wiki-traj.jsonl``.  Only the latter are
loaded.  All trajectory rows have ``key``, ``id``, and ``conversations``;
messages have only ``role``, ``content``, and the source ``loss`` flag.  The
source uses system/user/assistant roles, serial one-call turns, and places the
tool result in the user message immediately after a call.  The loss flag is a
turn-level upstream training mask retained unchanged in ``raw``; it is not
promoted into canonical messages or loader-added annotations.

Protocol and reasoning
----------------------
The model calls one visible ``use_mcp_tool`` wrapper with ``server_name``,
``tool_name``, and nested ``arguments``.  Server-qualified definitions are
embedded in one exact system template.  Conversion moves the callable
description and complete catalog into the wrapper definition, maps the three
explicit parameter declarations into its schema, and removes the XML transport
grammar that canonical ``tool_calls`` replaces.  Unrelated system preamble,
date, and objectives remain unchanged.  It does not flatten MCP operations into
renamed functions.  A source-specific validator parses the eight embedded
server/tool schemas and validates the nested arguments.

Only a complete top-level ``<use_mcp_tool>`` block at the end of an assistant
message is a call.  Its next source-user message becomes the single linked tool
result; ordinary users remain users.  Malformed XML, JSON, undeclared server
or tool names, invalid nested arguments, and calls without an immediate result
are quarantined.  Arbitrary JSON arrays in a result remain one result.

Native ``<think>``/``<reasoning>`` spans remain in assistant content unless
``FilterConfig.strip_thinking`` is true.  Shared removal is surgical and never
removes the visible MCP call. Omitting filter_config enables stripping; an
explicit FilterConfig uses its own value, whose field default is False.
The final-view pipeline rechecks structure,
serial singleton linkage, wrapper capability, nested contracts, system context,
and a nonempty terminal assistant.  Upstream loss flags do not change loader
acceptance or impose a trainer loss policy.  The catalog declares no dedicated
think tool; Python and search reasoning mechanisms remain visible.  Full
histories are already released, so the loader performs no prefix reconstruction.

Ordered pipeline
----------------
1. Read the pinned trajectory JSONL in physical order and preserve each row in
   ``raw`` without mutable aliases.
2. Match the exact embedded wrapper/catalog template and parse all definitions.
3. Convert strict terminal XML calls and their immediately following source-user
   results; preserve all other chronology and source text.
4. Validate shared structure/linkage/capability, the source MCP catalog, and the
   source clock and sandbox context before transforms.
5. Optionally strip native reasoning and apply the system override, then repeat
   all affected final-view checks and require a terminal assistant answer.
6. Apply shared deterministic Levels 1/1.5/2 curation within the one released
   trajectory file; report Levels 3–5 as audit-only counts plus source, drops,
   transforms, curation, and final count.

The context gate retains the release's single visible ``Today is`` value and
accepts a Python sandbox identifier only after the exact preceding successful
``create_sandbox`` result makes that identifier visible.  Error results do not
activate state.  It does not treat assistant prose, future results, placeholder
IDs, or unrelated JSON values as runtime state, and does not attempt generic
truth judgments over web evidence.

Release interpretation and use
------------------------------
The released cohorts have English and Chinese labels despite the card's English
language declaration. These labels do not independently classify language or
define benchmark splits. The card's ``test`` configuration name is not evidence
of a held-out evaluation set; curation uses the neutral scope ``released``.
The source clock must stay in the initial system after an override: a later or
different date cannot replace it. Template recognition requires the complete
fixed catalog and transport framing; unfamiliar inserted instructions reject.

Retaining source loss flags in raw does not endorse positive supervision of every
assistant. The trainer owns masks, history policy and model-specific rendering.
Answer truth, search effectiveness, critic quality, inherited-source terms and
downstream benefit are not certified. Name-based audits see one shared wrapper;
coarse groups do not establish that its underlying searches are duplicates.

iter_load() reads the trajectory file in source order and emits after scoped
disk-backed curation; close it on early termination. load() materializes results.
The two reasoning-mode configurations and complete counters are in
``tests/fixtures/loaders/toolmind_web``. Conversion and final-view tests are in
``tests/unit/test_loader_toolmind_web.py``; the separate full-source projection
in ``tests/e2e/test_toolmind_web_projection.py`` reuses strict JSON/native-boundary
primitives and does not independently reimplement every rejection or curation
decision. Transformation counters can include candidates later excluded.
"""

import ast
from collections import Counter
from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache, partial
import re
from typing import Any

from ..format import ConversationSample
from .base import FilterConfig, LoadReport
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .normalization import Reject, json_bytes, parse_json
from .pipeline import (
    Pipeline,
    RowState,
    Stage,
    _code_spans,
    link_calls,
    native_reasoning_spans,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_structure,
    validate_termination,
)
from .schema import check_arguments, compile_schema
from .source import jsonl_lines

DATASET_ID = "Nanbeige/ToolMind-Web-QA"
DATASET_REVISION = "2690dcdfdd82ab147aad4a65ab231d4e344f5cd0"
SOURCE = "open-wiki-traj.jsonl"


@dataclass(slots=True)
class ToolMindWebConfig:
    """The pinned release exposes one loadable trajectory file."""


_TOOL_SECTION = "# Tool-Use Formatting Instructions \n"
_OBJECTIVES = "# General Objective\n"
_CATALOG = "Here are the functions available in JSONSchema format:\n\n\n"
_SERVER = re.compile(r"^## Server name: ([^\n]+)$", re.MULTILINE)
_TOOL = re.compile(r"^### Tool name: ([^\n]+)$", re.MULTILINE)
_CALL = re.compile(
    r"(?s)<use_mcp_tool>\s*"
    r"<server_name>([^<>]+)</server_name>\s*"
    r"<tool_name>([^<>]+)</tool_name>\s*"
    r"<arguments>\s*(.*?)\s*</arguments>\s*"
    r"</use_mcp_tool>\s*"
)
_CONTROL = re.compile(r"</?(?:use_mcp_tool|server_name|tool_name|arguments)>")
_XML_LIKE_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9_.:-]*)(?:\s[^>]*?)?\s*(/?)>")
_DATE = re.compile(r"Today is: ([^\n]+)")
_SANDBOX_CREATED = re.compile(r"Sandbox created with sandbox_id: ([A-Za-z0-9-]+)")
_TRANSPORT_AND_CATALOG_HEADER = (
    "# Tool-Use Formatting Instructions \n\n"
    "Tool-use is formatted using XML-style tags. The tool-use is enclosed in "
    "<use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within "
    "its own set of tags.\n\n"
    "The Model Context Protocol (MCP) connects to servers that provide additional "
    "tools and resources to extend your capabilities. You can use the server's tools "
    "via the `use_mcp_tool`.\n\n"
    "Description: \nRequest to use a tool provided by a MCP server. Each MCP server "
    "can provide multiple tools with different capabilities. Tools have defined input "
    "schemas that specify required and optional parameters.\n\n"
    "Parameters:\n"
    "- server_name: (required) The name of the MCP server providing the tool\n"
    "- tool_name: (required) The name of the tool to execute\n"
    "- arguments: (required) A JSON object containing the tool's input parameters, "
    "following the tool's input schema, quotes within string must be properly escaped, "
    "ensure it's valid JSON\n\n"
    "Usage:\n<use_mcp_tool>\n<server_name>server name here</server_name>\n"
    "<tool_name>tool name here</tool_name>\n<arguments>\n{\n"
    '"param1": "value1",\n"param2": "value2 \\"escaped string\\""\n'
    "}\n</arguments>\n</use_mcp_tool>\n\n"
    "Important Notes:\n"
    "- Tool-use must be placed **at the end** of your response, **top-level**, and not "
    "nested within other tags.\n"
    "- Always adhere to this format for the tool use to ensure proper parsing and "
    "execution.\n\n"
    "String and scalar parameters should be specified as is, while lists and objects "
    "should use JSON format. Note that spaces for string values are not stripped. The "
    "output is not expected to be valid XML and is parsed with regular expressions.\n"
    + _CATALOG
)
_PARAMETER_LINES = {
    "server_name": "The name of the MCP server providing the tool",
    "tool_name": "The name of the tool to execute",
    "arguments": (
        "A JSON object containing the tool's input parameters, following the "
        "tool's input schema, quotes within string must be properly escaped, "
        "ensure it's valid JSON"
    ),
}
_EXPECTED_CATALOG_KEYS = (
    ("tool-python", "create_sandbox"),
    ("tool-python", "run_command"),
    ("tool-python", "run_python_code"),
    ("tool-python", "upload_file_from_local_to_sandbox"),
    ("tool-python", "download_file_from_internet_to_sandbox"),
    ("tool-python", "download_file_from_sandbox_to_local"),
    ("search_and_scrape_webpage", "google_search"),
    ("jina_scrape_llm_summary", "scrape_and_extract_info"),
)
_EXPECTED_CATALOG = """## Server name: tool-python
### Tool name: create_sandbox
Description: Create a linux sandbox.

Args:
    timeout: Time in seconds before the sandbox is automatically shutdown. The default is 600 seconds.

Returns:
    The sandbox_id of the newly created sandbox. You should use this sandbox_id to run other tools in the sandbox.
Input JSON schema: {'properties': {'timeout': {'default': 600, 'title': 'Timeout'}}, 'type': 'object'}
### Tool name: run_command
Description: Execute a lightweight shell command in the linux sandbox (no long-running, blocking, or resource-heavy processes).

Args:
    command: The command to execute.
    sandbox_id: The id of the sandbox to execute the command in. To create a new sandbox, use tool `create_sandbox`.

Returns:
    A CommandResult object containing the result of the command execution, format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)
Input JSON schema: {'properties': {'command': {'title': 'Command', 'type': 'string'}, 'sandbox_id': {'title': 'Sandbox Id', 'type': 'string'}}, 'required': ['command', 'sandbox_id'], 'type': 'object'}
### Tool name: run_python_code
Description: Run short, safe python code in a sandbox and return the execution result (avoid long loops or heavy tasks; must finish quickly).

Args:
    code_block: The python code to run.
    sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.

Returns:
    A CommandResult object containing the result of the command execution, format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)
Input JSON schema: {'properties': {'code_block': {'title': 'Code Block', 'type': 'string'}, 'sandbox_id': {'title': 'Sandbox Id', 'type': 'string'}}, 'required': ['code_block', 'sandbox_id'], 'type': 'object'}
### Tool name: upload_file_from_local_to_sandbox
Description: Upload a local file to the `/home/user` dir of the remote python interpreter.

Args:
    sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
    local_file_path: The path of the file on local machine to upload.
    sandbox_file_path: The path of directory to upload the file to in the sandbox. Default is `/home/user/`.

Returns:
    The path of the uploaded file in the remote python interpreter if the upload is successful.
Input JSON schema: {'properties': {'sandbox_id': {'title': 'Sandbox Id', 'type': 'string'}, 'local_file_path': {'title': 'Local File Path', 'type': 'string'}, 'sandbox_file_path': {'default': '/home/user', 'title': 'Sandbox File Path', 'type': 'string'}}, 'required': ['sandbox_id', 'local_file_path'], 'type': 'object'}
### Tool name: download_file_from_internet_to_sandbox
Description: Download a file from the internet to the `/home/user` dir of the sandbox (avoid large or slow URLs).

Args:
    sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
    url: The URL of the file to download.
    sandbox_file_path: The path of directory to download the file to in the sandbox. Default is `/home/user/`.

Returns:
    The path of the downloaded file in the sandbox if the download is successful.
Input JSON schema: {'properties': {'sandbox_id': {'title': 'Sandbox Id', 'type': 'string'}, 'url': {'title': 'Url', 'type': 'string'}, 'sandbox_file_path': {'default': '/home/user', 'title': 'Sandbox File Path', 'type': 'string'}}, 'required': ['sandbox_id', 'url'], 'type': 'object'}
### Tool name: download_file_from_sandbox_to_local
Description: Download a file from the sandbox to local system. Files in sandbox cannot be processed by tools from other servers - only local files and internet URLs can be processed by them.

Args:
    sandbox_id: The id of the sandbox to download the file from. To have a sandbox, use tool `create_sandbox`.
    sandbox_file_path: The path of the file to download on the sandbox.
    local_filename: Optional filename to save as. If not provided, uses the original filename from sandbox_file_path.

Returns:
    The local path of the downloaded file if successful, otherwise error message.
Input JSON schema: {'properties': {'sandbox_id': {'title': 'Sandbox Id', 'type': 'string'}, 'sandbox_file_path': {'title': 'Sandbox File Path', 'type': 'string'}, 'local_filename': {'default': None, 'title': 'Local Filename', 'type': 'string'}}, 'required': ['sandbox_id', 'sandbox_file_path'], 'type': 'object'}

## Server name: search_and_scrape_webpage
### Tool name: google_search
Description:\x20
Tool to perform web searches via Serper API and retrieve rich results.

It is able to retrieve organic search results, people also ask,
related searches, and knowledge graph.

Args:
    q: Search query string
    gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
    hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
    location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
    num: Number of results to return (default: 10)
    tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)
    page: Page number of results to return (default: 1)
    autocorrect: Whether to autocorrect spelling in query

Returns:
    Dictionary containing search results and metadata.

Input JSON schema: {'properties': {'q': {'title': 'Q', 'type': 'string'}, 'gl': {'default': 'us', 'title': 'Gl', 'type': 'string'}, 'hl': {'default': 'en', 'title': 'Hl', 'type': 'string'}, 'location': {'default': None, 'title': 'Location', 'type': 'string'}, 'num': {'default': None, 'title': 'Num', 'type': 'integer'}, 'tbs': {'default': None, 'title': 'Tbs', 'type': 'string'}, 'page': {'default': None, 'title': 'Page', 'type': 'integer'}, 'autocorrect': {'default': None, 'title': 'Autocorrect', 'type': 'boolean'}}, 'required': ['q'], 'title': 'google_searchArguments', 'type': 'object'}

## Server name: jina_scrape_llm_summary
### Tool name: scrape_and_extract_info
Description:\x20
Scrape content from a URL, including web pages, PDFs, code files, and other supported resources, and extract meaningful information using an LLM.
If you need to extract information from a PDF, please use this tool.

Args:
    url (str): The URL to scrape content from. Supports various types of URLs such as web pages, PDFs, raw text/code files (e.g., GitHub, Gist), and similar sources.
    info_to_extract (str): The specific types of information to extract (usually a question)
    custom_headers (Dict[str, str]): Additional headers to include in the scraping request

Returns:
    Dict[str, Any]: A dictionary containing:
        - success (bool): Whether the operation was successful
        - url (str): The original URL
        - extracted_info (str): The extracted information
        - error (str): Error message if the operation failed
        - scrape_stats (Dict): Statistics about the scraped content
        - model_used (str): The model used for summarization
        - tokens_used (int): Number of tokens used (if available)

Input JSON schema: {'properties': {'url': {'title': 'Url', 'type': 'string'}, 'info_to_extract': {'title': 'Info To Extract', 'type': 'string'}, 'custom_headers': {'additionalProperties': {'type': 'string'}, 'default': None, 'title': 'Custom Headers', 'type': 'object'}}, 'required': ['url', 'info_to_extract'], 'title': 'scrape_and_extract_infoArguments', 'type': 'object'}

"""


def _python_schema(text: str) -> dict[str, Any]:
    """Parse the released Python-literal schema without accepting duplicate keys."""
    try:
        expression = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise Reject("malformed_embedded_tool_schema") from exc
    for node in ast.walk(expression):
        if isinstance(node, ast.Dict):
            keys = []
            for key in node.keys:
                if key is None:
                    raise Reject("malformed_embedded_tool_schema")
                try:
                    value = ast.literal_eval(key)
                except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
                    raise Reject("malformed_embedded_tool_schema") from exc
                if not isinstance(value, str) or value in keys:
                    raise Reject("malformed_embedded_tool_schema")
                keys.append(value)
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise Reject("malformed_embedded_tool_schema") from exc
    if not isinstance(value, dict):
        raise Reject("malformed_embedded_tool_schema")
    json_bytes(value)
    return value


@lru_cache(maxsize=32)
def _catalog(section: str) -> dict[tuple[str, str], Any]:
    marker = section.find(_CATALOG)
    if marker < 0:
        raise Reject("unknown_embedded_tool_template")
    text = section[marker + len(_CATALOG) :]
    if text != _EXPECTED_CATALOG:
        raise Reject("unknown_embedded_tool_catalog")
    servers = list(_SERVER.finditer(text))
    if not servers or servers[0].start() != 0:
        raise Reject("malformed_embedded_tool_catalog")
    result: dict[tuple[str, str], Any] = {}
    ordered_keys: list[tuple[str, str]] = []
    for server_index, server_match in enumerate(servers):
        server_name = server_match.group(1)
        server_end = (
            servers[server_index + 1].start()
            if server_index + 1 < len(servers)
            else len(text)
        )
        server_body = text[server_match.end() : server_end]
        tools = list(_TOOL.finditer(server_body))
        if not tools or tools[0].start() != 1:
            raise Reject("malformed_embedded_tool_catalog")
        for tool_index, tool_match in enumerate(tools):
            tool_name = tool_match.group(1)
            tool_end = (
                tools[tool_index + 1].start()
                if tool_index + 1 < len(tools)
                else len(server_body)
            )
            body = server_body[tool_match.end() : tool_end]
            schema_marker = "Input JSON schema: "
            if (
                body.count(schema_marker) != 1
                or not body.startswith("\nDescription: ")
                or "\n\nArgs:\n" not in body
                or "\n\nReturns:\n" not in body
            ):
                raise Reject("malformed_embedded_tool_catalog")
            schema_text = body.split(schema_marker, 1)[1].strip()
            key = (server_name, tool_name)
            if key in result:
                raise Reject("conflicting_embedded_tool_definition")
            result[key] = compile_schema(_python_schema(schema_text))
            ordered_keys.append(key)
    if tuple(ordered_keys) != _EXPECTED_CATALOG_KEYS:
        raise Reject("unknown_embedded_tool_catalog")
    return result


def _extract_system(
    content: Any,
) -> tuple[str, dict[str, Any], dict[tuple[str, str], Any]]:
    if not isinstance(content, str) or content.count(_TOOL_SECTION) != 1:
        raise Reject("unknown_embedded_tool_template")
    start = content.index(_TOOL_SECTION)
    end = content.find(_OBJECTIVES, start)
    if end < 0 or len(_DATE.findall(content[:start])) != 1:
        raise Reject("unknown_embedded_tool_template")
    section = content[start:end]
    if not section.startswith(_TRANSPORT_AND_CATALOG_HEADER):
        raise Reject("unknown_embedded_tool_template")
    catalog = _catalog(section)
    semantic_start = _TRANSPORT_AND_CATALOG_HEADER.index(
        "The Model Context Protocol (MCP) connects"
    )
    parameters_start = _TRANSPORT_AND_CATALOG_HEADER.index(
        "\n\nParameters:", semantic_start
    )
    catalog_start = section.index(_CATALOG)
    description = (
        _TRANSPORT_AND_CATALOG_HEADER[semantic_start:parameters_start]
        + "\n\n"
        + section[catalog_start:]
    )
    normalized_system = content[:start] + content[end:]
    wrapper = {
        "type": "function",
        "function": {
            "name": "use_mcp_tool",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": _PARAMETER_LINES["server_name"],
                    },
                    "tool_name": {
                        "type": "string",
                        "description": _PARAMETER_LINES["tool_name"],
                    },
                    "arguments": {
                        "type": "object",
                        "description": _PARAMETER_LINES["arguments"],
                    },
                },
                "required": ["server_name", "tool_name", "arguments"],
            },
        },
    }
    return normalized_system, wrapper, catalog


def _source_row(row: Any) -> list[dict[str, Any]]:
    if not isinstance(row, dict) or set(row) != {"key", "id", "conversations"}:
        raise Reject("unknown_source_row_fields")
    if not isinstance(row["key"], str) or not row["key"]:
        raise Reject("invalid_source_key")
    if not isinstance(row["id"], str) or not row["id"]:
        raise Reject("invalid_source_id")
    messages = row["conversations"]
    if not isinstance(messages, list) or not messages:
        raise Reject("invalid_source_messages")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content", "loss"}:
            raise Reject("unknown_source_message_fields")
        if message["role"] not in {"system", "user", "assistant"}:
            raise Reject("unknown_source_role")
        if (
            not isinstance(message["content"], str)
            or type(message["loss"]) is not int
            or message["loss"] not in (0, 1)
        ):
            raise Reject("invalid_source_message")
        if message["loss"] == 1 and message["role"] != "assistant":
            raise Reject("invalid_source_loss_mask")
    if messages[0]["role"] != "system" or messages[-1]["role"] != "assistant":
        raise Reject("invalid_source_chronology")
    expected = "user"
    for message in messages[1:]:
        if message["role"] != expected:
            raise Reject("invalid_source_chronology")
        expected = "assistant" if expected == "user" else "user"
    return messages


def _has_open_markup_container(
    content: str, native_spans: tuple[tuple[int, int], ...]
) -> bool:
    """Detect a source tag left open at the terminal call boundary.

    Native reasoning and Markdown literals cannot contain the visible call, so
    their exact spans are skipped.  An orphan closing tag cannot enclose the
    call; a matching close consumes its nearest same-name opener.  This checks
    only whether any named opener remains unmatched at the call boundary, not
    whether the surrounding prose is generally well-formed XML.
    """
    protected = sorted((*native_spans, *_code_spans(content)))
    openings: list[str] = []

    def scan(start: int, end: int) -> None:
        for match in _XML_LIKE_TAG.finditer(content, start, end):
            closing, name, self_closing = match.groups()
            if self_closing:
                continue
            if not closing:
                openings.append(name)
                continue
            for index in range(len(openings) - 1, -1, -1):
                if openings[index] == name:
                    openings.pop(index)
                    break

    cursor = 0
    for start, end in protected:
        if end <= cursor:
            continue
        if start > cursor:
            scan(cursor, start)
        cursor = max(cursor, end)
    scan(cursor, len(content))
    return bool(openings)


def _convert_row(
    row: Any, line: int | str, transforms: Counter[str] | None = None
) -> ConversationSample:
    counts = transforms if transforms is not None else Counter()
    source = _source_row(row)
    system, wrapper, catalog = _extract_system(source[0]["content"])
    counts["embedded_mcp_catalogs_extracted"] += 1
    counts["xml_wire_templates_removed"] += 1
    canonical: list[dict[str, Any]] = []
    pending_result = False
    for index, message in enumerate(source):
        role = message["role"]
        content = message["content"]
        if index == 0:
            canonical.append({"role": "system", "content": system})
            continue
        if pending_result:
            if role != "user":
                raise Reject("missing_source_tool_result")
            canonical.append({"role": "tool", "content": content})
            pending_result = False
            counts["source_user_results_normalized"] += 1
            continue
        if role != "assistant":
            canonical.append({"role": role, "content": content})
            continue
        start = content.rfind("<use_mcp_tool>")
        if start < 0:
            if _CONTROL.search(content):
                raise Reject("malformed_source_tool_markup")
            canonical.append({"role": "assistant", "content": content})
            continue
        match = _CALL.fullmatch(content[start:])
        if match is None or _CONTROL.search(content[:start]):
            raise Reject("malformed_source_tool_markup")
        probe = "<think></think>"
        spans = native_reasoning_spans(content[:start] + probe)
        if not spans or spans[-1] != (
            len(content[:start]),
            len(content[:start]) + len(probe),
        ):
            raise Reject("non_top_level_source_tool_call")
        if _has_open_markup_container(content[:start], spans[:-1]):
            raise Reject("non_top_level_source_tool_call")
        server_name, tool_name, arguments_text = match.groups()
        arguments = parse_json(arguments_text)
        if not isinstance(arguments, dict):
            raise Reject("non_object_arguments")
        validator = catalog.get((server_name, tool_name))
        if validator is None:
            raise Reject("undefined_mcp_tool")
        check_arguments(arguments, validator)
        outer = {
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": arguments,
        }
        canonical.append(
            {
                "role": "assistant",
                "content": content[:start],
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "use_mcp_tool",
                            "arguments": json_bytes(outer).decode(),
                        },
                    }
                ],
            }
        )
        pending_result = True
        counts["source_xml_calls_normalized"] += 1
    if pending_result:
        raise Reject("missing_source_tool_result")
    return ConversationSample(
        messages=deepcopy(canonical),
        tools=[wrapper],
        dataset=DATASET_ID,
        sample_id=row["id"],
        raw=row,
    )


def _validate_mcp_calls(state: RowState) -> None:
    if len(state.sample.tools) != 1:
        raise Reject("malformed_mcp_wrapper")
    function = state.sample.tools[0].get("function")
    if not isinstance(function, dict) or function.get("name") != "use_mcp_tool":
        raise Reject("malformed_mcp_wrapper")
    description = function.get("description")
    if not isinstance(description, str):
        raise Reject("malformed_mcp_wrapper")
    catalog = _catalog(description)
    for index, message in enumerate(state.sample.messages):
        if message["role"] != "assistant":
            continue
        for call_index, call in enumerate(message.get("tool_calls", [])):
            arguments = state.parsed_arguments[index, call_index]
            if set(arguments) != {"server_name", "tool_name", "arguments"}:
                raise Reject("invalid_mcp_wrapper_arguments")
            server_name = arguments["server_name"]
            tool_name = arguments["tool_name"]
            nested = arguments["arguments"]
            if not isinstance(server_name, str) or not isinstance(tool_name, str):
                raise Reject("invalid_mcp_wrapper_arguments")
            if not isinstance(nested, dict):
                raise Reject("non_object_arguments")
            validator = catalog.get((server_name, tool_name))
            if validator is None:
                raise Reject("undefined_mcp_tool")
            check_arguments(nested, validator)


def _validate_visible_context(state: RowState) -> None:
    """Retain the release's explicit environment clock before every target.

    The QA construction is temporally aligned and every released trajectory has
    exactly one initial ``Today is`` value.  This bounded check does not judge
    arbitrary web-result truth or natural-language temporal reasoning.
    """
    source = _source_row(state.sample.raw)
    source_clocks = _DATE.findall(source[0]["content"])
    visible_system = state.sample.messages[0] if state.sample.messages else {}
    visible_clocks = (
        _DATE.findall(visible_system.get("content", ""))
        if visible_system.get("role") == "system"
        and isinstance(visible_system.get("content"), str)
        else []
    )
    if (
        len(source_clocks) != 1
        or len(visible_clocks) != 1
        or visible_clocks[0] != source_clocks[0]
    ):
        raise Reject("missing_or_ambiguous_source_clock")

    active_sandboxes: set[str] = set()
    for index, message in enumerate(state.sample.messages):
        if message["role"] != "assistant" or not message.get("tool_calls"):
            continue
        outer = state.parsed_arguments[index, 0]
        if outer["server_name"] != "tool-python":
            continue
        tool_name = outer["tool_name"]
        nested = outer["arguments"]
        if tool_name == "create_sandbox":
            result = state.sample.messages[index + 1]["content"]
            match = _SANDBOX_CREATED.fullmatch(result)
            if match is not None:
                active_sandboxes.add(match.group(1))
            continue
        if "sandbox_id" in nested:
            sandbox_id = nested["sandbox_id"]
            if not isinstance(sandbox_id, str) or sandbox_id not in active_sandboxes:
                raise Reject("ungrounded_sandbox_id")


def _pipeline(filters: FilterConfig) -> Pipeline:
    linkage = partial(
        link_calls,
        positional=False,
        parallel=False,
        align_results=filters.align_results,
    )
    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("structure", validate_structure),
        ("linkage", linkage),
        ("capabilities", validate_capabilities),
        ("mcp_contracts", _validate_mcp_calls),
        ("visible_context", _validate_visible_context),
    ]
    if filters.strip_thinking:
        stages.append(("reasoning", remove_reasoning))
    if filters.system_message_override is not None:
        stages.append(
            (
                "system_override",
                partial(override_system, override=filters.system_message_override),
            )
        )
    stages.extend(
        [
            ("final_structure", validate_structure),
            ("final_linkage", linkage),
            ("final_capabilities", validate_capabilities),
            ("final_mcp_contracts", _validate_mcp_calls),
            ("final_visible_context", _validate_visible_context),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def iter_load(
    dataset_config: ToolMindWebConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dataset_config is not None and not isinstance(dataset_config, ToolMindWebConfig):
        raise TypeError("dataset_config must be ToolMindWebConfig or None")
    filters = filter_config or FilterConfig(strip_thinking=True)
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    report = LoadReport(
        dataset=DATASET_ID,
        raw_count=0,
        stage1_count=0,
        filter_config=filters,
        strip_thinking_applied=filters.strip_thinking,
    )
    parse_drops: Counter[str] = Counter()
    conversion_drops: Counter[str] = Counter()
    conversion_transforms: Counter[str] = Counter()
    pipeline = _pipeline(filters)

    def candidates() -> Iterator[CurationInput]:
        for line_number, line in enumerate(
            jsonl_lines(DATASET_ID, DATASET_REVISION, [SOURCE]), 1
        ):
            report.raw_count += 1
            try:
                row = parse_json(line)
            except Reject as exc:
                parse_drops[exc.reason] += 1
                continue
            try:
                sample = _convert_row(row, line_number, conversion_transforms)
            except Reject as exc:
                conversion_drops[exc.reason] += 1
                continue
            report.stage1_count += 1
            state = pipeline.process(sample)
            if state is not None:
                yield CurationInput(state.sample, state.batches, state.parsed_arguments)

    def output() -> Generator[ConversationSample, None, None]:
        result = curate(
            candidates(),
            scope=CurationScope(DATASET_ID, SOURCE, "released"),
            config=curation_config,
        )
        final_count = 0
        try:
            for sample in result:
                final_count += 1
                yield sample
        finally:
            result.close()
        report.final_count = final_count
        report.stage1_drop_reasons = dict(parse_drops + conversion_drops)
        report.filter_drop_reasons = dict(pipeline.drops)
        report.filtered_count = pipeline.passed["termination"]
        report.dataset_config_count = report.stage1_count
        report.dataset_config_transform_counts = {
            "source": {
                "revision": DATASET_REVISION,
                "files": [SOURCE],
                "subsets": {SOURCE: report.raw_count},
            },
            "pipeline_passed": dict(pipeline.passed),
            "pipeline_drops": dict(pipeline.stage_drops),
            "transformations": dict(conversion_transforms + pipeline.transforms),
            "curation": [entry.as_dict() for entry in result.reports.values()],
        }

    return output(), report


def load(
    dataset_config: ToolMindWebConfig | None = None,
    filter_config: FilterConfig | None = None,
    *,
    curation_config: CurationConfig | None = None,
    batch_size: int = 256,
) -> tuple[list[ConversationSample], LoadReport]:
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
