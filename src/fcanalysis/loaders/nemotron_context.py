"""Exact asynchronous input/lifecycle grammars observed in pinned Nemotron.

The definitions for get_serp_async and SimilarWeb GetCompleteDataAsync state
that they return task_id and require GetTaskResult polling to obtain completed
data. Their simulated dictionaries expose created/queued/processing/started/
task_created and pending/inprogress/succeeded states. data.gov.sg's initiate
and poll operations use the same datasetId/columnNames/filters request, returning
an initiation message followed by a download URL. Other listed operations are
status/fetch queries: their identifier must already be visible. A poll-only
status query does not itself start a job. Delivery/project/social poll tools
are not asynchronous compute protocols merely because of their names.

data.gov.sg's published contract documents matching initiate/poll arguments
and allows direct polling for non-CSV datasets:
https://guide.data.gov.sg/developer-guide/dataset-apis/download-dataset .
The matching MCP adapter forwards data.data as JSON text without adding state:
https://github.com/aniruddha-adhikary/gahmen-mcp/blob/main/src/datagovsg.tools.ts .
The exact initiation message and completed URL shapes below occur in the
pinned release; undocumented message variants remain unresolved.

These grammars cover released structural evidence, not arbitrary prose claims
of successful completion. No runtime grant is inferred from a discovery name.
"""

from typing import Any

from .context import collect_string_values, contains_literal_token
from .normalization import Reject, json_bytes, parse_json
from .pipeline import JobEvent, RowState, validate_lifecycle

_STARTS = frozenset(
    {"get_serp_async", "getcompletedata_async_creates_a_task_to_process"}
)
_POLL_ARGUMENT = {
    "gettaskresult_free_of_use": "task_id",
    "retrievetaskstatus": "task_id",
    "poll_session_results": "job_id",
    "get_job_by_task_id": "task_id",
    "get_job_status": "is_id",
    "get_call": "request_id",
    "jobs_id": "is_id",
    "retrieve_text": "transactionid",
    "datagovsg_poll_download": "datasetId",
    "datagovsg_initiate_download": "datasetId",
}
_PENDING = frozenset(
    {
        "created",
        "queued",
        "processing",
        "started",
        "task_created",
        "pending",
        "inprogress",
        "PENDING",
        "RECEIVED",
        "STARTED",
    }
)
_COMPLETE = frozenset({"succeeded", "completed", "success"})
_PAYLOADS = ("result", "results", "data", "output", "downloadUrl", "url")


def _result(content: str) -> Any:
    try:
        return parse_json(content)
    except Reject:
        return None


def validate_async(state: RowState) -> None:
    """Ground exact job inputs; complete every job started in the conversation."""
    names = {
        call["function"]["name"]
        for msg in state.sample.messages
        for call in msg.get("tool_calls", [])
    }
    if not (names & (_STARTS | _POLL_ARGUMENT.keys())):
        return
    visible: list[str] = []
    leaves: set[str] = set()
    started: set[str] = set()
    download_requests: dict[str, bytes] = {}
    events: list[JobEvent] = []
    bindings = {b.assistant_index: b for b in state.batches}

    for index, msg in enumerate(state.sample.messages):
        if msg["role"] in ("system", "user"):
            if isinstance(msg.get("content"), str):
                visible.append(msg["content"])
        elif msg["role"] == "tool":
            result = _result(msg["content"])
            if result is None:
                visible.append(msg["content"])
            else:
                collect_string_values(result, leaves)
        elif msg["role"] == "assistant" and index in bindings:
            batch = bindings[index]
            # Check all inputs before interpreting any result from this batch.
            for ci, call in enumerate(msg.get("tool_calls", [])):
                name = call["function"]["name"]
                args = state.parsed_arguments[index, ci]
                if name in _POLL_ARGUMENT:
                    value = args.get(_POLL_ARGUMENT[name])
                    if (
                        not isinstance(value, str)
                        or not value
                        or (
                            value not in leaves
                            and not contains_literal_token(value, visible)
                        )
                    ):
                        raise Reject("ungrounded_job_identifier")
                    if name == "datagovsg_poll_download" and value in download_requests:
                        if json_bytes(args) != download_requests[value]:
                            raise Reject("mismatched_download_request")
            for ci, ri in enumerate(batch.result_indices):
                name = msg["tool_calls"][ci]["function"]["name"]
                args = state.parsed_arguments[index, ci]
                if name not in _STARTS and name not in (
                    "gettaskresult_free_of_use",
                    "datagovsg_initiate_download",
                    "datagovsg_poll_download",
                ):
                    continue
                result = _result(state.sample.messages[ri]["content"])
                if not isinstance(result, dict):
                    raise Reject("unresolved_async_state")
                if name in _STARTS:
                    # A visible failure creates no successful background job.
                    if result.get("error") and not result.get("task_id"):
                        continue
                    task = result.get("task_id")
                    if not isinstance(task, str) or not task:
                        raise Reject("missing_started_job_identifier")
                    phase = result.get("status")
                    if phase not in _PENDING | _COMPLETE:
                        raise Reject("unresolved_async_state")
                    if task in started:
                        raise Reject("reused_started_job_identifier")
                    started.add(task)
                    events.append(JobEvent(task, "started"))
                    if phase in _COMPLETE:
                        events.append(
                            JobEvent(
                                task,
                                "completed",
                                payload=any(result.get(k) for k in _PAYLOADS),
                            )
                        )
                elif name == "gettaskresult_free_of_use":
                    task = args["task_id"]
                    if "task_id" in result and result["task_id"] != task:
                        raise Reject("mismatched_job_identifier")
                    phase = result.get("status")
                    if phase in _PENDING:
                        events.append(JobEvent(task, "pending"))
                    elif phase in _COMPLETE:
                        events.append(
                            JobEvent(
                                task,
                                "completed",
                                payload=any(result.get(k) for k in _PAYLOADS),
                            )
                        )
                    elif phase in ("failed", "error") and result.get("error"):
                        # The visible runtime has terminated this attempt in failure.
                        events.append(JobEvent(task, "abandoned"))
                    elif task in started:
                        raise Reject("unresolved_async_state")
                elif name == "datagovsg_initiate_download":
                    task = args["datasetId"]
                    if result.get("error"):
                        continue
                    if result != {
                        "message": "Download successfully initiated. Proceed to poll download"
                    }:
                        raise Reject("unresolved_async_state")
                    if task in download_requests:
                        raise Reject("ambiguous_repeated_download")
                    download_requests[task] = json_bytes(args)
                    events.append(JobEvent(task, "started"))
                elif name == "datagovsg_poll_download":
                    task = args["datasetId"]
                    if task not in download_requests:
                        continue
                    url = result.get("url")
                    if isinstance(url, str) and url:
                        events.append(JobEvent(task, "completed", payload=True))
                    else:
                        raise Reject("unresolved_async_state")
    validate_lifecycle(events, require_completion_for="started")
