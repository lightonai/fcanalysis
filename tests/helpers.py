"""Factory helpers for synthetic ConversationSample inputs in unit tests."""

from typing import Any

from fcanalysis.format import ConversationSample


def sample(
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    dataset: str = "test",
    sample_id: str | int = 0,
) -> ConversationSample:
    return ConversationSample(
        messages=messages or [],
        tools=tools or [],
        dataset=dataset,
        sample_id=sample_id,
    )


def assistant(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    msg.update(extra)
    return msg


def user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def system(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def tool_response(content: str = "ok", tool_call_id: str = "call_1") -> dict[str, Any]:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def call(
    name: str = "f",
    arguments: str | dict[str, Any] = "{}",
    call_id: str = "call_1",
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def func(
    name: str = "f",
    parameters: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
            or {"type": "object", "properties": {}, "required": []},
        },
    }
