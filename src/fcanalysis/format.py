from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ConversationSample:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    dataset: str
    sample_id: str | int
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
