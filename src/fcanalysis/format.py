from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ConversationSample:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    dataset: str
    sample_id: str | int
    # Loader-added curation facts. Unlike ``raw``, these annotations are part
    # of the normalized sample and may be serialized for downstream selection.
    # They must remain outside model input: chat templates and training
    # formatters must consume only ``messages`` and ``tools``.
    annotations: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
