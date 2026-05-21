"""Unit tests for nemotron_agentic_v2 loader-specific predicates.

The three predicates here mirror v1's implementations.
"""

from typing import Any

from fcanalysis.loaders.nemotron_agentic_v2 import (
    _has_consecutive_same_role,
    _has_duplicate_tool_names,
    _has_orphan_calls,
)


def _f(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}


class TestHasDuplicateToolNames:
    def test_empty(self) -> None:
        assert _has_duplicate_tool_names([]) is False

    def test_unique(self) -> None:
        assert _has_duplicate_tool_names([_f("a"), _f("b")]) is False

    def test_repeat(self) -> None:
        assert _has_duplicate_tool_names([_f("a"), _f("a")]) is True

    def test_skips_non_dict(self) -> None:
        assert _has_duplicate_tool_names([_f("a"), "garbage", _f("b")]) is False

    def test_skips_missing_function(self) -> None:
        assert _has_duplicate_tool_names([{"type": "function"}, _f("a")]) is False

    def test_skips_non_string_name(self) -> None:
        bad = {"type": "function", "function": {"name": 7}}
        assert _has_duplicate_tool_names([bad, _f("a")]) is False


class TestHasOrphanCalls:
    def test_balanced(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "f"}}
                ],
            },
            {"role": "tool", "content": "r", "tool_call_id": "1"},
        ]
        assert _has_orphan_calls(msgs) is False

    def test_orphan(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "f"}},
                    {"id": "2", "type": "function", "function": {"name": "g"}},
                ],
            },
            {"role": "tool", "content": "r", "tool_call_id": "1"},
        ]
        assert _has_orphan_calls(msgs) is True

    def test_zero_responses(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "f"}}
                ],
            },
        ]
        assert _has_orphan_calls(msgs) is True

    def test_no_assistant_tool_calls(self) -> None:
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "text"},
        ]
        assert _has_orphan_calls(msgs) is False


class TestHasConsecutiveSameRole:
    def test_empty(self) -> None:
        assert _has_consecutive_same_role([]) is False

    def test_single(self) -> None:
        assert _has_consecutive_same_role([{"role": "user"}]) is False

    def test_alternating(self) -> None:
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        assert _has_consecutive_same_role(msgs) is False

    def test_consecutive(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        assert _has_consecutive_same_role(msgs) is True
