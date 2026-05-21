"""Unit tests for nemotron_agentic_v1 loader-specific predicates.

The predicates operate on raw input shapes (raw_tools list, raw
messages list) rather than ConversationSample, because they are used
during the Stage 1 conversion pass before a ConversationSample exists.
"""

from typing import Any

from fcanalysis.loaders.nemotron_agentic_v1 import (
    _has_conflicting_duplicate_tools,
    _has_consecutive_same_role,
    _has_duplicate_tool_names,
    _has_orphan_calls,
)


def _f(name: str, **extras: object) -> dict[str, Any]:
    func: dict[str, Any] = {"name": name}
    func.update(extras)
    return {"type": "function", "function": func}


class TestHasConflictingDuplicateTools:
    def test_empty_tools(self) -> None:
        assert _has_conflicting_duplicate_tools([]) is False

    def test_unique_names(self) -> None:
        assert _has_conflicting_duplicate_tools([_f("a"), _f("b")]) is False

    def test_duplicate_name_same_body(self) -> None:
        assert _has_conflicting_duplicate_tools([_f("a"), _f("a")]) is False

    def test_duplicate_name_different_body(self) -> None:
        assert (
            _has_conflicting_duplicate_tools(
                [_f("a", description="x"), _f("a", description="y")]
            )
            is True
        )

    def test_canonicalization_ignores_key_order(self) -> None:
        a1 = {"type": "function", "function": {"name": "a", "description": "x"}}
        a2 = {"type": "function", "function": {"description": "x", "name": "a"}}
        assert _has_conflicting_duplicate_tools([a1, a2]) is False

    def test_non_dict_tool_skipped(self) -> None:
        assert _has_conflicting_duplicate_tools(["not a dict", _f("a")]) is False

    def test_missing_function_key_skipped(self) -> None:
        assert (
            _has_conflicting_duplicate_tools([{"type": "function"}, _f("a")]) is False
        )

    def test_non_string_name_skipped(self) -> None:
        tool = {"type": "function", "function": {"name": 123}}
        assert _has_conflicting_duplicate_tools([tool, _f("a")]) is False


class TestHasDuplicateToolNames:
    def test_empty(self) -> None:
        assert _has_duplicate_tool_names([]) is False

    def test_unique(self) -> None:
        assert _has_duplicate_tool_names([_f("a"), _f("b"), _f("c")]) is False

    def test_repeat(self) -> None:
        assert _has_duplicate_tool_names([_f("a"), _f("b"), _f("a")]) is True

    def test_repeat_with_same_body(self) -> None:
        # Even same-body duplicates count as duplicate names here.
        # (_has_conflicting_duplicate_tools is the stricter variant.)
        assert _has_duplicate_tool_names([_f("a"), _f("a")]) is True

    def test_skips_malformed_entries(self) -> None:
        assert _has_duplicate_tool_names([None, _f("a"), {"type": "function"}]) is False


class TestHasOrphanCalls:
    def test_empty(self) -> None:
        assert _has_orphan_calls([]) is False

    def test_balanced_single_turn(self) -> None:
        msgs = [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "f"}}
                ],
            },
            {"role": "tool", "content": "r", "tool_call_id": "1"},
        ]
        assert _has_orphan_calls(msgs) is False

    def test_too_few_responses(self) -> None:
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
            }
        ]
        assert _has_orphan_calls(msgs) is True

    def test_extra_responses_not_orphans(self) -> None:
        # More responses than calls is NOT an orphan situation (predicate only
        # flags when calls > responses).
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "f"}}
                ],
            },
            {"role": "tool", "content": "r1", "tool_call_id": "1"},
            {"role": "tool", "content": "r2", "tool_call_id": "1"},
        ]
        assert _has_orphan_calls(msgs) is False

    def test_multi_turn_second_orphan(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "f"}}
                ],
            },
            {"role": "tool", "content": "r", "tool_call_id": "1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "2", "type": "function", "function": {"name": "g"}},
                    {"id": "3", "type": "function", "function": {"name": "h"}},
                ],
            },
            {"role": "tool", "content": "r2", "tool_call_id": "2"},
        ]
        assert _has_orphan_calls(msgs) is True


class TestHasConsecutiveSameRole:
    def test_empty(self) -> None:
        assert _has_consecutive_same_role([]) is False

    def test_single_message(self) -> None:
        assert _has_consecutive_same_role([{"role": "user"}]) is False

    def test_alternating(self) -> None:
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
        ]
        assert _has_consecutive_same_role(msgs) is False

    def test_two_consecutive_users(self) -> None:
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
        ]
        assert _has_consecutive_same_role(msgs) is True

    def test_two_consecutive_assistants(self) -> None:
        msgs = [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        assert _has_consecutive_same_role(msgs) is True

    def test_two_consecutive_tools(self) -> None:
        # Tool/tool sequences ARE common (multiple responses after parallel calls),
        # but this predicate flags them anyway.
        msgs = [
            {"role": "tool", "content": "r1", "tool_call_id": "1"},
            {"role": "tool", "content": "r2", "tool_call_id": "2"},
        ]
        assert _has_consecutive_same_role(msgs) is True
