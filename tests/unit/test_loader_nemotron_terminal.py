"""Unit tests for nemotron_terminal loader-specific predicates/transforms."""

from fcanalysis.loaders.nemotron_terminal import _has_valid_commands, _strip_think


class TestHasValidCommands:
    def test_well_formed_single_command(self) -> None:
        parsed = {"commands": [{"keystrokes": "ls\n"}]}
        assert _has_valid_commands(parsed) is True

    def test_well_formed_multiple_commands(self) -> None:
        parsed = {
            "commands": [
                {"keystrokes": "cd /tmp\n"},
                {"keystrokes": "ls\n"},
            ]
        }
        assert _has_valid_commands(parsed) is True

    def test_missing_commands_key(self) -> None:
        assert _has_valid_commands({}) is False

    def test_commands_is_none(self) -> None:
        assert _has_valid_commands({"commands": None}) is False

    def test_commands_not_a_list(self) -> None:
        assert _has_valid_commands({"commands": "ls"}) is False
        assert _has_valid_commands({"commands": {"keystrokes": "ls"}}) is False

    def test_empty_commands_list(self) -> None:
        assert _has_valid_commands({"commands": []}) is False

    def test_command_missing_keystrokes(self) -> None:
        # Observed typo variant ystrokes must be rejected.
        parsed = {"commands": [{"ystrokes": "ls\n"}]}
        assert _has_valid_commands(parsed) is False

    def test_command_with_trailing_colon_typo(self) -> None:
        # Observed variant keystrokes: (trailing colon) is treated as a
        # different key and must be rejected.
        parsed = {"commands": [{"keystrokes:": "ls\n"}]}
        assert _has_valid_commands(parsed) is False

    def test_command_not_a_dict(self) -> None:
        parsed = {"commands": ["ls\n"]}
        assert _has_valid_commands(parsed) is False

    def test_mixed_valid_and_invalid(self) -> None:
        parsed = {
            "commands": [
                {"keystrokes": "ls\n"},
                {"ystrokes": "cd /\n"},
            ]
        }
        assert _has_valid_commands(parsed) is False


class TestStripThink:
    def test_removes_think_block(self) -> None:
        assert _strip_think("<think>hidden</think>visible") == "visible"

    def test_removes_multiple_think_blocks(self) -> None:
        result = _strip_think("a<think>x</think>b<think>y</think>c")
        assert result == "abc"

    def test_strips_outer_whitespace(self) -> None:
        assert _strip_think("  visible  ") == "visible"

    def test_strips_whitespace_left_by_think_removal(self) -> None:
        assert _strip_think("<think>x</think>   ") == ""

    def test_no_think_blocks_unchanged(self) -> None:
        assert _strip_think("plain text") == "plain text"

    def test_multiline_think_block(self) -> None:
        assert _strip_think("a<think>line1\nline2</think>b") == "ab"
