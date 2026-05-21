"""Unit tests for txt360 loader-specific filter predicates."""

from fcanalysis.loaders.txt360 import _has_qualifying_user


class TestHasQualifyingUser:
    def test_user_with_content(self) -> None:
        assert _has_qualifying_user([{"role": "user", "content": "hi"}]) is True

    def test_user_with_empty_string(self) -> None:
        assert _has_qualifying_user([{"role": "user", "content": ""}]) is False

    def test_user_with_whitespace_only(self) -> None:
        assert _has_qualifying_user([{"role": "user", "content": "   \t\n"}]) is False

    def test_user_with_none_content(self) -> None:
        assert _has_qualifying_user([{"role": "user", "content": None}]) is False

    def test_user_missing_content(self) -> None:
        assert _has_qualifying_user([{"role": "user"}]) is False

    def test_user_with_non_string_content(self) -> None:
        assert _has_qualifying_user([{"role": "user", "content": ["x"]}]) is False

    def test_no_user_messages(self) -> None:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
        ]
        assert _has_qualifying_user(msgs) is False

    def test_qualifying_user_after_non_qualifying(self) -> None:
        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "real question"},
        ]
        assert _has_qualifying_user(msgs) is True

    def test_empty_messages(self) -> None:
        assert _has_qualifying_user([]) is False
