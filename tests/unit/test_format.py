"""Unit tests for fcanalysis.format.ConversationSample."""

import dataclasses

from fcanalysis.format import ConversationSample


class TestConversationSample:
    def test_construction(self) -> None:
        s = ConversationSample(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            dataset="test",
            sample_id=1,
        )
        assert s.messages == [{"role": "user", "content": "hi"}]
        assert s.tools == [{"type": "function", "function": {"name": "f"}}]
        assert s.dataset == "test"
        assert s.sample_id == 1

    def test_raw_defaults_to_empty_dict(self) -> None:
        s = ConversationSample(messages=[], tools=[], dataset="test", sample_id="x")
        assert s.raw == {}

    def test_raw_is_distinct_per_instance(self) -> None:
        a = ConversationSample(messages=[], tools=[], dataset="t", sample_id=1)
        b = ConversationSample(messages=[], tools=[], dataset="t", sample_id=2)
        a.raw["k"] = "v"
        assert b.raw == {}

    def test_repr_excludes_raw(self) -> None:
        s = ConversationSample(
            messages=[], tools=[], dataset="d", sample_id=1, raw={"hidden": 1}
        )
        assert "hidden" not in repr(s)
        assert "raw" not in repr(s)

    def test_str_or_int_sample_id(self) -> None:
        a = ConversationSample(messages=[], tools=[], dataset="d", sample_id="abc")
        b = ConversationSample(messages=[], tools=[], dataset="d", sample_id=42)
        assert a.sample_id == "abc"
        assert b.sample_id == 42

    def test_asdict_roundtrip(self) -> None:
        s = ConversationSample(
            messages=[{"role": "user", "content": "u"}],
            tools=[],
            dataset="d",
            sample_id="s",
            raw={"r": 1},
        )
        d = dataclasses.asdict(s)
        rebuilt = ConversationSample(**d)
        assert rebuilt == s

    def test_equality(self) -> None:
        a = ConversationSample(messages=[], tools=[], dataset="d", sample_id=1)
        b = ConversationSample(messages=[], tools=[], dataset="d", sample_id=1)
        c = ConversationSample(messages=[], tools=[], dataset="d", sample_id=2)
        assert a == b
        assert a != c
