"""Unit tests for fcanalysis.overlap: dedup primitives and pipelines.

Layer 2 P0. The seed key and decision sequence functions produce the
content fingerprints that determine the final training-mix sample
counts (200K v1 after within-dedup, 187K txt360 after cross-dedup).
Any regression here changes the mix downstream.
"""

import hashlib

from fcanalysis.overlap import (
    compute_seed_key,
    dedup_cross,
    dedup_within,
    extract_decision_sequence,
    find_duplicates,
    normalize_text,
)

from tests.helpers import assistant, call, func, sample, system, tool_response, user


class TestNormalizeText:
    def test_lowercases(self) -> None:
        assert normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self) -> None:
        assert normalize_text("hi, there!") == "hi there"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("a  b\t c\n\nd") == "a b c d"

    def test_nfc_combines_diacritics(self) -> None:
        # Combining-acute "e" + U+0301 normalizes to precomposed "é"
        decomposed = "café"
        precomposed = "café"
        assert normalize_text(decomposed) == normalize_text(precomposed)

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_text("   \t\n  ") == ""

    def test_only_punctuation(self) -> None:
        assert normalize_text("!?.,;") == ""

    def test_unicode_letter_preserved(self) -> None:
        # Non-ASCII letters survive (not in ASCII punctuation table).
        assert normalize_text("Ångström") == "ångström"


class TestComputeSeedKey:
    def test_deterministic(self) -> None:
        s = sample(messages=[user("hello")], tools=[func("f"), func("g")])
        assert compute_seed_key(s) == compute_seed_key(s)

    def test_different_user_message_changes_key(self) -> None:
        a = sample(messages=[user("hello")], tools=[func("f")])
        b = sample(messages=[user("world")], tools=[func("f")])
        assert compute_seed_key(a) != compute_seed_key(b)

    def test_different_tools_changes_key(self) -> None:
        a = sample(messages=[user("hi")], tools=[func("f")])
        b = sample(messages=[user("hi")], tools=[func("g")])
        assert compute_seed_key(a) != compute_seed_key(b)

    def test_tool_order_does_not_matter(self) -> None:
        a = sample(messages=[user("hi")], tools=[func("a"), func("b")])
        b = sample(messages=[user("hi")], tools=[func("b"), func("a")])
        assert compute_seed_key(a) == compute_seed_key(b)

    def test_case_and_punct_normalized(self) -> None:
        a = sample(messages=[user("Hello, world!")], tools=[func("F")])
        b = sample(messages=[user("hello world")], tools=[func("f")])
        assert compute_seed_key(a) == compute_seed_key(b)

    def test_normalize_false_preserves_case(self) -> None:
        a = sample(messages=[user("Hi")], tools=[func("F")])
        b = sample(messages=[user("hi")], tools=[func("f")])
        assert compute_seed_key(a, normalize=False) != compute_seed_key(
            b, normalize=False
        )

    def test_first_user_only(self) -> None:
        # Only the first user message contributes; later turns are ignored.
        a = sample(
            messages=[
                user("first"),
                assistant(content="ok"),
                user("second"),
            ],
            tools=[func("f")],
        )
        b = sample(messages=[user("first")], tools=[func("f")])
        assert compute_seed_key(a) == compute_seed_key(b)

    def test_system_message_does_not_affect_key(self) -> None:
        a = sample(messages=[system("S"), user("u")], tools=[func("f")])
        b = sample(messages=[user("u")], tools=[func("f")])
        assert compute_seed_key(a) == compute_seed_key(b)

    def test_output_is_sha256_hex(self) -> None:
        s = sample(messages=[user("x")], tools=[func("f")])
        key = compute_seed_key(s)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
        # Manual recomputation to lock the construction
        payload = "x" + "\0" + "f"
        assert key == hashlib.sha256(payload.encode()).hexdigest()


class TestExtractDecisionSequence:
    def test_empty_when_no_calls(self) -> None:
        s = sample(messages=[user("u"), assistant(content="text only")])
        assert extract_decision_sequence(s) == ()

    def test_single_call(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments='{"x":1}')])]
        )
        assert extract_decision_sequence(s) == (("f", ("x",)),)

    def test_multiple_args_sorted(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments='{"b":1,"a":2}')])]
        )
        assert extract_decision_sequence(s) == (("f", ("a", "b")),)

    def test_call_order_preserved(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[
                        call(name="first", call_id="1", arguments="{}"),
                        call(name="second", call_id="2", arguments="{}"),
                    ]
                )
            ]
        )
        assert extract_decision_sequence(s) == (("first", ()), ("second", ()))

    def test_function_name_normalized(self) -> None:
        # normalize_text lowercases and strips ASCII punctuation; underscore
        # is in string.punctuation, so "Get_Item!" collapses to "getitem".
        s = sample(
            messages=[assistant(tool_calls=[call(name="Get_Item!", arguments="{}")])]
        )
        seq = extract_decision_sequence(s)
        assert seq == (("getitem", ()),)

    def test_unparseable_arguments_become_empty_keys(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments="{not json")])]
        )
        assert extract_decision_sequence(s) == (("f", ()),)

    def test_non_dict_arguments_become_empty_keys(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(name="f", arguments="[1,2]")])])
        assert extract_decision_sequence(s) == (("f", ()),)

    def test_dict_arguments_passed_through(self) -> None:
        tc = call(name="f")
        tc["function"]["arguments"] = {"y": 1, "x": 2}
        s = sample(messages=[assistant(tool_calls=[tc])])
        assert extract_decision_sequence(s) == (("f", ("x", "y")),)

    def test_multi_turn_calls_concatenated(self) -> None:
        s = sample(
            messages=[
                assistant(tool_calls=[call(name="a", call_id="1", arguments="{}")]),
                tool_response(tool_call_id="1"),
                assistant(tool_calls=[call(name="b", call_id="2", arguments="{}")]),
            ]
        )
        assert extract_decision_sequence(s) == (("a", ()), ("b", ()))


class TestDedupWithin:
    def test_empty(self) -> None:
        out, stats = dedup_within([])
        assert out == []
        assert stats.input_samples == 0
        assert stats.output_samples == 0
        assert stats.removed_samples == 0
        assert stats.unique_seed_keys == 0
        assert stats.duplicate_seed_keys == 0

    def test_all_unique_keys(self) -> None:
        s1 = sample(messages=[user("a")], tools=[func("f")], sample_id=1)
        s2 = sample(messages=[user("b")], tools=[func("f")], sample_id=2)
        s3 = sample(messages=[user("c")], tools=[func("f")], sample_id=3)
        out, stats = dedup_within([s1, s2, s3])
        assert len(out) == 3
        assert stats.removed_samples == 0
        assert stats.unique_seed_keys == 3
        assert stats.duplicate_seed_keys == 0

    def test_keeps_first_of_duplicates(self) -> None:
        s1 = sample(messages=[user("same")], tools=[func("f")], sample_id="first")
        s2 = sample(messages=[user("same")], tools=[func("f")], sample_id="second")
        s3 = sample(messages=[user("same")], tools=[func("f")], sample_id="third")
        out, stats = dedup_within([s1, s2, s3])
        assert len(out) == 1
        assert out[0].sample_id == "first"
        assert stats.removed_samples == 2
        assert stats.unique_seed_keys == 1
        assert stats.duplicate_seed_keys == 1

    def test_order_preserved_for_kept(self) -> None:
        a = sample(messages=[user("a")], tools=[func("f")], sample_id="a")
        b = sample(messages=[user("b")], tools=[func("f")], sample_id="b")
        c = sample(messages=[user("c")], tools=[func("f")], sample_id="c")
        out, _ = dedup_within([b, a, c])
        assert [s.sample_id for s in out] == ["b", "a", "c"]

    def test_mixed(self) -> None:
        a1 = sample(messages=[user("a")], tools=[func("f")], sample_id="a1")
        a2 = sample(messages=[user("a")], tools=[func("f")], sample_id="a2")
        b1 = sample(messages=[user("b")], tools=[func("f")], sample_id="b1")
        b2 = sample(messages=[user("b")], tools=[func("f")], sample_id="b2")
        b3 = sample(messages=[user("b")], tools=[func("f")], sample_id="b3")
        c1 = sample(messages=[user("c")], tools=[func("f")], sample_id="c1")
        out, stats = dedup_within([a1, b1, a2, b2, c1, b3])
        assert [s.sample_id for s in out] == ["a1", "b1", "c1"]
        assert stats.input_samples == 6
        assert stats.output_samples == 3
        assert stats.removed_samples == 3
        assert stats.unique_seed_keys == 3
        assert stats.duplicate_seed_keys == 2


class TestDedupCross:
    def test_empty_primary_keeps_all_secondary(self) -> None:
        s = sample(messages=[user("a")], tools=[func("f")])
        out, stats = dedup_cross([], [s])
        assert out == [s]
        assert stats.removed_samples == 0
        assert stats.shared_seed_keys == 0
        assert stats.primary_seed_keys == 0

    def test_empty_secondary(self) -> None:
        p = sample(messages=[user("a")], tools=[func("f")])
        out, stats = dedup_cross([p], [])
        assert out == []
        assert stats.removed_samples == 0
        assert stats.shared_seed_keys == 0
        assert stats.primary_seed_keys == 1

    def test_full_overlap_removes_all(self) -> None:
        p = sample(messages=[user("a")], tools=[func("f")], sample_id="p")
        s = sample(messages=[user("a")], tools=[func("f")], sample_id="s")
        out, stats = dedup_cross([p], [s])
        assert out == []
        assert stats.removed_samples == 1
        assert stats.shared_seed_keys == 1

    def test_no_overlap_keeps_all(self) -> None:
        p = sample(messages=[user("a")], tools=[func("f")])
        s = sample(messages=[user("b")], tools=[func("f")])
        out, stats = dedup_cross([p], [s])
        assert out == [s]
        assert stats.removed_samples == 0
        assert stats.shared_seed_keys == 0

    def test_partial_overlap(self) -> None:
        p1 = sample(messages=[user("a")], tools=[func("f")], sample_id="p1")
        p2 = sample(messages=[user("b")], tools=[func("f")], sample_id="p2")
        s1 = sample(messages=[user("a")], tools=[func("f")], sample_id="s1")
        s2 = sample(messages=[user("c")], tools=[func("f")], sample_id="s2")
        s3 = sample(messages=[user("b")], tools=[func("f")], sample_id="s3")
        out, stats = dedup_cross([p1, p2], [s1, s2, s3])
        assert [s.sample_id for s in out] == ["s2"]
        assert stats.removed_samples == 2
        assert stats.shared_seed_keys == 2
        assert stats.primary_seed_keys == 2
        assert stats.secondary_input_samples == 3
        assert stats.secondary_output_samples == 1

    def test_primary_not_mutated(self) -> None:
        p = sample(messages=[user("a")], tools=[func("f")], sample_id="p")
        s = sample(messages=[user("a")], tools=[func("f")], sample_id="s")
        primary_in = [p]
        dedup_cross(primary_in, [s])
        assert primary_in == [p]

    def test_uses_normalized_key(self) -> None:
        # Case + punctuation normalize to the same key.
        p = sample(messages=[user("Hello!")], tools=[func("f")])
        s = sample(messages=[user("hello")], tools=[func("f")])
        out, stats = dedup_cross([p], [s])
        assert out == []
        assert stats.shared_seed_keys == 1


class TestFindDuplicates:
    def test_no_duplicates(self) -> None:
        samples = [
            sample(messages=[user("a")], tools=[func("f")], sample_id=1),
            sample(messages=[user("b")], tools=[func("f")], sample_id=2),
        ]
        report = find_duplicates(samples)
        assert report.total_samples == 2
        assert report.unique_seed_keys == 2
        assert report.duplicate_seed_keys == 0
        assert report.redundant_groups == 0
        assert report.augmented_groups == 0

    def test_redundant_group_same_decisions(self) -> None:
        # Same seed key, same single call: classified redundant.
        s1 = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="f", arguments='{"x":1}')]),
            ],
            tools=[func("f")],
            sample_id=1,
        )
        s2 = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="f", arguments='{"x":2}')]),
            ],
            tools=[func("f")],
            sample_id=2,
        )
        report = find_duplicates([s1, s2])
        assert report.duplicate_seed_keys == 1
        assert report.redundant_groups == 1
        assert report.augmented_groups == 0
        assert report.redundant_extra_samples == 1

    def test_augmented_group_different_decisions(self) -> None:
        s1 = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="f", arguments="{}")]),
            ],
            tools=[func("f"), func("g")],
            sample_id=1,
        )
        s2 = sample(
            messages=[
                user("u"),
                assistant(tool_calls=[call(name="g", arguments="{}")]),
            ],
            tools=[func("f"), func("g")],
            sample_id=2,
        )
        report = find_duplicates([s1, s2])
        assert report.duplicate_seed_keys == 1
        assert report.redundant_groups == 0
        assert report.augmented_groups == 1
        assert report.augmented_extra_samples == 1
