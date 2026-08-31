"""Validate the semantic correction invariant.

Invariant: a turn's final label is anti  <=>  a usable correction lives at
verification.correction. The only allowed exceptions are legacy old-stage-2
failures (anti with no correction). A justified turn must NEVER carry a
correction. Prints PASS/FAIL; exit code 1 on violation so a chain can gate on it.

Pass either one semantic-result JSONL file or a directory containing JSONL
files. An empty input is an error rather than a vacuous pass.
"""

import argparse
import json
import sys
from pathlib import Path


def is_anti(category: str) -> bool:
    return bool(category) and (
        category.startswith("ANTI_") or category == "OTHER_UNJUSTIFIED"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="semantic-result JSONL file or directory containing JSONL files",
    )
    return parser.parse_args()


def result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if files:
            return files
        raise ValueError(f"no JSONL files found in {path}")
    raise ValueError(f"result path does not exist: {path}")


def main() -> int:
    args = parse_args()
    try:
        files = result_files(args.results)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    anti = anti_corr = anti_nocorr = just = just_corr = contested = 0
    classified_turns = 0
    violations: list[tuple[object, object, object, str]] = []
    for filename in files:
        with filename.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"ERROR: invalid JSON at {filename}:{line_number}: {exc}",
                        file=sys.stderr,
                    )
                    return 2
                if "error" in result:
                    continue
                for classification in result.get("classifications", []):
                    classified_turns += 1
                    verification = classification.get("verification") or {}
                    has_correction = isinstance(verification, dict) and bool(
                        verification.get("correction")
                    )
                    if classification.get("contested"):
                        contested += 1
                    if is_anti(classification.get("category", "")):
                        anti += 1
                        if has_correction:
                            anti_corr += 1
                        else:
                            anti_nocorr += 1
                    else:
                        just += 1
                        if has_correction:
                            just_corr += 1
                            if len(violations) < 5:
                                violations.append(
                                    (
                                        result.get("sample_id"),
                                        classification.get("turn_index"),
                                        classification.get("category"),
                                        "JUSTIFIED w/ corr",
                                    )
                                )

    if classified_turns == 0:
        print("ERROR: no classified turns found", file=sys.stderr)
        return 2

    print(
        f"anti={anti}  anti_with_corr={anti_corr}  "
        f"anti_without_corr={anti_nocorr} (legacy)"
    )
    print(f"justified={just}  justified_with_corr={just_corr} (MUST be 0)")
    print(f"contested={contested}")
    ok = just_corr == 0
    print("INVARIANT:", "PASS" if ok else "FAIL")
    for violation in violations:
        print("  violation:", violation)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
