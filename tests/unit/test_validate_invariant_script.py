"""CLI tests for the semantic correction-invariant validator."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_invariant.py"


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_result(path: Path, classification: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"sample_id": "s1", "classifications": [classification]}) + "\n"
    )


def test_missing_input_is_an_error(tmp_path: Path) -> None:
    result = _run(tmp_path / "missing")

    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_empty_directory_is_an_error(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.returncode == 2
    assert "no JSONL files" in result.stderr


def test_valid_justified_result_passes(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "result.jsonl",
        {"turn_index": 0, "category": "JUSTIFIED_DIRECT_ANSWER"},
    )

    result = _run(tmp_path)

    assert result.returncode == 0
    assert "INVARIANT: PASS" in result.stdout


def test_justified_result_with_correction_fails(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "result.jsonl",
        {
            "turn_index": 0,
            "category": "JUSTIFIED_DIRECT_ANSWER",
            "verification": {"correction": "Use the tool."},
        },
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "INVARIANT: FAIL" in result.stdout
