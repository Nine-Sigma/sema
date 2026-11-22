"""The sensitive-value hook refuses staged text, messages and ranges by key name only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCAN = Path(__file__).parents[2] / "scripts" / "git-hooks" / "scan-sensitive.sh"
VALUE = "toy-value-0001"


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".env").write_text(f"OTHER=ignored\nPARTNER_TOY={VALUE}\n")
    (tmp_path / ".sensitive-patterns").write_text("# note\ntoy-fragment\n")
    return tmp_path


def _scan(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCAN), *args], cwd=repo, capture_output=True, text=True, check=False
    )


def _stage(repo: Path, text: str) -> None:
    (repo / "f.txt").write_text(text)
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)


def test_staged_text_holding_a_partner_value_is_refused_by_key_name(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, f"host = {VALUE}\n")
    result = _scan(repo, "staged")
    assert result.returncode == 1
    assert "PARTNER_TOY" in result.stderr and VALUE not in result.stderr + result.stdout


def test_a_listed_pattern_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "a toy-fragment here\n")
    assert _scan(repo, "staged").returncode == 1


def test_clean_staged_text_and_other_env_keys_pass(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "host = ignored\n")
    assert _scan(repo, "staged").returncode == 0


def test_a_commit_message_holding_a_value_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    message = repo / "MSG"
    message.write_text(f"feat: point at {VALUE}\n")
    assert _scan(repo, "message", str(message)).returncode == 1
