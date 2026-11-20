import json
import shutil
import subprocess
from pathlib import Path

import pytest

from showcase.ontology_query_benchmark import freeze

pytestmark = pytest.mark.unit

PATTERNS = ("src/toy/*.py", "pyproject.toml", "uv.lock", "config/bench.json",
            "config/*_spec.json", "questions/*.json", "gold.json", "cache/*")
CLOSURE_FILES = (
    "src/toy/__init__.py",
    "src/toy/workflow.py",
    "pyproject.toml",
    "uv.lock",
    "config/bench.json",
    "config/people_spec.json",
    "config/events_spec.json",
    "questions/regular.json",
    "questions/temporal.json",
    "questions/hierarchy.json",
    "gold.json",
    "cache/schema.json",
    "cache/kg.json",
)
OUTSIDE_CLOSURE = (
    "config/bench_old.json",
    "src/toy/__pycache__/x.cpython-312.pyc",
    "runs/report.json",
)


def _harness(root: Path) -> Path:
    for rel in CLOSURE_FILES + OUTSIDE_CLOSURE:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {rel}\n")
    return root


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


def test_freeze_copies_exactly_the_declared_closure_with_hashes(tmp_path: Path) -> None:
    source = _harness(tmp_path / "harness")
    target = freeze.freeze(source, tmp_path / "frozen", PATTERNS)
    closure = json.loads((target / "closure.json").read_text())
    assert sorted(closure["files"]) == sorted(CLOSURE_FILES)
    for rel, digest in closure["files"].items():
        assert (target / rel).read_bytes() == (source / rel).read_bytes()
        assert digest == freeze.sha256_file(source / rel)
    assert target.name == closure["closure_id"]
    assert not (target / OUTSIDE_CLOSURE[0]).exists()


def test_freeze_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    source = _harness(tmp_path / "harness")
    first = freeze.freeze(source, tmp_path / "frozen", PATTERNS)
    assert freeze.freeze(source, tmp_path / "frozen", PATTERNS) == first
    (source / "uv.lock").write_text("changed\n")
    assert freeze.freeze(source, tmp_path / "frozen", PATTERNS) != first


def test_freeze_fails_when_a_declared_input_is_missing(tmp_path: Path) -> None:
    source = _harness(tmp_path / "harness")
    (source / "gold.json").unlink()
    with pytest.raises(freeze.FreezeError, match="gold.json"):
        freeze.freeze(source, tmp_path / "frozen", PATTERNS)


def test_freeze_fails_when_a_source_changes_during_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _harness(tmp_path / "harness")
    real_copy = shutil.copy2

    def copy_then_edit(src: str, dst: str) -> object:
        result = real_copy(src, dst)
        if str(src).endswith("uv.lock"):
            Path(src).write_text("edited mid-copy\n")
        return result

    monkeypatch.setattr(freeze.shutil, "copy2", copy_then_edit)
    with pytest.raises(freeze.FreezeError, match="source changed.*uv.lock"):
        freeze.freeze(source, tmp_path / "frozen", PATTERNS)
    assert not list((tmp_path / "frozen").glob("*/closure.json"))


def test_freeze_fails_when_a_copy_differs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _harness(tmp_path / "harness")
    real_copy = shutil.copy2

    def corrupt(src: str, dst: str) -> object:
        result = real_copy(src, dst)
        if str(src).endswith("pyproject.toml"):
            Path(dst).write_text("corrupt\n")
        return result

    monkeypatch.setattr(freeze.shutil, "copy2", corrupt)
    with pytest.raises(freeze.FreezeError, match="copy differs.*pyproject.toml"):
        freeze.freeze(source, tmp_path / "frozen", PATTERNS)


def test_freeze_records_git_head_and_dirty_files(tmp_path: Path) -> None:
    source = _harness(tmp_path / "harness")
    _git(source, "init", "-q")
    _git(source, "add", "-A")
    _git(source, "commit", "-q", "-m", "base")
    (source / "pyproject.toml").write_text("dirty\n")
    target = freeze.freeze(source, tmp_path / "frozen", PATTERNS)
    closure = json.loads((target / "closure.json").read_text())
    assert len(closure["git_head"]) == 40
    assert any(line.endswith("pyproject.toml") for line in closure["git_dirty"])
