"""R29 OMOP-coupling guard: engine core + generic spine carry no OMOP literals.

US-000 neutralized the four known engine-core leaks. US-001 formalizes the
guard: a single policy constant (denylist + core paths + allowlist) scanned by
``scripts/check_engine_coupling.py``, wired here as a unit-marked test so
``uv run pytest`` fails on a leak. The denylist is case-explicit so US-000's
migration cannot be bypassed by casing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_engine_coupling import (
    R29_POLICY,
    EngineCouplingPolicy,
    find_violations,
    scan,
)

pytestmark = pytest.mark.unit


def _fixture_policy() -> EngineCouplingPolicy:
    """Policy aimed at a synthetic 'core' tree with an allowlisted subdir."""
    return EngineCouplingPolicy(
        denylist=R29_POLICY.denylist,
        core_paths=("core", "core/engine.py"),
        allowlist=("core/policies",),
    )


def test_fixture_leak_is_flagged(tmp_path: Path) -> None:
    """A core file carrying OMOP literals yields a non-empty violation list."""
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "leaky.py").write_text(
        "x = 'OncoTree'  # condition_occurrence\n", encoding="utf-8"
    )
    violations = scan(tmp_path, _fixture_policy())
    literals = {v.literal for v in violations}
    assert "OncoTree" in literals
    assert "condition_occurrence" in literals
    assert all(v.path == "core/leaky.py" for v in violations)


def test_clean_fixture_passes(tmp_path: Path) -> None:
    """A core file free of denylist literals yields no violations."""
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "clean.py").write_text(
        "vocab = 'snomed value set'\n", encoding="utf-8"
    )
    assert scan(tmp_path, _fixture_policy()) == []


def test_allowlisted_file_is_not_flagged(tmp_path: Path) -> None:
    """Literals inside an allowlisted policy module are permitted."""
    (tmp_path / "core" / "policies").mkdir(parents=True)
    (tmp_path / "core" / "policies" / "omop.py").write_text(
        "relationship = 'Maps to'\nstd = 'standard_concept'\n", encoding="utf-8"
    )
    assert scan(tmp_path, _fixture_policy()) == []


def test_missing_core_path_is_skipped(tmp_path: Path) -> None:
    """Configured-but-not-yet-created core paths do not raise (later stories)."""
    assert scan(tmp_path, _fixture_policy()) == []


def test_denylist_is_case_explicit() -> None:
    """Casing variants are enumerated so migration cannot be bypassed."""
    for literal in ("OncoTree", "oncotree", "ONCOTREE", "Maps to"):
        assert literal in R29_POLICY.denylist


def test_engine_core_and_spine_are_r29_clean() -> None:
    """The live tree must be green: no OMOP/OncoTree literal in any core path."""
    violations = find_violations()
    assert violations == [], f"R29 coupling leaks: {violations}"
