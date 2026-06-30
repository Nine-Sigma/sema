"""R29 baseline: engine core must be free of OMOP/OncoTree literals.

US-000 migration check. A case-explicit denylist is scanned against the
engine-core Python sources; after migration the violation list must be EMPTY.
US-001 formalizes the full guard (allowlist, core-paths, fixture-positive
test, suite wiring); this test freezes the post-migration clean state.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Case-explicit so a casing variant cannot bypass the migration check.
DENYLIST: tuple[str, ...] = (
    "OncoTree",
    "oncotree",
    "ONCOTREE",
    "cBioPortal",
    "cbioportal",
    "cBio",
    "cbio",
    "Maps to",
    "standard_concept",
    "condition_occurrence",
    "concept_id",
)

ENGINE_CORE = Path(__file__).resolve().parents[2] / "src" / "sema" / "engine"


def _scan_for_literals(
    root: Path, denylist: tuple[str, ...]
) -> list[tuple[str, int, str]]:
    """Return (relative_path, line_number, literal) for every denylist hit."""
    violations: list[tuple[str, int, str]] = []
    for py_file in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(
            py_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for literal in denylist:
                if literal in line:
                    violations.append((py_file.name, lineno, literal))
    return violations


def test_scanner_flags_a_seeded_literal(tmp_path: Path) -> None:
    """The scanner itself must catch a planted OMOP/OncoTree literal."""
    leaky = tmp_path / "leaky.py"
    leaky.write_text("vocab = 'oncotree'\n", encoding="utf-8")
    assert _scan_for_literals(tmp_path, DENYLIST) == [("leaky.py", 1, "oncotree")]


def test_engine_core_has_no_omop_coupling() -> None:
    """No OMOP/OncoTree literal may live in src/sema/engine/ (R29)."""
    violations = _scan_for_literals(ENGINE_CORE, DENYLIST)
    assert violations == [], f"engine-core OMOP/OncoTree leaks: {violations}"
