"""Generic benchmark code and its tests name no source: every source value is profile data.

The scan covers every line (code, strings, comments, docstrings) of the package's Python
files, its tests and every neutral fixture file. The profile directory and the opt-in
source-profile test hold source values by design; this file holds the denylist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "showcase" / "ontology_query_benchmark"
TESTS = Path(__file__).parent
FIXTURES = REPO / "tests" / "fixtures" / "ontology_query_benchmark"
FIXTURE_SUFFIXES = (".py", ".json", ".yaml")
EXEMPT_DIRS = (PACKAGE / "profiles",)
EXEMPT_FILES = (Path(__file__), TESTS / "test_source_profile.py")
DENYLIST = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?<![a-z])partner", "prtcc", "partner-workspace", "timbr-partner", "timbr_poc", "standard_library",
        "omop", "tumor", "r3_platinum", "cancer_registry", "demographics", "cpd_v2",
        "v3_20261021", "w4s", "feedface", "deadbeef", "000000000000000",
        "databricks-claude", "databricks-bge",
    )
) + (re.compile(r"\bQ14\b"),)


def scanned_files() -> list[Path]:
    fixtures = [p for p in FIXTURES.rglob("*") if p.suffix in FIXTURE_SUFFIXES]
    files = sorted({*PACKAGE.rglob("*.py"), *TESTS.rglob("*.py"), *fixtures})
    return [
        path for path in files
        if path not in EXEMPT_FILES and not any(path.is_relative_to(d) for d in EXEMPT_DIRS)
    ]


def literal_hits(paths: list[Path]) -> list[str]:
    hits = []
    for path in paths:
        shown = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hits += [
                f"{shown}:{lineno}: {found.group(0)}"
                for pattern in DENYLIST
                if (found := pattern.search(line))
            ]
    return hits


def test_the_scan_covers_package_and_tests() -> None:
    names = {path.name for path in scanned_files()}
    assert {"runner.py", "sema_context_utils.py", "test_runner.py", "profile.yaml",
            "benchmark_workflow.py"} <= names  # fmt: skip


def test_the_guard_reports_a_planted_literal_with_its_line(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text('HOST = "generic"\nWAREHOUSE = "feedface00000000"\n')
    hits = literal_hits([planted])
    assert len(hits) == 1 and hits[0].endswith("planted.py:2: feedface")


def test_generic_code_and_tests_name_no_source() -> None:
    hits = literal_hits(scanned_files())
    assert not hits, f"{len(hits)} source-specific literals:\n" + "\n".join(hits)
