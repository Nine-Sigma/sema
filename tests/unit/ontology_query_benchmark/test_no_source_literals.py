"""Generic benchmark code and its tests name no source: every source value is profile data.

The scan covers every line (code, strings, comments, docstrings) of every package file of
any type, the package tests and every neutral fixture file. The profile directory and the opt-in
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
EXEMPT_FILES = (Path(__file__), TESTS / "test_source_profile.py")
DENYLIST = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?<![a-z])partner", "prtcc", "partner-workspace", "timbr-partner", "timbr_poc", "standard_library",
        "omop", "tumor", "r3_platinum", "cancer_registry", "demographics", "cpd_v2",
        "v3_20261021", "w4s", "feedface", "deadbeef", "000000000000000",
        "databricks-claude", "databricks-bge", "naaccr", "vb_vocabulary", r"\bCPD\b",
        "partner_wh", "concept_athena_id", "partner_dev", r"UNV\d", r"\bpartnerplatform\b",
        r"\bathena\b", r"\bpb_", "_r3_", "partner-harness", "cafebabe", "cancer_plus", "sonnet-5",
    )
) + (re.compile(r"\bQ14\b"),)


def scanned_files(package: Path = PACKAGE) -> list[Path]:
    fixtures = [p for p in FIXTURES.rglob("*") if p.suffix in FIXTURE_SUFFIXES]
    sources = [p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files = sorted({*sources, *TESTS.rglob("*.py"), *fixtures})
    return [
        path for path in files
        if path not in EXEMPT_FILES and not path.is_relative_to(package / "profiles")
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


@pytest.mark.parametrize("literal", [
    "naaccr_sequence_number_UNV00001374", "vb_vocabulary_test_2026_09_25_candidate_ancestor",
    "adapts the CPD product spec", "partner_wh", "concept_athena_id",
    "partner_dev.x.y", "UNV00001374", "a partnerplatform export", "the Athena vocabulary",
    "pb_person", "person_r3_copy", "partner-harness", "cafebabe", "cancer_plus",
    "claude-sonnet-5",
])
def test_the_guard_reports_each_known_source_identifier(tmp_path: Path, literal: str) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(f"VALUE = {literal!r}\n")
    assert literal_hits([planted])


def test_the_guard_scans_every_package_file_type_outside_the_profiles(tmp_path: Path) -> None:
    package = tmp_path / "package"
    for relative in ("notes.sql", "profiles/source.yaml", "__pycache__/m.cpython-312.pyc"):
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
    (package / "notes.sql").write_text("SELECT 1\nFROM cancer_registry.s.t\n")
    (package / "profiles/source.yaml").write_text("relation: cancer_registry.s.t\n")
    (package / "__pycache__/m.cpython-312.pyc").write_bytes(b"\x00\xffcancer_registry")
    hits = [h for h in literal_hits(scanned_files(package)) if h.startswith(str(package))]
    assert hits == [f"{package / 'notes.sql'}:2: cancer_registry"]


def test_generic_code_and_tests_name_no_source() -> None:
    hits = literal_hits(scanned_files())
    assert not hits, f"{len(hits)} source-specific literals:\n" + "\n".join(hits)
