"""Generality gate: the source files the DuckDB connector unit changes name no source.

The literals are read at test time from every source fixture and profile, so a source
added later is covered without editing this test.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from pathlib import Path

import duckdb
import pytest
import yaml
from sqlglot import exp

from tests.unit.source_literals_env import (
    REPO_ROOT,
    TABULAR_SUFFIXES,
    Hit,
    excluded_names,
    literal_index,
    literals_of,
    matches,
    source_texts,
    structured_literals,
)

pytestmark = pytest.mark.unit

SRC = REPO_ROOT / "src" / "sema"
UNIT_FILES = tuple(
    SRC / relative
    for relative in (
        "connectors/duckdb.py",
        "connectors/duckdb_utils.py",
        "models/config.py",
        "models/extraction.py",
        "models/physical_key.py",
        "pipeline/build.py",
        "pipeline/orchestrate.py",
    )
)
LITERAL_ROOTS = (
    REPO_ROOT / "tests" / "fixtures" / "intake",
    REPO_ROOT / "showcase" / "open_data_load" / "profiles",
    REPO_ROOT / "showcase" / "ontology_query_benchmark" / "profiles",
)
STRUCTURED_SUFFIXES = (".yaml", ".yml", ".json")
DATABASE_SUFFIXES = (".duckdb", ".db")
# Free text: read as a literal list, a query's aliases and a note's words flag generic code.
UNREAD_SUFFIXES = (".md", ".py", ".sql", ".ttl")
SQLITE_CATALOG = (
    "SELECT m.name, p.name FROM sqlite_master AS m, pragma_table_info(m.name) AS p "
    "WHERE m.type IN ('table', 'view')"
)
DUCKDB_CATALOG = "SELECT table_name, column_name FROM information_schema.columns"
# Hits that predate this unit, kept exact so a new hit fails and a moved default shrinks it.
PREDATING_HITS = {
    ("config.py", "databricks-bge-large-en"): (
        "d395b094 embedding model default; predates unit 3, see unit-3 STATE.md Waiting on Dean"
    ),
    ("config.py", "5.4"): (
        "aed40d37 cdm_version default; predates unit 3, see unit-3 STATE.md Waiting on Dean"
    ),
}


def literal_files(roots: Iterable[Path] = LITERAL_ROOTS) -> list[Path]:
    return sorted(path for root in roots for path in root.rglob("*") if path.is_file())


def _catalog_names(path: Path) -> set[str]:
    if path.suffix.lower() == ".duckdb":
        with duckdb.connect(str(path), read_only=True) as con:
            rows = con.execute(DUCKDB_CATALOG).fetchall()
    else:
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as con:
            rows = con.execute(SQLITE_CATALOG).fetchall()
    return {name for row in rows for name in row}


def file_literals(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    text = "" if suffix in TABULAR_SUFFIXES + DATABASE_SUFFIXES else path.read_text("utf-8")
    if suffix in TABULAR_SUFFIXES:
        return literals_of(path)
    if suffix == ".json":
        return structured_literals(json.loads(text))
    if suffix in STRUCTURED_SUFFIXES:
        return structured_literals(yaml.safe_load(text))
    if suffix in DATABASE_SUFFIXES:
        return _catalog_names(path)
    return set()


def warehouse_vocabulary() -> set[str]:
    """Product words a profile also uses: connector module names and SQL type names."""
    connectors = {path.stem for path in (SRC / "connectors").glob("*.py")}
    return connectors | {kind.value.casefold() for kind in exp.DataType.Type}


def source_literals(roots: Iterable[Path] = LITERAL_ROOTS) -> dict[tuple[str, ...], str]:
    excluded = excluded_names() | warehouse_vocabulary()
    texts = (text for path in literal_files(roots) for text in file_literals(path))
    return literal_index(texts, excluded)


def scan_hits(paths: Iterable[Path], literals: Mapping[tuple[str, ...], str]) -> list[Hit]:
    return sorted(
        {
            Hit(path.name, text.lineno, literal)
            for path in paths
            for text in source_texts(path.read_text(encoding="utf-8"))
            for literal in matches(text, literals)
        }
    )


def baseline_drift(
    hits: list[Hit], baseline: Mapping[tuple[str, str], str]
) -> tuple[list[Hit], set[tuple[str, str]]]:
    """Hits outside the baseline, and baseline entries that no longer hit."""
    found = {(hit.path, hit.literal) for hit in hits}
    new = [hit for hit in hits if (hit.path, hit.literal) not in baseline]
    return new, set(baseline) - found


def test_the_scan_covers_every_file_the_unit_changes() -> None:
    missing = [str(path) for path in UNIT_FILES if not path.is_file()]
    assert not missing, f"renamed or removed, update the scanned set: {missing}"


def test_every_literal_file_is_read_or_its_type_is_named_unread() -> None:
    read_suffixes = TABULAR_SUFFIXES + STRUCTURED_SUFFIXES + DATABASE_SUFFIXES
    for path in literal_files():
        suffix = path.suffix.lower()
        assert suffix in read_suffixes + UNREAD_SUFFIXES, f"no reader for {path}"
        assert suffix in UNREAD_SUFFIXES or file_literals(path), f"{path} yielded no literal"


def test_database_files_yield_their_table_and_column_names(tmp_path: Path) -> None:
    with duckdb.connect(str(tmp_path / "a.duckdb")) as con:
        con.execute("CREATE TABLE widget_ledger (ledger_amount INTEGER)")
    with closing(sqlite3.connect(tmp_path / "b.db")) as con:
        con.execute("CREATE TABLE gadget_log (log_note TEXT)")
    names = set(source_literals([tmp_path]).values())
    assert {"widget_ledger", "ledger_amount", "gadget_log", "log_note"} <= names


def test_product_warehouse_vocabulary_is_not_a_literal(tmp_path: Path) -> None:
    (tmp_path / "p.yaml").write_text("source: DuckDB\ntype: BIGINT\nrelation: gadget_log\n")
    assert set(source_literals([tmp_path]).values()) == {"gadget_log"}


def test_a_planted_source_literal_is_reported_with_its_line(tmp_path: Path) -> None:
    literals = source_literals()
    single = max((text for key, text in literals.items() if len(key) == 1), key=len)
    multi = min((text for key, text in literals.items() if len(key) > 1), key=len)
    planted = tmp_path / "planted.py"
    planted.write_text(f"KIND = 'generic'\nNAME = {single.upper()!r}\n# reads {multi}\n")
    hits = {(hit.lineno, hit.literal) for hit in scan_hits([planted], literals)}
    assert {(2, single), (3, multi)} <= hits
    assert all(lineno != 1 for lineno, _ in hits)


def test_the_baseline_reports_a_new_hit_and_a_stale_entry(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text("OLD = 'known_ledger'\nNEW = 'fresh_ledger'\n")
    literals = literal_index(["known_ledger", "fresh_ledger"], set())
    baseline = {("planted.py", "known_ledger"): "kept", ("planted.py", "gone_ledger"): "moved"}
    new, stale = baseline_drift(scan_hits([planted], literals), baseline)
    assert new == [Hit("planted.py", 2, "fresh_ledger")]
    assert stale == {("planted.py", "gone_ledger")}


def test_files_the_unit_changes_hold_only_the_baselined_literals() -> None:
    new, stale = baseline_drift(scan_hits(UNIT_FILES, source_literals()), PREDATING_HITS)
    assert new == [], "source literals in generic code:\n" + "\n".join(
        f"  {hit.path}:{hit.lineno}: {hit.literal!r}" for hit in new
    )
    assert not stale, f"no longer hit, remove from PREDATING_HITS: {sorted(stale)}"
