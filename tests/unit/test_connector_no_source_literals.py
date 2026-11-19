"""Generality gate: the source files the DuckDB connector unit and the coded-column unit
change name no source.

The literals are read at test time from every source fixture and profile, so a source
added later is covered without editing this test.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing
from pathlib import Path

import duckdb
import pytest
import yaml
from sqlglot import exp

from sema.engine.embeddings import INDEX_CONFIGS
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
        "cli_eval.py",
        "eval/discovery_scorecard.py",
        "eval/discovery_scorecard_utils.py",
        "intake/run/scorecard.py",
        "connectors/duckdb.py",
        "connectors/duckdb_utils.py",
        "consumers/nl2sql/prompting_utils.py",
        "engine/coded_column_utils.py",
        "engine/join_detector.py",
        "engine/join_detector_utils.py",
        "engine/warehouse_lookup.py",
        "graph/concept_value_set.py",
        "graph/loader.py",
        "models/assertions.py",
        "models/config.py",
        "models/extraction.py",
        "models/physical_key.py",
        "pipeline/build.py",
        "pipeline/coded_columns.py",
        "pipeline/orchestrate.py",
        "pipeline/dedup_utils.py",
        "pipeline/retrieval.py",
        "pipeline/retrieval_utils.py",
        "pipeline/term_expansion_utils.py",
        "pipeline/context.py",
        "pipeline/context_utils.py",
        "models/context.py",
        "consumers/nl2sql/consumer.py",
        "consumers/nl2sql/prompting.py",
        "connectors/base.py",
        "consumers/base.py",
        "engine/identity_utils.py",
        "connectors/databricks_utils.py",
        "graph/queries.py",
        "pipeline/identity.py",
        "graph/identity_read_forms.py",
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
    ("discovery_scorecard.py", "hierarchy"): (
        "fact kind id fixed in the unit 6 contract; intake fixture specs use it as a section word"
    ),
    ("discovery_scorecard.py", "many_to_one"): (
        "join cardinality word fixed in the unit 6 contract (ruling I); the scorecard fixture's oracle uses it"
    ),
    **dict.fromkeys(
        [("discovery_scorecard.py", "kind"), ("discovery_scorecard_utils.py", "kind"),
         ("scorecard.py", "kind")],
        "the oracle file format's field name (unit 6 contract); unit 5's code fixture uses it as a word",
    ),
    ("config.py", "databricks-bge-large-en"): (
        "d395b094 embedding model default; predates unit 3, see unit-3 STATE.md Waiting on Dean"
    ),
    ("config.py", "5.4"): (
        "aed40d37 cdm_version default; predates unit 3, see unit-3 STATE.md Waiting on Dean"
    ),
    **dict.fromkeys(
        [("context_utils.py", "source"), ("retrieval.py", "index"), ("retrieval.py", "source"),
         ("retrieval.py", "values"), ("retrieval_utils.py", "source"),
         ("term_expansion_utils.py", "source")],
        "a generic candidate key a profile also uses as a word; predates unit 6 (f792d5c)",
    ),
    ("loader.py", "column name"): "Column node keys in Cypher; predates unit 5",
    ("loader.py", "predicate"): "Assertion field name; predates unit 5",
    ("loader.py", "source"): "Assertion field name; predates unit 5",
    ("databricks_utils.py", "values"): (
        "HAS_TOP_VALUES payload field name, a generic word a profile also uses; predates unit 7, "
        "baselined 2026-11-19 when N3 added the file"
    ),
    ("queries.py", "column name"): (
        "Column node keys in Cypher; predates unit 7, scanned from 2026-11-19 (N7)"
    ),
    ("queries.py", "maps to"): "a comment's words; predates unit 7, scanned from 2026-11-19 (N7)",
}
# Source fixture modules: their public constants name the connector fixture's source.
CONNECTOR_FIXTURES = tuple(sorted((REPO_ROOT / "tests" / "support").glob("connector_fixture*.py")))


def literal_files(roots: Iterable[Path] = LITERAL_ROOTS) -> list[Path]:
    """Source files only: bytecode that importing a fixture builder leaves is skipped."""
    return sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts
    )


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


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            yield from _strings(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _strings(item)


def module_literals(path: Path) -> set[str]:
    """Every string a fixture module's public constants hold, dataclass fields included."""
    module = importlib.import_module(f"tests.support.{path.stem}")
    return {
        text
        for name, value in vars(module).items()
        if name.isupper() and not name.startswith("_")
        for text in _strings(value)
    }


def warehouse_vocabulary() -> set[str]:
    """Product words a profile or fixture also uses: connector module names, SQL type names
    and the graph's vector index names, which a stand-in graph answers by name."""
    connectors = {path.stem for path in (SRC / "connectors").glob("*.py")}
    indexes = {name for name, _ in INDEX_CONFIGS}
    return connectors | indexes | {kind.value.casefold() for kind in exp.DataType.Type}


def source_literals(
    roots: Iterable[Path] = LITERAL_ROOTS, modules: Iterable[Path] = ()
) -> dict[tuple[str, ...], str]:
    excluded = excluded_names() | warehouse_vocabulary()
    texts = [text for path in literal_files(roots) for text in file_literals(path)]
    texts += [text for path in modules for text in module_literals(path)]
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


def files_without_reader(roots: Iterable[Path] = LITERAL_ROOTS) -> list[Path]:
    known = TABULAR_SUFFIXES + STRUCTURED_SUFFIXES + DATABASE_SUFFIXES + UNREAD_SUFFIXES
    return [path for path in literal_files(roots) if path.suffix.lower() not in known]


def test_every_literal_file_is_read_or_its_type_is_named_unread() -> None:
    assert files_without_reader() == [], "no reader for these literal files"
    for path in literal_files():
        unread = path.suffix.lower() in UNREAD_SUFFIXES
        assert unread or file_literals(path), f"{path} yielded no literal"


def test_bytecode_caches_are_skipped_and_an_unknown_type_still_fails(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "build_fixture.cpython-312.pyc").write_bytes(b"\xa7\r\r\n\xff\xfe\x00")
    (tmp_path / "p.yaml").write_text("relation: gadget_log\n")
    assert set(source_literals([tmp_path]).values()) == {"gadget_log"}
    (tmp_path / "notes.weird").write_text("gadget_log\n")
    assert files_without_reader([tmp_path]) == [tmp_path / "notes.weird"]


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


def test_a_planted_connector_fixture_name_is_reported(tmp_path: Path) -> None:
    literals = source_literals(modules=CONNECTOR_FIXTURES)
    fixture_only = set(literals.values()) - set(source_literals().values())
    single = max((text for text in fixture_only if " " not in text), key=len)
    multi = min((text for text in fixture_only if " " in text), key=len)
    planted = tmp_path / "planted.py"
    planted.write_text(f"KIND = 'generic'\nNAME = {single!r}\n# reads {multi}\n")
    hits = {(hit.lineno, hit.literal) for hit in scan_hits([planted], literals)}
    assert {(2, single), (3, multi)} <= hits
    assert all(lineno != 1 for lineno, _ in hits)


def test_files_the_unit_changes_hold_only_the_baselined_literals() -> None:
    literals = source_literals(modules=CONNECTOR_FIXTURES)
    new, stale = baseline_drift(scan_hits(UNIT_FILES, literals), PREDATING_HITS)
    assert new == [], "source literals in generic code:\n" + "\n".join(
        f"  {hit.path}:{hit.lineno}: {hit.literal!r}" for hit in new
    )
    assert not stale, f"no longer hit, remove from PREDATING_HITS: {sorted(stale)}"
