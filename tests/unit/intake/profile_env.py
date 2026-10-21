"""Shared setup for the profiling tool tests: landed fixtures, recording provider, ledger."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.binding.spec_sections import RelationRef
from sema.ingest.duckdb_staging import Staging
from tests.unit.binding.fixtures.concept_library import build_library, query_fn


FIXTURES = Path(__file__).parents[2] / "fixtures" / "intake"
A2_WORKBOOK = FIXTURES / "tabular" / "codes_a2.xlsx"
A3_DB = FIXTURES / "semsql" / "ontology.db"

CONCEPTS = RelationRef(provider="staging", name="land_a2.codes_a2__Concepts")
VALUES = RelationRef(provider="staging", name="land_a2.codes_a2__Values")
HIERARCHY = RelationRef(provider="staging", name="land_a2.codes_a2__Hierarchy")
A3_EDGES = RelationRef(provider="staging", name="land_a3.semsql_edges")
CYCLE = RelationRef(provider="staging", name="land_t.cycle")
WIDE = RelationRef(provider="staging", name="land_t.wide")
ODD = RelationRef(provider="staging", name="land_t.odd rel")
ODD_COLUMN = 'col "q" name'
SYN_CONCEPT = RelationRef(provider="warehouse", name="lib.concept")

A2_LABELS = {"Red", "Green", "Blue", "Teal (old)", "Small", "Medium", "Large", "Round", "Square"}
A2_CONCEPT_TEXT = {
    ("Attribute", "Any attribute of an item"),
    ("Color", "Colour of an item"),
    ("Dimension", "A measurable extent"),
    ("Size", "Size of an item"),
    ("Shape", "Outline of an item"),
}
WIDE_DISTINCT = 150
SAMPLE_CAP = 20
A3_DEPTH = 2


@dataclass
class Env:
    staging: Staging
    warehouse: duckdb.DuckDBPyConnection
    ledger_dir: Path
    issued: list[str] = field(default_factory=list)

    def provider(self) -> Any:
        from sema.intake.relations import RelationProvider

        run = query_fn(self.warehouse)

        def recorded(sql: str) -> list[tuple[Any, ...]]:
            self.issued.append(sql)
            return run(sql)

        return RelationProvider(warehouse=recorded, staging=self.staging)

    def sql_lines(self) -> list[dict[str, Any]]:
        path = self.ledger_dir / "sql.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def statements(self) -> list[str]:
        return [line["statement"] for line in self.sql_lines()]


@dataclass(frozen=True)
class BoundStructure:
    schema_relations: frozenset[str]


class FakeVocabulary:
    def __init__(self, known: set[tuple[str, str]]) -> None:
        self.known = known
        self.calls: list[tuple[str, str]] = []

    def concept_by_code(self, vocabulary: str, code: str) -> object | None:
        self.calls.append((vocabulary, code))
        return object() if (vocabulary, code) in self.known else None


def _extra_relations(staging: Staging) -> None:
    staging.execute('CREATE SCHEMA IF NOT EXISTS "land_t"')
    staging.execute('CREATE TABLE "land_t"."cycle" (child VARCHAR, parent VARCHAR)')
    staging.execute("""INSERT INTO "land_t"."cycle" VALUES ('X', 'Y'), ('Y', 'X')""")
    staging.execute(
        'CREATE TABLE "land_t"."wide" AS SELECT'
        " 'v' || LPAD(CAST(i AS VARCHAR), 3, '0') AS v, i % 2 AS flag"
        f" FROM range({WIDE_DISTINCT}) t(i)"
    )
    quoted = ODD_COLUMN.replace('"', '""')
    staging.execute(f'CREATE TABLE "land_t"."odd rel" ("{quoted}" VARCHAR)')
    staging.execute("""INSERT INTO "land_t"."odd rel" VALUES ('a'), ('bb'), (NULL)""")
    staging.execute(f'COMMENT ON COLUMN "land_t"."odd rel"."{quoted}" IS \'a note\'')


def open_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    from sema.intake.loaders.semsql import land_semsql
    from sema.intake.loaders.tabular import land_tabular

    staging = Staging(str(tmp_path / "staging.duckdb"), schemas=())
    land_tabular(A2_WORKBOOK, staging, schema="land_a2")
    land_semsql(A3_DB, staging, schema="land_a3")
    _extra_relations(staging)
    warehouse = build_library()
    environment = Env(staging, warehouse, tmp_path / "ledger" / "run-1")
    original = staging.execute

    def recorded(sql: str, params: Any = None) -> Any:
        environment.issued.append(sql)
        return original(sql, params)

    monkeypatch.setattr(staging, "execute", recorded)
    try:
        yield environment
    finally:
        warehouse.close()
        staging.close()


def make_policy(tier: str = "metadata_only", **kwargs: Any) -> Any:
    from sema.intake.profile import EvidencePolicy

    return EvidencePolicy(tier=tier, **kwargs)


def make_tools(env: Env, policy: Any = None, stage: str = "structure", **kwargs: Any) -> Any:
    from sema.intake.ledger import Ledger
    from sema.intake.profile import ProfileTools

    ledger = Ledger(env.ledger_dir.parent).run(env.ledger_dir.name)
    return ProfileTools(
        env.provider(), policy or make_policy(), ledger, stage=stage, **kwargs
    )


def column_of(description: Any, name: str) -> Any:
    return next(c for c in description.columns if c.name == name)


def reads(statement: str, relation_sql: str) -> bool:
    return re.search(r"\bFROM\s+" + re.escape(relation_sql), statement, re.IGNORECASE) is not None


def value_bearing(env: Env) -> list[dict[str, Any]]:
    return [line for line in env.sql_lines() if line["value_bearing"]]
