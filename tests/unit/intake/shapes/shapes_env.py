"""Shared setup for the shape template tests: landed sources, lazy profiles, helpers.

``sema.intake.shapes`` is imported lazily, so a missing module fails each test on its own
instead of failing collection.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import openpyxl

from sema.intake.landing import LandedRelation, LandResult, Provenance, utc_now
from sema.intake.ledger import Ledger
from sema.intake.loaders.base import DEFAULT_POLICY_CLASS
from sema.intake.profile import EvidencePolicy, ProfileTools
from sema.intake.relations import RelationProvider

FIXTURES = Path(__file__).parents[3] / "fixtures" / "intake"
A2_WORKBOOK = FIXTURES / "tabular" / "codes_a2.xlsx"
A3_DB = FIXTURES / "semsql" / "ontology.db"
ACRIS = FIXTURES / "acris"
OMOP_CSV = FIXTURES / "omop" / "OMOP_CDMv5.4_Field_Level.csv"

TEMPLATE_IDS = (
    "concept_list",
    "concept_with_values_denormalized",
    "concept_and_value_set",
    "edge_list_hierarchy",
    "level_columns",
    "path_codes",
    "semsql",
    "athena_bundle",
    "entity_model",
    "ignore",
)
MULTI_RELATION_IDS = frozenset(
    {"concept_and_value_set", "edge_list_hierarchy", "entity_model", "semsql", "athena_bundle"}
)
IGNORE = "ignore"

Run = Callable[[str], list[tuple[Any, ...]]]


def shapes(name: str | None = None) -> ModuleType:
    return importlib.import_module("sema.intake.shapes" + (f".{name}" if name else ""))


def base() -> ModuleType:
    return shapes("base")


def make_tools(
    provider: RelationProvider, ledger_dir: Path, land: LandResult | None = None
) -> ProfileTools:
    ledger = Ledger(ledger_dir.parent).run(ledger_dir.name)
    policy = EvidencePolicy(tier="metadata_only")
    return ProfileTools(provider, policy, ledger, stage="structure", land=land)


@dataclass
class Source:
    land: LandResult
    tools: ProfileTools
    provider: str
    ledger_dir: Path
    cached: tuple[Any, ...] | None = field(default=None, repr=False)

    def profiles(self) -> tuple[Any, ...]:
        if self.cached is None:
            built = base().profile_relations(self.land, self.tools, provider=self.provider)
            self.cached = tuple(built)
        return self.cached

    def ref(self, name: str) -> str:
        return f"{self.land.relation(name).schema}.{name}"

    def profile(self, name: str) -> Any:
        return next(p for p in self.profiles() if p.landed.name == name)

    def group(self, *names: str) -> list[Any]:
        return [self.profile(name) for name in names]

    def sheet(self, suffix: str) -> list[Any]:
        return [p for p in self.profiles() if p.landed.name.endswith(f"__{suffix}")]

    def stem_groups(self) -> list[list[Any]]:
        groups: dict[str, list[Any]] = {}
        for profile in self.profiles():
            groups.setdefault(profile.stem, []).append(profile)
        return list(groups.values())

    def sql_lines(self) -> list[dict[str, Any]]:
        path = self.ledger_dir / "sql.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def value_bearing(self) -> list[dict[str, Any]]:
        return [line for line in self.sql_lines() if line["value_bearing"]]


Memory = Callable[..., Source]


def staging_source(staging: Any, land: LandResult, ledger_dir: Path) -> Source:
    provider = RelationProvider(warehouse=None, staging=staging)
    return Source(land, make_tools(provider, ledger_dir), "staging", ledger_dir)


def warehouse_source(con: duckdb.DuckDBPyConnection, schema: str, ledger_dir: Path) -> Source:
    from tests.unit.binding.fixtures.concept_library import query_fn

    land = land_of_schema(lambda sql: con.execute(sql).fetchall(), schema)
    provider = RelationProvider(warehouse=query_fn(con), staging=None)
    return Source(land, make_tools(provider, ledger_dir, land), "warehouse", ledger_dir)


def land_of_schema(run: Run, schema: str) -> LandResult:
    """A LandResult for tables created by SQL rather than by a loader."""
    tables = run(
        "SELECT table_name FROM information_schema.tables"
        f" WHERE table_schema = '{schema}' ORDER BY table_name"
    )
    relations = tuple(_landed(run, schema, str(row[0])) for row in tables)
    return LandResult(
        medium="tabular",
        source_identity=(("schema", schema),),
        relations=relations,
        policy_class=DEFAULT_POLICY_CLASS,
    )


def _landed(run: Run, schema: str, table: str) -> LandedRelation:
    columns = run(
        "SELECT column_name, data_type FROM information_schema.columns"
        f" WHERE table_schema = '{schema}' AND table_name = '{table}' ORDER BY ordinal_position"
    )
    count = run(f'SELECT COUNT(*) FROM "{schema}"."{table}"')[0][0]
    provenance = Provenance(
        source_uri=f"memory:{schema}",
        loader="test",
        loader_version="1",
        landed_at=utc_now(),
        policy_class=DEFAULT_POLICY_CLASS,
        locator_kind="relation",
        locator=table,
    )
    return LandedRelation(
        schema=schema,
        name=table,
        columns=tuple((str(name), str(kind)) for name, kind in columns),
        row_count=int(count),
        provenance=provenance,
    )


def write_workbook(path: Path, sheets: dict[str, list[list[object]]]) -> Path:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        worksheet = workbook.create_sheet(title)
        for row in rows:
            worksheet.append(row)
    workbook.save(path)
    return path


def candidates(template_id: str, group: Sequence[Any], source: Source) -> list[Any]:
    result: list[Any] = shapes(template_id).candidates(list(group), source.tools)
    return result


def verify(template_id: str, binding: Any, group: Sequence[Any], source: Source) -> Any:
    return shapes(template_id).verify(binding, list(group), source.tools)


def over(bindings: Iterable[Any], relation: str) -> list[Any]:
    return [b for b in bindings if relation in b.relations]


def binding(template: str, relations: Sequence[str], **slots: str) -> Any:
    return base().SlotBinding(
        template=template,
        section=None,
        slots=dict(slots),
        evidence={},
        relations=tuple(relations),
    )


def every_template_candidates(group: Sequence[Any], source: Source) -> list[Any]:
    found: list[Any] = []
    for template_id in TEMPLATE_IDS:
        if template_id != IGNORE:
            found += candidates(template_id, group, source)
    return found
