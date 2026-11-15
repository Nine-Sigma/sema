"""A7: the OMOP CDM field-level CSV imported as an entity model, against slice 0."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_models import ManifestEntity, ParsedManifest
from sema.targets.adapters.manifest_parser import parse_manifest_raw
from tests.unit.binding.fixtures.concept_library_v2 import captured_log

pytestmark = pytest.mark.unit

REPO = Path(__file__).parents[3]
OMOP = REPO / "tests" / "fixtures" / "intake" / "omop"
FIELD_CSV = OMOP / "OMOP_CDMv5.4_Field_Level.csv"
SPEC = OMOP / "cdm_field_level.spec.yaml"
SLICE0 = REPO / "showcase" / "cbioportal_to_omop" / "manifests" / "omop_condition_slice0.yaml"
EXPECTED_DELTAS = {
    ("omop.condition_occurrence", "condition_start_date", "nullable"),
    ("omop.condition_occurrence", "condition_start_date", "required"),
    ("omop.condition_occurrence", "condition_concept_id", "synonyms"),
    ("omop.condition_occurrence", "condition_concept_id", "vocabulary_binding"),
}


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory) -> tuple[ParsedManifest, list[str]]:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec
    from sema.intake.land.loaders.tabular import land_tabular
    from sema.intake.land.relations import RelationProvider

    path = tmp_path_factory.mktemp("a7") / "staging.duckdb"
    with Staging(str(path), schemas=()) as staging, captured_log() as messages:
        land_tabular(FIELD_CSV, staging, schema="land_cdm")
        result = import_concept_source(
            load_concept_source_spec(SPEC),
            RelationProvider(warehouse=None, staging=staging),
            dataset_map={},
        )
    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))
    return parsed, messages


def _csv_rows() -> list[dict[str, str]]:
    with FIELD_CSV.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _by_table(rows: list[dict[str, str]], predicate: Any) -> dict[str, set[Any]]:
    found: dict[str, set[Any]] = {}
    for row in rows:
        table = f"omop.{row['cdmTableName'].lower()}"
        found.setdefault(table, set())
        value = predicate(row)
        if value is not None:
            found[table].add(value)
    return found


def _required(row: dict[str, str]) -> str | None:
    return row["cdmFieldName"].lower() if row["isRequired"] == "Yes" else None


def _foreign_key(row: dict[str, str]) -> tuple[str, str, str] | None:
    if not row["fkTableName"]:
        return None
    return (
        row["cdmFieldName"].lower(),
        f"omop.{row['fkTableName'].lower()}",
        row["fkFieldName"].lower(),
    )


def _entity(parsed: ParsedManifest, name: str) -> ManifestEntity:
    return next(e for e in parsed.entities if e.qualified_name == name)


def _deltas(ours: ManifestEntity, theirs: ManifestEntity, scope: set[str]) -> set[tuple[str, str, str]]:
    deltas: set[tuple[str, str, str]] = set()
    props = {p.name: p for p in ours.properties}
    for expected in theirs.properties:
        mine = props[expected.name]
        for aspect in ("type", "nullable", "synonyms", "vocabulary_binding"):
            if getattr(mine, aspect) != getattr(expected, aspect):
                deltas.add((theirs.qualified_name, expected.name, aspect))
    assert ours.obligation is not None and theirs.obligation is not None
    required_ours = set(ours.obligation.required_fields) & scope
    for name in required_ours ^ set(theirs.obligation.required_fields):
        deltas.add((theirs.qualified_name, name, "required"))
    if ours.obligation.primary_key != theirs.obligation.primary_key:
        deltas.add((theirs.qualified_name, "*", "primary_key"))
    return deltas


def _scoped_keys(entity: ManifestEntity, entities: set[str], scope: set[str]) -> Any:
    assert entity.obligation is not None
    fks = {
        (fk.referenced_entity, tuple(map(tuple, fk.join_keys)))
        for fk in entity.obligation.foreign_keys
        if fk.referenced_entity in entities
    }
    domains = {
        (d.property_name, d.domain_id)
        for d in entity.obligation.domain_constraints
        if d.property_name in scope
    }
    return fks, domains


def test_person_and_condition_occurrence_match_slice0_with_listed_deltas(
    imported: tuple[ParsedManifest, list[str]],
) -> None:
    parsed, _ = imported
    slice0 = parse_manifest_raw(yaml.safe_load(SLICE0.read_text()))
    names = {e.qualified_name for e in slice0.entities}

    deltas: set[tuple[str, str, str]] = set()
    for theirs in slice0.entities:
        ours = _entity(parsed, theirs.qualified_name)
        scope = {p.name for p in theirs.properties}
        deltas |= _deltas(ours, theirs, scope)
        assert _scoped_keys(ours, names, scope) == _scoped_keys(theirs, names, scope)

    assert deltas == EXPECTED_DELTAS


def test_required_fields_and_foreign_keys_across_all_tables(
    imported: tuple[ParsedManifest, list[str]],
) -> None:
    parsed, _ = imported
    rows = _csv_rows()
    required = _by_table(rows, _required)
    foreign = _by_table(rows, _foreign_key)

    assert {e.qualified_name for e in parsed.entities} == set(required)
    for entity in parsed.entities:
        name = entity.qualified_name
        assert {p.name for p in entity.properties if not p.nullable} == required[name]
        if entity.obligation is None:
            continue
        assert set(entity.obligation.required_fields) == required[name]
        assert {
            (local, fk.referenced_entity, remote)
            for fk in entity.obligation.foreign_keys
            for local, remote in fk.join_keys
        } == foreign[name]


def test_tables_without_an_obligation_are_listed(
    imported: tuple[ParsedManifest, list[str]],
) -> None:
    parsed, messages = imported
    rows = _csv_rows()
    required = _by_table(rows, _required)
    keyed = _by_table(rows, lambda r: r["cdmFieldName"] if r["isPrimaryKey"] == "Yes" else None)
    expected = {name for name in required if not required[name] or not keyed[name]}

    missing = {e.qualified_name for e in parsed.entities if e.obligation is None}

    assert missing == expected
    assert len(missing) == 11
    for name in missing:
        assert [m for m in messages if name in m and "no obligation" in m]
