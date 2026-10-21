"""A1 offline fixture: synthesized standard-library relations and their hand-written spec."""

from __future__ import annotations

import importlib.util
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_parser import parse_manifest_raw

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parents[2] / "fixtures" / "intake" / "partner_sl"
DATABASE = FIXTURE / "library.duckdb"
SPEC = FIXTURE / "standard_library.spec.yaml"
RUN = FIXTURE / "standard_library.run.yaml"
BUILDER = FIXTURE / "build_fixture.py"
SCHEMA = "standard_library"
COLUMNS = {
    "standard_library_data_element": [
        "domain_prt_id",
        "data_element_prt_id",
        "data_element_prt_preferred_label",
        "data_element_title",
        "data_element_definition",
        "data_element_data_type_id",
        "data_element_phi",
        "data_element_synonym_name",
        "permissible_value_prt_id",
        "permissible_value_prt_preferred_label",
        "permissible_value_title",
        "permissible_value_vocabulary_id",
        "permissible_value_concept_code",
        "permissible_value_synonym_name",
    ],
    "standard_library_concept": [
        "concept_prt_id",
        "concept_athena_id",
        "vocabulary_id",
        "concept_code",
        "definition",
    ],
    "standard_library_concept_relationship": [
        "concept_prt_id_1",
        "concept_athena_id_1",
        "concept_prt_id_2",
        "concept_athena_id_2",
        "relationship_id",
        "is_hierarchical",
        "defines_ancestry",
    ],
    "data_element_mappings": [
        "unified_column_data_element_prt_id",
        "mapped_source_dataset_label",
        "mapped_source_dataset_rid",
        "mapped_source_columns",
        "mapped_project_label",
        "unified_dataset_location",
    ],
    "permissible_value_mappings_edited": [
        "data_element_prt_id",
        "given_value",
        "mapped_permissible_value_prt_ids",
        "mapping_status",
    ],
}
CONCEPTS = {
    "PRTL2140000129": "Administrative gender",
    "PRT4179218": "Birth sex",
    "PRT44803968": "Ethnicity",
    "PRTL2140007288": "Ethnicity category",
    "PRT4110772": "Gender identity",
    "PRTL2140007289": "Hispanic or Latino ethnicity group",
    "PRT4181605": "Preferred language",
    "PRT4013886": "Race",
    "PRT4216292": "Race category",
    "PRT35917103": "Race, cancer registry",
    "PRT35918387": "Spanish/Hispanic Origin, cancer registry",
    "PRT4031696": "Spiritual or religious belief",
}
RACE = "PRT4013886"
RACE_REFS = {("omop", "4013886"), ("SNOMED", "103579009")}


@dataclass(frozen=True)
class _Imported:
    spec: Any
    result: Any


def _copy(tmp_path_factory: pytest.TempPathFactory, name: str) -> Path:
    target = tmp_path_factory.mktemp(name) / DATABASE.name
    shutil.copyfile(DATABASE, target)
    return target


def _builder() -> ModuleType:
    loaded = importlib.util.spec_from_file_location("partner_sl_build_fixture", BUILDER)
    assert loaded is not None and loaded.loader is not None
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    return module


def _tables(staging: Staging) -> dict[str, list[str]]:
    rows = staging.execute(
        "SELECT table_name, column_name FROM duckdb_columns() "
        "WHERE schema_name = ? ORDER BY table_name, column_index",
        [SCHEMA],
    ).fetchall()
    tables: dict[str, list[str]] = {}
    for table, column in rows:
        tables.setdefault(table, []).append(column)
    return tables


def _rows(staging: Staging) -> dict[str, list[tuple[Any, ...]]]:
    return {
        table: staging.execute(
            f'SELECT * FROM {SCHEMA}."{table}" ORDER BY ALL'
        ).fetchall()
        for table in COLUMNS
    }


@pytest.fixture(scope="module")
def staging(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Staging]:
    with Staging(str(_copy(tmp_path_factory, "partner_sl")), schemas=()) as opened:
        yield opened


@pytest.fixture(scope="module")
def imported(staging: Staging) -> _Imported:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec, load_dataset_map
    from sema.intake.relations import RelationProvider

    spec = load_concept_source_spec(SPEC)
    assert spec.authored_bindings is not None
    dataset_map = load_dataset_map(spec.authored_bindings.dataset_map)
    provider = RelationProvider(warehouse=None, staging=staging)
    result = import_concept_source(spec, provider, dataset_map=dataset_map)
    return _Imported(spec=spec, result=result)


def test_fixture_holds_the_library_relations_with_their_columns(
    staging: Staging,
) -> None:
    assert _tables(staging) == COLUMNS


def test_fixture_holds_the_twelve_concepts_by_id_and_label(staging: Staging) -> None:
    rows = staging.execute(
        f"SELECT DISTINCT data_element_prt_id, data_element_prt_preferred_label "
        f"FROM {SCHEMA}.standard_library_data_element"
    ).fetchall()

    assert dict(rows) == CONCEPTS
    assert len(rows) == len(CONCEPTS)


def test_every_relation_holds_rows(staging: Staging) -> None:
    assert all(rows for rows in _rows(staging).values())


def test_builder_reproduces_the_committed_rows(
    staging: Staging, tmp_path: Path
) -> None:
    rebuilt = tmp_path / DATABASE.name
    _builder().build(rebuilt)

    with Staging(str(rebuilt), schemas=()) as fresh:
        assert _rows(fresh) == _rows(staging)


def test_spec_matches_the_run_descriptor(imported: _Imported) -> None:
    run = yaml.safe_load(RUN.read_text())
    spec = imported.spec

    assert (spec.id, spec.identity_system, spec.release) == (
        run["id"],
        run["identity_system"],
        str(run["release"]),
    )
    assert spec.scheme is not None
    assert spec.scheme.qualified_name == run["scheme"]["qualified_name"]
    assert spec.scheme.description == run["scheme"]["description"]


def test_spec_reads_every_relation_from_staging(imported: _Imported) -> None:
    refs = imported.spec.relation_refs()

    assert {ref.provider for ref in refs} == {"staging"}
    assert {ref.name for ref in refs} == {f"{SCHEMA}.{name}" for name in COLUMNS}


def test_import_yields_one_concept_property_per_concept(imported: _Imported) -> None:
    (scheme,) = imported.result.manifest["entities"]
    concepts = {p["concept_ref"]["id"]: p for p in scheme["properties"]}

    assert scheme["kind"] == "CONCEPT_SCHEME"
    assert set(concepts) == set(CONCEPTS)
    race_refs = {(r["system"], r["id"]) for r in concepts[RACE]["external_refs"]}
    assert race_refs == RACE_REFS


def test_import_yields_one_value_set_per_concept(
    imported: _Imported, staging: Staging
) -> None:
    manifest = imported.result.manifest
    expected = Counter(
        f"VS_{concept}"
        for (concept,) in staging.execute(
            f"SELECT data_element_prt_id FROM {SCHEMA}.standard_library_data_element"
        ).fetchall()
    )

    assert {v["name"] for v in manifest["vocabularies"]} == set(expected)
    assert Counter(t["vocabulary"] for t in manifest["terms"]) == expected


def test_import_carries_relationships_and_authored_evidence(
    imported: _Imported,
) -> None:
    result = imported.result

    assert result.manifest["relationships"]
    assert any(not row["far_end_loaded"] for row in result.manifest["relationships"])
    bindings = result.authored.column_bindings
    assert {b.resolved for b in bindings} == {True, False}
    mappings = result.authored.value_mappings
    assert {m.approved for m in mappings} == {True, False}


def test_manifest_parses(imported: _Imported) -> None:
    raw = yaml.safe_load(yaml.safe_dump(imported.result.manifest))
    parsed = parse_manifest_raw(raw)

    assert len(parsed.vocabularies) == len(CONCEPTS)


def test_manifest_normalizes_to_concept_properties(
    imported: _Imported, tmp_path: Path
) -> None:
    from sema.targets.adapters.manifest import ManifestTargetAdapter
    from sema.targets.normalizer import TargetModelNormalizer

    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(imported.result.manifest, sort_keys=False))
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(path))

    (scheme,) = normalized.entities
    kinds = [p.property_kind.value for p in scheme.properties]
    assert kinds == ["CONCEPT"] * len(CONCEPTS)


def test_source_note_states_the_rows_are_synthesized() -> None:
    note = (FIXTURE / "SOURCE.md").read_text()

    assert "synthesized" in note
    assert "no partner data values" in note
