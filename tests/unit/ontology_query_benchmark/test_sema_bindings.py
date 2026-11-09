"""Supplied product-spec bindings: provenance, approval policy and refusal findings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    BindingStatus,
    SpecPinError,
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build_utils import (
    LibraryFacts,
    import_library_facts,
)

pytestmark = pytest.mark.unit

CONCEPT_SPEC = Path(__file__).parents[2] / "fixtures/ontology_query_benchmark/concept_source.yaml"
PEOPLE = "pt_people"
EVENTS = "pt_events"
RID = "rid_L1"
CONVENTIONS = ("People means distinct rids.", "A distinct event is two keys.")


def _pass_through(entity: str, name: str, de_id: str) -> dict[str, Any]:
    col = f"{name}_{de_id}"
    return {
        "entity_id": entity, "column_id": f"{name}-v1", "column_name": col,
        "physical_type": "STRING", "cardinality": "single",
        "strategy": "pass_through", "de_id": de_id, "concept_name": name,
        "value_column": col, "label_column": f"{col}_label",
        "mapping_type_column": f"{col}_mapping_type", "pv_id_column": col,
    }


def _scalar(entity: str, column_id: str, col: str, type_: str) -> dict[str, Any]:
    return {
        "entity_id": entity, "column_id": column_id, "column_name": col,
        "physical_type": type_, "cardinality": "single", "strategy": "scalar",
    }


def _people_spec() -> dict[str, Any]:
    return {
        "product_id": "people",
        "entities": [{"entity_id": "person", "grain": "one row per rid",
                      "key_columns": [RID]}],
        "fields": [
            _scalar("person", "record-id-v1", RID, "STRING"),
            _pass_through("person", "gender", "G1"),
            _pass_through("person", "vital", "V1"),
            _pass_through("person", "city", "C1"),
            {**_scalar("person", "derived-v1", "derived_x", "STRING"),
             "strategy": "derived"},
            _scalar("person", "ghost-v1", "ghost_col", "STRING"),
            _scalar("person", "age-v1", "age_years", "BIGINT"),
            {**_scalar("person", "birth-date-v1", "birth_date_N1", "DATE"), "de_id": "N1"},
        ],
        "relationships": [],
        "exposure": {"denied_columns": ["first_name"]},
    }


def _events_spec() -> dict[str, Any]:
    return {
        "product_id": "events",
        "entities": [{"entity_id": "event", "grain": "one row per event row",
                      "key_columns": ["row_id"]}],
        "fields": [
            _scalar("event", "row-surrogate-id-v1", "row_id", "STRING"),
            _scalar("event", "record-id-v1", RID, "STRING"),
            _pass_through("event", "site", "S1"),
        ],
        "relationships": [{
            "relationship_id": "person", "source_entity_id": "event",
            "target_entity_id": "person", "cardinality": "many_to_one",
            "source_columns": [RID], "target_columns": [RID],
            "target_product_id": "people",
        }],
        "exposure": {"denied_columns": []},
    }


def _columns(spec: dict[str, Any]) -> dict[str, str]:
    cols: dict[str, str] = {}
    for f in spec["fields"]:
        cols[f["column_name"]] = f["physical_type"].lower()
        if f["strategy"] == "pass_through":
            cols[f["label_column"]] = "string"
            cols[f["mapping_type_column"]] = "string"
    return cols


def catalog_fixture() -> dict[str, dict[str, str]]:
    people = _columns(_people_spec())
    del people["ghost_col"]
    people["age_years"] = "string"
    people["first_name"] = "string"
    events = _columns(_events_spec())
    events["site_S1_concept_code"] = "string"
    return {PEOPLE: people, EVENTS: events}


ELEMENTS: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "G1": ("Administrative gender", "Gender for billing.",
           [("L8532", "Female"), ("L8507", "Male")]),
    "V1": ("Vital status", "Dead or alive.", [("L1", "Alive"), ("L2", "Deceased")]),
    "S1": ("Site of event", "Site code.",
           [("P1", "North"), ("P2", "South"), ("P3", "East")]),
}

_ELEMENT_DDL = (
    "CREATE TABLE lib.lib_element ("
    "element_id VARCHAR, element_label VARCHAR, element_title VARCHAR, "
    "element_definition VARCHAR, element_type VARCHAR, element_sensitive BOOLEAN, "
    "element_synonyms VARCHAR[], member_id VARCHAR, member_label VARCHAR, "
    "member_title VARCHAR, member_vocabulary VARCHAR, member_code VARCHAR, "
    "member_synonyms VARCHAR[])"
)
_CONCEPT_DDL = (
    "CREATE TABLE lib.lib_concept (concept_id VARCHAR, external_id VARCHAR, "
    "concept_code VARCHAR, vocabulary_id VARCHAR, definition VARCHAR)"
)


def write_projection(
    tmp_path: Path, elements: dict[str, tuple[str, str, list[tuple[str, str]]]] = ELEMENTS
) -> tuple[Path, Path]:
    """A scoped projection in the workload layout plus its extract receipt."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "projection.duckdb"
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA lib")
    con.execute(_ELEMENT_DDL)
    con.execute(_CONCEPT_DDL)
    con.executemany(
        "INSERT INTO lib.lib_element VALUES "
        "(?, ?, ?, ?, 'value set', false, [], ?, ?, ?, NULL, NULL, [])",
        [(de, label, f"{de}: {label}", definition, pv, pv_label, pv_label)
         for de, (label, definition, members) in elements.items()
         for pv, pv_label in members],
    )
    con.close()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "complete": True,
        "duckdb": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }))
    return path, receipt


def facts_fixture(tmp_path: Path) -> LibraryFacts:
    return import_library_facts(CONCEPT_SPEC, *write_projection(tmp_path))


def write_fixture(tmp_path: Path) -> Path:
    """Write a two-product sidecar; return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pins = []
    for product, table, spec in (
        ("people", PEOPLE, _people_spec()),
        ("events", EVENTS, _events_spec()),
    ):
        path = tmp_path / f"{product}.json"
        path.write_text(json.dumps(spec))
        pins.append({"product": product, "table": table, "path": path.name,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    sidecar = {
        "schema_version": 1, "sidecar_version": "t-1",
        "authority": "Spec test authority",
        "approval_policy": {"kind": "supplied_approved",
                            "executable_strategies": ["scalar", "pass_through"]},
        "model_role": "BENCH_S", "concept_vocabulary": "lib",
        "physical": {"catalog": "cat", "schema": "sch"},
        "specs": pins,
        "conventions": {"path": "rules.json", "file_sha256": "x",
                        "conventions_sha256": hashlib.sha256(
                            "\n".join(CONVENTIONS).encode()).hexdigest()},
        "spec_name": "SPEC",
        "target_model": {"id": "toy-s", "namespace": "toy", "display_name": "Toy arm S"},
        "grain": {"product": "events", "columns": [RID, "site_S1"],
                  "surrogate_row_id": "row_id",
                  "note": "Event grain columns (supplied): {columns}. "
                          "{surrogate_row_id} is the surrogate row id, not the event grain."},
        "rules_scope": "people counting and null handling",
        "membership_relation": "lib.sl.data_element",
        "membership_sql": "SELECT member_id, member_label FROM {relation} "
                          "WHERE element_id = '{element_id}'",
        "catalog_companion_suffix": "_concept_code",
        "hierarchy_heading": "TOY HIERARCHY REFERENCE",
        "hierarchy_reference": ["concept_ancestor: ancestor_concept_id is broader."],
        "context_char_budget": 16000,
        "value_list_limit": 40,
    }
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar))
    return sidecar_path


def resolution_fixture(tmp_path: Path) -> BindingResolution:
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    return resolve_bindings(sidecar, load_products(sidecar), catalog_fixture())


def test_approved_binding_carries_library_identity_and_provenance(
    tmp_path: Path,
) -> None:
    gender = resolution_fixture(tmp_path).by_column(PEOPLE, "gender_G1")
    assert gender is not None
    assert gender.status is BindingStatus.APPROVED
    assert (gender.library_id, gender.de_id) == ("gender-v1", "G1")
    assert gender.relation == f"cat.sch.{PEOPLE}"
    assert gender.companions == ("gender_G1_label", "gender_G1_mapping_type")
    assert gender.spec.sha256 == hashlib.sha256(
        (tmp_path / "people.json").read_bytes()).hexdigest()
    assert gender.authority == "Spec test authority"


def test_unapproved_strategy_is_refused_with_a_finding(tmp_path: Path) -> None:
    resolution = resolution_fixture(tmp_path)
    derived = resolution.by_column(PEOPLE, "derived_x")
    assert derived is not None and derived.status is BindingStatus.REFUSED
    assert any("derived_x" in f and "strategy" in f for f in resolution.findings)


def test_missing_column_and_type_mismatch_are_refused(tmp_path: Path) -> None:
    resolution = resolution_fixture(tmp_path)
    ghost = resolution.by_column(PEOPLE, "ghost_col")
    age = resolution.by_column(PEOPLE, "age_years")
    assert ghost is not None and ghost.status is BindingStatus.REFUSED
    assert age is not None and age.status is BindingStatus.REFUSED
    assert any(
        "ghost_col" in f and "no physical column" in f for f in resolution.findings
    )
    assert any("age_years" in f and "type" in f for f in resolution.findings)


def test_approved_set_excludes_refused_bindings(tmp_path: Path) -> None:
    approved = {b.column for b in resolution_fixture(tmp_path).approved()}
    assert {"gender_G1", RID, "site_S1"} <= approved
    assert not approved & {"derived_x", "ghost_col", "age_years"}


def test_spec_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    sidecar_path = write_fixture(tmp_path)
    (tmp_path / "people.json").write_text("{}")
    with pytest.raises(SpecPinError, match="people"):
        load_products(load_sidecar(sidecar_path, tmp_path))


def test_competing_bindings_for_one_column_are_ambiguous(tmp_path: Path) -> None:
    sidecar_path = write_fixture(tmp_path)
    spec = _people_spec()
    spec["fields"].append(_scalar("person", "other-v1", "gender_G1", "STRING"))
    path = tmp_path / "people.json"
    path.write_text(json.dumps(spec))
    raw = json.loads(sidecar_path.read_text())
    raw["specs"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(raw))
    sidecar = load_sidecar(sidecar_path, tmp_path)
    resolution = resolve_bindings(sidecar, load_products(sidecar), catalog_fixture())
    assert "gender_G1" not in {b.column for b in resolution.approved()}
    assert any("gender_G1" in f and "ambiguous" in f for f in resolution.findings)


def test_catalog_concept_code_column_is_a_marked_companion(tmp_path: Path) -> None:
    resolution = resolution_fixture(tmp_path)
    site = resolution.by_column(EVENTS, "site_S1")
    gender = resolution.by_column(PEOPLE, "gender_G1")
    assert site is not None and gender is not None
    assert site.catalog_companions == ("site_S1_concept_code",)
    assert site.companions == ("site_S1_label", "site_S1_mapping_type")
    assert gender.catalog_companions == ()
