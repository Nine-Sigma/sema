"""Supplied CPD bindings: provenance, approval policy and refusal findings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from showcase.partner_timbr_benchmark.sema_bindings import (
    BindingResolution,
    BindingStatus,
    SpecPinError,
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.partner_timbr_benchmark.sema_build_utils import (
    CONCEPT_SPEC_PATH,
    LibraryFacts,
    import_library_facts,
)

pytestmark = pytest.mark.unit

DEMO = "pb_demo"
TUMOR = "pb_tumor"
MRN = "mrn_PRT1"
CONVENTIONS = ("Patients means distinct MRNs.", "A distinct tumor is six keys.")


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


def _demo_spec() -> dict[str, Any]:
    return {
        "product_id": "demographics",
        "entities": [{"entity_id": "person", "grain": "one row per MRN",
                      "key_columns": [MRN]}],
        "fields": [
            _scalar("person", "medical-record-number-v1", MRN, "STRING"),
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


def _tumor_spec() -> dict[str, Any]:
    return {
        "product_id": "cancer_registry",
        "entities": [{"entity_id": "primary_tumor", "grain": "one row per tumor row",
                      "key_columns": ["row_id"]}],
        "fields": [
            _scalar("primary_tumor", "row-surrogate-id-v1", "row_id", "STRING"),
            _scalar("primary_tumor", "medical-record-number-v1", MRN, "STRING"),
            _pass_through("primary_tumor", "site", "S1"),
        ],
        "relationships": [{
            "relationship_id": "person", "source_entity_id": "primary_tumor",
            "target_entity_id": "person", "cardinality": "many_to_one",
            "source_columns": [MRN], "target_columns": [MRN],
            "target_product_id": "demographics",
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
    demo = _columns(_demo_spec())
    del demo["ghost_col"]
    demo["age_years"] = "string"
    demo["first_name"] = "string"
    tumor = _columns(_tumor_spec())
    tumor["site_S1_concept_code"] = "string"
    return {DEMO: demo, TUMOR: tumor}


ELEMENTS: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "G1": ("Administrative gender", "Gender for billing.",
           [("PRT8532", "Female"), ("PRT8507", "Male")]),
    "V1": ("Vital status", "Dead or alive.", [("PRT1", "Alive"), ("PRT2", "Deceased")]),
    "S1": ("Site of primary tumor", "ICD-O-3 site.",
           [("P1", "Lung"), ("P2", "Breast"), ("P3", "Colon")]),
}

_DATA_ELEMENT_DDL = (
    "CREATE TABLE standard_library.standard_library_data_element ("
    "data_element_prt_id VARCHAR, data_element_prt_preferred_label VARCHAR, "
    "data_element_title VARCHAR, data_element_definition VARCHAR, "
    "data_element_data_type_id VARCHAR, data_element_phi BOOLEAN, "
    "data_element_synonym_name VARCHAR[], permissible_value_prt_id VARCHAR, "
    "permissible_value_prt_preferred_label VARCHAR, permissible_value_title VARCHAR, "
    "permissible_value_vocabulary_id VARCHAR, permissible_value_concept_code VARCHAR, "
    "permissible_value_synonym_name VARCHAR[])"
)
_CONCEPT_DDL = (
    "CREATE TABLE standard_library.standard_library_concept (concept_prt_id VARCHAR, "
    "concept_athena_id VARCHAR, concept_code VARCHAR, vocabulary_id VARCHAR, "
    "definition VARCHAR)"
)


def write_projection(
    tmp_path: Path, elements: dict[str, tuple[str, str, list[tuple[str, str]]]] = ELEMENTS
) -> tuple[Path, Path]:
    """A scoped projection in the workload layout plus its extract receipt."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "projection.duckdb"
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA standard_library")
    con.execute(_DATA_ELEMENT_DDL)
    con.execute(_CONCEPT_DDL)
    con.executemany(
        "INSERT INTO standard_library.standard_library_data_element VALUES "
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
    return import_library_facts(CONCEPT_SPEC_PATH, *write_projection(tmp_path))


def write_fixture(tmp_path: Path) -> Path:
    """Write a two-product sidecar; return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pins = []
    for product, table, spec in (
        ("demographics", DEMO, _demo_spec()),
        ("cancer_registry", TUMOR, _tumor_spec()),
    ):
        path = tmp_path / f"{product}.json"
        path.write_text(json.dumps(spec))
        pins.append({"product": product, "table": table, "path": path.name,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    sidecar = {
        "schema_version": 1, "sidecar_version": "t-1",
        "authority": "CPD test authority",
        "approval_policy": {"kind": "supplied_approved",
                            "executable_strategies": ["scalar", "pass_through"]},
        "model_role": "BENCH_S", "concept_vocabulary": "lib",
        "physical": {"catalog": "cat", "schema": "sch"},
        "harness_root": str(tmp_path), "specs": pins,
        "conventions": {"path": "w4s.json", "file_sha256": "x",
                        "conventions_sha256": hashlib.sha256(
                            "\n".join(CONVENTIONS).encode()).hexdigest()},
        "tumor_grain": {"product": "cancer_registry", "columns": [MRN, "site_S1"],
                        "surrogate_row_id": "row_id"},
        "membership_relation": "lib.sl.data_element",
        "hierarchy_reference": ["concept_ancestor: ancestor_concept_id is broader."],
        "context_char_budget": 16000,
        "value_list_limit": 40,
    }
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar))
    return sidecar_path


def resolution_fixture(tmp_path: Path) -> BindingResolution:
    sidecar = load_sidecar(write_fixture(tmp_path))
    return resolve_bindings(sidecar, load_products(sidecar), catalog_fixture())


def test_approved_binding_carries_library_identity_and_provenance(
    tmp_path: Path,
) -> None:
    gender = resolution_fixture(tmp_path).by_column(DEMO, "gender_G1")
    assert gender is not None
    assert gender.status is BindingStatus.APPROVED
    assert (gender.library_id, gender.de_id) == ("gender-v1", "G1")
    assert gender.relation == f"cat.sch.{DEMO}"
    assert gender.companions == ("gender_G1_label", "gender_G1_mapping_type")
    assert gender.spec.sha256 == hashlib.sha256(
        (tmp_path / "demographics.json").read_bytes()).hexdigest()
    assert gender.authority == "CPD test authority"


def test_unapproved_strategy_is_refused_with_a_finding(tmp_path: Path) -> None:
    resolution = resolution_fixture(tmp_path)
    derived = resolution.by_column(DEMO, "derived_x")
    assert derived is not None and derived.status is BindingStatus.REFUSED
    assert any("derived_x" in f and "strategy" in f for f in resolution.findings)


def test_missing_column_and_type_mismatch_are_refused(tmp_path: Path) -> None:
    resolution = resolution_fixture(tmp_path)
    ghost = resolution.by_column(DEMO, "ghost_col")
    age = resolution.by_column(DEMO, "age_years")
    assert ghost is not None and ghost.status is BindingStatus.REFUSED
    assert age is not None and age.status is BindingStatus.REFUSED
    assert any(
        "ghost_col" in f and "no physical column" in f for f in resolution.findings
    )
    assert any("age_years" in f and "type" in f for f in resolution.findings)


def test_approved_set_excludes_refused_bindings(tmp_path: Path) -> None:
    approved = {b.column for b in resolution_fixture(tmp_path).approved()}
    assert {"gender_G1", MRN, "site_S1"} <= approved
    assert not approved & {"derived_x", "ghost_col", "age_years"}


def test_spec_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    sidecar_path = write_fixture(tmp_path)
    (tmp_path / "demographics.json").write_text("{}")
    with pytest.raises(SpecPinError, match="demographics"):
        load_products(load_sidecar(sidecar_path))


def test_competing_bindings_for_one_column_are_ambiguous(tmp_path: Path) -> None:
    sidecar_path = write_fixture(tmp_path)
    spec = _demo_spec()
    spec["fields"].append(_scalar("person", "other-v1", "gender_G1", "STRING"))
    path = tmp_path / "demographics.json"
    path.write_text(json.dumps(spec))
    raw = json.loads(sidecar_path.read_text())
    raw["specs"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(raw))
    sidecar = load_sidecar(sidecar_path)
    resolution = resolve_bindings(sidecar, load_products(sidecar), catalog_fixture())
    assert "gender_G1" not in {b.column for b in resolution.approved()}
    assert any("gender_G1" in f and "ambiguous" in f for f in resolution.findings)


def test_catalog_concept_code_column_is_a_marked_companion(tmp_path: Path) -> None:
    resolution = resolution_fixture(tmp_path)
    site = resolution.by_column(TUMOR, "site_S1")
    gender = resolution.by_column(DEMO, "gender_G1")
    assert site is not None and gender is not None
    assert site.catalog_companions == ("site_S1_concept_code",)
    assert site.companions == ("site_S1_label", "site_S1_mapping_type")
    assert gender.catalog_companions == ()
