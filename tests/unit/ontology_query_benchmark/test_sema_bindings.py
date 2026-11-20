"""Supplied product-spec bindings: provenance, approval policy and refusal findings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    BindingStatus,
    SpecPinError,
    load_products,
    load_sidecar,
    resolve_bindings,
)
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    EVENTS,
    ID_SUFFIX,
    PEOPLE,
    RID,
    _people_spec,
    _scalar,
    catalog_fixture,
    write_fixture,
)

pytestmark = pytest.mark.unit


def resolution_fixture(tmp_path: Path) -> BindingResolution:
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    return resolve_bindings(sidecar, load_products(sidecar), catalog_fixture())


def test_approved_binding_carries_library_identity_and_provenance(
    tmp_path: Path,
) -> None:
    gender = resolution_fixture(tmp_path).by_column(PEOPLE, "gender_G1")
    assert gender is not None
    assert gender.status is BindingStatus.APPROVED
    assert (gender.library_id, gender.de_id) == (f"gender{ID_SUFFIX}", "G1")
    assert gender.relation == f"cat.sch.{PEOPLE}"
    assert gender.companions == (("label", "gender_G1_text"), ("mapping_type", "gender_G1_how"))
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


def _repin_people(tmp_path: Path, spec: dict[str, Any]) -> Path:
    """Rewrite the people spec and its sidecar pin; return the sidecar path."""
    sidecar_path = write_fixture(tmp_path)
    path = tmp_path / "people.json"
    path.write_text(json.dumps(spec))
    raw = json.loads(sidecar_path.read_text())
    raw["specs"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(raw))
    return sidecar_path


def test_competing_bindings_for_one_column_are_ambiguous(tmp_path: Path) -> None:
    spec = _people_spec()
    spec["fields"].append(_scalar("person", f"other{ID_SUFFIX}", "gender_G1", "STRING"))
    sidecar = load_sidecar(_repin_people(tmp_path, spec), tmp_path)
    resolution = resolve_bindings(sidecar, load_products(sidecar), catalog_fixture())
    assert "gender_G1" not in {b.column for b in resolution.approved()}
    assert any("gender_G1" in f and "ambiguous" in f for f in resolution.findings)


def test_catalog_concept_code_column_is_a_marked_companion(tmp_path: Path) -> None:
    resolution = resolution_fixture(tmp_path)
    site = resolution.by_column(EVENTS, "site_S1")
    gender = resolution.by_column(PEOPLE, "gender_G1")
    assert site is not None and gender is not None
    assert site.catalog_companions == ("site_S1_concept_code",)
    assert site.companion_columns == ("site_S1_label", "site_S1_mapping_type")
    assert gender.catalog_companions == ()


def test_a_spec_that_breaks_the_declared_format_is_refused_at_load(tmp_path: Path) -> None:
    spec = _people_spec()
    del spec["fields"][1]["strategy"]
    sidecar = load_sidecar(_repin_people(tmp_path, spec), tmp_path)
    with pytest.raises(ValueError, match="people.json"):
        load_products(sidecar)


def test_a_spec_declaring_two_entity_identities_is_refused_at_load(tmp_path: Path) -> None:
    spec = _people_spec()
    identity = {"key_columns": [RID]}
    spec["entities"] = [{"entity_id": name, "key_columns": [RID], "identity": identity}
                        for name in ("person", "member")]  # fmt: skip
    sidecar = load_sidecar(_repin_people(tmp_path, spec), tmp_path)
    with pytest.raises(ValueError, match="one entity identity"):
        load_products(sidecar)


def test_an_entity_identity_is_read_from_the_spec(tmp_path: Path) -> None:
    identity = {"key_columns": [RID, "site_S1"], "exclude_null_keys": True}
    sidecar = load_sidecar(write_fixture(tmp_path, events_identity=identity), tmp_path)
    events = next(p for p in load_products(sidecar) if p.pin.product == "events")
    (entity,) = events.doc.entities
    assert entity.entity_id == "event"
    assert entity.identity is not None
    assert entity.identity.key_columns == (RID, "site_S1")
    assert entity.identity.exclude_null_keys is True
