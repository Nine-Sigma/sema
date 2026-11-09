"""Arm S sidecar: an identity overlay beside a frozen product spec, and the warehouse dialect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.models.target.identity import EntityIdentity
from showcase.ontology_query_benchmark.product_spec import ProductSpecDoc, ProductSpecError
from showcase.ontology_query_benchmark.sema_bindings import load_products, load_sidecar
from tests.unit.ontology_query_benchmark.test_sema_bindings import RID, write_fixture

pytestmark = pytest.mark.unit


OVERLAY_IDENTITY = {"key_columns": [RID, "site_S1"], "exclude_null_keys": True}


def _events_doc(tmp_path: Path, **fixture: Any) -> ProductSpecDoc:
    sidecar = load_sidecar(write_fixture(tmp_path, **fixture), tmp_path)
    return next(p.doc for p in load_products(sidecar) if p.pin.product == "events")


def test_an_entity_identity_is_read_from_the_sidecar_overlay(tmp_path: Path) -> None:
    overlay = {"entity_id": "event", "identity": OVERLAY_IDENTITY}
    entity = _events_doc(tmp_path, events_overlay=overlay).identity_entity
    assert entity is not None
    assert entity.entity_id == "event"
    assert entity.identity == EntityIdentity.model_validate(OVERLAY_IDENTITY)


def test_an_overlay_equal_to_the_spec_identity_is_accepted(tmp_path: Path) -> None:
    overlay = {"entity_id": "event", "identity": OVERLAY_IDENTITY}
    doc = _events_doc(tmp_path, events_identity=OVERLAY_IDENTITY, events_overlay=overlay)
    assert doc.identity_entity is not None
    assert doc.identity_entity.identity == EntityIdentity.model_validate(OVERLAY_IDENTITY)


@pytest.mark.parametrize("overlay", [
    {"entity_id": "event", "identity": {"key_columns": [RID]}},
    {"entity_id": "visit", "identity": OVERLAY_IDENTITY},
])  # fmt: skip
def test_an_overlay_that_differs_from_the_spec_identity_is_refused_at_load(
    tmp_path: Path, overlay: dict[str, Any]
) -> None:
    with pytest.raises(ProductSpecError, match="conflicts with the spec identity"):
        _events_doc(tmp_path, events_identity=OVERLAY_IDENTITY, events_overlay=overlay)


def test_an_overlay_naming_no_spec_entity_is_refused_at_load(tmp_path: Path) -> None:
    overlay = {"entity_id": "visit", "identity": OVERLAY_IDENTITY}
    with pytest.raises(ProductSpecError, match="names no entity"):
        _events_doc(tmp_path, events_overlay=overlay)


def test_an_overlay_with_an_unknown_key_is_refused_at_load(tmp_path: Path) -> None:
    overlay = {"entity_id": "event", "identity": OVERLAY_IDENTITY, "grain": "x"}
    with pytest.raises(ValueError, match="grain"):
        _events_doc(tmp_path, events_overlay=overlay)


def test_the_sidecar_names_its_warehouse_sql_dialect(tmp_path: Path) -> None:
    sidecar = load_sidecar(write_fixture(tmp_path, sql_dialect="databricks"), tmp_path)
    assert sidecar.sql_dialect == "databricks"


@pytest.mark.parametrize("dialect", [None, "", "not_a_dialect"])
def test_a_sidecar_without_a_known_sql_dialect_is_refused_at_load(
    tmp_path: Path, dialect: str | None
) -> None:
    with pytest.raises((KeyError, ValueError), match="sql_dialect|dialect"):
        load_sidecar(write_fixture(tmp_path, sql_dialect=dialect), tmp_path)
