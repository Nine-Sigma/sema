"""One vocabulary release per hierarchy, and an anchor slot a concept code fills."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from showcase.ontology_query_benchmark.sema_hierarchy_sql import membership_sql
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    ANCHOR,
    add_typed_vocabulary,
    hierarchy,
    typed_hierarchy,
    warehouse,
)
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import _build, _facts, _reader

pytestmark = pytest.mark.unit

MAPPING = "wh.voc.relationship"


def test_a_relation_outside_the_declared_release_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match=MAPPING):
        hierarchy(release="voc.a")


def test_an_exempt_relation_must_be_one_the_hierarchy_declares() -> None:
    with pytest.raises(ValidationError, match="wh.other.x"):
        hierarchy(release="voc", release_exempt=["wh.other.x"])


def test_relations_sharing_the_release_load() -> None:
    assert hierarchy(release="voc").release == "voc"


def test_an_undeclared_release_is_a_build_finding(tmp_path: Path) -> None:
    _, manifest = _build(tmp_path, _facts(tmp_path, hierarchy()), _reader())

    assert any("no vocabulary release" in f for f in manifest.findings)


def test_an_exempt_relation_is_a_build_finding(tmp_path: Path) -> None:
    spec = hierarchy(release="voc.a", release_exempt=[MAPPING, "wh.voc.concept"])
    _, manifest = _build(tmp_path, _facts(tmp_path, spec), _reader())

    exempt = [f for f in manifest.findings if "outside vocabulary release voc.a" in f]
    assert len(exempt) == 2 and MAPPING in exempt[0]


def test_a_shared_release_adds_no_release_finding(tmp_path: Path) -> None:
    _, manifest = _build(tmp_path, _facts(tmp_path, hierarchy(release="voc")), _reader())

    assert not any("release" in f for f in manifest.findings)


def test_an_anchor_code_fills_the_ancestor_slot() -> None:
    con = warehouse()
    text_ids = membership_sql(hierarchy(), "dx", anchor=ANCHOR)
    typed_ids = membership_sql(typed_hierarchy(), "dx", anchor="1")

    found = con.execute(f"SELECT DISTINCT dx FROM wh.data.events WHERE {text_ids}").fetchall()
    assert {row[0] for row in found} == {"L10", "L11"}
    typed = add_typed_vocabulary(con).execute(
        f"SELECT dx FROM (VALUES ('L10'), ('L11'), ('L12')) v(dx) WHERE {typed_ids}"
    ).fetchall()
    assert typed == [("L10",)]
    assert "{anchor_id}" in membership_sql(hierarchy(), "dx")
