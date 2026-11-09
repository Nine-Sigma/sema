"""One vocabulary release per hierarchy, and an anchor slot a concept code fills."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from pydantic import ValidationError

from showcase.ontology_query_benchmark.profile import Attribution, ConceptHierarchy
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from showcase.ontology_query_benchmark.sema_hierarchy_sql import membership_sql
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    ANCHOR,
    add_hierarchy_slice,
    add_typed_vocabulary,
    hierarchy,
    repin,
    typed_hierarchy,
    warehouse,
)
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONCEPT_SPEC,
    ELEMENTS,
    write_projection,
)
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import SITE, _build, _facts, _reader

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


def _facts_without(tmp_path: Path, spec: ConceptHierarchy, concept: str) -> LibraryFacts:
    """The synthetic slice with one mapping target's concept row missing."""
    projection, receipt = write_projection(tmp_path, {**ELEMENTS, "S1": SITE})
    add_hierarchy_slice(projection, receipt)
    with duckdb.connect(str(projection)) as con:
        con.execute("DELETE FROM concept_hierarchy.concept WHERE cid = ?", [concept])
    repin(projection, receipt)
    return import_library_facts(CONCEPT_SPEC, projection, receipt,
                                spec_attribution=Attribution.ASSISTED, hierarchy=spec)  # fmt: skip


def test_an_exempt_mapping_whose_targets_all_have_concepts_adds_no_finding(
    tmp_path: Path,
) -> None:
    spec = hierarchy(release="voc.a", release_exempt=[MAPPING, "wh.voc.concept"])
    _, manifest = _build(tmp_path, _facts(tmp_path, spec), _reader())

    assert not any("release" in f or "mapping targets" in f for f in manifest.findings)


def test_mapping_targets_without_a_concept_row_are_a_measured_finding(tmp_path: Path) -> None:
    spec = hierarchy(release="voc.a", release_exempt=[MAPPING, "wh.voc.concept"])
    _, manifest = _build(tmp_path, _facts_without(tmp_path, spec, "C7"), _reader())

    [finding] = [f for f in manifest.findings if "mapping targets" in f]
    assert finding.startswith("1 of 5 mapping targets")
    assert MAPPING in finding and "wh.voc.concept" in finding


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
