"""A member whose own id is already a standard concept reaches the hierarchy through itself,
with no mapping row: in the extracted slice, as a graph bridge and in the fragment."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from showcase.ontology_query_benchmark.profile import Attribution
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    MEMBERS,
    WarehouseCursor,
    add_hierarchy_slice,
    hierarchy,
    repin,
    warehouse,
)
from tests.unit.ontology_query_benchmark.test_intake_extract_hierarchy import _extract, _slice
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONCEPT_SPEC,
    ELEMENTS,
    write_projection,
)
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import _bridges, _build, _reader
from tests.unit.ontology_query_benchmark.test_sema_hierarchy import _facts as _mapped_facts
from tests.unit.ontology_query_benchmark.test_sema_hierarchy_anchor import (
    EVENTS_SQL,
    READABLE,
    anchor,
    context,
    fragments,
    members,
)

pytestmark = pytest.mark.unit

# C9 is a standard concept under C8 and C7 a non-standard one; neither has a mapping row.
OWN_IDS = ("C9", "C7")
# The mapping-only text b90375f rendered for this spec, before the identity route existed.
MAPPED = ("site_S1 IN (SELECT m.id_1 FROM wh.voc.relationship m LEFT JOIN wh.voc.relationship r"
          " ON r.id_1 = m.id_2 AND r.rel IN ('Replaced by') JOIN wh.voc.ancestor a ON a.des IN"
          " (m.id_2, r.id_2) JOIN wh.voc.concept s ON s.cid = a.des AND s.flag IN ('S')"
          " WHERE m.rel IN ('Maps to') AND a.anc = {slot})")  # fmt: skip
MAPPED_ROUTE = ("A member reaches a concept through wh.voc.relationship where rel in ('Maps to'),"
                " then its replacement where rel in ('Replaced by'). Only concepts with"
                " wh.voc.concept.flag in ('S') count.")  # fmt: skip


def _facts(tmp_path: Path) -> LibraryFacts:
    site = ("Site of event", "Site code.", [(m, f"site {m}") for m in (*MEMBERS, *OWN_IDS)])
    projection, receipt = write_projection(tmp_path, {**ELEMENTS, "S1": site})
    add_hierarchy_slice(projection, receipt)
    with duckdb.connect(str(projection)) as con:
        con.executemany("INSERT INTO concept_hierarchy.concept VALUES (?, ?, ?)",
                        [("C8", "Unrelated", "S"), ("C9", "Unrelated child", "S")])  # fmt: skip
        con.execute("INSERT INTO concept_hierarchy.edge VALUES ('C8', 'C9')")
    repin(projection, receipt)
    return import_library_facts(
        CONCEPT_SPEC, projection, receipt,
        spec_attribution=Attribution.ASSISTED, hierarchy=hierarchy(),
    )


def _events() -> duckdb.DuckDBPyConnection:
    con = warehouse()
    con.executemany("INSERT INTO wh.data.events VALUES (?, ?)", [(98, "C9"), (99, "C7")])
    return con


def test_the_slice_takes_a_member_that_is_a_standard_concept_and_its_ancestors(
    tmp_path: Path,
) -> None:
    receipt = _extract(tmp_path, WarehouseCursor(warehouse((*MEMBERS, *OWN_IDS))))

    assert receipt["complete"] is True, receipt["incomplete"]
    assert {"C8", "C9"} <= {row[0] for row in _slice(tmp_path, receipt, "concept")}
    assert ("C8", "C9") in _slice(tmp_path, receipt, "edge")


def test_a_standard_member_bridges_to_itself_and_a_non_standard_one_does_not(
    tmp_path: Path,
) -> None:
    loader, _ = _build(tmp_path, _facts(tmp_path), _reader())

    assert ("lib", "C9", "std", "C9") in _bridges(loader)
    assert not {b for b in _bridges(loader) if b[1] == "C7"}


def test_the_fragment_selects_a_standard_member_under_its_anchor(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path), READABLE, [
        anchor("C8", "Unrelated", "std", 1), anchor("C1", "Neoplasm", "std", 2),
    ]).text  # fmt: skip

    filled = fragments(text, "site_S1")
    assert [members(_events(), EVENTS_SQL, f) for f in filled] == [{"L10", "L11"}, {"C9"}]


def test_without_a_member_that_is_a_concept_the_section_keeps_the_mapping_only_text(
    tmp_path: Path,
) -> None:
    text = context(tmp_path, _mapped_facts(tmp_path, hierarchy()), READABLE,
                   [anchor("C1", "Neoplasm", "std", 2)]).text  # fmt: skip

    assert fragments(text, "site_S1", filled=False) == [MAPPED.format(slot="{anchor_id}")]
    assert fragments(text, "site_S1") == [MAPPED.format(slot="'C1'")]
    assert f"  {MAPPED_ROUTE} In wh.voc.ancestor" in text


def test_a_member_that_is_a_concept_states_the_identity_route(tmp_path: Path) -> None:
    text = context(tmp_path, _facts(tmp_path), READABLE, [anchor("C8", "Unrelated", "std", 1)]).text

    assert ", or directly when its own id is a concept." in text
