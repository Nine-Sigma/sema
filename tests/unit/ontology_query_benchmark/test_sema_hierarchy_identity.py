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
