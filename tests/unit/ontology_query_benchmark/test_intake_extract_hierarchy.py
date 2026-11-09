"""The projection adds a bounded hierarchy slice selected by membership, never the closure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

from showcase.ontology_query_benchmark.intake_extract import extract
from showcase.ontology_query_benchmark.profile import load_profile
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    HIERARCHY,
    PARENT_EDGES,
    WarehouseCursor,
    closure_rows,
    hierarchy,
    spec_with,
    warehouse,
    write_product_spec,
)

pytestmark = pytest.mark.unit


def _extract(tmp_path: Path, cursor: WarehouseCursor, **kwargs: Any) -> dict[str, Any]:
    spec = kwargs.pop("spec", spec_with(hierarchy()))
    return extract(lambda: cursor, spec, [write_product_spec(tmp_path / "events.json")],
                   tmp_path / "out", **kwargs)  # fmt: skip


def _slice(tmp_path: Path, receipt: dict[str, Any], table: str) -> set[tuple[Any, ...]]:
    with duckdb.connect(str(tmp_path / "out" / receipt["duckdb"]["path"]), read_only=True) as con:
        return set(con.execute(f"SELECT * FROM concept_hierarchy.{table}").fetchall())


def test_the_profile_loads_a_structured_concept_hierarchy(tmp_path: Path) -> None:
    raw = yaml.safe_load(PROFILE_PATH.read_text())
    raw["extract"]["concept_hierarchy"] = HIERARCHY
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))

    loaded = load_profile(path).extract.concept_hierarchy

    assert loaded is not None and loaded.mapping is not None
    assert loaded.mapping.replacement_types == ("Replaced by",)
    assert loaded.relations == ("wh.voc.relationship", "wh.voc.ancestor", "wh.voc.concept")


def test_the_slice_holds_member_mappings_their_ancestors_and_parent_edges(
    tmp_path: Path,
) -> None:
    receipt = _extract(tmp_path, WarehouseCursor(warehouse()))

    assert receipt["complete"] is True, receipt["incomplete"]
    assert _slice(tmp_path, receipt, "mapping") == {
        ("L10", "C3", "Maps to"), ("L11", "C4", "Maps to"), ("C4", "C5", "Replaced by"),
        ("L12", "C6", "Maps to"), ("L13", "C7", "Maps to"),
    }
    assert {row[0] for row in _slice(tmp_path, receipt, "concept")} == {
        "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    }
    assert _slice(tmp_path, receipt, "edge") == {
        ("C1", "C2"), ("C2", "C3"), ("C2", "C5"), ("C2", "C7"),
    }
    entry = receipt["relations"]["concept_hierarchy.edge"]
    assert entry["source"] == "wh.voc.ancestor"
    assert entry["expected_rows"] == entry["returned_rows"] == 4


def test_the_slice_edges_reproduce_the_warehouse_closure_of_the_slice(tmp_path: Path) -> None:
    con = warehouse()
    receipt = _extract(tmp_path, WarehouseCursor(con))
    concepts = {row[0] for row in _slice(tmp_path, receipt, "concept")}

    derived = closure_rows(sorted(_slice(tmp_path, receipt, "edge")), self_rows=False)

    warehouse_rows = con.execute("SELECT anc, des, lvl FROM wh.voc.ancestor WHERE lvl > 0")
    assert derived == {row for row in warehouse_rows.fetchall() if row[1] in concepts}
    assert ("C8", "C9", 1) not in derived and len(PARENT_EDGES) == 5


def test_a_slice_over_the_row_limit_writes_no_database(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, WarehouseCursor(warehouse()), row_limit=6)

    assert receipt["complete"] is False
    assert receipt["duckdb"] is None
    assert any(m.startswith("concept_hierarchy.concept: 7 counted rows exceed")
               for m in receipt["incomplete"])  # fmt: skip


def test_a_hierarchy_relation_that_moves_during_the_read_is_incomplete(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, WarehouseCursor(warehouse(), moved="ancestor"))

    assert receipt["complete"] is False
    assert any(m.startswith("concept_hierarchy.edge: updated_at moved")
               for m in receipt["incomplete"])  # fmt: skip


def test_without_a_hierarchy_only_the_library_relations_are_projected(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, WarehouseCursor(warehouse()), spec=spec_with(None))

    assert receipt["complete"] is True
    assert set(receipt["relations"]) == {"lib_element", "lib_concept"}
    with duckdb.connect(str(tmp_path / "out" / receipt["duckdb"]["path"]), read_only=True) as con:
        schemas = {r[0] for r in con.execute("SELECT schema_name FROM duckdb_schemas()").fetchall()}
    assert "concept_hierarchy" not in schemas
