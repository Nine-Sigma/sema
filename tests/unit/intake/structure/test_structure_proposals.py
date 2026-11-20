"""Structure output: proposal targets and the shape refusal (S3 Decision 4).

``StructureResult.proposal_targets`` is a serialized field: the qualified names of the
``no_template`` relations and of the relations ignored as ``no_header``, in landing order.
A header-less relation stays in the ignore partition while it is listed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.intake.land.loaders.tabular import land_tabular
from tests.unit.intake.structure.shapes.shapes_env import Source, warehouse_source
from tests.unit.intake.structure_env import (
    assert_partition,
    fresh_tools,
    land,
    reasons,
    ref,
    run,
    structure,
)

pytestmark = pytest.mark.unit

NO_TEMPLATE = "no_template"
# A numeric-only grid passes no header test, so it lands with positional columns.
GRID = "a_grid"
GRID_CSV = "10,20,30\n11,21,31\n12,22,32\n"
# One repeating label column: no ignore profile and no template matches it.
NOTES = "b_notes"
NOTES_CSV = "note\nalpha\nalpha\nbeta\n"
# Integer ids beside labels: a concept list that is not a dictionary.
ITEMS = "c_items"
ITEMS_CSV = "id,label\n1,First item\n2,Second item\n3,Third item\n"


@pytest.fixture
def mixed(tmp_path: Path) -> Iterator[Source]:
    directory = tmp_path / "csv"
    directory.mkdir()
    for name, text in ((GRID, GRID_CSV), (NOTES, NOTES_CSV), (ITEMS, ITEMS_CSV)):
        (directory / f"{name}.csv").write_text(text)
    staging, source = land(tmp_path / "landed", "mixed", land_tabular, directory)
    yield source
    staging.close()


def _structured(source: Source, tmp_path: Path) -> Any:
    return run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))


def test_proposal_targets_is_a_serialized_field() -> None:
    model = structure().StructureResult

    assert "proposal_targets" in model.model_fields


def test_headerless_relation_is_ignored_and_listed_as_a_proposal_target(
    mixed: Source, tmp_path: Path
) -> None:
    grid = ref(mixed, GRID)

    result = _structured(mixed, tmp_path)

    assert mixed.land.headerless == (GRID,)
    assert reasons(result)[grid] == "no_header"
    assert grid in result.proposal_targets
    assert_partition(mixed, result)


def test_no_template_relation_is_ambiguous_and_listed_as_a_proposal_target(
    mixed: Source, tmp_path: Path
) -> None:
    notes = ref(mixed, NOTES)

    result = _structured(mixed, tmp_path)

    (ambiguity,) = [a for a in result.ambiguities if a.subject == notes]
    assert ambiguity.members == (NO_TEMPLATE,)
    assert notes in result.proposal_targets


def test_proposal_targets_hold_only_those_relations_in_landing_order(
    mixed: Source, tmp_path: Path
) -> None:
    targets = {ref(mixed, GRID), ref(mixed, NOTES)}
    landed = [ref(mixed, relation.name) for relation in mixed.land.relations]

    result = _structured(mixed, tmp_path)

    expected = tuple(name for name in landed if name in targets)
    assert result.proposal_targets == expected
    assert ref(mixed, ITEMS) not in result.proposal_targets


def test_proposal_targets_survive_a_round_trip(mixed: Source, tmp_path: Path) -> None:
    result = _structured(mixed, tmp_path)

    model = structure().StructureResult
    restored = model.model_validate_json(result.model_dump_json())

    assert restored.proposal_targets == result.proposal_targets


def test_shape_is_refused_while_a_section_is_unresolved(tmp_path: Path) -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA ci")
    con.execute(
        "CREATE TABLE ci.items AS SELECT * FROM (VALUES (1, 'First item'),"
        " (2, 'Second item'), (3, 'Third item')) t(id, label)"
    )
    source = warehouse_source(con, "ci", tmp_path / "ledger" / "landing")
    result = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))
    con.close()

    assert [entry.binding.template for entry in result.bound] == ["concept_list"]
    with pytest.raises(ValueError):
        result.to_shape()
