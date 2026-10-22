"""The accuracy scorer over synthetic imports (design decision 1, generality-gate spec).

Every input is an invented manifest mapping and term staging export text; the cell kinds
are the import's own names and the term tables are the store's.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from sema.intake.scorecard import KindScore, Scorecard, render, score
from tests.unit.intake.scorecard_env import (
    CONCEPT,
    CREW,
    ENTITY_CASES,
    HARBOR,
    PROPERTY_CASES,
    RELATIONSHIP,
    SYNONYM,
    VOYAGE,
    Floor,
    Table,
    change_definitions,
    concept_row,
    entity,
    entity_of,
    export,
    fk,
    floor_misses,
    manifest,
    oracle_manifest,
    oracle_tables,
    prop,
    prop_of,
)

pytestmark = pytest.mark.unit

FIXED_KINDS = (
    "entities", "type", "nullable", "definition", "decoded_values", "required_fields",
    "primary_key", "foreign_keys", "domain_constraints", "declared",
)
TERM_KINDS = (f"terms:{CONCEPT}", f"terms:{RELATIONSHIP}", f"terms:{SYNONYM}")
EQUAL_CELLS = {
    "entities": 3, "type": 8, "nullable": 8, "definition": 3, "decoded_values": 1,
    "required_fields": 2, "primary_key": 2, "foreign_keys": 1, "domain_constraints": 1,
    "declared": 1, f"terms:{CONCEPT}": 3, f"terms:{RELATIONSHIP}": 1, f"terms:{SYNONYM}": 1,
}
ALL_CELLS = sum(EQUAL_CELLS.values())

Manifest = dict[str, Any]
Tables = dict[str, Table]


def _changed(change: Callable[[Manifest], None]) -> Manifest:
    built = copy.deepcopy(oracle_manifest())
    change(built)
    return built


def _score_manifest(assembled: Manifest) -> Scorecard:
    staged = export(oracle_tables())
    return score(oracle_manifest(), assembled, staged, staged)


def _score_terms(assembled: Tables) -> Scorecard:
    return score(oracle_manifest(), oracle_manifest(), export(oracle_tables()), export(assembled))


def _counts(kind: KindScore) -> tuple[int, int, int, int]:
    return kind.equal, kind.missing, kind.wrong, kind.extra


def _port_type(built: Manifest) -> None:
    prop_of(built, VOYAGE, "port")["type"] = "integer"


def _identical() -> Scorecard:
    return _score_manifest(oracle_manifest())


def test_identical_imports_score_one_on_every_kind() -> None:
    card = _identical()

    assert set(card.kinds) == {*FIXED_KINDS, *TERM_KINDS}
    for name, kind in card.kinds.items():
        assert (kind.precision, kind.recall) == (1.0, 1.0), name
        assert _counts(kind) == (EQUAL_CELLS[name], 0, 0, 0), name
    assert (card.precision, card.recall) == (1.0, 1.0)


def test_one_differing_cell_is_wrong_and_nothing_else_moves() -> None:
    card = _score_manifest(_changed(_port_type))
    identical = _identical()

    assert _counts(card.kinds["type"]) == (7, 0, 1, 0)
    assert card.kinds["type"].precision == pytest.approx(7 / 8)
    assert card.kinds["type"].recall == pytest.approx(7 / 8)
    assert {k: v for k, v in card.kinds.items() if k != "type"} == {
        k: v for k, v in identical.kinds.items() if k != "type"
    }


def _swap_crew_for_harbor(built: Manifest) -> None:
    built["entities"] = [e for e in built["entities"] if e["qualified_name"] != CREW]
    built["entities"].append(entity(HARBOR, prop("berth")))


def test_entity_set_scores_by_qualified_name() -> None:
    card = _score_manifest(_changed(_swap_crew_for_harbor))

    assert _counts(card.kinds["entities"]) == (2, 1, 0, 1)
    assert card.kinds["entities"].precision == pytest.approx(2 / 3)
    assert card.kinds["entities"].recall == pytest.approx(2 / 3)


def test_an_absent_entity_makes_its_cells_missing_and_an_extra_one_its_cells_extra() -> None:
    card = _score_manifest(_changed(_swap_crew_for_harbor))

    assert _counts(card.kinds["type"]) == (6, 2, 0, 1)
    assert _counts(card.kinds["nullable"]) == (6, 2, 0, 1)
    assert _counts(card.kinds["declared"]) == (0, 1, 0, 0)


@pytest.mark.parametrize(("change", "kind", "counts"), PROPERTY_CASES)
def test_property_cells_score_per_entity_property(
    change: Callable[[Manifest], None], kind: str, counts: tuple[int, int, int, int]
) -> None:
    assert _counts(_score_manifest(_changed(change)).kinds[kind]) == counts


def test_precision_and_recall_count_each_verdict_on_its_side() -> None:
    definition = _score_manifest(_changed(change_definitions)).kinds["definition"]

    assert definition.precision == pytest.approx(1 / 3)
    assert definition.recall == pytest.approx(1 / 3)


@pytest.mark.parametrize(("change", "expected"), ENTITY_CASES)
def test_entity_set_cells_score_per_entity(
    change: Callable[[Manifest], None], expected: dict[str, tuple[int, int, int, int]]
) -> None:
    card = _score_manifest(_changed(change))

    assert {kind: _counts(card.kinds[kind]) for kind in expected} == expected


def test_foreign_keys_compare_as_a_set_of_referenced_entity_and_join_keys() -> None:
    extra = fk(CREW, ("hull_code", "hull_code"))

    def add(at: int) -> Callable[[Manifest], None]:
        return lambda m: entity_of(m, VOYAGE)["obligation"]["foreign_keys"].insert(at, extra)

    staged = export(oracle_tables())
    card = score(_changed(add(1)), _changed(add(0)), staged, staged)

    assert _counts(card.kinds["foreign_keys"]) == (1, 0, 0, 0)


def _replace(table: str, rows: tuple[tuple[str, ...], ...]) -> Tables:
    tables = oracle_tables()
    tables[table] = Table(tables[table].header, rows)
    return tables


def _concepts(*pairs: tuple[str, str]) -> Tables:
    return _replace(CONCEPT, tuple(concept_row(code, name) for code, name in pairs))


TERM_CASES: list[Any] = [
    pytest.param(
        _concepts(("S", "Sloop"), ("K", "Cutter"), ("Y", "Yawl")),
        CONCEPT, (2, 0, 1, 0), id="concept-key-kept-value-wrong",
    ),
    pytest.param(
        _concepts(("S", "Sloop"), ("Q", "Ketch"), ("Y", "Yawl")),
        CONCEPT, (2, 1, 0, 1), id="concept-key-changed",
    ),
    pytest.param(
        _replace(RELATIONSHIP, (("rig_kinds:Y", "rig_kinds:K", "maps_to"),)),
        RELATIONSHIP, (0, 1, 0, 1), id="relationship-every-column-is-key",
    ),
    pytest.param(
        _replace(SYNONYM, (("rig_kinds:S", "Single mast", "fr"),)),
        SYNONYM, (0, 1, 0, 1), id="other-table-full-row",
    ),
    pytest.param(
        {k: v for k, v in oracle_tables().items() if k != SYNONYM},
        SYNONYM, (0, 1, 0, 0), id="table-absent",
    ),
    pytest.param(
        _replace(SYNONYM, (("rig_kinds:S", "Single mast", "en"),) * 2),
        SYNONYM, (1, 0, 0, 1), id="duplicate-row-is-its-own-cell",
    ),
]


@pytest.mark.parametrize(("assembled", "table", "counts"), TERM_CASES)
def test_term_rows_score_per_table_by_its_key_columns(
    assembled: Tables, table: str, counts: tuple[int, int, int, int]
) -> None:
    card = _score_terms(assembled)

    assert _counts(card.kinds[f"terms:{table}"]) == counts
    assert {k: v for k, v in card.kinds.items() if k != f"terms:{table}"} == {
        k: v for k, v in _identical().kinds.items() if k != f"terms:{table}"
    }


def test_rows_sharing_a_key_are_each_a_cell() -> None:
    shared = export(_concepts(("S", "Sloop"), ("S", "Schooner"), ("K", "Ketch"), ("Y", "Yawl")))
    card = score(oracle_manifest(), oracle_manifest(), shared, shared)

    assert _counts(card.kinds[f"terms:{CONCEPT}"]) == (4, 0, 0, 0)


def test_a_table_only_the_assembled_import_stages_scores_extra_rows() -> None:
    tables = oracle_tables() | {"term_note": Table(("note:VARCHAR",), (("a",), ("b",)))}

    assert _counts(_score_terms(tables).kinds["terms:term_note"]) == (0, 0, 0, 2)


def test_a_zero_denominator_scores_one() -> None:
    dropped = _score_terms(_replace(RELATIONSHIP, ())).kinds[f"terms:{RELATIONSHIP}"]
    empty = score(manifest(), manifest(), "", "")

    assert (dropped.precision, dropped.recall) == (1.0, 0.0)
    assert set(empty.kinds) == set(FIXED_KINDS)
    for name, kind in empty.kinds.items():
        assert (kind.precision, kind.recall, _counts(kind)) == (1.0, 1.0, (0, 0, 0, 0)), name
    assert (empty.precision, empty.recall) == (1.0, 1.0)


def test_overall_numbers_are_over_every_cell_of_every_kind() -> None:
    wrong = _score_manifest(_changed(_port_type))
    missing = _score_terms(_replace(RELATIONSHIP, ()))
    entities = _score_manifest(_changed(lambda m: m["entities"].append(entity(HARBOR))))

    assert (wrong.precision, wrong.recall) == pytest.approx(
        ((ALL_CELLS - 1) / ALL_CELLS, (ALL_CELLS - 1) / ALL_CELLS)
    )
    assert (missing.precision, missing.recall) == pytest.approx(
        (1.0, (ALL_CELLS - 1) / ALL_CELLS)
    )
    assert (entities.precision, entities.recall) == pytest.approx(
        (ALL_CELLS / (ALL_CELLS + 1), 1.0)
    )


def _row(text: str, name: str) -> list[str]:
    (line,) = [line for line in text.splitlines() if line.split()[:1] == [name]]
    return line.split()


def test_render_prints_one_row_per_kind_and_an_overall_row() -> None:
    card = _score_manifest(_changed(change_definitions))
    text = render(card)

    for name, kind in card.kinds.items():
        cells = _row(text, name)
        assert float(cells[1]) == pytest.approx(kind.precision, abs=0.01), name
        assert float(cells[2]) == pytest.approx(kind.recall, abs=0.01), name
        assert [int(cell) for cell in cells[3:7]] == list(_counts(kind)), name
    overall = _row(text, "overall")
    assert float(overall[1]) == pytest.approx(card.precision, abs=0.01)
    assert float(overall[2]) == pytest.approx(card.recall, abs=0.01)
    for column in ("precision", "recall", "equal", "missing", "wrong", "extra"):
        assert column in text


def test_scorecard_and_kind_score_are_frozen() -> None:
    card = _identical()

    with pytest.raises(ValidationError):
        setattr(card, "precision", 0.0)
    with pytest.raises(ValidationError):
        setattr(card.kinds["type"], "equal", 0)


FLOOR = Floor(precision=0.9, recall=0.8, ignored=1)


def test_floor_misses_is_empty_when_the_leg_holds_its_floor() -> None:
    held = Scorecard(kinds={}, precision=0.9, recall=0.8)

    assert floor_misses(held, 1, FLOOR) == ()


@pytest.mark.parametrize(
    ("precision", "recall", "ignored", "named"),
    [
        (0.89, 0.8, 1, ("precision below",)),
        (0.9, 0.79, 1, ("recall below",)),
        (0.9, 0.8, 2, ("ignored above",)),
        (0.91, 0.8, 1, ("precision above",)),
        (0.9, 0.81, 1, ("recall above",)),
        (0.9, 0.8, 0, ("ignored below",)),
        (0.5, 0.5, 4, ("precision below", "recall below", "ignored above")),
    ],
)
def test_floor_misses_names_each_deviation_either_way(
    precision: float, recall: float, ignored: int, named: tuple[str, ...]
) -> None:
    card = Scorecard(kinds={}, precision=precision, recall=recall)

    misses = floor_misses(card, ignored, FLOOR)

    assert tuple(" ".join(miss.split()[0:4:3]) for miss in misses) == named
