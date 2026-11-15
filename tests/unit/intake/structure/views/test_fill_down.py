"""fill_down: a continuation-row dictionary read as one a template can verify (Decision 4).

Pinned API, imported lazily so a missing module fails each test on its own:

- ``sema.intake.structure.views.VIEWS``: the closed tuple of view modules, shaped like ``TEMPLATES``.
- ``sema.intake.structure.views.View``: protocol with ``ID``, ``apply``, and ``verify``.
- ``sema.intake.structure.views.ViewSpec``: the recordable definition, ``view``, ``relation`` (the
  landed relation's ref name), ``columns``; it round-trips through JSON for the ledger.
- ``verify(spec, group, tools) -> Verdict``: checks the landed relation the spec names;
  fails with ``fill_down_anchor`` per filled column whose first landed row is null.
- ``apply(spec, group, tools) -> RelationProfile``: materializes the derived staging
  relation and returns its profile, ready for a template's ``verify``.
"""

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.land.landing import LOCATOR_COLUMN
from sema.intake.land.loaders.tabular import land_tabular
from tests.unit.intake.structure.shapes.shapes_env import Source, binding, staging_source, verify

pytestmark = pytest.mark.unit

FILL_DOWN = "fill_down"
ANCHOR = "fill_down_anchor"
DENORMALIZED = "concept_with_values_denormalized"
HEADER = ["variable", "description", "code", "label"]
# Physical rows 2..12: the last variable's continuation rows are 10..12, so an order by the
# locator's text (row 10 before row 2) differs from landed row order.
CONTINUATION_ROWS: tuple[tuple[int, str | None, str | None, str, str], ...] = (
    (2, "colour", "surface colour", "1", "red"),
    (3, None, None, "2", "green"),
    (4, None, None, "3", "blue"),
    (5, "size", "overall size", "1", "small"),
    (6, None, None, "2", "medium"),
    (7, None, None, "3", "large"),
    (8, None, None, "9", "unknown"),
    (9, "shape", "outline shape", "1", "round"),
    (10, None, None, "2", "square"),
    (11, None, None, "3", "oval"),
    (12, None, None, "9", "unknown"),
)
VARIABLE_BY_ROW = {2: "colour", 5: "size", 9: "shape"}
DESCRIPTION_BY_ROW = {2: "surface colour", 5: "overall size", 9: "outline shape"}
# The first landed row has no variable: nothing above it to fill from.
UNANCHORED_ROWS: tuple[tuple[int, str | None, str | None, str, str], ...] = (
    (2, None, "first description", "1", "red"),
    (3, "colour", None, "2", "green"),
    (4, None, None, "3", "blue"),
)


def views(name: str | None = None) -> ModuleType:
    return importlib.import_module("sema.intake.structure.views" + (f".{name}" if name else ""))


def _write_csv(path: Path, rows: Sequence[tuple[Any, ...]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for _, *cells in rows:
            writer.writerow(["" if cell is None else cell for cell in cells])
    return path


def _land(root: Path, stem: str, rows: Sequence[tuple[Any, ...]]) -> Iterator[Source]:
    root.mkdir(parents=True, exist_ok=True)
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = land_tabular(_write_csv(root / f"{stem}.csv", rows), staging, schema=f"land_{stem}")
    yield staging_source(staging, landed, root / "ledger" / "run")
    staging.close()


@pytest.fixture()
def dictionary(tmp_path: Path) -> Iterator[Source]:
    yield from _land(tmp_path / "dictionary", "codebook", CONTINUATION_ROWS)


@pytest.fixture()
def unanchored(tmp_path: Path) -> Iterator[Source]:
    yield from _land(tmp_path / "unanchored", "codebook", UNANCHORED_ROWS)


def _spec(source: Source, *columns: str) -> Any:
    relation = source.profiles()[0].ref.name
    return views().ViewSpec(view=FILL_DOWN, relation=relation, columns=columns)


def _rows(source: Source, relation: str) -> dict[int, dict[str, Any]]:
    """Every row of ``relation`` keyed by its landed physical row number."""
    staging = source.tools.provider.staging
    quoted = ".".join(f'"{part}"' for part in relation.split("."))
    cursor = staging.execute(f"SELECT * FROM {quoted}")
    names = [column[0] for column in cursor.description]
    by_row: dict[int, dict[str, Any]] = {}
    for values in cursor.fetchall():
        row = dict(zip(names, values, strict=True))
        by_row[int(str(row[LOCATOR_COLUMN]).rpartition("!")[2])] = row
    return by_row


def _filled(by_row: dict[int, str]) -> dict[int, str]:
    filled, current = {}, ""
    for number, *_ in CONTINUATION_ROWS:
        current = by_row.get(number, current)
        filled[number] = current
    return filled


def _apply(source: Source, *columns: str) -> Any:
    return views(FILL_DOWN).apply(_spec(source, *columns), list(source.profiles()), source.tools)


def test_views_lists_fill_down_only() -> None:
    registry = views()

    assert [view.ID for view in registry.VIEWS] == [FILL_DOWN]
    assert registry.VIEWS[0] is views(FILL_DOWN)
    assert callable(registry.VIEWS[0].apply) and callable(registry.VIEWS[0].verify)
    assert hasattr(registry, "View")


def test_view_spec_round_trips_through_json(dictionary: Source) -> None:
    spec = _spec(dictionary, "variable", "description")

    recorded = json.loads(json.dumps(spec.model_dump(mode="json")))

    assert recorded == {
        "view": FILL_DOWN,
        "relation": dictionary.profiles()[0].ref.name,
        "columns": ["variable", "description"],
    }
    assert views().ViewSpec.model_validate(recorded) == spec


def test_anchored_columns_verify(dictionary: Source) -> None:
    spec = _spec(dictionary, "variable", "description")

    verdict = views(FILL_DOWN).verify(spec, list(dictionary.profiles()), dictionary.tools)

    assert verdict.status == "pass"
    assert verdict.failures == ()


def test_each_null_takes_the_last_value_above_in_landed_row_order(dictionary: Source) -> None:
    view = _apply(dictionary, "variable", "description")

    rows = _rows(dictionary, view.ref.name)

    assert sorted(rows) == [number for number, *_ in CONTINUATION_ROWS]
    assert {n: r["variable"] for n, r in rows.items()} == _filled(VARIABLE_BY_ROW)
    assert {n: r["description"] for n, r in rows.items()} == _filled(DESCRIPTION_BY_ROW)
    assert {n: str(r["code"]) for n, r in rows.items()} == {
        number: code for number, _, _, code, _ in CONTINUATION_ROWS
    }


def test_only_named_columns_are_filled(dictionary: Source) -> None:
    view = _apply(dictionary, "variable")

    rows = _rows(dictionary, view.ref.name)

    assert {n: r["variable"] for n, r in rows.items()} == _filled(VARIABLE_BY_ROW)
    assert {n: r["description"] for n, r in rows.items()} == {
        number: DESCRIPTION_BY_ROW.get(number) for number, *_ in CONTINUATION_ROWS
    }


def test_view_is_a_derived_staging_relation_beside_the_landed_one(dictionary: Source) -> None:
    landed = dictionary.profiles()[0]
    before = _rows(dictionary, landed.ref.name)

    view = _apply(dictionary, "variable")

    assert view.ref.provider == "staging"
    assert view.ref.name != landed.ref.name
    assert view.row_count == landed.row_count == len(CONTINUATION_ROWS)
    assert [c.name for c in view.columns] == [c.name for c in landed.columns]
    assert _rows(dictionary, landed.ref.name) == before


def test_apply_is_repeatable(dictionary: Source) -> None:
    first = _apply(dictionary, "variable")

    second = _apply(dictionary, "variable")

    assert second.ref == first.ref
    assert _rows(dictionary, second.ref.name) == _rows(dictionary, first.ref.name)


def test_template_verifies_over_the_view_and_not_over_the_landed_relation(
    dictionary: Source,
) -> None:
    landed = dictionary.profiles()[0]
    view = _apply(dictionary, "variable")
    over_view = binding(DENORMALIZED, [view.ref.name], relation=view.ref.name, group_by="variable")
    over_landed = binding(
        DENORMALIZED, [landed.ref.name], relation=landed.ref.name, group_by="variable"
    )

    passed = verify(DENORMALIZED, over_view, [view], dictionary)
    refused = verify(DENORMALIZED, over_landed, [landed], dictionary)

    assert passed.status == "pass", passed.failures
    assert refused.status == "fail"
    assert "group_by_non_null" in {f.check for f in refused.failures}


def test_unanchored_column_is_refused_with_fill_down_anchor(unanchored: Source) -> None:
    spec = _spec(unanchored, "variable")

    verdict = views(FILL_DOWN).verify(spec, list(unanchored.profiles()), unanchored.tools)

    assert verdict.status == "fail"
    assert [(f.check, f.relation, f.column) for f in verdict.failures] == [
        (ANCHOR, spec.relation, "variable")
    ]


def test_anchor_is_checked_per_filled_column(unanchored: Source) -> None:
    spec = _spec(unanchored, "description", "variable")

    verdict = views(FILL_DOWN).verify(spec, list(unanchored.profiles()), unanchored.tools)

    assert verdict.status == "fail"
    assert [(f.check, f.column) for f in verdict.failures] == [(ANCHOR, "variable")]


def test_apply_refuses_an_unanchored_spec_with_its_failures(unanchored: Source) -> None:
    spec = _spec(unanchored, "variable")

    with pytest.raises(views().ViewRefusedError) as refused:
        views(FILL_DOWN).apply(spec, list(unanchored.profiles()), unanchored.tools)

    assert [(f.check, f.column) for f in refused.value.failures] == [(ANCHOR, "variable")]


def test_a_view_over_a_warehouse_relation_is_refused(dictionary: Source) -> None:
    from sema.binding.spec_sections import RelationRef

    landed = dictionary.profiles()[0]
    remote = replace(landed, ref=RelationRef(provider="warehouse", name=landed.ref.name))
    spec = _spec(dictionary, "variable")

    verdict = views(FILL_DOWN).verify(spec, [remote], dictionary.tools)
    with pytest.raises(views().ViewRefusedError, match="staging") as refused:
        views(FILL_DOWN).apply(spec, [remote], dictionary.tools)

    assert verdict.status == "fail"
    assert [f.check for f in verdict.failures] == ["fill_down_staging"]
    assert [f.check for f in refused.value.failures] == ["fill_down_staging"]
