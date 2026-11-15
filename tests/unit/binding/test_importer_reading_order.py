"""Reading order is the landed physical row order, never the provider's scan order."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from sema.binding.spec import ConceptSourceSpec, RelationRef
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.landing import LOCATOR_COLUMN
from sema.intake.land.relations import RelationProvider
from tests.unit.binding.fixtures.concept_library_v2 import captured_log

pytestmark = pytest.mark.unit

LABEL = "book.xlsx!Columns"
# Insertion order differs from both the numeric locator order and the string
# order, where "!10" sorts before "!6".
ROWS = (("6", "gamma"), ("10", "delta"), ("2", "alpha"), ("3", "beta"))
LOCATOR_ORDER = ["alpha", "beta", "gamma", "delta"]
UNLOCATED = (("b", "second"), ("c", "third"), ("a", "first"))


def _create(
    staging: Staging, name: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    staging.execute("CREATE SCHEMA IF NOT EXISTS land")
    quoted = ", ".join(f'"{column}" VARCHAR' for column in columns)
    staging.execute(f'CREATE TABLE land."{name}" ({quoted})')
    for row in rows:
        marks = ", ".join("?" for _ in row)
        staging.execute(f'INSERT INTO land."{name}" VALUES ({marks})', list(row))


def _located(rows: Sequence[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return [(*values, f"{LABEL}!{number}") for number, *values in rows]


@pytest.fixture
def provider(tmp_path: Path) -> Iterator[RelationProvider]:
    with Staging(str(tmp_path / "order.duckdb"), schemas=()) as staging:
        _create(staging, "located", ("label", LOCATOR_COLUMN), _located(ROWS))
        _create(staging, "unlocated", ("code", "label"), UNLOCATED)
        yield RelationProvider(warehouse=None, staging=staging)


def _ref(name: str) -> RelationRef:
    return RelationRef(provider="staging", name=f"land.{name}")


def _labels(provider: RelationProvider, name: str, where: str | None = None) -> list[Any]:
    from sema.binding.importer_utils import fetch_in_order

    return [row["label"] for row in fetch_in_order(provider, _ref(name), ["label"], where)]


def test_located_relation_reads_in_numeric_locator_order(provider: RelationProvider) -> None:
    assert _labels(provider, "located") == LOCATOR_ORDER


def test_locator_order_holds_under_a_filter(provider: RelationProvider) -> None:
    assert _labels(provider, "located", "label <> 'beta'") == ["alpha", "gamma", "delta"]


def test_unlocated_relation_reads_in_selected_column_order_and_logs_once(
    provider: RelationProvider,
) -> None:
    from sema.binding.importer_utils import fetch_in_order

    with captured_log() as messages:
        rows = fetch_in_order(provider, _ref("unlocated"), ["code", "label"])

    assert [row["code"] for row in rows] == ["a", "b", "c"]
    logged = [m for m in messages if "land.unlocated" in m and "row order" in m]
    assert len(logged) == 1


def test_header_token_tie_follows_locator_order(tmp_path: Path) -> None:
    from sema.binding.importer_entity_utils import read_header

    body = (
        ("3", "Report Title", "Inserted first"),
        ("2", "Dataset Name", "Physically first"),
        ("4", "Summary", "One row per shipment."),
    )
    with Staging(str(tmp_path / "header.duckdb"), schemas=()) as staging:
        _create(staging, "header", ("Field", "Value", LOCATOR_COLUMN), _located(body))
        header = read_header(
            RelationProvider(warehouse=None, staging=staging), _ref("header"), None
        )

    assert header.name == "Physically first"
    assert "Report Title" in header.prose
    assert [key for key, _ in header.pairs] == [
        "Field",
        "Dataset Name",
        "Report Title",
        "Summary",
    ]


def _per_file_spec() -> ConceptSourceSpec:
    return ConceptSourceSpec.model_validate(
        {
            "spec_version": 2,
            "id": "ex-order",
            "identity_system": "ex",
            "release": "1",
            "default_provider": "staging",
            "entity_model": {
                "layout": "per_file",
                "entities": [{"relation": "land.columns", "entity": "orders"}],
                "roles": {"property": "field", "description": "meaning"},
            },
        }
    )


def test_per_file_properties_follow_locator_order(tmp_path: Path) -> None:
    from sema.binding.importer import import_concept_source

    body = (
        ("10", "status", "Order state"),
        ("2", "order_id", "Order key"),
        ("6", "placed_at", "When the order was placed"),
    )
    with Staging(str(tmp_path / "per_file.duckdb"), schemas=()) as staging:
        _create(staging, "columns", ("field", "meaning", LOCATOR_COLUMN), _located(body))
        result = import_concept_source(
            _per_file_spec(),
            RelationProvider(warehouse=None, staging=staging),
            dataset_map={},
        )

    fields = [field.name for field in result.table_schemas["ex.orders"].fields]
    assert fields == ["order_id", "placed_at", "status"]
