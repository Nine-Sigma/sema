"""A10: the whole ACRIS directory lands, 15 workbooks of 4 sheets and 5 code tables."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

ACRIS = Path(__file__).parents[2] / "fixtures" / "intake" / "acris"
SCHEMA = "land_acris"
WORKBOOKS = 15
CODE_TABLE_ROWS = {
    "Country_Codes": 250,
    "Document_Control_Codes": 126,
    "Property_Types_Codes": 49,
    "State_Codes": 54,
    "UCC_Collateral_Codes": 8,
}
COLUMN_INFO_NAMES = [
    "Column Name",
    "Column Description",
    "Term, Acronym, or Code Definitions",
    "Additional Notes (where applicable, include the range of possible values, units"
    " of measure, how to interpret null/zero values, whether there are specific"
    " relationships between columns, and information on column source)",
]


@pytest.fixture(scope="module")
def landed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    from sema.intake.loaders.tabular import land_tabular

    path = tmp_path_factory.mktemp("acris") / "staging.duckdb"
    with Staging(str(path), schemas=()) as staging:
        yield land_tabular(ACRIS, staging, schema=SCHEMA)


def _sheet(landed: Any, suffix: str) -> list[Any]:
    return [r for r in landed.relations if r.name.endswith(f"__{suffix}")]


def _names(relation: Any) -> list[str]:
    return [name for name, _ in relation.columns if name != "_sema_locator"]


def test_every_sheet_and_code_table_lands(landed: Any) -> None:
    assert len(landed.relations) == 65 == WORKBOOKS * 4 + len(CODE_TABLE_ROWS)
    assert landed.headerless == ()


def test_column_info_header_is_row_2_with_collapsed_names(landed: Any) -> None:
    column_info = _sheet(landed, "Column_Info")

    assert len(column_info) == WORKBOOKS
    assert {r.header_row for r in column_info} == {2}
    assert all(_names(r) == COLUMN_INFO_NAMES for r in column_info)


def test_dataset_info_header_is_row_5(landed: Any) -> None:
    dataset_info = _sheet(landed, "Dataset_Info")

    assert len(dataset_info) == WORKBOOKS
    assert {r.header_row for r in dataset_info} == {5}


def test_dropdown_options_header_is_row_2_with_21_rows(landed: Any) -> None:
    dropdowns = _sheet(landed, "Dropdown_Options")

    assert len(dropdowns) == WORKBOOKS
    assert {(r.header_row, r.row_count) for r in dropdowns} == {(2, 21)}


def test_revision_history_lands_empty_without_its_trailing_columns(landed: Any) -> None:
    revisions = _sheet(landed, "Dataset_Revision_History")

    assert len(revisions) == WORKBOOKS
    assert {r.row_count for r in revisions} == {0}
    assert {len(_names(r)) for r in revisions} == {4}
    assert all(r.empty_columns == () for r in revisions)


def test_no_relation_has_positional_columns(landed: Any) -> None:
    positional = [
        (r.name, name) for r in landed.relations for name in _names(r)
        if name.startswith("_col_")
    ]

    assert positional == []


def test_code_tables_land_with_header_row_1_and_their_row_counts(landed: Any) -> None:
    counts = {name: landed.relation(name).row_count for name in CODE_TABLE_ROWS}

    assert counts == CODE_TABLE_ROWS
    assert {landed.relation(name).header_row for name in CODE_TABLE_ROWS} == {1}
