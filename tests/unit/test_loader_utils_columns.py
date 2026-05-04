"""Tests for `fetch_columns_by_schema` in `sema.graph.loader_utils`."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.graph.loader_utils import fetch_columns_by_schema
from sema.models.extraction import ExtractedColumn

pytestmark = pytest.mark.unit


def test_fetch_columns_by_schema_returns_extracted_columns():
    loader = MagicMock()
    loader._run_read.return_value = [
        {
            "name": "patient_id", "table_name": "patient",
            "catalog": "workspace",
            "schema_name": "cbioportal_msk_chord_2024",
            "data_type": "string", "nullable": False,
            "comment": None,
        },
        {
            "name": "patient_id", "table_name": "sample",
            "catalog": "workspace",
            "schema_name": "cbioportal_msk_chord_2024",
            "data_type": "string", "nullable": True,
            "comment": "fk to patient.patient_id",
        },
    ]

    cols = fetch_columns_by_schema(
        loader, "cbioportal_msk_chord_2024",
    )

    assert len(cols) == 2
    assert all(isinstance(c, ExtractedColumn) for c in cols)
    assert cols[0].name == "patient_id"
    assert cols[0].table_name == "patient"
    assert cols[0].nullable is False
    assert cols[1].comment == "fk to patient.patient_id"


def test_fetch_columns_passes_schema_filter_to_query():
    loader = MagicMock()
    loader._run_read.return_value = []

    fetch_columns_by_schema(loader, "some_schema")

    loader._run_read.assert_called_once()
    _args, kwargs = loader._run_read.call_args
    assert kwargs.get("schema_name") == "some_schema"


def test_fetch_columns_returns_empty_list_when_no_results():
    loader = MagicMock()
    loader._run_read.return_value = []
    assert fetch_columns_by_schema(loader, "empty_schema") == []
