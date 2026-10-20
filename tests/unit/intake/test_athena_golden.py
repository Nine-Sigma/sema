"""Athena bundle staging output stays byte-identical across the loader move."""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging
from tests.support.athena_bundle import export_schema, make_minimal_bundle

pytestmark = pytest.mark.unit

GOLDEN = Path(__file__).parents[2] / "fixtures" / "intake" / "athena" / "golden_export.csv"


def showcase_export(tmp_path: Path) -> str:
    from showcase.cbioportal_to_omop.omop_ingest import ingest_vocabulary

    bundle = tmp_path / "athena"
    make_minimal_bundle(bundle, with_synonyms=True)
    with Staging(str(tmp_path / "vocab.duckdb")) as staging:
        ingest_vocabulary(bundle, staging)
        return export_schema(staging)


def test_showcase_ingest_vocabulary_export_equals_golden(tmp_path: Path) -> None:
    assert showcase_export(tmp_path) == GOLDEN.read_text()
