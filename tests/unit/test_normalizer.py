"""Tests for AssertionNormalizer: DTO -> assertion conversion."""

import pytest

from sema.models.assertions import AssertionPredicate
from sema.models.extraction import (
    ExtractedColumn,
    ExtractedForeignKey,
    ExtractedSampleRows,
    ExtractedTable,
    ExtractedTag,
    ExtractedTopValues,
)
from sema.pipeline.normalizer import AssertionNormalizer

pytestmark = pytest.mark.unit


@pytest.fixture
def normalizer() -> AssertionNormalizer:
    return AssertionNormalizer(
        workspace="ws.example.com",
        run_id="run-1",
        platform="databricks",
    )


class TestTableNormalization:
    def test_table_exists_assertion(self, normalizer: AssertionNormalizer) -> None:
        table = ExtractedTable(name="patients", catalog="cat", schema="sch")
        assertions = normalizer.normalize_table(table)
        assert len(assertions) == 1
        a = assertions[0]
        assert a.predicate == AssertionPredicate.TABLE_EXISTS
        assert a.subject_ref == "databricks://ws.example.com/cat/sch/patients"

    def test_table_with_comment(self, normalizer: AssertionNormalizer) -> None:
        table = ExtractedTable(
            name="patients", catalog="cat", schema="sch",
            comment="Patient records",
        )
        assertions = normalizer.normalize_table(table)
        assert len(assertions) == 2
        comments = [a for a in assertions if a.predicate == AssertionPredicate.HAS_COMMENT]
        assert len(comments) == 1
        assert comments[0].payload["value"] == "Patient records"


class TestColumnNormalization:
    def test_column_exists_and_datatype(self, normalizer: AssertionNormalizer) -> None:
        col = ExtractedColumn(
            name="patient_id", table_name="patients",
            catalog="cat", schema="sch", data_type="INT",
        )
        assertions = normalizer.normalize_column(col)
        predicates = {a.predicate for a in assertions}
        assert AssertionPredicate.COLUMN_EXISTS in predicates
        assert AssertionPredicate.HAS_DATATYPE in predicates
        col_ref = "databricks://ws.example.com/cat/sch/patients/patient_id"
        assert all(a.subject_ref == col_ref for a in assertions)

    def test_column_with_comment(self, normalizer: AssertionNormalizer) -> None:
        col = ExtractedColumn(
            name="dx_code", table_name="tbl",
            catalog="cat", schema="sch", data_type="STRING",
            comment="Diagnosis code",
        )
        assertions = normalizer.normalize_column(col)
        comments = [a for a in assertions if a.predicate == AssertionPredicate.HAS_COMMENT]
        assert len(comments) == 1


class TestForeignKeyNormalization:
    def test_joins_to_assertion(self, normalizer: AssertionNormalizer) -> None:
        fk = ExtractedForeignKey(
            from_table="encounters", from_columns=["patient_id"],
            to_table="patients", to_columns=["id"],
            from_catalog="cat", from_schema="sch",
            to_catalog="cat", to_schema="sch",
        )
        assertions = normalizer.normalize_foreign_key(fk)
        assert len(assertions) == 1
        a = assertions[0]
        assert a.predicate == AssertionPredicate.JOINS_TO
        assert "patient_id" in a.payload["on_column"]
        assert a.object_ref == "databricks://ws.example.com/cat/sch/patients"


class TestTopValuesNormalization:
    def test_top_values_assertion(self, normalizer: AssertionNormalizer) -> None:
        tv = ExtractedTopValues(
            column_name="status", table_name="tbl",
            catalog="cat", schema="sch",
            values=[{"value": "active"}, {"value": "inactive"}],
            approx_distinct=2,
        )
        assertions = normalizer.normalize_top_values(tv)
        assert len(assertions) == 1
        a = assertions[0]
        assert a.predicate == AssertionPredicate.HAS_TOP_VALUES
        assert len(a.payload["values"]) == 2


class TestSampleRowsNormalization:
    def test_sample_rows_assertion(self, normalizer: AssertionNormalizer) -> None:
        sr = ExtractedSampleRows(
            table_name="tbl", catalog="cat", schema="sch",
            rows=[["1", "John"], ["2", "Jane"]],
            column_names=["id", "name"],
        )
        assertions = normalizer.normalize_sample_rows(sr)
        assert len(assertions) == 1
        a = assertions[0]
        assert a.predicate == AssertionPredicate.HAS_SAMPLE_ROWS
        assert a.subject_ref == "databricks://ws.example.com/cat/sch/tbl"


class TestTagNormalization:
    def test_column_tag(self, normalizer: AssertionNormalizer) -> None:
        tag = ExtractedTag(
            table_name="tbl", column_name="col",
            tag_key="pii", tag_value="true",
            catalog="cat", schema="sch",
        )
        assertions = normalizer.normalize_tag(tag)
        assert len(assertions) == 1
        assert assertions[0].subject_ref.endswith("/col")

    def test_table_tag(self, normalizer: AssertionNormalizer) -> None:
        tag = ExtractedTag(
            table_name="tbl", column_name=None,
            tag_key="domain", tag_value="clinical",
            catalog="cat", schema="sch",
        )
        assertions = normalizer.normalize_tag(tag)
        assert len(assertions) == 1
        assert assertions[0].subject_ref.endswith("/tbl")
