import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.connectors.databricks import DatabricksConnector, TableWorkItem
from sema.models.assertions import AssertionPredicate
from sema.models.config import DatabricksConfig, ProfilingConfig


@pytest.fixture
def mock_cursor():
    cursor = MagicMock()
    cursor.description = []
    cursor.fetchall = MagicMock(return_value=[])
    return cursor


@pytest.fixture
def mock_connection(mock_cursor):
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def connector(mock_connection):
    with patch("sema.connectors.databricks.sql_connect") as mock_connect:
        mock_connect.return_value = mock_connection
        config = DatabricksConfig(
            host="https://test.databricks.com",
            token="dapi123",
            http_path="/sql/1.0/warehouses/test",
        )
        conn = DatabricksConnector(config=config, profiling=ProfilingConfig())
        conn._connection = mock_connection
        yield conn


class TestCatalogDiscovery:
    def test_list_catalogs(self, connector, mock_cursor):
        mock_cursor.fetchall.return_value = [("catalog_a",), ("catalog_b",)]
        result = connector.list_catalogs()
        assert result == ["catalog_a", "catalog_b"]

    def test_discover_schemas(self, connector, mock_cursor):
        mock_cursor.fetchall.return_value = [("clinical",), ("staging",)]
        result = connector._discover_schemas("cdm")
        assert result == ["clinical", "staging"]


class TestTableExtraction:
    def test_extract_emits_table_exists(self, connector, mock_cursor):
        # SHOW TABLES returns table list
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],  # SHOW SCHEMAS
            [("cbioportal_omop", "cancer_diagnosis", False)],  # SHOW TABLES
            # information_schema.columns
            [("dx_type_cd", "STRING", "YES", "Diagnosis type code")],
            [],  # FK constraints
            [],  # tags
            [(10,)],  # APPROX_COUNT_DISTINCT for dx_type_cd
            [("CRC", 100), ("BRCA", 80)],  # top-k values
            [("CRC", "Stage III", "2024-01-15")],  # sample rows
        ]
        mock_cursor.description = [("col1",), ("col2",), ("col3",)]

        assertions = connector.extract(catalog="cdm")

        table_assertions = [a for a in assertions if a.predicate == AssertionPredicate.TABLE_EXISTS]
        assert len(table_assertions) == 1
        assert table_assertions[0].subject_ref == "unity://cdm.clinical.cancer_diagnosis"
        assert table_assertions[0].source == "unity_catalog"
        assert table_assertions[0].payload["table_type"] == "TABLE"

    def test_extract_emits_column_exists(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("cbioportal_omop", "cancer_diagnosis", False)],
            [("dx_type_cd", "STRING", "YES", "Diagnosis type code"),
             ("patient_id", "STRING", "NO", None)],
            [], [],
            [(10,)], [("CRC", 100)],  # dx_type_cd profiling
            [(50000,)],  # patient_id high cardinality - skip sampling
            [("CRC", "P1")],  # sample rows
        ]
        mock_cursor.description = [("col1",), ("col2",)]

        assertions = connector.extract(catalog="cdm")

        col_assertions = [a for a in assertions if a.predicate == AssertionPredicate.COLUMN_EXISTS]
        assert len(col_assertions) == 2
        dx_col = next(a for a in col_assertions if "dx_type_cd" in a.subject_ref)
        assert dx_col.payload["data_type"] == "STRING"
        assert dx_col.payload["nullable"] is True
        assert dx_col.payload["comment"] == "Diagnosis type code"

    def test_extract_emits_datatype(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("cbioportal_omop", "cancer_diagnosis", False)],
            [("date_col", "DATE", "YES", None)],
            [], [],
            [(5000,)],  # high cardinality - date column
            [("2024-01-01", "2024-01-02")],  # sample rows
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        dt_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_DATATYPE]
        assert len(dt_assertions) == 1
        assert dt_assertions[0].payload["value"] == "DATE"


class TestConstraintExtraction:
    def test_fk_constraints_emit_joins_to(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("cbioportal_omop", "cancer_diagnosis", False)],
            [("patient_id", "STRING", "NO", None)],
            # FK constraints: from_table, from_col, to_table, to_col
            [("cancer_diagnosis", "patient_id", "patients", "patient_id")],
            [],  # tags
            [(50000,)],  # high cardinality
            [("P1",)],  # sample rows
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        join_assertions = [a for a in assertions if a.predicate == AssertionPredicate.JOINS_TO]
        assert len(join_assertions) == 1
        assert join_assertions[0].object_ref is not None
        assert "patients" in join_assertions[0].object_ref
        assert join_assertions[0].payload["on_column"] == "patient_id"

    def test_no_fk_constraints(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("cbioportal_omop", "cancer_diagnosis", False)],
            [("col1", "STRING", "YES", None)],
            [],  # no FK constraints
            [],  # no tags
            [(5,)], [("val1", 10)],  # profiling
            [("val1",)],  # sample rows
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        join_assertions = [a for a in assertions if a.predicate == AssertionPredicate.JOINS_TO]
        assert len(join_assertions) == 0


class TestTagExtraction:
    def test_tags_emit_has_tag(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("cbioportal_omop", "cancer_diagnosis", False)],
            [("dx_type_cd", "STRING", "YES", None)],
            [],  # no FK
            [("dx_type_cd", "pii", "false"), ("dx_type_cd", "domain", "oncology")],  # tags
            [(10,)], [("CRC", 50)],
            [("CRC",)],
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        tag_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_TAG]
        assert len(tag_assertions) == 2
        tag_values = {a.payload["tag_key"]: a.payload["tag_value"] for a in tag_assertions}
        assert tag_values["pii"] == "false"
        assert tag_values["domain"] == "oncology"


class TestColumnProfiling:
    def test_categorical_column_gets_top_values(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("schema", "tbl", False)],
            [("status", "STRING", "YES", None)],
            [], [],
            [(5,)],  # approx distinct = 5, below threshold
            [("active", 100), ("inactive", 50), ("pending", 10)],  # top-k
            [("active",)],
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        top_val_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_TOP_VALUES]
        assert len(top_val_assertions) == 1
        values = top_val_assertions[0].payload["values"]
        assert len(values) == 3
        assert values[0]["value"] == "active"
        assert values[0]["frequency"] == 100

    def test_high_cardinality_skips_sampling(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("schema", "tbl", False)],
            [("patient_id", "STRING", "NO", None)],
            [], [],
            [(50000,)],  # approx distinct = 50000, above threshold
            [("P1",)],  # sample rows
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        top_val_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_TOP_VALUES]
        assert len(top_val_assertions) == 0


class TestSampleRows:
    def test_sample_rows_emitted(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("schema", "tbl", False)],
            [("col1", "STRING", "YES", None)],
            [], [],
            [(3,)], [("a", 10), ("b", 5)],
            [("a",), ("b",)],  # sample rows
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        sample_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_SAMPLE_ROWS]
        assert len(sample_assertions) == 1
        assert len(sample_assertions[0].payload["rows"]) == 2


class TestScopeFiltering:
    def test_schema_filter(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("schema", "tbl1", False)],  # only clinical tables
            [("col1", "STRING", "YES", None)],
            [], [],
            [(3,)], [("a", 10)],
            [("a",)],
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm", schemas=["clinical"])

        table_assertions = [a for a in assertions if a.predicate == AssertionPredicate.TABLE_EXISTS]
        assert all("clinical" in a.subject_ref for a in table_assertions)

    def test_table_pattern_filter(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("cbioportal_omop", "cancer_diagnosis", False), ("cbioportal_omop", "cancer_surgery", False), ("cbioportal_omop", "patient_demographics", False)],
            # cancer_diagnosis columns
            [("col1", "STRING", "YES", None)],
            [], [],
            [(3,)], [("a", 10)],
            [("a",)],
            # cancer_surgery columns
            [("col1", "STRING", "YES", None)],
            [], [],
            [(3,)], [("a", 10)],
            [("a",)],
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm", table_pattern="cancer_*")

        table_assertions = [a for a in assertions if a.predicate == AssertionPredicate.TABLE_EXISTS]
        table_names = [a.subject_ref for a in table_assertions]
        assert all("cancer_" in name for name in table_names)
        assert not any("patient_" in name for name in table_names)


class TestDiscoverTables:
    def test_discover_returns_work_items(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",), ("staging",)],  # SHOW SCHEMAS
            [("cbioportal_omop", "cancer_diagnosis", False), ("cbioportal_omop", "patients", False)],  # clinical tables
            [("cbioportal_omop", "staging_raw", False)],  # staging tables
        ]

        items = connector.discover_tables(catalog="cdm")

        assert len(items) == 3
        assert all(isinstance(item, TableWorkItem) for item in items)
        assert items[0].catalog == "cdm"
        assert items[0].schema == "clinical"
        assert items[0].table_name == "cancer_diagnosis"
        assert items[0].fqn == "unity://cdm.clinical.cancer_diagnosis"
        assert items[1].table_name == "patients"
        assert items[2].schema == "staging"

    def test_discover_respects_schema_filter(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("cbioportal_omop", "tbl1", False)],  # clinical tables only
        ]

        items = connector.discover_tables(catalog="cdm", schemas=["clinical"])

        assert len(items) == 1
        assert items[0].schema == "clinical"

    def test_discover_respects_table_pattern(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],  # SHOW SCHEMAS
            [("cbioportal_omop", "cancer_diagnosis", False), ("cbioportal_omop", "cancer_surgery", False), ("cbioportal_omop", "patient_demographics", False)],
        ]

        items = connector.discover_tables(catalog="cdm", table_pattern="cancer_*")

        assert len(items) == 2
        assert all("cancer_" in item.table_name for item in items)

    def test_discover_produces_no_assertions(self, connector, mock_cursor):
        """discover_tables should be lightweight — no SQL profiling or column queries."""
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("cbioportal_omop", "tbl1", False)],
        ]

        items = connector.discover_tables(catalog="cdm")

        assert len(items) == 1
        # Only 2 queries: SHOW SCHEMAS + SHOW TABLES
        assert mock_cursor.execute.call_count == 2


class TestExtractTable:
    def test_extract_table_produces_assertions(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            # columns
            [("dx_type_cd", "STRING", "YES", "Diagnosis type code")],
            [],  # FK constraints
            [],  # tags
            [(10,)],  # APPROX_COUNT_DISTINCT
            [("CRC", 100), ("BRCA", 80)],  # top-k values
            [("CRC", "Stage III")],  # sample rows
        ]
        mock_cursor.description = [("col1",), ("col2",)]

        work_item = TableWorkItem(catalog="cdm", schema="clinical", table_name="cancer_diagnosis", fqn="unity://cdm.clinical.cancer_diagnosis")
        assertions = connector.extract_table(work_item)

        table_assertions = [a for a in assertions if a.predicate == AssertionPredicate.TABLE_EXISTS]
        assert len(table_assertions) == 1
        assert table_assertions[0].subject_ref == "unity://cdm.clinical.cancer_diagnosis"

        col_assertions = [a for a in assertions if a.predicate == AssertionPredicate.COLUMN_EXISTS]
        assert len(col_assertions) == 1
        assert col_assertions[0].payload["data_type"] == "STRING"

        dt_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_DATATYPE]
        assert len(dt_assertions) == 1

        comment_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_COMMENT]
        assert len(comment_assertions) == 1

        top_val_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_TOP_VALUES]
        assert len(top_val_assertions) == 1
        assert len(top_val_assertions[0].payload["values"]) == 2

        sample_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_SAMPLE_ROWS]
        assert len(sample_assertions) == 1

    def test_extract_table_with_fks_and_tags(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("patient_id", "STRING", "NO", None)],  # columns
            [("cancer_diagnosis", "patient_id", "patients", "patient_id")],  # FK
            [("patient_id", "pii", "true")],  # tags
            [(50000,)],  # high cardinality — no top-k
            [("P1",)],  # sample rows
        ]
        mock_cursor.description = [("col1",)]

        work_item = TableWorkItem(catalog="cdm", schema="clinical", table_name="cancer_diagnosis", fqn="unity://cdm.clinical.cancer_diagnosis")
        assertions = connector.extract_table(work_item)

        join_assertions = [a for a in assertions if a.predicate == AssertionPredicate.JOINS_TO]
        assert len(join_assertions) == 1
        assert "patients" in join_assertions[0].object_ref

        tag_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_TAG]
        assert len(tag_assertions) == 1
        assert tag_assertions[0].payload["tag_key"] == "pii"


class TestExtractBackwardCompatibility:
    def test_extract_matches_discover_plus_extract_table(self, connector, mock_cursor):
        """extract() should produce the same assertions as discover + extract_table."""
        # Set up for two tables
        base_side_effects = [
            [("clinical",)],  # SHOW SCHEMAS
            [("cbioportal_omop", "tbl1", False), ("cbioportal_omop", "tbl2", False)],  # SHOW TABLES
            # tbl1
            [("col1", "STRING", "YES", None)],  # columns
            [], [],  # FK, tags
            [(3,)], [("a", 10)],  # profiling
            [("a",)],  # sample rows
            # tbl2
            [("col2", "INT", "NO", None)],  # columns
            [], [],  # FK, tags
            [(100,)],  # high cardinality
            [("1",)],  # sample rows
        ]
        mock_cursor.description = [("col1",)]

        # Run extract()
        mock_cursor.fetchall.side_effect = list(base_side_effects)
        extract_assertions = connector.extract(catalog="cdm")

        extract_predicates = sorted([(a.subject_ref, a.predicate.value) for a in extract_assertions])

        # Run discover + extract_table
        mock_cursor.fetchall.side_effect = list(base_side_effects)
        work_items = connector.discover_tables(catalog="cdm")

        manual_assertions = []
        for item in work_items:
            manual_assertions.extend(connector.extract_table(item))

        manual_predicates = sorted([(a.subject_ref, a.predicate.value) for a in manual_assertions])

        # Same set of (subject_ref, predicate) pairs
        assert extract_predicates == manual_predicates


class TestErrorHandling:
    def test_connection_failure(self):
        with patch("sema.connectors.databricks.sql_connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")
            config = DatabricksConfig(
                host="https://bad.databricks.com",
                token="bad",
                http_path="/bad",
            )
            with pytest.raises(ConnectionError, match="Connection refused"):
                DatabricksConnector(config=config, profiling=ProfilingConfig())

    def test_query_failure_skips_entire_table(self, connector, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [("clinical",)],
            [("schema", "tbl1", False), ("schema", "tbl2", False)],
            Exception("Permission denied on tbl1"),
            # tbl2 succeeds
            [("col1", "STRING", "YES", None)],
            [], [],
            [(3,)], [("a", 10)],
            [("a",)],
        ]
        mock_cursor.description = [("col1",)]

        assertions = connector.extract(catalog="cdm")

        # tbl1 is entirely skipped (all-or-nothing per table)
        table_assertions = [a for a in assertions if a.predicate == AssertionPredicate.TABLE_EXISTS]
        assert len(table_assertions) == 1
        assert "tbl2" in table_assertions[0].subject_ref

        # Only tbl2 has column assertions
        col_assertions = [a for a in assertions if a.predicate == AssertionPredicate.COLUMN_EXISTS]
        assert all("tbl2" in a.subject_ref for a in col_assertions)
