"""Tests for PhysicalKey, CanonicalRef, and NodeKey."""

import pytest

from sema.models.physical_key import CanonicalRef, NodeKey, PhysicalKey

pytestmark = pytest.mark.unit


class TestPhysicalKey:
    def test_table_key(self) -> None:
        pk = PhysicalKey("ds1", "catalog", "schema", "table")
        assert pk.table_key == "ds1/catalog/schema/table"

    def test_table_key_no_schema(self) -> None:
        pk = PhysicalKey("ds1", "mydb", None, "users")
        assert pk.table_key == "ds1/mydb/users"

    def test_column_key(self) -> None:
        pk = PhysicalKey("ds1", "catalog", "schema", "table", "col")
        assert pk.column_key == "ds1/catalog/schema/table/col"

    def test_column_key_none_when_no_column(self) -> None:
        pk = PhysicalKey("ds1", "catalog", "schema", "table")
        assert pk.column_key is None

    def test_frozen(self) -> None:
        pk = PhysicalKey("ds1", "catalog", "schema", "table")
        with pytest.raises(AttributeError):
            pk.table = "other"  # type: ignore[misc]


class TestCanonicalRefDatabricks:
    def test_table_ref(self) -> None:
        pk = CanonicalRef.parse(
            "databricks://workspace/catalog/schema/table"
        )
        assert pk.datasource_id == "workspace"
        assert pk.catalog_or_db == "catalog"
        assert pk.schema == "schema"
        assert pk.table == "table"
        assert pk.column is None

    def test_column_ref(self) -> None:
        pk = CanonicalRef.parse(
            "databricks://workspace/catalog/schema/table/column"
        )
        assert pk.column == "column"

    def test_dotted_column_ref_normalized(self) -> None:
        pk = CanonicalRef.parse(
            "databricks://workspace/catalog/schema/table.column"
        )
        assert pk.table == "table"
        assert pk.column == "column"

    def test_datasource_id_override(self) -> None:
        pk = CanonicalRef.parse(
            "databricks://workspace/catalog/schema/table",
            datasource_id="custom_ds",
        )
        assert pk.datasource_id == "custom_ds"


class TestCanonicalRefPostgres:
    def test_table_ref(self) -> None:
        pk = CanonicalRef.parse(
            "postgres://localhost:5432/mydb/public/users"
        )
        assert pk.datasource_id == "localhost:5432/mydb"
        assert pk.catalog_or_db == "mydb"
        assert pk.schema == "public"
        assert pk.table == "users"
        assert pk.column is None

    def test_column_ref(self) -> None:
        pk = CanonicalRef.parse(
            "postgres://localhost:5432/mydb/public/users/email"
        )
        assert pk.column == "email"


class TestCanonicalRefMySQL:
    def test_table_ref_no_schema(self) -> None:
        pk = CanonicalRef.parse(
            "mysql://localhost:3306/mydb/users"
        )
        assert pk.datasource_id == "localhost:3306/mydb"
        assert pk.catalog_or_db == "mydb"
        assert pk.schema is None
        assert pk.table == "users"

    def test_column_ref(self) -> None:
        pk = CanonicalRef.parse(
            "mysql://localhost:3306/mydb/users/email"
        )
        assert pk.column == "email"
        assert pk.schema is None


class TestCanonicalRefLegacyUnity:
    def test_basic(self) -> None:
        pk = CanonicalRef.parse("unity://catalog.schema.table")
        assert pk.datasource_id == "unity"
        assert pk.catalog_or_db == "catalog"
        assert pk.schema == "schema"
        assert pk.table == "table"

    def test_with_column(self) -> None:
        pk = CanonicalRef.parse("unity://catalog.schema.table.column")
        assert pk.column == "column"

    def test_datasource_override(self) -> None:
        pk = CanonicalRef.parse(
            "unity://catalog.schema.table",
            datasource_id="my_workspace",
        )
        assert pk.datasource_id == "my_workspace"


class TestCanonicalRefErrors:
    def test_invalid_ref(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse ref"):
            CanonicalRef.parse("not_a_valid_ref")

    def test_unknown_protocol(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse ref"):
            CanonicalRef.parse("ftp://host/path")


class TestNodeKey:
    def test_entity_key(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl")
        key = NodeKey.entity(pk)
        assert key == {
            "datasource_id": "ds1",
            "table_key": "ds1/cat/sch/tbl",
        }

    def test_property_key(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl", "col")
        key = NodeKey.property(pk)
        assert key == {
            "datasource_id": "ds1",
            "column_key": "ds1/cat/sch/tbl/col",
        }

    def test_property_key_requires_column(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl")
        with pytest.raises(ValueError, match="no column"):
            NodeKey.property(pk)

    def test_vocabulary_key(self) -> None:
        key = NodeKey.vocabulary("ICD-10")
        assert key == {"name": "ICD-10"}

    def test_term_key(self) -> None:
        key = NodeKey.term("ICD-10", "C34.1")
        assert key == {"vocabulary_name": "ICD-10", "code": "C34.1"}

    def test_alias_key(self) -> None:
        key = NodeKey.alias("target_123", "BP")
        assert key == {"target_key": "target_123", "text": "BP"}

    def test_valueset_key(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl", "col")
        key = NodeKey.valueset(pk)
        assert key == {
            "datasource_id": "ds1",
            "column_key": "ds1/cat/sch/tbl/col",
        }

    def test_joinpath_key_deterministic(self) -> None:
        k1 = NodeKey.joinpath(
            "ds1", "table_a", "table_b",
            [("col_a", "col_b")],
        )
        k2 = NodeKey.joinpath(
            "ds1", "table_a", "table_b",
            [("col_a", "col_b")],
        )
        assert k1 == k2
        assert k1["datasource_id"] == "ds1"
        assert k1["from_table"] == "table_a"
        assert k1["to_table"] == "table_b"
        assert len(k1["join_columns_hash"]) == 12

    def test_joinpath_key_different_columns(self) -> None:
        k1 = NodeKey.joinpath(
            "ds1", "a", "b", [("x", "y")]
        )
        k2 = NodeKey.joinpath(
            "ds1", "a", "b", [("p", "q")]
        )
        assert k1["join_columns_hash"] != k2["join_columns_hash"]

    def test_table_key_with_schema(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl")
        key = NodeKey.table(pk)
        assert key == {
            "datasource_id": "ds1",
            "catalog": "cat",
            "name": "tbl",
            "schema_name": "sch",
        }

    def test_table_key_no_schema(self) -> None:
        pk = PhysicalKey("ds1", "mydb", None, "users")
        key = NodeKey.table(pk)
        assert key == {
            "datasource_id": "ds1",
            "catalog": "mydb",
            "name": "users",
        }
        assert "schema_name" not in key

    def test_column_key_with_schema(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl", "col")
        key = NodeKey.column(pk)
        assert key == {
            "datasource_id": "ds1",
            "catalog": "cat",
            "table_name": "tbl",
            "name": "col",
            "schema_name": "sch",
        }

    def test_column_key_no_schema(self) -> None:
        pk = PhysicalKey("ds1", "mydb", None, "tbl", "col")
        key = NodeKey.column(pk)
        assert key == {
            "datasource_id": "ds1",
            "catalog": "mydb",
            "table_name": "tbl",
            "name": "col",
        }
        assert "schema_name" not in key

    def test_column_key_requires_column(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl")
        with pytest.raises(ValueError, match="no column"):
            NodeKey.column(pk)

    def test_valueset_key_requires_column(self) -> None:
        pk = PhysicalKey("ds1", "cat", "sch", "tbl")
        with pytest.raises(ValueError, match="no column"):
            NodeKey.valueset(pk)
