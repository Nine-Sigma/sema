import pytest

pytestmark = pytest.mark.unit

from sema.models.constants import (
    GENERIC_JOIN_COLUMNS,
    MATCH_TYPE_BOOST,
    REF_PATTERN,
    RefParts,
    SourcePrecedence,
    parse_ref,
    parse_unity_ref,
    parse_unity_ref_strict,
    source_precedence,
    translate_ref,
)


class TestSourcePrecedence:
    def test_enum_values(self) -> None:
        assert SourcePrecedence.HUMAN == 100
        assert SourcePrecedence.ATLAN == 80
        assert SourcePrecedence.DBT == 60
        assert SourcePrecedence.UNITY_CATALOG == 40
        assert SourcePrecedence.LLM_INTERPRETATION == 20
        assert SourcePrecedence.HEURISTIC == 15
        assert SourcePrecedence.PATTERN_MATCH == 10

    def test_helper_known_source(self) -> None:
        assert source_precedence("human") == 100
        assert source_precedence("dbt") == 60

    def test_helper_unknown_source_returns_default(self) -> None:
        assert source_precedence("some_new_source") == 0

    def test_helper_custom_default(self) -> None:
        assert source_precedence("unknown", default=5) == 5


class TestParseRef:
    def test_databricks_table_ref(self) -> None:
        parts = parse_ref("databricks://my-ws/cdm/clinical/cancer_diagnosis")
        assert parts.platform == "databricks"
        assert parts.workspace == "my-ws"
        assert parts.catalog == "cdm"
        assert parts.schema == "clinical"
        assert parts.table == "cancer_diagnosis"
        assert parts.column is None

    def test_databricks_column_ref(self) -> None:
        parts = parse_ref(
            "databricks://my-ws/cdm/clinical/cancer_diagnosis/dx_type_cd"
        )
        assert parts.platform == "databricks"
        assert parts.workspace == "my-ws"
        assert parts.catalog == "cdm"
        assert parts.schema == "clinical"
        assert parts.table == "cancer_diagnosis"
        assert parts.column == "dx_type_cd"

    def test_invalid_ref_returns_none(self) -> None:
        assert parse_ref("not-a-valid-ref") is None

    def test_ref_pattern_matches(self) -> None:
        m = REF_PATTERN.match("databricks://ws/cat/sch/tbl")
        assert m is not None
        assert m.group("platform") == "databricks"

    def test_ref_parts_is_named_tuple(self) -> None:
        parts = RefParts(
            platform="databricks",
            workspace="ws",
            catalog="cat",
            schema="sch",
            table="tbl",
            column=None,
        )
        assert parts.platform == "databricks"
        assert parts.column is None


class TestTranslateRef:
    def test_table_ref(self) -> None:
        old = "unity://cdm.clinical.cancer_diagnosis"
        result = translate_ref(old, "my-workspace")
        assert result == "databricks://my-workspace/cdm/clinical/cancer_diagnosis"

    def test_column_ref(self) -> None:
        old = "unity://cdm.clinical.cancer_diagnosis.dx_type_cd"
        result = translate_ref(old, "my-workspace")
        assert result == "databricks://my-workspace/cdm/clinical/cancer_diagnosis/dx_type_cd"

    def test_non_unity_ref_passthrough(self) -> None:
        ref = "databricks://ws/cdm/clinical/tbl"
        assert translate_ref(ref, "ws") == ref

    def test_unknown_format_passthrough(self) -> None:
        ref = "some-random-string"
        assert translate_ref(ref, "ws") == ref


class TestLegacyParseUnityRef:
    def test_valid_ref_with_column(self) -> None:
        assert parse_unity_ref("unity://catalog.schema.table.column") == (
            "catalog", "schema", "table", "column",
        )

    def test_valid_ref_without_column(self) -> None:
        assert parse_unity_ref("unity://catalog.schema.table") == (
            "catalog", "schema", "table", None,
        )

    def test_invalid_ref_returns_empty_strings(self) -> None:
        assert parse_unity_ref("not-a-unity-ref") == (
            "", "", "not-a-unity-ref", None,
        )

    def test_strict_valid_ref(self) -> None:
        assert parse_unity_ref_strict("unity://catalog.schema.table") == (
            "catalog", "schema", "table", None,
        )

    def test_strict_invalid_ref_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid ref"):
            parse_unity_ref_strict("not-a-unity-ref")


class TestConstants:
    def test_generic_join_columns_contents(self) -> None:
        expected = {"id", "name", "type", "status", "value", "code",
                    "description", "created_at", "updated_at"}
        assert GENERIC_JOIN_COLUMNS == expected

    def test_generic_join_columns_immutable(self) -> None:
        with pytest.raises(AttributeError):
            GENERIC_JOIN_COLUMNS.add("new_col")  # type: ignore[attr-defined]

    def test_match_type_boost_values(self) -> None:
        assert MATCH_TYPE_BOOST["lexical_exact"] == 0.3
        assert MATCH_TYPE_BOOST["vector"] == 0.0
        assert MATCH_TYPE_BOOST["graph"] == -0.1

    def test_match_type_boost_immutable(self) -> None:
        with pytest.raises(TypeError):
            MATCH_TYPE_BOOST["new_key"] = 1.0  # type: ignore[index]
