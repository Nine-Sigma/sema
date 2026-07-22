"""S1-05 — deterministic source-row surrogate PK.

The row PK is a content hash of the STABLE source-row identity
(``source_schema, source_table, source_row_ref``) — never of the resolved
identity (``person_id``), so it survives a Stage-B dedup rebuild that recomputes
person_id while preserving the row PK. It is globally unique across studies
(schema is in the hash) and idempotent across re-runs.

The Python reference is the contract; the compiled SQL expression must reproduce
it exactly on the fit engine (DuckDB), and render the equivalent ``conv``-based
form on the live engine (Spark/Databricks). All three agree by construction:
``int(md5(payload)[:15], 16)``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sema.compile.row_surrogate import (
    surrogate_row_id,
    surrogate_row_id_expr,
)

pytestmark = pytest.mark.unit


class TestReference:
    def test_deterministic(self) -> None:
        a = surrogate_row_id("sch", "tbl", "SAMPLE-1")
        b = surrogate_row_id("sch", "tbl", "SAMPLE-1")
        assert a == b

    def test_positive_bigint_range(self) -> None:
        val = surrogate_row_id("sch", "tbl", "SAMPLE-1")
        assert 0 < val < 2**63

    def test_row_ref_changes_id(self) -> None:
        assert surrogate_row_id("sch", "tbl", "S-1") != surrogate_row_id(
            "sch", "tbl", "S-2"
        )

    def test_schema_changes_id_global_uniqueness(self) -> None:
        # Same row ref in two studies -> different PK (schema is in the hash),
        # so a multi-study condition table never collides.
        assert surrogate_row_id("gbm", "sample", "S-1") != surrogate_row_id(
            "msk", "sample", "S-1"
        )

    def test_table_boundary_no_collision(self) -> None:
        # Separator prevents ("a","b") from colliding with ("ab","").
        assert surrogate_row_id("a", "b", "x") != surrogate_row_id("ab", "", "x")


class TestDuckDBExpression:
    def test_sql_expr_matches_reference(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "s.duckdb"))
        conn.execute('CREATE SCHEMA "raw"')
        conn.execute('CREATE TABLE "raw"."sample" (sample_id VARCHAR)')
        conn.execute(
            "INSERT INTO \"raw\".\"sample\" VALUES ('P-1'), ('P-2'), ('P-3')"
        )
        expr = surrogate_row_id_expr(
            source_row_column="sample_id",
            source_schema="raw",
            source_table="sample",
            source_alias="src",
            dialect="duckdb",
        )
        sql = (
            f'SELECT sample_id, {expr.sql(dialect="duckdb")} AS rid '
            'FROM "raw"."sample" AS src'
        )
        rows = conn.execute(sql).fetchall()
        for sample_id, rid in rows:
            assert rid == surrogate_row_id("raw", "sample", sample_id)

    def test_ids_are_distinct_per_row(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "s.duckdb"))
        conn.execute('CREATE SCHEMA "raw"')
        conn.execute('CREATE TABLE "raw"."sample" (sample_id VARCHAR)')
        conn.execute(
            "INSERT INTO \"raw\".\"sample\" "
            "SELECT 'S-' || i FROM range(1, 1001) t(i)"
        )
        expr = surrogate_row_id_expr(
            source_row_column="sample_id",
            source_schema="raw",
            source_table="sample",
            source_alias="src",
            dialect="duckdb",
        )
        distinct, total = conn.execute(
            f'SELECT COUNT(DISTINCT {expr.sql(dialect="duckdb")}), COUNT(*) '
            'FROM "raw"."sample" AS src'
        ).fetchone()
        assert distinct == total == 1000


class TestSparkExpression:
    def test_spark_render_uses_conv(self) -> None:
        expr = surrogate_row_id_expr(
            source_row_column="sample_id",
            source_schema="raw",
            source_table="sample",
            source_alias="src",
            dialect="spark",
        )
        sql = expr.sql(dialect="spark").upper()
        assert "CONV" in sql
        assert "MD5" in sql
        assert "BIGINT" in sql
