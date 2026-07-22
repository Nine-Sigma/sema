"""S1-05 — deterministic source-row surrogate PK (generic, R29-scanned).

A row's surrogate primary key is a content hash of its STABLE source-row
identity (``source_schema``, ``source_table``, ``source_row_ref``) — NOT of any
resolved identity. This is what lets a Stage-B dedup rebuild recompute the row's
resolved-entity FK while keeping the row PK stable, and what keeps the PK unique
across studies (the schema is in the hash).

MD5 is used because it is byte-identical across the fit engine (DuckDB) and the
live engine (Spark/Databricks); the first 15 hex digits (60 bits) parse to an
always-positive BIGINT. The Python reference and both SQL renderings agree by
construction: ``int(md5(payload)[:15], 16)``.
"""

from __future__ import annotations

import hashlib

import sqlglot.expressions as exp

# Unit separator joins the identity parts so ("a","b") can never collide with
# ("ab","") for the same row ref.
_SEP = "\x1f"
# 15 hex digits = 60 bits -> always fits a positive signed BIGINT.
_HEX_DIGITS = 15


def _payload(source_schema: str, source_table: str, source_row_ref: str) -> str:
    return f"{source_schema}{_SEP}{source_table}{_SEP}{source_row_ref}"


def surrogate_row_id(
    source_schema: str, source_table: str, source_row_ref: str
) -> int:
    """The canonical (reference) surrogate PK for one source row."""
    digest = hashlib.md5(_payload(source_schema, source_table, source_row_ref).encode())
    return int(digest.hexdigest()[:_HEX_DIGITS], 16)


def surrogate_row_id_expr(
    *,
    source_row_column: str,
    source_schema: str,
    source_table: str,
    source_alias: str,
    dialect: str,
) -> exp.Expression:
    """A SQLGlot expression computing :func:`surrogate_row_id` over a source column.

    The schema/table are folded into a constant prefix (they are fixed per
    compile); only the row-ref column varies. DuckDB parses the hex prefix via a
    ``'0x'``-cast; Spark/Databricks via ``conv(..., 16, 10)``. Both yield the same
    integer as the Python reference.
    """
    prefix = f"{source_schema}{_SEP}{source_table}{_SEP}"
    row_col = exp.cast(
        exp.column(source_row_column, table=source_alias), to="VARCHAR"
    )
    joined = exp.Concat(expressions=[exp.Literal.string(prefix), row_col])
    hexed = exp.func("substr", exp.func("md5", joined), exp.Literal.number(1),
                     exp.Literal.number(_HEX_DIGITS))
    if dialect in ("spark", "databricks", "hive"):
        return exp.cast(
            exp.func("conv", hexed, exp.Literal.number(16), exp.Literal.number(10)),
            to="BIGINT",
        )
    zero_x = exp.Concat(expressions=[exp.Literal.string("0x"), hexed])
    return exp.cast(zero_x, to="BIGINT")
