"""Canonical identity primitives for physical and semantic graph nodes.

PhysicalKey is the structured identity for warehouse assets.
CanonicalRef parses connector-native URIs into PhysicalKey.
NodeKey builds Neo4j merge keys from PhysicalKey + semantic scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Final


@dataclass(frozen=True)
class PhysicalKey:
    """Structured identity for a physical warehouse asset.

    All graph merge operations use this instead of raw URI strings.
    ``schema`` is None for databases without schema levels (e.g. MySQL).
    ``column`` is None for table-level keys.
    """

    datasource_id: str
    catalog_or_db: str
    schema: str | None
    table: str
    column: str | None = None

    @property
    def table_key(self) -> str:
        parts = [self.datasource_id, self.catalog_or_db]
        if self.schema is not None:
            parts.append(self.schema)
        parts.append(self.table)
        return "/".join(parts)

    @property
    def column_key(self) -> str | None:
        if self.column is None:
            return None
        return f"{self.table_key}/{self.column}"


# ---------------------------------------------------------------------------
# CanonicalRef: parse any connector URI into PhysicalKey
# ---------------------------------------------------------------------------

_DATABRICKS_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:databricks|duckdb)://(?P<ws>[^/]+)/(?P<cat>[^/]+)"
    r"/(?P<sch>[^/]+)/(?P<tbl>[^/]+)(?:/(?P<col>.+))?$"
)

_POSTGRES_RE: Final[re.Pattern[str]] = re.compile(
    r"^postgres://(?P<host>[^/]+)/(?P<db>[^/]+)"
    r"/(?P<sch>[^/]+)/(?P<tbl>[^/]+)(?:/(?P<col>.+))?$"
)

_MYSQL_RE: Final[re.Pattern[str]] = re.compile(
    r"^mysql://(?P<host>[^/]+)/(?P<db>[^/]+)"
    r"/(?P<tbl>[^/]+)(?:/(?P<col>.+))?$"
)

_UNITY_RE: Final[re.Pattern[str]] = re.compile(
    r"^unity://(?P<cat>[^.]+)\.(?P<sch>[^.]+)\.(?P<tbl>[^.]+)"
    r"(?:\.(?P<col>.+))?$"
)

# Handles dotted column refs: databricks://ws/cat/sch/table.column
_DATABRICKS_DOTTED_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:databricks|duckdb)://(?P<ws>[^/]+)/(?P<cat>[^/]+)"
    r"/(?P<sch>[^/]+)/(?P<tbl>[^./]+)\.(?P<col>.+)$"
)


def _try_databricks(
    ref: str, ds_override: str | None,
) -> PhysicalKey | None:
    """Try dotted column format first, then slash-delimited."""
    for pattern in (_DATABRICKS_DOTTED_RE, _DATABRICKS_RE):
        m = pattern.match(ref)
        if m:
            return PhysicalKey(
                datasource_id=ds_override or m.group("ws"),
                catalog_or_db=m.group("cat"),
                schema=m.group("sch"),
                table=m.group("tbl"),
                column=m.group("col"),
            )
    return None


def _try_postgres(
    ref: str, ds_override: str | None,
) -> PhysicalKey | None:
    m = _POSTGRES_RE.match(ref)
    if not m:
        return None
    ds = ds_override or f"{m.group('host')}/{m.group('db')}"
    return PhysicalKey(
        datasource_id=ds, catalog_or_db=m.group("db"),
        schema=m.group("sch"), table=m.group("tbl"),
        column=m.group("col"),
    )


def _try_mysql(
    ref: str, ds_override: str | None,
) -> PhysicalKey | None:
    m = _MYSQL_RE.match(ref)
    if not m:
        return None
    ds = ds_override or f"{m.group('host')}/{m.group('db')}"
    return PhysicalKey(
        datasource_id=ds, catalog_or_db=m.group("db"),
        schema=None, table=m.group("tbl"), column=m.group("col"),
    )


def _try_unity(
    ref: str, ds_override: str | None,
) -> PhysicalKey | None:
    m = _UNITY_RE.match(ref)
    if not m:
        return None
    return PhysicalKey(
        datasource_id=ds_override or "unity",
        catalog_or_db=m.group("cat"), schema=m.group("sch"),
        table=m.group("tbl"), column=m.group("col"),
    )


_PARSERS = [_try_databricks, _try_postgres, _try_mysql, _try_unity]

# Schemes whose parse reads back <datasource>/<catalog>/<schema>/<table>[/<column>].
_RENDERED_SCHEMES: Final[frozenset[str]] = frozenset({"databricks", "duckdb"})


def _ref_part(value: str | None) -> str:
    if not value or "/" in value:
        raise ValueError(f"Reference part must be non-empty and hold no '/': {value!r}")
    return value


def _column_part(value: str) -> str:
    # Parse reads everything after the table as the column, so a '/' here round-trips.
    if not value:
        raise ValueError("Reference column part must be non-empty")
    return value


class CanonicalRef:
    """Parse connector-native URIs into PhysicalKey, and build the one canonical form."""

    @staticmethod
    def datasource(scheme: str, datasource_id: str) -> str:
        """Render the datasource-level reference ``<scheme>://<datasource>``."""
        if scheme not in _RENDERED_SCHEMES:
            raise ValueError(f"No reference renderer for scheme: {scheme}")
        return f"{scheme}://{_ref_part(datasource_id)}"

    @staticmethod
    def build(scheme: str, key: PhysicalKey) -> str:
        """Render ``key`` as ``<scheme>://<ds>/<catalog>/<schema>/<table>[/<column>]``."""
        parts = [
            CanonicalRef.datasource(scheme, key.datasource_id),
            *map(_ref_part, [key.catalog_or_db, key.schema, key.table]),
        ]
        if key.column is not None:
            parts.append(_column_part(key.column))
        return "/".join(parts)

    @staticmethod
    def with_column(table_ref: str, column: str) -> str:
        """Build the reference of ``column`` in the table that ``table_ref`` names."""
        key = CanonicalRef.parse(table_ref)
        if key.column is not None:
            raise ValueError(f"Reference already names a column: {table_ref}")
        scheme = table_ref.split("://", 1)[0]
        return CanonicalRef.build(scheme, replace(key, column=column))

    @staticmethod
    def parse(ref: str, datasource_id: str | None = None) -> PhysicalKey:
        """Parse a connector URI into PhysicalKey.

        Tries each connector format in order. Raises ValueError
        if none match.
        """
        for parser in _PARSERS:
            result = parser(ref, datasource_id)
            if result is not None:
                return result
        raise ValueError(f"Cannot parse ref: {ref}")


# ---------------------------------------------------------------------------
# NodeKey: build Neo4j merge keys from PhysicalKey + semantic scope
# ---------------------------------------------------------------------------


class NodeKey:
    """Build Neo4j MERGE keys for physical and semantic nodes."""

    @staticmethod
    def table(pk: PhysicalKey) -> dict[str, Any]:
        d: dict[str, Any] = {
            "datasource_id": pk.datasource_id,
            "catalog": pk.catalog_or_db,
            "name": pk.table,
        }
        if pk.schema is not None:
            d["schema_name"] = pk.schema
        return d

    @staticmethod
    def column(pk: PhysicalKey) -> dict[str, Any]:
        if pk.column is None:
            raise ValueError("PhysicalKey has no column")
        d: dict[str, Any] = {
            "datasource_id": pk.datasource_id,
            "catalog": pk.catalog_or_db,
            "table_name": pk.table,
            "name": pk.column,
        }
        if pk.schema is not None:
            d["schema_name"] = pk.schema
        return d

    @staticmethod
    def entity(pk: PhysicalKey) -> dict[str, str]:
        return {
            "datasource_id": pk.datasource_id,
            "table_key": pk.table_key,
        }

    @staticmethod
    def property(pk: PhysicalKey) -> dict[str, str]:
        col_key = pk.column_key
        if col_key is None:
            raise ValueError("PhysicalKey has no column for property key")
        return {
            "datasource_id": pk.datasource_id,
            "column_key": col_key,
        }

    @staticmethod
    def vocabulary(name: str) -> dict[str, str]:
        return {"name": name}

    @staticmethod
    def term(vocabulary_name: str, code: str) -> dict[str, str]:
        return {"vocabulary_name": vocabulary_name, "code": code}

    @staticmethod
    def alias(target_key: str, text: str) -> dict[str, str]:
        return {"target_key": target_key, "text": text}

    @staticmethod
    def valueset(pk: PhysicalKey) -> dict[str, str]:
        col_key = pk.column_key
        if col_key is None:
            raise ValueError("PhysicalKey has no column for valueset key")
        return {
            "datasource_id": pk.datasource_id,
            "column_key": col_key,
        }
