"""Canonical identity primitives for physical and semantic graph nodes.

PhysicalKey is the structured identity for warehouse assets.
CanonicalRef parses connector-native URIs into PhysicalKey.
NodeKey builds Neo4j merge keys from PhysicalKey + semantic scope.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
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
    r"^databricks://(?P<ws>[^/]+)/(?P<cat>[^/]+)"
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
    r"^databricks://(?P<ws>[^/]+)/(?P<cat>[^/]+)"
    r"/(?P<sch>[^/]+)/(?P<tbl>[^.]+)\.(?P<col>.+)$"
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


class CanonicalRef:
    """Parse any connector-native URI into a PhysicalKey."""

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

    @staticmethod
    def joinpath(
        datasource_id: str,
        from_table: str,
        to_table: str,
        join_columns: list[tuple[str, str]],
    ) -> dict[str, str]:
        cols_str = ",".join(
            f"{a}={b}" for a, b in sorted(join_columns)
        )
        cols_hash = hashlib.sha256(cols_str.encode()).hexdigest()[:12]
        return {
            "datasource_id": datasource_id,
            "from_table": from_table,
            "to_table": to_table,
            "join_columns_hash": cols_hash,
        }
