"""S1-01 — generic identity-registry frozen schema, row model, SQL builders.

The two-level identity contract lives here as module-level constants so the
S1-01 drift test can freeze it from one place:
``(source_namespace, source_entity_key) -> source_entity_uid -> entity_id``.

Domain literals never appear in this module (D6/R29): a row carries whatever
source namespace / entity key the resolver writes; the OMOP ``person`` binding
is supplied by the policy/compile layer, not baked in here. ``entity_id`` is the
registry-assigned canonical id and is deliberately NOT in the grain, so Stage B
can collapse two uids onto one entity_id with an UPDATE, not a PK change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

# Two-level frozen columns — order is the table column order. Changing this set
# (add/remove/rename) is a contract change caught by the S1-01 drift test.
FROZEN_COLUMNS: tuple[str, ...] = (
    "source_namespace",
    "source_entity_key",
    "source_entity_uid",
    "entity_id",
    "run_id",
)

# Transactional unique key for atomic get-or-create (D7). ``entity_id`` is
# registry-assigned (not derived) and ``run_id`` is provenance — neither is here.
GRAIN_KEY: tuple[str, ...] = (
    "source_namespace",
    "source_entity_key",
)

_COLUMN_TYPES: dict[str, str] = {
    "source_namespace": "VARCHAR",
    "source_entity_key": "VARCHAR",
    "source_entity_uid": "VARCHAR",
    "entity_id": "BIGINT",
    "run_id": "VARCHAR",
}

# Unit separator: joins namespace+key for the uid hash so ("ab","c") and
# ("a","bc") never collide.
_UID_SEP = "\x1f"


def source_entity_uid(source_namespace: str, source_entity_key: str) -> str:
    """Stable per-source-entity surrogate — a content hash of the level-1 key.

    Deterministic across runs/machines (unlike ``hash()``): the same source
    entity always yields the same uid, and it never collapses across entities.
    """
    payload = f"{source_namespace}{_UID_SEP}{source_entity_key}".encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class IdentityAssignment:
    """One identity-registry row: a source entity mapped to a canonical id."""

    source_namespace: str
    source_entity_key: str
    source_entity_uid: str
    entity_id: int
    run_id: str

    def __post_init__(self) -> None:
        if not self.source_entity_key:
            raise ValueError(
                "source_entity_key must be non-empty; a blank source key routes "
                "to NO_MAP upstream, never to a synthetic identity"
            )
        if self.entity_id <= 0:
            raise ValueError(f"entity_id must be a positive int, got {self.entity_id}")


def from_row(row: tuple[Any, ...]) -> IdentityAssignment:
    """Build an IdentityAssignment from a DB row ordered by FROZEN_COLUMNS."""
    fields = dict(zip(FROZEN_COLUMNS, row, strict=True))
    return IdentityAssignment(
        source_namespace=str(fields["source_namespace"]),
        source_entity_key=str(fields["source_entity_key"]),
        source_entity_uid=str(fields["source_entity_uid"]),
        entity_id=int(fields["entity_id"]),
        run_id=str(fields["run_id"]),
    )


def _qualified(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def create_table_sql(schema: str, table: str) -> str:
    cols = ",\n  ".join(f'"{c}" {_COLUMN_TYPES[c]}' for c in FROZEN_COLUMNS)
    pk = ", ".join(f'"{c}"' for c in GRAIN_KEY)
    return (
        f"CREATE TABLE IF NOT EXISTS {_qualified(schema, table)} (\n"
        f"  {cols},\n"
        f"  PRIMARY KEY ({pk})\n)"
    )


def insert_ignore_sql(schema: str, table: str) -> str:
    """Get-or-create: an already-assigned grain key is left untouched (D7)."""
    cols = ", ".join(f'"{c}"' for c in FROZEN_COLUMNS)
    placeholders = ", ".join("?" for _ in FROZEN_COLUMNS)
    conflict = ", ".join(f'"{c}"' for c in GRAIN_KEY)
    return (
        f"INSERT INTO {_qualified(schema, table)} ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO NOTHING"
    )


def select_all_sql(schema: str, table: str) -> str:
    cols = ", ".join(f'"{c}"' for c in FROZEN_COLUMNS)
    return f"SELECT {cols} FROM {_qualified(schema, table)}"


def select_by_grain_sql(schema: str, table: str) -> str:
    cols = ", ".join(f'"{c}"' for c in FROZEN_COLUMNS)
    where = " AND ".join(f'"{c}" = ?' for c in GRAIN_KEY)
    return f"SELECT {cols} FROM {_qualified(schema, table)} WHERE {where}"


def max_entity_id_sql(schema: str, table: str) -> str:
    return f'SELECT COALESCE(MAX("entity_id"), 0) FROM {_qualified(schema, table)}'
