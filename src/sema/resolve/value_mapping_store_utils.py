"""§1.5(a) value-mapping store — frozen schema, row model, and SQL builders.

The frozen column list and grain key live here as module-level constants so the
US-005 drift test can freeze the contract from one place. Domain literals never
appear in this module: a row carries whatever source vocabulary / target
property the resolver writes (the OMOP/OncoTree specifics stay in the policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResolutionStatus(str, Enum):
    """Per-decision resolution outcome (mirrors :mod:`sema.eval`)."""

    RESOLVED = "RESOLVED"
    NO_MAP = "NO_MAP"


# §1.5(a) frozen columns — order is the table column order. Changing this set
# (add/remove/rename) is a contract change caught by the US-005 drift test.
FROZEN_COLUMNS: tuple[str, ...] = (
    "source_vocabulary",
    "normalized_source_value",
    "target_property_ref",
    "target_field",
    "vocab_binding",
    "concept_id",
    "vocab_release",
    "valid_start",
    "valid_end",
    "resolution_status",
    "no_map_reason",
    "confidence",
    "status",
    "resolver_policy_ref",
    "run_id",
)

# §1.5(a) unique grain key. ``run_id`` is provenance, deliberately NOT in here.
GRAIN_KEY: tuple[str, ...] = (
    "source_vocabulary",
    "normalized_source_value",
    "target_property_ref",
    "resolver_policy_ref",
    "vocab_release",
)

_COLUMN_TYPES: dict[str, str] = {
    "source_vocabulary": "VARCHAR",
    "normalized_source_value": "VARCHAR",
    "target_property_ref": "VARCHAR",
    "target_field": "VARCHAR",
    "vocab_binding": "VARCHAR",
    "concept_id": "BIGINT",
    "vocab_release": "VARCHAR",
    "valid_start": "VARCHAR",
    "valid_end": "VARCHAR",
    "resolution_status": "VARCHAR",
    "no_map_reason": "VARCHAR",
    "confidence": "DOUBLE",
    "status": "VARCHAR",
    "resolver_policy_ref": "VARCHAR",
    "run_id": "VARCHAR",
}


@dataclass(frozen=True)
class ValueMapping:
    """One resolved decision: a §1.5(a) value-mapping-store row.

    ``concept_id`` is NULL iff ``resolution_status == NO_MAP``; ``no_map_reason``
    is non-null iff NO_MAP. ``status`` is the lifecycle status carried from the
    winning assertion.
    """

    source_vocabulary: str
    normalized_source_value: str
    target_property_ref: str
    target_field: str
    vocab_binding: str
    concept_id: int | None
    vocab_release: str
    valid_start: str | None
    valid_end: str | None
    resolution_status: ResolutionStatus
    no_map_reason: str | None
    confidence: float
    status: Any  # sema.models.planner.lifecycle.Status (avoid import cycle)
    resolver_policy_ref: str
    run_id: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.resolution_status is ResolutionStatus.NO_MAP:
            if self.concept_id is not None:
                raise ValueError("NO_MAP rows MUST have concept_id = NULL")
            if not self.no_map_reason:
                raise ValueError("NO_MAP rows MUST carry a no_map_reason")
        else:
            if self.concept_id is None:
                raise ValueError("RESOLVED rows MUST have a non-null concept_id")
            if self.no_map_reason is not None:
                raise ValueError("RESOLVED rows MUST NOT carry a no_map_reason")


def _status_value(status: Any) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def to_params(mapping: ValueMapping) -> list[Any]:
    """Bind values in FROZEN_COLUMNS order for an INSERT."""
    return [
        mapping.source_vocabulary,
        mapping.normalized_source_value,
        mapping.target_property_ref,
        mapping.target_field,
        mapping.vocab_binding,
        mapping.concept_id,
        mapping.vocab_release,
        mapping.valid_start,
        mapping.valid_end,
        mapping.resolution_status.value,
        mapping.no_map_reason,
        mapping.confidence,
        _status_value(mapping.status),
        mapping.resolver_policy_ref,
        mapping.run_id,
    ]


def from_row(row: tuple[Any, ...], status_cls: Any) -> ValueMapping:
    """Build a ValueMapping from a DB row ordered by FROZEN_COLUMNS."""
    fields = dict(zip(FROZEN_COLUMNS, row, strict=True))
    return ValueMapping(
        source_vocabulary=fields["source_vocabulary"],
        normalized_source_value=fields["normalized_source_value"],
        target_property_ref=fields["target_property_ref"],
        target_field=fields["target_field"],
        vocab_binding=fields["vocab_binding"],
        concept_id=fields["concept_id"],
        vocab_release=fields["vocab_release"],
        valid_start=fields["valid_start"],
        valid_end=fields["valid_end"],
        resolution_status=ResolutionStatus(fields["resolution_status"]),
        no_map_reason=fields["no_map_reason"],
        confidence=fields["confidence"],
        status=status_cls(fields["status"]),
        resolver_policy_ref=fields["resolver_policy_ref"],
        run_id=fields["run_id"],
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


def upsert_sql(schema: str, table: str) -> str:
    cols = ", ".join(f'"{c}"' for c in FROZEN_COLUMNS)
    placeholders = ", ".join("?" for _ in FROZEN_COLUMNS)
    conflict = ", ".join(f'"{c}"' for c in GRAIN_KEY)
    updates = ", ".join(
        f'"{c}" = excluded."{c}"' for c in FROZEN_COLUMNS if c not in GRAIN_KEY
    )
    return (
        f"INSERT INTO {_qualified(schema, table)} ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )


def select_all_sql(schema: str, table: str) -> str:
    cols = ", ".join(f'"{c}"' for c in FROZEN_COLUMNS)
    return f"SELECT {cols} FROM {_qualified(schema, table)}"


def select_by_grain_sql(schema: str, table: str) -> str:
    cols = ", ".join(f'"{c}"' for c in FROZEN_COLUMNS)
    where = " AND ".join(f'"{c}" = ?' for c in GRAIN_KEY)
    return f"SELECT {cols} FROM {_qualified(schema, table)} WHERE {where}"
