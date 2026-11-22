"""Project authored target joins into retrieval `:JoinPath` nodes (M5).

Retrieval reads joins only from ``(:JoinPath)-[:USES]->(:Table)`` — the target
materializer writes ``:TargetObligation`` instead, so FK joins are invisible.
This is a *deterministic manifest projection* (not heuristic FK detection,
which stays Slice-3): each obligation foreign key and each entity-level
declared join becomes one ``:JoinPath`` with ``USES`` edges to both tables.

Generic: reads the normalized model's obligations, entity declared joins, and
a physical schema binding; names no target domain.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from sema.graph.loader import GraphLoader
from sema.graph.target_retrieval_loader import PhysicalSchemaBinding
from sema.log import logger
from sema.models.physical_key import PhysicalKey
from sema.models.target.normalized import NormalizedTargetModel

_OBLIGATION_SOURCE = "manifest"
_OBLIGATION_CONFIDENCE = 1.0
_DECLARED_SOURCE = "manifest_declared"
# Above retrieval's structural and semantic thresholds so declared joins are
# used, below an enforced obligation join so retrieval can rank them apart.
DECLARED_JOIN_CONFIDENCE = 0.8


@dataclass(frozen=True)
class JoinPathCounts:
    written: int
    skipped: int


@dataclass(frozen=True)
class _AuthoredJoin:
    entity: str
    referenced_entity: str
    join_keys: tuple[tuple[str, str], ...]
    source: str
    confidence: float


@dataclass(frozen=True)
class _BoundJoin:
    entity_table: str
    referenced_table: str
    join: _AuthoredJoin


def materialize_target_join_paths(
    loader: GraphLoader,
    normalized: NormalizedTargetModel,
    binding: PhysicalSchemaBinding,
) -> JoinPathCounts:
    bound: list[_BoundJoin] = []
    skipped = 0
    for join in _authored_joins(normalized):
        entity_table = binding.table_for(join.entity)
        referenced_table = binding.table_for(join.referenced_entity)
        if entity_table is None or referenced_table is None:
            skipped += 1
            continue
        bound.append(_BoundJoin(entity_table, referenced_table, join))
    pair_counts = Counter((b.entity_table, b.referenced_table) for b in bound)
    for item in bound:
        shared = pair_counts[(item.entity_table, item.referenced_table)] > 1
        _write_join_path(loader, binding, item, _path_name(item, shared))
    if skipped:
        logger.warning(
            "Skipped {} target join(s) whose entity has no physical binding",
            skipped,
        )
    return JoinPathCounts(written=len(bound), skipped=skipped)


def _authored_joins(normalized: NormalizedTargetModel) -> Iterator[_AuthoredJoin]:
    for obligation in normalized.obligations:
        for fk in obligation.foreign_keys:
            yield _AuthoredJoin(
                obligation.target_entity, fk.referenced_entity,
                tuple(fk.join_keys), _OBLIGATION_SOURCE, _OBLIGATION_CONFIDENCE,
            )
    for entity in normalized.entities:
        for declared in entity.declared_joins:
            yield _AuthoredJoin(
                entity.ref.qualified_name, declared.referenced_entity,
                tuple(declared.join_keys), _DECLARED_SOURCE,
                DECLARED_JOIN_CONFIDENCE,
            )


def _path_name(item: _BoundJoin, shared_pair: bool) -> str:
    # A pair with one join keeps the original name so existing graphs keep
    # their path identities; only pairs with several joins add the key columns.
    name = f"{item.referenced_table}__{item.entity_table}"
    if not shared_pair:
        return name
    columns = "__".join(f"{local}_{remote}" for local, remote in item.join.join_keys)
    return f"{name}__{columns}"


def _write_join_path(
    loader: GraphLoader,
    binding: PhysicalSchemaBinding,
    item: _BoundJoin,
    name: str,
) -> None:
    entity_ref = _table_ref(binding, item.entity_table)
    referenced_ref = _table_ref(binding, item.referenced_table)
    predicates = [
        {
            "left_table": entity_ref,
            "left_column": local,
            "right_table": referenced_ref,
            "right_column": remote,
            "operator": "=",
        }
        for local, remote in item.join.join_keys
    ]
    loader.upsert_join_path(
        name, predicates, hop_count=1, source=item.join.source,
        confidence=item.join.confidence, source_schema=binding.schema,
    )
    for table in (item.entity_table, item.referenced_table):
        loader.add_join_path_uses(
            name, _table_key(binding, table), source_schema=binding.schema
        )


def _table_ref(binding: PhysicalSchemaBinding, table: str) -> str:
    return f"{binding.catalog}.{binding.schema}.{table}"


def _table_key(binding: PhysicalSchemaBinding, table: str) -> PhysicalKey:
    # The binding names no datasource; the graph writers match catalog, schema, table.
    return PhysicalKey("", binding.catalog, binding.schema, table)


__all__ = [
    "DECLARED_JOIN_CONFIDENCE",
    "JoinPathCounts",
    "materialize_target_join_paths",
]
