"""Project authored FK obligations into retrieval `:JoinPath` nodes (M5).

Retrieval reads joins only from ``(:JoinPath)-[:USES]->(:Table)`` — the target
materializer writes ``:TargetObligation`` instead, so FK joins are invisible.
This is a *deterministic manifest projection* (not heuristic FK detection,
which stays Slice-3): each declared foreign key becomes one ``:JoinPath`` with
``USES`` edges to both physical tables.

Generic: reads the normalized model's obligations + a physical schema binding;
names no target domain.
"""

from __future__ import annotations

from sema.graph.loader import GraphLoader
from sema.graph.target_retrieval_loader import PhysicalSchemaBinding
from sema.models.planner.target_model import ForeignKeyObligation
from sema.models.target.normalized import NormalizedTargetModel

_SOURCE = "manifest"
_CONFIDENCE = 1.0


def materialize_target_join_paths(
    loader: GraphLoader,
    normalized: NormalizedTargetModel,
    binding: PhysicalSchemaBinding,
) -> None:
    for obligation in normalized.obligations:
        entity_table = binding.table_for(obligation.target_entity)
        if entity_table is None:
            continue
        for fk in obligation.foreign_keys:
            referenced_table = binding.table_for(fk.referenced_entity)
            if referenced_table is None:
                continue
            _write_join_path(
                loader, binding, entity_table, referenced_table, fk
            )


def _write_join_path(
    loader: GraphLoader,
    binding: PhysicalSchemaBinding,
    entity_table: str,
    referenced_table: str,
    fk: ForeignKeyObligation,
) -> None:
    entity_ref = _table_ref(binding, entity_table)
    referenced_ref = _table_ref(binding, referenced_table)
    name = f"{referenced_table}__{entity_table}"
    predicates = [
        {
            "left": f"{entity_ref}.{local}",
            "right": f"{referenced_ref}.{referenced}",
            "op": "=",
        }
        for local, referenced in fk.join_keys
    ]
    loader.upsert_join_path(
        name, predicates, hop_count=1, source=_SOURCE,
        confidence=_CONFIDENCE, source_schema=binding.schema,
    )
    loader.add_join_path_uses(name, entity_ref, source_schema=binding.schema)
    loader.add_join_path_uses(
        name, referenced_ref, source_schema=binding.schema
    )


def _table_ref(binding: PhysicalSchemaBinding, table: str) -> str:
    return f"{binding.catalog}.{binding.schema}.{table}"


__all__ = ["materialize_target_join_paths"]
