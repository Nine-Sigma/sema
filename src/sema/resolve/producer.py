"""US-009: deterministic VOCAB_LOOKUP assertion + graph-edge producer.

Reads per-code decisions from the value-mapping store (written solely by
US-006 — this producer NEVER writes the store), emits one column-level
``MappingAssertion(pattern=VOCAB_LOOKUP)``, and materialises the ``:FieldMap``
with ``MAPS_TO`` -> the target ``:Property`` and ``DERIVED_FROM`` -> the source
``:Property`` (the ``MAPS_TO`` edge always originates at the ``:FieldMap``).

Generic spine module: it names no domain literal — refs and node ids are passed
in as data. The graph writes reuse :mod:`sema.graph.planner_loader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sema.graph.planner_loader import (
    cypher_create_field_map_derived_from,
    cypher_create_field_map_maps_to,
    write_field_map,
)
from sema.models.planner._enums import ModelRole
from sema.models.planner.field_map import FieldMap
from sema.models.planner.mapping_plan import MappingAssertion
from sema.resolve.engine_utils import ResolveContext
from sema.resolve.policy import ResolverPolicy
from sema.resolve.producer_utils import (
    NoResolvedDecisionError,
    build_column_assertion,
    decisions_for_binding,
    field_map_from_assertion,
)
from sema.resolve.value_mapping_store_utils import ValueMapping

__all__ = ["MappingNodes", "NoResolvedDecisionError", "VocabLookupProducer"]


@dataclass(frozen=True)
class MappingNodes:
    """Graph identities of the source/target :Property nodes being linked."""

    source_property_id: str
    target_property_id: str


class VocabLookupProducer:
    """Turns this run's value-mapping decisions into one graph mapping (US-009)."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def produce(
        self,
        decisions: Sequence[ValueMapping],
        policy: ResolverPolicy,
        context: ResolveContext,
        nodes: MappingNodes,
    ) -> MappingAssertion:
        # ``decisions`` are THIS run's mappings (the caller's scope), never the
        # whole store: folding store.read_all() would let a stale same-policy row
        # leak its status/confidence into the assertion — or mask an empty run.
        scoped = decisions_for_binding(list(decisions), policy, context)
        assertion = build_column_assertion(scoped, policy, context)
        field_map = field_map_from_assertion(assertion)
        self._write_graph(field_map, _field_map_id(assertion), nodes)
        return assertion

    def _write_graph(
        self, field_map: FieldMap, field_map_id: str, nodes: MappingNodes
    ) -> None:
        write_field_map(self._session, field_map, field_map_id=field_map_id)
        edges = (
            cypher_create_field_map_maps_to(
                field_map_id, nodes.target_property_id, ModelRole.TARGET
            ),
            cypher_create_field_map_derived_from(
                field_map_id, nodes.source_property_id, ModelRole.SOURCE
            ),
        )
        for stmt, params in edges:
            self._session.run(stmt, **params)


def _field_map_id(assertion: MappingAssertion) -> str:
    return f"field_map:{assertion.id}"
