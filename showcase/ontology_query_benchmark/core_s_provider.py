"""Arm S's context from SEMA's own query path, for a connector-built graph.

The text is the core prompt's context sections for the question's pruned SCO, without the
core frame, rules or question (the harness adds its own). The DuckDB dialect is named
because the sealed-set runner executes through the DuckDB connector.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sema.connectors.duckdb import DuckDBConnector
from sema.consumers.nl2sql.prompting import render_sections
from sema.models.context import (
    JoinPredicate,
    PhysicalAsset,
    ResolvedEntity,
    ResolvedProperty,
    SemanticContextObject,
)
from sema.models.target.identity import EntityIdentity
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine


@dataclass(frozen=True)
class CoreSContext:
    text: str
    identity: str
    findings: tuple[Any, ...] = ()


class CoreSContextProvider:
    """An `SProvider`: `(question, is_denied) -> SContextLike` through the core query path."""

    def __init__(self, driver: Any, embedder: Any = None) -> None:
        self.engine = RetrievalEngine(driver, embedder=embedder)

    def __call__(self, question: str, is_denied: Callable[[str], bool]) -> CoreSContext:
        sco = without_denied(prune_to_sco(self.engine.retrieve(question)), is_denied)
        text = "\n".join(render_sections(sco, DuckDBConnector.sql_dialect or ""))
        return CoreSContext(text, hashlib.sha256(text.encode()).hexdigest())


def without_denied(sco: SemanticContextObject,
                   is_denied: Callable[[str], bool]) -> SemanticContextObject:  # fmt: skip
    """The SCO with every denied column removed, and every fact that names one."""

    def keep(*columns: str) -> bool:
        return not any(is_denied(c) for c in columns)

    return sco.model_copy(update={
        "physical_assets": [_asset_kept(a, keep) for a in sco.physical_assets],
        "entities": [_entity_kept(e, keep) for e in sco.entities],
        "governed_values": [g for g in sco.governed_values if keep(g.column)],
        "join_paths": [j for j in sco.join_paths if all(
            keep(p.left_column, p.right_column) for p in j.join_predicates)],
        "hierarchy_anchors": [h for h in sco.hierarchy_anchors if keep(h.column)],
        "column_matches": [m for m in sco.column_matches
                           if keep(m.column, m.alternative_column)],
        "result_shape": sco.result_shape and sco.result_shape.model_copy(update={
            "columns": [c for c in sco.result_shape.columns if keep(c)]}),
    })  # fmt: skip


Keep = Callable[..., bool]


def _asset_kept(asset: PhysicalAsset, keep: Keep) -> PhysicalAsset:
    """A key or a shared entity that names a denied column is dropped whole."""
    return asset.model_copy(update={
        "columns": [c for c in asset.columns if keep(c)],
        "key_columns": asset.key_columns if keep(*asset.key_columns) else [],
        "shared_entities": [s.model_copy(update={"identity": _rules_kept(s.identity, keep)})
                            for s in asset.shared_entities if keep(*s.key_columns)],
    })


def _entity_kept(entity: ResolvedEntity, keep: Keep) -> ResolvedEntity:
    """The entity's identity goes when a key column is denied, and an ordering rule goes when
    a column it reads is denied."""
    properties = [_property_kept(p, keep) for p in entity.properties if keep(p.physical_column)]
    identity = entity.identity
    if not keep(*entity.key_columns, *(identity.key_columns if identity else ())):
        return entity.model_copy(update={
            "properties": properties, "key_columns": [], "identity": None, "null_key_rows": None})
    return entity.model_copy(update={"properties": properties,
                                     "identity": _rules_kept(identity, keep)})


def _rules_kept(identity: EntityIdentity | None, keep: Keep) -> EntityIdentity | None:
    """An ordering rule goes when a column it reads is denied."""
    if identity is None:
        return None
    return identity.model_copy(update={
        "attributes": tuple(a for a in identity.attributes if keep(*a.columns))})


def _property_kept(prop: ResolvedProperty, keep: Keep) -> ResolvedProperty:
    return prop.model_copy(update={"labels": [
        label for label in prop.labels if keep(label.column, *_columns(label.via))]})


def _columns(predicates: list[JoinPredicate]) -> list[str]:
    return [c for p in predicates for c in (p.left_column, p.right_column)]
