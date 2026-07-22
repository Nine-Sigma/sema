"""S1-08 — the FK-closed OMOP-shape materialization chain (DuckDB or Databricks).

Wires the independently-built S1-01…S1-07 units into one ordered pass that lands
a production-shaped ``omop.person`` + ``omop.condition_occurrence`` for one study:

    resolve identity (S1-02) -> registry get-or-create (S1-01)
        -> [bridge registry into Databricks (S1-08)]
        -> FK-closed materialize (S1-06) -> Gate-D-lite (S1-07)

``backend`` selects the warehouse. DuckDB (the fit engine) reads the registry
in-place — the registry connection IS the target connection, and no bridge runs.
Databricks (the live engine) can only join a co-located registry, so the
DuckDB-canonical registry is bridged into a Delta table first (``bridge=True``).

Generic spine (D6/R29): every OMOP physical name arrives via specs from the
allowlisted policy layer; this module names no OMOP/showcase literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_backend import DUCKDB_FK_BACKEND, FkBackend, FkCursor
from sema.compile.fk_closed_compiler import FkClosedCompiler, FkClosedResult
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
)
from sema.eval.staging_qa import run_fk_closed_qa
from sema.eval.staging_qa_utils import StagingQAReport
from sema.resolve.identity_bridge import bridge_identity_registry
from sema.resolve.identity_registry import IdentityRegistry
from sema.resolve.identity_resolver import (
    DeterministicIdentityResolver,
    IdentitySourceRow,
)

__all__ = ["OmopShapeRequest", "OmopShapeResult", "run_omop_shape_fit"]


@dataclass(frozen=True)
class OmopShapeRequest:
    """Everything the FK-closed OMOP-shape chain needs, all supplied as data."""

    source: ChildSourceSpec
    source_row_count: int
    distinct_patient_keys: Sequence[str]
    parent: ParentTableSpec
    child: ChildTableSpec
    registry_spec: RegistryJoinSpec
    decisions: Sequence[StagingDecision]
    required_fields: Sequence[str]
    no_map_default: int
    missing_key_reason: str
    run_id: str


@dataclass(frozen=True)
class OmopShapeResult:
    """The artifacts and verdicts produced by one FK-closed materialization."""

    fk: FkClosedResult
    qa: StagingQAReport
    review_count: int
    registry_rows_bridged: int


def run_omop_shape_fit(
    request: OmopShapeRequest,
    *,
    registry: IdentityRegistry,
    target_cursor: FkCursor,
    backend: FkBackend = DUCKDB_FK_BACKEND,
    bridge: bool = False,
) -> OmopShapeResult:
    """Resolve identity, (bridge), materialize the FK-closed shape, run Gate-D-lite."""
    resolver = DeterministicIdentityResolver(
        registry, missing_key_reason=request.missing_key_reason
    )
    identity_rows = [
        IdentitySourceRow(
            source_namespace=request.source.schema,
            source_entity_key=key,
            source_row_ref=key,
        )
        for key in request.distinct_patient_keys
    ]
    resolution = resolver.resolve(identity_rows, run_id=request.run_id)

    bridged = 0
    if bridge:
        bridged = bridge_identity_registry(
            target_cursor,
            registry.read_all(),
            schema=request.registry_spec.schema,
            table=request.registry_spec.table,
        )

    fk = FkClosedCompiler(
        no_map_default=request.no_map_default, backend=backend
    ).materialize(
        target_cursor,
        parent=request.parent,
        child=request.child,
        source=request.source,
        registry=request.registry_spec,
        decisions=request.decisions,
    )
    qa = run_fk_closed_qa(
        target_cursor,
        parent=request.parent,
        child=request.child,
        source=request.source,
        required_fields=request.required_fields,
        source_row_count=request.source_row_count,
        backend=backend,
    )
    return OmopShapeResult(
        fk=fk,
        qa=qa,
        review_count=resolution.review_count,
        registry_rows_bridged=bridged,
    )
