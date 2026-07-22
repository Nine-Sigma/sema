"""FK-closed two-table fit + Stage B identity collapse (DuckDB or Databricks).

Generic platform capability: wires the identity engine, FK-closed compiler, and
Gate-D-lite into one ordered pass that lands an FK-target parent + an FK-closed
child for one source, and a Stage B pass that collapses shared identities and
rebuilds the affected sources:

    resolve identity -> registry get-or-create
        -> [bridge registry into Databricks]
        -> FK-closed materialize -> Gate-D-lite

``backend`` selects the warehouse. DuckDB (the fit engine) reads the registry
in-place — the registry connection IS the target connection, and no bridge runs.
Databricks (the live engine) can only join a co-located registry, so the
DuckDB-canonical registry is bridged into a Delta table first (``bridge=True``).

Domain-generic (D6/R29): every physical name arrives via specs supplied by the
caller (a showcase policy layer); this module names no target-model literal.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from sema.resolve.identity_collapse import CollapseResult, collapse_identities
from sema.resolve.identity_registry import IdentityRegistry
from sema.resolve.identity_resolver import (
    DeterministicIdentityResolver,
    IdentitySourceRow,
)

__all__ = [
    "FkClosedFitRequest",
    "FkClosedFitResult",
    "StageBResult",
    "run_fk_closed_fit",
    "run_stage_b_collapse",
]


@dataclass(frozen=True)
class FkClosedFitRequest:
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
class FkClosedFitResult:
    """The artifacts and verdicts produced by one FK-closed materialization."""

    fk: FkClosedResult
    qa: StagingQAReport
    review_count: int
    registry_rows_bridged: int


def run_fk_closed_fit(
    request: FkClosedFitRequest,
    *,
    registry: IdentityRegistry,
    target_cursor: FkCursor,
    backend: FkBackend = DUCKDB_FK_BACKEND,
    bridge: bool = False,
) -> FkClosedFitResult:
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
    return FkClosedFitResult(
        fk=fk,
        qa=qa,
        review_count=resolution.review_count,
        registry_rows_bridged=bridged,
    )


@dataclass(frozen=True)
class StageBResult:
    """The outcome of one Stage B collapse + rebuild over a multi-study corpus."""

    collapse: CollapseResult
    fk: FkClosedResult
    qa: Sequence[StagingQAReport]
    registry_rows_bridged: int


def _shape_key(request: FkClosedFitRequest) -> tuple[object, ...]:
    return (
        request.parent,
        request.child,
        request.registry_spec,
        tuple(request.decisions),
        request.no_map_default,
    )


def _require_uniform_shape(
    requests: Sequence[FkClosedFitRequest], head: FkClosedFitRequest
) -> None:
    """Fail if any request disagrees with ``head`` on the borrowed rebuild shape.

    ``rebuild_after_collapse`` applies ``head``'s parent/child/registry specs,
    decisions, and no-map default to every source. A divergent request would be
    silently rewritten under the head's policy, so reject it up front. Compared
    pairwise because ``decisions`` / ``null_columns`` are unhashable.
    """
    head_key = _shape_key(head)
    offenders = [
        r.source.schema for r in requests if _shape_key(r) != head_key
    ]
    if offenders:
        raise ValueError(
            "run_stage_b_collapse requires all studies to share the same shape "
            f"(parent/child/registry/decisions/no_map_default); diverging: {offenders}"
        )


def run_stage_b_collapse(
    requests: Sequence[FkClosedFitRequest],
    *,
    namespace_grouping: Mapping[str, str],
    registry: IdentityRegistry,
    target_cursor: FkCursor,
    backend: FkBackend = DUCKDB_FK_BACKEND,
    bridge: bool = False,
) -> StageBResult:
    """Collapse shared-identity persons, then rebuild every study's child rows.

    Deterministic same-namespace collapse (S1-10): remaps the registry's
    uid→entity_id level, then rebuilds ``condition_occurrence`` for all studies
    through the collapsed registry (PKs preserved, FKs recomputed) and shrinks
    ``person`` to the survivors. Idempotent — a re-run collapses nothing and
    rebuilds to identical counts. All studies must target the same shape, so the
    parent/child/registry specs are taken from the first request.
    """
    if not requests:
        raise ValueError("run_stage_b_collapse requires at least one study request")
    head = requests[0]
    _require_uniform_shape(requests, head)
    collapse = collapse_identities(registry, namespace_grouping=namespace_grouping)

    bridged = 0
    if bridge:
        bridged = bridge_identity_registry(
            target_cursor,
            registry.read_all(),
            schema=head.registry_spec.schema,
            table=head.registry_spec.table,
        )

    fk = FkClosedCompiler(
        no_map_default=head.no_map_default, backend=backend
    ).rebuild_after_collapse(
        target_cursor,
        parent=head.parent,
        child=head.child,
        sources=[r.source for r in requests],
        registry=head.registry_spec,
        decisions=head.decisions,
    )
    qa = [
        run_fk_closed_qa(
            target_cursor,
            parent=r.parent,
            child=r.child,
            source=r.source,
            required_fields=r.required_fields,
            source_row_count=r.source_row_count,
            backend=backend,
        )
        for r in requests
    ]
    return StageBResult(
        collapse=collapse, fk=fk, qa=qa, registry_rows_bridged=bridged
    )
