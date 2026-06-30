"""US-012A: the DuckDB end-to-end fit chain (no Databricks).

``run_fit`` wires the independently-built spine units into one local pass so
interface drift between them is caught before the live Databricks gate (US-013):

    resolve (US-006) -> produce (US-009) -> assemble (US-008)
        -> compile + staging-write (US-010) -> Gate D-lite QA (US-011)
        -> eval report (US-012)

It writes NO Databricks objects: the value-mapping store and the §1.5(b)
staging table both live in DuckDB. The TARGET binding is consumed as data
(built by :mod:`sema.pipeline.fit_slice0_utils` from the already-authored
manifest) — this module re-loads nothing and names no domain literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from sema.compile.compiler import TransformCompiler
from sema.compile.compiler_utils import (
    CompileContext,
    SourceTableSpec,
    StagingColumns,
)
from sema.compile.staging_backend import (
    DUCKDB_BACKEND,
    StagingBackend,
    StagingCursor,
)
from sema.eval.mapping_goldset import GoldSet
from sema.eval.mapping_report import build_mapping_report, decisions_from_store
from sema.eval.mapping_report_utils import MappingReport
from sema.eval.staging_qa import run_staging_qa
from sema.eval.staging_qa_utils import StagingQAReport
from sema.models.planner.field_map import RowIdentity
from sema.models.planner.mapping_plan import MappingAssertion, MappingPlan
from sema.models.planner.target_model import TargetObligation
from sema.resolve.assembler import Slice0PlanAssembler
from sema.resolve.engine import VocabularyResolver
from sema.resolve.engine_utils import (
    ResolveContext,
    staging_decision_from_value_mapping,
)
from sema.resolve.policy import ResolverPolicy
from sema.resolve.producer import MappingNodes, VocabLookupProducer
from sema.resolve.value_mapping_store import ValueMappingStore

__all__ = ["FitRequest", "FitResult", "run_fit"]


@dataclass(frozen=True)
class FitRequest:
    """Everything the local fit chain needs, all supplied as data."""

    source: SourceTableSpec
    source_codes: list[str]
    source_row_count: int
    policy: ResolverPolicy
    resolve_context: ResolveContext
    compile_context: CompileContext
    staging_columns: StagingColumns
    obligation: TargetObligation
    row_identity: RowIdentity
    staging_schema: str
    staging_table: str
    gold: GoldSet
    nodes: MappingNodes


@dataclass(frozen=True)
class FitResult:
    """The artifacts and verdicts produced by one local fit pass."""

    assertion: MappingAssertion
    plan: MappingPlan
    rows_staged: int
    source_row_count: int
    qa: StagingQAReport
    report: MappingReport
    staging_schema: str
    staging_table: str
    store_columns: list[str]


class _RecordingSession:
    """In-memory stand-in for a Neo4j session (the smoke writes no graph).

    The producer (US-009) materialises ``:FieldMap``/``MAPS_TO``/``DERIVED_FROM``
    against a graph session; for the DuckDB-only gate we record the writes
    instead of requiring Neo4j. The producer's assertion/field-map projection is
    still exercised, so drift in those shapes is caught.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def run(self, statement: str, **params: Any) -> None:
        self.writes.append((statement, params))


def run_fit(
    resolver: VocabularyResolver,
    request: FitRequest,
    *,
    value_mapping_conn: duckdb.DuckDBPyConnection,
    staging_conn: StagingCursor,
    staging_backend: StagingBackend = DUCKDB_BACKEND,
) -> FitResult:
    """Run resolve -> produce -> assemble -> compile -> QA -> eval.

    ``staging_backend`` selects the warehouse the §1.5(b) staging table is
    written to (DuckDB by default, Databricks for the live US-013 run). The
    value-mapping store stays on DuckDB (its canonical home, US-005) regardless.
    """
    ctx = request.resolve_context
    store = ValueMappingStore(value_mapping_conn)
    resolver.resolve_and_store(request.source_codes, store, ctx)

    producer = VocabLookupProducer(_RecordingSession())
    assertion = producer.produce(store, request.policy, ctx, request.nodes)
    plan = Slice0PlanAssembler().assemble(
        [assertion], request.obligation, request.row_identity
    )

    decisions = [
        staging_decision_from_value_mapping(mapping) for mapping in store.read_all()
    ]
    compiler = TransformCompiler()
    compiled = compiler.compile(
        plan, request.staging_columns, request.source, request.compile_context, decisions
    )
    rows_staged = compiler.execute(
        staging_conn,
        compiled,
        columns=request.staging_columns,
        source=request.source,
        staging_schema=request.staging_schema,
        staging_table=request.staging_table,
        backend=staging_backend,
    )

    qa = run_staging_qa(
        staging_conn,
        columns=request.staging_columns,
        staging_schema=request.staging_schema,
        staging_table=request.staging_table,
        source_schema=request.source.schema,
        source_table=request.source.table,
        expected_row_count=request.source_row_count,
        gold_set=request.gold,
        backend=staging_backend,
    )

    report = build_mapping_report(
        request.gold,
        decisions_from_store(
            store,
            target_property_ref=ctx.target_property_ref,
            resolver_policy_ref=ctx.resolver_policy_ref,
            vocab_release=ctx.vocab_release,
        ),
    )
    return FitResult(
        assertion=assertion,
        plan=plan,
        rows_staged=rows_staged,
        source_row_count=request.source_row_count,
        qa=qa,
        report=report,
        staging_schema=request.staging_schema,
        staging_table=request.staging_table,
        store_columns=store.column_names(),
    )
