"""US-014 integration: the no-vocab dim_customer target end-to-end on DuckDB.

Loads ``dim_customer.yaml`` (hermetic InMemoryGraphWriter), then compiles +
executes the projection against a temp DuckDB loaded from the ``customers.csv``
fixture. Asserts the rows materialise with exactly the target columns (no
vocabulary columns) and that the R29 coupling guard stays green. Uses a temp
DuckDB file, so it always runs under ``-m integration`` (no external resource).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from scripts.check_engine_coupling import find_violations
from sema.compile.compiler import TransformCompiler
from sema.models.planner._enums import MaterializationMode, PrimaryKeyStrategy
from sema.models.planner.field_map import RowIdentity
from sema.models.planner.lifecycle import PlanVerdict, Status
from sema.models.planner.mapping_plan import MappingAssertion
from sema.models.planner.patterns import (
    ConstantValue,
    DerivedExpression,
    DirectCopyPayload,
    MappingPattern,
)
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.models.planner.target_model import TargetObligation
from sema.resolve.assembler import Slice0PlanAssembler
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.loader import load_target
from sema.targets.materializer import InMemoryGraphWriter
from sema.targets.materializer_ops import VocabularyBindingOp

from datetime import datetime, timezone

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
MANIFEST = _REPO / "src" / "sema" / "targets" / "manifests" / "dim_customer.yaml"
CSV = _REPO / "tests" / "data" / "fixtures" / "customers.csv"

_TARGET_COLUMNS = ["customer_key", "customer_name", "signup_year", "source_system"]


def _assertions() -> list[MappingAssertion]:
    prov = Provenance(
        run=RunProvenance(
            run_id="us014-live",
            target_model_version="dim-customer-0.1.0",
            target_schema_snapshot_hash="t",
            context_card_version="cc",
            prompt_template_version="pt",
            few_shot_set_version="fs",
            constraint_version="cv",
            llm_model="none",
        ),
        source=SourceScope(
            source_id="customers", source_schema_hash="s", source_profile_hash="p"
        ),
        timestamp=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )
    common = dict(confidence=1.0, provenance=prov, status=Status.auto_accepted)
    return [
        MappingAssertion(
            id="a1",
            source_field_ref="source.customers.customer_id",
            target_property_ref="target.dim_customer.customer_key",
            pattern=MappingPattern.DIRECT_COPY,
            payload=DirectCopyPayload(source_field_ref="source.customers.customer_id"),
            **common,
        ),
        MappingAssertion(
            id="a2",
            source_field_ref="source.customers.full_name",
            target_property_ref="target.dim_customer.customer_name",
            pattern=MappingPattern.DIRECT_COPY,
            payload=DirectCopyPayload(source_field_ref="source.customers.full_name"),
            **common,
        ),
        MappingAssertion(
            id="a3",
            source_field_ref="source.customers.signup_year_text",
            target_property_ref="target.dim_customer.signup_year",
            pattern=MappingPattern.DERIVED,
            payload=DerivedExpression(
                source_field_refs=["source.customers.signup_year_text"],
                expression_ast={
                    "cast": {
                        "source_field_ref": "source.customers.signup_year_text",
                        "to_type": "INTEGER",
                    }
                },
            ),
            **common,
        ),
        MappingAssertion(
            id="a4",
            source_field_ref="source.customers.customer_id",
            target_property_ref="target.dim_customer.source_system",
            pattern=MappingPattern.CONSTANT,
            payload=ConstantValue(literal_value="CRM", target_type="string"),
            **common,
        ),
    ]


def _obligation() -> TargetObligation:
    return TargetObligation(
        target_entity="target.dim_customer",
        required_fields=[f"target.dim_customer.{c}" for c in _TARGET_COLUMNS],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )


def _row_identity() -> RowIdentity:
    return RowIdentity(
        target_row_key_rule="source.customers.customer_id",
        source_lineage=["source.customers.customer_id"],
        materialization_mode=MaterializationMode.REPLACE_PARTITION,
    )


def test_no_vocab_manifest_loads_without_binding() -> None:
    writer = InMemoryGraphWriter()
    load_target(ManifestTargetAdapter(MANIFEST), writer=writer)
    assert not [op for op in writer.ops if isinstance(op, VocabularyBindingOp)]


def test_dim_customer_flows_through_spine_to_staging(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "dim.duckdb"))
    conn.execute('CREATE SCHEMA "raw"')
    conn.execute(
        f"CREATE TABLE \"raw\".\"customers\" AS "
        f"SELECT * FROM read_csv_auto('{CSV}', header=true)"
    )
    source_count = conn.execute('SELECT COUNT(*) FROM "raw"."customers"').fetchone()[0]

    plan = Slice0PlanAssembler().assemble(
        _assertions(), _obligation(), _row_identity()
    )
    assert plan.derive_verdict() is PlanVerdict.compilable

    compiler = TransformCompiler()
    compiled = compiler.compile_projection(
        plan, source_schema="raw", source_table="customers"
    )
    written = compiler.execute_projection(
        conn,
        compiled,
        staging_schema="analytics_staging",
        staging_table="dim_customer",
    )
    assert written == source_count

    cols = [
        r[1]
        for r in conn.execute(
            'PRAGMA table_info("analytics_staging"."dim_customer")'
        ).fetchall()
    ]
    assert cols == _TARGET_COLUMNS

    rows = conn.execute(
        "SELECT customer_key, customer_name, signup_year, source_system "
        'FROM "analytics_staging"."dim_customer" ORDER BY customer_key'
    ).fetchall()
    assert rows[0] == (1, "Ada Lovelace", 1843, "CRM")
    assert all(isinstance(r[2], int) for r in rows)  # DERIVED cast applied
    assert {r[3] for r in rows} == {"CRM"}  # CONSTANT


def test_r29_guard_stays_green() -> None:
    assert find_violations() == []
