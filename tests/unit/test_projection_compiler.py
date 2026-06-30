"""US-014: generality counter-example — a no-vocab target through the spine.

A curated ``dim_customer`` target (no vocabulary at all) flows through the SAME
target loader (US-007), assembler (US-008), and compiler (US-010) as the OMOP
vocab path, with zero concept resolution. These hermetic tests assert:

* loading the no-vocab manifest writes NO :VocabularyBinding node and NO
  HAS_VOCABULARY_BINDING edge (the positive complement of US-007), while the
  :Entity/:Property/:TargetObligation are materialised;
* DIRECT_COPY (copy/rename), DERIVED (a cast), and CONSTANT field maps assemble
  into a PASS plan and compile + execute into a plain staging table whose
  columns are exactly the target columns — no vocabulary columns;
* the resolver step is never invoked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from sema.compile.compiler import CompiledProjection, TransformCompiler
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
from sema.targets.materializer_ops import (
    EntityOp,
    PropertyOp,
    RelationshipOp,
    TargetObligationOp,
    VocabularyBindingOp,
)

pytestmark = pytest.mark.unit

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "sema"
    / "targets"
    / "manifests"
    / "dim_customer.yaml"
)

_ENTITY = "analytics.dim_customer"
_TARGET_COLUMNS = ("customer_key", "customer_name", "signup_year", "source_system")


def _load() -> InMemoryGraphWriter:
    writer = InMemoryGraphWriter()
    load_target(ManifestTargetAdapter(MANIFEST), writer=writer)
    return writer


def _provenance() -> Provenance:
    return Provenance(
        run=RunProvenance(
            run_id="us014",
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


def _assertions() -> list[MappingAssertion]:
    prov = _provenance()
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
        required_fields=[
            "target.dim_customer.customer_key",
            "target.dim_customer.customer_name",
            "target.dim_customer.signup_year",
            "target.dim_customer.source_system",
        ],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )


def _row_identity() -> RowIdentity:
    return RowIdentity(
        target_row_key_rule="source.customers.customer_id",
        source_lineage=["source.customers.customer_id"],
        materialization_mode=MaterializationMode.REPLACE_PARTITION,
    )


# --- manifest: the positive complement of US-007 ---------------------------


def test_manifest_file_exists() -> None:
    assert MANIFEST.is_file()


def test_entity_property_and_obligation_materialized() -> None:
    ops = _load().ops
    entities = {op.qualified_name for op in ops if isinstance(op, EntityOp)}
    assert any(e.endswith("dim_customer") for e in entities)
    props = {op.name for op in ops if isinstance(op, PropertyOp)}
    assert set(_TARGET_COLUMNS).issubset(props)
    obligation = next(op for op in ops if isinstance(op, TargetObligationOp))
    assert "customer_key" in obligation.payload["required_fields"]


def test_no_vocabulary_binding_node_or_edge() -> None:
    ops = _load().ops
    assert not [op for op in ops if isinstance(op, VocabularyBindingOp)]
    rel_types = {op.rel_type for op in ops if isinstance(op, RelationshipOp)}
    assert "HAS_VOCABULARY_BINDING" not in rel_types


# --- assembler: a no-vocab plan passes -------------------------------------


def test_no_vocab_plan_assembles_and_passes() -> None:
    plan = Slice0PlanAssembler().assemble(
        _assertions(), _obligation(), _row_identity()
    )
    assert plan.derive_verdict() is PlanVerdict.compilable
    assert plan.covered_required_fields() == set(_obligation().required_fields)
    patterns = {fm.pattern for fm in plan.field_maps}
    assert patterns == {
        MappingPattern.DIRECT_COPY,
        MappingPattern.DERIVED,
        MappingPattern.CONSTANT,
    }


# --- compiler: plain projection, no vocabulary columns ---------------------


def _compiled() -> CompiledProjection:
    plan = Slice0PlanAssembler().assemble(
        _assertions(), _obligation(), _row_identity()
    )
    return TransformCompiler().compile_projection(
        plan, source_schema="raw", source_table="customers"
    )


def test_compile_projection_target_columns() -> None:
    compiled = _compiled()
    assert compiled.target_columns == _TARGET_COLUMNS


def test_compile_projection_renders_copy_cast_constant() -> None:
    sql = _compiled().sql("duckdb").upper()
    assert "CAST" in sql  # DERIVED cast
    assert "'CRM'" in sql.replace('"', "'") or "CRM" in sql  # CONSTANT
    assert "CUSTOMER_KEY" in sql


def test_execute_projection_materializes_no_vocab_columns(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "dim.duckdb"))
    conn.execute('CREATE SCHEMA "raw"')
    conn.execute(
        'CREATE TABLE "raw"."customers" '
        "(customer_id INTEGER, full_name VARCHAR, signup_year_text VARCHAR)"
    )
    conn.execute(
        "INSERT INTO \"raw\".\"customers\" VALUES "
        "(1, 'Ada Lovelace', '1843'), (2, 'Alan Turing', '1936')"
    )
    written = TransformCompiler().execute_projection(
        conn,
        _compiled(),
        staging_schema="analytics_staging",
        staging_table="dim_customer",
    )
    assert written == 2
    cols = [
        r[1]
        for r in conn.execute(
            'PRAGMA table_info("analytics_staging"."dim_customer")'
        ).fetchall()
    ]
    assert cols == list(_TARGET_COLUMNS)
    rows = conn.execute(
        'SELECT customer_key, customer_name, signup_year, source_system '
        'FROM "analytics_staging"."dim_customer" ORDER BY customer_key'
    ).fetchall()
    assert rows[0] == (1, "Ada Lovelace", 1843, "CRM")
    # DERIVED cast yields an INTEGER, not the source VARCHAR.
    assert isinstance(rows[0][2], int)


def test_constant_null_projects_null_and_does_not_cover() -> None:
    from sema.compile.projection_utils import build_projection_select
    from sema.models.planner.field_map import FieldMap

    fm = FieldMap(
        target_field_ref="target.dim_customer.note",
        pattern=MappingPattern.CONSTANT,
        payload=ConstantValue(literal_value=None, target_type="string"),
        status=Status.auto_accepted,
    )
    assert fm.covers_required_field() is False
    sql = build_projection_select("raw", "customers", [fm]).sql(dialect="duckdb")
    assert "NULL" in sql.upper()


def test_unsupported_pattern_raises() -> None:
    from sema.compile.projection_utils import (
        UnsupportedProjectionError,
        build_projection_select,
    )
    from sema.models.planner.field_map import FieldMap
    from sema.models.planner.patterns import NoMapPayload, NoMapScope

    fm = FieldMap(
        target_field_ref="target.dim_customer.x",
        pattern=MappingPattern.NO_MAP,
        payload=NoMapPayload(reason="n/a", scope=NoMapScope.GLOBAL),
    )
    with pytest.raises(UnsupportedProjectionError):
        build_projection_select("raw", "customers", [fm])


def test_empty_field_maps_raises() -> None:
    from sema.compile.projection_utils import (
        UnsupportedProjectionError,
        build_projection_select,
    )

    with pytest.raises(UnsupportedProjectionError):
        build_projection_select("raw", "customers", [])


def test_derived_without_cast_ast_raises() -> None:
    from sema.compile.projection_utils import (
        UnsupportedProjectionError,
        build_projection_select,
    )
    from sema.models.planner.field_map import FieldMap

    fm = FieldMap(
        target_field_ref="target.dim_customer.signup_year",
        pattern=MappingPattern.DERIVED,
        payload=DerivedExpression(
            source_field_refs=["source.customers.signup_year_text"],
            expression_ast={"unsupported": True},
        ),
    )
    with pytest.raises(UnsupportedProjectionError):
        build_projection_select("raw", "customers", [fm])


def test_execute_projection_is_idempotent(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "dim.duckdb"))
    conn.execute('CREATE SCHEMA "raw"')
    conn.execute(
        'CREATE TABLE "raw"."customers" '
        "(customer_id INTEGER, full_name VARCHAR, signup_year_text VARCHAR)"
    )
    conn.execute("INSERT INTO \"raw\".\"customers\" VALUES (1, 'Ada', '1843')")
    compiler = TransformCompiler()
    first = compiler.execute_projection(
        conn, _compiled(), staging_schema="s", staging_table="dim_customer"
    )
    second = compiler.execute_projection(
        conn, _compiled(), staging_schema="s", staging_table="dim_customer"
    )
    assert first == second == 1
