"""US-010: transform compiler (MappingPlan -> SQLGlot -> staging write).

Asserts §1.5(b): the compiler inlines resolved decisions as a ``JOIN (VALUES ...)``,
builds the AST once and renders BOTH DuckDB and Databricks dialects, projects
``<target_concept_column>`` = the store value (NULL for NO_MAP rows), and executes
an idempotent temp-build + scoped-swap against DuckDB. The compiler module names no
showcase literal; column names arrive from the policy-owned ``StagingColumns``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest
import sqlglot

pytestmark = pytest.mark.unit


# --- fixtures ---------------------------------------------------------------


def _provenance() -> object:
    from sema.models.planner.provenance import (
        Provenance,
        RunProvenance,
        SourceScope,
    )

    run = RunProvenance(
        run_id="run-1",
        target_model_version="omop-cdm-5.4",
        target_schema_snapshot_hash="t-abc",
        vocab_release="omop-2026-q1",
        context_card_version="cards-v3",
        prompt_template_version="tpl-7",
        few_shot_set_version="fs-12",
        constraint_version="rules-v2",
        llm_model="claude-opus-4.7",
        embedding_model="bge-large",
    )
    source = SourceScope(
        source_id="src_study",
        source_schema_hash="s-abc",
        source_profile_hash="p-abc",
    )
    return Provenance(
        run=run,
        source=source,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _vocab_lookup_plan() -> object:
    from sema.models.planner._enums import MaterializationMode, PrimaryKeyStrategy
    from sema.models.planner.field_map import RowIdentity
    from sema.models.planner.lifecycle import Status
    from sema.models.planner.mapping_plan import MappingAssertion
    from sema.models.planner.patterns import MappingPattern, VocabLookup
    from sema.models.planner.target_model import TargetObligation
    from sema.resolve.assembler import Slice0PlanAssembler

    assertion = MappingAssertion(
        id="a-cc",
        source_field_ref="src.sample.code",
        target_property_ref="target.stage.concept",
        pattern=MappingPattern.VOCAB_LOOKUP,
        payload=VocabLookup(
            vocabulary_ref="vocab.snomed",
            source_value_ref="src.sample.code",
            domain_constraint_ref="target.stage.domain=Condition",
            require_standard=True,
            allow_zero_default=False,
            resolver_policy_ref="omop.oncotree_to_snomed_condition",
        ),
        confidence=1.0,
        provenance=_provenance(),
        status=Status.auto_accepted,
    )
    obligation = TargetObligation(
        target_entity="target.stage",
        required_fields=["target.stage.concept"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    row_identity = RowIdentity(
        target_row_key_rule="src.sample.code",
        source_lineage=["src.sample.code"],
        materialization_mode=MaterializationMode.REPLACE_PARTITION,
    )
    return Slice0PlanAssembler().assemble([assertion], obligation, row_identity)


def _columns() -> object:
    from sema.compile.compiler_utils import StagingColumns

    return StagingColumns(
        source_value_column="source_oncotree_code",
        target_concept_column="condition_concept_id",
    )


def _source() -> object:
    from sema.compile.compiler_utils import SourceTableSpec

    return SourceTableSpec(
        schema="src_study",
        table="sample",
        value_column="ONCOTREE_CODE",
        patient_key_column="PATIENT_ID",
    )


def _context() -> object:
    from sema.compile.compiler_utils import CompileContext

    return CompileContext(
        resolver_policy_ref="omop.oncotree_to_snomed_condition",
        vocab_release="omop-2026-q1",
        run_id="run-1",
    )


def _decisions() -> list[object]:
    from sema.compile.compiler_utils import StagingDecision

    return [
        StagingDecision(
            normalized_source_value="LUAD",
            target_value=4314337,
            resolution_status="RESOLVED",
            no_map_reason=None,
            status="auto_accepted",
        ),
        StagingDecision(
            normalized_source_value="ZZZZ",
            target_value=None,
            resolution_status="NO_MAP",
            no_map_reason="no standard target concept survived the domain gate",
            status="auto_accepted",
        ),
    ]


# --- AST / dialect rendering ------------------------------------------------


class TestCompileSql:
    def test_inlines_decisions_as_values_join(self) -> None:
        from sema.compile.compiler import TransformCompiler

        compiled = TransformCompiler().compile(
            _vocab_lookup_plan(), _columns(), _source(), _context(), _decisions()
        )
        sql = compiled.sql("duckdb")
        assert "VALUES" in sql
        assert "'LUAD'" in sql and "4314337" in sql
        assert "LEFT JOIN" in sql

    def test_renders_both_dialects_from_one_ast(self) -> None:
        from sema.compile.compiler import TransformCompiler

        compiled = TransformCompiler().compile(
            _vocab_lookup_plan(), _columns(), _source(), _context(), _decisions()
        )
        duck = compiled.sql("duckdb")
        dbx = compiled.sql("databricks")
        assert duck != dbx
        # both parse under their own dialect
        sqlglot.parse_one(duck, dialect="duckdb")
        sqlglot.parse_one(dbx, dialect="databricks")

    def test_projects_policy_named_columns(self) -> None:
        from sema.compile.compiler import TransformCompiler

        compiled = TransformCompiler().compile(
            _vocab_lookup_plan(), _columns(), _source(), _context(), _decisions()
        )
        sql = compiled.sql("duckdb")
        assert "AS source_oncotree_code" in sql
        assert "AS condition_concept_id" in sql
        assert "AS resolver_policy_ref" in sql

    def test_unsupported_pattern_raises(self) -> None:
        from sema.compile.compiler import TransformCompiler, UnsupportedTransformError
        from sema.models.planner._enums import TargetArtifactKind

        with pytest.raises(UnsupportedTransformError):
            TransformCompiler().compile(
                _vocab_lookup_plan(),
                _columns(),
                _source(),
                _context(),
                _decisions(),
                artifact_kind=TargetArtifactKind.GRAPH_NODE,
            )

    def test_plan_without_vocab_lookup_raises(self) -> None:
        from sema.compile.compiler import TransformCompiler, UnsupportedTransformError
        from sema.models.planner._enums import MaterializationMode, PrimaryKeyStrategy
        from sema.models.planner.field_map import FieldMap, RowIdentity
        from sema.models.planner.mapping_plan import MappingPlan
        from sema.models.planner.patterns import DirectCopyPayload, MappingPattern
        from sema.models.planner.target_model import TargetObligation

        plan = MappingPlan(
            id="plan::stage::direct",
            source_scope_ref="src_study",
            obligation=TargetObligation(
                target_entity="target.stage",
                required_fields=["target.stage.name"],
                primary_key=PrimaryKeyStrategy.NATURAL_KEY,
            ),
            row_identity=RowIdentity(
                target_row_key_rule="src.sample.id",
                source_lineage=["src.sample.id"],
                materialization_mode=MaterializationMode.REPLACE_PARTITION,
            ),
            field_maps=[
                FieldMap(
                    target_field_ref="target.stage.name",
                    pattern=MappingPattern.DIRECT_COPY,
                    payload=DirectCopyPayload(source_field_ref="src.sample.name"),
                )
            ],
        )
        with pytest.raises(UnsupportedTransformError):
            TransformCompiler().compile(
                plan, _columns(), _source(), _context(), _decisions()
            )


class TestStagingDecisionProjection:
    def _value_mapping(self, *, no_map: bool) -> object:
        from sema.models.planner.lifecycle import Status
        from sema.resolve.value_mapping_store_utils import (
            ResolutionStatus,
            ValueMapping,
        )

        return ValueMapping(
            source_vocabulary="OncoTree",
            normalized_source_value="LUAD" if not no_map else "ZZZZ",
            target_property_ref="target.stage.concept",
            target_field="condition_concept_id",
            vocab_binding="binding.condition",
            concept_id=None if no_map else 4314337,
            vocab_release="omop-2026-q1",
            valid_start=None,
            valid_end=None,
            resolution_status=(
                ResolutionStatus.NO_MAP if no_map else ResolutionStatus.RESOLVED
            ),
            no_map_reason="dead end" if no_map else None,
            confidence=1.0,
            status=Status.auto_accepted,
            resolver_policy_ref="omop.oncotree_to_snomed_condition",
            run_id="run-1",
        )

    def test_resolved_projection(self) -> None:
        from sema.resolve.engine_utils import staging_decision_from_value_mapping

        decision = staging_decision_from_value_mapping(self._value_mapping(no_map=False))
        assert decision.normalized_source_value == "LUAD"
        assert decision.target_value == 4314337
        assert decision.resolution_status == "RESOLVED"
        assert decision.no_map_reason is None
        assert decision.status == "auto_accepted"

    def test_no_map_projection_has_null_target(self) -> None:
        from sema.resolve.engine_utils import staging_decision_from_value_mapping

        decision = staging_decision_from_value_mapping(self._value_mapping(no_map=True))
        assert decision.target_value is None
        assert decision.resolution_status == "NO_MAP"
        assert decision.no_map_reason == "dead end"


# --- execution against DuckDB ----------------------------------------------


def _seed_source(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute('CREATE SCHEMA IF NOT EXISTS "src_study"')
    conn.execute(
        'CREATE TABLE "src_study"."sample" '
        "(ONCOTREE_CODE VARCHAR, PATIENT_ID VARCHAR)"
    )
    conn.executemany(
        'INSERT INTO "src_study"."sample" VALUES (?, ?)',
        [("LUAD", "P1"), ("LUAD", "P2"), ("ZZZZ", "P3")],
    )


class TestExecute:
    def _run(self, conn: duckdb.DuckDBPyConnection) -> object:
        from sema.compile.compiler import TransformCompiler

        compiler = TransformCompiler()
        compiled = compiler.compile(
            _vocab_lookup_plan(), _columns(), _source(), _context(), _decisions()
        )
        return compiler.execute(
            conn,
            compiled,
            columns=_columns(),
            source=_source(),
            staging_schema="sema_staging",
            staging_table="condition_staging",
        )

    def test_row_count_matches_source(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "x.duckdb"))
        _seed_source(conn)
        written = self._run(conn)
        assert written == 3
        total = conn.execute(
            'SELECT COUNT(*) FROM "sema_staging"."condition_staging"'
        ).fetchone()[0]
        assert total == 3

    def test_no_map_rows_have_null_target(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "x.duckdb"))
        _seed_source(conn)
        self._run(conn)
        null_codes = conn.execute(
            'SELECT source_oncotree_code FROM "sema_staging"."condition_staging" '
            "WHERE condition_concept_id IS NULL"
        ).fetchall()
        assert {r[0] for r in null_codes} == {"ZZZZ"}
        resolved = conn.execute(
            'SELECT condition_concept_id FROM "sema_staging"."condition_staging" '
            "WHERE source_oncotree_code = 'LUAD'"
        ).fetchall()
        assert {r[0] for r in resolved} == {4314337}

    def test_rerun_is_idempotent(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "x.duckdb"))
        _seed_source(conn)
        self._run(conn)
        first = conn.execute(
            'SELECT * FROM "sema_staging"."condition_staging" ORDER BY 1, 4, 5'
        ).fetchall()
        self._run(conn)
        second = conn.execute(
            'SELECT * FROM "sema_staging"."condition_staging" ORDER BY 1, 4, 5'
        ).fetchall()
        assert first == second
        assert len(second) == 3

    def test_swap_is_scoped_to_source_table(self, tmp_path: Path) -> None:
        from sema.compile.compiler import TransformCompiler
        from sema.compile.compiler_utils import SourceTableSpec

        conn = duckdb.connect(str(tmp_path / "x.duckdb"))
        _seed_source(conn)
        # sibling study in the same schema must survive a swap of src_study.sample
        conn.execute(
            'CREATE TABLE "src_study"."sample2" '
            "(ONCOTREE_CODE VARCHAR, PATIENT_ID VARCHAR)"
        )
        conn.execute(
            'INSERT INTO "src_study"."sample2" VALUES (?, ?)', ["LUAD", "Q1"]
        )
        self._run(conn)
        sibling_source = SourceTableSpec(
            schema="src_study", table="sample2", value_column="ONCOTREE_CODE"
        )
        compiler = TransformCompiler()
        compiled = compiler.compile(
            _vocab_lookup_plan(), _columns(), sibling_source, _context(), _decisions()
        )
        compiler.execute(
            conn,
            compiled,
            columns=_columns(),
            source=sibling_source,
            staging_schema="sema_staging",
            staging_table="condition_staging",
        )
        rows = conn.execute(
            'SELECT source_table, COUNT(*) FROM "sema_staging"."condition_staging" '
            "GROUP BY source_table ORDER BY source_table"
        ).fetchall()
        assert rows == [("sample", 3), ("sample2", 1)]


class TestStagingSchema:
    def test_create_table_uses_frozen_b_columns(self) -> None:
        from sema.compile.compiler_utils import staging_column_order

        order = staging_column_order(_columns())
        assert order == (
            "source_schema",
            "source_table",
            "source_row_ref",
            "source_patient_key",
            "source_oncotree_code",
            "condition_concept_id",
            "resolver_policy_ref",
            "vocab_release",
            "resolution_status",
            "no_map_reason",
            "status",
            "run_id",
        )
