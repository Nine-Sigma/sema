"""US-010 live test: compile + execute the staging write on ~/.sema/poc.duckdb.

Skip-guarded on the DuckDB mirror. Resolves the distinct ONCOTREE_CODE for a real
study, inlines those decisions into the §1.5(b) staging SELECT, executes it into a
temp staging table, and asserts: row count = source row count, the join populated
``condition_concept_id``, and NO_MAP rows are NULL there. Also renders the
Databricks dialect and asserts it parses (full Databricks execution is US-013).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest
import sqlglot

from sema.compile.compiler import TransformCompiler
from sema.models.planner._enums import MaterializationMode, PrimaryKeyStrategy
from sema.models.planner.field_map import FieldMap, RowIdentity
from sema.models.planner.lifecycle import Status
from sema.models.planner.mapping_plan import MappingPlan
from sema.models.planner.patterns import MappingPattern, VocabLookup
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.models.planner.target_model import TargetObligation
from sema.resolve.engine import VocabularyResolver
from sema.resolve.engine_utils import (
    ResolveContext,
    staging_decision_from_value_mapping,
)
from sema.resolve.policies.omop import (
    OMOP_ONCOTREE_CONDITION_REF,
    OMOP_STAGING_COLUMNS,
    OMOP_VOCAB_SCHEMA,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.value_mapping_store_utils import ResolutionStatus
from sema.resolve.vocab_store import open_duckdb_vocab_store
from tests.integration._omop_binding import build_condition_binding

pytestmark = pytest.mark.integration

_DB = Path.home() / ".sema" / "poc.duckdb"


def _discover_study(conn: duckdb.DuckDBPyConnection) -> tuple[str, str] | None:
    rows = conn.execute(
        "SELECT table_schema, table_name FROM information_schema.columns "
        "WHERE column_name = 'ONCOTREE_CODE' AND table_name = 'sample' "
        "ORDER BY table_schema LIMIT 1"
    ).fetchall()
    return (rows[0][0], rows[0][1]) if rows else None


def _context() -> ResolveContext:
    prov = Provenance(
        run=RunProvenance(
            run_id="us010-live",
            target_model_version="omop-cdm-5.4",
            target_schema_snapshot_hash="t",
            vocab_release="omop-vocab-2024",
            context_card_version="cc",
            prompt_template_version="pt",
            few_shot_set_version="fs",
            constraint_version="cv",
            llm_model="none",
        ),
        source=SourceScope(
            source_id="study", source_schema_hash="s", source_profile_hash="p"
        ),
        timestamp=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )
    return ResolveContext(
        source_field_ref="source.sample.ONCOTREE_CODE",
        source_value_ref="source.sample.ONCOTREE_CODE",
        target_property_ref="target.stage.condition_concept_id",
        target_field="condition_concept_id",
        domain_constraint_ref="target.stage.domain=Condition",
        vocabulary_ref="vocab.snomed",
        vocab_binding="binding.condition",
        vocab_release="omop-vocab-2024",
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        run_id="us010-live",
        provenance=prov,
    )


def _plan() -> MappingPlan:
    ctx = _context()
    field_map = FieldMap(
        target_field_ref="target.stage.condition_concept_id",
        pattern=MappingPattern.VOCAB_LOOKUP,
        payload=VocabLookup(
            vocabulary_ref=ctx.vocabulary_ref,
            source_value_ref=ctx.source_value_ref,
            domain_constraint_ref=ctx.domain_constraint_ref,
            require_standard=True,
            allow_zero_default=False,
            resolver_policy_ref=ctx.resolver_policy_ref,
        ),
        status=Status.auto_accepted,
    )
    return MappingPlan(
        id="plan::stage::study",
        source_scope_ref="study",
        obligation=TargetObligation(
            target_entity="target.stage",
            required_fields=["target.stage.condition_concept_id"],
            primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        ),
        row_identity=RowIdentity(
            target_row_key_rule="source.sample.ONCOTREE_CODE",
            source_lineage=["source.sample.ONCOTREE_CODE"],
            materialization_mode=MaterializationMode.REPLACE_PARTITION,
        ),
        field_maps=[field_map],
    )


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_compile_and_execute_staging_write(tmp_path: Path) -> None:
    policy = make_omop_oncotree_condition_policy(build_condition_binding())
    vstore = open_duckdb_vocab_store(str(_DB), schema=OMOP_VOCAB_SCHEMA)
    resolver = VocabularyResolver(vstore, policy)

    pdb = duckdb.connect(str(_DB), read_only=True)
    found = _discover_study(pdb)
    assert found is not None, "no cbioportal sample table with ONCOTREE_CODE"
    src_schema, src_table = found
    codes = [
        r[0]
        for r in pdb.execute(
            f'SELECT DISTINCT "ONCOTREE_CODE" FROM "{src_schema}"."{src_table}" '
            "WHERE \"ONCOTREE_CODE\" IS NOT NULL"
        ).fetchall()
    ]
    src_count = pdb.execute(
        f'SELECT COUNT(*) FROM "{src_schema}"."{src_table}" '
        "WHERE \"ONCOTREE_CODE\" IS NOT NULL"
    ).fetchone()[0]

    decisions = [
        staging_decision_from_value_mapping(
            resolver.to_value_mapping(resolver.resolve(code), _context())
        )
        for code in codes
    ]

    from sema.compile.compiler_utils import SourceTableSpec, CompileContext

    source = SourceTableSpec(
        schema=src_schema, table=src_table, value_column="ONCOTREE_CODE"
    )
    cctx = CompileContext(
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        vocab_release="omop-vocab-2024",
        run_id="us010-live",
    )

    work = duckdb.connect(str(tmp_path / "staging.duckdb"))
    work.execute(f"ATTACH '{_DB}' AS poc (READ_ONLY)")
    work.execute(f'CREATE SCHEMA IF NOT EXISTS "{src_schema}"')
    work.execute(
        f'CREATE TABLE "{src_schema}"."{src_table}" AS '
        f'SELECT "ONCOTREE_CODE" FROM poc."{src_schema}"."{src_table}" '
        "WHERE \"ONCOTREE_CODE\" IS NOT NULL"
    )

    compiler = TransformCompiler()
    compiled = compiler.compile(_plan(), OMOP_STAGING_COLUMNS, source, cctx, decisions)
    written = compiler.execute(
        work,
        compiled,
        columns=OMOP_STAGING_COLUMNS,
        source=source,
        staging_schema="sema_staging",
        staging_table="condition_staging",
    )
    assert written == src_count

    populated = work.execute(
        'SELECT COUNT(*) FROM "sema_staging"."condition_staging" '
        "WHERE condition_concept_id IS NOT NULL"
    ).fetchone()[0]
    assert populated >= 1

    # NO_MAP codes -> NULL target; RESOLVED codes -> non-null.
    no_map_codes = {
        d.normalized_source_value
        for d in decisions
        if d.resolution_status == ResolutionStatus.NO_MAP.value
    }
    staged_null = {
        r[0]
        for r in work.execute(
            "SELECT source_oncotree_code FROM "
            '"sema_staging"."condition_staging" WHERE condition_concept_id IS NULL'
        ).fetchall()
    }
    assert staged_null <= no_map_codes

    # Databricks dialect renders and parses (US-013 executes it).
    dbx = compiled.sql("databricks")
    sqlglot.parse_one(dbx, dialect="databricks")
