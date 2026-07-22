"""US-012A: showcase wiring for the local fit chain.

Builds a :class:`~showcase.cbioportal_to_omop.slice0_fit.FitRequest` from the already-authored
OMOP target manifest (US-007) plus a source study. The TARGET binding is read
from the manifest as data (via :class:`ManifestTargetAdapter`) — the fit chain
re-materialises nothing. The OMOP showcase column names live in the allowlisted
policy layer (``resolve/policies/omop.py``), imported here, not invented.

Lives under ``pipeline/`` (not an R29-scanned core path); even so it stays
data-driven — the source value column and target domain arrive from the caller
and the binding, never as engine-owned literals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

from sema.compile.compiler_utils import CompileContext, SourceTableSpec
from sema.eval.mapping_goldset import GoldSet
from sema.models.planner._enums import (
    MaterializationMode,
    TargetArtifactKind,
)
from sema.models.planner.field_map import RowIdentity
from sema.models.planner.mapping_plan import MappingAssertion
from sema.models.planner.patterns import ConstantValue, MappingPattern
from sema.models.planner.lifecycle import Status
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.models.target.refs import TargetEntityRef, TargetPropertyRef
from sema.models.target.vocab_binding import VocabularyBindingDecl
from showcase.cbioportal_to_omop.slice0_fit import FitRequest
from sema.resolve.engine_utils import ResolveContext
from showcase.cbioportal_to_omop.omop_policy import (
    OMOP_STAGING_COLUMNS,
    SLICE0_CONDITION_CONCEPT_FIELD,
    SLICE0_RESOLVER_POLICY_FIELD,
    SLICE0_VOCAB_RELEASE_FIELD,
    make_slice0_staging_obligation,
)
from sema.resolve.policy import ResolverPolicy
from sema.resolve.policies import resolve_policy
from sema.resolve.producer import MappingNodes
from sema.targets.adapters.manifest import ManifestTargetAdapter

__all__ = [
    "build_slice0_fit_request",
    "discover_study",
    "enumerate_source",
    "load_binding",
]

DEFAULT_ENTITY = "omop.condition_occurrence"
DEFAULT_PROPERTY = "condition_concept_id"
DEFAULT_VOCAB_RELEASE = "omop-vocab-2024"
DEFAULT_STAGING_SCHEMA = "sema_staging"
DEFAULT_STAGING_TABLE = "condition_staging"
DEFAULT_RUN_ID = "sema-fit"


def load_binding(
    manifest_path: str | Path,
    *,
    entity_qname: str = DEFAULT_ENTITY,
    property_name: str = DEFAULT_PROPERTY,
) -> VocabularyBindingDecl:
    """Read the property's vocabulary binding from the authored manifest."""
    adapter = ManifestTargetAdapter(Path(manifest_path))
    model_id = adapter.describe().target_model_id
    ref = TargetPropertyRef(
        entity_ref=TargetEntityRef(
            target_model_id=model_id,
            qualified_name=entity_qname,
            kind=TargetArtifactKind.TABLE_ROW,
        ),
        property_name=property_name,
    )
    bindings = list(adapter.load_vocabulary_bindings(ref))
    if not bindings:
        raise ValueError(
            f"manifest {Path(manifest_path).name} declares no vocabulary binding "
            f"for {entity_qname}.{property_name}"
        )
    return bindings[0]


def discover_study(
    conn: duckdb.DuckDBPyConnection,
    *,
    value_column: str,
    source_table: str = "sample",
) -> tuple[str, str] | None:
    """Find the first schema whose ``source_table`` carries ``value_column``."""
    rows = conn.execute(
        "SELECT table_schema, table_name FROM information_schema.columns "
        "WHERE column_name = ? AND table_name = ? ORDER BY table_schema LIMIT 1",
        [value_column, source_table],
    ).fetchall()
    return (rows[0][0], rows[0][1]) if rows else None


def enumerate_source(
    conn: duckdb.DuckDBPyConnection,
    *,
    schema: str,
    table: str,
    value_column: str,
) -> tuple[list[str], int]:
    """Return the distinct non-null source codes and the total source row count."""
    codes = [
        r[0]
        for r in conn.execute(
            f'SELECT DISTINCT "{value_column}" FROM "{schema}"."{table}" '
            f'WHERE "{value_column}" IS NOT NULL ORDER BY "{value_column}"'
        ).fetchall()
    ]
    row = conn.execute(
        f'SELECT COUNT(*) FROM "{schema}"."{table}" '
        f'WHERE "{value_column}" IS NOT NULL'
    ).fetchone()
    return codes, int(row[0]) if row else 0


def build_slice0_fit_request(
    *,
    manifest_path: str | Path,
    source_schema: str,
    source_table: str,
    value_column: str,
    source_codes: list[str],
    source_row_count: int,
    gold: GoldSet,
    vocab_release: str = DEFAULT_VOCAB_RELEASE,
    run_id: str = DEFAULT_RUN_ID,
    staging_schema: str = DEFAULT_STAGING_SCHEMA,
    staging_table: str = DEFAULT_STAGING_TABLE,
) -> tuple[ResolverPolicy, FitRequest]:
    """Assemble the policy + FitRequest for the OncoTree->OMOP Condition showcase."""
    binding = load_binding(manifest_path)
    policy = resolve_policy(binding)
    context = _resolve_context(
        binding, source_schema, source_table, value_column, vocab_release, run_id
    )
    request = FitRequest(
        source=SourceTableSpec(
            schema=source_schema, table=source_table, value_column=value_column
        ),
        source_codes=source_codes,
        source_row_count=source_row_count,
        policy=policy,
        resolve_context=context,
        compile_context=CompileContext(
            resolver_policy_ref=context.resolver_policy_ref,
            vocab_release=vocab_release,
            run_id=run_id,
        ),
        staging_columns=OMOP_STAGING_COLUMNS,
        obligation=make_slice0_staging_obligation(),
        constant_assertions=_constant_assertions(context),
        row_identity=RowIdentity(
            target_row_key_rule=context.source_value_ref,
            source_lineage=[context.source_value_ref],
            materialization_mode=MaterializationMode.REPLACE_PARTITION,
        ),
        staging_schema=staging_schema,
        staging_table=staging_table,
        gold=gold,
        nodes=MappingNodes(
            source_property_id=context.source_field_ref,
            target_property_id=context.target_property_ref,
        ),
    )
    return policy, request


def _constant_assertions(context: ResolveContext) -> list[MappingAssertion]:
    """Build the two run-constant staging field maps (§1.5(e) coverage).

    ``resolver_policy_ref`` and ``vocab_release`` are stamped onto every staging
    row as compile-time constants; representing them as CONSTANT assertions makes
    the ``MappingPlan`` a faithful description of the staging output and lets the
    assembler's coverage gate check all three required fields end-to-end.
    """
    specs = (
        (SLICE0_RESOLVER_POLICY_FIELD, context.resolver_policy_ref, "resolver-policy"),
        (SLICE0_VOCAB_RELEASE_FIELD, context.vocab_release, "vocab-release"),
    )
    return [
        MappingAssertion(
            id=f"const::{slug}::{context.run_id}",
            source_field_ref=context.source_value_ref,
            target_property_ref=target_ref,
            pattern=MappingPattern.CONSTANT,
            payload=ConstantValue(literal_value=value, target_type="VARCHAR"),
            confidence=1.0,
            provenance=context.provenance,
            status=Status.auto_accepted,
        )
        for target_ref, value, slug in specs
    ]


def _resolve_context(
    binding: VocabularyBindingDecl,
    source_schema: str,
    source_table: str,
    value_column: str,
    vocab_release: str,
    run_id: str,
) -> ResolveContext:
    source_ref = f"source.{source_table}.{value_column}"
    target_ref = SLICE0_CONDITION_CONCEPT_FIELD
    policy_ref = binding.resolver_policy_ref or ""
    return ResolveContext(
        source_field_ref=source_ref,
        source_value_ref=source_ref,
        target_property_ref=target_ref,
        target_field=binding.property_name,
        domain_constraint_ref=f"target.stage.domain={binding.domain}",
        vocabulary_ref=f"vocab.{binding.vocabulary.name.lower()}",
        vocab_binding=f"binding.{binding.property_name}",
        vocab_release=vocab_release,
        resolver_policy_ref=policy_ref,
        run_id=run_id,
        provenance=_provenance(source_schema, vocab_release, run_id),
    )


def _provenance(source_id: str, vocab_release: str, run_id: str) -> Provenance:
    return Provenance(
        run=RunProvenance(
            run_id=run_id,
            target_model_version="omop-cdm-5.4",
            target_schema_snapshot_hash="t",
            vocab_release=vocab_release,
            context_card_version="cc",
            prompt_template_version="pt",
            few_shot_set_version="fs",
            constraint_version="cv",
            llm_model="none",
        ),
        source=SourceScope(
            source_id=source_id, source_schema_hash="s", source_profile_hash="p"
        ),
        timestamp=datetime.now(timezone.utc),
    )
