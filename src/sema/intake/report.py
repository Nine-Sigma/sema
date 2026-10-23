"""Dry run: what an import would write, counted and rendered (design Decision 13).

Term staging goes to a namespace unique to the run, dropped in ``finally``, so the spec's
real namespace and the graph are never written. The report carries no path or timestamp,
so two runs of one spec serialize the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sema.binding.importer import ConceptImport, import_concept_source, write_import
from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import RelationRef
from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import quote_ident
from sema.intake.ledger import RunLedger
from sema.intake.profile import EvidencePolicy, ProfileTools
from sema.intake.relations import RelationProvider
from sema.intake.report_utils import (
    EDGE_COLUMNS,
    TermEntry,
    concept_count,
    duplicate_codes,
    entity_counts,
    external_ref_codes,
    hierarchy_edges,
    import_terms,
    lifecycle_counts,
    longest_labels,
    near_duplicates,
    null_required_rows,
    seeded_sample,
    staged_terms,
    vocabulary_of_concepts,
    without_obligation,
    write_edges,
)
from sema.log import logger
from sema.targets.adapters.manifest_models import ParsedManifest
from sema.targets.adapters.manifest_parser import parse_manifest_raw

REPORT_SEED = 13
SEEDED_TERMS = 10
LONGEST_TERMS = 5
STAGE = "dry_run"
SCRATCH_MARK = "__dryrun_"
RUN_ID_CHARS = 8
REPORT_ARTIFACT = "dry_run_report"
HIERARCHY_TABLE = "hierarchy_edges"
Unverifiable = Literal["unverifiable"]
ResolveRate = float | Unverifiable
UNVERIFIABLE: Unverifiable = "unverifiable"
MANIFEST_FILE = "manifest.yaml"
SIDECAR_FILE = "authored_evidence.json"


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RenderedTerm(_Report):
    code: str | None
    display: str | None


class VocabularyReport(_Report):
    term_count: int
    seeded: list[RenderedTerm]
    longest: list[RenderedTerm]
    near_duplicates: list[RenderedTerm]
    null_required: list[RenderedTerm]
    duplicate_codes: list[str]


class LifecycleCounts(_Report):
    active: int
    retired: int
    replaced: int


class EntityReport(_Report):
    properties: int
    foreign_keys: int
    domains: int
    inline_value_sets: int


class DryRunReport(_Report):
    concept_count: int
    vocabulary_count: int
    vocabularies: dict[str, VocabularyReport]
    root_count: int
    max_depth: int
    orphan_parents: int
    lifecycle: LifecycleCounts
    external_refs: dict[str, ResolveRate]
    entities: dict[str, EntityReport]
    without_obligation: dict[str, str]


@dataclass(frozen=True)
class DryRun:
    """The report with the scratch import and staged terms it counted, for the verifier."""

    report: DryRunReport
    imported: ConceptImport
    staged: tuple[TermEntry, ...]


def scratch_namespace(spec: ConceptSourceSpec, run_id: str) -> str:
    return f"{spec.external_namespace}{SCRATCH_MARK}{run_id[:RUN_ID_CHARS]}"


def dry_run(
    spec: ConceptSourceSpec, provider: RelationProvider, staging: Staging, ledger: RunLedger
) -> DryRun:
    scratch = scratch_namespace(spec, ledger.directory.name)
    storage = spec.term_storage.model_copy(update={"external_namespace": scratch})
    try:
        scratch_spec = spec.model_copy(update={"term_storage": storage})
        ran = _report(scratch_spec, provider, staging, ledger)
    finally:
        staging.execute(f"DROP SCHEMA IF EXISTS {quote_ident(scratch)} CASCADE")
    report = ran.report
    ledger.artifact(REPORT_ARTIFACT, report)
    logger.info(
        "Dry run of {}: {} vocabularies, {} terms, {} entities; scratch namespace {} dropped",
        spec.id,
        report.vocabulary_count,
        report.lifecycle.active + report.lifecycle.retired,
        len(report.entities),
        scratch,
    )
    return ran


def _report(
    spec: ConceptSourceSpec, provider: RelationProvider, staging: Staging, ledger: RunLedger
) -> DryRun:
    imported = import_concept_source(spec, provider, dataset_map={}, staging=staging)
    _write_scratch_files(imported)
    parsed = parse_manifest_raw(imported.manifest)
    staged = tuple(staged_terms(staging, spec.external_namespace))
    terms = import_terms(parsed, imported) + list(staged)
    tools = ProfileTools(
        RelationProvider(warehouse=None, staging=staging),
        EvidencePolicy(tier="metadata_only"),
        ledger,
        stage=STAGE,
    )
    roots, depth, orphans = _hierarchy(parsed, staging, spec.external_namespace, tools)
    nulls = null_required_rows(spec, provider, ledger, STAGE, vocabulary_of_concepts(parsed))
    active, retired, replaced = lifecycle_counts(terms)
    schemas = imported.table_schemas
    report = DryRunReport(
        concept_count=concept_count(parsed),
        vocabulary_count=len(parsed.vocabularies),
        vocabularies={
            v.name: _vocabulary(
                [t for t in terms if t.vocabulary == v.name], nulls.get(v.name, [])
            )
            for v in parsed.vocabularies
        },
        root_count=roots,
        max_depth=depth,
        orphan_parents=orphans,
        lifecycle=LifecycleCounts(active=active, retired=retired, replaced=replaced),
        external_refs=_external_refs(parsed, tools),
        entities={name: _entity(*entity_counts(schema)) for name, schema in schemas.items()},
        without_obligation=without_obligation(parsed, schemas),
    )
    return DryRun(report, imported, staged)


def _write_scratch_files(imported: ConceptImport) -> None:
    with TemporaryDirectory() as directory:
        scratch = Path(directory)
        write_import(imported, scratch / MANIFEST_FILE, scratch / SIDECAR_FILE)


def _hierarchy(
    parsed: ParsedManifest, staging: Staging, namespace: str, tools: ProfileTools
) -> tuple[int, int, int]:
    """Roots and depth over the emitted hierarchical relationships, then orphan parents."""
    edges = hierarchy_edges(parsed)
    if not edges.edges:
        return 0, 0, edges.orphan_parents
    write_edges(staging, namespace, HIERARCHY_TABLE, edges)
    ref = RelationRef(provider="staging", name=f"{namespace}.{HIERARCHY_TABLE}")
    check = tools.check_hierarchy(ref, *EDGE_COLUMNS)
    return check.root_count, check.max_depth, edges.orphan_parents


def _external_refs(
    parsed: ParsedManifest, tools: ProfileTools
) -> dict[str, ResolveRate]:
    rates: dict[str, ResolveRate] = {}
    for system, codes in external_ref_codes(parsed).items():
        check = tools.resolve_external_refs(system, codes)
        rate = check.resolve_rate
        rates[system] = UNVERIFIABLE if check.unverifiable or rate is None else rate
    return rates


def _vocabulary(
    terms: Sequence[TermEntry], nulls: Sequence[tuple[str | None, str | None]]
) -> VocabularyReport:
    """Counts cover retired terms; rendering covers active ones, whose display every
    storage keeps, so an inline and a staged run of one spec render alike."""
    active = [term for term in terms if not term.retired]
    return VocabularyReport(
        term_count=len(terms),
        seeded=_rendered(seeded_sample(active, SEEDED_TERMS, REPORT_SEED)),
        longest=_rendered(longest_labels(active, LONGEST_TERMS)),
        near_duplicates=_rendered(near_duplicates(active)),
        null_required=[RenderedTerm(code=code, display=display) for code, display in nulls],
        duplicate_codes=duplicate_codes(terms),
    )


def _rendered(terms: Sequence[TermEntry]) -> list[RenderedTerm]:
    return [RenderedTerm(code=term.code, display=term.display) for term in terms]


def _entity(properties: int, foreign_keys: int, domains: int, values: int) -> EntityReport:
    return EntityReport(
        properties=properties, foreign_keys=foreign_keys, domains=domains, inline_value_sets=values
    )
