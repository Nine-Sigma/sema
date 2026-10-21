"""Dry-run verifier: the import's manifest checked for completeness, duplicates, and lifecycle.

Active terms are the manifest's terms and retired ones the sidecar's retirements, so the
checks read the import output and the landed row counts only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from sema.binding.importer import ConceptImport
from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import RelationRef
from sema.intake.profile import ProfileTools
from sema.intake.shapes.base_utils import failure
from sema.intake.structure import StructureResult
from sema.intake.verify.base import Failure, Verdict
from sema.models.target.refs import VocabularySource
from sema.targets.adapters.manifest_exceptions import ManifestError
from sema.targets.adapters.manifest_models import ParsedManifest
from sema.targets.adapters.manifest_parser import parse_manifest_raw

IMPORT_COMPLETED = "import_completed"
COUNTS_IN_RANGE = "counts_in_range"
UNIQUE_ACTIVE_CODE = "unique_active_code"
MANIFEST_PARSES = "manifest_parses"
INLINE_UNDER_CAP = "inline_under_cap"
REPLACED_BY_ACTIVE = "replaced_by_active"
CHECKS = (
    IMPORT_COMPLETED,
    COUNTS_IN_RANGE,
    UNIQUE_ACTIVE_CODE,
    MANIFEST_PARSES,
    INLINE_UNDER_CAP,
    REPLACED_BY_ACTIVE,
)
TERM_KIND = "term"

Pair = tuple[str, str]


def verify_dry_run(
    spec: ConceptSourceSpec,
    imported: ConceptImport,
    *,
    structure: StructureResult,
    tools: ProfileTools,
) -> Verdict:
    parsed, failures = _parsed(imported.manifest)
    if parsed is None:
        return Verdict.of(failures=failures, ambiguities=[], floors={})
    active = [(term.vocabulary, term.code) for term in parsed.terms]
    failures += _duplicates(active, spec)
    failures += _over_cap(parsed, spec)
    failures += _retirements(imported, spec, set(active), _active_concepts(parsed))
    failures += _counts(spec, structure, tools, len(active) + _retired_terms(imported))
    return Verdict.of(failures=failures, ambiguities=[], floors={})


def term_relations(spec: ConceptSourceSpec) -> list[RelationRef]:
    found = [spec.value_sets.relation] if spec.value_sets is not None else []
    return found + [entry.relation for entry in spec.code_lists or ()]


def _parsed(manifest: dict[str, Any]) -> tuple[ParsedManifest | None, list[Failure]]:
    if not manifest:
        return None, [failure(IMPORT_COMPLETED, "", manifest="empty")]
    try:
        return parse_manifest_raw(manifest), []
    except ManifestError as error:
        return None, [failure(MANIFEST_PARSES, "", error=str(error))]


def _vocabulary_relation(spec: ConceptSourceSpec, vocabulary: str) -> str:
    listed = next((e for e in spec.code_lists or () if e.vocabulary_id == vocabulary), None)
    if listed is not None:
        return listed.relation.name
    return spec.value_sets.relation.name if spec.value_sets is not None else ""


def _duplicates(active: Sequence[Pair], spec: ConceptSourceSpec) -> list[Failure]:
    return [
        failure(
            UNIQUE_ACTIVE_CODE, _vocabulary_relation(spec, vocabulary),
            vocabulary=vocabulary, code=code, active=count,
        )
        for (vocabulary, code), count in Counter(active).items()
        if count > 1
    ]


def _over_cap(parsed: ParsedManifest, spec: ConceptSourceSpec) -> list[Failure]:
    cap = spec.term_storage.inline_cap
    inline = {v.name for v in parsed.vocabularies if v.source == VocabularySource.INLINE}
    counts = Counter(term.vocabulary for term in parsed.terms if term.vocabulary in inline)
    return [
        failure(
            INLINE_UNDER_CAP, _vocabulary_relation(spec, name), vocabulary=name, terms=n, cap=cap
        )
        for name, n in counts.items()
        if n > cap
    ]


def _active_concepts(parsed: ParsedManifest) -> set[str]:
    return {
        prop.concept_ref.id
        for entity in parsed.entities
        for prop in entity.properties
        if prop.concept_ref is not None
    }


def _retired_terms(imported: ConceptImport) -> int:
    return sum(r.kind == TERM_KIND for r in imported.authored.retirements)


def _retirements(
    imported: ConceptImport, spec: ConceptSourceSpec, active: set[Pair], concepts: set[str]
) -> list[Failure]:
    concept_relation = spec.concepts.relation.name if spec.concepts is not None else ""
    failures = []
    for retired in imported.authored.retirements:
        if retired.replaced_by is None:
            continue
        vocabulary = retired.vocabulary or ""
        if retired.kind == TERM_KIND:
            found = (vocabulary, retired.replaced_by) in active
            relation = _vocabulary_relation(spec, vocabulary)
        else:
            found, relation = retired.replaced_by in concepts, concept_relation
        if not found:
            failures.append(failure(
                REPLACED_BY_ACTIVE, relation, kind=retired.kind, vocabulary=vocabulary,
                code=retired.code, replaced_by=retired.replaced_by,
            ))
    return failures


def _counts(
    spec: ConceptSourceSpec, structure: StructureResult, tools: ProfileTools, terms: int
) -> list[Failure]:
    """Terms, active and retired, number at most the rows of the bound term relations."""
    bound = {name for entry in structure.bound for name in entry.binding.relations}
    failures = []
    rows = 0
    for ref in term_relations(spec):
        if ref.name not in bound:
            failures.append(failure(COUNTS_IN_RANGE, ref.name, bound=False))
            continue
        rows += tools.count_where(ref, [])[0]
    if terms > rows:
        failures.append(failure(COUNTS_IN_RANGE, "", terms=terms, rows=rows))
    return failures
