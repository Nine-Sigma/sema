"""Enrich: model descriptions, types, synonyms, and decoded values for bound relations.

Runs SEMA's semantic engine stages B and C with every domain layer off, through the intake
call path, after Roles and before Dry run. Stage A is never called: the bound entity and
its key role supply it. Nothing here changes a structural decision; the entries go to the
``enrichment`` ledger artifact, and Import fills only the fields the source left empty.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from sema.binding.spec import ConceptSourceSpec
from sema.engine.semantic import SemanticEngine
from sema.engine.stage_utils import PromptLayers
from sema.intake.roles.eligibility_spec_utils import RolesResult
from sema.intake.report.enrichment_utils import (
    ADDED,
    DECODED_VALUES,
    DESCRIPTION,
    SYNONYMS,
    EngineClient,
    Enrichment,
    EnrichmentEntry,
    Subject,
    Versions,
    enrichment_subjects,
    restore_names,
    subject_entries,
)
from sema.intake.calls.ledger import RunLedger
from sema.intake.calls.model_calls import ModelCalls
from sema.intake.land.profile import ProfileTools
from sema.log import logger
from sema.models.stages import StageAResult, StageBBatchResult, StageCBatchResult, StageCResult

__all__ = [
    "ARTIFACT",
    "Enrichment",
    "EnrichmentEntry",
    "StageBBatchResult_v1",
    "StageCBatchResult_v1",
    "StageCResult_v1",
    "apply_enrichment",
    "enrich",
    "stage_a_for",
]

ARTIFACT = "enrichment"
STAGE_B_VERSION = "# prompt: enrich_stage_b v1"
STAGE_C_VERSION = "# prompt: enrich_stage_c v1"
SUPPLIED_CONFIDENCE = 1.0
ENGINE_LAYERS = PromptLayers(
    enable_domain_bias=False,
    enable_vocab_hints=False,
    enable_few_shot=False,
    enable_inline_examples=False,
)
MANIFEST_FIELDS = {DESCRIPTION: "definition", SYNONYMS: SYNONYMS, DECODED_VALUES: DECODED_VALUES}


class StageBBatchResult_v1(StageBBatchResult):  # noqa: N801 - the version is the call key
    pass


class StageCBatchResult_v1(StageCBatchResult):  # noqa: N801
    pass


class StageCResult_v1(StageCResult):  # noqa: N801
    pass


VERSIONS: Versions = {
    StageBBatchResult: (StageBBatchResult_v1, STAGE_B_VERSION),
    StageCBatchResult: (StageCBatchResult_v1, STAGE_C_VERSION),
    StageCResult: (StageCResult_v1, STAGE_C_VERSION),
}


def enrich(
    spec: ConceptSourceSpec,
    roles: RolesResult,
    tools: ProfileTools,
    calls: ModelCalls,
    ledger: RunLedger,
    *,
    enabled: bool = False,
) -> Enrichment | None:
    """The enrichment artifact for every bound subject; None, with no call, when off."""
    if not enabled:
        return None
    client = EngineClient(calls, VERSIONS)
    engine = SemanticEngine(
        llm_client=client, domain_context=None, prompt_layers=ENGINE_LAYERS
    )
    entries: list[EnrichmentEntry] = []
    for subject in enrichment_subjects(spec, roles, tools):
        entries += _enrich_subject(engine, client, subject)
    enrichment = Enrichment(entries=tuple(entries))
    ledger.artifact(ARTIFACT, enrichment)
    logger.info(f"Enrichment of {spec.id}: {len(entries)} model entries")
    return enrichment


def stage_a_for(entity: str, key: tuple[str, ...]) -> StageAResult:
    grain = ", ".join(key) if key else entity
    return StageAResult(
        primary_entity=entity,
        grain_hypothesis=f"one row per {grain}",
        confidence=SUPPLIED_CONFIDENCE,
    )


def apply_enrichment(
    manifest: Mapping[str, Any], enrichment: Enrichment | None
) -> dict[str, Any]:
    """A new manifest whose empty property fields carry the added model text.

    An alternative never applies, nor does a semantic type, which the manifest has no
    field for; a stated value always wins.
    """
    applied = copy.deepcopy(dict(manifest))
    if enrichment is None:
        return applied
    properties = {
        (entity["qualified_name"], prop["name"]): prop
        for entity in applied.get("entities", [])
        for prop in entity.get("properties", [])
    }
    for entry in enrichment.entries:
        target = MANIFEST_FIELDS.get(entry.field)
        prop = properties.get((entry.entity, entry.property))
        if entry.kind != ADDED or target is None or prop is None or prop.get(target):
            continue
        prop[target] = copy.deepcopy(entry.value)
    return applied


def _enrich_subject(
    engine: SemanticEngine, client: EngineClient, subject: Subject
) -> list[EnrichmentEntry]:
    stage_a = stage_a_for(subject.name, subject.key)
    metadata = subject.table_metadata()
    stage_b = engine.run_stage_b(metadata, stage_a)
    restore_names(stage_b, subject, client)
    stage_c = engine.run_stage_c(metadata, stage_a, stage_b)
    return subject_entries(subject, stage_b, stage_c, client)
