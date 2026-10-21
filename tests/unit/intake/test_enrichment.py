"""Enrichment: SEMA's L2 engine stages B and C over bound relations, with domain layers off.

Pinned API of ``sema.intake.enrichment`` (task 10.2 implements it):

- ``enrich(spec, roles, tools, calls, ledger, *, enabled: bool = False) -> Enrichment | None``.
  Off by default: with ``enabled`` False it returns None, makes no model call, and writes
  no artifact. The graph's Enrich node (task 11) passes the run's flag through.
- Each call goes through ``ModelCalls.call`` at stage ``"enrich"``; the adapter is the
  engine's client, so the ledger's ``call`` event records the path that served it.
- The engine is ``SemanticEngine(domain_context=None, prompt_layers=PromptLayers(
  enable_domain_bias=False, enable_vocab_hints=False, enable_few_shot=False))``; only
  ``run_stage_b`` and ``run_stage_c`` run. ``stage_a_for(entity, key) -> StageAResult``
  supplies stage A: ``primary_entity`` is the entity, ``grain_hypothesis`` is
  ``"one row per <key columns joined by ', '>"`` (``"one row per <entity>"`` with no key),
  confidence 1.0.
- Subjects: an entity-model binding enriches each imported entity (named by its manifest
  ``qualified_name``), its properties being the dictionary's property rows with the stated
  description as the column comment. Every other bound relation is its own subject (named
  by its qualified landed name), its properties being its landed columns, and its key the
  column chosen for the lexicon's key role.
- ``Enrichment.entries`` is a tuple of ``EnrichmentEntry(entity, property, field, value,
  source="model", confidence, call_key, kind)``, with ``field`` one of ``description``
  (the stage B label), ``semantic_type``, ``synonyms``, ``decoded_values`` (stage C), and
  ``kind`` ``"added"`` where the source states nothing for that field or ``"alternative"``
  where it does. The result is written as the ``enrichment`` ledger artifact.
- Versioned keys (the intake-model-calls rule): the adapter prepends ``# prompt:
  enrich_stage_b v1`` (``enrich_stage_c v1`` for stage C) as the first line of the
  engine's rendered prompt, and calls with intake subclasses of the engine models named
  ``StageBBatchResult_v1``, ``StageCBatchResult_v1``, and ``StageCResult_v1`` (defined in
  ``sema.intake.enrichment``; the engine's models are not edited). ``ModelCalls.call``
  keys on ``output_model.__name__``, so no src change to the call path is needed.
- ``apply_enrichment(manifest, enrichment) -> dict`` is what Import calls on the imported
  manifest: it returns a new manifest in which each ``added`` entry fills the property's
  ``definition``, ``synonyms``, or ``decoded_values`` only where that field is empty, and
  never applies an ``alternative``, a ``semantic_type``, or anything when ``enrichment``
  is None.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.binding.importer import import_concept_source
from sema.engine.vocabulary import VOCABULARY_PATTERNS
from sema.intake.eligibility import assemble_spec
from sema.intake.ledger_utils import prompt_hash
from sema.targets.adapters.manifest_parser import parse_manifest_raw
from tests.unit.intake.enrichment_env import (
    ARTIFACT,
    CONFIDENCE,
    ENRICH_STAGE,
    SEMANTIC_TYPE,
    SERVED_BY,
    STAGE_B_VERSION_LINE,
    EnrichRun,
    a2_run,
    acris_run,
    domain_blocks,
    domain_types,
    label_for,
    manifest_property,
    synonym_for,
)

pytestmark = pytest.mark.unit

A2_CONCEPTS = "land_a2.codes_a2__Concepts"
A2_KEY = "concept_code"
ACRIS_MASTER = "acris.acris_real_property_master"
BOROUGH = "Recorded Borough"
BOROUGH_DESCRIPTION = "Borough where document was recorded"
BOROUGHS = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens"}
INLINE_DOMAIN_EXAMPLES = ("diagnosis coding system", "gene symbol namespace")


@pytest.fixture(scope="module")
def a2_enriched(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[EnrichRun, Any]]:
    with a2_run(tmp_path_factory.mktemp("enrich_a2")) as run:
        yield run, run.enrich_watched()


@pytest.fixture(scope="module")
def acris_enriched(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[EnrichRun, Any]]:
    with acris_run(tmp_path_factory.mktemp("enrich_acris")) as run:
        yield run, run.enrich_watched()


def _imported_manifest(run: EnrichRun) -> dict[str, Any]:
    spec = run.stages.spec
    return dict(import_concept_source(spec, run.stages.landed.provider, dataset_map={}).manifest)


def _entry(enrichment: Any, entity: str, prop: str, field: str) -> Any:
    found = [
        e
        for e in enrichment.entries
        if (e.entity, e.property, e.field) == (entity, prop, field)
    ]
    assert len(found) == 1, f"{len(found)} entries for {entity}.{prop} {field}"
    return found[0]


# -- Off by default ---------------------------------------------------------------------


def test_enrich_is_off_unless_enabled() -> None:
    from sema.intake.enrichment import enrich

    assert inspect.signature(enrich).parameters["enabled"].default is False


def test_off_by_default_makes_no_call_and_writes_nothing(tmp_path: Path) -> None:
    from sema.intake.enrichment import apply_enrichment

    with a2_run(tmp_path) as run:
        result = run.enrich()
        manifest = _imported_manifest(run)

        assert result is None
        run.client.invoke.assert_not_called()
        assert run.call_records() == {}
        assert not run.artifact_path().exists()
        assert run.stages.landed.record().model_calls == 0
        assert apply_enrichment(manifest, None) == manifest


# -- Stage A is supplied, not called ----------------------------------------------------


def test_stage_a_result_names_the_bound_entity_and_key() -> None:
    from sema.intake.enrichment import stage_a_for

    keyed = stage_a_for(A2_CONCEPTS, (A2_KEY,))
    unkeyed = stage_a_for(ACRIS_MASTER, ())

    assert (keyed.primary_entity, keyed.grain_hypothesis) == (
        A2_CONCEPTS,
        f"one row per {A2_KEY}",
    )
    assert keyed.confidence == 1.0
    assert unkeyed.grain_hypothesis == f"one row per {ACRIS_MASTER}"


def test_no_stage_a_call_and_the_prompt_names_the_bound_entity(
    a2_enriched: tuple[EnrichRun, Any],
) -> None:
    run, _ = a2_enriched
    prompts = run.prompts()

    assert prompts, "enrichment made no call"
    assert "StageAResult" not in run.asked_models()
    assert all(r["schema_name"] != "StageAResult" for r in run.call_records().values())
    named = [p for p in prompts if f"Entity: {A2_CONCEPTS}" in p]
    assert len(named) == 1
    assert f"Grain: one row per {A2_KEY}" in named[0]


def test_the_engine_runs_with_every_domain_layer_off(
    a2_enriched: tuple[EnrichRun, Any],
) -> None:
    run, _ = a2_enriched

    assert run.prompt_kwargs, "stage B rendered no prompt through the engine"
    for kwargs in run.prompt_kwargs:
        layers = kwargs["layers"]
        assert kwargs["domain_context"] is None
        assert not (layers.enable_domain_bias or layers.enable_vocab_hints)
        assert not layers.enable_few_shot


# -- The rendered prompt holds no domain layer ------------------------------------------


def test_prompt_holds_no_domain_layer_or_vocabulary_family_list(
    a2_enriched: tuple[EnrichRun, Any],
) -> None:
    run, _ = a2_enriched
    blocks, types = domain_blocks(), domain_types()

    for prompt in run.prompts():
        assert [b for b in blocks if b in prompt] == []
        assert sorted(t for t in types if t in prompt) == []


def test_prompt_holds_no_code_pattern(a2_enriched: tuple[EnrichRun, Any]) -> None:
    run, _ = a2_enriched

    for prompt in run.prompts():
        for vocab in VOCABULARY_PATTERNS:
            assert vocab.pattern.pattern not in prompt
            assert not re.search(rf"(?<!\w){re.escape(vocab.name)}(?!\w)", prompt), vocab.name


def test_prompt_holds_no_inline_domain_example(a2_enriched: tuple[EnrichRun, Any]) -> None:
    run, _ = a2_enriched

    for prompt in run.prompts():
        assert [e for e in INLINE_DOMAIN_EXAMPLES if e in prompt] == []


# -- A2: structure unchanged, model text added ------------------------------------------


def test_a2_enrichment_leaves_shape_roles_and_sections_unchanged(
    a2_enriched: tuple[EnrichRun, Any],
) -> None:
    from sema.intake.enrichment import apply_enrichment

    run, enrichment = a2_enriched
    roles, before = run.stages.roles, run.roles_before
    assert before is not None and run.stages.spec is not None

    assert roles.structure == before.structure
    assert roles.eligibilities == before.eligibilities
    assert roles.joins == before.joins
    reassembled = assemble_spec(run.descriptor, roles)
    assert reassembled is not None
    assert reassembled.model_dump() == run.stages.spec.model_dump()
    manifest = _imported_manifest(run)
    assert apply_enrichment(manifest, enrichment) == manifest


def test_a2_enrichment_adds_model_sourced_text(a2_enriched: tuple[EnrichRun, Any]) -> None:
    run, enrichment = a2_enriched
    keys = set(run.call_records())

    assert enrichment.entries
    assert {e.source for e in enrichment.entries} == {"model"}
    assert {e.call_key for e in enrichment.entries} <= keys
    label = _entry(enrichment, A2_CONCEPTS, "concept_name", "description")
    assert (label.value, label.confidence, label.kind) == (
        label_for("concept_name"),
        CONFIDENCE,
        "added",
    )
    typed = _entry(enrichment, A2_CONCEPTS, "concept_name", "semantic_type")
    assert typed.value == SEMANTIC_TYPE


def test_a2_enrichment_is_the_ledger_artifact_with_each_call_path(
    a2_enriched: tuple[EnrichRun, Any],
) -> None:
    from sema.intake.enrichment import Enrichment

    run, enrichment = a2_enriched
    calls = [e for e in run.events("call") if e["stage"] == ENRICH_STAGE]

    assert run.artifact_path().name == f"{ARTIFACT}.json"
    assert Enrichment.model_validate_json(run.artifact_path().read_text()) == enrichment
    assert len(calls) == len(run.prompts())
    assert {e["payload"]["path"] for e in calls} == {SERVED_BY}


def test_enrichment_calls_are_keyed_by_versioned_prompt_and_output_model(
    a2_enriched: tuple[EnrichRun, Any],
) -> None:
    run, _ = a2_enriched
    records = run.call_records()

    assert records
    for key, record in records.items():
        prompt, schema_name = record["prompt"], record["schema_name"]
        first, rest = prompt.split("\n", 1)
        assert first == STAGE_B_VERSION_LINE
        assert schema_name == "StageBBatchResult_v1"
        assert prompt_hash(record["model"], prompt, schema_name) == key
        bumped = f"{first[: -len('v1')]}v2\n{rest}"
        assert prompt_hash(record["model"], bumped, schema_name) != key


# -- ACRIS: source text wins ------------------------------------------------------------


def test_acris_prompt_carries_the_entity_and_its_stated_description(
    acris_enriched: tuple[EnrichRun, Any],
) -> None:
    run, _ = acris_enriched
    (prompt,) = [p for p in run.prompts() if f"Entity: {ACRIS_MASTER}\n" in p]

    (line,) = [ln for ln in prompt.splitlines() if ln.startswith(f"  {BOROUGH} (")]
    assert line.endswith(f"— {BOROUGH_DESCRIPTION}")


def test_acris_stated_description_is_kept_and_the_model_text_is_an_alternative(
    acris_enriched: tuple[EnrichRun, Any],
) -> None:
    from sema.intake.enrichment import apply_enrichment

    run, enrichment = acris_enriched
    alternative = _entry(enrichment, ACRIS_MASTER, BOROUGH, "description")
    applied = apply_enrichment(_imported_manifest(run), enrichment)
    prop = manifest_property(applied, ACRIS_MASTER, BOROUGH)

    assert (alternative.kind, alternative.value, alternative.source) == (
        "alternative",
        label_for(BOROUGH),
        "model",
    )
    assert prop["definition"] == BOROUGH_DESCRIPTION
    assert prop["decoded_values"] == BOROUGHS
    assert label_for(BOROUGH) not in str(applied)


def test_acris_model_synonyms_fill_only_empty_fields(
    acris_enriched: tuple[EnrichRun, Any],
) -> None:
    from sema.intake.enrichment import apply_enrichment

    run, enrichment = acris_enriched
    manifest = _imported_manifest(run)
    added = _entry(enrichment, ACRIS_MASTER, BOROUGH, "synonyms")
    applied = apply_enrichment(manifest, enrichment)

    assert manifest_property(manifest, ACRIS_MASTER, BOROUGH).get("synonyms", []) == []
    assert (added.kind, added.value) == ("added", [synonym_for(BOROUGH)])
    assert manifest_property(applied, ACRIS_MASTER, BOROUGH)["synonyms"] == [
        synonym_for(BOROUGH)
    ]
    assert manifest_property(manifest, ACRIS_MASTER, BOROUGH).get("synonyms", []) == []
    assert parse_manifest_raw(applied).entities
