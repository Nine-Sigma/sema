"""End-to-end loader pipeline tests: load_target orchestration + decisions + cards."""

from __future__ import annotations

import pytest

from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.context_card import TargetContextCard
from sema.models.target.enrichment import EnrichmentStatus, Facet
from sema.models.target.properties import TargetPropertyDecl
from sema.models.target.refs import VocabularyRef, VocabularySource
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.exceptions import CardContentDriftError, LoaderStageOrderError
from sema.targets.hashing import compute_card_hash
from sema.targets.loader import load_target
from sema.targets.materializer import (
    InMemoryGraphWriter,
    StageGuard,
    TargetModelMaterializer,
)

from tests.unit.targets.conftest import (
    ScriptedAdapter,
    default_completeness,
    make_descriptor,
    make_obligation,
    make_table_row_entity,
)

pytestmark = pytest.mark.unit


def _basic_adapter(target_model_id: str = "fake-target") -> ScriptedAdapter:
    entity = make_table_row_entity(target_model_id=target_model_id)
    obligation = make_obligation()
    return ScriptedAdapter(make_descriptor(target_model_id), [entity], [obligation])


def test_load_target_emits_loaded_target_with_pinned_hash() -> None:
    writer = InMemoryGraphWriter()
    loaded = load_target(_basic_adapter(), writer=writer)
    assert len(loaded.target_schema_snapshot_hash) == 64
    assert loaded.entity_refs[0].qualified_name == "fake.person"
    assert len(loaded.enrichment_decisions) == 1


def test_one_decision_record_per_entity() -> None:
    a = make_table_row_entity(qualified_name="fake.alpha")
    b = make_table_row_entity(qualified_name="fake.beta")
    c = make_table_row_entity(qualified_name="fake.gamma")
    obligations = [
        make_obligation("fake.alpha"),
        make_obligation("fake.beta"),
        make_obligation("fake.gamma"),
    ]
    adapter = ScriptedAdapter(make_descriptor(), [a, b, c], obligations)
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    assert len(loaded.enrichment_decisions) == 3


def test_decision_covers_five_facets_exactly() -> None:
    loaded = load_target(_basic_adapter(), writer=InMemoryGraphWriter())
    record = loaded.enrichment_decisions[0]
    assert set(record.decisions.keys()) == set(Facet)


def test_complete_facet_yields_not_required() -> None:
    completeness = SemanticCompletenessAnnotations(
        structure=SemanticCompleteness.COMPLETE,
        obligations=SemanticCompleteness.COMPLETE,
        vocabulary_bindings=SemanticCompleteness.COMPLETE,
        semantic_aliases=SemanticCompleteness.COMPLETE,
        terms=SemanticCompleteness.COMPLETE,
    )
    descriptor = make_descriptor(completeness=completeness)
    entity = make_table_row_entity()
    adapter = ScriptedAdapter(descriptor, [entity], [make_obligation()])
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    statuses = {f: d.status for f, d in loaded.enrichment_decisions[0].decisions.items()}
    assert all(s is EnrichmentStatus.not_required for s in statuses.values())


def test_partial_facet_yields_required_deferred() -> None:
    loaded = load_target(_basic_adapter(), writer=InMemoryGraphWriter())
    record = loaded.enrichment_decisions[0]
    assert (
        record.decisions[Facet.semantic_aliases].status is EnrichmentStatus.required_deferred
    )


def test_external_facet_yields_not_required() -> None:
    loaded = load_target(_basic_adapter(), writer=InMemoryGraphWriter())
    record = loaded.enrichment_decisions[0]
    assert record.decisions[Facet.terms].status is EnrichmentStatus.not_required


def test_skip_facets_opt_out_yields_required_skipped() -> None:
    loaded = load_target(
        _basic_adapter(),
        writer=InMemoryGraphWriter(),
        skip_facets=["semantic_aliases"],
    )
    record = loaded.enrichment_decisions[0]
    assert (
        record.decisions[Facet.semantic_aliases].status is EnrichmentStatus.required_skipped
    )
    assert record.decisions[Facet.structure].status is EnrichmentStatus.not_required


def test_inline_synonyms_yield_supplied_by_adapter() -> None:
    entity = make_table_row_entity(
        properties=[
            TargetPropertyDecl(
                name="person_id",
                type="string",
                nullable=False,
                synonyms=["alias1", "alias2"],
            )
        ]
    )
    adapter = ScriptedAdapter(make_descriptor(), [entity], [make_obligation()])
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    record = loaded.enrichment_decisions[0]
    assert (
        record.decisions[Facet.semantic_aliases].status
        is EnrichmentStatus.supplied_by_adapter
    )


def test_inline_terms_with_binding_yield_supplied_by_adapter_for_terms() -> None:
    entity = make_table_row_entity()
    binding = VocabularyBindingDecl(
        entity_ref=entity.ref,
        property_name="person_id",
        vocabulary=VocabularyRef(name="GENDER_CV", source=VocabularySource.INLINE),
    )
    term = TargetTermDecl(
        vocabulary=VocabularyRef(name="GENDER_CV", source=VocabularySource.INLINE),
        code="M",
        display="Male",
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [entity],
        [make_obligation()],
        bindings=[binding],
        terms=[term],
    )
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    record = loaded.enrichment_decisions[0]
    assert record.decisions[Facet.terms].status is EnrichmentStatus.supplied_by_adapter


def test_aggregate_card_version_single_entity_returns_literal() -> None:
    loaded = load_target(_basic_adapter(), writer=InMemoryGraphWriter())
    assert loaded.aggregate_context_card_version == "0.0.0+synthesized"


def test_aggregate_card_version_multi_entity_is_deterministic_hash() -> None:
    a = make_table_row_entity(qualified_name="fake.alpha")
    b = make_table_row_entity(qualified_name="fake.beta")
    cards = [
        TargetContextCard(entity_ref=a.ref, card_version="1.0.0", description="A"),
        TargetContextCard(entity_ref=b.ref, card_version="2.1.3", description="B"),
    ]
    adapter = ScriptedAdapter(
        make_descriptor(),
        [a, b],
        [make_obligation("fake.alpha"), make_obligation("fake.beta")],
        cards=cards,
    )
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    aggregate = loaded.aggregate_context_card_version
    assert len(aggregate) == 64
    int(aggregate, 16)
    loaded2 = load_target(adapter, writer=InMemoryGraphWriter())
    assert loaded2.aggregate_context_card_version == aggregate


def test_card_content_drift_detected() -> None:
    entity = make_table_row_entity()
    card = TargetContextCard(
        entity_ref=entity.ref, card_version="1.0.0", description="Original"
    )
    adapter = ScriptedAdapter(
        make_descriptor(), [entity], [make_obligation()], cards=[card]
    )
    persisted = {
        (entity.ref.target_model_id, entity.ref.qualified_name, "1.0.0"): "deadbeef" * 8,
    }
    with pytest.raises(CardContentDriftError):
        load_target(
            adapter, writer=InMemoryGraphWriter(), persisted_card_hashes=persisted
        )


def test_persisted_card_hashes_keyed_by_target_model_id() -> None:
    """Two distinct target_model_ids that happen to declare the same
    qualified_name + card_version MUST not cross-poison each other's
    drift checks."""
    entity = make_table_row_entity(target_model_id="omop")
    other_entity = make_table_row_entity(target_model_id="acris")
    card = TargetContextCard(
        entity_ref=entity.ref, card_version="1.0.0", description="omop card"
    )
    other_card = TargetContextCard(
        entity_ref=other_entity.ref, card_version="1.0.0", description="acris card"
    )
    adapter = ScriptedAdapter(
        make_descriptor(target_model_id="omop"),
        [entity],
        [make_obligation()],
        cards=[card],
    )
    other_adapter = ScriptedAdapter(
        make_descriptor(target_model_id="acris"),
        [other_entity],
        [make_obligation()],
        cards=[other_card],
    )
    persisted = {
        ("omop", entity.ref.qualified_name, "1.0.0"): compute_card_hash(card),
    }
    load_target(
        adapter, writer=InMemoryGraphWriter(), persisted_card_hashes=persisted
    )
    load_target(
        other_adapter,
        writer=InMemoryGraphWriter(),
        persisted_card_hashes=persisted,
    )


def test_card_drift_under_same_target_model_id_still_raises() -> None:
    entity = make_table_row_entity(target_model_id="omop")
    card = TargetContextCard(
        entity_ref=entity.ref, card_version="1.0.0", description="X"
    )
    adapter = ScriptedAdapter(
        make_descriptor(target_model_id="omop"),
        [entity],
        [make_obligation()],
        cards=[card],
    )
    persisted = {
        ("omop", entity.ref.qualified_name, "1.0.0"): "deadbeef" * 8,
    }
    with pytest.raises(CardContentDriftError):
        load_target(
            adapter, writer=InMemoryGraphWriter(), persisted_card_hashes=persisted
        )


def test_card_version_bump_permits_content_change() -> None:
    entity = make_table_row_entity()
    card_v1 = TargetContextCard(
        entity_ref=entity.ref, card_version="1.0.0", description="Old"
    )
    adapter_v1 = ScriptedAdapter(
        make_descriptor(), [entity], [make_obligation()], cards=[card_v1]
    )
    loaded_v1 = load_target(adapter_v1, writer=InMemoryGraphWriter())
    persisted = {
        (entity.ref.target_model_id, entity.ref.qualified_name, "1.0.0"): compute_card_hash(card_v1),
    }
    card_v2 = TargetContextCard(
        entity_ref=entity.ref, card_version="1.1.0", description="New"
    )
    adapter_v2 = ScriptedAdapter(
        make_descriptor(), [entity], [make_obligation()], cards=[card_v2]
    )
    loaded_v2 = load_target(
        adapter_v2,
        writer=InMemoryGraphWriter(),
        persisted_card_hashes=persisted,
    )
    assert loaded_v1.aggregate_context_card_version == "1.0.0"
    assert loaded_v2.aggregate_context_card_version == "1.1.0"


def test_lazy_subset_hash_differs_from_full() -> None:
    a = make_table_row_entity(qualified_name="fake.alpha")
    b = make_table_row_entity(qualified_name="fake.beta")
    obligations = [make_obligation("fake.alpha"), make_obligation("fake.beta")]
    adapter = ScriptedAdapter(make_descriptor(), [a, b], obligations)
    full = load_target(adapter, writer=InMemoryGraphWriter())
    subset = load_target(
        adapter, writer=InMemoryGraphWriter(), selected_refs=[a.ref]
    )
    assert full.target_schema_snapshot_hash != subset.target_schema_snapshot_hash


def test_stage_spy_observes_normalize_hash_materialize_order() -> None:
    events: list[str] = []
    load_target(
        _basic_adapter(), writer=InMemoryGraphWriter(), stage_spy=events.append
    )
    assert events == [
        "normalize_started",
        "normalize_completed",
        "hash_started",
        "hash_completed",
        "materialize_started",
        "materialize_completed",
    ]


def test_calling_materializer_before_hash_raises() -> None:
    writer = InMemoryGraphWriter()
    guard = StageGuard()
    adapter = _basic_adapter()
    from sema.targets.normalizer import TargetModelNormalizer

    normalized = TargetModelNormalizer.normalize(adapter)
    guard.transition_to(StageGuard.NORMALIZED)
    with pytest.raises(LoaderStageOrderError):
        TargetModelMaterializer.write(
            normalized,
            "0" * 64,
            writer,
            enrichment_decisions=[],
            stage_guard=guard,
        )
