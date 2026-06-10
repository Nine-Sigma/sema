"""Materializer behavior: write sequence, idempotency, endpoint properties."""

from __future__ import annotations

import pytest

from sema.models.planner._enums import PrimaryKeyStrategy
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.properties import TargetPropertyDecl
from sema.targets.loader import load_target
from sema.targets.materializer import InMemoryGraphWriter
from sema.targets.materializer_ops import (
    ConstraintOp,
    ContextCardOp,
    EnrichmentDecisionOp,
    EntityOp,
    PropertyOp,
    TargetObligationOp,
    VocabularyBindingOp,
)

from tests.unit.targets.conftest import (
    ScriptedAdapter,
    make_descriptor,
    make_graph_edge_entity,
    make_graph_node_entity,
    make_obligation,
    make_table_row_entity,
)

pytestmark = pytest.mark.unit


def _basic_adapter() -> ScriptedAdapter:
    entity = make_table_row_entity()
    obligation = make_obligation()
    return ScriptedAdapter(make_descriptor(), [entity], [obligation])


def test_materializer_writes_entity_property_obligation_decision() -> None:
    writer = InMemoryGraphWriter()
    load_target(_basic_adapter(), writer=writer)
    op_types = [type(op).__name__ for op in writer.ops]
    assert "EntityOp" in op_types
    assert "PropertyOp" in op_types
    assert "TargetObligationOp" in op_types
    assert "EnrichmentDecisionOp" in op_types


def test_idempotent_write_sequence() -> None:
    adapter = _basic_adapter()
    writer1 = InMemoryGraphWriter()
    writer2 = InMemoryGraphWriter()
    load_target(adapter, writer=writer1)
    load_target(adapter, writer=writer2)
    assert _normalize_ops(writer1.ops) == _normalize_ops(writer2.ops)


def _normalize_ops(ops: list[object]) -> list[tuple[str, str]]:
    return [(type(op).__name__, _digest_for_op(op)) for op in ops]


def _digest_for_op(op: object) -> str:
    if isinstance(op, EnrichmentDecisionOp):
        return f"{op.entity_ref}|{op.target_schema_snapshot_hash}"
    if isinstance(op, EntityOp):
        return f"{op.qualified_name}|{op.target_schema_snapshot_hash}"
    if isinstance(op, PropertyOp):
        return (
            f"{op.parent_entity_qualified_name}.{op.name}|"
            f"{op.target_schema_snapshot_hash}|{op.property_kind}"
        )
    if isinstance(op, TargetObligationOp):
        return f"{op.target_entity}|{op.target_schema_snapshot_hash}"
    return repr(op)


def test_endpoint_properties_emitted_with_endpoint_kind() -> None:
    llc = make_graph_node_entity(qualified_name="acris.LLC")
    prop_node = make_graph_node_entity(qualified_name="acris.Property")
    edge = make_graph_edge_entity(
        target_model_id="fake-target",
        qualified_name="acris.OWNS",
        subject_target=llc.ref,
        object_target=prop_node.ref,
        columnar_properties=[
            TargetPropertyDecl(name="valid_from", type="date", nullable=False)
        ],
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [llc, prop_node, edge],
        [
            make_obligation("acris.LLC", required_fields=["name"]),
            make_obligation("acris.Property", required_fields=["name"]),
            TargetObligationDecl(
                target_entity="acris.OWNS",
                required_fields=["subject", "object", "valid_from"],
                primary_key=PrimaryKeyStrategy.NATURAL_KEY,
            ),
        ],
    )
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer)
    edge_props = [
        op
        for op in writer.ops
        if isinstance(op, PropertyOp) and op.parent_entity_qualified_name == "acris.OWNS"
    ]
    by_name = {p.name: p for p in edge_props}
    assert {"subject", "object", "valid_from"} <= set(by_name)
    subject = by_name["subject"]
    assert subject.property_kind == "ENDPOINT"
    assert subject.endpoint_role == "subject"
    assert subject.endpoint_target_entity_qualified_name == "acris.LLC"
    assert subject.materialized_as_edge_property is False
    valid_from = by_name["valid_from"]
    assert valid_from.property_kind == "COLUMN"
    assert valid_from.materialized_as_edge_property is True


def test_entity_op_carries_target_provenance_and_compact_status() -> None:
    writer = InMemoryGraphWriter()
    loaded = load_target(_basic_adapter(), writer=writer)
    entity_ops = [op for op in writer.ops if isinstance(op, EntityOp)]
    assert len(entity_ops) == 1
    op = entity_ops[0]
    assert op.target_model_id == "fake-target"
    assert op.target_schema_snapshot_hash == loaded.target_schema_snapshot_hash
    assert set(op.enrichment_status.keys()) == {
        "structure",
        "obligations",
        "vocabulary_bindings",
        "semantic_aliases",
        "terms",
    }


def test_materializer_writes_constraint_for_each_domain_constraint() -> None:
    from sema.models.planner.target_model import DomainConstraint

    entity = make_table_row_entity(
        properties=[
            TargetPropertyDecl(name="person_id", type="string", nullable=False),
            TargetPropertyDecl(
                name="gender_concept_id", type="string", nullable=False
            ),
        ],
    )
    obligation = TargetObligationDecl(
        target_entity="fake.person",
        required_fields=["person_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        domain_constraints=[
            DomainConstraint(
                property_name="gender_concept_id", domain_id="omop.Gender"
            ),
        ],
    )
    adapter = ScriptedAdapter(make_descriptor(), [entity], [obligation])
    writer = InMemoryGraphWriter()
    loaded = load_target(adapter, writer=writer)
    constraint_ops = [op for op in writer.ops if isinstance(op, ConstraintOp)]
    assert len(constraint_ops) == 1
    cop = constraint_ops[0]
    assert cop.target_model_id == "fake-target"
    assert cop.target_schema_snapshot_hash == loaded.target_schema_snapshot_hash
    assert cop.attached_property_id == "fake.person.gender_concept_id"
    assert cop.constraint_kind == "domain_binding"
    assert cop.payload == {"domain_id": "omop.Gender"}


def test_materializer_skips_constraints_for_endpoint_properties() -> None:
    from sema.models.planner.target_model import DomainConstraint

    llc = make_graph_node_entity(qualified_name="acris.LLC")
    prop_node = make_graph_node_entity(qualified_name="acris.Property")
    edge = make_graph_edge_entity(
        target_model_id="fake-target",
        qualified_name="acris.OWNS",
        subject_target=llc.ref,
        object_target=prop_node.ref,
    )
    edge_obligation = TargetObligationDecl(
        target_entity="acris.OWNS",
        required_fields=["subject", "object"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        domain_constraints=[
            DomainConstraint(property_name="subject", domain_id="ignored"),
        ],
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [llc, prop_node, edge],
        [
            make_obligation("acris.LLC", required_fields=["name"]),
            make_obligation("acris.Property", required_fields=["name"]),
            edge_obligation,
        ],
    )
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer)
    constraint_ops = [op for op in writer.ops if isinstance(op, ConstraintOp)]
    edge_constraints = [
        c for c in constraint_ops if c.attached_property_id.startswith("acris.OWNS.")
    ]
    assert edge_constraints == []


def test_constraint_op_idempotent_across_runs() -> None:
    from sema.models.planner.target_model import DomainConstraint

    entity = make_table_row_entity(
        properties=[
            TargetPropertyDecl(name="person_id", type="string", nullable=False),
            TargetPropertyDecl(
                name="gender_concept_id", type="string", nullable=False
            ),
        ],
    )
    obligation = TargetObligationDecl(
        target_entity="fake.person",
        required_fields=["person_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        domain_constraints=[
            DomainConstraint(
                property_name="gender_concept_id", domain_id="omop.Gender"
            ),
        ],
    )
    adapter = ScriptedAdapter(make_descriptor(), [entity], [obligation])
    w1 = InMemoryGraphWriter()
    w2 = InMemoryGraphWriter()
    load_target(adapter, writer=w1)
    load_target(adapter, writer=w2)
    c1 = [op for op in w1.ops if isinstance(op, ConstraintOp)]
    c2 = [op for op in w2.ops if isinstance(op, ConstraintOp)]
    assert c1 == c2


def test_has_property_relationship_carries_full_hash_versioned_keys() -> None:
    from sema.targets.materializer_ops import RelationshipOp

    writer = InMemoryGraphWriter()
    loaded = load_target(_basic_adapter(), writer=writer)
    rels = [
        op
        for op in writer.ops
        if isinstance(op, RelationshipOp) and op.rel_type == "HAS_PROPERTY"
    ]
    assert len(rels) == 1
    rel = rels[0]
    h = loaded.target_schema_snapshot_hash
    for keys in (rel.from_keys, rel.to_keys):
        assert keys["target_model_id"] == "fake-target"
        assert keys["target_model_version"] == "1.0.0"
        assert keys["target_schema_snapshot_hash"] == h


def test_has_obligation_relationship_emitted_with_full_keys() -> None:
    from sema.targets.materializer_ops import RelationshipOp

    writer = InMemoryGraphWriter()
    loaded = load_target(_basic_adapter(), writer=writer)
    rels = [
        op
        for op in writer.ops
        if isinstance(op, RelationshipOp) and op.rel_type == "HAS_OBLIGATION"
    ]
    assert len(rels) == 1
    rel = rels[0]
    h = loaded.target_schema_snapshot_hash
    for keys in (rel.from_keys, rel.to_keys):
        assert keys["target_model_id"] == "fake-target"
        assert keys["target_model_version"] == "1.0.0"
        assert keys["target_schema_snapshot_hash"] == h


def test_has_enrichment_decision_relationship_carries_full_keys() -> None:
    from sema.targets.materializer_ops import RelationshipOp

    writer = InMemoryGraphWriter()
    loaded = load_target(_basic_adapter(), writer=writer)
    rels = [
        op
        for op in writer.ops
        if isinstance(op, RelationshipOp) and op.rel_type == "HAS_ENRICHMENT_DECISION"
    ]
    assert len(rels) == 1
    rel = rels[0]
    h = loaded.target_schema_snapshot_hash
    for keys in (rel.from_keys, rel.to_keys):
        assert keys["target_model_id"] == "fake-target"
        assert keys["target_model_version"] == "1.0.0"
        assert keys["target_schema_snapshot_hash"] == h


def test_materializer_emits_current_flip_op_for_loaded_logical_keys() -> None:
    from sema.targets.materializer_ops import CurrentFlipOp

    writer = InMemoryGraphWriter()
    loaded = load_target(_basic_adapter(), writer=writer)
    flips = [op for op in writer.ops if isinstance(op, CurrentFlipOp)]
    assert len(flips) == 1
    flip = flips[0]
    assert flip.target_model_id == "fake-target"
    assert flip.target_model_version == "1.0.0"
    assert flip.current_snapshot_hash == loaded.target_schema_snapshot_hash
    assert "fake.person" in flip.entity_qualified_names


def test_lazy_subset_flip_scoped_to_selected_only() -> None:
    from sema.targets.materializer_ops import CurrentFlipOp

    e1 = make_table_row_entity(qualified_name="omop.person")
    e2 = make_table_row_entity(qualified_name="omop.observation")
    o1 = make_obligation("omop.person")
    o2 = make_obligation("omop.observation")
    adapter = ScriptedAdapter(make_descriptor(), [e1, e2], [o1, o2])
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer, selected_refs=[e1.ref])
    flips = [op for op in writer.ops if isinstance(op, CurrentFlipOp)]
    assert len(flips) == 1
    assert flips[0].entity_qualified_names == ("omop.person",)


def test_vocabulary_binding_op_emitted_with_full_hooks() -> None:
    from sema.models.target.refs import VocabularyRef, VocabularySource
    from sema.models.target.vocab_binding import VocabularyBindingDecl

    entity = make_table_row_entity(
        properties=[
            TargetPropertyDecl(name="person_id", type="string", nullable=False),
            TargetPropertyDecl(
                name="gender_concept_id", type="string", nullable=False
            ),
        ],
    )
    binding = VocabularyBindingDecl(
        entity_ref=entity.ref,
        property_name="gender_concept_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
        domain="Gender",
        require_standard=True,
        allow_zero_default=True,
        effective_date_ref="visit.start_date",
        resolver_policy_ref="omop.snomed.gender.v1",
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [entity],
        [make_obligation(target_entity=entity.ref.qualified_name)],
        bindings=[binding],
    )
    writer = InMemoryGraphWriter()
    loaded = load_target(adapter, writer=writer)
    binding_ops = [op for op in writer.ops if isinstance(op, VocabularyBindingOp)]
    assert len(binding_ops) == 1
    bop = binding_ops[0]
    assert bop.parent_entity_qualified_name == entity.ref.qualified_name
    assert bop.property_name == "gender_concept_id"
    assert bop.vocabulary_name == "SNOMED"
    assert bop.vocabulary_source == "EXTERNAL"
    assert bop.domain == "Gender"
    assert bop.require_standard is True
    assert bop.allow_zero_default is True
    assert bop.effective_date_ref == "visit.start_date"
    assert bop.resolver_policy_ref == "omop.snomed.gender.v1"
    assert bop.target_schema_snapshot_hash == loaded.target_schema_snapshot_hash


def test_vocabulary_binding_relationship_emitted() -> None:
    from sema.models.target.refs import VocabularyRef, VocabularySource
    from sema.models.target.vocab_binding import VocabularyBindingDecl
    from sema.targets.materializer_ops import RelationshipOp

    entity = make_table_row_entity()
    binding = VocabularyBindingDecl(
        entity_ref=entity.ref,
        property_name="person_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [entity],
        [make_obligation()],
        bindings=[binding],
    )
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer)
    rels = [
        op
        for op in writer.ops
        if isinstance(op, RelationshipOp) and op.rel_type == "HAS_VOCABULARY_BINDING"
    ]
    assert len(rels) == 1
    assert rels[0].from_label == "Property"
    assert rels[0].to_label == "VocabularyBinding"


def test_context_card_op_emitted_with_content_and_hash() -> None:
    writer = InMemoryGraphWriter()
    loaded = load_target(_basic_adapter(), writer=writer)
    card_ops = [op for op in writer.ops if isinstance(op, ContextCardOp)]
    assert len(card_ops) == 1
    cop = card_ops[0]
    assert cop.entity_qualified_name == "fake.person"
    assert cop.card_version == "0.0.0+synthesized"
    assert len(cop.card_hash) == 64
    assert cop.target_schema_snapshot_hash == loaded.target_schema_snapshot_hash


def test_context_card_relationship_emitted_from_entity() -> None:
    from sema.targets.materializer_ops import RelationshipOp

    writer = InMemoryGraphWriter()
    load_target(_basic_adapter(), writer=writer)
    rels = [
        op
        for op in writer.ops
        if isinstance(op, RelationshipOp) and op.rel_type == "HAS_CONTEXT_CARD"
    ]
    assert len(rels) == 1
    assert rels[0].from_label == "Entity"
    assert rels[0].to_label == "ContextCard"


def test_loaded_target_exposes_context_cards_and_hashes() -> None:
    loaded = load_target(_basic_adapter(), writer=InMemoryGraphWriter())
    assert len(loaded.context_cards) == 1
    assert loaded.context_cards[0].entity_ref.qualified_name == "fake.person"
    assert "fake.person" in loaded.card_hashes
    assert len(loaded.card_hashes["fake.person"]) == 64


def test_current_flip_op_includes_binding_card_term_keys() -> None:
    from sema.models.target.refs import VocabularyRef, VocabularySource
    from sema.models.target.term import TargetTermDecl
    from sema.models.target.vocab_binding import VocabularyBindingDecl
    from sema.targets.materializer_ops import CurrentFlipOp

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
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer)
    flips = [op for op in writer.ops if isinstance(op, CurrentFlipOp)]
    assert len(flips) == 1
    flip = flips[0]
    assert ("fake.person", "person_id", "GENDER_CV") in flip.vocabulary_binding_keys
    assert any(k[0] == "fake.person" for k in flip.context_card_keys)
    assert ("GENDER_CV", "M") in flip.term_keys


def test_decision_op_decisions_json_round_trips() -> None:
    import json

    writer = InMemoryGraphWriter()
    load_target(_basic_adapter(), writer=writer)
    decision_ops = [op for op in writer.ops if isinstance(op, EnrichmentDecisionOp)]
    assert len(decision_ops) == 1
    parsed = json.loads(decision_ops[0].decisions_json)
    assert set(parsed.keys()) == {
        "structure",
        "obligations",
        "vocabulary_bindings",
        "semantic_aliases",
        "terms",
    }
    assert parsed["semantic_aliases"]["status"] == "required_deferred"
