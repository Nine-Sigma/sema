"""US-007: OMOP condition_occurrence TARGET via the authored Slice-0 manifest.

Hermetic (InMemoryGraphWriter) assertions that loading the authored
manifest through the MERGED targets/ framework materialises a
model_role=TARGET :Entity/:Property/:TargetObligation, a
:VocabularyBinding (property_name=condition_concept_id, domain=Condition,
require_standard) and a HAS_VOCABULARY_BINDING edge, and that the
binding's ``resolver_policy_ref`` resolves to the US-004 OMOP/OncoTree
policy. No OMOP literal lives in code here — all of it is in the YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.policies import resolve_policy
from showcase.cbioportal_to_omop.omop_policy import OMOP_ONCOTREE_CONDITION_REF
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
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.unit

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "showcase" / "cbioportal_to_omop" / "manifests"
    / "omop_condition_slice0.yaml"
)

_CONDITION_PROPERTY = "condition_concept_id"
_CONDITION_DOMAIN = "Condition"
_CONDITION_VOCABULARY = "OMOP-Condition"
_START_DATE_PROPERTY = "condition_start_date"
_EXPECTED_MODEL_VERSION = "0.2.0"


def _load() -> InMemoryGraphWriter:
    writer = InMemoryGraphWriter()
    load_target(ManifestTargetAdapter(MANIFEST), writer=writer)
    return writer


def _entity_qname() -> str:
    ops = [op for op in _load().ops if isinstance(op, EntityOp)]
    names = [op.qualified_name for op in ops]
    return next(n for n in names if n.endswith("condition_occurrence"))


def test_manifest_file_exists() -> None:
    assert MANIFEST.is_file()


def test_manifest_loads_without_dangling_ref() -> None:
    # Repoint regression: the binding `vocabulary` slot is resolved against the
    # top-level `vocabularies:` declaration. Renaming the binding to the
    # domain-governed sentinel WITHOUT renaming the declaration would fall back
    # to INLINE and raise DanglingRefError in the normalizer. A clean load
    # proves both were renamed together.
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(MANIFEST))
    decl = next(
        b
        for b in normalized.vocabulary_bindings
        if b.property_name == _CONDITION_PROPERTY
    )
    assert decl.vocabulary.name == _CONDITION_VOCABULARY


def test_condition_occurrence_entity_materialized() -> None:
    qname = _entity_qname()
    assert qname.endswith("condition_occurrence")


def test_condition_concept_id_property_materialized() -> None:
    qname = _entity_qname()
    props = {
        op.name
        for op in _load().ops
        if isinstance(op, PropertyOp)
        and op.parent_entity_qualified_name == qname
    }
    assert _CONDITION_PROPERTY in props


def test_target_obligation_requires_condition_concept_id() -> None:
    qname = _entity_qname()
    obligation = next(
        op
        for op in _load().ops
        if isinstance(op, TargetObligationOp) and op.target_entity == qname
    )
    assert _CONDITION_PROPERTY in obligation.payload["required_fields"]


def test_model_version_bumped_to_0_2_0_for_d4() -> None:
    # D4: the nullable-date contract change is an explicit version bump, not a
    # silent runtime relaxation.
    prop = next(op for op in _load().ops if isinstance(op, PropertyOp))
    assert prop.target_model_version == _EXPECTED_MODEL_VERSION


def test_condition_start_date_is_nullable_d4() -> None:
    # S1-00 proved no absolute date (nor anchor) exists in the cBio source;
    # D4 versions the contract to nullable rather than fabricating a date.
    qname = _entity_qname()
    prop = next(
        op
        for op in _load().ops
        if isinstance(op, PropertyOp)
        and op.parent_entity_qualified_name == qname
        and op.name == _START_DATE_PROPERTY
    )
    assert prop.nullable is True


def test_condition_start_date_dropped_from_required_fields_d4() -> None:
    qname = _entity_qname()
    obligation = next(
        op
        for op in _load().ops
        if isinstance(op, TargetObligationOp) and op.target_entity == qname
    )
    required = obligation.payload["required_fields"]
    assert _START_DATE_PROPERTY not in required
    # the other three OMOP-shape requirements stay required.
    assert {"condition_occurrence_id", "person_id", _CONDITION_PROPERTY}.issubset(
        set(required)
    )


def test_vocabulary_binding_op_carries_condition_binding() -> None:
    binding = next(
        op
        for op in _load().ops
        if isinstance(op, VocabularyBindingOp)
        and op.property_name == _CONDITION_PROPERTY
    )
    assert binding.domain == _CONDITION_DOMAIN
    assert binding.require_standard is True
    # bug-369 F2: the target is OMOP-standard-Condition (vocabulary-agnostic),
    # declared machine-readably; the vocabulary slot names the governance scope.
    assert binding.standard_domain_governed is True
    assert binding.vocabulary_name == _CONDITION_VOCABULARY
    assert binding.resolver_policy_ref == OMOP_ONCOTREE_CONDITION_REF


def test_has_vocabulary_binding_edge_present_not_misspelled() -> None:
    rels = [op for op in _load().ops if isinstance(op, RelationshipOp)]
    rel_types = {r.rel_type for r in rels}
    assert "HAS_VOCABULARY_BINDING" in rel_types
    assert "HAS_VOCAB_BINDING" not in rel_types
    binding_rel = next(r for r in rels if r.rel_type == "HAS_VOCABULARY_BINDING")
    assert binding_rel.from_label == "Property"
    assert binding_rel.to_label == "VocabularyBinding"
    assert binding_rel.to_keys["property_name"] == _CONDITION_PROPERTY


def test_normalized_binding_resolves_to_us004_policy() -> None:
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(MANIFEST))
    decl: VocabularyBindingDecl = next(
        b
        for b in normalized.vocabulary_bindings
        if b.property_name == _CONDITION_PROPERTY
    )
    assert decl.domain == _CONDITION_DOMAIN
    assert decl.require_standard is True
    policy = resolve_policy(decl)
    assert policy.source_vocabulary == "OncoTree"
    assert policy.target_domain == _CONDITION_DOMAIN
