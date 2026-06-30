"""US-009 unit tests: the deterministic VOCAB_LOOKUP assertion producer.

Hermetic. A fake decision source (the value-mapping store, written solely by
US-006) returns configured ``ValueMapping`` rows; a fake graph session records
the Cypher writes. The producer reads per-code decisions, emits ONE column-level
``MappingAssertion(pattern=VOCAB_LOOKUP)`` carrying ``source_field_ref`` and
``target_property_ref``, and materialises a ``:FieldMap`` with ``MAPS_TO`` ->
the target ``:Property`` and ``DERIVED_FROM`` -> the source ``:Property``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sema.models.planner.lifecycle import Status
from sema.models.planner.mapping_plan import MappingAssertion
from sema.models.planner.patterns import MappingPattern, VocabLookup
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.resolve.engine_utils import ResolveContext
from sema.resolve.policies.omop import (
    OMOP_ONCOTREE_CONDITION_REF,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.policy import ResolverPolicy
from sema.resolve.producer import (
    MappingNodes,
    NoResolvedDecisionError,
    VocabLookupProducer,
)
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping

pytestmark = pytest.mark.unit

_SOURCE_FIELD = "source.sample.cancer_type_code"
_TARGET_PROPERTY = "target.condition_occurrence.condition_concept_id"
_TARGET_FIELD = "condition_concept_id"
_VOCAB_RELEASE = "OMOP_2024"


class FakeDecisionSource:
    """Read-only view over configured decisions (no writes)."""

    def __init__(self, decisions: list[ValueMapping]) -> None:
        self._decisions = decisions
        self.writes = 0

    def read_all(self) -> list[ValueMapping]:
        return list(self._decisions)

    def upsert(self, _: object) -> int:  # pragma: no cover - guards against writes
        self.writes += 1
        return 0


class FakeSession:
    """Records every ``run`` call (query + named parameters)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, query: str, **params: object) -> None:
        self.calls.append((query, params))


def _binding() -> object:
    from sema.models.planner._enums import TargetArtifactKind
    from sema.models.target.refs import (
        TargetEntityRef,
        VocabularyRef,
        VocabularySource,
    )
    from sema.models.target.vocab_binding import VocabularyBindingDecl

    return VocabularyBindingDecl(
        entity_ref=TargetEntityRef(
            target_model_id="omop_condition_slice0",
            qualified_name="omop.condition_occurrence",
            kind=TargetArtifactKind.TABLE_ROW,
        ),
        property_name=_TARGET_FIELD,
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
        domain="Condition",
        require_standard=True,
        allow_zero_default=False,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
    )


@pytest.fixture()
def policy() -> ResolverPolicy:
    return make_omop_oncotree_condition_policy(_binding())


def _provenance() -> Provenance:
    return Provenance(
        run=RunProvenance(
            run_id="run-1",
            target_model_version="v1",
            target_schema_snapshot_hash="h1",
            vocab_release=_VOCAB_RELEASE,
            context_card_version="cc1",
            prompt_template_version="pt1",
            few_shot_set_version="fs1",
            constraint_version="cv1",
            llm_model="none",
        ),
        source=SourceScope(
            source_id="cbioportal",
            source_schema_hash="sh1",
            source_profile_hash="sp1",
        ),
        timestamp=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )


def _context() -> ResolveContext:
    return ResolveContext(
        source_field_ref=_SOURCE_FIELD,
        source_value_ref=_SOURCE_FIELD,
        target_property_ref=_TARGET_PROPERTY,
        target_field=_TARGET_FIELD,
        domain_constraint_ref="target.condition_occurrence.domain=Condition",
        vocabulary_ref="target.vocabulary.SNOMED",
        vocab_binding="omop.condition_occurrence.condition_concept_id",
        vocab_release=_VOCAB_RELEASE,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        run_id="run-1",
        provenance=_provenance(),
    )


def _nodes() -> MappingNodes:
    return MappingNodes(source_property_id="src-prop-1", target_property_id="tgt-prop-1")


def _decision(
    code: str,
    *,
    concept_id: int | None,
    resolution: ResolutionStatus,
    status: Status,
    confidence: float,
) -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref=_TARGET_PROPERTY,
        target_field=_TARGET_FIELD,
        vocab_binding="omop.condition_occurrence.condition_concept_id",
        concept_id=concept_id,
        vocab_release=_VOCAB_RELEASE,
        valid_start=None,
        valid_end=None,
        resolution_status=resolution,
        no_map_reason=None if resolution is ResolutionStatus.RESOLVED else "dead end",
        confidence=confidence,
        status=status,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        run_id="run-1",
    )


def _resolved(code: str = "LUAD", cid: int = 45768916) -> ValueMapping:
    return _decision(
        code,
        concept_id=cid,
        resolution=ResolutionStatus.RESOLVED,
        status=Status.auto_accepted,
        confidence=1.0,
    )


def _no_map(code: str = "DEAD") -> ValueMapping:
    return _decision(
        code,
        concept_id=None,
        resolution=ResolutionStatus.NO_MAP,
        status=Status.auto_accepted,
        confidence=1.0,
    )


def _produce(
    decisions: list[ValueMapping], policy: ResolverPolicy
) -> tuple[MappingAssertion, FakeSession, FakeDecisionSource]:
    source = FakeDecisionSource(decisions)
    session = FakeSession()
    producer = VocabLookupProducer(session)
    assertion = producer.produce(source, policy, _context(), _nodes())
    return assertion, session, source


def test_produce_emits_single_vocab_lookup_assertion(policy: ResolverPolicy) -> None:
    assertion, _, _ = _produce([_resolved(), _no_map()], policy)
    assert assertion.pattern is MappingPattern.VOCAB_LOOKUP
    assert assertion.source_field_ref == _SOURCE_FIELD
    assert assertion.target_property_ref == _TARGET_PROPERTY
    assert assertion.status is Status.auto_accepted


def test_assertion_payload_is_vocab_lookup_with_policy_ref(
    policy: ResolverPolicy,
) -> None:
    assertion, _, _ = _produce([_resolved()], policy)
    assert isinstance(assertion.payload, VocabLookup)
    assert assertion.payload.resolver_policy_ref == OMOP_ONCOTREE_CONDITION_REF


def test_produce_materializes_field_map_node(policy: ResolverPolicy) -> None:
    _, session, _ = _produce([_resolved()], policy)
    field_map_writes = [c for c in session.calls if "MERGE (n:FieldMap" in c[0]]
    assert len(field_map_writes) == 1
    props = field_map_writes[0][1]["props"]
    assert props["pattern"] == MappingPattern.VOCAB_LOOKUP.value
    assert props["target_field_ref"] == _TARGET_PROPERTY


def test_produce_writes_maps_to_to_target_property(policy: ResolverPolicy) -> None:
    _, session, _ = _produce([_resolved()], policy)
    maps_to = next(c for c in session.calls if "MAPS_TO" in c[0])
    assert "(f:FieldMap" in maps_to[0]  # edge originates at the FieldMap
    assert maps_to[1]["p_id"] == "tgt-prop-1"


def test_produce_writes_derived_from_source_property(policy: ResolverPolicy) -> None:
    _, session, _ = _produce([_resolved()], policy)
    derived = next(c for c in session.calls if "DERIVED_FROM" in c[0])
    assert "(f:FieldMap" in derived[0]
    assert derived[1]["p_id"] == "src-prop-1"


def test_field_map_and_edges_share_one_id(policy: ResolverPolicy) -> None:
    _, session, _ = _produce([_resolved()], policy)
    fm_id = next(c for c in session.calls if "MERGE (n:FieldMap" in c[0])[1]["props"][
        "id"
    ]
    maps_to = next(c for c in session.calls if "MAPS_TO" in c[0])
    derived = next(c for c in session.calls if "DERIVED_FROM" in c[0])
    assert maps_to[1]["fm_id"] == fm_id
    assert derived[1]["fm_id"] == fm_id


def test_review_pending_decision_escalates_assertion(policy: ResolverPolicy) -> None:
    tie = _decision(
        "AMB",
        concept_id=100,
        resolution=ResolutionStatus.RESOLVED,
        status=Status.review_pending,
        confidence=0.5,
    )
    assertion, _, _ = _produce([_resolved(), tie], policy)
    assert assertion.status is Status.review_pending
    assert assertion.confidence < 1.0


def test_no_resolved_decisions_raises(policy: ResolverPolicy) -> None:
    source = FakeDecisionSource([_no_map("DEAD"), _no_map("GONE")])
    producer = VocabLookupProducer(FakeSession())
    with pytest.raises(NoResolvedDecisionError):
        producer.produce(source, policy, _context(), _nodes())


def test_producer_never_writes_the_value_mapping_store(
    policy: ResolverPolicy,
) -> None:
    _, _, source = _produce([_resolved()], policy)
    assert source.writes == 0


def test_decisions_for_other_bindings_are_ignored(policy: ResolverPolicy) -> None:
    other = _resolved("KIRC", cid=1)
    object.__setattr__(other, "target_property_ref", "target.other.x")
    assertion, session, _ = _produce([_resolved(), other], policy)
    field_map_writes = [c for c in session.calls if "MERGE (n:FieldMap" in c[0]]
    assert len(field_map_writes) == 1
    assert assertion.target_property_ref == _TARGET_PROPERTY
