"""US-006 unit tests: the deterministic vocabulary resolver (§4 algorithm).

Hermetic: a fake VocabStore (records calls, returns configured ConceptRows) and
a real OMOP/OncoTree policy. One test per §4 stage — candidate generation (1a),
'Maps to' standardization (2), the domain gate (3) — plus the orchestrated
RESOLVED / NO_MAP / ambiguous-tie outcomes, the code-bearing short-circuit (no
LLM on the hot path), and the sole-writer store round-trip.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

import pytest

from sema.models.planner._enums import TargetArtifactKind
from sema.models.planner.lifecycle import Status
from sema.models.planner.patterns import MappingPattern, VocabLookup
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.models.target.refs import TargetEntityRef, VocabularyRef, VocabularySource
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.candidates import generate_candidates
from sema.resolve.domain_gate import apply_domain_gate
from sema.resolve.engine import VocabularyResolver
from sema.resolve.engine_utils import CodeResolution, ResolveContext
from sema.resolve.policies.omop import (
    OMOP_ONCOTREE_CONDITION_REF,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.policy import ResolverPolicy
from sema.resolve.standardize import standardize
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import ResolutionStatus
from sema.resolve.vocab_store_utils import ConceptRow

pytestmark = pytest.mark.unit


class FakeVocabStore:
    """Records calls and returns configured rows; no real backend."""

    def __init__(
        self,
        *,
        by_code: dict[tuple[str, str], ConceptRow] | None = None,
        maps_to: dict[str, list[ConceptRow]] | None = None,
    ) -> None:
        self._by_code = by_code or {}
        self._maps_to = maps_to or {}
        self.code_calls: list[tuple[str, str]] = []
        self.maps_to_calls: list[dict[str, Any]] = []

    def concept_by_code(self, vocabulary: str, code: str) -> ConceptRow | None:
        self.code_calls.append((vocabulary, code))
        return self._by_code.get((vocabulary, code))

    def maps_to_targets(
        self,
        concept_id: str,
        *,
        relationship_id: str,
        standard_flag: str | None = None,
        only_valid: bool = False,
    ) -> list[ConceptRow]:
        self.maps_to_calls.append(
            {
                "concept_id": concept_id,
                "relationship_id": relationship_id,
                "standard_flag": standard_flag,
                "only_valid": only_valid,
            }
        )
        return list(self._maps_to.get(concept_id, []))


def _binding() -> VocabularyBindingDecl:
    return VocabularyBindingDecl(
        entity_ref=TargetEntityRef(
            target_model_id="omop_condition_slice0",
            qualified_name="omop.condition_occurrence",
            kind=TargetArtifactKind.TABLE_ROW,
        ),
        property_name="condition_concept_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
        domain="Condition",
        require_standard=True,
        allow_zero_default=False,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
    )


@pytest.fixture()
def policy() -> ResolverPolicy:
    return make_omop_oncotree_condition_policy(_binding())


def _source(code: str = "LUAD", cid: str = "777926") -> ConceptRow:
    return ConceptRow(
        id=cid,
        name="Lung Adenocarcinoma",
        domain="Condition",
        vocabulary="OncoTree",
        standard=None,
        code=code,
        invalid_reason=None,
    )


def _standard(
    cid: str = "45768916", domain: str = "Condition", standard: str = "S"
) -> ConceptRow:
    return ConceptRow(
        id=cid,
        name="Adenocarcinoma of lung",
        domain=domain,
        vocabulary="SNOMED",
        standard=standard,
        code="254626006",
        invalid_reason=None,
    )


def _provenance() -> Provenance:
    return Provenance(
        run=RunProvenance(
            run_id="run-1",
            target_model_version="v1",
            target_schema_snapshot_hash="h1",
            vocab_release="OMOP_2024",
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
        source_field_ref="source.sample.cancer_type_code",
        source_value_ref="source.sample.cancer_type_code",
        target_property_ref="target.condition_occurrence.condition_concept_id",
        target_field="condition_concept_id",
        domain_constraint_ref="target.condition_occurrence.domain=Condition",
        vocabulary_ref="target.vocabulary.SNOMED",
        vocab_binding="omop.condition_occurrence.condition_concept_id",
        vocab_release="OMOP_2024",
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        run_id="run-1",
        provenance=_provenance(),
    )


# ── stage 1a: candidate generation ─────────────────────────────────────────
def test_candidates_exact_code_match_in_source_vocabulary(policy: ResolverPolicy):
    store = FakeVocabStore(by_code={("OncoTree", "LUAD"): _source()})
    rows = generate_candidates(store, policy, "LUAD")
    assert [r.id for r in rows] == ["777926"]
    # R9: matched in the SOURCE vocabulary, never the target (SNOMED).
    assert store.code_calls == [("OncoTree", "LUAD")]


def test_candidates_unknown_code_is_empty(policy: ResolverPolicy):
    store = FakeVocabStore()
    assert generate_candidates(store, policy, "ZZZZ") == []


# ── stage 2: 'Maps to' standardization ──────────────────────────────────────
def test_standardize_walks_maps_to_with_standard_and_validity(policy: ResolverPolicy):
    target = _standard()
    store = FakeVocabStore(maps_to={"777926": [target]})
    out = standardize(store, policy, _source())
    assert [r.id for r in out] == ["45768916"]
    call = store.maps_to_calls[0]
    assert call["relationship_id"] == "Maps to"
    assert call["standard_flag"] == "S"
    assert call["only_valid"] is True


def test_standardize_keeps_a_candidate_that_is_already_standard(
    policy: ResolverPolicy,
):
    already = _standard(cid="500")  # standard='S', valid, in domain
    store = FakeVocabStore(maps_to={"500": []})
    out = standardize(store, policy, already)
    assert [r.id for r in out] == ["500"]


def test_standardize_omits_standard_flag_when_not_required():
    binding = _binding().model_copy(update={"require_standard": False})
    policy = make_omop_oncotree_condition_policy(binding)
    store = FakeVocabStore(maps_to={"777926": [_standard()]})
    standardize(store, policy, _source())
    assert store.maps_to_calls[0]["standard_flag"] is None


# ── stage 3: the domain gate ────────────────────────────────────────────────
def test_domain_gate_rejects_wrong_domain(policy: ResolverPolicy):
    keep = _standard(cid="1", domain="Condition")
    drop = _standard(cid="2", domain="Measurement")
    out = apply_domain_gate(policy, [keep, drop])
    assert [r.id for r in out] == ["1"]


# ── orchestration: RESOLVED / NO_MAP / tie ──────────────────────────────────
def test_resolve_one_survivor_is_auto_accepted_resolved(policy: ResolverPolicy):
    store = FakeVocabStore(
        by_code={("OncoTree", "LUAD"): _source()},
        maps_to={"777926": [_standard()]},
    )
    resolution = VocabularyResolver(store, policy).resolve("LUAD")
    assert resolution.resolution_status is ResolutionStatus.RESOLVED
    assert resolution.concept_id == 45768916
    assert resolution.status is Status.auto_accepted
    assert resolution.confidence == 1.0
    assert resolution.no_map_reason is None


def test_resolve_zero_survivors_is_first_class_no_map(policy: ResolverPolicy):
    store = FakeVocabStore(
        by_code={("OncoTree", "DEAD"): _source(code="DEAD")},
        maps_to={"777926": []},
    )
    resolution = VocabularyResolver(store, policy).resolve("DEAD")
    assert resolution.resolution_status is ResolutionStatus.NO_MAP
    assert resolution.concept_id is None
    assert resolution.no_map_reason


def test_resolve_unknown_code_is_no_map(policy: ResolverPolicy):
    resolution = VocabularyResolver(FakeVocabStore(), policy).resolve("ZZZZ")
    assert resolution.resolution_status is ResolutionStatus.NO_MAP


def test_resolve_tie_escalates_to_zone2_review(policy: ResolverPolicy):
    store = FakeVocabStore(
        by_code={("OncoTree", "AMB"): _source(code="AMB")},
        maps_to={"777926": [_standard(cid="200"), _standard(cid="100")]},
    )
    resolution = VocabularyResolver(store, policy).resolve("AMB")
    assert resolution.resolution_status is ResolutionStatus.RESOLVED
    assert resolution.status is Status.review_pending  # Zone-2, not auto
    assert resolution.concept_id == 100  # deterministic placeholder pending review
    assert resolution.confidence < 1.0


def test_code_bearing_hot_path_invokes_no_llm(policy: ResolverPolicy):
    def explode(_: Sequence[ConceptRow]) -> ConceptRow | None:
        raise AssertionError("disambiguator must not fire on the single-survivor path")

    store = FakeVocabStore(
        by_code={("OncoTree", "LUAD"): _source()},
        maps_to={"777926": [_standard()]},
    )
    resolver = VocabularyResolver(store, policy, disambiguator=explode)
    assert resolver.resolve("LUAD").status is Status.auto_accepted


# ── emit: assertion + value-mapping ─────────────────────────────────────────
def test_to_assertion_emits_vocab_lookup_on_survivor(policy: ResolverPolicy):
    store = FakeVocabStore(
        by_code={("OncoTree", "LUAD"): _source()},
        maps_to={"777926": [_standard()]},
    )
    resolver = VocabularyResolver(store, policy)
    resolution = resolver.resolve("LUAD")
    assertion = resolver.to_assertion(resolution, _context())
    assert assertion.pattern is MappingPattern.VOCAB_LOOKUP
    assert assertion.status is Status.auto_accepted
    assert isinstance(assertion.payload, VocabLookup)
    assert assertion.payload.resolver_policy_ref == OMOP_ONCOTREE_CONDITION_REF
    assert assertion.source_field_ref == "source.sample.cancer_type_code"
    assert assertion.target_property_ref.endswith("condition_concept_id")


def test_to_assertion_rejects_no_map(policy: ResolverPolicy):
    store = FakeVocabStore(by_code={}, maps_to={})
    resolver = VocabularyResolver(store, policy)
    resolution = resolver.resolve("ZZZZ")
    with pytest.raises(ValueError):
        resolver.to_assertion(resolution, _context())


def test_resolve_and_store_is_sole_writer(policy: ResolverPolicy, tmp_path):
    store = FakeVocabStore(
        by_code={
            ("OncoTree", "LUAD"): _source(),
            ("OncoTree", "DEAD"): _source(code="DEAD", cid="999999"),
        },
        maps_to={"777926": [_standard()]},
    )
    resolver = VocabularyResolver(store, policy)
    import duckdb

    conn = duckdb.connect(str(tmp_path / "vm.duckdb"))
    vm_store = ValueMappingStore(conn)
    written = resolver.resolve_and_store(["LUAD", "DEAD"], vm_store, _context())
    assert {m.normalized_source_value for m in written} == {"LUAD", "DEAD"}

    rows = {m.normalized_source_value: m for m in vm_store.read_all()}
    assert rows["LUAD"].resolution_status is ResolutionStatus.RESOLVED
    assert rows["LUAD"].concept_id == 45768916
    assert rows["DEAD"].resolution_status is ResolutionStatus.NO_MAP
    assert rows["DEAD"].concept_id is None
    vm_store.close()


def test_resolve_and_store_upserts_same_grain(policy: ResolverPolicy, tmp_path):
    store = FakeVocabStore(
        by_code={("OncoTree", "LUAD"): _source()},
        maps_to={"777926": [_standard()]},
    )
    resolver = VocabularyResolver(store, policy)
    import duckdb

    conn = duckdb.connect(str(tmp_path / "vm.duckdb"))
    vm_store = ValueMappingStore(conn)
    resolver.resolve_and_store(["LUAD"], vm_store, _context())
    resolver.resolve_and_store(["LUAD"], vm_store, _context())
    assert vm_store.count() == 1
    vm_store.close()
