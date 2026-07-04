"""US-004: per-vocabulary resolver policy object.

The ResolverPolicy captures SOURCE-side resolution rules (which source
vocabulary a code is matched in, which relationship standardizes to the
target, which flag marks a standard concept) while reading the TARGET-side
obligation (domain, require_standard, allow_zero_default) from the loaded
:class:`VocabularyBindingDecl` — never duplicating those fields.

R9: ``resolver_policy_ref`` names the SOURCE vocabulary (OncoTree); the
binding's singular ``vocabulary`` is the TARGET governance scope
(OMOP-Condition).
"""

from __future__ import annotations

import pytest

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.refs import (
    TargetEntityRef,
    VocabularyRef,
    VocabularySource,
)
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.policies import UnknownResolverPolicyError, resolve_policy
from sema.resolve.policies.omop import (
    OMOP_ONCOTREE_CONDITION_REF,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.policy import Candidate, ResolverPolicy

pytestmark = pytest.mark.unit

_TARGET_VOCABULARY = "OMOP-Condition"


def _binding(
    *,
    domain: str | None = "Condition",
    require_standard: bool = True,
    allow_zero_default: bool = False,
    resolver_policy_ref: str | None = OMOP_ONCOTREE_CONDITION_REF,
) -> VocabularyBindingDecl:
    entity_ref = TargetEntityRef(
        target_model_id="omop_condition_slice0",
        qualified_name="omop.condition_occurrence",
        kind=TargetArtifactKind.TABLE_ROW,
    )
    return VocabularyBindingDecl(
        entity_ref=entity_ref,
        property_name="condition_concept_id",
        vocabulary=VocabularyRef(
            name=_TARGET_VOCABULARY, source=VocabularySource.EXTERNAL
        ),
        domain=domain,
        require_standard=require_standard,
        allow_zero_default=allow_zero_default,
        standard_domain_governed=True,
        resolver_policy_ref=resolver_policy_ref,
    )


def _policy(**kwargs: object) -> ResolverPolicy:
    return make_omop_oncotree_condition_policy(_binding(**kwargs))


def test_exposes_source_side_literals() -> None:
    policy = _policy()
    assert policy.source_vocabulary == "OncoTree"
    assert policy.maps_to_relationship == "Maps to"
    assert policy.standard_flag == "S"


def test_reads_target_obligation_from_binding_not_duplicated() -> None:
    policy = _policy(domain="Condition", require_standard=True)
    assert policy.target_domain == "Condition"
    assert policy.require_standard is True
    # Change the binding -> the policy reflects it (no duplicated field).
    other = _policy(domain="Drug", require_standard=False)
    assert other.target_domain == "Drug"
    assert other.require_standard is False


def test_allow_zero_default_is_per_obligation_and_off_by_default() -> None:
    assert _policy().allow_zero_default is False
    assert _policy(allow_zero_default=True).allow_zero_default is True


def test_validity_predicate_accepts_standard_valid_in_domain() -> None:
    policy = _policy()
    candidate = Candidate(standard_value="S", is_invalid=False, domain="Condition")
    assert policy.is_standard(candidate) is True
    assert policy.is_valid(candidate) is True
    assert policy.in_target_domain(candidate) is True
    assert policy.accepts(candidate) is True


def test_validity_predicate_rejects_invalid_concept() -> None:
    policy = _policy()
    candidate = Candidate(standard_value="S", is_invalid=True, domain="Condition")
    assert policy.is_valid(candidate) is False
    assert policy.accepts(candidate) is False


def test_validity_predicate_rejects_non_standard_when_required() -> None:
    policy = _policy(require_standard=True)
    candidate = Candidate(standard_value=None, is_invalid=False, domain="Condition")
    assert policy.is_standard(candidate) is False
    assert policy.accepts(candidate) is False


def test_validity_predicate_allows_non_standard_when_not_required() -> None:
    policy = _policy(require_standard=False)
    candidate = Candidate(standard_value=None, is_invalid=False, domain="Condition")
    assert policy.accepts(candidate) is True


def test_domain_gate_rejects_other_domain() -> None:
    policy = _policy(domain="Condition")
    candidate = Candidate(standard_value="S", is_invalid=False, domain="Drug")
    assert policy.in_target_domain(candidate) is False
    assert policy.accepts(candidate) is False


def test_domain_gate_open_when_binding_has_no_domain() -> None:
    policy = _policy(domain=None)
    candidate = Candidate(standard_value="S", is_invalid=False, domain="Anything")
    assert policy.in_target_domain(candidate) is True


def test_r9_source_vocabulary_distinct_from_target() -> None:
    binding = _binding()
    policy = make_omop_oncotree_condition_policy(binding)
    # The resolver matches the source code in the SOURCE vocabulary...
    vocab, code = policy.candidate_lookup("LUAD")
    assert vocab == "OncoTree"
    assert code == "LUAD"
    # ...which is distinct from the binding's TARGET governance scope.
    assert binding.vocabulary.name == _TARGET_VOCABULARY
    assert policy.source_vocabulary != binding.vocabulary.name


def test_resolve_policy_dispatches_on_resolver_policy_ref() -> None:
    binding = _binding()
    policy = resolve_policy(binding)
    assert isinstance(policy, ResolverPolicy)
    assert policy.source_vocabulary == "OncoTree"
    assert policy.binding is binding


def test_resolve_policy_unknown_ref_raises() -> None:
    binding = _binding(resolver_policy_ref="omop.nonexistent")
    with pytest.raises(UnknownResolverPolicyError):
        resolve_policy(binding)


def test_resolve_policy_missing_ref_raises() -> None:
    binding = _binding(resolver_policy_ref=None)
    with pytest.raises(UnknownResolverPolicyError):
        resolve_policy(binding)
