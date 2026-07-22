"""F1: deterministic contract-conformance gate over resolved decisions.

The gate re-verifies every RESOLVED concept_id against the authoritative vocab
(standard + valid + target domain) — the self-consistency check that makes
``sema fit --strict`` meaningful WITHOUT gold labels. NO_MAP rows are counted
but never checked for a target concept.
"""

from __future__ import annotations

import pytest

from sema.eval.conformance import assert_contract_conformance
from sema.models.planner._enums import TargetArtifactKind
from sema.models.planner.lifecycle import Status
from sema.models.target.refs import TargetEntityRef, VocabularyRef, VocabularySource
from sema.models.target.vocab_binding import VocabularyBindingDecl
from showcase.cbioportal_to_omop.omop_policy import (
    OMOP_ONCOTREE_CONDITION_REF,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping
from sema.resolve.vocab_store_utils import ConceptRow

pytestmark = pytest.mark.unit


class FakeConceptLookup:
    """Stands in for VocabStore.concepts_by_ids; preserves missing ids as None."""

    def __init__(self, rows: dict[str, ConceptRow]) -> None:
        self._rows = rows

    def concepts_by_ids(self, ids: list[str]) -> dict[str, ConceptRow | None]:
        return {i: self._rows.get(i) for i in ids}


def _policy():
    binding = VocabularyBindingDecl(
        entity_ref=TargetEntityRef(
            target_model_id="omop_condition_slice0",
            qualified_name="omop.condition_occurrence",
            kind=TargetArtifactKind.TABLE_ROW,
        ),
        property_name="condition_concept_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
        domain="Condition",
        require_standard=True,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
    )
    return make_omop_oncotree_condition_policy(binding)


def _mapping(value: str, concept_id: int | None, status: ResolutionStatus) -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=value,
        target_property_ref="target.stage.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="binding.condition_concept_id",
        concept_id=concept_id,
        vocab_release="omop-vocab-2024",
        valid_start=None,
        valid_end=None,
        resolution_status=status,
        no_map_reason=None if status is ResolutionStatus.RESOLVED else "no curated crosswalk",
        confidence=1.0,
        status=Status.auto_accepted,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        run_id="run-1",
    )


def _concept(cid: str, *, standard: str | None = "S", domain: str | None = "Condition",
             invalid: str | None = None) -> ConceptRow:
    return ConceptRow(
        id=cid, name="x", domain=domain, vocabulary="SNOMED",
        standard=standard, code="c", invalid_reason=invalid,
    )


def test_conformant_run_passes_and_counts_no_map():
    mappings = [
        _mapping("LUAD", 100, ResolutionStatus.RESOLVED),
        _mapping("PANEC", None, ResolutionStatus.NO_MAP),
    ]
    lookup = FakeConceptLookup({"100": _concept("100")})
    report = assert_contract_conformance(mappings, lookup, _policy())
    assert report.passed
    assert report.violations == ()
    assert report.resolved_count == 1
    assert report.no_map_count == 1


def test_missing_concept_is_violation():
    mappings = [_mapping("LUAD", 999, ResolutionStatus.RESOLVED)]
    report = assert_contract_conformance(mappings, FakeConceptLookup({}), _policy())
    assert not report.passed
    assert len(report.violations) == 1
    assert "absent" in report.violations[0].reason


def test_non_standard_concept_is_violation():
    lookup = FakeConceptLookup({"100": _concept("100", standard="C")})
    report = assert_contract_conformance(
        [_mapping("LUAD", 100, ResolutionStatus.RESOLVED)], lookup, _policy()
    )
    assert not report.passed
    assert "standard" in report.violations[0].reason


def test_invalid_concept_is_violation():
    lookup = FakeConceptLookup({"100": _concept("100", invalid="D")})
    report = assert_contract_conformance(
        [_mapping("LUAD", 100, ResolutionStatus.RESOLVED)], lookup, _policy()
    )
    assert not report.passed
    assert "invalid" in report.violations[0].reason


def test_wrong_domain_concept_is_violation():
    lookup = FakeConceptLookup({"100": _concept("100", domain="Measurement")})
    report = assert_contract_conformance(
        [_mapping("LUAD", 100, ResolutionStatus.RESOLVED)], lookup, _policy()
    )
    assert not report.passed
    assert "domain" in report.violations[0].reason


def test_no_map_only_run_is_conformant():
    report = assert_contract_conformance(
        [_mapping("PANEC", None, ResolutionStatus.NO_MAP)], FakeConceptLookup({}), _policy()
    )
    assert report.passed
    assert report.resolved_count == 0
    assert report.no_map_count == 1
