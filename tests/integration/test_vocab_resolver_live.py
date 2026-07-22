"""US-006 live test: the §4 resolver against the real ~/.sema/poc.duckdb mirror.

Skip-guarded on the DuckDB file. Confirms a real OncoTree code resolves to a
standard SNOMED Condition concept, at least one known dead-end yields NO_MAP,
and records OncoTree→SNOMED path coverage over the gold-set codes (coverage is
*reported* here; US-012 asserts the threshold).

The gold set is intentionally UNLABELLED at this stage (the human oracle gate),
so this test does not grade against ``gold_concept_id`` — it smoke-checks the
deterministic walk and reports coverage, never self-certifies precision.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.eval.mapping_goldset import load_gold_set
from sema.resolve.engine import VocabularyResolver
from showcase.cbioportal_to_omop.omop_policy import (
    OMOP_VOCAB_SCHEMA,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.value_mapping_store_utils import ResolutionStatus
from sema.resolve.vocab_store import open_duckdb_vocab_store
from tests.integration._omop_binding import build_condition_binding

pytestmark = pytest.mark.integration

_DB = Path.home() / ".sema" / "poc.duckdb"
_GOLD = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "gold"
    / "oncotree_condition_slice0.jsonl"
)


@pytest.fixture()
def resolver() -> VocabularyResolver:
    policy = make_omop_oncotree_condition_policy(build_condition_binding())
    store = open_duckdb_vocab_store(str(_DB), schema=OMOP_VOCAB_SCHEMA)
    return VocabularyResolver(store, policy)


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_luad_resolves_to_standard_snomed_condition(
    resolver: VocabularyResolver,
) -> None:
    resolution = resolver.resolve("LUAD")
    assert resolution.resolution_status is ResolutionStatus.RESOLVED
    assert resolution.concept_id is not None
    # The standard SNOMED Condition target verified live in US-003.
    assert resolution.concept_id == 45768916


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_unknown_code_yields_no_map(resolver: VocabularyResolver) -> None:
    resolution = resolver.resolve("NOT_A_REAL_ONCOTREE_CODE")
    assert resolution.resolution_status is ResolutionStatus.NO_MAP
    assert resolution.concept_id is None


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_path_coverage_over_gold_codes_recorded(
    resolver: VocabularyResolver,
) -> None:
    gold_rows = load_gold_set(_GOLD)
    codes = sorted({row.oncotree_code for row in gold_rows})
    resolved = 0
    no_map = 0
    for code in codes:
        outcome = resolver.resolve(code)
        if outcome.resolution_status is ResolutionStatus.RESOLVED:
            resolved += 1
        else:
            no_map += 1
    coverage = resolved / len(codes) if codes else 0.0
    # Recorded, not asserted (US-012 owns the threshold); at least one of each
    # outcome class is expected over the real OncoTree slice.
    print(
        f"OncoTree→SNOMED path coverage: {resolved}/{len(codes)} "
        f"({coverage:.1%}); NO_MAP={no_map}"
    )
    assert resolved >= 1
