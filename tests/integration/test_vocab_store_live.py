"""US-003 live test: VocabStore against the real ~/.sema/poc.duckdb mirror.

Skip-guarded on the DuckDB file. Confirms a real OncoTree code resolves, that
'Maps to' standardization yields >=1 standard SNOMED Condition concept, and
records query latency on the 54.9M-row concept_relationship table.

The relationship name ('Maps to') and standard flag ('S') here stand in for the
US-004 ResolverPolicy values; in production the resolver (US-006) supplies them
from the policy, never the store.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sema.resolve.policies.omop import (
    OMOP_VOCAB_SCHEMA,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.vocab_store import open_duckdb_vocab_store
from tests.integration._omop_binding import build_condition_binding

pytestmark = pytest.mark.integration

_DB = Path.home() / ".sema" / "poc.duckdb"


@pytest.fixture()
def policy():  # type: ignore[no-untyped-def]
    return make_omop_oncotree_condition_policy(build_condition_binding())


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_concept_by_code_resolves_real_oncotree_code(policy) -> None:  # type: ignore[no-untyped-def]
    store = open_duckdb_vocab_store(str(_DB), schema=OMOP_VOCAB_SCHEMA)
    try:
        row = store.concept_by_code(policy.source_vocabulary, "LUAD")
    finally:
        store.close()
    assert row is not None
    assert row.code == "LUAD"
    assert row.domain == policy.target_domain


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_maps_to_yields_standard_snomed_condition(policy) -> None:  # type: ignore[no-untyped-def]
    store = open_duckdb_vocab_store(str(_DB), schema=OMOP_VOCAB_SCHEMA)
    try:
        source = store.concept_by_code(policy.source_vocabulary, "LUAD")
        assert source is not None
        start = time.perf_counter()
        targets = store.maps_to_targets(
            source.id,
            relationship_id=policy.maps_to_relationship,
            standard_flag=policy.standard_flag,
            only_valid=True,
        )
        elapsed = time.perf_counter() - start
    finally:
        store.close()

    assert targets, "expected >=1 'Maps to' standard target"
    assert all(t.standard == policy.standard_flag for t in targets)
    assert all(t.domain == policy.target_domain for t in targets)
    assert any(t.vocabulary == "SNOMED" for t in targets)
    # Bounded latency on the 54.9M concept_relationship table (recorded).
    assert elapsed < 30.0, f"maps_to latency {elapsed:.2f}s exceeded bound"
