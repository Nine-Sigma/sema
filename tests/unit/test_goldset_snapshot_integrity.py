"""US-002 / G-07: snapshot integrity — the contract that must NEVER red on ingest.

The shipped artifact is checked against its own declaration, and the declared
scope is enumerated from a DuckDB fixture built in this test. Both previous
integration tests skipped without ``~/.sema/poc.duckdb``, so the whole contract
was unverified anywhere but one developer's machine — and "an unlisted study is
ignored" cannot be staged against a live personal database at all.

The paired **benchmark freshness** contract lives in
``test_goldset_benchmark_freshness.py``. It is allowed to red; this file is not.
"""

from __future__ import annotations

import duckdb
import pytest

from sema.eval.goldset_snapshot import load_current_snapshot, load_snapshot, snapshot_dir
from sema.eval.goldset_source import SourceKind, SourceSpec, enumerate_scoped_codes
from sema.eval.mapping_goldset import GoldSet
from sema.eval.mapping_goldset_utils import GoldLabel, TierState

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def snapshot():  # type: ignore[no-untyped-def]
    return load_current_snapshot()


@pytest.fixture()
def staged_fixture(snapshot):  # type: ignore[no-untyped-def]
    """A DuckDB standing in for the declared scope, plus one UNLISTED study."""
    spec = snapshot.header.source_of_truth
    con = duckdb.connect(":memory:")
    schema, table = spec.table.split(".")
    con.execute(f"CREATE SCHEMA {schema}")
    con.execute(
        f"CREATE TABLE {spec.table} "
        f"({spec.scope_column} VARCHAR, {spec.code_column} VARCHAR)"
    )
    declared = spec.scope_values[0]
    con.execute("CREATE TABLE _counts (code VARCHAR, n BIGINT)")
    con.executemany(
        "INSERT INTO _counts VALUES (?, ?)",
        [[e.code, e.frozen_row_count] for e in snapshot.universe],
    )
    con.execute(
        f"INSERT INTO {spec.table} SELECT ?, c.code FROM _counts c, range(c.n)",
        [declared],
    )
    con.execute(
        f"INSERT INTO {spec.table} VALUES (?, ?)",
        ["cbioportal_a_study_ingested_later", "A_BRAND_NEW_CODE"],
    )
    try:
        yield con
    finally:
        con.close()


def test_the_shipped_snapshot_loads_and_satisfies_its_invariants(snapshot) -> None:  # type: ignore[no-untyped-def]
    """Unique codes, canonical order, disjoint states, digests — asserted on load."""
    assert snapshot.header.snapshot_version
    assert len(snapshot.rows) == len({r.oncotree_code for r in snapshot.rows})


def test_the_frozen_populations_are_declared_not_recomputed(snapshot) -> None:  # type: ignore[no-untyped-def]
    """A frequency tier recomputed at test time re-introduces the drift it fixed."""
    header = snapshot.header
    assert header.tier_codes, "the tier is an explicit code list in the header"
    assert header.tier_achieved_row_share >= header.tier_target_row_share
    assert set(header.tier_codes) <= {e.code for e in snapshot.universe}


def test_every_in_tier_and_challenge_code_has_a_row(snapshot) -> None:  # type: ignore[no-untyped-def]
    rows = snapshot.by_code()
    for code, state in snapshot.states_by_code().items():
        if state in (TierState.IN_TIER, TierState.CHALLENGE):
            assert code in rows, f"{state.value} code {code} has no gold row"


def test_the_four_states_partition_the_universe_manifest(snapshot) -> None:  # type: ignore[no-untyped-def]
    """Storage is not eligibility: all 509 codes are frozen, few are eligible."""
    states = snapshot.states_by_code()
    manifest = {e.code for e in snapshot.universe}
    row_codes = {r.oncotree_code for r in snapshot.rows}

    assert set(states) == manifest | row_codes
    retired = {c for c, s in states.items() if s is TierState.RETIRED}
    assert retired.isdisjoint(manifest), "a code in scope cannot also be retired"
    buckets = [{c for c, s in states.items() if s is state} for state in TierState]
    assert sum(len(b) for b in buckets) == len(states), "states must be disjoint"


def test_out_of_tier_codes_are_accounted_for_not_missing(snapshot) -> None:  # type: ignore[no-untyped-def]
    gold = GoldSet(snapshot.rows)
    accounted = set(gold.out_of_tier_codes()) | set(gold.retired_codes())

    assert accounted.isdisjoint(gold.unlabelled_codes())
    assert gold.total_eligible_codes == len(snapshot.header.tier_codes)


def test_the_oracle_is_still_a_human_gate(snapshot) -> None:  # type: ignore[no-untyped-def]
    """Sema never labels its own gold set — a green suite by autogeneration is worse."""
    for row in snapshot.rows:
        if row.gold_label is GoldLabel.UNLABELLED:
            assert row.gold_concept_id is None
        else:
            assert row.curator and row.evidence, "a label without provenance is not an oracle"


def test_the_declared_scope_reproduces_the_universe_manifest(staged_fixture, snapshot) -> None:  # type: ignore[no-untyped-def]
    observed = dict(enumerate_scoped_codes(staged_fixture, snapshot.header.source_of_truth))

    assert observed == {e.code: e.frozen_row_count for e in snapshot.universe}


def test_ingesting_an_unlisted_study_changes_no_assertion_outcome(staged_fixture, snapshot) -> None:  # type: ignore[no-untyped-def]
    """The whole point: a new study must not red the snapshot-integrity suite."""
    spec = snapshot.header.source_of_truth
    before = dict(enumerate_scoped_codes(staged_fixture, spec))

    staged_fixture.execute(
        f"INSERT INTO {spec.table} SELECT 'cbioportal_yet_another_study', 'FLOOD' "
        "FROM range(5000)"
    )

    assert dict(enumerate_scoped_codes(staged_fixture, spec)) == before
    assert "FLOOD" not in {e.code for e in snapshot.universe}


def test_the_sealed_prior_snapshot_still_loads() -> None:
    """History stays readable: the pre-drift artifact is a snapshot, not a deletion."""
    prior = load_snapshot(snapshot_dir("2026-08-11-raw2study"))

    assert len(prior.rows) == 64
    assert prior.header.source_of_truth.kind is SourceKind.RAW_SAMPLES
    assert prior.header.tier_achieved_row_share == 1.0


def test_a_retired_code_keeps_its_row(snapshot) -> None:  # type: ignore[no-untyped-def]
    """GBM left the staging scope; deleting its row would discard human labour."""
    retired = GoldSet(snapshot.rows).retired_codes()

    assert retired == ["GBM"]
    assert snapshot.by_code()["GBM"].row_count > 0


def test_source_specs_of_both_kinds_describe_the_same_gold_set(snapshot) -> None:  # type: ignore[no-untyped-def]
    """G-01: source_of_truth is executable, so the two shapes are interchangeable."""
    staging = snapshot.header.source_of_truth
    raw = SourceSpec(
        kind=SourceKind.RAW_SAMPLES,
        table="sample",
        code_column="ONCOTREE_CODE",
        scope_values=staging.scope_values,
    )

    assert SourceSpec.from_dict(raw.as_dict()) == raw
    assert staging.kind is SourceKind.STAGING
    assert staging.scope_column == "source_schema"
