"""US-002 integration: the gold set covers its DECLARED scope, live.

The two tests this file used to hold both read ``~/.sema/poc.duckdb`` at run
time, and ``discover_oncotree_schemas()`` auto-discovered every ``cbioportal_*``
schema — so ingesting a study silently changed the denominator and reddened the
suite. That is exactly what happened when ``msk_impact_50k_2026`` landed.

The contract itself now lives off this machine, in
``tests/unit/test_goldset_snapshot_integrity.py`` (fixture-backed, never reds on
ingest) and ``test_goldset_benchmark_freshness.py`` (may red, distinct message).
What remains here is the part that genuinely needs the live database: that the
declared scope, enumerated for real, is the scope the snapshot froze.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_DUCKDB = Path.home() / ".sema" / "poc.duckdb"


@pytest.fixture()
def live():  # type: ignore[no-untyped-def]
    from sema.eval.goldset_snapshot import load_current_snapshot
    from sema.eval.goldset_source import enumerate_scoped_codes

    if not _DUCKDB.exists():
        pytest.skip("~/.sema/poc.duckdb not present")
    duckdb = pytest.importorskip("duckdb")
    snapshot = load_current_snapshot()
    con = duckdb.connect(str(_DUCKDB), read_only=True)
    try:
        yield snapshot, dict(enumerate_scoped_codes(con, snapshot.header.source_of_truth))
    finally:
        con.close()


def test_every_code_in_the_frozen_tier_is_present_in_the_gold_set(live) -> None:  # type: ignore[no-untyped-def]
    snapshot, observed = live
    gold_codes = {r.oncotree_code for r in snapshot.rows}

    missing = set(snapshot.header.tier_codes) - gold_codes
    assert not missing, f"in-tier codes absent from the gold set: {sorted(missing)}"


def test_a_code_below_the_tier_is_accounted_for_not_a_coverage_failure(live) -> None:  # type: ignore[no-untyped-def]
    """Coverage is a budget against a declared tier, not an absolute over the tail."""
    from sema.eval.mapping_goldset import GoldSet

    snapshot, observed = live
    gold = GoldSet(snapshot.rows)
    out_of_tier = set(observed) - set(snapshot.header.tier_codes)

    assert out_of_tier, "the tail exists; the tier is a budget, not the universe"
    assert out_of_tier.isdisjoint(gold.unlabelled_codes())
    assert gold.coverage_fraction() == pytest.approx(
        gold.labelled_count / len(snapshot.header.tier_codes)
    )


def test_the_declared_scope_still_enumerates_the_frozen_universe(live) -> None:  # type: ignore[no-untyped-def]
    """Membership drift is an error; row_count drift is a reported number."""
    snapshot, observed = live
    frozen = {e.code for e in snapshot.universe}

    assert set(observed) == frozen


def test_row_count_drift_is_reported_never_asserted_equal(live) -> None:  # type: ignore[no-untyped-def]
    from sema.eval.goldset_drift import goldset_drift_report

    snapshot, observed = live
    report = goldset_drift_report(snapshot, observed, snapshot.header.source_of_truth)

    assert report.universe_drift is not None, "drift is a number, not an assertion"
    assert not report.scope_changed
