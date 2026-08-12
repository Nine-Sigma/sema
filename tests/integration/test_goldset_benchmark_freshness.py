"""US-002 / G-07: benchmark freshness — the contract that MAY red.

Deliberately separated from snapshot integrity. "Ingesting a study never reds the
suite" and "fail when the frozen tier stops representing the data" cannot both
hold unconditionally: a large enough ingest must trip the second. Left implicit,
the implementation would silently encode whichever reading makes the tests pass.

So: ``tests/unit/test_goldset_snapshot_integrity.py`` never reds on ingest, and
this file is free to. Its failure says "benchmark stale — re-snapshot", never
"gold set drifted", and both thresholds are declared in the artifact header
rather than picked at review time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_DUCKDB = Path.home() / ".sema" / "poc.duckdb"


@pytest.fixture()
def observed():  # type: ignore[no-untyped-def]
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


def test_the_frozen_benchmark_still_represents_the_live_data(observed) -> None:  # type: ignore[no-untyped-def]
    from sema.eval.goldset_drift import goldset_drift_report

    snapshot, counts = observed
    report = goldset_drift_report(snapshot, counts, snapshot.header.source_of_truth)

    assert not report.is_stale, (
        f"{report.staleness_reason} This is NOT gold-set corruption: the artifact is "
        f"internally consistent (snapshot {snapshot.header.snapshot_version}); the "
        "benchmark has simply stopped representing the current population."
    )


def test_weight_drift_is_reported_not_asserted_to_zero(observed) -> None:  # type: ignore[no-untyped-def]
    """row_count is a scoring weight. Drift is a number someone reads, not a gate."""
    from sema.eval.goldset_drift import goldset_drift_report

    snapshot, counts = observed
    report = goldset_drift_report(snapshot, counts, snapshot.header.source_of_truth)

    assert report.eligible_drift is not None
    assert report.universe_drift is not None
    assert isinstance(report.per_code, dict)
    assert report.as_dict()["drift"]["eligible_populations"] is not None


def test_a_scope_change_suppresses_drift_rather_than_averaging_it(observed) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from sema.eval.goldset_drift import goldset_drift_report

    snapshot, counts = observed
    widened = replace(
        snapshot.header.source_of_truth,
        scope_values=snapshot.header.source_of_truth.scope_values + ("cbioportal_elsewhere",),
    )

    report = goldset_drift_report(snapshot, counts, widened)

    assert report.scope_changed
    assert report.universe_drift is None, "per-code drift across scopes is meaningless"
