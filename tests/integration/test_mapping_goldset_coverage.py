"""US-002 integration: the gold set must cover every observed ONCOTREE_CODE.

Skip-guarded on ``~/.sema/poc.duckdb``. Enumerates distinct ONCOTREE_CODE from
the loaded cbioportal_* studies and asserts the gold-set artifact covers every
observed code (or records it as a known-unlabelled gap), so the gold set cannot
silently drift from the data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_DUCKDB = Path.home() / ".sema" / "poc.duckdb"
_GOLD = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "gold"
    / "oncotree_condition_slice0.jsonl"
)


@pytest.fixture()
def cursor():  # type: ignore[no-untyped-def]
    if not _DUCKDB.exists():
        pytest.skip("~/.sema/poc.duckdb not present")
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(str(_DUCKDB), read_only=True)
    try:
        yield con
    finally:
        con.close()


def test_gold_set_covers_every_observed_code(cursor) -> None:  # type: ignore[no-untyped-def]
    from sema.eval.mapping_goldset import enumerate_distinct_codes, load_gold_set

    observed = {code for code, _ in enumerate_distinct_codes(cursor)}
    assert observed, "no ONCOTREE_CODE observed in poc.duckdb"

    gold_codes = {r.oncotree_code for r in load_gold_set(_GOLD)}
    missing = observed - gold_codes
    assert not missing, f"gold set drifted from data; uncovered codes: {sorted(missing)}"


def test_gold_row_counts_match_live_data(cursor) -> None:  # type: ignore[no-untyped-def]
    from sema.eval.mapping_goldset import enumerate_distinct_codes, load_gold_set

    observed = dict(enumerate_distinct_codes(cursor))
    gold = {r.oncotree_code: r.row_count for r in load_gold_set(_GOLD)}
    for code, rc in observed.items():
        assert gold.get(code) == rc, f"{code}: gold row_count {gold.get(code)} != {rc}"
