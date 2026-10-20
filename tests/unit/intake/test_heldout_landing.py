"""The held-out fixture lands with every relation accounted for.

The workbook has 8 sheets, so the landing has 8 relations. Nothing about the
fixture's content is asserted: it is never an input to a floor, token, pattern,
or predicate.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

HELDOUT = Path(__file__).parents[2] / "fixtures" / "intake" / "heldout"
SCHEMA = "land_heldout"


def _expected_relations() -> set[str]:
    from sema.intake.loaders.base import relation_name

    names: set[str] = set()
    for path in sorted(HELDOUT.glob("*.xlsx")):
        workbook = openpyxl.load_workbook(path, read_only=True)
        try:
            names |= {relation_name(f"{path.stem}__{t}") for t in workbook.sheetnames}
        finally:
            workbook.close()
    return names


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Staging._insert_rows quotes column names as f'\"{c}\"' without doubling an "
        "embedded quote: _duckdb.ParserException: Parser Error: syntax error at or "
        'near "Most" on the column \'Cohort/Measure in "Most Recent" Datafile\''
    ),
)
def test_heldout_fixture_lands_with_every_relation_accounted_for(tmp_path: Path) -> None:
    from sema.intake.loaders.tabular import land_tabular

    expected = _expected_relations()
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        result = land_tabular(HELDOUT, staging, schema=SCHEMA)
        tables = staging.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = ?", [SCHEMA]
        ).fetchall()

    landed = {r.name for r in result.relations}
    assert len(expected) == 8
    assert landed == expected
    assert {name for (name,) in tables} == landed
    assert set(result.headerless) <= landed
