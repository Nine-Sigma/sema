"""The structure stage's join checks grow at most linearly with a relation's width.

A relation of N unique, non-null floating-point columns holds no join key: a float is
never an identifier, and a parent column whose every value is distinct cannot be a
parent link. Structure still checks every ordered column pair as an id and a parent,
N(N-1) joins, so a wide relation stalls the stage. Counts are compared at two widths;
no wall-clock time is measured.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from sema.intake.ledger import SQL_FILE
from tests.unit.intake.shapes.shapes_env import warehouse_source
from tests.unit.intake.structure_env import fresh_tools, run

pytestmark = pytest.mark.unit

SCHEMA = "sw"
ROWS = 20
WIDTH = 10
# Constant per-relation work (profiles, ignore checks) may differ between widths.
SLACK = 4


def unique_floats(width: int) -> str:
    """Each column holds distinct values in its own range, so no two columns share one."""
    cells = ", ".join(f"r + {i} * 1000 + 0.25 AS col_{i:04d}" for i in range(width))
    return f"CREATE TABLE {SCHEMA}.wide AS SELECT {cells} FROM range({ROWS}) t(r)"


def statements(ledger_dir: Path) -> int:
    path = ledger_dir / SQL_FILE
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return sum(1 for line in lines if json.loads(line)["statement"])


def counted(width: int, root: Path) -> tuple[int, int]:
    """Join checks and SQL statements of one structure run over a relation of ``width``."""
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE SCHEMA {SCHEMA}")
    con.execute(unique_floats(width))
    source = warehouse_source(con, SCHEMA, root / "landing")
    tools = fresh_tools(source, root / "structure")
    with patch.object(tools, "check_join", wraps=tools.check_join) as joins:
        run(source, tools)
    con.close()
    return joins.call_count, statements(tools.ledger.directory)


@pytest.fixture(scope="module")
def widths(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[int, tuple[int, int]]]:
    root = tmp_path_factory.mktemp("structure_wide")
    yield {width: counted(width, root / str(width)) for width in (WIDTH, 2 * WIDTH)}


def test_join_checks_and_statements_grow_at_most_linearly_with_width(
    widths: dict[int, tuple[int, int]],
) -> None:
    (narrow_joins, narrow_sql), (broad_joins, broad_sql) = widths[WIDTH], widths[2 * WIDTH]

    assert broad_joins <= 2 * narrow_joins + SLACK, (
        f"{narrow_joins} join checks at {WIDTH} columns, {broad_joins} at {2 * WIDTH}"
    )
    assert broad_sql <= 2 * narrow_sql + SLACK, (
        f"{narrow_sql} statements at {WIDTH} columns, {broad_sql} at {2 * WIDTH}"
    )
