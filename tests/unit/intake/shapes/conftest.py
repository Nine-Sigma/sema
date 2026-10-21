"""Session fixtures: every source the shape templates are tested over, landed once."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import duckdb
import pytest

from sema.ingest.duckdb_staging import Staging
from tests.unit.intake.shapes.shapes_env import (
    A2_WORKBOOK,
    A3_DB,
    ACRIS,
    OMOP_CSV,
    Source,
    staging_source,
    warehouse_source,
)


def _staging(factory: pytest.TempPathFactory, name: str) -> tuple[Staging, Path]:
    root = factory.mktemp(name)
    return Staging(str(root / "staging.duckdb"), schemas=()), root / "ledger" / "run"


@pytest.fixture(scope="session")
def a2(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    from sema.intake.loaders.tabular import land_tabular

    staging, ledger = _staging(tmp_path_factory, "a2")
    yield staging_source(staging, land_tabular(A2_WORKBOOK, staging, schema="land_a2"), ledger)
    staging.close()


@pytest.fixture(scope="session")
def a3(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    from sema.intake.loaders.semsql import land_semsql

    staging, ledger = _staging(tmp_path_factory, "a3")
    yield staging_source(staging, land_semsql(A3_DB, staging, schema="land_a3"), ledger)
    staging.close()


@pytest.fixture(scope="session")
def acris(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    from sema.intake.loaders.tabular import land_tabular

    staging, ledger = _staging(tmp_path_factory, "acris")
    yield staging_source(staging, land_tabular(ACRIS, staging, schema="land_acris"), ledger)
    staging.close()


@pytest.fixture(scope="session")
def omop(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    from sema.intake.loaders.tabular import land_tabular

    staging, ledger = _staging(tmp_path_factory, "omop")
    yield staging_source(staging, land_tabular(OMOP_CSV, staging, schema="land_omop"), ledger)
    staging.close()


@pytest.fixture(scope="session")
def athena(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    from sema.intake.loaders.athena import land_athena_bundle
    from tests.support.athena_bundle import make_minimal_bundle

    staging, ledger = _staging(tmp_path_factory, "athena")
    bundle = ledger.parents[1] / "bundle"
    make_minimal_bundle(bundle)
    yield staging_source(staging, land_athena_bundle(bundle, staging, schema="land_ath"), ledger)
    staging.close()


@pytest.fixture(scope="session")
def syn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    from tests.unit.binding.fixtures.concept_library import build_library

    con = build_library()
    yield warehouse_source(con, "lib", tmp_path_factory.mktemp("syn") / "ledger" / "run")
    con.close()


@pytest.fixture(scope="session")
def v2(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    from tests.unit.binding.fixtures.concept_library_v2 import build_library_v2

    con = build_library_v2()
    yield warehouse_source(con, "land", tmp_path_factory.mktemp("v2") / "ledger" / "run")
    con.close()


@pytest.fixture(scope="session")
def cycle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA loop")
    con.execute("CREATE TABLE loop.cycle (child VARCHAR, parent VARCHAR)")
    con.execute("INSERT INTO loop.cycle VALUES ('A', 'B'), ('B', 'A')")
    yield warehouse_source(con, "loop", tmp_path_factory.mktemp("cycle") / "ledger" / "run")
    con.close()


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[Callable[..., Source]]:
    """Build a source over in-memory tables: ``memory(schema, *statements)``."""
    connections: list[duckdb.DuckDBPyConnection] = []

    def build(schema: str, *statements: str) -> Source:
        con = duckdb.connect(":memory:")
        connections.append(con)
        con.execute(f"CREATE SCHEMA {schema}")
        for statement in statements:
            con.execute(statement)
        return warehouse_source(con, schema, tmp_path / schema / "ledger" / "run")

    yield build
    for con in connections:
        con.close()
