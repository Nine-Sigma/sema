"""A warehouse schema is a run source (ingestion-cli: "a source path or warehouse schema").

Pinned graph surface for a warehouse run (a decision recorded with the orchestrator):
``graph.discover(None, warehouse=<Connector>, catalog=<name>, schema=<name>, ...)`` lands
the schema through ``loaders.warehouse.land_warehouse`` and profiles through the connector.
The connector is a live object, so like ``client`` and ``embedder`` it is passed to every
later call on the run (``resume``, ``approve``, ``approve_values``, ``reject``) as
``warehouse=``; ``catalog`` and ``schema`` are state scalars, so later calls do not repeat
them. A resume re-executes Land through the connector it is given.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.models.schema_metadata import SchemaMetadata
from tests.unit.intake.catalog_env import (
    ITEMS_TABLE,
    VALUES_TABLE,
    WAREHOUSE,
    connector_over,
    warehouse_catalog,
    write_descriptor,
)
from tests.unit.intake.graph_env import (
    GraphRun,
    ProcessExit,
    graph,
    routed_client,
    settings,
)
from tests.unit.intake.warehouse_env import CATALOG, SCHEMA

pytestmark = pytest.mark.unit

EXIT_RUN_ID = "warehouse-run-exits-in-structure"
STAGES_MODULE = "sema.intake.run.graph_stages"


@pytest.fixture
def warehouse() -> Iterator[tuple[duckdb.DuckDBPyConnection, SchemaMetadata]]:
    con, metadata = warehouse_catalog(status=False)
    try:
        yield con, metadata
    finally:
        con.close()


def _dies_once_in_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process dies the first time Structure runs; later processes run it normally."""
    module = importlib.import_module(STAGES_MODULE)
    real, calls = module.run_structure, []

    def dying(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise ProcessExit("structure")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "run_structure", dying)


def test_warehouse_run_lands_through_the_connector_and_stops_at_review(
    tmp_path: Path, warehouse: tuple[duckdb.DuckDBPyConnection, SchemaMetadata]
) -> None:
    connector = connector_over(*warehouse)
    client = routed_client()

    outcome = graph().discover(
        None, warehouse=connector, catalog=CATALOG, schema=SCHEMA,
        descriptor=write_descriptor(tmp_path / "inputs"), client=client,
        **settings(tmp_path),
    )

    run = GraphRun(tmp_path, outcome.run_id, client, outcome)
    assert outcome.node == "review"
    assert connector.metadata_reads == [(CATALOG, SCHEMA)]
    land = run.artifact("land_result")
    assert land["medium"] == "warehouse"
    assert {(r["schema"], r["name"]) for r in land["relations"]} == {
        (WAREHOUSE, ITEMS_TABLE), (WAREHOUSE, VALUES_TABLE),
    }
    assert connector.queries, "profiling never reached the warehouse"
    row = run.row()
    assert (row.source, row.stage, row.status) == (WAREHOUSE, "review", "waiting_review")


def test_warehouse_run_resumes_by_reexecuting_land_through_the_connector(
    tmp_path: Path,
    warehouse: tuple[duckdb.DuckDBPyConnection, SchemaMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dies_once_in_structure(monkeypatch)
    first = connector_over(*warehouse)
    with pytest.raises(ProcessExit):
        graph().discover(
            None, warehouse=first, catalog=CATALOG, schema=SCHEMA,
            descriptor=write_descriptor(tmp_path / "inputs"), client=routed_client(),
            run_id=EXIT_RUN_ID, **settings(tmp_path),
        )
    run = GraphRun(tmp_path, EXIT_RUN_ID, routed_client())
    assert run.nodes()[-1] == "structure"
    assert first.metadata_reads == [(CATALOG, SCHEMA)]
    mark, second = run.mark(), connector_over(*warehouse)

    run.outcome = graph().resume(
        EXIT_RUN_ID, warehouse=second, client=run.client, **settings(tmp_path)
    )

    assert run.outcome.node == "review"
    (resumed,) = run.events("resume")
    assert resumed["stage"] == "structure"
    assert run.nodes(mark)[0] == "structure"
    assert "land" not in run.nodes(mark)
    assert second.metadata_reads == [(CATALOG, SCHEMA)]
    assert first.metadata_reads == [(CATALOG, SCHEMA)]
    assert second.queries, "the resumed run profiled through a connector it was not given"
