"""`sema target discover --warehouse <catalog>.<schema>` (ingestion-cli: "a source path or
warehouse schema").

Pinned surface: ``--warehouse`` names one schema as ``<catalog>.<schema>`` and is exclusive
with SOURCE; both or neither is a usage error (exit 2) whose error line names SOURCE and
``--warehouse``, raised before any run is registered or connector built. The connector comes
from the existing factory (``sema.pipeline.build.DatabricksConnectorFactory``, as
``sema target project`` builds it), Land reads the schema through ``land_warehouse``, the
connector profiles the run, and the command stops at Review printing the run id, the
open-entry count, and the run metrics. The registry records the source as
``<catalog>.<schema>``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sema.intake.calls.registry import Registry
from sema.pipeline.build import DatabricksConnectorFactory
from tests.unit.cli_intake_env import Cli, intake_cli, open_count
from tests.unit.intake.catalog_env import (
    ITEMS_TABLE,
    VALUES_TABLE,
    WAREHOUSE,
    connector_over,
    warehouse_catalog,
    write_catalog,
    write_descriptor,
)
from tests.unit.intake.graph_env import REGISTRY_FILE
from tests.unit.intake.warehouse_env import CATALOG, SCHEMA, FakeConnector

pytestmark = pytest.mark.unit


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    return intake_cli(tmp_path / "runs", monkeypatch)


@dataclass
class Factory:
    """The fake connector the patched factory returns, and each connector it created."""

    connector: FakeConnector
    created: list[FakeConnector] = field(default_factory=list)


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[Factory]:
    con, metadata = warehouse_catalog(status=False)
    patched = Factory(connector_over(con, metadata))

    def create(_factory: DatabricksConnectorFactory) -> FakeConnector:
        patched.created.append(patched.connector)
        return patched.connector

    monkeypatch.setattr(DatabricksConnectorFactory, "create", create)
    try:
        yield patched
    finally:
        con.close()


def _error_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.startswith("Error:")]
    assert lines, output
    return lines[-1]


def test_discover_a_warehouse_schema_stops_at_review_and_prints_the_summary(
    cli: Cli, factory: Factory
) -> None:
    descriptor = write_descriptor(cli.root.parent / "inputs")

    printed = cli.ok("discover", "--warehouse", WAREHOUSE, "--descriptor", descriptor)

    run = cli.graph_run(printed["run_id"])
    assert (printed["stopped_at"], printed["status"]) == ("review", "waiting_review")
    assert printed["run_metrics"] == run.artifact("run_metrics")
    assert printed["open_entries"] == open_count(run)
    assert factory.created == [factory.connector]
    assert factory.connector.metadata_reads == [(CATALOG, SCHEMA)]
    assert factory.connector.queries, "profiling never reached the warehouse"
    land = run.artifact("land_result")
    assert land["medium"] == "warehouse"
    assert {r["name"] for r in land["relations"]} == {ITEMS_TABLE, VALUES_TABLE}
    (row,) = Registry(cli.root / REGISTRY_FILE).list()
    assert (row.run_id, row.source) == (printed["run_id"], WAREHOUSE)
    assert (row.stage, row.status) == ("review", "waiting_review")


def test_discover_with_a_source_and_a_warehouse_is_a_usage_error(
    cli: Cli, factory: Factory
) -> None:
    source, descriptor = write_catalog(cli.root.parent / "inputs")

    result = cli.run(
        "discover", source, "--warehouse", WAREHOUSE, "--descriptor", descriptor
    )

    assert result.exit_code == 2
    error = _error_line(result.output)
    assert "SOURCE" in error and "--warehouse" in error
    assert factory.created == []
    assert Registry(cli.root / REGISTRY_FILE).list() == ()
    assert cli.client.invoke.call_count == 0


def test_discover_with_neither_a_source_nor_a_warehouse_is_a_usage_error(
    cli: Cli, factory: Factory
) -> None:
    descriptor = write_descriptor(cli.root.parent / "inputs")

    result = cli.run("discover", "--descriptor", descriptor)

    assert result.exit_code == 2
    error = _error_line(result.output)
    assert "SOURCE" in error and "--warehouse" in error
    assert factory.created == []
    assert Registry(cli.root / REGISTRY_FILE).list() == ()


def test_approve_on_a_warehouse_run_lands_again_through_a_new_connector(
    cli: Cli, factory: Factory
) -> None:
    descriptor = write_descriptor(cli.root.parent / "inputs")
    printed = cli.ok("discover", "--warehouse", WAREHOUSE, "--descriptor", descriptor)
    run = cli.graph_run(printed["run_id"])
    assert printed["stopped_at"] == "review"
    mark = run.mark()

    approved = cli.approve(run)

    assert (approved["stopped_at"], approved["status"]) == (None, "approved")
    assert factory.created == [factory.connector, factory.connector]
    assert factory.connector.metadata_reads == [(CATALOG, SCHEMA)] * 2
    assert "import" in run.nodes(mark)
    assert (run.ledger_dir / "import" / "manifest.yaml").is_file()


def test_discover_a_warehouse_without_a_descriptor_is_a_usage_error(
    cli: Cli, factory: Factory
) -> None:
    result = cli.run("discover", "--warehouse", WAREHOUSE)

    assert result.exit_code == 2
    assert "--descriptor" in _error_line(result.output)
    assert factory.created == []
    assert Registry(cli.root / REGISTRY_FILE).list() == ()
    assert cli.client.invoke.call_count == 0


@pytest.mark.parametrize("value", ["lib", "memory.lib.extra", ".lib"])
def test_a_warehouse_value_not_naming_catalog_and_schema_is_a_usage_error(
    cli: Cli, factory: Factory, value: str
) -> None:
    descriptor = write_descriptor(cli.root.parent / "inputs")

    result = cli.run("discover", "--warehouse", value, "--descriptor", descriptor)

    assert result.exit_code == 2
    assert "<catalog>.<schema>" in _error_line(result.output)
    assert factory.created == []
    assert Registry(cli.root / REGISTRY_FILE).list() == ()
