"""A relation allowlist scopes warehouse Land before any row is counted (relation-profiling)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.unit.intake.catalog_env import (
    GROUP_VALUES_TABLE,
    GROUPS_TABLE,
    ITEMS_TABLE,
    VALUES_TABLE,
    WAREHOUSE,
    connector_over,
    two_catalogs,
    write_descriptor,
)
from tests.unit.intake.graph_env import GraphRun, graph, routed_client, settings
from tests.unit.intake.warehouse_env import CATALOG, SCHEMA, TABLE, FakeConnector, schema_metadata

pytestmark = pytest.mark.unit

SCOPE_ARTIFACT = Path("r") / "artifacts" / "land_scope.json"


def _land(tmp_path: Path, connector: FakeConnector, relations: tuple[str, ...] | None) -> object:
    from sema.intake.calls.ledger import Ledger
    from sema.intake.land.loaders.warehouse import land_warehouse

    return land_warehouse(
        connector, catalog=CATALOG, schema=SCHEMA, ledger=Ledger(tmp_path).run("r"),
        relations=relations,
    )


def test_allowlist_lands_and_counts_only_the_selected_relation(tmp_path: Path) -> None:
    connector = FakeConnector(schema_metadata(), query=lambda sql: [(3,)])

    land = _land(tmp_path, connector, (TABLE,))

    assert [r.name for r in land.relations] == [TABLE]  # type: ignore[attr-defined]
    assert connector.metadata_reads == [(CATALOG, SCHEMA)]
    assert len(connector.queries) == 1 and "codes" not in connector.queries[0]
    scope = json.loads((tmp_path / SCOPE_ARTIFACT).read_text())
    assert scope == {
        "namespace": f"{CATALOG}.{SCHEMA}",
        "selected": [TABLE],
        "excluded": ["codes"],
        "missing": [],
    }


def test_missing_required_relation_fails_before_any_statement(tmp_path: Path) -> None:
    connector = FakeConnector(schema_metadata(), query=lambda sql: [(3,)])

    with pytest.raises(ValueError, match=re.escape("absent_table")):
        _land(tmp_path, connector, (TABLE, "absent_table"))

    assert connector.queries == []
    assert not (tmp_path / "r" / "sql.jsonl").exists()
    scope = json.loads((tmp_path / SCOPE_ARTIFACT).read_text())
    assert (scope["selected"], scope["missing"]) == ([TABLE], ["absent_table"])


def test_no_allowlist_lands_every_object_and_writes_no_scope(tmp_path: Path) -> None:
    connector = FakeConnector(schema_metadata(), query=lambda sql: [(3,)])

    land = _land(tmp_path, connector, None)

    assert sorted(r.name for r in land.relations) == ["codes", TABLE]  # type: ignore[attr-defined]
    assert len(connector.queries) == 2
    assert not (tmp_path / SCOPE_ARTIFACT).exists()


def test_descriptor_relations_scope_a_warehouse_run(tmp_path: Path) -> None:
    connector = connector_over(*two_catalogs())
    client = routed_client()
    descriptor = write_descriptor(
        tmp_path / "inputs", relations=[ITEMS_TABLE, VALUES_TABLE]
    )

    outcome = graph().discover(
        None, warehouse=connector, catalog=CATALOG, schema=SCHEMA,
        descriptor=descriptor, client=client, **settings(tmp_path),
    )

    run = GraphRun(tmp_path, outcome.run_id, client, outcome)
    land = run.artifact("land_result")
    assert {(r["schema"], r["name"]) for r in land["relations"]} == {
        (WAREHOUSE, ITEMS_TABLE), (WAREHOUSE, VALUES_TABLE),
    }
    excluded = re.compile(rf"\b({GROUPS_TABLE}|{GROUP_VALUES_TABLE})\b")
    assert not [q for q in connector.queries if excluded.search(q)]
    assert sorted(run.artifact("land_scope")["excluded"]) == [GROUP_VALUES_TABLE, GROUPS_TABLE]


def test_an_empty_allowlist_is_refused() -> None:
    from pydantic import ValidationError

    from sema.intake.roles.eligibility_utils import RunDescriptor

    stated = {"id": "d", "identity_system": "s", "release": "1"}

    with pytest.raises(ValidationError, match="relations"):
        RunDescriptor.model_validate({**stated, "relations": []})
    assert RunDescriptor.model_validate(stated).relations is None
