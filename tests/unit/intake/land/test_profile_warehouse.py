"""Warehouse catalog facts come from the landed relation; one evidence tier (Decision 8)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.binding.spec_sections import RelationRef
from tests.unit.binding.fixtures.concept_library import query_fn
from tests.unit.intake.profile_env import Env, column_of, make_tools, open_env, value_bearing
from tests.unit.intake.warehouse_env import (
    CATALOG,
    CODES_ROWS,
    CONCEPT_COLUMNS,
    SCHEMA,
    TABLE,
    FakeConnector,
    create_codes,
    schema_metadata,
)

pytestmark = pytest.mark.unit

WAREHOUSE_CONCEPT = RelationRef(provider="warehouse", name=f"{CATALOG}.{SCHEMA}.{TABLE}")
CONCEPT_ROWS = 7
AGGREGATE = re.compile(r"^SELECT COUNT\(\*\)(, COUNT\(.+\))* FROM \S+$", re.DOTALL)
CATALOG_STATEMENT = re.compile(r"\bDESCRIBE\b|\bSHOW\b|\binformation_schema\b", re.IGNORECASE)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    yield from open_env(tmp_path, monkeypatch)


def _landed_tools(env: Env) -> tuple[Any, FakeConnector]:
    from sema.intake.calls.ledger import Ledger
    from sema.intake.land.loaders.warehouse import land_warehouse

    create_codes(env.warehouse)
    connector = FakeConnector(schema_metadata(), query=query_fn(env.warehouse))
    land_ledger = Ledger(env.ledger_dir.parent).run("land")
    land = land_warehouse(connector, catalog=CATALOG, schema=SCHEMA, ledger=land_ledger)
    return make_tools(env, land=land), connector


def test_warehouse_catalog_comes_from_the_landed_relation(env: Env) -> None:
    tools, connector = _landed_tools(env)

    catalog = tools.catalog(WAREHOUSE_CONCEPT)

    assert [(c.name, c.type, c.comment) for c in catalog] == [
        (c.name, c.data_type, c.comment) for c in CONCEPT_COLUMNS
    ]
    assert env.issued == [] and env.sql_lines() == []
    assert connector.metadata_reads == [(CATALOG, SCHEMA)]


def test_warehouse_describe_issues_no_describe_and_logs_only_aggregates(env: Env) -> None:
    tools, connector = _landed_tools(env)

    described = tools.describe_relation(WAREHOUSE_CONCEPT)

    assert described.row_count == CONCEPT_ROWS
    first = CONCEPT_COLUMNS[0]
    profiled = column_of(described, first.name)
    assert (profiled.type, profiled.comment) == (first.data_type, first.comment)
    assert column_of(described, "athena_id").comment is None
    statements = env.statements()
    assert statements == env.issued
    assert not [s for s in statements if CATALOG_STATEMENT.search(s)]
    assert [bool(AGGREGATE.match(s)) for s in statements] == [True]
    assert {line["provider"] for line in env.sql_lines()} == {"warehouse"}
    assert not value_bearing(env)
    assert connector.metadata_reads == [(CATALOG, SCHEMA)]


def test_warehouse_land_passes_the_land_verifier(env: Env) -> None:
    from sema.intake.verify.land import verify_land

    tools, _ = _landed_tools(env)
    land = tools.land

    verdict = verify_land(land, tools, provider="warehouse")

    assert [f.check for f in verdict.failures] == []
    assert verdict.status == "pass"
    assert land.relation(TABLE).row_count == CONCEPT_ROWS
    assert land.relation("codes").row_count == CODES_ROWS


def test_structure_ignores_no_warehouse_relation_that_has_rows(env: Env) -> None:
    from sema.intake.structure.shapes.ignore import NO_ROWS, REASON
    from sema.intake.structure.structure import run_structure

    tools, _ = _landed_tools(env)

    result = run_structure(tools.land, tools, provider="warehouse")

    assert [b.relations for b in result.ignored if b.slots.get(REASON) == NO_ROWS] == []


def test_intake_policy_from_a_bounded_values_build_config() -> None:
    from sema.intake.land.profile import EvidencePolicy
    from sema.models.config import BuildConfig

    config = BuildConfig(evidence_policy="bounded_values", max_distinct_values_read=37)

    policy = EvidencePolicy.from_build_config(config)

    assert policy.tier == "bounded_values"
    assert policy.max_distinct_values_read == 37
