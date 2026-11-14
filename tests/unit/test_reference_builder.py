"""One reference builder: the inverse of CanonicalRef.parse, used by L1 and L2 alike."""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import SecretStr

from sema.connectors.databricks import DatabricksConnector
from sema.engine.semantic import SemanticEngine
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.config import Neo4jConfig
from sema.models.physical_key import CanonicalRef, PhysicalKey
from sema.pipeline.build_utils import _build_table_metadata
from tests.support.connector_fixture import (
    CATALOG,
    fixture_build_config,
    fixture_environment,
    make_fake_llm_client,
)

pytestmark = pytest.mark.unit

TABLE_KEY = PhysicalKey("host.example", "cat", "sch", "tbl")


@pytest.mark.parametrize("key", [
    TABLE_KEY,
    replace(TABLE_KEY, column="col"),
    replace(TABLE_KEY, column="a.b"),
])
def test_build_is_the_inverse_of_parse(key: PhysicalKey) -> None:
    assert CanonicalRef.parse(CanonicalRef.build("databricks", key)) == key


def test_build_renders_the_slash_form() -> None:
    assert CanonicalRef.build("databricks", replace(TABLE_KEY, column="col")) == (
        "databricks://host.example/cat/sch/tbl/col"
    )


@pytest.mark.parametrize(("scheme", "key"), [
    ("nosuch", TABLE_KEY),
    ("databricks", replace(TABLE_KEY, table="")),
    ("databricks", replace(TABLE_KEY, schema=None)),
    ("databricks", replace(TABLE_KEY, column="")),
    ("databricks", replace(TABLE_KEY, catalog_or_db="a/b")),
    ("databricks", replace(TABLE_KEY, column="a/b")),
])
def test_build_rejects_unrenderable_keys(scheme: str, key: PhysicalKey) -> None:
    with pytest.raises(ValueError):
        CanonicalRef.build(scheme, key)


def test_with_column_builds_over_the_parsed_table_key() -> None:
    table_ref = CanonicalRef.build("databricks", TABLE_KEY)
    assert CanonicalRef.with_column(table_ref, "col") == CanonicalRef.build(
        "databricks", replace(TABLE_KEY, column="col"),
    )


def test_with_column_rejects_a_column_ref() -> None:
    column_ref = CanonicalRef.build("databricks", replace(TABLE_KEY, column="col"))
    with pytest.raises(ValueError):
        CanonicalRef.with_column(column_ref, "other")


def _column_refs(assertions: list[Assertion], table_ref: str) -> set[str]:
    return {a.subject_ref for a in assertions if a.subject_ref != table_ref}


def test_every_l2_column_ref_equals_the_l1_column_ref_on_the_fixture() -> None:
    neo4j = Neo4jConfig(uri="bolt://unused", user="unused", password=SecretStr("unused"))
    config = fixture_build_config(neo4j)
    with fixture_environment():
        connector = DatabricksConnector(config.databricks, config.profiling)
        work_items = connector.discover_tables(CATALOG, config.schemas)
        engine = SemanticEngine(llm_client=make_fake_llm_client(), run_id="fixture")
        checked = 0
        for item in work_items:
            l1 = connector.extract_table(item)
            l1_columns = {
                a.subject_ref for a in l1 if a.predicate == AssertionPredicate.COLUMN_EXISTS
            }
            l2 = engine.interpret_table(_build_table_metadata(l1, item.fqn))
            assert l1_columns
            assert _column_refs(l2, item.fqn) == l1_columns, item.fqn
            checked += 1
    assert checked == len(work_items) > 0
