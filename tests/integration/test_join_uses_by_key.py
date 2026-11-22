"""`USES` and entity links land on the merge-key match only.

The decoy is a table of the same name in a second schema whose `ref` holds the
bare table name, so a writer that matched on `ref` would link the decoy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from sema.graph.join_materializer import materialize_join_paths
from sema.graph.loader import GraphLoader
from sema.graph.target_join_paths import materialize_target_join_paths
from sema.graph.target_retrieval_loader import EntityTableBinding, PhysicalSchemaBinding
from sema.models.assertions import Assertion, AssertionPredicate, AssertionStatus
from sema.models.physical_key import PhysicalKey
from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.declared import DeclaredJoinDecl
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.refs import TargetEntityRef

pytestmark = pytest.mark.integration

CATALOG = "cat"
MAIN = "main"
DECOY = "mirror"
REFERENCED, REFERENCING = "account", "transfer"
COLUMNS = {REFERENCED: "account_id", REFERENCING: "from_account_id"}


@pytest.fixture
def loader(clean_neo4j: Any) -> GraphLoader:
    loader = GraphLoader(clean_neo4j)
    for schema, ref_of in ((MAIN, _slash_ref), (DECOY, lambda table: table)):
        for table, column in COLUMNS.items():
            loader.upsert_table(table, schema, CATALOG, ref=ref_of(table))
            loader.upsert_column(column, table, schema, CATALOG, "STRING")
            loader.upsert_entity(
                name=f"{schema}-{table}", description=None, source="test",
                confidence=0.9, table_name=table, schema_name=schema,
                catalog=CATALOG, source_schema=schema,
            )
    return loader


def _slash_ref(table: str) -> str:
    return f"databricks://ds/{CATALOG}/{MAIN}/{table}"


def _declared_join() -> Assertion:
    predicate = {
        "left_table": REFERENCED, "left_column": COLUMNS[REFERENCED],
        "right_table": REFERENCING, "right_column": COLUMNS[REFERENCING],
        "operator": "=",
    }
    return Assertion(
        id="declared", subject_ref=_slash_ref(REFERENCING),
        object_ref=_slash_ref(REFERENCED), predicate=AssertionPredicate.JOINS_TO,
        payload={
            "join_predicates": [predicate], "hop_count": 1,
            "from_ref": _slash_ref(REFERENCED), "to_ref": _slash_ref(REFERENCING),
            "constraint_name": "fk_transfer_account",
        },
        source="connector", confidence=0.95, status=AssertionStatus.AUTO,
        run_id="run-1", observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _read(driver: Any, query: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [record.data() for record in session.run(query)]


def _uses(driver: Any) -> set[tuple[str, str, str | None, str]]:
    rows = _read(driver, (
        "MATCH (jp:JoinPath)-[:USES]->(n) RETURN jp.name AS jp, "
        "coalesce(n.table_name, n.name) AS table, "
        "CASE WHEN n:Column THEN n.name END AS column, n.schema_name AS schema"
    ))
    return {(r["jp"], r["table"], r["column"], r["schema"]) for r in rows}


def _entity_links(driver: Any) -> set[tuple[str, str]]:
    rows = _read(driver, (
        "MATCH (:JoinPath)-[r:FROM_ENTITY|TO_ENTITY]->(e:Entity) "
        "RETURN type(r) AS rel, e.name AS entity"
    ))
    return {(r["rel"], r["entity"]) for r in rows}


def test_source_join_uses_tables_and_columns_by_merge_key(
    loader: GraphLoader, clean_neo4j: Any,
) -> None:
    materialize_join_paths(
        loader, {("s", "joins_to"): [_declared_join()]}, source_schema=MAIN,
    )
    name = "account/account_id=transfer/from_account_id"
    assert _uses(clean_neo4j) == {
        (name, REFERENCED, None, MAIN), (name, REFERENCING, None, MAIN),
        (name, REFERENCED, COLUMNS[REFERENCED], MAIN),
        (name, REFERENCING, COLUMNS[REFERENCING], MAIN),
    }
    assert _entity_links(clean_neo4j) == {
        ("FROM_ENTITY", f"{MAIN}-{REFERENCED}"), ("TO_ENTITY", f"{MAIN}-{REFERENCING}"),
    }


def test_writers_match_the_key_not_the_ref(loader: GraphLoader, clean_neo4j: Any) -> None:
    loader.upsert_join_path("jp", [], 1, "test", 1.0, source_schema=MAIN)
    table_key = PhysicalKey("ds", CATALOG, MAIN, REFERENCED)
    loader.add_join_path_uses("jp", table_key, source_schema=MAIN)
    loader.add_join_path_uses(
        "jp", PhysicalKey("ds", CATALOG, MAIN, REFERENCED, COLUMNS[REFERENCED]),
        source_schema=MAIN,
    )
    loader.add_join_path_entity_links(
        "jp", table_key, PhysicalKey("ds", CATALOG, MAIN, REFERENCING), source_schema=MAIN,
    )
    assert _uses(clean_neo4j) == {
        ("jp", REFERENCED, None, MAIN), ("jp", REFERENCED, COLUMNS[REFERENCED], MAIN),
    }
    assert _entity_links(clean_neo4j) == {
        ("FROM_ENTITY", f"{MAIN}-{REFERENCED}"), ("TO_ENTITY", f"{MAIN}-{REFERENCING}"),
    }


def _target_model() -> NormalizedTargetModel:
    def entity(name: str, joins: list[DeclaredJoinDecl]) -> TargetEntityDecl:
        ref = TargetEntityRef(
            target_model_id="m", qualified_name=name, kind=TargetArtifactKind.TABLE_ROW,
        )
        return TargetEntityDecl(ref=ref, declared_joins=joins)

    join = DeclaredJoinDecl(
        referenced_entity="ns.referenced",
        join_keys=[(COLUMNS[REFERENCING], COLUMNS[REFERENCED])], evidence="declared",
    )
    none = SemanticCompleteness.NONE
    return NormalizedTargetModel(
        descriptor=TargetModelDescriptor(
            target_model_id="m", target_model_version="1", display_name="M",
            completeness=SemanticCompletenessAnnotations(
                structure=SemanticCompleteness.COMPLETE, obligations=none,
                vocabulary_bindings=none, semantic_aliases=none, terms=none,
            ),
        ),
        entities=[entity("ns.referenced", []), entity("ns.referencing", [join])],
    )


def test_projection_keeps_its_name_and_uses_both_tables(
    loader: GraphLoader, clean_neo4j: Any,
) -> None:
    binding = PhysicalSchemaBinding(
        catalog=CATALOG, schema=MAIN, entity_tables=(
            EntityTableBinding("ns.referenced", REFERENCED),
            EntityTableBinding("ns.referencing", REFERENCING),
        ),
    )
    materialize_target_join_paths(loader, _target_model(), binding)
    name = f"{REFERENCED}__{REFERENCING}"
    assert _uses(clean_neo4j) == {
        (name, REFERENCED, None, MAIN), (name, REFERENCING, None, MAIN),
    }
