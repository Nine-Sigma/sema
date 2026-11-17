"""Per fact kind: what SEMA produced, how it compares with an oracle, and how much rechecks.

`produced` and an oracle are `Cells` per fact kind, keyed as the contract's oracle file format
keys them, and scored by the intake scorecard's verdict arithmetic. A kind absent from
`produced` prints `not produced`; a kind the oracle holds no claim for prints `no oracle`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from sema.connectors.base import Connector
from sema.eval.discovery_scorecard_utils import (
    GRAPH_JOINS,
    declared_join_key,
    graph_join_key,
    join_holds,
    unique_keys,
    unique_keys_query,
)
from sema.intake.run.scorecard import VERDICTS, Cells, KindScore, kind_score, verdicts

FACT_KINDS = (
    "candidate_key", "entity_coarser_than_row", "entity_shared_across_relations",
    "null_key_policy", "attribute_resolution_rule", "label_source", "observed_values",
    "column_to_concept_binding", "join", "bridge", "hierarchy", "mapping_coverage",
    "value_encoding", "mapping_cardinality", "measure_semantics", "time_roles",
    "universe_and_special_values", "term_default",
)
JOIN = "join"
CANDIDATE_KEY = "candidate_key"
KEY_ONLY = "key_only"
UNIQUE = "unique"
MANY_TO_ONE = "many_to_one"
ONE_TO_ONE = "one_to_one"
NOT_PRODUCED = "not produced"
NO_ORACLE = "no oracle"
HEADER = ("kind", "produced", *VERDICTS, "rechecked")


class KindRow(BaseModel):
    """One fact kind's cells: `produced` None is not produced, `score` None is no oracle."""

    model_config = ConfigDict(frozen=True)

    produced: int | None
    score: KindScore | None
    rechecked: int


class DiscoveryCard(BaseModel):
    """One row per fact kind, in `FACT_KINDS` order."""

    model_config = ConfigDict(frozen=True)

    kinds: dict[str, KindRow]


def score_discovery(
    produced: Mapping[str, Cells], oracle: Mapping[str, Cells], rechecked: Mapping[str, int],
) -> DiscoveryCard:
    return DiscoveryCard(kinds={
        kind: _row(produced.get(kind), oracle.get(kind), rechecked.get(kind, 0))
        for kind in FACT_KINDS
    })


def _row(made: Cells | None, claims: Cells | None, rechecked: int) -> KindRow:
    score = kind_score(verdicts(claims, _resolved(made or {}, claims))) if claims else None
    return KindRow(produced=None if made is None else len(made), score=score,
                   rechecked=rechecked)


def _resolved(made: Cells, claims: Cells) -> Cells:
    """A key-only fact takes the oracle's value on its key: the key is the whole claim made."""
    return {
        key: claims.get(key, value) if value == KEY_ONLY else value
        for key, value in made.items()
    }


def render(card: DiscoveryCard) -> str:
    """A markdown table: one row per fact kind under the cell header, no total row."""
    rows = [HEADER, tuple("---" for _ in HEADER)]
    rows += [(kind, *_cells(row)) for kind, row in card.kinds.items()]
    return "".join(f"| {' | '.join(row)} |\n" for row in rows)


def _cells(row: KindRow) -> tuple[str, ...]:
    produced = NOT_PRODUCED if row.produced is None else str(row.produced)
    if row.score is None:
        return (produced, *(NO_ORACLE for _ in VERDICTS), str(row.rechecked))
    return (produced, *(str(getattr(row.score, v)) for v in VERDICTS), str(row.rechecked))


def facts_from_graph(
    driver: Any, catalog: str | None = None, schemas: list[str] | None = None,
) -> dict[str, Cells]:
    """The build's facts for the catalog and schemas, in oracle keys; no schemas means every
    schema of the catalog. A join with no recorded cardinality is key-only."""
    with driver.session() as session:
        records = list(session.run(GRAPH_JOINS, catalog=catalog, schemas=schemas or None))
    return {JOIN: {
        graph_join_key(record["predicates"]): record["cardinality"] or KEY_ONLY
        for record in records
    }}


def declared_constraints(
    connector: Connector, catalog: str, schemas: list[str] | None,
) -> dict[str, Cells]:
    """Declared unique keys and foreign keys; a foreign key on unique columns is one to one."""
    tables = connector.discover_tables(catalog, schemas or None)
    candidate_keys = {
        key: UNIQUE
        for schema in sorted({table.schema for table in tables})
        for key in unique_keys(connector.execute_query(unique_keys_query(catalog, schema)))
    }
    joins = {
        key: ONE_TO_ONE if key[:2] in candidate_keys else MANY_TO_ONE
        for table in tables
        for foreign_key in connector.extract_foreign_keys_dto(catalog, table.schema,
                                                              table.table_name)
        for key in [declared_join_key(foreign_key)]
    }
    return {CANDIDATE_KEY: candidate_keys, JOIN: joins}


def recheck(
    connector: Connector, catalog: str, schemas: list[str] | None,
    produced: Mapping[str, Cells],
) -> dict[str, int]:
    """Produced facts whose evidence recomputes on the data; only joins have a recheck."""
    tables = {
        table.table_name.lower(): (table.schema, table.table_name)
        for table in connector.discover_tables(catalog, schemas or None)
    }
    joins = produced.get(JOIN, {})
    return {JOIN: sum(join_holds(connector, catalog, tables, key) for key in joins)}
