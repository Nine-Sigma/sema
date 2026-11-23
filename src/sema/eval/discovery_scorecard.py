"""Per fact kind: what SEMA produced, how it compares with an oracle, and how much rechecks.

`produced` and an oracle are `Cells` per fact kind, keyed as the contract's oracle file format
keys them, and scored by the intake scorecard's verdict arithmetic. A kind absent from
`produced` prints `not produced`; a kind the oracle holds no claim for prints `no oracle`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from sema.connectors.base import Connector
from sema.engine.join_detector_utils import containment
from sema.engine.warehouse_lookup import RelationEvidence, WarehouseValues
from sema.eval.discovery_scorecard_identity_utils import (
    ATTRIBUTE_RULE,
    CANDIDATE_KEY,
    ENTITY_COARSER,
    ENTITY_SHARED,
    IDENTITY_ASSERTIONS,
    KEYED_ENTITY_EDGES,
    NULL_KEY_POLICY,
    SINGLE,
    UNIQUE,
    EntityRead,
    KeyRead,
    entity_reads,
    identity_counts,
    key_read,
    rule_word,
)
from sema.eval.discovery_scorecard_utils import (
    CODED_BINDINGS,
    GRAPH_JOINS,
    BindingRead,
    binding_read,
    code_relation,
    columns_by_row_count,
    columns_key,
    declared_join_key,
    graph_join_key,
    join_holds,
    label_holds,
    listed_codes,
)
from sema.intake.run.scorecard import VERDICTS, Cells, Key, KindScore, kind_score, verdicts
from sema.models.assertions import REJECTED, STATED_GRADES, VERIFIED, AssertionPredicate

FACT_KINDS = (
    "candidate_key", "entity_coarser_than_row", "entity_shared_across_relations",
    "null_key_policy", "attribute_resolution_rule", "label_source", "observed_values",
    "column_to_concept_binding", "join", "bridge", "hierarchy", "mapping_coverage",
    "value_encoding", "mapping_cardinality", "measure_semantics", "time_roles",
    "universe_and_special_values", "term_default",
)
JOIN = "join"
BINDING = "column_to_concept_binding"
LABEL_SOURCE = "label_source"
MAPPING_COVERAGE = "mapping_coverage"
OBSERVED_VALUES = "observed_values"
CODED_KINDS = (BINDING, LABEL_SOURCE, MAPPING_COVERAGE, OBSERVED_VALUES)
DICTIONARY_ONLY = "dictionary_only"
KEY_ONLY = "key_only"
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


def render(
    card: DiscoveryCard, *, declared: int | None = None, discovered: int | None = None,
) -> str:
    """A markdown table: one row per fact kind under the cell header, no total row; then the
    graph joins read from a declaration and those discovered on the data, when given."""
    rows = [HEADER, tuple("---" for _ in HEADER)]
    rows += [(kind, *_cells(row)) for kind, row in card.kinds.items()]
    table = "".join(f"| {' | '.join(row)} |\n" for row in rows)
    if declared is None and discovered is None:
        return table
    return f"{table}\njoins from declarations: {declared}\njoins discovered: {discovered}\n"


def _cells(row: KindRow) -> tuple[str, ...]:
    produced = NOT_PRODUCED if row.produced is None else str(row.produced)
    if row.score is None:
        return (produced, *(NO_ORACLE for _ in VERDICTS), str(row.rechecked))
    return (produced, *(str(getattr(row.score, v)) for v in VERDICTS), str(row.rechecked))


def facts_from_graph(
    driver: Any, catalog: str | None = None, schemas: list[str] | None = None,
) -> dict[str, Cells]:
    """The build's joins for the catalog and schemas, in oracle keys; no schemas means every
    schema of the catalog. A join with no recorded cardinality is key-only."""
    return {JOIN: {
        graph_join_key(record["predicates"]): record["cardinality"] or KEY_ONLY
        for record in _graph_joins(driver, catalog, schemas)
    }}


def joins_from_graph(
    driver: Any, catalog: str | None = None, schemas: list[str] | None = None,
) -> list[tuple[str, str | None]]:
    """Each graph join's predicates text and the source that wrote it."""
    return [(record["predicates"], record["join_source"])
            for record in _graph_joins(driver, catalog, schemas)]


def _graph_joins(driver: Any, catalog: str | None, schemas: list[str] | None) -> list[Any]:
    with driver.session() as session:
        return list(session.run(GRAPH_JOINS, catalog=catalog, schemas=schemas or None))


def declared_constraints(
    connector: Connector, catalog: str, schemas: list[str] | None,
) -> dict[str, Cells]:
    """Declared unique keys and foreign keys; a foreign key on unique columns is one to one."""
    tables = connector.discover_tables(catalog, schemas or None)
    candidate_keys: Cells = {
        (relation.lower(), columns_key(columns)): UNIQUE
        for schema in sorted({table.schema for table in tables})
        for relation, columns in connector.extract_unique_keys(catalog, schema)
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
    produced: Mapping[str, Cells], *, join_predicates: Iterable[str] = (),
) -> dict[str, int]:
    """Produced joins and identity facts whose evidence recomputes on the data; a join on
    several columns rechecks only when `join_predicates` holds its column pairs."""
    tables = {
        table.table_name.lower(): (table.schema, table.table_name)
        for table in connector.discover_tables(catalog, schemas or None)
    }
    joins = produced.get(JOIN, {})
    paired = {graph_join_key(text): text for text in join_predicates}
    identity = identity_counts(RelationEvidence(connector, catalog), tables, produced)
    return {JOIN: sum(join_holds(connector, catalog, tables, key, paired.get(key))
                      for key in joins), **identity}


def read_identity(
    driver: Any, catalog: str | None = None, schemas: list[str] | None = None,
) -> tuple[list[KeyRead], list[EntityRead]]:
    """Every key and entity fact the build wrote for the catalog and schemas."""
    scope = {"catalog": catalog, "schemas": schemas or None}
    with driver.session() as session:
        keys, entities = (
            list(session.run(IDENTITY_ASSERTIONS, predicate=predicate, **scope))
            for predicate in (AssertionPredicate.HAS_KEY.value,
                              AssertionPredicate.HAS_IDENTITY.value)
        )
        edges = list(session.run(KEYED_ENTITY_EDGES, **scope))
    return [key_read(record) for record in keys], entity_reads(entities, edges)


def identity_facts(keys: Iterable[KeyRead], entities: Iterable[EntityRead]) -> dict[str, Cells]:
    """Key and entity facts in oracle keys: a fact graded `proven` or `supported`, and every
    NULL-key count whose fact is not rejected, since a count is a measurement."""
    keys, entities = list(keys), list(entities)
    stated = [read for read in entities if read.verification in STATED_GRADES]
    measured = [(read.relation, read.key_columns, read.null_rows, read.verification)
                for read in entities]
    measured += [(read.relation, read.columns, read.null_rows, read.verification) for read in keys]
    return {
        CANDIDATE_KEY: {(read.relation.lower(), columns_key(read.columns)): UNIQUE
                        for read in keys if read.verification in STATED_GRADES},
        ENTITY_COARSER: {(read.relation.lower(),): columns_key(identity.key_columns)
                         for read in stated if (identity := read.identity) is not None},
        ENTITY_SHARED: _shared_entities(stated, entities),
        NULL_KEY_POLICY: {(relation.lower(), columns_key(columns)): rows
                          for relation, columns, rows, grade in measured
                          if columns and rows is not None and grade != REJECTED},
        ATTRIBUTE_RULE: {fact: rule for read in stated
                         for fact, rule in _attribute_rules(read).items()},
    }


def _shared_entities(stated: list[EntityRead], entities: list[EntityRead]) -> Cells:
    """Each stated owner edge's entity, keyed by its relation and key, to the other relations
    whose edge on that entity carries key columns and is proven, supported or ungraded (a
    pre-existing edge the step only gave key columns); an L2 edge carries none."""
    shared: Cells = {}
    for owner in stated:
        if owner.owner is not True or not owner.key_columns:
            continue
        others = sorted({read.relation.lower() for read in entities
                         if read.entity == owner.entity and read.key_columns
                         and read.owner is not True
                         and (read.verification is None or read.verification in STATED_GRADES)
                         and read.relation != owner.relation})
        if others:
            shared[(owner.relation.lower(), columns_key(owner.key_columns))] = others
    return shared


def _attribute_rules(read: EntityRead) -> Cells:
    """An entity coarser than its row: `single` for each column no entity group holds two values
    of, then the rule `identity.attributes` names for each resolved column."""
    if read.identity is None:
        return {}
    relation = read.relation.lower()
    rules: Cells = {(relation, column.lower()): SINGLE
                    for column, share in read.conflict_share.items() if share == 0}
    rules.update({(relation, attribute.column.lower()): word
                  for attribute in read.identity.attributes
                  if (word := rule_word(attribute)) is not None})
    return rules


def read_coded_bindings(
    driver: Any, catalog: str | None = None, schemas: list[str] | None = None,
) -> list[BindingRead]:
    """Every coded column's binding the build wrote for the catalog and schemas."""
    with driver.session() as session:
        records = list(session.run(
            CODED_BINDINGS, predicate=AssertionPredicate.HAS_VALUE_SET.value,
            verified=VERIFIED, catalog=catalog, schemas=schemas or None,
        ))
    return [read for record in records if (read := binding_read(record)) is not None]


def loaded_code_lists(
    connector: Connector, catalog: str, schemas: list[str] | None,
    reads: Iterable[BindingRead], code_lists: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Each verified binding's dictionary code list that is loaded as a relation, to that
    relation's name: the one relation of the scope with a key column on the data whose values
    are exactly the list's codes. A column that merely contains the codes is a chance match.
    A column of n rows whose values are exactly n codes holds each code once: it is a key."""
    values = WarehouseValues(connector.execute_query, catalog)
    wanted = {
        read.code_source: codes for read in reads
        if read.verification == VERIFIED and code_relation(read.code_source) is None
        and (codes := listed_codes(values, code_lists, read.code_source))
    }
    if not wanted:
        return {}
    sized = columns_by_row_count(connector, catalog, schemas,
                                 {len(codes) for codes in wanted.values()})
    loaded: dict[str, str] = {}
    for source, codes in wanted.items():
        relations = {table.lower() for schema, table, column in sized.get(len(codes), [])
                     if set(values.counts((schema, table, column)) or {}) == codes}
        if len(relations) == 1:
            loaded[source] = relations.pop()
    return loaded


def coded_facts(
    reads: Iterable[BindingRead], loaded: Mapping[str, str] | None = None,
) -> dict[str, Cells]:
    """The verified bindings' facts in oracle keys; an unverified binding states none. A
    dictionary code list in `loaded` is named by the relation it is loaded as."""
    verified = [read for read in reads if read.verification == VERIFIED]
    return {
        BINDING: {_column(read): _code_list(read.code_source, loaded or {}) for read in verified},
        LABEL_SOURCE: {
            _column(read): label for read in verified if (label := _label(read)) is not None
        },
        MAPPING_COVERAGE: {_column(read): read.listed_row_share for read in verified},
        OBSERVED_VALUES: {
            (*_column(read), code): rows
            for read in verified for code, rows in read.member_rows.items() if rows
        },
    }


def _column(read: BindingRead) -> Key:
    return (read.relation.lower(), read.column.lower())


def _code_list(code_source: str, loaded: Mapping[str, str]) -> str:
    """A code relation's name, or the dictionary vocabulary's loaded relation or its own."""
    key = code_relation(code_source)
    return (key.table if key else loaded.get(code_source, code_source)).lower()


def _label(read: BindingRead) -> list[str] | str | None:
    if read.label is not None:
        return [read.label[1].lower(), read.label[2].lower()]
    return DICTIONARY_ONLY if read.dictionary_label else None


def recheck_coded(
    connector: Connector, catalog: str, reads: Iterable[BindingRead],
    code_lists: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Verified coded facts whose evidence recomputes on the data: the containment holds at
    the floor, the listed row share and each observed value's rows recount, the code
    determines its label column. A dictionary-only label names no column to recheck."""
    values = WarehouseValues(connector.execute_query, catalog)
    counts: Counter[str] = Counter()
    for read in reads:
        if read.verification != VERIFIED:
            continue
        observed = values.counts((read.schema, read.relation, read.column)) or {}
        listed = listed_codes(values, code_lists, read.code_source)
        if observed and listed is not None:
            measured = containment(observed, listed)
            counts[BINDING] += measured.holds
            counts[MAPPING_COVERAGE] += measured.listed_row_share == read.listed_row_share
        counts[OBSERVED_VALUES] += sum(
            observed.get(code) == rows for code, rows in read.member_rows.items() if rows
        )
        counts[LABEL_SOURCE] += label_holds(values, read)
    return {kind: counts[kind] for kind in CODED_KINDS}
