"""Build step: each loaded relation's key and the entity its rows describe, graded on the data.

Runs once every table of the run is built and its joins are materialized. Key nominations are
the source's declared keys, the column lists intake's spec states (each code list's code column,
each foreign key's reference columns) placed by column name, each single column whose distinct
count equals the row count, and the row key the proposer names. With a model, one proposal per
relation names the entity its rows describe; a count on the data accepts or rejects every
nomination (contract F2, F4). The `has_key` and `has_identity` assertions and their read
forms record the result.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
import uuid

from sema.connectors.base import Connector
from sema.engine.identity_prompt import ColumnProfile, identity_prompt, propose
from sema.engine.identity_utils import (
    ACCEPTED,
    CONTAINED,
    DECLARED_KEY,
    SPEC_LIST,
    Columns,
    Counts,
    EntityFact,
    EntityScope,
    KeyFact,
    Relation,
    by_name,
    entity_facts,
    grade_keys,
    identity_payload,
    key_holds,
    key_payload,
    named_columns,
    one_smaller,
    same_named,
)
from sema.engine.join_detector_utils import normalized_name
from sema.engine.warehouse_lookup import RelationEvidence
from sema.graph.identity_read_forms import (
    read_entity_names,
    write_entity_edges,
    write_table_keys,
)
from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import fetch_columns_by_schema
from sema.llm_client import IdentityProposal
from sema.log import logger
from sema.models.assertions import Assertion, AssertionPredicate, AssertionStatus
from sema.models.config import BuildConfig
from sema.models.extraction import ExtractedColumn
from sema.models.physical_key import CanonicalRef
from sema.pipeline.coded_columns import Dictionary, read_dictionary

# A relation's candidate keys are few; the declared and stated sets come first, so past this
# cap only further uncorroborated single columns go unmeasured.
MAX_KEY_SETS = 8
_SOURCE = "identity"
_CONFIDENCE = 0.95



def run_identity(
    loader: GraphLoader, connector: Connector, config: BuildConfig, schemas: list[str],
    run_id: str, client: Any | None = None,
) -> None:
    """Grade each relation's nominated keys and, with a model, the entity its rows describe;
    write the `has_key` and `has_identity` assertions and their read forms."""
    if not config.enable_fk_detection:
        return
    dictionary = read_dictionary(config.concept_spec, config.concept_staging)
    relations: dict[Relation, list[ExtractedColumn]] = {}
    for schema in schemas:
        for column in fetch_columns_by_schema(loader, schema):
            relations.setdefault((column.schema, column.table_name), []).append(column)
    counts = Counts(RelationEvidence(connector, config.catalog))
    declared = _declared(connector, config.catalog, schemas)
    stated = _stated(dictionary, relations)
    facts = {
        relation: _measured(relation, _nominated(
            relation, columns, declared.get((relation[0], relation[1].lower()), []),
            [names for names, _ in stated.get(relation, [])], counts,
        ), counts)
        for relation, columns in relations.items()
    }
    _corroborate_stated(facts, stated)
    _corroborate_contained(facts, relations, counts)
    if client is None:
        _write(loader, relations, facts, run_id)
        return
    named = read_entity_names(loader, [r for cs in relations.values() if (r := _table_ref(cs))])
    l2 = {relation: named.get(_table_ref(columns) or "", (None, None))
          for relation, columns in relations.items()}
    proposals = _proposals(client, relations, facts, l2, counts)
    _nominate_row_keys(facts, proposals, relations, counts)
    keys = _write(loader, relations, facts, run_id)
    scope = EntityScope(relations, keys, counts,
                        _reference_lists(connector, config.catalog, dictionary))
    entities = entity_facts(scope, proposals, {r: names[0] for r, names in l2.items()})
    _write_entities(loader, relations, entities, run_id)


def _declared(
    connector: Connector, catalog: str, schemas: list[str],
) -> dict[Relation, list[Columns]]:
    """Each declared key by (schema, lowercase relation); a schema whose read fails has none."""
    found: dict[Relation, list[Columns]] = {}
    for schema in schemas:
        try:
            keys = connector.extract_unique_keys(catalog, schema)
        except Exception as exc:  # each connector raises its own driver's errors
            logger.warning(f"Declared keys of {schema} not read: {exc}")
            continue
        for relation, columns in keys:
            found.setdefault((schema, relation.lower()), []).append(tuple(columns))
    return found


def _stated(
    dictionary: Dictionary | None, relations: dict[Relation, list[ExtractedColumn]],
) -> dict[Relation, list[tuple[Columns, Columns]]]:
    """Each spec list placed on every relation holding all its columns by normalized name:
    the relation's own names, and the list as the spec spells it."""
    if dictionary is None:
        return {}
    lists = [(code,) for code, _ in dictionary.roles.values()] + dictionary.reference_keys
    placed: dict[Relation, list[tuple[Columns, Columns]]] = {}
    for relation, columns in relations.items():
        by_name = {normalized_name(c.name): c.name for c in columns}
        for spec in dict.fromkeys(lists):
            names = tuple(by_name.get(normalized_name(n), "") for n in spec)
            if all(names):
                placed.setdefault(relation, []).append((names, spec))
    return placed


def _nominated(
    relation: Relation, columns: list[ExtractedColumn], declared: list[Columns],
    stated: list[Columns], counts: Counts,
) -> dict[frozenset[str], KeyFact]:
    """Declared keys, then stated lists, then single columns unique with no NULL (a profile's
    distinct count leaves NULL out), up to `MAX_KEY_SETS`."""
    by_lower = {c.name.lower(): c.name for c in columns}
    nominated: dict[frozenset[str], KeyFact] = {}

    def add(names: Columns, signal: str | None = None) -> None:
        fact = nominated.get(frozenset(names))
        if fact is None and len(nominated) < MAX_KEY_SETS:
            fact = nominated[frozenset(names)] = KeyFact(names, None, None, None, None)
        if fact is not None and signal is not None:
            fact.signals.append(signal)

    for key in declared:
        names = tuple(by_lower.get(c.lower(), "") for c in key)
        if all(names):
            add(names, DECLARED_KEY)
    for names in stated:
        add(names)
    rows = counts.rows(relation)
    for column in columns:
        if len(nominated) >= MAX_KEY_SETS or rows is None:
            break
        single = (column.name,)
        if counts.distinct(relation, single) == rows and counts.null_rows(relation, single) == 0:
            add(single)
    return nominated


def _measured(
    relation: Relation, nominated: dict[frozenset[str], KeyFact], counts: Counts,
) -> dict[frozenset[str], KeyFact]:
    for fact in nominated.values():
        fact.rows = counts.rows(relation)
        fact.distinct = counts.distinct(relation, fact.columns)
        fact.null_rows = counts.null_rows(relation, fact.columns)
        smaller = [counts.distinct(relation, s) for s in one_smaller(fact.columns)]
        fact.holds = key_holds(fact.rows, fact.distinct, fact.null_rows, smaller)
    return nominated


def _corroborate_stated(
    facts: dict[Relation, dict[frozenset[str], KeyFact]],
    stated: dict[Relation, list[tuple[Columns, Columns]]],
) -> None:
    """A spec list corroborates the one relation of the scope it is a key on; a list that is
    a key on several, or undecided on any, corroborates none."""
    placed: dict[Columns, list[KeyFact]] = {}
    for relation, lists in stated.items():
        for names, spec in lists:
            if (fact := facts[relation].get(frozenset(names))) is not None:
                placed.setdefault(spec, []).append(fact)
    for on in placed.values():
        passing = [f for f in on if f.holds]
        if len(passing) == 1 and all(f.holds is not None for f in on):
            passing[0].signals.append(SPEC_LIST)


def _corroborate_contained(
    facts: dict[Relation, dict[frozenset[str], KeyFact]],
    relations: dict[Relation, list[ExtractedColumn]], counts: Counts,
) -> None:
    """A second relation whose same-named columns are contained in a passing set corroborates it."""
    names = by_name(relations)
    for relation, nominated in facts.items():
        for fact in (f for f in nominated.values() if f.holds):
            for other, theirs in same_named(relation, fact.columns, names):
                if counts.within(other, theirs, relation, fact.columns):
                    fact.signals.append(f"{CONTAINED}:{other[1]}")


def _write(
    loader: GraphLoader, relations: dict[Relation, list[ExtractedColumn]],
    facts: dict[Relation, dict[frozenset[str], KeyFact]], run_id: str,
) -> dict[Relation, KeyFact | None]:
    keys: dict[Relation, KeyFact | None] = {}
    for relation, nominated in facts.items():
        graded = list(nominated.values())
        keys[relation] = key = grade_keys(graded)
        subject = _table_ref(relations[relation])
        if graded and subject is not None:
            loader.store_assertion(_assertion(
                subject, AssertionPredicate.HAS_KEY, key_payload(graded, key), relation, run_id,
            ), source_schema=relation[0])
    write_table_keys(loader, [
        (relations[relation][0].catalog, *relation,
         (list(key.columns), key.verification) if key else None)
        for relation, key in keys.items()
    ])
    return keys


def _assertion(
    subject: str, predicate: AssertionPredicate, payload: dict[str, Any], relation: Relation,
    run_id: str,
) -> Assertion:
    return Assertion(
        id=str(uuid.uuid4()), subject_ref=subject, predicate=predicate, payload=payload,
        source=_SOURCE, confidence=_CONFIDENCE, status=AssertionStatus.AUTO, run_id=run_id,
        observed_at=datetime.now(timezone.utc), source_schema=relation[0],
    )


def _proposals(
    client: Any, relations: dict[Relation, list[ExtractedColumn]],
    facts: dict[Relation, dict[frozenset[str], KeyFact]],
    l2: dict[Relation, tuple[str | None, str | None]], counts: Counts,
) -> dict[Relation, IdentityProposal]:
    """One proposer call per relation, a relation with no key or a single-column key too: a
    surrogate key says nothing about the entity."""
    proposals: dict[Relation, IdentityProposal] = {}
    for relation, columns in relations.items():
        key = grade_keys(list(facts[relation].values()))
        profiles = [ColumnProfile(c.name, c.data_type, c.comment,
                                  counts.distinct(relation, (c.name,)),
                                  counts.null_rows(relation, (c.name,))) for c in columns]
        prompt = identity_prompt(relation[1], counts.rows(relation), profiles,
                                 key.columns if key else None, *l2[relation])
        proposal = propose(client, prompt, _table_ref(columns) or relation[1])
        if proposal is not None:
            proposals[relation] = proposal
    return proposals


def _nominate_row_keys(
    facts: dict[Relation, dict[frozenset[str], KeyFact]],
    proposals: dict[Relation, IdentityProposal],
    relations: dict[Relation, list[ExtractedColumn]], counts: Counts,
) -> None:
    """A proposed row key is one more nomination, under the same cap and the key predicate."""
    for relation, proposal in proposals.items():
        names = named_columns(relations[relation], proposal.row_key)
        nominated = facts[relation]
        if names is None or frozenset(names) in nominated or len(nominated) >= MAX_KEY_SETS:
            continue
        added = {frozenset(names): KeyFact(names, None, None, None, None)}
        nominated.update(_measured(relation, added, counts))
        _corroborate_contained({relation: added}, relations, counts)


def _reference_lists(
    connector: Connector, catalog: str, dictionary: Dictionary | None,
) -> Callable[[Relation, Columns], bool]:
    """Whether a declared foreign key of the relation, or a spec reference list, names exactly
    these columns; a relation's declared keys are read once, when first asked."""
    spec = {frozenset(map(normalized_name, ref))
            for ref in (dictionary.reference_keys if dictionary else [])}
    declared: dict[Relation, set[frozenset[str]]] = {}

    def referenced(relation: Relation, columns: Columns) -> bool:
        names = frozenset(map(normalized_name, columns))
        if relation not in declared:
            declared[relation] = _declared_references(connector, catalog, relation)
        return names in spec or names in declared[relation]

    return referenced


def _declared_references(
    connector: Connector, catalog: str, relation: Relation,
) -> set[frozenset[str]]:
    try:
        keys = connector.extract_foreign_keys_dto(catalog, *relation)
    except Exception as exc:  # each connector raises its own driver's errors
        logger.warning(f"Declared foreign keys of {relation[1]} not read: {exc}")
        return set()
    return {frozenset(normalized_name(a) for a, _ in key.column_pairs) for key in keys}


def _write_entities(
    loader: GraphLoader, relations: dict[Relation, list[ExtractedColumn]],
    entities: list[EntityFact], run_id: str,
) -> None:
    """One `has_identity` per relation and entity; an edge for each accepted one."""
    edges = []
    for fact in entities:
        subject, owner = _table_ref(relations[fact.relation]), _table_ref(relations[fact.owner])
        if subject is None or owner is None:
            continue
        loader.store_assertion(_assertion(
            subject, AssertionPredicate.HAS_IDENTITY, identity_payload(fact, owner),
            fact.relation, run_id,
        ), source_schema=fact.relation[0])
        if fact.verification in ACCEPTED:
            edges.append({
                "catalog": relations[fact.relation][0].catalog, "schema": fact.relation[0],
                "table": fact.relation[1], "entity": fact.entity,
                "owner": fact.owner == fact.relation, "key_columns": list(fact.key_columns),
                "verification": fact.verification,
                "identity": fact.identity.model_dump_json() if fact.identity else None,
                "null_key_rows": fact.null_rows or None,
            })
    write_entity_edges(loader, edges, _SOURCE)


def _table_ref(columns: list[ExtractedColumn]) -> str | None:
    ref = next((c.ref for c in columns if c.ref), None)
    if ref is None:
        return None
    return CanonicalRef.build(ref.split("://", 1)[0], replace(CanonicalRef.parse(ref), column=None))
