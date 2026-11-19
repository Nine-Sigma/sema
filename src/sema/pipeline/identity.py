"""Build step: each loaded relation's key, nominated and graded on the data, zero model calls.

Runs once every table of the run is built and its joins are materialized. Nominations are the
source's declared keys, the column lists intake's spec states (each code list's code column,
each foreign key's reference columns) placed by column name, and each single column whose
distinct count equals the row count. A count on the data accepts or rejects each one (contract
F2, F4); the relation's `has_key` assertion and its `Table` read form record the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import uuid

from sema.connectors.base import Connector
from sema.engine.identity_utils import (
    CONTAINED,
    DECLARED_KEY,
    SPEC_LIST,
    KeyFact,
    grade_keys,
    key_holds,
    key_payload,
    one_smaller,
)
from sema.engine.join_detector_utils import COVERAGE_THRESHOLD, normalized_name
from sema.engine.warehouse_lookup import RelationEvidence
from sema.graph.identity_read_forms import write_table_keys
from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import fetch_columns_by_schema
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

Relation = tuple[str, str]
Columns = tuple[str, ...]


@dataclass
class _Counts:
    """Each count once per build: a single column's distinct count is both its profile and a
    smaller set of a two-column nomination."""

    evidence: RelationEvidence
    _held: dict[tuple[str, Relation, Columns], int | None] = field(default_factory=dict)

    def rows(self, relation: Relation) -> int | None:
        return self._count("rows", relation, ())

    def distinct(self, relation: Relation, columns: Columns) -> int | None:
        return self._count("distinct", relation, tuple(sorted(columns)))

    def null_rows(self, relation: Relation, columns: Columns) -> int | None:
        return self._count("null_rows", relation, tuple(sorted(columns)))

    def _count(self, measure: str, relation: Relation, columns: Columns) -> int | None:
        held = (measure, relation, columns)
        if held not in self._held:
            if measure == "rows":
                self._held[held] = self.evidence.rows(relation)
            elif measure == "distinct":
                self._held[held] = self.evidence.distinct(relation, columns)
            else:
                self._held[held] = self.evidence.null_rows(relation, columns)
        return self._held[held]


def run_identity(
    loader: GraphLoader, connector: Connector, config: BuildConfig, schemas: list[str],
    run_id: str,
) -> None:
    """Grade each relation's nominated keys and write its `has_key` assertion and key."""
    if not config.enable_fk_detection:
        return
    dictionary = read_dictionary(config.concept_spec, config.concept_staging)
    relations: dict[Relation, list[ExtractedColumn]] = {}
    for schema in schemas:
        for column in fetch_columns_by_schema(loader, schema):
            relations.setdefault((column.schema, column.table_name), []).append(column)
    counts = _Counts(RelationEvidence(connector, config.catalog))
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
    _write(loader, relations, facts, run_id)


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
    stated: list[Columns], counts: _Counts,
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
    relation: Relation, nominated: dict[frozenset[str], KeyFact], counts: _Counts,
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
    relations: dict[Relation, list[ExtractedColumn]], counts: _Counts,
) -> None:
    """A second relation whose same-named columns are contained in a passing set corroborates it."""
    names = {r: {normalized_name(c.name): c.name for c in cs} for r, cs in relations.items()}
    for relation, nominated in facts.items():
        for fact in (f for f in nominated.values() if f.holds):
            for other, by_name in names.items():
                theirs = tuple(by_name.get(normalized_name(c), "") for c in fact.columns)
                if other == relation or not all(theirs):
                    continue
                held = counts.evidence.contained(other, theirs, relation, fact.columns)
                if held and held[0] and held[1] / held[0] >= COVERAGE_THRESHOLD:
                    fact.signals.append(f"{CONTAINED}:{other[1]}")


def _write(
    loader: GraphLoader, relations: dict[Relation, list[ExtractedColumn]],
    facts: dict[Relation, dict[frozenset[str], KeyFact]], run_id: str,
) -> None:
    keys: dict[Relation, KeyFact | None] = {}
    for relation, nominated in facts.items():
        graded = list(nominated.values())
        keys[relation] = key = grade_keys(graded)
        subject = _table_ref(relations[relation])
        if not graded or subject is None:
            continue
        loader.store_assertion(Assertion(
            id=str(uuid.uuid4()), subject_ref=subject,
            predicate=AssertionPredicate.HAS_KEY, payload=key_payload(graded, key),
            source=_SOURCE, confidence=_CONFIDENCE, status=AssertionStatus.AUTO,
            run_id=run_id, observed_at=datetime.now(timezone.utc), source_schema=relation[0],
        ), source_schema=relation[0])
    write_table_keys(loader, [
        (relations[relation][0].catalog, *relation,
         (list(key.columns), key.verification) if key else None)
        for relation, key in keys.items()
    ])


def _table_ref(columns: list[ExtractedColumn]) -> str | None:
    ref = next((c.ref for c in columns if c.ref), None)
    if ref is None:
        return None
    return CanonicalRef.build(ref.split("://", 1)[0], replace(CanonicalRef.parse(ref), column=None))
