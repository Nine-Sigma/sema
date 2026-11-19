"""The identity facts behind the discovery scorecard: their graph reads and their rechecks.

A key or entity fact is read as the build graph holds it (contract F6): a `has_key` or
`has_identity` assertion, its relation's `key_columns`, and the relation's `ENTITY_ON_TABLE`
edges. A recheck recounts a fact through `RelationEvidence` and decides it with the build's own
predicates in `engine.identity_utils`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from sema.engine.identity_utils import entity_holds, key_holds, one_smaller
from sema.engine.join_detector_utils import COVERAGE_THRESHOLD
from sema.engine.warehouse_lookup import RelationEvidence, TableKey
from sema.eval.discovery_scorecard_utils import COLUMN_SEPARATOR
from sema.intake.run.scorecard import Cells, Key
from sema.models.target.identity import EntityIdentity, ResolutionRule, ResolvedAttribute

CANDIDATE_KEY = "candidate_key"
ENTITY_COARSER = "entity_coarser_than_row"
ENTITY_SHARED = "entity_shared_across_relations"
NULL_KEY_POLICY = "null_key_policy"
ATTRIBUTE_RULE = "attribute_resolution_rule"
IDENTITY_KINDS = (CANDIDATE_KEY, ENTITY_COARSER, ENTITY_SHARED, NULL_KEY_POLICY, ATTRIBUTE_RULE)
UNIQUE = "unique"
SINGLE = "single"
LATEST = "latest:"
STATED_GRADES = frozenset({"proven", "supported"})
REJECTED = "rejected"
# shortcut: the predicate words N4 and N5 add to `AssertionPredicate`; use the enum once merged.
HAS_KEY = "has_key"
HAS_IDENTITY = "has_identity"
IDENTITY_ASSERTIONS = (
    "MATCH (a:Assertion {predicate: $predicate}) WHERE a.status <> 'superseded' "
    "MATCH (t:Table {ref: a.subject_ref}) "
    "WHERE ($catalog IS NULL OR t.catalog = $catalog) "
    "AND ($schemas IS NULL OR t.schema_name IN $schemas) "
    "RETURN t.name AS relation, a.payload AS payload, properties(t).key_columns AS table_key, "
    "[(e:Entity)-[r:ENTITY_ON_TABLE]->(t) | [e.name, properties(r)]] AS edges"
)


class KeyRead(NamedTuple):
    """A key fact: the relation, its key columns, its grade and its NULL-key row count."""

    relation: str
    columns: tuple[str, ...]
    verification: str | None
    null_rows: int | None


class EntityRead(NamedTuple):
    """An entity on a relation: its edge's key columns, `owner` and `EntityIdentity`, the
    grade, and the NULL-key rows and conflict share per non-key column its evidence counts.
    An edge L2 wrote has no key columns and no grade."""

    relation: str
    entity: str
    key_columns: tuple[str, ...] | None
    owner: bool | None
    identity: EntityIdentity | None
    verification: str | None
    null_rows: int | None
    conflict_share: dict[str, float]


def _evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload.get("evidence") or {})


def key_read(record: Mapping[str, Any]) -> KeyRead:
    """A `has_key` assertion. The relation's key is its `key_columns`: another accepted set is
    an alternative, so it states no key and keeps only its NULL count."""
    payload = json.loads(record["payload"])
    columns, grade = tuple(payload["columns"]), payload.get("verification")
    if grade in STATED_GRADES and list(columns) != list(record["table_key"] or []):
        grade = None
    return KeyRead(record["relation"], columns, grade, _evidence(payload).get("null_rows"))


def entity_read(record: Mapping[str, Any]) -> EntityRead:
    """A `has_identity` assertion: its entity's edge on the relation states the fact, and the
    assertion's evidence its counts."""
    payload = json.loads(record["payload"])
    entity, evidence = str(payload.get("entity")), _evidence(payload)
    edge: dict[str, Any] = next(
        (properties for name, properties in record["edges"] if name == entity), {})
    key, identity = edge.get("key_columns") or payload.get("key_columns"), edge.get("identity")
    return EntityRead(
        relation=record["relation"], entity=entity, key_columns=tuple(key) if key else None,
        owner=edge.get("owner"),
        identity=EntityIdentity.model_validate_json(identity) if identity else None,
        verification=edge.get("verification", payload.get("verification")),
        null_rows=evidence.get("null_rows"),
        conflict_share=dict(evidence.get("conflict_share") or {}),
    )


def columns_of(key: str) -> list[str]:
    return key.split(COLUMN_SEPARATOR)


def rule_word(attribute: ResolvedAttribute) -> str | None:
    """An attribute rule in the F1 form; a `min_by` rule has none."""
    if attribute.rule is ResolutionRule.MAX_BY:
        return f"{LATEST}{(attribute.by or '').lower()}"
    return None if attribute.rule is ResolutionRule.MIN_BY else attribute.rule.value


@dataclass
class IdentityRecheck:
    """Per identity kind, the produced facts whose counts the data re-proves; `tables` maps a
    lowercased relation name to its (schema, table)."""

    evidence: RelationEvidence
    tables: Mapping[str, TableKey]

    def counts(self, produced: Mapping[str, Cells]) -> dict[str, int]:
        entities = produced.get(ENTITY_COARSER, {})
        entity_keys = {key[0]: str(value) for key, value in entities.items()}
        checks: dict[str, Callable[[Key, Any], bool]] = {
            CANDIDATE_KEY: lambda key, _: self._key(self.tables[key[0]], columns_of(key[1])),
            ENTITY_COARSER: lambda key, value: self._entity(self.tables[key[0]],
                                                            columns_of(str(value))),
            ENTITY_SHARED: self._shared,
            NULL_KEY_POLICY: lambda key, value: self.evidence.null_rows(
                self.tables[key[0]], columns_of(key[1])) == value,
            ATTRIBUTE_RULE: lambda key, value: self._rule(key, value, entity_keys.get(key[0])),
        }
        return {
            kind: sum(key[0] in self.tables and check(key, value)
                      for key, value in produced.get(kind, {}).items())
            for kind, check in checks.items()
        }

    def _smaller(self, relation: TableKey, columns: Sequence[str]) -> list[int | None]:
        return [self.evidence.distinct(relation, subset) for subset in one_smaller(columns)]

    def _key(self, relation: TableKey, columns: Sequence[str]) -> bool:
        return key_holds(
            self.evidence.rows(relation), self.evidence.distinct(relation, columns),
            self.evidence.null_rows(relation, columns), self._smaller(relation, columns),
        ) is True

    def _entity(self, relation: TableKey, columns: Sequence[str]) -> bool:
        return entity_holds(
            self.evidence.rows(relation), self.evidence.distinct(relation, columns),
            self._smaller(relation, columns),
        ) is True

    def _shared(self, key: Key, others: Any) -> bool:
        """Every other relation's non-NULL key tuples are contained in the owner's at the floor."""
        owner, columns = self.tables[key[0]], columns_of(key[1])
        return bool(others) and all(
            other in self.tables and self._contained(self.tables[other], owner, columns)
            for other in others
        )

    def _contained(self, relation: TableKey, owner: TableKey, columns: Sequence[str]) -> bool:
        counted = self.evidence.contained(relation, columns, owner, columns)
        return counted is not None and counted[0] > 0 and (
            counted[1] / counted[0] >= COVERAGE_THRESHOLD)

    def _rule(self, key: Key, value: Any, entity_key: str | None) -> bool:
        """`single`: no entity group holds two values; `latest:<by>`: the ordering column picks
        exactly one row of each entity. No count accepts `min` or `max`."""
        if entity_key is None:
            return False
        relation, columns = self.tables[key[0]], columns_of(entity_key)
        if value == SINGLE:
            counted = self.evidence.conflicts(relation, columns, [key[1]])
            return counted is not None and counted[1][key[1]] == 0
        if not str(value).startswith(LATEST):
            return False
        ordered = [*columns, str(value).removeprefix(LATEST)]
        rows = self.evidence.rows(relation)
        return rows is not None and self.evidence.distinct(relation, ordered) == rows
