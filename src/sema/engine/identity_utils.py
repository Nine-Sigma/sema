"""Identity predicates over the counts of `warehouse_lookup.RelationEvidence` (contract F4),
and the entity facts they grade.

Each predicate returns True when it holds, False when it fails, and None when it cannot decide:
a count is missing because its statement timed out, or a key has a NULL part.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from sema.engine.join_detector_utils import COVERAGE_THRESHOLD, normalized_name
from sema.engine.normalize_utils import normalize_name
from sema.engine.warehouse_lookup import RelationEvidence
from sema.llm_client import IdentityProposal
from sema.models.extraction import ExtractedColumn
from sema.models.target.identity import EntityIdentity, ResolutionRule, ResolvedAttribute

Relation = tuple[str, str]
Columns = tuple[str, ...]


def one_smaller(columns: Sequence[str]) -> list[tuple[str, ...]]:
    """Each subset one column smaller; uniqueness is monotone, so these stand for every proper
    subset."""
    if len(columns) < 2:
        return []
    return list(combinations(columns, len(columns) - 1))


def key_holds(
    rows: int | None, distinct: int | None, null_rows: int | None,
    smaller_distincts: Sequence[int | None],
) -> bool | None:
    """Unique, minimal, and with no NULL part; `smaller_distincts` counts `one_smaller`'s sets."""
    if rows is None or distinct is None:
        return None
    if distinct != rows or rows in smaller_distincts:
        return False
    if null_rows is None or None in smaller_distincts:
        return None
    return True if null_rows == 0 else None


def entity_holds(
    rows: int | None, distinct: int | None, smaller_distincts: Sequence[int | None],
) -> bool | None:
    """Coarser than the row, and dropping any one key column lowers the distinct count."""
    if rows is None or distinct is None:
        return None
    if distinct >= rows or any(n is not None and n >= distinct for n in smaller_distincts):
        return False
    return None if None in smaller_distincts else True


def conflict_shares(groups: int, conflicting: Mapping[str, int]) -> dict[str, float]:
    """Each attribute's share of the key groups that hold more than one non-NULL value of it."""
    return {name: count / groups if groups else 0.0 for name, count in conflicting.items()}


DECLARED_KEY, SPEC_LIST, CONTAINED = "declared_key", "spec_list", "contained"
# F2's signal order: a lower rank outranks a higher one; no signal ranks last.
_SIGNAL_RANK = {DECLARED_KEY: 0, SPEC_LIST: 1, CONTAINED: 2}
PROVEN, SUPPORTED, OPEN, REJECTED = "proven", "supported", "open", "rejected"
ACCEPTED = (PROVEN, SUPPORTED)


@dataclass
class KeyFact:
    """One nominated column set on a relation: the key predicate's answer, its counts, and the
    signals that corroborate it."""

    columns: tuple[str, ...]
    holds: bool | None
    rows: int | None
    distinct: int | None
    null_rows: int | None
    signals: list[str] = field(default_factory=list)
    verification: str = OPEN


def signal_rank(signals: Sequence[str]) -> int:
    """A signal is its kind, optionally followed by ':' and what it names."""
    return min((_SIGNAL_RANK[s.split(":", 1)[0]] for s in signals), default=len(_SIGNAL_RANK))


def grade_keys(facts: Sequence[KeyFact]) -> KeyFact | None:
    """Grade every set; the key is the one passing set at the top signal rank, else None."""
    for fact in facts:
        fact.verification = grade(fact.holds, bool(fact.signals))
    passing = [f for f in facts if f.holds]
    if not passing:
        return None
    top = min(signal_rank(f.signals) for f in passing)
    tied = [f for f in passing if signal_rank(f.signals) == top]
    if len(tied) == 1:
        return tied[0]
    for fact in tied:
        fact.verification = OPEN
    return None


def key_payload(facts: Sequence[KeyFact], key: KeyFact | None) -> dict[str, Any]:
    """The relation's `has_key` payload: the key, else its first open set, else its first set;
    every other graded set is an alternative."""
    head = key or next((f for f in facts if f.verification == OPEN), facts[0])
    return {
        "columns": list(head.columns), "verification": head.verification,
        "evidence": {"rows": head.rows, "distinct": head.distinct,
                     "null_rows": head.null_rows, "signals": list(head.signals)},
        "alternatives": [{"columns": list(f.columns), "verification": f.verification}
                         for f in facts if f is not head],
    }


def grade(holds: bool | None, corroborated: bool) -> str:
    """F2: undecided is open, a failed predicate rejected, a corroborated one proven."""
    if holds is None:
        return OPEN
    if not holds:
        return REJECTED
    return PROVEN if corroborated else SUPPORTED


@dataclass
class EntityFact:
    """One entity on one relation: the entity's name, its key as the relation spells it, the
    relation that owns it, and the counts behind its grade. `identity` is set when the entity
    is coarser than this relation's row and accepted."""

    relation: Relation
    entity: str
    key_columns: Columns
    owner: Relation
    verification: str
    rows: int | None = None
    groups: int | None = None
    null_rows: int | None = None
    conflict_share: dict[str, float] = field(default_factory=dict)
    identity: EntityIdentity | None = None


def entity_identity(
    key_columns: Sequence[str], conflicting: Mapping[str, int], latest_by: str | None,
) -> EntityIdentity:
    """Each attribute with conflicts takes the greatest `latest_by` when that column picks one
    row of each entity (the caller checked it), else stays unresolved; one with none needs no
    rule. No count accepts a min or max, so neither is stated."""
    attributes = tuple(
        ResolvedAttribute(column=name, rule=ResolutionRule.MAX_BY, by=latest_by)
        for name, count in conflicting.items() if count and latest_by is not None
    )
    return EntityIdentity(key_columns=tuple(key_columns), attributes=attributes)


def identity_payload(fact: EntityFact, owner_ref: str) -> dict[str, Any]:
    """The relation's `has_identity` payload (contract F5)."""
    return {
        "entity": fact.entity, "key_columns": list(fact.key_columns), "owner": owner_ref,
        "identity": fact.identity.model_dump(mode="json") if fact.identity else None,
        "verification": fact.verification,
        "evidence": {"rows": fact.rows, "groups": fact.groups, "null_rows": fact.null_rows,
                     "conflict_share": dict(fact.conflict_share)},
        "alternatives": [],
    }


@dataclass
class Counts:
    """Each count once per build: a single column's distinct count is both its profile and a
    smaller set of a two-column nomination."""

    evidence: RelationEvidence
    _held: dict[tuple[str, Relation, Columns], int | None] = field(default_factory=dict)
    _within: dict[tuple[Relation, Columns, Relation, Columns], bool] = field(
        default_factory=dict)

    def rows(self, relation: Relation) -> int | None:
        return self._count("rows", relation, ())

    def distinct(self, relation: Relation, columns: Columns) -> int | None:
        return self._count("distinct", relation, tuple(sorted(columns)))

    def null_rows(self, relation: Relation, columns: Columns) -> int | None:
        return self._count("null_rows", relation, tuple(sorted(columns)))

    def within(
        self, relation: Relation, columns: Columns, owner: Relation, owner_columns: Columns,
    ) -> bool:
        """The relation's distinct non-NULL tuples are held by the owner's at the floor."""
        held = (relation, columns, owner, owner_columns)
        if held not in self._within:
            found = self.evidence.contained(relation, columns, owner, owner_columns)
            self._within[held] = bool(
                found and found[0] and found[1] / found[0] >= COVERAGE_THRESHOLD)
        return self._within[held]

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


def by_name(
    relations: dict[Relation, list[ExtractedColumn]],
) -> dict[Relation, dict[str, str]]:
    return {r: {normalized_name(c.name): c.name for c in cs} for r, cs in relations.items()}


def same_named(
    relation: Relation, columns: Columns, by_name: dict[Relation, dict[str, str]],
) -> Iterator[tuple[Relation, Columns]]:
    """Each other relation holding every column by normalized name, as it spells them."""
    for other, names in by_name.items():
        theirs = tuple(names.get(normalized_name(c), "") for c in columns)
        if other != relation and all(theirs):
            yield other, theirs


def named_columns(
    columns: Sequence[ExtractedColumn], names: Sequence[str] | None,
) -> Columns | None:
    """A proposal's column names as the relation spells them; None when one is not a column."""
    by_lower = {c.name.lower(): c.name for c in columns}
    found = tuple(by_lower.get(name.lower(), "") for name in names or ())
    if not found or not all(found) or len(set(found)) < len(found):
        return None
    return found


@dataclass
class EntityScope:
    """What grading an entity reads: the build's relations by normalized column name, their
    accepted keys, the counts, and whether a reference list names a relation's columns."""

    relations: dict[Relation, list[ExtractedColumn]]
    keys: Mapping[Relation, KeyFact | None]
    counts: Counts
    referenced: Callable[[Relation, Columns], bool]
    names: dict[Relation, dict[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.names = by_name(self.relations)


def entity_facts(
    scope: EntityScope, proposals: Mapping[Relation, IdentityProposal],
    l2_names: Mapping[Relation, str | None],
) -> list[EntityFact]:
    """Each proposed entity graded on its relation, then each accepted entity on every
    relation that shares it. A proposal naming the row key states no coarser entity; one
    naming another relation's key is that relation's entity, under its L2 name."""
    row_owned: dict[Relation, tuple[str, KeyFact]] = {}
    facts: list[EntityFact] = []
    for relation, proposal in proposals.items():
        key = scope.keys.get(relation)
        named = named_columns(scope.relations[relation], proposal.entity_key)
        if named is not None and key is not None and set(named) == set(key.columns):
            row_owned[relation] = (proposal.entity, key)
        elif named is not None:
            facts.append(_coarser(scope, relation, named, proposal))
    for fact in facts:
        if (owned := _key_owner(scope, fact)) is not None:
            fact.owner = owned[0]
            row_owned.setdefault(owned[0], (fact.entity, owned[1]))
    coarse_owned = [f for f in facts if f.owner == f.relation and f.identity is not None]
    for owner, (proposed, key) in row_owned.items():
        name = normalize_name(l2_names.get(owner) or proposed)
        for fact in (f for f in facts if f.owner == owner and f.relation != owner):
            fact.entity = name
        held = EntityFact(owner, name, key.columns, owner, key.verification, key.rows,
                          key.distinct, key.null_rows)
        shares = _shares(scope, held, facts)
        if shares or any(f.owner == owner and f.relation != owner for f in facts):
            facts.append(held)
        facts.extend(shares)
    for fact in coarse_owned:
        facts.extend(_shares(scope, fact, facts))
    return _one_per_relation(facts)


def _one_per_relation(facts: Sequence[EntityFact]) -> list[EntityFact]:
    """F6 holds one edge per entity and relation: a fact with an identity outranks an accepted
    one, which outranks the rest; the first of equals stays."""
    kept: dict[tuple[str, Relation], EntityFact] = {}
    for fact in facts:
        held = kept.setdefault((fact.entity, fact.relation), fact)
        if _rank(fact) > _rank(held):
            kept[(fact.entity, fact.relation)] = fact
    return list(kept.values())


def _rank(fact: EntityFact) -> tuple[bool, bool]:
    return fact.identity is not None, fact.verification in ACCEPTED


def _coarser(
    scope: EntityScope, relation: Relation, named: Columns, proposal: IdentityProposal,
) -> EntityFact:
    """The entity predicate on the proposed key; a second relation whose same-named columns
    the key contains corroborates it."""
    counts = scope.counts
    rows, groups = counts.rows(relation), counts.distinct(relation, named)
    holds = entity_holds(rows, groups, [counts.distinct(relation, s) for s in one_smaller(named)])
    corroborated = bool(holds) and any(
        counts.within(other, theirs, relation, named)
        for other, theirs in same_named(relation, named, scope.names))
    fact = EntityFact(relation, normalize_name(proposal.entity), named, relation,
                      grade(holds, corroborated), rows, groups, counts.null_rows(relation, named))
    if fact.verification in ACCEPTED:
        _resolve(scope, fact, proposal.latest_by)
    return fact


def _resolve(scope: EntityScope, fact: EntityFact, latest_by: str | None) -> None:
    """Conflict shares of every non-key column, and the rules an ordering column accepts."""
    columns = scope.relations[fact.relation]
    attributes = [c.name for c in columns if c.name not in fact.key_columns]
    held = scope.counts.evidence.conflicts(fact.relation, fact.key_columns, attributes)
    groups, conflicting = held if held is not None else (0, {})
    by = (named_columns(columns, [latest_by]) or (None,))[0] if latest_by else None
    ordered = by is not None and by not in fact.key_columns and (
        scope.counts.distinct(fact.relation, (*fact.key_columns, by)) == fact.rows)
    fact.conflict_share = conflict_shares(groups, conflicting)
    fact.identity = entity_identity(fact.key_columns, conflicting, by if ordered else None)


def _key_owner(scope: EntityScope, fact: EntityFact) -> tuple[Relation, KeyFact] | None:
    """The relation whose accepted key is this accepted entity's key and holds its values."""
    if fact.identity is None:
        return None
    for other, theirs in same_named(fact.relation, fact.key_columns, scope.names):
        key = scope.keys.get(other)
        if key is not None and set(theirs) == set(key.columns) and scope.counts.within(
                fact.relation, fact.key_columns, other, theirs):
            return other, key
    return None


def _shares(
    scope: EntityScope, owned: EntityFact, facts: Sequence[EntityFact],
) -> list[EntityFact]:
    """Each other relation whose same-named columns the owner's key contains; columns that are
    a relation's own key share nothing unless a reference list names them."""
    counts, owner, key = scope.counts, owned.owner, owned.key_columns
    found = []
    for other, theirs in same_named(owner, key, scope.names):
        own = scope.keys.get(other)
        if any(f.relation == other and f.owner == owner for f in facts) or (
                own is not None and set(theirs) == set(own.columns)
                and not scope.referenced(other, theirs)):
            continue
        if counts.within(other, theirs, owner, key):
            found.append(EntityFact(other, owned.entity, theirs, owner, owned.verification,
                                    counts.rows(other), counts.distinct(other, theirs),
                                    counts.null_rows(other, theirs)))
    return found
