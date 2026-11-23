"""Declared joins: the entities, joins and inline lists a dictionary states, and the loaded
relation each declared join's entities are, decided by names and counts on the data."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from sema.engine.identity_utils import Counts, Relation
from sema.engine.join_detector_utils import COVERAGE_THRESHOLD, normalized_name


@dataclass(frozen=True)
class DictionaryEntity:
    name: str
    properties: tuple[str, ...]


@dataclass(frozen=True)
class DeclaredJoin:
    """A join the dictionary states, with its ordered (local, remote) column pairs."""

    referencing: str
    referenced: str
    pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class InlineList:
    entity: str
    property: str
    codes: dict[str, str]


@dataclass(frozen=True)
class DeclaredBindings:
    """Each accepted join is (join, referencing relation, referenced relation, the referenced
    side's rows with a NULL part)."""

    accepted: tuple[tuple[DeclaredJoin, Relation, Relation, int], ...]
    open: tuple[DeclaredJoin, ...]
    rejected: tuple[DeclaredJoin, ...]
    bound: dict[str, Relation]
    unplaced: dict[str, tuple[str, ...]]


def bind_declared(
    entities: Sequence[DictionaryEntity], joins: Sequence[DeclaredJoin],
    columns: Mapping[Relation, Sequence[str]], counts: Counts,
) -> DeclaredBindings:
    """Accept a join when exactly one pair of candidate relations holds it on the data."""
    properties = {entity.name: entity.properties for entity in entities}
    accepted: list[tuple[DeclaredJoin, Relation, Relation, int]] = []
    open_joins: list[DeclaredJoin] = []
    rejected: list[DeclaredJoin] = []
    for join in joins:
        local = _candidates(properties.get(join.referencing, ()), [p[0] for p in join.pairs],
                            columns)
        remote = _candidates(properties.get(join.referenced, ()), [p[1] for p in join.pairs],
                             columns)
        pairs = list(product(local, remote))
        checked = [_checked(child, parent, counts) for child, parent in pairs]
        passing = [(child[0], parent[0], found[0])
                   for (child, parent), found in zip(pairs, checked, strict=True)
                   if found is not None and found[1]]
        if any(found is None for found in checked) or len(passing) > 1:
            open_joins.append(join)
        elif passing:
            accepted.append((join, *passing[0]))
        else:
            rejected.append(join)
    bound = _bound(accepted)
    unplaced = {
        name: missing for name, relation in bound.items()
        if (missing := _unmatched(properties.get(name, ()), columns[relation]))
    }
    return DeclaredBindings(tuple(accepted), tuple(open_joins), tuple(rejected), bound, unplaced)


def _candidates(
    properties: Sequence[str], named: Sequence[str], columns: Mapping[Relation, Sequence[str]],
) -> list[tuple[Relation, tuple[str, ...]]]:
    """Relations holding every named column whose names overlap the entity's at the floor; each
    with the named columns as the relation spells them."""
    wanted = {normalized_name(p) for p in properties}
    found = []
    for relation, names in columns.items():
        spelled = {normalized_name(c): c for c in names}
        if not wanted or not all(normalized_name(n) in spelled for n in named):
            continue
        if len(wanted & set(spelled)) / len(wanted | set(spelled)) >= COVERAGE_THRESHOLD:
            found.append((relation, tuple(spelled[normalized_name(n)] for n in named)))
    return found


def _checked(
    child: tuple[Relation, tuple[str, ...]], parent: tuple[Relation, tuple[str, ...]],
    counts: Counts,
) -> tuple[int, bool] | None:
    """The referenced side's rows with a NULL part, and whether its tuples do not repeat and
    hold the referencing ones; None when a count is missing."""
    relation, names = parent
    rows, nulls = counts.rows(relation), counts.null_rows(relation, names)
    distinct = counts.evidence.contained(relation, names, relation, names)
    if rows is None or nulls is None or distinct is None:
        return None
    holds = rows - nulls == distinct[0] and counts.within(child[0], child[1], relation, names)
    return nulls, holds


def _bound(
    accepted: Sequence[tuple[DeclaredJoin, Relation, Relation, int]],
) -> dict[str, Relation]:
    """Each entity the accepted joins bind to one relation; one bound to two stays unbound."""
    seen: dict[str, set[Relation]] = {}
    for join, referencing, referenced, _ in accepted:
        seen.setdefault(join.referencing, set()).add(referencing)
        seen.setdefault(join.referenced, set()).add(referenced)
    return {name: next(iter(found)) for name, found in seen.items() if len(found) == 1}


def _unmatched(properties: Sequence[str], names: Sequence[str]) -> tuple[str, ...]:
    held = {normalized_name(c) for c in names}
    return tuple(p for p in properties if normalized_name(p) not in held)
