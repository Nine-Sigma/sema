"""Structure stage helpers: stem groups, per-group candidates, and the scoped join graph."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import combinations

from pydantic import BaseModel, ConfigDict

from sema.intake.landing import SHEET_SEPARATOR, LandResult
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes import TEMPLATES, apply_precedence, ignore
from sema.intake.shapes.base import JOIN_RESOLVE_FLOOR, RelationProfile, SlotBinding
from sema.intake.shapes.base_utils import may_resolve
from sema.intake.shapes.precedence import STAGE, finding_hints, member
from sema.intake.verify.base import Ambiguity, Failure

TABULAR = "tabular"
NO_TEMPLATE = "no_template"

Profiles = Sequence[RelationProfile]


class JoinEdge(BaseModel):
    """``resolve_rate`` is the share of ``from_column``'s distinct values found in ``to_column``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_relation: str
    from_column: str
    to_relation: str
    to_column: str
    resolve_rate: float
    within_group: bool
    merged: bool = False


def stem_groups(land: LandResult) -> list[tuple[str, ...]]:
    """One group per file stem; a landing of another medium is one group."""
    names = [f"{relation.schema}.{relation.name}" for relation in land.relations]
    if land.medium != TABULAR:
        return [tuple(names)] if names else []
    groups: dict[tuple[str, str], list[str]] = {}
    for relation, name in zip(land.relations, names, strict=True):
        stem, separator, _ = relation.name.rpartition(SHEET_SEPARATOR)
        groups.setdefault((relation.schema, stem if separator else name), []).append(name)
    return [tuple(group) for group in groups.values()]


def group_candidates(group: Profiles, tools: ProfileTools) -> list[SlotBinding]:
    if not group:
        return []
    return [
        binding.model_copy(update={"section": getattr(template, "SECTION", None)})
        for template in TEMPLATES
        if template.ID != ignore.ID
        for binding in template.candidates(group, tools)
    ]


def settle(
    group: Profiles, bindings: Sequence[SlotBinding]
) -> tuple[list[SlotBinding], list[Ambiguity]]:
    """Precedence, then every relation no kept binding covers is listed with what touched it."""
    decided = apply_precedence(bindings)
    covered = {name for binding in decided.candidates for name in binding.relations}
    covered |= {ambiguity.subject for ambiguity in decided.ambiguities}
    left = [
        _left_open(profile.ref.name, bindings)
        for profile in group
        if profile.ref.name not in covered
    ]
    return list(decided.candidates), [*decided.ambiguities, *left]


def _left_open(relation: str, bindings: Sequence[SlotBinding]) -> Ambiguity:
    touching = [binding for binding in bindings if relation in binding.relations]
    members = tuple(sorted({member(binding) for binding in touching})) or (NO_TEMPLATE,)
    return Ambiguity(
        stage=STAGE, subject=relation, members=members, hints=finding_hints(touching)
    )


def describe_failure(found: Failure) -> str:
    where = f"{found.relation}.{found.column}" if found.column else found.relation
    evidence = ", ".join(f"{key}={value}" for key, value in sorted(found.evidence.items()))
    return f"{found.check} on {where or 'the hint'} ({evidence})"


def is_id_signature(profile: RelationProfile, column: ColumnProfile) -> bool:
    return profile.row_count > 0 and column.is_unique and column.null_rate == 0.0


def names_match(a: str, b: str) -> bool:
    """Equal, or the shorter is a suffix of the longer starting at a word boundary."""
    short, long = sorted((a, b), key=len)
    if short == long:
        return True
    return long.endswith(short) and not long[-len(short) - 1].isalnum()


def within_edges(group: Profiles, tools: ProfileTools) -> list[JoinEdge]:
    """Slot evidence: any column into an id-signature column of another relation."""
    edges: list[JoinEdge] = []
    for a, b in combinations(group, 2):
        for left, right in _column_pairs(a, b):
            edges += _checked(a, left, b, right, tools, within_group=True)
    return edges


def cross_edges(
    groups: Sequence[Profiles], claimed: set[str], tools: ProfileTools
) -> list[JoinEdge]:
    """Between groups, id-signature columns at both ends; claimed pairs are evidence only."""
    edges: list[JoinEdge] = []
    for a, b in _cross_pairs(groups, claimed):
        for left, right in _column_pairs(a, b):
            if is_id_signature(a, left) and is_id_signature(b, right):
                edges += _checked(a, left, b, right, tools, within_group=False)
    return edges


def _cross_pairs(
    groups: Sequence[Profiles], claimed: set[str]
) -> Iterator[tuple[RelationProfile, RelationProfile]]:
    for first, second in combinations(groups, 2):
        for a in first:
            yield from (
                (a, b)
                for b in second
                if a.ref.provider == b.ref.provider
                and (a.ref.name in claimed) == (b.ref.name in claimed)
            )


def _column_pairs(
    a: RelationProfile, b: RelationProfile
) -> Iterator[tuple[ColumnProfile, ColumnProfile]]:
    for left in a.columns:
        yield from (
            (left, right)
            for right in b.columns
            if left.type == right.type and names_match(left.name, right.name)
        )


def _checked(
    a: RelationProfile,
    left: ColumnProfile,
    b: RelationProfile,
    right: ColumnProfile,
    tools: ProfileTools,
    *,
    within_group: bool,
) -> list[JoinEdge]:
    into_b = is_id_signature(b, right) and may_resolve(left, right)
    into_a = is_id_signature(a, left) and may_resolve(right, left)
    if not (into_b or into_a):
        return []
    joined = tools.check_join(a.ref, left.name, b.ref, right.name)
    edges = []
    if into_b:
        ends = (a.ref.name, left.name, b.ref.name, right.name)
        edges.append(_edge(ends, joined.resolve_a_to_b, within_group))
    if into_a:
        ends = (b.ref.name, right.name, a.ref.name, left.name)
        edges.append(_edge(ends, joined.resolve_b_to_a, within_group))
    return edges


def _edge(ends: tuple[str, str, str, str], rate: float, within_group: bool) -> JoinEdge:
    from_relation, from_column, to_relation, to_column = ends
    return JoinEdge(
        from_relation=from_relation,
        from_column=from_column,
        to_relation=to_relation,
        to_column=to_column,
        resolve_rate=rate,
        within_group=within_group,
    )


def merging(edge: JoinEdge, claimed: set[str]) -> bool:
    unclaimed = edge.from_relation not in claimed and edge.to_relation not in claimed
    return not edge.within_group and unclaimed and edge.resolve_rate >= JOIN_RESOLVE_FLOOR


def union(count: int, links: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Union-find over group indices; each result lists the indices of one merged group."""
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in links:
        parent[root(first)] = root(second)
    merged: dict[int, list[int]] = {}
    for index in range(count):
        merged.setdefault(root(index), []).append(index)
    return list(merged.values())
