"""edge_list_hierarchy: concept ids with a resolving parent column or edge relation, acyclic.

A parent column keeps the binding on the concept relation; an edge relation spans two and
holds exactly two columns that resolve into the ids, because a third makes it a level table.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

from sema.binding.spec_sections import ParentColumnHierarchy, RelationRef
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import (
    JOIN_RESOLVE_FLOOR,
    RelationProfile,
    Slot,
    SlotBinding,
    is_id,
)
from sema.intake.shapes.base_utils import (
    JOIN_FLOOR,
    ColumnPair,
    failure,
    may_resolve,
    ranges_overlapping,
    section_of,
    slot_columns,
    slot_relations,
    verdict,
)
from sema.intake.verify.base import Failure, Verdict

ID = "edge_list_hierarchy"
SECTION = section_of(ParentColumnHierarchy)
CONCEPT_RELATION, IDENTIFIER, PARENT = "concept_relation", "id", "parent"
EDGE_RELATION, CHILD = "edge_relation", "child"
# The edge relation is optional: a proposal naming one relation reads the parent column form.
SLOTS = (
    Slot(CONCEPT_RELATION, "relation"),
    Slot(EDGE_RELATION, "relation"),
    Slot(IDENTIFIER, "column"),
    Slot(PARENT, "column"),
    Slot(CHILD, "column"),
)
EDGE_LINK_COUNT = 2
Checked = tuple[list[Failure], dict[str, float | str]]


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = []
    for concepts in group:
        keys = [column for column in concepts.columns if is_id(concepts, column)]
        pairs = ranges_overlapping(concepts, _parent_pairs(concepts, keys), tools)
        for key in keys:
            parents = [parent for parent, paired in pairs if paired is key]
            found += _parent_bindings(concepts, key, parents, tools)
            for edges in (p for p in group if p.ref.name != concepts.ref.name):
                found += _edge_bindings(concepts, key, edges, tools)
    return found


def verify(
    binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
) -> Verdict:
    edge_form = EDGE_RELATION in binding.slots
    slots = (CONCEPT_RELATION, EDGE_RELATION) if edge_form else (CONCEPT_RELATION,)
    found, failures = slot_relations(group, binding, slots)
    if failures:
        return verdict(failures, JOIN_FLOOR)
    concepts, edges = found[0], found[-1]
    keys, failures = slot_columns(concepts, binding, (IDENTIFIER,))
    links, missing = slot_columns(edges, binding, (CHILD, PARENT) if edge_form else (PARENT,))
    if failures or missing:
        return verdict(failures + missing, JOIN_FLOOR)
    key = keys[0]
    if not is_id(concepts, key):
        failures.append(failure("id_is_unique", concepts.ref.name, key.name))
    if edge_form:
        failures += _link_count_failures(concepts, key, edges, tools)
        failures += _edge_check(concepts, key, edges, links[0], links[1], tools)[0]
    else:
        failures += _parent_check(concepts, key, links[0], tools)[0]
    return verdict(failures, JOIN_FLOOR)


def _parent_pairs(
    concepts: RelationProfile, keys: Sequence[ColumnProfile]
) -> list[ColumnPair]:
    return [
        (parent, key)
        for key in keys
        for parent in concepts.columns
        if parent.name != key.name and may_resolve(parent, key)
    ]


def _parent_bindings(
    concepts: RelationProfile,
    key: ColumnProfile,
    parents: Sequence[ColumnProfile],
    tools: ProfileTools,
) -> list[SlotBinding]:
    found = []
    for parent in parents:
        failures, evidence = _parent_check(concepts, key, parent, tools)
        if not failures:
            slots = {CONCEPT_RELATION: concepts.ref.name, IDENTIFIER: key.name}
            found.append(_binding({**slots, PARENT: parent.name}, evidence, concepts))
    return found


def _edge_bindings(
    concepts: RelationProfile, key: ColumnProfile, edges: RelationProfile, tools: ProfileTools
) -> list[SlotBinding]:
    resolving = _links(concepts, key, edges, tools)
    if len(resolving) != EDGE_LINK_COUNT:
        return []
    found = []
    for child, parent in permutations(resolving, 2):
        failures, evidence = _acyclic(edges.ref, child.name, parent.name, tools)
        if not failures:
            slots = {CONCEPT_RELATION: concepts.ref.name, IDENTIFIER: key.name}
            slots |= {EDGE_RELATION: edges.ref.name, CHILD: child.name, PARENT: parent.name}
            found.append(_binding(slots, evidence, concepts, edges))
    return found


def _links(
    concepts: RelationProfile, key: ColumnProfile, edges: RelationProfile, tools: ProfileTools
) -> list[ColumnProfile]:
    return [
        column
        for column in edges.columns
        if may_resolve(column, key) and not _resolve_failures(edges, column, concepts, key, tools)
    ]


def _link_count_failures(
    concepts: RelationProfile, key: ColumnProfile, edges: RelationProfile, tools: ProfileTools
) -> list[Failure]:
    count = len(_links(concepts, key, edges, tools))
    if count == EDGE_LINK_COUNT:
        return []
    return [failure("edge_link_count", edges.ref.name, link_count=count)]


def _parent_check(
    concepts: RelationProfile, key: ColumnProfile, parent: ColumnProfile, tools: ProfileTools
) -> Checked:
    if parent.name == key.name:
        return [failure("distinct_columns", concepts.ref.name, parent.name)], {}
    failures = _resolve_failures(concepts, parent, concepts, key, tools)
    if failures:
        return failures, {}
    return _acyclic(concepts.ref, key.name, parent.name, tools)


def _edge_check(
    concepts: RelationProfile,
    key: ColumnProfile,
    edges: RelationProfile,
    child: ColumnProfile,
    parent: ColumnProfile,
    tools: ProfileTools,
) -> Checked:
    if child.name == parent.name:
        return [failure("distinct_columns", edges.ref.name, parent.name)], {}
    failures = _resolve_failures(edges, child, concepts, key, tools)
    failures += _resolve_failures(edges, parent, concepts, key, tools)
    if failures:
        return failures, {}
    return _acyclic(edges.ref, child.name, parent.name, tools)


def _resolve_failures(
    source: RelationProfile,
    column: ColumnProfile,
    concepts: RelationProfile,
    key: ColumnProfile,
    tools: ProfileTools,
) -> list[Failure]:
    relation = source.ref.name
    if column.type != key.type:
        return [failure("key_types_match", relation, column.name, id_type=key.type)]
    joined = tools.check_join(source.ref, column.name, concepts.ref, key.name)
    if joined.resolve_a_to_b < JOIN_RESOLVE_FLOOR:
        return [failure("join_resolves", relation, column.name, resolve=joined.resolve_a_to_b)]
    return []


def _acyclic(ref: RelationRef, child: str, parent: str, tools: ProfileTools) -> Checked:
    checked = tools.check_hierarchy(ref, child, parent)
    evidence: dict[str, float | str] = {
        "root_count": float(checked.root_count),
        "max_depth": float(checked.max_depth),
    }
    if checked.cycle_count:
        return [failure("no_cycles", ref.name, parent, cycle_count=checked.cycle_count)], evidence
    return [], evidence


def _binding(
    slots: dict[str, str], evidence: dict[str, float | str], *spanned: RelationProfile
) -> SlotBinding:
    relations = tuple(profile.ref.name for profile in spanned)
    return SlotBinding(template=ID, slots=slots, evidence=evidence, relations=relations)
