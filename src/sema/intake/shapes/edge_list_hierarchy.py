"""edge_list_hierarchy: concept ids with a resolving parent column or edge relation, acyclic.

A parent column keeps the binding on the concept relation; an edge relation spans two.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

from sema.binding.spec_sections import RelationRef
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.shapes.base import JOIN_RESOLVE_FLOOR, RelationProfile, SlotBinding, is_id
from sema.intake.shapes.base_utils import (
    JOIN_FLOOR,
    failure,
    may_resolve,
    slot_columns,
    slot_relations,
    verdict,
)
from sema.intake.verify.base import Failure, Verdict

ID = "edge_list_hierarchy"
CONCEPT_RELATION, IDENTIFIER, PARENT = "concept_relation", "id", "parent"
EDGE_RELATION, CHILD = "edge_relation", "child"
Checked = tuple[list[Failure], dict[str, float | str]]


def candidates(group: Sequence[RelationProfile], tools: ProfileTools) -> list[SlotBinding]:
    found = []
    for concepts in group:
        for key in (column for column in concepts.columns if is_id(concepts, column)):
            found += _parent_bindings(concepts, key, tools)
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
        failures += _edge_check(concepts, key, edges, links[0], links[1], tools)[0]
    else:
        failures += _parent_check(concepts, key, links[0], tools)[0]
    return verdict(failures, JOIN_FLOOR)


def _parent_bindings(
    concepts: RelationProfile, key: ColumnProfile, tools: ProfileTools
) -> list[SlotBinding]:
    found = []
    for parent in (column for column in concepts.columns if may_resolve(column, key)):
        if parent.name == key.name:
            continue
        failures, evidence = _parent_check(concepts, key, parent, tools)
        if not failures:
            slots = {CONCEPT_RELATION: concepts.ref.name, IDENTIFIER: key.name}
            found.append(_binding({**slots, PARENT: parent.name}, evidence, concepts))
    return found


def _edge_bindings(
    concepts: RelationProfile, key: ColumnProfile, edges: RelationProfile, tools: ProfileTools
) -> list[SlotBinding]:
    resolving = [
        column
        for column in edges.columns
        if may_resolve(column, key) and not _resolve_failures(edges, column, concepts, key, tools)
    ]
    found = []
    for child, parent in permutations(resolving, 2):
        failures, evidence = _acyclic(edges.ref, child.name, parent.name, tools)
        if not failures:
            slots = {CONCEPT_RELATION: concepts.ref.name, IDENTIFIER: key.name}
            slots |= {EDGE_RELATION: edges.ref.name, CHILD: child.name, PARENT: parent.name}
            found.append(_binding(slots, evidence, concepts, edges))
    return found


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
