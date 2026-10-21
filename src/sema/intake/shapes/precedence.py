"""Candidate precedence (Decision 17): multi-relation bindings claim their relations.

A loader-fixed binding spans its whole namespace, so it claims every relation the loader
landed, and those relations leave every other template's candidate set first: a loader that
fixes the relation names has already decided the shape. A claimed relation leaves the
single-relation candidate sets. A relation stays ambiguous when two claims include it, or
when it is unclaimed and holds more than one single-relation binding. Nothing here reads a
name or prefers a template by identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sema.intake.shapes import (
    athena_bundle,
    concept_and_value_set,
    edge_list_hierarchy,
    entity_model,
    semsql,
)
from sema.intake.shapes.base import SlotBinding
from sema.intake.verify.base import Ambiguity

MULTI_RELATION = frozenset(
    {
        concept_and_value_set.ID,
        edge_list_hierarchy.ID,
        entity_model.ID,
        semsql.ID,
        athena_bundle.ID,
    }
)
LOADER_FIXED = frozenset({semsql.ID, athena_bundle.ID})
STAGE = "structure"


@dataclass(frozen=True)
class Precedence:
    candidates: tuple[SlotBinding, ...]
    ambiguities: tuple[Ambiguity, ...]


def claims(binding: SlotBinding) -> bool:
    return binding.template in MULTI_RELATION and len(set(binding.relations)) > 1


def apply_precedence(bindings: Sequence[SlotBinding]) -> Precedence:
    bindings = _without_loader_fixed_relations(bindings)
    claiming = [b for b in bindings if claims(b)]
    claimed = _by_relation(claiming)
    single = [b for b in bindings if not claims(b) and not set(b.relations) & claimed.keys()]
    contested = {
        relation: members
        for relation, members in (claimed | _by_relation(single)).items()
        if len(members) > 1
    }
    kept = [b for b in claiming + single if not set(b.relations) & contested.keys()]
    ambiguities = tuple(_ambiguity(relation, contested[relation]) for relation in sorted(contested))
    return Precedence(tuple(sorted(kept, key=_order)), ambiguities)


def member(binding: SlotBinding) -> str:
    slots = ", ".join(f"{slot}={value}" for slot, value in sorted(binding.slots.items()))
    return f"{binding.template}({slots})"


def _without_loader_fixed_relations(bindings: Sequence[SlotBinding]) -> list[SlotBinding]:
    fixed = {r for b in bindings if b.template in LOADER_FIXED for r in b.relations}
    return [b for b in bindings if b.template in LOADER_FIXED or not set(b.relations) & fixed]


def _by_relation(bindings: Sequence[SlotBinding]) -> dict[str, list[SlotBinding]]:
    found: dict[str, list[SlotBinding]] = {}
    for binding in bindings:
        for relation in dict.fromkeys(binding.relations):
            found.setdefault(relation, []).append(binding)
    return found


def _ambiguity(relation: str, members: Sequence[SlotBinding]) -> Ambiguity:
    return Ambiguity(stage=STAGE, subject=relation, members=tuple(sorted(map(member, members))))


def _order(binding: SlotBinding) -> tuple[str, tuple[str, ...], str]:
    return binding.template, binding.relations, member(binding)
