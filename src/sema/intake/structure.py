"""Structure: what a landed namespace permits, computed from aggregates in a fixed order.

The six steps (design Decision 3) are stem groups, profiles and signatures, ignore, template
candidates with precedence, the scoped join graph, and binding. A hint pins what it names
as ``source: human`` and still passes each template's ``verify``. Everything code cannot
decide is listed as an ambiguity, never chosen.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import Provider, ShapeEntry
from sema.intake.landing import LandResult
from sema.intake.profile import ProfileTools
from sema.intake.shapes import TEMPLATES, entity_model, filled_sections, ignore
from sema.intake.shapes.base import (
    RelationProfile,
    RelationSignature,
    SlotBinding,
    profile_relations,
)
from sema.intake.shapes.base_utils import JOIN_FLOOR, SIGNATURE_FLOORS, failure
from sema.intake.shapes.precedence import claims
from sema.intake.structure_hint_utils import hint_bindings
from sema.intake.structure_utils import (
    JoinEdge,
    cross_edges,
    describe_failure,
    group_candidates,
    merging,
    settle,
    stem_groups,
    union,
    within_edges,
)
from sema.intake.verify.base import Ambiguity, Failure, FloorValue
from sema.log import logger

__all__ = [
    "STEPS",
    "BoundEntry",
    "HintRejectedError",
    "JoinEdge",
    "StructureResult",
    "run_structure",
]

STEPS = ("stem_groups", "profiles", "ignore", "candidates", "join_graph", "binding")
CODE_BOUND_CONFIDENCE = 1.0
HINT_OVERLAP = "hint_overlap"
SCHEMA_SLOTS = (entity_model.DICTIONARY, entity_model.HEADER, entity_model.RELATION)

BindingSource = Literal["code", "human"]


class HintRejectedError(ValueError):
    """The hint names a binding the predicates reject; nothing is bound from it."""

    def __init__(self, failures: Sequence[Failure]) -> None:
        self.failures = tuple(failures)
        super().__init__("hint rejected: " + "; ".join(map(describe_failure, self.failures)))


class BoundEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding: SlotBinding
    confidence: float = Field(default=CODE_BOUND_CONFIDENCE, ge=0.0, le=1.0)
    source: BindingSource


class StructureResult(BaseModel):
    """``groups`` lists every relation, ignored ones included, in landing order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    signatures: dict[str, RelationSignature]
    bound: tuple[BoundEntry, ...]
    ignored: tuple[SlotBinding, ...]
    ambiguities: tuple[Ambiguity, ...]
    edges: tuple[JoinEdge, ...]
    floors: dict[str, FloorValue]

    @property
    def schema_relations(self) -> tuple[str, ...]:
        """Dictionary and header relations of bound entity models (design Decision 18)."""
        names = [
            entry.binding.slots[slot]
            for entry in self.bound
            if entry.binding.template == entity_model.ID
            for slot in SCHEMA_SLOTS
            if slot in entry.binding.slots
        ]
        return tuple(dict.fromkeys(names))

    def to_shape(self) -> list[ShapeEntry]:
        pairs: dict[tuple[str, str], None] = {}
        for entry in self.bound:
            sections = filled_sections(entry.binding)
            if not sections:
                raise ValueError(
                    f"the section of {entry.binding.template} over"
                    f" {', '.join(entry.binding.relations)} is unresolved until Roles"
                )
            pairs |= {(section, entry.binding.template): None for section in sections}
        return [ShapeEntry(section=section, template=template) for section, template in pairs]


@dataclass(frozen=True)
class _Graph:
    groups: list[tuple[str, ...]]
    open_groups: list[list[RelationProfile]]
    candidates: list[list[SlotBinding]]
    edges: list[JoinEdge]


def run_structure(
    land: LandResult,
    tools: ProfileTools,
    *,
    provider: Provider = "staging",
    hint: ConceptSourceSpec | None = None,
) -> StructureResult:
    named = stem_groups(land)
    profiles = profile_relations(land, tools, provider)
    by_name = {profile.ref.name: profile for profile in profiles}
    pinned = pin_hint(hint, profiles, tools) if hint is not None else []
    held = {name for binding in pinned for name in binding.relations}
    ignored = [b for b in ignore.candidates(profiles, tools) if b.relations[0] not in held]
    removed = held | {binding.relations[0] for binding in ignored}
    open_groups = [[by_name[name] for name in group if name not in removed] for group in named]
    found = [group_candidates(group, tools) for group in open_groups]
    claimed = _claimed([*pinned, *(binding for group in found for binding in group)])
    graph = _join_graph(named, open_groups, found, claimed, tools)
    bound, ambiguities = _bind(pinned, graph)
    logger.info(
        "structure over {} relations: {} bound, {} ignored, {} ambiguous, {} groups",
        len(profiles), len(bound), len(ignored), len(ambiguities), len(graph.groups),
    )
    return StructureResult(
        order=STEPS,
        groups=tuple(graph.groups),
        signatures={profile.ref.name: profile.signature for profile in profiles},
        bound=tuple(bound),
        ignored=tuple(ignored),
        ambiguities=tuple(ambiguities),
        edges=tuple(graph.edges),
        floors=stage_floors(),
    )


def pin_hint(
    hint: ConceptSourceSpec, profiles: Sequence[RelationProfile], tools: ProfileTools
) -> list[SlotBinding]:
    bindings, failures = hint_bindings(hint)
    templates = {template.ID: template for template in TEMPLATES}
    for binding in bindings:
        failures += templates[binding.template].verify(binding, profiles, tools).failures
    counts = Counter(name for binding in bindings for name in binding.relations)
    failures += [failure(HINT_OVERLAP, name, bindings=n) for name, n in counts.items() if n > 1]
    if failures:
        raise HintRejectedError(failures)
    logger.info("hint pins {} bindings over {} relations", len(bindings), len(counts))
    return bindings


def stage_floors() -> dict[str, FloorValue]:
    floors: dict[str, FloorValue] = {**SIGNATURE_FLOORS, **JOIN_FLOOR}
    for template in TEMPLATES:
        floors |= getattr(template, "FLOORS", {})
    return floors


def _claimed(bindings: Iterable[SlotBinding]) -> set[str]:
    return {name for binding in bindings if claims(binding) for name in binding.relations}


def _join_graph(
    named: Sequence[tuple[str, ...]],
    open_groups: Sequence[list[RelationProfile]],
    found: Sequence[list[SlotBinding]],
    claimed: set[str],
    tools: ProfileTools,
) -> _Graph:
    within = [edge for group in open_groups for edge in within_edges(group, tools)]
    crossing = [
        edge.model_copy(update={"merged": True}) if merging(edge, claimed) else edge
        for edge in cross_edges(open_groups, claimed, tools)
    ]
    index = {profile.ref.name: i for i, group in enumerate(open_groups) for profile in group}
    links = [(index[e.from_relation], index[e.to_relation]) for e in crossing if e.merged]
    graph = _Graph([], [], [], [*within, *crossing])
    for members in union(len(named), links):
        group = [profile for i in members for profile in open_groups[i]]
        graph.groups.append(tuple(name for i in members for name in named[i]))
        graph.open_groups.append(group)
        single = len(members) == 1
        graph.candidates.append(found[members[0]] if single else group_candidates(group, tools))
    return graph


def _bind(
    pinned: Sequence[SlotBinding], graph: _Graph
) -> tuple[list[BoundEntry], list[Ambiguity]]:
    bound = [BoundEntry(binding=binding, source="human") for binding in pinned]
    ambiguities: list[Ambiguity] = []
    for group, found in zip(graph.open_groups, graph.candidates, strict=True):
        kept, left = settle(group, found)
        bound += [BoundEntry(binding=binding, source="code") for binding in kept]
        ambiguities += left
    return bound, ambiguities
