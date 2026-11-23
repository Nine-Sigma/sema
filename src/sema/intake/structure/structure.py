"""Structure: what a landed namespace permits, computed from aggregates in a fixed order.

The six steps (design Decision 3) are stem groups, profiles and signatures, ignore, template
candidates with precedence, the scoped join graph, and binding. A hint pins what it names
as ``source: human`` and still passes each template's ``verify``. Everything code cannot
decide is listed as an ambiguity, never chosen.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import Provider, ShapeEntry
from sema.intake.structure.described_data_utils import (
    SCHEMA_SLOTS,
    described_beside,
    described_by,
    per_file_dictionaries,
    schema_slot_relations,
    uses_described,
)
from sema.intake.land.landing import LandResult
from sema.intake.land.profile import ProfileTools
from sema.intake.structure.shapes import TEMPLATES, entity_model, filled_sections, ignore
from sema.intake.structure.shapes.base import (
    CLASSES,
    RelationProfile,
    RelationSignature,
    SlotBinding,
    profile_relations,
)
from sema.intake.structure.shapes.base_utils import JOIN_FLOOR, SIGNATURE_FLOORS, failure
from sema.intake.structure.shapes.precedence import claims
from sema.intake.structure.structure_hint_utils import hint_bindings
from sema.intake.structure.structure_utils import (
    NO_TEMPLATE,
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
from sema.intake.verify.base import Ambiguity, Failure, FloorClass, FloorValue
from sema.intake.structure.views.base import ViewSpec
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

BindingSource = Literal["code", "human", "model"]


class HintRejectedError(ValueError):
    """The hint names a binding the predicates reject; nothing is bound from it."""

    def __init__(self, failures: Sequence[Failure]) -> None:
        self.failures = tuple(failures)
        super().__init__("hint rejected: " + "; ".join(map(describe_failure, self.failures)))


class BoundEntry(BaseModel):
    """``view`` is the view a model proposal was verified over; the binding names the landed
    relation, never the view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    binding: SlotBinding
    confidence: float = Field(default=CODE_BOUND_CONFIDENCE, ge=0.0, le=1.0)
    source: BindingSource
    call_key: str | None = None
    view: ViewSpec | None = None


class StructureResult(BaseModel):
    """``groups`` lists every relation, ignored ones included, in landing order.

    ``proposal_targets`` are the relations a layout proposal may bind: ``no_template``
    relations and relations ignored as ``no_header``, in landing order.

    ``ambiguities`` lists open entries only; ``settled`` keeps each entry a model or a
    reviewer settled, with its choice, confidence, source, and call key.

    ``described_by`` maps each relation a bound dictionary describes to that dictionary;
    the described-data gate kept it out of every dictionary and header slot (S5 decision 23).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    order: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    signatures: dict[str, RelationSignature]
    bound: tuple[BoundEntry, ...]
    ignored: tuple[SlotBinding, ...]
    ambiguities: tuple[Ambiguity, ...]
    edges: tuple[JoinEdge, ...]
    floors: dict[str, FloorValue]
    proposal_targets: tuple[str, ...] = ()
    settled: tuple[Ambiguity, ...] = ()
    described_by: dict[str, str] = {}

    def with_settled(self) -> StructureResult:
        """Entries carrying a choice or a source leave the open list for ``settled``."""
        moving = [a.choice is not None or a.source is not None for a in self.ambiguities]
        if not any(moving):
            return self
        pairs = list(zip(self.ambiguities, moving, strict=True))
        update = {
            "ambiguities": tuple(a for a, moved in pairs if not moved),
            "settled": (*self.settled, *(a for a, moved in pairs if moved)),
        }
        result: StructureResult = self.model_copy(update=update)
        return result

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
    bound, ambiguities, described = _bind_described(pinned, graph, by_name, tools)
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
        proposal_targets=_proposal_targets(profiles, ignored, ambiguities),
        described_by=described,
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


def stage_floor_classes() -> dict[str, FloorClass]:
    """The class of every constant ``stage_floors`` lists; one name has one class."""
    classes: dict[str, FloorClass] = dict(CLASSES)
    for template in TEMPLATES:
        for name, found in getattr(template, "CLASSES", {}).items():
            if classes.setdefault(name, found) != found:
                raise ValueError(f"{name} is classed both {classes[name]} and {found}")
    return classes


def _proposal_targets(
    profiles: Sequence[RelationProfile],
    ignored: Sequence[SlotBinding],
    ambiguities: Sequence[Ambiguity],
) -> tuple[str, ...]:
    targets = {a.subject for a in ambiguities if a.members == (NO_TEMPLATE,)}
    targets |= {b.relations[0] for b in ignored if b.slots.get(ignore.REASON) == ignore.NO_HEADER}
    return tuple(profile.ref.name for profile in profiles if profile.ref.name in targets)


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


def _bind_described(
    pinned: Sequence[SlotBinding],
    graph: _Graph,
    by_name: dict[str, RelationProfile],
    tools: ProfileTools,
) -> tuple[list[BoundEntry], list[Ambiguity], dict[str, str]]:
    """Bind with described data out of every dictionary and header slot.

    Only a dictionary that ends up bound may describe a relation, so a describing dictionary
    left unbound returns what it described to the candidates and the groups bind again.
    """
    eligible = [[b for b in g if not described_beside(b, by_name, tools)] for g in graph.candidates]
    graph = replace(graph, candidates=eligible)
    found = [binding for group in graph.candidates for binding in group]
    dictionaries = [by_name[name] for name in per_file_dictionaries([*pinned, *found])]
    targets = [by_name[name] for name in schema_slot_relations(found)]
    described = described_by(dictionaries, targets, tools)
    while True:
        candidates = [[b for b in g if not uses_described(b, described)] for g in graph.candidates]
        bound, ambiguities = _bind(pinned, replace(graph, candidates=candidates))
        in_force = set(per_file_dictionaries(entry.binding for entry in bound))
        kept = {name: by for name, by in described.items() if by in in_force}
        if kept == described:
            return bound, ambiguities, described
        described = kept


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
