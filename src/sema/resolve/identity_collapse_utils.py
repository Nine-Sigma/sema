"""S1-10 — deterministic identity-collapse planning (generic, R29-scanned).

Computes which registry rows should share one canonical ``entity_id``. Two source
entities collapse ONLY when they carry the exact same source key within the SAME
identity namespace — the safe, high-precision, no-probability case (D3). The
grouping of ``source_namespace`` → identity namespace arrives as data (an
institution may spread patients across several study schemas), so this module
names no domain literal and never guesses across namespaces.

Determinism (replay-safe): the survivor is the MIN ``entity_id`` in a group, so
collapse only ever lowers ids, never mints new ones, and a re-run over an
already-collapsed registry produces an empty plan.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sema.resolve.identity_registry_utils import IdentityAssignment

SourceKey = tuple[str, str]


def identity_namespace_of(
    source_namespace: str, namespace_grouping: Mapping[str, str]
) -> str:
    """Map a source namespace to its identity namespace (itself, if ungrouped).

    The default — an ungrouped source namespace is its own identity namespace —
    is the safe one: an un-configured study can never collapse with another.
    """
    return namespace_grouping.get(source_namespace, source_namespace)


@dataclass(frozen=True)
class CollapseGroup:
    """One set of source entities that resolve to a single canonical id."""

    identity_namespace: str
    source_entity_key: str
    survivor_entity_id: int
    retired: tuple[tuple[SourceKey, int], ...]

    @property
    def retired_entity_ids(self) -> tuple[int, ...]:
        return tuple(entity_id for _, entity_id in self.retired)


@dataclass(frozen=True)
class CollapsePlan:
    """The deterministic set of entity_id remaps a collapse would apply."""

    groups: tuple[CollapseGroup, ...]

    @property
    def remaps(self) -> dict[SourceKey, int]:
        return {
            grain: group.survivor_entity_id
            for group in self.groups
            for grain, _ in group.retired
        }

    @property
    def retired_entity_ids(self) -> tuple[int, ...]:
        return tuple(
            entity_id for group in self.groups for entity_id in group.retired_entity_ids
        )

    @property
    def collapsed_person_count(self) -> int:
        """How many canonical ids are retired (i.e. persons removed by dedup)."""
        return len(self.retired_entity_ids)

    @property
    def remapped_row_count(self) -> int:
        return len(self.remaps)


def compute_collapse_plan(
    rows: Sequence[IdentityAssignment], *, namespace_grouping: Mapping[str, str]
) -> CollapsePlan:
    """Group registry rows by ``(identity_namespace, key)`` into collapse groups.

    Only buckets holding ≥2 distinct ``entity_id``s become groups; the survivor is
    the min id and every other member is remapped onto it. Cross-namespace pairs
    land in different buckets by construction, so they can never collapse.
    """
    buckets: dict[SourceKey, list[IdentityAssignment]] = defaultdict(list)
    for row in rows:
        namespace = identity_namespace_of(row.source_namespace, namespace_grouping)
        buckets[(namespace, row.source_entity_key)].append(row)

    groups = tuple(
        group
        for bucket_key, members in sorted(buckets.items())
        if (group := _group_from_bucket(bucket_key, members)) is not None
    )
    return CollapsePlan(groups=groups)


def _group_from_bucket(
    bucket_key: SourceKey, members: Sequence[IdentityAssignment]
) -> CollapseGroup | None:
    if len({m.entity_id for m in members}) < 2:
        return None
    survivor = min(m.entity_id for m in members)
    retired = tuple(
        ((m.source_namespace, m.source_entity_key), m.entity_id)
        for m in sorted(members, key=lambda m: (m.source_namespace, m.source_entity_key))
        if m.entity_id != survivor
    )
    identity_namespace, source_entity_key = bucket_key
    return CollapseGroup(
        identity_namespace=identity_namespace,
        source_entity_key=source_entity_key,
        survivor_entity_id=survivor,
        retired=retired,
    )
