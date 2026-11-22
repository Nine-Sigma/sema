"""Rows of the code-domains report, from the build graph's evidence and value-set links.

A binding is one (column, proposed value set) pair the `has_value_set` assertions propose; a
link is one `HAS_VALUE_SET` edge with its members. Each link is its own row, carrying the
evidence of the binding the build linked on its column; each binding the build did not link
is a row with the numbers its containment assertion recorded and the reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

BUILD_GRAPH = "build_graph"
CONTAINMENT = "containment"


@dataclass(frozen=True)
class Evidence:
    """One `has_value_set` assertion: its column as `table.column`, None when unplaced."""

    column: str | None
    value_set: str
    kind: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class Member:
    code: str
    label: str | None
    rows: int | None
    listed: bool | None


@dataclass(frozen=True)
class GraphLink:
    column: str
    value_set: str
    props: Mapping[str, Any]
    members: tuple[Member, ...]


@dataclass(frozen=True)
class Row:
    column: str | None
    value_set: str
    evidence: tuple[str, ...]
    linked: bool
    verification: str | None
    labels: bool
    listed_value_share: float | None
    listed_row_share: float | None
    null_rows: int | None
    reason: str


@dataclass
class Binding:
    column: str | None
    value_set: str
    kinds: set[str] = field(default_factory=set)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def linked(self) -> bool:
        return bool(self.payload.get("linked"))

    def reasons(self) -> list[str]:
        prop = self.payload.get("property")
        reasons = [f"property {prop!r}"] if prop and self.column is None else []
        if self.payload.get("reason"):
            reasons.append(str(self.payload["reason"]))
        if self.payload.get("alternatives"):
            reasons.append("alternatives: " + ", ".join(self.payload["alternatives"]))
        return reasons


def bindings(evidence: Iterable[Evidence]) -> list[Binding]:
    """One binding per column and value set; each unplaced property stays its own."""
    found: dict[tuple[Any, ...], Binding] = {}
    for e in evidence:
        unplaced = e.payload.get("property") if e.column is None else None
        binding = found.setdefault((e.column, e.value_set, unplaced),
                                   Binding(e.column, e.value_set))
        binding.kinds.add(e.kind)
        if e.kind == CONTAINMENT or not binding.payload:
            binding.payload = dict(e.payload)
    return list(found.values())


def _listed(member: Member) -> bool:
    return member.listed is not False


def member_shares(members: tuple[Member, ...]) -> tuple[float | None, float | None]:
    """Shares of observed values and rows the value set lists, when members carry rows."""
    if any(m.rows is None for m in members):
        return None, None
    observed = [m for m in members if m.rows]
    total = sum(m.rows or 0 for m in observed)
    if not observed or not total:
        return None, None
    listed = [m for m in observed if _listed(m)]
    return len(listed) / len(observed), sum(m.rows or 0 for m in listed) / total


def labeled(members: tuple[Member, ...]) -> bool:
    listed = [m for m in members if _listed(m)]
    return bool(listed) and all(m.label for m in listed)


def link_row(link: GraphLink, binding: Binding | None) -> Row:
    value_share, row_share = member_shares(link.members)
    listed = [m for m in link.members if _listed(m)]
    reasons = [] if link.members else [f"value set {link.value_set} has no members"]
    if link.members and not labeled(link.members):
        named = sum(1 for m in listed if m.label)
        reasons.append(f"{named} of {len(listed)} listed members carry a label")
    reasons += binding.reasons() if binding else []
    return Row(
        column=link.column,
        value_set=binding.value_set if binding else link.value_set,
        evidence=tuple(sorted(binding.kinds)) if binding else (BUILD_GRAPH,),
        linked=bool(link.members),
        verification=link.props.get("verification"),
        labels=labeled(link.members),
        listed_value_share=link.props.get("listed_value_share", value_share),
        listed_row_share=link.props.get("listed_row_share", row_share),
        null_rows=link.props.get("null_rows"),
        reason="; ".join(reasons),
    )


def binding_row(binding: Binding) -> Row:
    numbers = binding.payload
    return Row(
        column=binding.column,
        value_set=binding.value_set,
        evidence=tuple(sorted(binding.kinds)),
        linked=False,
        verification=numbers.get("verification"),
        labels=False,
        listed_value_share=numbers.get("listed_value_share"),
        listed_row_share=numbers.get("listed_row_share"),
        null_rows=numbers.get("null_rows"),
        reason="; ".join([*binding.reasons(), "no HAS_VALUE_SET"]),
    )
