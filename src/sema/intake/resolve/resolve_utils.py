"""Structure resolution helpers: entries to send, member lookups, choices, the prompt, and
the values a prompt shows."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from string import Template
from typing import Any, TypeVar

from sema.binding.spec_sections import RelationRef
from sema.intake.resolve.output_models_utils import ABSTAIN, Choice
from sema.intake.land.profile import ProfileTools
from sema.intake.land.profile_utils import ColumnProfile, RowSample, ValueSample
from sema.intake.structure.shapes.base import RelationProfile, SlotBinding, is_text
from sema.intake.structure.shapes.precedence import STAGE, member
from sema.intake.structure.structure import BindingSource, BoundEntry, StructureResult
from sema.intake.structure.structure_utils import NO_TEMPLATE, describe_failure, group_candidates
from sema.intake.verify.base import Ambiguity, Failure

PROMPT_PACKAGE = "sema.intake.resolve"
STRUCTURE_PROMPT = "prompts/structure_v1.md"
MODEL_SOURCE: BindingSource = "model"
NOTHING = "none"
HUMAN = "human"
FIRST_ROWS_HEADING = "## First rows"

Bindings = dict[str, dict[str, SlotBinding]]
SchemaText = dict[str, list[tuple[object, ...]]]
Attempt = TypeVar("Attempt")


@dataclass(frozen=True)
class OpenEntry:
    """``position`` indexes the structure's ambiguity list; ``bindings`` maps member to binding."""

    position: int
    ambiguity: Ambiguity
    bindings: dict[str, SlotBinding]

    def spanned(self) -> tuple[str, ...]:
        names = [self.ambiguity.subject]
        names += [name for b in self.bindings.values() for name in b.relations]
        return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class Settlement:
    """One parsed answer: updated ambiguities by position and the bindings chosen by member."""

    key: str
    updates: dict[int, Ambiguity]
    chosen: dict[str, BoundEntry]

    def spanned(self) -> set[str]:
        return {name for entry in self.chosen.values() for name in entry.binding.relations}


def member_bindings(
    structure: StructureResult, profiles: Sequence[RelationProfile], tools: ProfileTools
) -> Bindings:
    """Each relation's candidate bindings by member, recomputed as the verifier does."""
    removed = {n for e in structure.bound if e.source == HUMAN for n in e.binding.relations}
    removed |= {name for binding in structure.ignored for name in binding.relations}
    by_name = {profile.ref.name: profile for profile in profiles}
    found: Bindings = {}
    for group in structure.groups:
        members = [by_name[n] for n in group if n in by_name and n not in removed]
        for binding in group_candidates(members, tools):
            for name in binding.relations:
                found.setdefault(name, {})[member(binding)] = binding
    return found


def open_entries(structure: StructureResult, found: Bindings) -> list[OpenEntry]:
    """Open structure entries whose members are bindings; `no_template` goes to layouts."""
    entries = []
    for position, ambiguity in enumerate(structure.ambiguities):
        if ambiguity.stage != STAGE or ambiguity.choice is not None:
            continue
        if ambiguity.members == (NO_TEMPLATE,):
            continue
        candidates = found.get(ambiguity.subject, {})
        missing = [m for m in ambiguity.members if m not in candidates]
        if missing:
            raise ValueError(
                f"structure entry {ambiguity.subject} lists members the landed relations"
                f" do not produce: {missing}"
            )
        bindings = {m: candidates[m] for m in ambiguity.members}
        entries.append(OpenEntry(position, ambiguity, bindings))
    return entries


def settle_choices(batch: Sequence[OpenEntry], choices: Sequence[Choice], key: str) -> Settlement:
    updates: dict[int, Ambiguity] = {}
    chosen: dict[str, BoundEntry] = {}
    for entry, (choice, confidence, reason) in zip(batch, choices, strict=True):
        if choice == ABSTAIN:
            updates[entry.position] = entry.ambiguity.model_copy(update={"reason": reason})
            continue
        fields = dict(
            choice=choice, confidence=confidence, source=MODEL_SOURCE, call_key=key, reason=reason
        )
        updates[entry.position] = entry.ambiguity.model_copy(update=fields)
        binding = entry.bindings[choice]
        bound = BoundEntry(
            binding=binding, confidence=confidence, source=MODEL_SOURCE, call_key=key
        )
        chosen.setdefault(choice, bound)
    return Settlement(key, updates, chosen)


def reopened(settlement: Settlement, failures: Sequence[Failure]) -> Settlement:
    """Every choice spanning a failing relation goes back to open, with the failure as reason."""
    failing = {found.relation for found in failures}
    dropped = {
        choice: entry
        for choice, entry in settlement.chosen.items()
        if failing & set(entry.binding.relations)
    }
    updates = dict(settlement.updates)
    for position, ambiguity in settlement.updates.items():
        if ambiguity.choice not in dropped:
            continue
        spans = set(dropped[ambiguity.choice].binding.relations)
        notes = [describe_failure(f) for f in failures if f.relation in spans]
        reason = "; ".join(dict.fromkeys(notes))
        updates[position] = open_with(ambiguity, reason)
    chosen = {c: e for c, e in settlement.chosen.items() if c not in dropped}
    return Settlement(settlement.key, updates, chosen)


def open_with(ambiguity: Ambiguity, reason: str) -> Ambiguity:
    fields = dict(choice=None, confidence=None, source=None, call_key=None, reason=reason)
    return ambiguity.model_copy(update=fields)


def entry_reason(problem: str, error: str | None) -> str:
    """A failed call's error text is kept in the reason and the ledger, never in a prompt,
    where it would make the next key differ between runs."""
    return problem if error is None else f"{problem}: {error}"


def failure_notes(
    batch: Sequence[OpenEntry], settlement: Settlement, failures: Sequence[Failure]
) -> list[str]:
    """Each failed check with the choices of this batch that span its relation."""
    notes = []
    for found in failures:
        spanning = [
            f"entry {index} chose `{update.choice}`"
            for index, entry in enumerate(batch)
            if (update := settlement.updates.get(entry.position)) is not None
            and update.choice is not None
            and found.relation in settlement.chosen[update.choice].binding.relations
        ]
        note = describe_failure(found)
        notes.append(f"{note}; {', '.join(spanning)}" if spanning else note)
    return list(dict.fromkeys(notes))


def retried(
    ask: Callable[[Sequence[Sequence[str]]], Attempt],
    notes: Callable[[Attempt], list[str]],
    max_retries: int,
    *,
    ended: Callable[[Attempt], bool],
) -> tuple[Attempt, bool]:
    """Ask, then re-ask with every earlier attempt's notes until one has none; the last
    attempt and whether it passed. An attempt passes when ``notes`` returns nothing.

    An ``ended`` attempt (a failed or refused call, not a choice that failed a check) is
    not re-asked: a second call cannot answer what the first could not, and a refused
    one would be refused again."""
    failed: list[list[str]] = []
    for _ in range(max_retries):
        attempt = ask(failed)
        if ended(attempt):
            return attempt, False
        found = notes(attempt)
        if not found:
            return attempt, True
        failed.append(found)
    attempt = ask(failed)
    return attempt, not ended(attempt) and not notes(attempt)


def render_template(prompt: str, sections: Mapping[str, str]) -> str:
    template = resources.files(PROMPT_PACKAGE).joinpath(prompt).read_text("utf-8")
    return Template(template).substitute(sections)


def render_prompt(
    batch: Sequence[OpenEntry],
    in_force: Sequence[BoundEntry],
    relations: str,
    failed: Sequence[Sequence[str]],
) -> str:
    sections = {
        "in_force": bullets(f"`{member(e.binding)}` ({e.source})" for e in in_force),
        "failures": history(failed),
        "relations": relations,
        "entries": "\n\n".join(_entry(index, entry) for index, entry in enumerate(batch)),
    }
    return render_template(STRUCTURE_PROMPT, sections)


def bullets(lines: Iterable[str]) -> str:
    rendered = [f"- {line}" for line in lines]
    return "\n".join(rendered) if rendered else NOTHING


@dataclass(frozen=True)
class ShownValues:
    """What a prompt shows of one relation's values: distinct values per column, then the
    first landed rows of the columns that showed any (S5 decision 5)."""

    lines: tuple[str, ...] = ()
    rows: RowSample | None = None

    def values(self) -> str:
        return bullets(self.lines)

    @property
    def has_rows(self) -> bool:
        return self.rows is not None and bool(self.rows.rows)

    def first_rows(self) -> str:
        """The block after Values; empty with no rows, so no heading stands without a value."""
        if self.rows is None or not self.has_rows:
            return ""
        columns = " | ".join(self.rows.columns)
        lead = f"The first {len(self.rows.rows)} rows in landed order, as {columns}:"
        table = bullets(" | ".join(map(_cell, row)) for row in self.rows.rows)
        return f"\n\n{FIRST_ROWS_HEADING}\n\n{lead}\n\n{table}"


def shown_values(
    tools: ProfileTools, ref: RelationRef, columns: Sequence[str], n: int
) -> ShownValues:
    """Values of the columns the policy approves; a withheld column issues no statement."""
    approved = [column for column in columns if tools.values_approved(ref, column)]
    samples = [
        sample
        for column in approved
        if isinstance(sample := tools.sample_values(ref, column, n), ValueSample) and sample.values
    ]
    lines = tuple(f"{s.column}: {', '.join(map(str, s.values))}" for s in samples)
    shown = [sample.column for sample in samples]
    rows = tools.sample_rows(ref, shown, n) if shown else None
    return ShownValues(lines, rows if isinstance(rows, RowSample) else None)


def _cell(value: Any) -> str:
    return "null" if value is None else str(value)


def history(failed: Sequence[Sequence[str]]) -> str:
    """Every earlier answer's failed checks, oldest first."""
    answers = [f"answer {n}:\n{bullets(notes)}" for n, notes in enumerate(failed, start=1)]
    return "\n\n".join(answers) or NOTHING


def render_relations(
    relations: Sequence[RelationProfile],
    schema_text: Mapping[str, Sequence[tuple[object, ...]]],
    *,
    hidden: Mapping[str, int] | None = None,
    kept: Mapping[str, Collection[str]] | None = None,
) -> str:
    """``hidden`` counts each relation's schema rows not shown; with ``kept``, only the
    columns it lists for a relation keep their facts and the rest are named."""
    return "\n\n".join(
        _relation(
            p,
            schema_text.get(p.ref.name),
            (hidden or {}).get(p.ref.name, 0),
            None if kept is None else kept.get(p.ref.name, ()),
        )
        for p in relations
    )


def schema_text(tools: ProfileTools, relations: Sequence[RelationProfile]) -> SchemaText:
    """Text of the bound schema relations among ``relations``, only when the run may read it."""
    readable = tools.structure.schema_relations if tools.structure is not None else ()
    return {
        profile.ref.name: tools.read_schema_text(profile.ref, text_columns(profile))
        for profile in relations
        if profile.ref.name in readable and text_columns(profile)
    }


def _entry(index: int, entry: OpenEntry) -> str:
    ambiguity = entry.ambiguity
    members = [
        f"- `{m}` spans {', '.join(entry.bindings[m].relations)}" for m in ambiguity.members
    ]
    exclusions = [f"{name}: {why}" for name, why in sorted(ambiguity.exclusions.items())]
    return "\n".join(
        [
            f"### entry {index}: {ambiguity.subject}",
            "members:",
            *members,
            f"hints:\n{bullets(ambiguity.hints)}" if ambiguity.hints else f"hints: {NOTHING}",
            f"exclusions: {'; '.join(exclusions) or NOTHING}",
        ]
    )


def _relation(
    profile: RelationProfile,
    schema_rows: Sequence[tuple[object, ...]] | None,
    hidden: int = 0,
    kept: Collection[str] | None = None,
) -> str:
    lines = [f"### {profile.ref.name}", f"rows: {profile.row_count}", "columns:"]
    keep = None if kept is None else set(kept)
    lines += [f"- {_column(c)}" for c in profile.columns if keep is None or c.name in keep]
    if named := [c.name for c in profile.columns if keep is not None and c.name not in keep]:
        lines.append(
            f"named only, {len(named)} columns no open entry offers and no claim holds:"
            f" {', '.join(named)}"
        )
    if schema_rows:
        lines.append("schema text:")
        lines += [f"- {' | '.join(map(str, row))}" for row in schema_rows]
    if hidden:
        lines.append(
            f"{hidden} more rows not shown; the rows above come first in order of the text"
            " columns"
        )
    return "\n".join(lines)


def _column(column: ColumnProfile) -> str:
    facts = [
        f"type {column.type}",
        f"distinct {column.distinct_count}",
        f"distinct ratio {column.distinct_ratio:.4g}",
        f"null rate {column.null_rate:.4g}",
    ]
    if column.mean_length is not None:
        facts.append(f"mean length {column.mean_length:.4g}")
    flags = (("unique", column.is_unique), ("constant", column.is_constant))
    facts += [flag for flag, on in flags if on]
    if column.comment:
        facts.append(f"comment {column.comment!r}")
    return f"{column.name}: {', '.join(facts)}"


def text_columns(profile: RelationProfile) -> list[str]:
    return [column.name for column in profile.columns if is_text(column)]
