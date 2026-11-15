"""The Roles prompt (roles v2): one relation's evidence, compacted in a stated order.

Every column of the asked relation is listed with its facts: aggregates, comment, the
roles whose lexicon tokens its name matches, its embedding rank per role when an embedder
ran, and its findings (S5 decisions 1, 3, 18). A schema relation shows a fixed sample of
``SCHEMA_TEXT_SAMPLE_ROWS`` rows with the rows not shown (decision 6). The asked relation's
approved columns show their values and first landed rows (decision 5). Over the budget,
the order is: comments are truncated, then the schema sample shrinks to its first rows,
then the first rows are dropped, then columns keep only name and type; each omission is
stated in the prompt, and a step with nothing to drop is not taken.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from pydantic import BaseModel

from sema.intake import lexicon
from sema.intake.eligibility_utils import RoleEligibility, column_findings, lexicon_hits
from sema.intake.model_calls import fits
from sema.intake.profile import SAMPLE_CAP, ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.resolve_utils import (
    NOTHING,
    SchemaText,
    ShownValues,
    bullets,
    history,
    render_template,
    shown_values,
    text_columns,
)
from sema.intake.shapes.base import RelationProfile
from sema.intake.verify.base import stated_findings
from sema.log import logger

ROLES_PROMPT = "prompts/roles_v2.md"
# Policy, not fixture-derived: the schema rows one role-map prompt shows per relation.
SCHEMA_TEXT_SAMPLE_ROWS = 20
# Policy: the compacted schema sample and comment length.
SHRUNK_SCHEMA_ROWS = 5
COMMENT_CHARS = 80


@dataclass(frozen=True)
class Compaction:
    """One step of the compaction order; level 0 is the full rendering."""

    comment_chars: int | None = None
    schema_rows: int = SCHEMA_TEXT_SAMPLE_ROWS
    first_rows: bool = True
    full_facts: bool = True

    def omissions(self) -> list[str]:
        stated = []
        if self.comment_chars is not None:
            stated.append(f"comments are truncated to {self.comment_chars} characters")
        if self.schema_rows < SCHEMA_TEXT_SAMPLE_ROWS:
            stated.append(f"schema text shows its first {self.schema_rows} rows")
        if not self.first_rows:
            stated.append("the first rows of the approved columns are not shown")
        if not self.full_facts:
            stated.append("columns show only their name and type")
        return stated


COMPACTION_ORDER = (
    Compaction(),
    Compaction(COMMENT_CHARS),
    Compaction(COMMENT_CHARS, SHRUNK_SCHEMA_ROWS),
    Compaction(COMMENT_CHARS, SHRUNK_SCHEMA_ROWS, first_rows=False),
    Compaction(COMMENT_CHARS, SHRUNK_SCHEMA_ROWS, first_rows=False, full_facts=False),
)


@dataclass
class RoleEvidence:
    """The relations one ask shows; the asked relation's columns carry role facts from
    ``entries``. The schema sample and the shown values are read once, at creation."""

    tools: ProfileTools
    relations: Sequence[RelationProfile]
    asked: str
    entries: Sequence[RoleEligibility]
    texts: SchemaText = field(init=False)
    shown: ShownValues = field(init=False)

    def __post_init__(self) -> None:
        readable = self.tools.structure.schema_relations if self.tools.structure else ()
        self.texts = {
            p.ref.name: self.tools.read_schema_text(
                p.ref, text_columns(p), SCHEMA_TEXT_SAMPLE_ROWS
            )
            for p in self.relations
            if p.ref.name in readable and text_columns(p)
        }
        asked = next((p for p in self.relations if p.ref.name == self.asked), None)
        columns = [column.name for column in asked.columns] if asked else []
        self.shown = (
            shown_values(self.tools, asked.ref, columns, SAMPLE_CAP) if asked else ShownValues()
        )

    def first_rows(self, compaction: Compaction) -> str:
        return self.shown.first_rows() if compaction.first_rows else ""

    def width(self, relation: str) -> int:
        return next((len(p.columns) for p in self.relations if p.ref.name == relation), 0)

    def render(self, compaction: Compaction, *, shuffled: bool = False) -> str:
        """``shuffled`` lists each relation's columns in its seeded order (S5 decision 4)."""
        return "\n\n".join(self._relation(p, compaction, shuffled) for p in self.relations)

    def _relation(self, profile: RelationProfile, compaction: Compaction, shuffled: bool) -> str:
        lines = [f"### {profile.ref.name}", f"rows: {profile.row_count}", "columns:"]
        roles = self._roles() if profile.ref.name == self.asked else ()
        ranks = _ranks(self.entries) if roles else {}
        for column in seeded_order(profile) if shuffled else profile.columns:
            facts = _facts(column, compaction)
            if roles and compaction.full_facts:
                facts += _role_facts(column, roles, ranks.get(column.name, ()))
            lines.append(f"- {column.name}: {'; '.join(facts)}")
        rows = self.texts.get(profile.ref.name, [])[: compaction.schema_rows]
        if rows:
            lines.append("schema text:")
            lines += [f"- {' | '.join(map(str, row))}" for row in rows]
        hidden = profile.row_count - len(rows)
        if rows and hidden:
            lines.append(f"{hidden} more rows not shown; the rows above come first in text order")
        return "\n".join(lines)

    def _roles(self) -> tuple[str, ...]:
        known = {entry.name for entry in lexicon.ROLES}
        asked = {e.role for e in self.entries if e.relation == self.asked and e.subject is None}
        return tuple(name for name in (e.name for e in lexicon.ROLES) if name in asked & known)


def seeded_order(profile: RelationProfile) -> tuple[ColumnProfile, ...]:
    """The columns permuted by a seed from the relation signature's digest, so a relation
    gets the same order on every run; a permutation that leaves the order as listed is
    reversed, because the second ask must differ from the first to be a second reading."""
    digest = hashlib.sha256(json.dumps(profile.signature).encode()).digest()
    columns = list(profile.columns)
    random.Random(int.from_bytes(digest, "big")).shuffle(columns)
    ordered = tuple(columns)
    return ordered[::-1] if ordered == profile.columns else ordered


def _facts(column: ColumnProfile, compaction: Compaction) -> list[str]:
    if not compaction.full_facts:
        return [f"type {column.type}"]
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
        limit = compaction.comment_chars
        comment = column.comment if limit is None else column.comment[:limit]
        facts.append(f"comment {comment!r}")
    return [", ".join(facts)]


def _role_facts(column: ColumnProfile, roles: Sequence[str], ranks: Sequence[str]) -> list[str]:
    found = []
    if hits := [role for role in roles if lexicon_hits(role, [column.name])]:
        found.append(f"lexicon tokens of {', '.join(hits)}")
    if ranks:
        found.append(f"embedding {', '.join(ranks)}")
    if readings := column_findings(column):
        found.append(stated_findings(readings))
    return found


def _ranks(entries: Iterable[RoleEligibility]) -> dict[str, list[str]]:
    """Per column, each role an embedder ranks it first for, with the margin over the
    second; a tie at the top separates nothing, so it is not stated."""
    ranks: dict[str, list[str]] = {}
    for entry in entries:
        for hint in entry.hints:
            if hint.source != "embedding" or not hint.rank or not hint.margin:
                continue
            (name, score), count = hint.rank[0], len(hint.rank)
            text = (
                f"{entry.role} rank 1 of {count}, score {score:.3f},"
                f" margin {hint.margin:.3f} over the second"
            )
            ranks.setdefault(name, []).append(text)
    return ranks


def render_roles_prompt(
    relation: str,
    claims: str,
    evidence: str,
    fields: str,
    failed: Sequence[Sequence[str]],
    *,
    values: str = NOTHING,
    first_rows: str = "",
) -> str:
    sections = {
        "relation": relation,
        "in_force": claims,
        "failures": history(failed),
        "relations": evidence,
        "samples": values,
        "first_rows": first_rows,
        "fields": fields,
    }
    return render_template(ROLES_PROMPT, sections)


def fitted(
    render: Callable[[Compaction], str], output_model: type[BaseModel], *, rows: bool = False
) -> str:
    """The first step's rendering that fits the budget, else the most compacted one, with
    each omission stated; ``ModelCalls`` refuses a prompt that still does not fit. With no
    ``rows`` shown, the step that drops them is not taken."""
    order: tuple[Compaction, ...] = COMPACTION_ORDER
    if not rows:
        order = tuple(dict.fromkeys(replace(c, first_rows=True) for c in COMPACTION_ORDER))
    for level, compaction in enumerate(order):
        prompt = render(compaction)
        omitted = compaction.omissions()
        if omitted:
            prompt += "\n\n## Omitted to fit the budget\n\n" + bullets(omitted)
        if fits(prompt, output_model):
            break
    if level:
        logger.info(f"{output_model.__name__} prompt compacted to step {level} of the order")
    return prompt


def claim_lines(claims: Iterable[RoleEligibility]) -> str:
    return bullets(_claim(entry) for entry in claims)


def _claim(entry: RoleEligibility) -> str:
    contest = "; a claim you may contest" if entry.source == "lexicon" else ""
    return f"`{entry.role}`: `{entry.choice}` ({entry.source}{contest})"


def field_lines(described: Mapping[str, str]) -> str:
    return bullets(f"`{name}`: {text}" for name, text in described.items())
