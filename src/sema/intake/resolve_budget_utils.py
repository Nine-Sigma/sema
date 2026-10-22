"""Prompt budget: the evidence a resolve prompt shows, compacted only while it is over budget.

Compaction runs in a stated order, one level at a time, and stops at the first rendering
that fits ``MAX_PROMPT_TOKENS``; level 0 is the full rendering, so a prompt under the
budget is unchanged. Level 1 states an entry offering every column of its relation but its
exclusions instead of listing them. Level 2 shows each schema relation's text read in order
of its text columns, at most ``MAX_SCHEMA_TEXT_ROWS`` rows, and counts the rows not shown.
Level 3 keeps full facts only for columns an open entry offers or a claim in force holds
and names the rest. A rendering still over budget is returned as is; ``ModelCalls``
refuses it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from sema.intake.model_calls import fits
from sema.intake.profile import ProfileTools
from sema.intake.resolve_utils import (
    SchemaText,
    render_relations,
    schema_text,
    text_columns,
)
from sema.intake.shapes.base import RelationProfile
from sema.log import logger

# Policy: the rows of one schema relation a compacted prompt shows; not fixture-derived.
MAX_SCHEMA_TEXT_ROWS = 1000
STATEMENTS, CAPPED_TEXT, NAMED_ONLY = 1, 2, 3
LEVELS = (0, STATEMENTS, CAPPED_TEXT, NAMED_ONLY)


@dataclass
class PromptEvidence:
    """The relations one ask shows; their full schema text is read once, at creation."""

    tools: ProfileTools
    relations: Sequence[RelationProfile]
    texts: SchemaText = field(init=False)
    _capped: SchemaText | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.texts = schema_text(self.tools, self.relations)

    def render(self, level: int, kept: Mapping[str, Collection[str]] | None = None) -> str:
        if level < CAPPED_TEXT:
            return render_relations(self.relations, self.texts)
        capped = self._capped_text()
        hidden = {
            p.ref.name: p.row_count - len(capped[p.ref.name])
            for p in self.relations
            if p.ref.name in capped
        }
        named = kept if level >= NAMED_ONLY else None
        return render_relations(self.relations, capped, hidden=hidden, kept=named)

    def _capped_text(self) -> SchemaText:
        if self._capped is None:
            by_name = {p.ref.name: p for p in self.relations}
            self._capped = {
                name: self.tools.read_schema_text(
                    by_name[name].ref, text_columns(by_name[name]), MAX_SCHEMA_TEXT_ROWS
                )
                for name in self.texts
            }
        return self._capped


def fitted(render: Callable[[int], str], output_model: type[BaseModel]) -> str:
    """The first level's rendering that fits the budget, else the most compacted one."""
    for level in LEVELS:
        prompt = render(level)
        if fits(prompt, output_model):
            break
    if level:
        logger.info(f"{output_model.__name__} prompt compacted to level {level} of the budget")
    return prompt
