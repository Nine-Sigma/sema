"""Layout proposal helpers: the output model, the slot binding, the prompt, structure edits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from sema.intake.output_models import LITERAL_PREFIX, SLOT_PREFIX, layout_model
from sema.intake.output_models_utils import NONE
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ValueSample
from sema.intake.resolve_utils import bullets, history, render_relations, render_template
from sema.intake.shapes import TEMPLATES, ignore
from sema.intake.shapes.base import RelationProfile, Slot, SlotBinding, Template
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.structure_utils import NO_TEMPLATE
from sema.intake.views import VIEWS

LAYOUT_PROMPT = "prompts/layout_v1.md"
VALUE_SAMPLE = 20


def proposable() -> tuple[Template, ...]:
    """Templates that declare slots; one that declares none cannot be proposed."""
    return tuple(t for t in TEMPLATES if getattr(t, "SLOTS", ()))


def slots_of(template: str) -> tuple[Slot, ...]:
    found = next((t for t in TEMPLATES if t.ID == template), None)
    return tuple(getattr(found, "SLOTS", ()))


def proposal_model(profile: RelationProfile, group: Sequence[str]) -> type[BaseModel]:
    """The closed sets one relation's proposal may name: templates, views, columns, group."""
    declared = [slot for template in proposable() for slot in getattr(template, "SLOTS")]
    literals: dict[str, list[str]] = {}
    for slot in (s for s in declared if s.kind == "literal"):
        literals.setdefault(slot.name, []).extend(slot.values)
    return layout_model(
        templates=[template.ID for template in proposable()],
        views=[view.ID for view in VIEWS],
        columns=[column.name for column in profile.columns],
        relations=list(group),
        slots=list(dict.fromkeys(s.name for s in declared if s.kind == "column")),
        literals=literals,
    )


def answer_fields(answer: BaseModel, prefix: str) -> dict[str, str]:
    """The answer's ``<prefix><slot>`` fields by slot, left out when ``none``."""
    values: dict[str, Any] = answer.model_dump()
    return {
        name.removeprefix(prefix): str(value)
        for name, value in values.items()
        if name.startswith(prefix) and value != NONE
    }


def binding_of(answer: BaseModel) -> SlotBinding:
    relations: list[str] = getattr(answer, "relations")
    columns = answer_fields(answer, SLOT_PREFIX)
    literals = answer_fields(answer, LITERAL_PREFIX)
    return proposed_binding(str(getattr(answer, "template")), relations, columns, literals)


def proposed_binding(
    template: str,
    relations: Sequence[str],
    columns: Mapping[str, str],
    literals: Mapping[str, str],
) -> SlotBinding:
    """Relation slots from ``relations`` in declared order, the rest from their own fields."""
    declared = slots_of(template)
    stated = {
        s.name: literals[s.name]
        for s in declared
        if s.kind == "literal" and literals.get(s.name, NONE) != NONE
    }
    applying = [s for s in declared if s.when is None or stated.get(s.when[0]) == s.when[1]]
    relation_slots = [s.name for s in applying if s.kind == "relation"]
    spanned = tuple(dict.fromkeys(relations))[: len(relation_slots)]
    slots = dict(zip(relation_slots, spanned, strict=False))
    slots |= {
        s.name: columns[s.name]
        for s in applying
        if s.kind == "column" and columns.get(s.name, NONE) != NONE
    }
    slots |= {s.name: stated[s.name] for s in applying if s.name in stated}
    return SlotBinding(template=template, slots=slots, relations=spanned)


def relation_slot_count(template: str, binding: SlotBinding) -> int:
    """How many relation slots apply to ``binding`` under its literals."""
    return sum(
        1
        for s in slots_of(template)
        if s.kind == "relation" and (s.when is None or binding.slots.get(s.when[0]) == s.when[1])
    )


def value_lines(tools: ProfileTools, profile: RelationProfile) -> list[str]:
    """Values of the columns the run's policy lets it read; none under ``metadata_only``."""
    lines = []
    for column in profile.columns:
        sample = tools.sample_values(profile.ref, column.name, VALUE_SAMPLE)
        if isinstance(sample, ValueSample) and sample.values:
            lines.append(f"{column.name}: {', '.join(map(str, sample.values))}")
    return lines


def render_layout_prompt(
    profile: RelationProfile,
    group: Sequence[RelationProfile],
    values: Sequence[str],
    failed: Sequence[Sequence[str]],
) -> str:
    others = [p for p in group if p.ref.name != profile.ref.name]
    sections = dict(
        templates="\n".join(_template_line(t) for t in proposable()),
        views=bullets(f"`{view.ID}`" for view in VIEWS),
        failures=history(failed),
        relation=render_relations([profile], {}),
        group=render_relations(others, {}) if others else NONE,
        samples=bullets(values),
    )
    return render_template(LAYOUT_PROMPT, sections)


def _template_line(template: Template) -> str:
    slots: tuple[Slot, ...] = getattr(template, "SLOTS")
    parts = [_slot_text(slot) for slot in slots]
    return f"- `{template.ID}`: {'; '.join(parts)}"


def _slot_text(slot: Slot) -> str:
    text = f"{slot.kind} `{slot.name}`"
    if slot.values:
        text += f" (one of {', '.join(f'`{value}`' for value in slot.values)})"
    if slot.when is not None:
        text += f" when `{slot.when[0]}` is `{slot.when[1]}`"
    return text


def with_bound(structure: StructureResult, entry: BoundEntry) -> StructureResult:
    """The structure with ``entry`` bound, its no_header ignores and no_template entries gone."""
    spanned = set(entry.binding.relations)
    ignored = tuple(
        b
        for b in structure.ignored
        if not (spanned & set(b.relations) and b.slots.get(ignore.REASON) == ignore.NO_HEADER)
    )
    ambiguities = tuple(
        a
        for a in structure.ambiguities
        if not (a.subject in spanned and a.members == (NO_TEMPLATE,))
    )
    update = {"bound": (*structure.bound, entry), "ignored": ignored, "ambiguities": ambiguities}
    result: StructureResult = structure.model_copy(update=update)
    return result


def with_reason(structure: StructureResult, relation: str, reason: str) -> StructureResult:
    """The relation's open no_template entry carries ``reason`` for Review."""
    ambiguities = tuple(
        a.model_copy(update={"reason": reason})
        if a.subject == relation and a.members == (NO_TEMPLATE,) and a.choice is None
        else a
        for a in structure.ambiguities
    )
    result: StructureResult = structure.model_copy(update={"ambiguities": ambiguities})
    return result
