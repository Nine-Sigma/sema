"""Intake-authored output models: flat enums over the members code proved (Decision 2).

Every model is built at call time with `create_model`: enum fields at the top level,
arrays of primitives, `extra="forbid"`, and no defaults, so strict provider modes see
every field required and no schema carries a union, reference, or pattern.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from sema.intake.output_models_utils import (
    ABSTAIN,
    MAX_SCHEMA_KEYS,
    NONE,
    Choice,
    checked_field,
    checked_key_count,
    checked_name,
    enum_type,
    read_choices,
    refuse_sentinel,
)
from sema.intake.shapes import TEMPLATES

MAX_STRUCTURE_BATCH = 50
STRUCTURE_BATCH = "StructureBatch_v1"
ROLES_CHOICE = "RolesChoice_v1"
REFERENCE_CHOICE = "ReferenceChoice_v1"
LAYOUT_PROPOSAL = "LayoutProposal_v1"
ENTRY_CHOICE = "EntryChoice_v1"
ENTRY_PREFIX = "entry_"
ROLE_PREFIX = "role_"
REFERENCE_PREFIX = "reference_"
SLOT_PREFIX = "slot_"
LITERAL_PREFIX = "literal_"
LAYOUT_ANSWERS = ("propose", ABSTAIN, "needs_values")
FORBID_EXTRA = ConfigDict(extra="forbid")

Fields = dict[str, Any]

# The two reasoning lists take the remaining keys under the cap.
MAX_REFERENCE_BATCH = MAX_SCHEMA_KEYS - len(("confidences", "reasons"))


def _build(name: str, fields: Fields) -> type[BaseModel]:
    checked_key_count(name, len(fields))
    model: type[BaseModel] = create_model(name, __config__=FORBID_EXTRA, **fields)
    return model


def _abstainable(members: Sequence[str], what: str) -> Any:
    refuse_sentinel(members, ABSTAIN, what)
    return enum_type([*members, ABSTAIN])


def _optional(members: Sequence[str], what: str) -> Any:
    refuse_sentinel(members, NONE, what)
    return enum_type([*members, NONE])


def _reasoning_lists() -> Fields:
    return {"confidences": (list[float], ...), "reasons": (list[str], ...)}


def choice_model(name: str, members: Sequence[str]) -> type[BaseModel]:
    """One ambiguity entry: a choice among its members or `abstain`."""
    fields: Fields = {
        "choice": (_abstainable(members, "choice"), ...),
        "confidence": (float, ...),
        "reason": (str, ...),
    }
    return _build(checked_name(name), fields)


def entry_field(index: int) -> str:
    return f"{ENTRY_PREFIX}{index}"


def structure_batch_model(entries: Sequence[Sequence[str]]) -> type[BaseModel]:
    """One field per structure entry, each over that entry's own members."""
    if len(entries) > MAX_STRUCTURE_BATCH:
        raise ValueError(
            f"{len(entries)} structure entries exceed the batch limit {MAX_STRUCTURE_BATCH}"
        )
    fields: Fields = {
        entry_field(index): (_abstainable(members, entry_field(index)), ...)
        for index, members in enumerate(entries)
    }
    return _build(STRUCTURE_BATCH, fields | _reasoning_lists())


def role_field(role: str) -> str:
    return checked_field(ROLE_PREFIX, role)


def roles_model(roles: Mapping[str, Sequence[str]]) -> type[BaseModel]:
    """One field per role of a relation, each over the columns eligible for it."""
    fields: Fields = {
        role_field(role): (_abstainable(columns, role_field(role)), ...)
        for role, columns in roles.items()
    }
    return _build(ROLES_CHOICE, fields | _reasoning_lists())


def reference_field(index: int) -> str:
    return f"{REFERENCE_PREFIX}{index}"


def references_model(entries: Sequence[Sequence[str]]) -> type[BaseModel]:
    """One field per open reference entry of a relation, each over that entry's members."""
    if len(entries) > MAX_REFERENCE_BATCH:
        raise ValueError(
            f"{len(entries)} reference entries exceed the batch limit {MAX_REFERENCE_BATCH}"
        )
    fields: Fields = {
        reference_field(index): (_abstainable(members, reference_field(index)), ...)
        for index, members in enumerate(entries)
    }
    return _build(REFERENCE_CHOICE, fields | _reasoning_lists())


def slot_field(slot: str) -> str:
    return checked_field(SLOT_PREFIX, slot)


def literal_field(slot: str) -> str:
    return checked_field(LITERAL_PREFIX, slot)


def layout_model(
    templates: Sequence[str],
    views: Sequence[str],
    columns: Sequence[str],
    relations: Sequence[str],
    slots: Sequence[str],
    literals: Mapping[str, Sequence[str]] | None = None,
) -> type[BaseModel]:
    """A layout proposal over the closed template set, the view registry, and a group.

    ``slots`` are column slots over ``columns``; ``literals`` maps each literal slot to the
    values it may take.
    """
    refuse_sentinel(columns, NONE, "column")
    column = enum_type(columns)
    fields: Fields = {
        "answer": (enum_type(LAYOUT_ANSWERS), ...),
        "template": (_optional(templates, "template"), ...),
        "view": (_optional(views, "view"), ...),
        "view_columns": (list[column], ...),  # type: ignore[valid-type]
        **{slot_field(slot): (_optional(columns, slot_field(slot)), ...) for slot in slots},
        **{
            literal_field(slot): (_optional(values, literal_field(slot)), ...)
            for slot, values in (literals or {}).items()
        },
        "relations": (list[enum_type(relations)], ...),  # type: ignore[misc]
        "needs_values": (list[column], ...),  # type: ignore[valid-type]
        "confidence": (float, ...),
        "reason": (str, ...),
    }
    return _build(LAYOUT_PROPOSAL, fields)


def entry_choices(answer: BaseModel, count: int) -> list[Choice] | None:
    """A batch answer's choices in entry order, or None when it failed (lists misaligned)."""
    return read_choices(answer, [entry_field(index) for index in range(count)])


def role_choices(answer: BaseModel, roles: Sequence[str]) -> list[Choice] | None:
    """A roles answer's choices in the given role order, or None when it failed."""
    return read_choices(answer, [role_field(role) for role in roles])


def reference_choices(answer: BaseModel, count: int) -> list[Choice] | None:
    """A references answer's choices in entry order, or None when it failed."""
    return read_choices(answer, [reference_field(index) for index in range(count)])


def _sample_members(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index}" for index in range(count)]


def _sample_choice() -> type[BaseModel]:
    return choice_model(ENTRY_CHOICE, _sample_members("member", 3))


def _sample_structure_batch() -> type[BaseModel]:
    entries = [_sample_members(f"member_{i}", 3) for i in range(MAX_STRUCTURE_BATCH)]
    return structure_batch_model(entries)


def _sample_roles() -> type[BaseModel]:
    columns = _sample_members("column", 4)
    return roles_model({role: columns for role in _sample_members("role", 6)})


def _sample_references() -> type[BaseModel]:
    entries = [_sample_members(f"target_{i}", 3) for i in range(MAX_REFERENCE_BATCH)]
    return references_model(entries)


def _sample_layout() -> type[BaseModel]:
    return layout_model(
        templates=[template.ID for template in TEMPLATES],
        views=_sample_members("view", 2),
        columns=_sample_members("_col", 6),
        relations=_sample_members("relation", 3),
        slots=_sample_members("slot", 8),
        literals={"literal_0": _sample_members("value", 6)},
    )


OUTPUT_MODEL_FACTORIES: tuple[Callable[[], type[BaseModel]], ...] = (
    _sample_choice,
    _sample_structure_batch,
    _sample_roles,
    _sample_references,
    _sample_layout,
)
