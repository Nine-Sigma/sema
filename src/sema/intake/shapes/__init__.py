"""The closed set of shape templates (design Decisions 1, 4, and 17).

Adding a template is adding a module and listing it here.
"""

from __future__ import annotations

from sema.intake.shapes import (
    athena_bundle,
    concept_and_value_set,
    concept_list,
    concept_with_values_denormalized,
    edge_list_hierarchy,
    entity_model,
    ignore,
    level_columns,
    path_codes,
    semsql,
)
from sema.intake.shapes.base import SlotBinding, Template
from sema.intake.shapes.precedence import MULTI_RELATION, Precedence, apply_precedence

__all__ = [
    "MULTI_RELATION",
    "TEMPLATES",
    "Precedence",
    "SlotBinding",
    "Template",
    "apply_precedence",
    "filled_sections",
]

TEMPLATES: tuple[Template, ...] = (
    concept_list,
    concept_with_values_denormalized,
    concept_and_value_set,
    edge_list_hierarchy,
    level_columns,
    path_codes,
    semsql,
    athena_bundle,
    entity_model,
    ignore,
)


def filled_sections(binding: SlotBinding) -> tuple[str, ...]:
    """The binding's own section, else the sections its template's loader fills."""
    if binding.section is not None:
        return (binding.section,)
    template = next((t for t in TEMPLATES if t.ID == binding.template), None)
    return tuple(getattr(template, "SECTIONS", ()))
