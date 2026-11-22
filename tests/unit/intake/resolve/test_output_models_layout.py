"""The layout proposal output model: closed sets only, every field required."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from sema.intake.resolve.output_models import layout_model
from sema.intake.structure.shapes import TEMPLATES

pytestmark = pytest.mark.unit

TEMPLATE_IDS = [template.ID for template in TEMPLATES]


def _enum(model: type[BaseModel], field: str) -> list[str]:
    prop = model.model_json_schema()["properties"][field]
    target = prop.get("items", prop)
    return list(target["enum"])


# -- layout proposal --------------------------------------------------------


COLUMNS = ["_col_0", "_col_1", "_col_2"]
GROUP_RELATIONS = ["sheet_a", "sheet_b"]
SLOTS = ["id", "label", "parent"]


def _layout() -> type[BaseModel]:
    return layout_model(
        templates=TEMPLATE_IDS,
        views=["fill_down"],
        columns=COLUMNS,
        relations=GROUP_RELATIONS,
        slots=SLOTS,
    )


def _proposal(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "answer": "propose",
        "template": TEMPLATE_IDS[0],
        "view": "fill_down",
        "view_columns": ["_col_0"],
        "slot_id": "_col_0",
        "slot_label": "_col_1",
        "slot_parent": "none",
        "relations": ["sheet_a"],
        "needs_values": [],
        "confidence": 0.6,
        "reason": "variable on first rows, codes on continuation rows",
    }
    return {**payload, **overrides}


def test_layout_model_enumerates_only_the_closed_sets() -> None:
    model = _layout()
    assert model.__name__ == "LayoutProposal_v1"
    assert _enum(model, "answer") == ["propose", "abstain", "needs_values"]
    assert _enum(model, "template") == [*TEMPLATE_IDS, "none"]
    assert _enum(model, "view") == ["fill_down", "none"]
    assert _enum(model, "view_columns") == COLUMNS
    assert _enum(model, "relations") == GROUP_RELATIONS
    assert _enum(model, "needs_values") == COLUMNS
    for slot in SLOTS:
        assert _enum(model, f"slot_{slot}") == [*COLUMNS, "none"]


def test_layout_model_accepts_a_proposal_and_a_values_request() -> None:
    model = _layout()
    model.model_validate(_proposal())
    model.model_validate(
        _proposal(answer="needs_values", template="none", view="none",
                  needs_values=["_col_1", "_col_2"]),
    )


@pytest.mark.parametrize(
    "override",
    [
        {"view": "pivot"},
        {"template": "unknown_template"},
        {"slot_id": "_col_9"},
        {"relations": ["sheet_z"]},
        {"needs_values": ["_col_9"]},
        {"answer": "maybe"},
    ],
    ids=["unknown_view", "unknown_template", "slot_column", "relation", "values_column",
         "answer"],
)
def test_layout_model_rejects_values_outside_the_closed_sets(
    override: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _layout().model_validate(_proposal(**override))


def test_layout_model_rejects_a_missing_field() -> None:
    payload = _proposal()
    del payload["slot_parent"]
    with pytest.raises(ValidationError):
        _layout().model_validate(payload)
