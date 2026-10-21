"""Intake-authored output models: proven members only, flat, portable schemas."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, ValidationError

from sema.intake.output_models import (
    MAX_STRUCTURE_BATCH,
    OUTPUT_MODEL_FACTORIES,
    choice_model,
    entry_choices,
    layout_model,
    role_choices,
    roles_model,
    structure_batch_model,
)
from sema.intake.shapes import TEMPLATES

pytestmark = pytest.mark.unit

KEY_CAP = 64
BANNED_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "$ref", "pattern", "default"})
FREE_TEXT_FIELDS = frozenset({"reason", "reasons"})
VERSIONED_NAME = re.compile(r"_v\d+$")
TEMPLATE_IDS = [template.ID for template in TEMPLATES]


def _pydantic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """What ChatOpenAI hands the provider: it passes the class to the SDK unchanged."""
    return model.model_json_schema()


def _function_schema(model: type[BaseModel]) -> dict[str, Any]:
    """What ChatDatabricks sends for `with_structured_output(method="json_schema")`."""
    parameters: dict[str, Any] = convert_to_openai_function(model, strict=True)["parameters"]
    return parameters


SCHEMA_PATHS: tuple[Callable[[type[BaseModel]], dict[str, Any]], ...] = (
    _pydantic_schema,
    _function_schema,
)


def _nodes(schema: Any) -> Iterator[dict[str, Any]]:
    if isinstance(schema, dict):
        yield schema
        for value in schema.values():
            yield from _nodes(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _nodes(item)


def _property_key_count(schema: dict[str, Any]) -> int:
    return sum(len(node.get("properties", {})) for node in _nodes(schema))


def _portability_violations(schema: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for node in _nodes(schema):
        violations += [f"keyword {key}" for key in BANNED_KEYWORDS & node.keys()]
        if node.get("type") != "object" and "properties" not in node:
            continue
        properties = node.get("properties", {})
        if node.get("additionalProperties") is not False:
            violations.append(f"object {node.get('title', '?')} allows extra keys")
        if set(node.get("required", [])) != set(properties):
            violations.append(f"object {node.get('title', '?')} has optional fields")
        violations += _choice_violations(properties)
    if _property_key_count(schema) > KEY_CAP:
        violations.append(f"{_property_key_count(schema)} keys over {KEY_CAP}")
    return violations


def _choice_violations(properties: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for name, prop in properties.items():
        if name in FREE_TEXT_FIELDS:
            continue
        target = prop.get("items", {}) if prop.get("type") == "array" else prop
        if target.get("type") == "string" and "enum" not in target:
            violations.append(f"field {name} is free text, not an enum")
    return violations


def _enum(model: type[BaseModel], field: str) -> list[str]:
    prop = model.model_json_schema()["properties"][field]
    target = prop.get("items", prop)
    return list(target["enum"])


def _registered_models() -> list[type[BaseModel]]:
    return [factory() for factory in OUTPUT_MODEL_FACTORIES]


# -- per-entry choice model -------------------------------------------------


def test_choice_model_accepts_members_and_abstain() -> None:
    model = choice_model("SectionChoice_v1", ["label", "description"])
    for member in ("label", "description", "abstain"):
        parsed = model.model_validate(
            {"choice": member, "confidence": 0.7, "reason": "named in the header"},
        )
        assert parsed.choice == member  # type: ignore[attr-defined]
    assert _enum(model, "choice") == ["label", "description", "abstain"]


def test_choice_model_rejects_a_member_outside_the_entry() -> None:
    model = choice_model("SectionChoice_v1", ["label", "description"])
    with pytest.raises(ValidationError):
        model.model_validate({"choice": "notes", "confidence": 0.7, "reason": "r"})


@pytest.mark.parametrize("missing", ["choice", "confidence", "reason"])
def test_choice_model_rejects_a_missing_field(missing: str) -> None:
    model = choice_model("SectionChoice_v1", ["label", "description"])
    payload = {"choice": "label", "confidence": 0.7, "reason": "r"}
    del payload[missing]
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_choice_model_rejects_an_extra_field() -> None:
    model = choice_model("SectionChoice_v1", ["label", "description"])
    with pytest.raises(ValidationError):
        model.model_validate(
            {"choice": "label", "confidence": 0.7, "reason": "r", "note": "x"},
        )


def test_choice_model_name_carries_its_version() -> None:
    assert choice_model("SectionChoice_v1", ["a"]).__name__ == "SectionChoice_v1"
    with pytest.raises(ValueError):
        choice_model("SectionChoice", ["a"])


# -- structure batch --------------------------------------------------------


def _batch_entries(count: int) -> list[list[str]]:
    return [[f"member_{i}_a", f"member_{i}_b", f"member_{i}_c"] for i in range(count)]


def test_structure_batch_limit_is_fifty() -> None:
    assert MAX_STRUCTURE_BATCH == 50


@pytest.mark.parametrize("schema_of", SCHEMA_PATHS)
def test_fifty_entry_structure_batch_has_at_most_64_keys(
    schema_of: Callable[[type[BaseModel]], dict[str, Any]],
) -> None:
    model = structure_batch_model(_batch_entries(MAX_STRUCTURE_BATCH))
    assert _property_key_count(schema_of(model)) <= KEY_CAP


def test_structure_batch_over_the_limit_is_refused() -> None:
    with pytest.raises(ValueError):
        structure_batch_model(_batch_entries(MAX_STRUCTURE_BATCH + 1))


def test_structure_batch_enumerates_each_entry_separately() -> None:
    entries = _batch_entries(3)
    model = structure_batch_model(entries)
    assert model.__name__ == "StructureBatch_v1"
    for index, members in enumerate(entries):
        assert _enum(model, f"entry_{index}") == [*members, "abstain"]


def test_structure_batch_rejects_another_entrys_member() -> None:
    model = structure_batch_model(_batch_entries(2))
    valid = {
        "entry_0": "member_0_a",
        "entry_1": "abstain",
        "confidences": [0.9, 0.1],
        "reasons": ["one relation fits", "no template fits"],
    }
    model.model_validate(valid)
    with pytest.raises(ValidationError):
        model.model_validate({**valid, "entry_0": "member_1_a"})


def test_structure_batch_rejects_a_missing_entry() -> None:
    model = structure_batch_model(_batch_entries(2))
    with pytest.raises(ValidationError):
        model.model_validate(
            {"entry_0": "member_0_a", "confidences": [0.9], "reasons": ["r"]},
        )


# -- roles per relation -----------------------------------------------------


ROLES = {"code": ["key_col", "label_col"], "display": ["label_col", "notes_col"]}


def test_roles_model_enumerates_each_roles_eligible_columns() -> None:
    model = roles_model(ROLES)
    assert model.__name__ == "RolesChoice_v1"
    assert _enum(model, "role_code") == ["key_col", "label_col", "abstain"]
    assert _enum(model, "role_display") == ["label_col", "notes_col", "abstain"]


def test_roles_model_rejects_a_column_not_eligible_for_the_role() -> None:
    model = roles_model(ROLES)
    valid = {
        "role_code": "key_col",
        "role_display": "label_col",
        "confidences": [0.9, 0.8],
        "reasons": ["unique", "text"],
    }
    model.model_validate(valid)
    with pytest.raises(ValidationError):
        model.model_validate({**valid, "role_code": "notes_col"})
    with pytest.raises(ValidationError):
        model.model_validate({k: v for k, v in valid.items() if k != "role_display"})


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


# -- registry and portability -----------------------------------------------


def test_registry_covers_every_intake_output_model() -> None:
    names = [model.__name__ for model in _registered_models()]
    assert {"StructureBatch_v1", "RolesChoice_v1", "LayoutProposal_v1"} <= set(names)
    assert len(names) == len(set(names))
    assert all(VERSIONED_NAME.search(name) for name in names)


def test_registered_structure_batch_is_the_worst_case() -> None:
    batch = next(m for m in _registered_models() if m.__name__ == "StructureBatch_v1")
    entry_fields = [name for name in batch.model_fields if name.startswith("entry_")]
    assert len(entry_fields) == MAX_STRUCTURE_BATCH


@pytest.mark.parametrize("schema_of", SCHEMA_PATHS)
def test_every_registered_output_model_is_portable(
    schema_of: Callable[[type[BaseModel]], dict[str, Any]],
) -> None:
    failures = {
        model.__name__: violations
        for model in _registered_models()
        if (violations := _portability_violations(schema_of(model)))
    }
    assert failures == {}


def test_portability_check_catches_a_nested_optional_model() -> None:
    class Inner_v1(BaseModel):  # noqa: N801
        value: str | None = None

    class Outer_v1(BaseModel):  # noqa: N801
        inner: Inner_v1

    violations = _portability_violations(_pydantic_schema(Outer_v1))
    assert any("$ref" in v for v in violations)
    assert any("anyOf" in v for v in violations)
    assert any("optional" in v for v in violations)
    assert any("extra keys" in v for v in violations)


# -- reserved sentinels and field names -------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: choice_model("SectionChoice_v1", ["label", "abstain"]),
        lambda: structure_batch_model([["member_a"], ["member_b", "abstain"]]),
        lambda: roles_model({"code": ["key_col", "abstain"]}),
        lambda: layout_model(TEMPLATE_IDS, ["fill_down"], ["_col_0", "none"],
                             GROUP_RELATIONS, SLOTS),
        lambda: layout_model([*TEMPLATE_IDS, "none"], ["fill_down"], COLUMNS,
                             GROUP_RELATIONS, SLOTS),
        lambda: layout_model(TEMPLATE_IDS, ["none"], COLUMNS, GROUP_RELATIONS, SLOTS),
    ],
    ids=["choice", "batch", "roles", "layout_column", "layout_template", "layout_view"],
)
def test_a_member_equal_to_a_sentinel_is_refused(build: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match="'(abstain|none)'"):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: roles_model({"code key": ["key_col"]}),
        lambda: layout_model(TEMPLATE_IDS, ["fill_down"], COLUMNS, GROUP_RELATIONS,
                             ["parent-id"]),
    ],
    ids=["role", "slot"],
)
def test_a_field_name_that_is_not_an_identifier_is_refused(
    build: Callable[[], Any],
) -> None:
    with pytest.raises(ValueError):
        build()


def test_a_single_member_set_still_serializes_as_an_enum() -> None:
    model = layout_model(TEMPLATE_IDS, ["fill_down"], ["_col_0"], ["sheet_a"], ["id"])
    assert _enum(model, "relations") == ["sheet_a"]
    assert _enum(model, "view_columns") == ["_col_0"]
    for schema_of in SCHEMA_PATHS:
        assert _portability_violations(schema_of(model)) == []


# -- reading a batch answer -------------------------------------------------


def test_entry_choices_pairs_each_entry_with_its_confidence_and_reason() -> None:
    model = structure_batch_model(_batch_entries(2))
    answer = model.model_validate(
        {
            "entry_0": "member_0_a",
            "entry_1": "abstain",
            "confidences": [0.9, 0.1],
            "reasons": ["one relation fits", "no template fits"],
        },
    )
    assert entry_choices(answer, 2) == [
        ("member_0_a", 0.9, "one relation fits"),
        ("abstain", 0.1, "no template fits"),
    ]


@pytest.mark.parametrize(
    ("confidences", "reasons"),
    [([0.9], ["a", "b"]), ([0.9, 0.1], ["a"]), ([0.9, 0.1, 0.5], ["a", "b", "c"])],
    ids=["short_confidences", "short_reasons", "long_both"],
)
def test_entry_choices_is_none_when_the_lists_do_not_match_the_entries(
    confidences: list[float], reasons: list[str],
) -> None:
    model = structure_batch_model(_batch_entries(2))
    answer = model.model_validate(
        {"entry_0": "member_0_a", "entry_1": "abstain",
         "confidences": confidences, "reasons": reasons},
    )
    assert entry_choices(answer, 2) is None


def test_role_choices_reads_each_role_in_order() -> None:
    model = roles_model(ROLES)
    answer = model.model_validate(
        {"role_code": "key_col", "role_display": "abstain",
         "confidences": [0.9, 0.2], "reasons": ["unique", "no text column"]},
    )
    assert role_choices(answer, list(ROLES)) == [
        ("key_col", 0.9, "unique"),
        ("abstain", 0.2, "no text column"),
    ]
    short = model.model_validate({**answer.model_dump(), "reasons": ["unique"]})
    assert role_choices(short, list(ROLES)) is None


def test_a_model_over_the_key_cap_is_refused() -> None:
    roles = {f"role_{index}": ["key_col"] for index in range(KEY_CAP)}
    with pytest.raises(ValueError, match="key"):
        roles_model(roles)
