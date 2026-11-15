"""A partial spec's shape entries pin the slot bindings its sections state."""

from __future__ import annotations

from typing import Any

import pytest

from sema.binding.spec import ConceptSourceSpec
from sema.intake.structure.shapes import (
    concept_and_value_set,
    concept_list,
    concept_with_values_denormalized,
    edge_list_hierarchy,
    entity_model,
    level_columns,
    path_codes,
)
from sema.intake.structure.structure_hint_utils import HINT_SECTION, hint_bindings

pytestmark = pytest.mark.unit

TERMS = "land.terms"
VALUES = "land.values"
CODES = "land.codes"
LINKS = "land.links"
DICTIONARY = "land.dictionary"
CONCEPTS = {
    "relation": TERMS,
    "group_by": "family",
    "roles": {"concept_id": "term_id", "name": "term_name"},
}
VALUE_SETS = {
    "relation": VALUES,
    "roles": {"concept_id": "owner_id", "code": "code", "display": "label"},
}
CODE_LIST = {
    "vocabulary_id": "colours",
    "relation": CODES,
    "roles": {"code": "code", "display": "label"},
}
SCHEME = {"qualified_name": "terms", "description": "Terms."}
PARENT_ONLY = {"parent_column": "parent_id"}


def _hint(shape: list[tuple[str, str]], **sections: Any) -> ConceptSourceSpec:
    raw: dict[str, Any] = {
        "spec_version": 2,
        "id": "hinted",
        "identity_system": "hinted",
        "release": "1",
        "default_provider": "staging",
        "shape": [{"section": section, "template": template} for section, template in shape],
    }
    raw.update(sections)
    return ConceptSourceSpec.model_validate(raw)


def _concepts_hint(template: str, hierarchy: dict[str, Any]) -> ConceptSourceSpec:
    return _hint(
        [("hierarchy", template)], scheme=SCHEME, concepts=CONCEPTS, hierarchy=hierarchy
    )


def _only(hint: ConceptSourceSpec) -> Any:
    bindings, failures = hint_bindings(hint)
    assert failures == []
    [binding] = bindings
    return binding


def test_concept_list_pins_the_code_list_the_dotted_section_names() -> None:
    other = {**CODE_LIST, "vocabulary_id": "sizes", "relation": "land.sizes"}
    hint = _hint([("code_lists.colours", concept_list.ID)], code_lists=[other, CODE_LIST])
    binding = _only(hint)
    assert (binding.section, binding.relations) == ("code_lists.colours", (CODES,))
    assert binding.slots == {concept_list.RELATION: CODES}


def test_denormalized_concepts_pin_the_group_by_column() -> None:
    template = concept_with_values_denormalized
    hint = _hint([("concepts", template.ID)], scheme=SCHEME, concepts=CONCEPTS)
    binding = _only(hint)
    assert binding.relations == (TERMS,)
    assert binding.slots == {template.RELATION: TERMS, template.GROUP_BY: "family"}


def test_concept_and_value_set_pins_both_relations_and_their_keys() -> None:
    template = concept_and_value_set
    hint = _hint(
        [("concepts", template.ID)], scheme=SCHEME, concepts=CONCEPTS, value_sets=VALUE_SETS
    )
    binding = _only(hint)
    assert binding.relations == (TERMS, VALUES)
    assert binding.slots == {
        template.CONCEPT_RELATION: TERMS,
        template.VALUE_RELATION: VALUES,
        template.CONCEPT_KEY: "term_id",
        template.VALUE_KEY: "owner_id",
    }


def test_level_columns_number_the_levels_from_one_in_order() -> None:
    hierarchy = {"level_columns": ["top", "middle", "leaf"]}
    binding = _only(_concepts_hint(level_columns.ID, hierarchy))
    assert binding.relations == (TERMS,)
    assert binding.slots == {
        level_columns.RELATION: TERMS,
        f"{level_columns.LEVEL}1": "top",
        f"{level_columns.LEVEL}2": "middle",
        f"{level_columns.LEVEL}3": "leaf",
    }


def test_per_file_entity_model_pins_one_binding_per_entry_with_its_header() -> None:
    entries = [
        {"relation": "land.a_fields", "header": "land.a_about"},
        {"relation": "land.b_fields"},
    ]
    model = {"layout": "per_file", "entities": entries, "roles": {"property": "field"}}
    bindings, failures = hint_bindings(
        _hint([("entity_model", entity_model.ID)], entity_model=model)
    )
    assert failures == []
    assert [(b.relations, b.slots) for b in bindings] == [
        (
            ("land.a_fields", "land.a_about"),
            {
                entity_model.LAYOUT: entity_model.PER_FILE,
                entity_model.DICTIONARY: "land.a_fields",
                entity_model.HEADER: "land.a_about",
            },
        ),
        (
            ("land.b_fields",),
            {entity_model.LAYOUT: entity_model.PER_FILE, entity_model.DICTIONARY: "land.b_fields"},
        ),
    ]


def test_path_codes_pins_its_own_relation_column_and_separator() -> None:
    hierarchy = {"path_column": "path", "separator": "/", "relation": LINKS}
    binding = _only(_concepts_hint(path_codes.ID, hierarchy))
    assert (binding.template, binding.section, binding.relations) == (
        path_codes.ID, "hierarchy", (LINKS,)
    )
    assert binding.slots == {
        path_codes.RELATION: LINKS, path_codes.CODE: "path", path_codes.SEPARATOR: "/"
    }


def test_path_codes_without_a_relation_reads_the_concepts_relation() -> None:
    binding = _only(_concepts_hint(path_codes.ID, {"path_column": "path", "separator": "."}))
    assert binding.relations == (TERMS,)
    assert binding.slots[path_codes.RELATION] == TERMS


@pytest.mark.parametrize(
    "hierarchy",
    [PARENT_ONLY, {**PARENT_ONLY, "relation": TERMS}],
    ids=["no-relation", "relation-is-the-concepts-relation"],
)
def test_parent_column_on_the_concepts_relation_spans_that_relation_only(
    hierarchy: dict[str, Any],
) -> None:
    binding = _only(_concepts_hint(edge_list_hierarchy.ID, hierarchy))
    assert binding.relations == (TERMS,)
    assert binding.slots == {
        edge_list_hierarchy.CONCEPT_RELATION: TERMS,
        edge_list_hierarchy.IDENTIFIER: "term_id",
        edge_list_hierarchy.PARENT: "parent_id",
    }


def test_a_separate_edge_relation_adds_its_relation_and_child_column() -> None:
    hierarchy = {**PARENT_ONLY, "child_column": "child_id", "relation": LINKS}
    binding = _only(_concepts_hint(edge_list_hierarchy.ID, hierarchy))
    assert binding.relations == (TERMS, LINKS)
    assert binding.slots[edge_list_hierarchy.EDGE_RELATION] == LINKS
    assert binding.slots[edge_list_hierarchy.CHILD] == "child_id"
    assert binding.slots[edge_list_hierarchy.PARENT] == "parent_id"


def test_single_relation_entity_model_pins_entity_and_property_roles() -> None:
    model = {"relation": DICTIONARY, "roles": {"entity": "table", "property": "field"}}
    binding = _only(_hint([("entity_model", entity_model.ID)], entity_model=model))
    assert binding.relations == (DICTIONARY,)
    assert binding.slots == {
        entity_model.LAYOUT: entity_model.SINGLE_RELATION,
        entity_model.RELATION: DICTIONARY,
        entity_model.ENTITY: "table",
        entity_model.PROPERTY: "field",
    }


UNBUILDABLE = [
    pytest.param(
        "hierarchy",
        edge_list_hierarchy.ID,
        {**PARENT_ONLY, "relation": LINKS},
        id="edge-relation-without-child-column",
    ),
    pytest.param("hierarchy", "no_such_template", PARENT_ONLY, id="unknown-template"),
    pytest.param("concepts.family", concept_list.ID, PARENT_ONLY, id="dotted-path-into-a-section"),
    pytest.param("hierarchy", concept_list.ID, PARENT_ONLY, id="concept-list-on-a-hierarchy"),
    pytest.param(
        "hierarchy",
        concept_with_values_denormalized.ID,
        PARENT_ONLY,
        id="denormalized-on-a-hierarchy",
    ),
    pytest.param("concepts", concept_and_value_set.ID, PARENT_ONLY, id="no-value-sets-section"),
    pytest.param("concepts", level_columns.ID, PARENT_ONLY, id="level-columns-on-concepts"),
    pytest.param("concepts", path_codes.ID, PARENT_ONLY, id="path-codes-on-concepts"),
    pytest.param("concepts", edge_list_hierarchy.ID, PARENT_ONLY, id="edge-list-on-concepts"),
    pytest.param("concepts", entity_model.ID, PARENT_ONLY, id="entity-model-on-concepts"),
]


@pytest.mark.parametrize(("section", "template", "hierarchy"), UNBUILDABLE)
def test_an_entry_that_builds_nothing_is_a_hint_section_failure(
    section: str, template: str, hierarchy: dict[str, Any]
) -> None:
    hint = _hint([(section, template)], scheme=SCHEME, concepts=CONCEPTS, hierarchy=hierarchy)
    bindings, failures = hint_bindings(hint)
    assert bindings == []
    [failure] = failures
    assert failure.check == HINT_SECTION
    assert failure.evidence == {"section": section, "template": template}
