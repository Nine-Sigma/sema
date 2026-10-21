"""ConceptSourceSpec version 2: load-time behavior of the concept-source-spec delta."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from tests.unit.binding.fixtures.concept_library import SPEC_PATH

pytestmark = pytest.mark.unit


def _minimal(**extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "spec_version": 2,
        "id": "syn-library",
        "identity_system": "syn",
        "release": "2026-10",
        "scheme": {"qualified_name": "syn.codes", "description": "Codes"},
        "concepts": {
            "relation": "land.codes",
            "group_by": "code",
            "roles": {"concept_id": "code", "name": "label"},
        },
    }
    spec.update(extra)
    return spec


def _load(raw: dict[str, Any]) -> Any:
    from sema.binding.spec import ConceptSourceSpec

    return ConceptSourceSpec.model_validate(raw)


def _v1_raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(SPEC_PATH.read_text())["concept_source"]
    return raw


def _value_sets(**roles: Any) -> dict[str, Any]:
    return {
        "relation": "land.values",
        "roles": {"concept_id": "concept", "code": "code", "display": "label", **roles},
    }


def test_spec_version_defaults_to_1() -> None:
    spec = _load(_v1_raw())

    assert spec.spec_version == 1


def test_unknown_spec_version_names_supported_versions() -> None:
    with pytest.raises(ValidationError, match=r"\{1, 2\}"):
        _load(_minimal(spec_version=3))


def test_version_2_field_on_version_1_spec_is_rejected() -> None:
    raw = _v1_raw()
    raw["value_sets"]["roles"]["status"] = {
        "column": "state",
        "active_values": ["active"],
        "retired_values": ["deprecated"],
    }

    with pytest.raises(ValidationError, match=r"status.*spec_version: 2"):
        _load(raw)


def test_object_relation_on_version_1_spec_is_rejected() -> None:
    raw = _v1_raw()
    raw["concepts"]["relation"] = {"provider": "staging", "name": "lib.data_element"}

    with pytest.raises(ValidationError, match=r"spec_version: 2"):
        _load(raw)


def test_version_1_spec_still_requires_every_section() -> None:
    raw = _v1_raw()
    del raw["value_mappings"]

    with pytest.raises(ValidationError, match="value_mappings"):
        _load(raw)


def test_minimal_code_list_loads() -> None:
    spec = _load(_minimal())

    assert spec.value_sets is None
    assert spec.concept_details is None
    assert spec.term_details is None
    assert spec.relationships is None
    assert spec.authored_bindings is None
    assert spec.value_mappings is None


def test_neither_concepts_nor_entity_model_is_rejected() -> None:
    raw = _minimal()
    del raw["concepts"]

    with pytest.raises(ValidationError, match="concepts.*entity_model"):
        _load(raw)


def test_concepts_require_scheme() -> None:
    raw = _minimal()
    del raw["scheme"]

    with pytest.raises(ValidationError, match="scheme"):
        _load(raw)


def test_string_and_object_relations_become_relation_refs() -> None:
    from sema.binding.spec import RelationRef

    spec = _load(
        _minimal(
            default_provider="staging",
            value_sets={**_value_sets(), "relation": {"provider": "staging", "name": "x.v"}},
        )
    )

    assert spec.concepts.relation == RelationRef(provider="staging", name="land.codes")
    assert spec.value_sets.relation == RelationRef(provider="staging", name="x.v")


def test_default_provider_is_warehouse() -> None:
    spec = _load(_minimal())

    assert spec.default_provider == "warehouse"
    assert spec.concepts.relation.provider == "warehouse"


def test_unknown_provider_names_allowed_providers() -> None:
    raw = _minimal()
    raw["concepts"]["relation"] = {"provider": "s3", "name": "bucket.codes"}

    with pytest.raises(ValidationError, match="warehouse.*staging"):
        _load(raw)


@pytest.mark.parametrize(
    ("hierarchy", "form"),
    [
        (
            {"parent_column": "parent_id", "relationship_type": "BROADER"},
            "ParentColumnHierarchy",
        ),
        (
            {"level_columns": ["l1", "l2", "l3"], "relationship_type": "BROADER"},
            "LevelColumnsHierarchy",
        ),
        ({"path_column": "code", "separator": "."}, "PathCodesHierarchy"),
    ],
)
def test_each_hierarchy_form_loads(hierarchy: dict[str, Any], form: str) -> None:
    spec = _load(_minimal(hierarchy=hierarchy))

    assert type(spec.hierarchy).__name__ == form
    assert spec.hierarchy.relationship_type


def test_two_hierarchy_declarations_are_rejected() -> None:
    relationships = {
        "relation": "land.edges",
        "roles": {
            "from": {"system": "syn", "id": "child"},
            "to": {"system": "syn", "id": "parent"},
            "type": "rel",
            "hierarchical": "is_hierarchical",
        },
    }
    raw = _minimal(
        hierarchy={"parent_column": "parent_id", "relationship_type": "BROADER"},
        relationships=relationships,
    )

    with pytest.raises(ValidationError, match="hierarchy.*relationships"):
        _load(raw)


def test_lifecycle_and_lang_roles_load() -> None:
    status = {"column": "state", "active_values": ["A"], "retired_values": ["R"]}
    lifecycle = {
        "status": status,
        "valid_from": "start",
        "valid_to": "end",
        "replaced_by": "successor",
        "lang": "label_lang",
    }
    raw = _minimal(value_sets=_value_sets(**lifecycle))
    raw["concepts"]["roles"].update(lifecycle)

    spec = _load(raw)

    for roles in (spec.concepts.roles, spec.value_sets.roles):
        assert roles.status.retired_values == ["R"]
        assert (roles.valid_from, roles.valid_to) == ("start", "end")
        assert roles.replaced_by == "successor"
        assert roles.lang == "label_lang"


def test_mappings_default_exact_predicates() -> None:
    mappings = {
        "relation": "land.sssom",
        "roles": {
            "subject_id": "subject_id",
            "predicate_id": "predicate_id",
            "object_id": "object_id",
            "mapping_justification": "mapping_justification",
        },
    }

    spec = _load(_minimal(mappings=mappings))

    assert spec.mappings.exact_predicates == ["skos:exactMatch"]
    assert spec.mappings.roles.confidence is None


def test_term_storage_defaults() -> None:
    spec = _load(_minimal())

    assert spec.term_storage.inline_cap == 5000
    assert spec.external_namespace == "vocabulary_syn_library"


def test_term_storage_explicit_namespace() -> None:
    spec = _load(_minimal(term_storage={"inline_cap": 5, "external_namespace": "v"}))

    assert spec.term_storage.inline_cap == 5
    assert spec.external_namespace == "v"


def test_entity_model_requires_entity_property_and_type() -> None:
    raw = _minimal(entity_model={"relation": "land.dict", "roles": {"entity": "t"}})
    del raw["concepts"]

    with pytest.raises(ValidationError, match="property"):
        _load(raw)


def test_entity_model_without_concepts_loads() -> None:
    roles = {"entity": "t", "property": "c", "type": "dt", "required": "req"}
    raw = _minimal(entity_model={"relation": "land.dict", "roles": roles})
    del raw["concepts"]
    del raw["scheme"]

    spec = _load(raw)

    assert spec.concepts is None
    assert spec.entity_model.roles.required == "req"
    assert spec.entity_model.roles.primary_key is None


def test_shape_loads() -> None:
    spec = _load(_minimal(shape=[{"section": "concepts", "template": "concept_list"}]))

    assert spec.shape[0].template == "concept_list"


def test_cross_provider_scope_is_rejected() -> None:
    raw = _minimal(
        value_sets={**_value_sets(), "relation": {"provider": "staging", "name": "v"}}
    )

    with pytest.raises(ValidationError, match="value_sets.*concepts"):
        _load(raw)


def test_mappings_are_exempt_from_the_scope_provider_rule() -> None:
    mappings = {
        "relation": {"provider": "staging", "name": "landed.sssom"},
        "roles": {
            "subject_id": "s",
            "predicate_id": "p",
            "object_id": "o",
            "mapping_justification": "j",
        },
    }

    spec = _load(_minimal(mappings=mappings))

    assert spec.mappings.relation.provider == "staging"
    assert spec.concepts.relation.provider == "warehouse"


def test_value_mappings_on_another_provider_are_rejected_from_yaml(
    tmp_path: Any,
) -> None:
    from sema.binding.spec import load_concept_source_spec

    raw = _v1_raw()
    raw["spec_version"] = 2
    raw["value_mappings"]["relation"] = {"provider": "staging", "name": "landed.map"}
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump({"concept_source": raw}))

    with pytest.raises(ValidationError, match="value_mappings.*concepts"):
        load_concept_source_spec(path)


def test_hierarchy_without_concepts_is_rejected() -> None:
    roles = {"entity": "t", "property": "c", "type": "dt"}
    raw = _minimal(
        entity_model={"relation": "land.dict", "roles": roles},
        hierarchy={"parent_column": "parent_id"},
    )
    del raw["concepts"]

    with pytest.raises(ValidationError, match="hierarchy need a `concepts`"):
        _load(raw)


# S2 additions: per-file entity models and standalone code lists.

PER_FILE_ENTRY = {
    "relation": "land.orders__Columns",
    "header": "land.orders__Header",
    "entity": "Orders",
    "foreign_keys": [
        {
            "columns": ["Customer ID", "Order Date"],
            "entity": "Customers",
            "reference_columns": ["Customer ID", "Order Date"],
        }
    ],
    "domains": {"Status": "Order_Status"},
}


def _bare(**extra: Any) -> dict[str, Any]:
    raw = _minimal(**extra)
    del raw["concepts"]
    del raw["scheme"]
    return raw


def _per_file(roles: dict[str, Any] | None = None, **section: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "layout": "per_file",
        "entities": [PER_FILE_ENTRY],
        "roles": roles if roles is not None else {"property": "field"},
    }
    body.update(section)
    return _bare(entity_model=body)


def _code_list(vocabulary_id: str = "Country_Codes", **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "vocabulary_id": vocabulary_id,
        "relation": f"land.{vocabulary_id}",
        "roles": {"code": "code", "display": "label"},
    }
    entry.update(extra)
    return entry


def test_entity_model_layout_defaults_to_single_relation() -> None:
    roles = {"entity": "t", "property": "c", "type": "dt"}

    spec = _load(_bare(entity_model={"relation": "land.dict", "roles": roles}))

    assert spec.entity_model.layout == "single_relation"


def test_per_file_entry_loads_header_entity_foreign_keys_and_domains() -> None:
    from sema.binding.spec import RelationRef

    spec = _load(_per_file() | {"default_provider": "staging"})

    model = spec.entity_model
    assert model.layout == "per_file"
    (entry,) = model.entities
    assert entry.relation == RelationRef(provider="staging", name="land.orders__Columns")
    assert entry.header == RelationRef(provider="staging", name="land.orders__Header")
    assert entry.entity == "Orders"
    (fk,) = entry.foreign_keys
    assert fk.columns == ["Customer ID", "Order Date"]
    assert fk.entity == "Customers"
    assert fk.reference_columns == ["Customer ID", "Order Date"]
    assert entry.domains == {"Status": "Order_Status"}


def test_per_file_entry_optional_parts_default_empty() -> None:
    spec = _load(_per_file(entities=[{"relation": "land.orders__Columns"}]))

    (entry,) = spec.entity_model.entities
    assert entry.header is None
    assert entry.entity is None
    assert entry.foreign_keys == []
    assert entry.domains == {}


def test_per_file_relations_are_listed_by_relation_refs() -> None:
    spec = _load(_per_file() | {"default_provider": "staging"})

    names = {(ref.provider, ref.name) for ref in spec.relation_refs()}
    assert ("staging", "land.orders__Columns") in names
    assert ("staging", "land.orders__Header") in names


def test_per_file_without_entities_is_rejected() -> None:
    raw = _per_file()
    del raw["entity_model"]["entities"]

    with pytest.raises(ValidationError, match="entities"):
        _load(raw)


def test_header_keys_load_without_text() -> None:
    spec = _load(_per_file(header_keys={"name": "Table Name"}))

    keys = spec.entity_model.header_keys
    assert keys.name == "Table Name"
    assert keys.description is None
    assert keys.text == []


def test_header_keys_load_with_text() -> None:
    header_keys = {
        "name": "Table Name",
        "description": "Table Description",
        "text": ["How does this table connect?"],
    }

    spec = _load(_per_file(header_keys=header_keys))

    keys = spec.entity_model.header_keys
    assert (keys.name, keys.description) == ("Table Name", "Table Description")
    assert keys.text == ["How does this table connect?"]


def test_header_keys_default_to_none() -> None:
    spec = _load(_per_file())

    assert spec.entity_model.header_keys is None


def test_notes_role_loads_as_a_list_and_defaults_empty() -> None:
    with_notes = _load(_per_file(roles={"property": "field", "notes": ["a", "b"]}))
    without = _load(_per_file())

    assert with_notes.entity_model.roles.notes == ["a", "b"]
    assert without.entity_model.roles.notes == []


@pytest.mark.parametrize("layout", ["single_relation", "per_file"])
def test_type_role_is_optional_in_both_layouts(layout: str) -> None:
    if layout == "per_file":
        raw = _per_file()
    else:
        roles = {"entity": "t", "property": "c"}
        raw = _bare(entity_model={"relation": "land.dict", "roles": roles})

    spec = _load(raw)

    assert spec.entity_model.roles.type is None


def test_entity_role_under_per_file_is_rejected() -> None:
    raw = _per_file(roles={"entity": "t", "property": "field"})

    with pytest.raises(ValidationError, match="per_file entries name their entity"):
        _load(raw)


def test_entity_role_is_required_under_single_relation() -> None:
    raw = _bare(entity_model={"relation": "land.dict", "roles": {"property": "c"}})

    with pytest.raises(ValidationError, match="entity"):
        _load(raw)


def test_property_role_is_required_under_per_file() -> None:
    with pytest.raises(ValidationError, match="property"):
        _load(_per_file(roles={}))


def test_code_list_without_concept_loads() -> None:
    spec = _load(_bare(code_lists=[_code_list()], default_provider="staging"))

    (entry,) = spec.code_lists
    assert entry.vocabulary_id == "Country_Codes"
    assert entry.relation.provider == "staging"
    assert entry.relation.name == "land.Country_Codes"
    assert (entry.roles.code, entry.roles.display) == ("code", "label")
    assert entry.concept is None
    assert entry.filter is None


def test_code_list_with_concept_loads() -> None:
    concept = {"concept_id": "country", "name": "Country"}

    spec = _load(_minimal(code_lists=[_code_list(concept=concept)]))

    concept_block = spec.code_lists[0].concept
    assert (concept_block.concept_id, concept_block.name) == ("country", "Country")
    assert concept_block.definition is None


def test_code_list_roles_accept_value_set_options_but_no_concept_id() -> None:
    options = {"code": "c", "display": "d", "display_fallback": "e", "synonyms": "s"}
    loaded = _load(_bare(code_lists=[_code_list(roles=options)]))
    with_concept_id = {"code": "c", "display": "d", "concept_id": "x"}

    assert loaded.code_lists[0].roles.display_fallback == "e"
    with pytest.raises(ValidationError, match="concept_id"):
        _load(_bare(code_lists=[_code_list(roles=with_concept_id)]))


def test_code_list_relations_are_listed_by_relation_refs() -> None:
    spec = _load(_bare(code_lists=[_code_list(), _code_list("State_Codes")]))

    names = {ref.name for ref in spec.relation_refs()}
    assert {"land.Country_Codes", "land.State_Codes"} <= names


def test_duplicate_code_list_vocabulary_id_is_rejected_naming_it() -> None:
    raw = _bare(code_lists=[_code_list(), _code_list(relation="land.other")])

    with pytest.raises(ValidationError, match="Country_Codes"):
        _load(raw)


def test_spec_with_only_code_lists_loads() -> None:
    spec = _load(_bare(code_lists=[_code_list()]))

    assert spec.concepts is None
    assert spec.entity_model is None
    assert spec.scheme is None
    assert len(spec.code_lists) == 1


def test_spec_with_none_of_the_three_sections_names_all_three() -> None:
    with pytest.raises(ValidationError, match="concepts.*entity_model.*code_lists"):
        _load(_bare())


def test_code_lists_on_version_1_spec_is_rejected() -> None:
    raw = _v1_raw()
    raw["code_lists"] = [_code_list()]

    with pytest.raises(ValidationError, match=r"code_lists.*spec_version: 2"):
        _load(raw)


def test_per_file_entity_model_on_version_1_spec_is_rejected() -> None:
    raw = _v1_raw()
    raw["entity_model"] = {
        "layout": "per_file",
        "entities": [{"relation": "land.orders__Columns"}],
        "roles": {"property": "field"},
    }

    with pytest.raises(ValidationError, match=r"entity_model.*spec_version: 2"):
        _load(raw)


def test_code_lists_is_a_version_2_field() -> None:
    from sema.binding.spec import VERSION_2_FIELDS

    assert ("", "code_lists") in VERSION_2_FIELDS


def test_new_section_models_are_importable_from_spec() -> None:
    import sema.binding.spec as spec_module

    names = (
        "PerFileEntry",
        "DeclaredForeignKey",
        "HeaderKeys",
        "CodeListSpec",
        "CodeListRoles",
        "CodeListConcept",
    )
    for name in names:
        assert name in spec_module.__all__
        assert hasattr(spec_module, name)
