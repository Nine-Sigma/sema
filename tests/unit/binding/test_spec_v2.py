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
