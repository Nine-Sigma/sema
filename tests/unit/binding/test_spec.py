"""ConceptSourceSpec loads the design 5.2 spec shape from YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.binding.fixtures.concept_library import DATASET_MAP_PATH, SPEC_PATH

pytestmark = pytest.mark.unit


def test_spec_yaml_loads_into_concept_source_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec

    # A cwd away from the fixture proves dataset_map resolves against the spec file.
    monkeypatch.chdir(tmp_path)
    spec = load_concept_source_spec(SPEC_PATH)

    assert isinstance(spec, ConceptSourceSpec)
    assert spec.id == "syn-library"
    assert spec.identity_system == "syn"
    assert spec.release == "2026-09"
    assert spec.scheme.qualified_name == "syn_library.demographics"
    assert spec.concepts.relation == "lib.data_element"
    assert spec.concepts.group_by == "data_element_id"
    assert spec.concepts.roles.definition == "data_element_definition"
    assert spec.value_sets.roles.code == "value_id"
    assert spec.concept_details.on == ["data_element_id", "concept_id"]
    assert spec.term_details.roles.definition == "definition"
    assert spec.relationships.relation == "lib.concept_relationship"
    assert spec.authored_bindings.roles.columns == "source_columns"
    assert (
        Path(spec.authored_bindings.dataset_map).resolve() == DATASET_MAP_PATH.resolve()
    )
    assert spec.value_mappings.approved_statuses == ["Reviewed"]
