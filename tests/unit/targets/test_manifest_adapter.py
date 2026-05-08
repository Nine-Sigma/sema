"""ManifestTargetAdapter parser + adapter behavior."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from sema.models.planner._enums import PrimaryKeyStrategy, TargetArtifactKind
from sema.models.target.completeness import SemanticCompleteness
from sema.models.target.refs import VocabularyRef, VocabularySource
from sema.targets import TargetOntologyAdapter
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.adapters.manifest_exceptions import (
    ManifestContextCardError,
    ManifestEndpointError,
    ManifestReservedNameError,
    ManifestSchemaError,
    UnsupportedManifestExtensionError,
    UnsupportedManifestVersionError,
)
from sema.targets.loader import load_target
from sema.targets.materializer import InMemoryGraphWriter

pytestmark = pytest.mark.unit


def _write_yaml(tmp_path: Path, content: dict, name: str = "manifest.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content))
    return path


def _write_json(tmp_path: Path, content: dict, name: str = "manifest.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(content))
    return path


def _minimal_manifest() -> dict:
    return {
        "manifest_version": 1,
        "descriptor": {
            "target_model_id": "fake-target",
            "target_model_version": "1.0.0",
            "display_name": "Fake",
        },
        "entities": [
            {
                "qualified_name": "fake.person",
                "kind": "TABLE_ROW",
                "properties": [
                    {"name": "person_id", "type": "integer", "nullable": False}
                ],
                "obligation": {
                    "required_fields": ["person_id"],
                    "primary_key": "NATURAL_KEY",
                },
            }
        ],
    }


def test_minimal_yaml_manifest_parses(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _minimal_manifest())
    adapter = ManifestTargetAdapter(path)
    refs = list(adapter.discover_entities())
    assert len(refs) == 1
    assert refs[0].qualified_name == "fake.person"


def test_yaml_and_json_manifests_produce_equal_parsed_manifest(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    yaml_path = _write_yaml(tmp_path, payload)
    json_path = _write_json(tmp_path, payload)
    yaml_adapter = ManifestTargetAdapter(yaml_path)
    json_adapter = ManifestTargetAdapter(json_path)
    assert yaml_adapter.parsed_manifest == json_adapter.parsed_manifest


def test_unknown_manifest_version_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["manifest_version"] = 99
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(UnsupportedManifestVersionError, match="99"):
        ManifestTargetAdapter(path)


def test_missing_descriptor_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    del payload["descriptor"]
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestSchemaError, match="descriptor"):
        ManifestTargetAdapter(path)


def test_unknown_extension_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.txt"
    path.write_text("not a manifest")
    with pytest.raises(UnsupportedManifestExtensionError):
        ManifestTargetAdapter(path)


def test_full_dto_coverage_from_rich_manifest(tmp_path: Path) -> None:
    payload = _rich_manifest()
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    writer = InMemoryGraphWriter()
    loaded = load_target(adapter, writer=writer)
    assert len(loaded.entity_refs) >= 2


def test_llm_client_invoke_never_called_during_parsing(tmp_path: Path) -> None:
    spy = MagicMock()
    path = _write_yaml(tmp_path, _minimal_manifest())
    ManifestTargetAdapter(path)
    assert spy.invoke.call_count == 0


def test_external_vocabulary_binding_yields_no_terms(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["vocabularies"] = [{"name": "SNOMED", "source": "EXTERNAL"}]
    payload["entities"][0]["properties"][0]["vocabulary_binding"] = {
        "vocabulary": "SNOMED",
        "domain": "Identifier",
    }
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    vocab = VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL)
    with pytest.raises(NotImplementedError):
        next(adapter.iter_terms(vocab))


def test_inline_terms_produce_term_decls(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["vocabularies"] = [{"name": "GENDER_CV", "source": "INLINE"}]
    payload["terms"] = [
        {"vocabulary": "GENDER_CV", "code": "M", "display": "Male"},
        {"vocabulary": "GENDER_CV", "code": "F", "display": "Female"},
    ]
    payload["entities"][0]["properties"][0]["vocabulary_binding"] = {
        "vocabulary": "GENDER_CV",
        "domain": "Gender",
    }
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    vocab = VocabularyRef(name="GENDER_CV", source=VocabularySource.INLINE)
    inline_terms = list(adapter.iter_terms(vocab))
    assert {t.code for t in inline_terms} == {"M", "F"}


def test_descriptor_default_completeness_when_omitted(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _minimal_manifest())
    adapter = ManifestTargetAdapter(path)
    descriptor = adapter.describe()
    assert descriptor.completeness.structure is SemanticCompleteness.COMPLETE
    assert descriptor.completeness.obligations is SemanticCompleteness.COMPLETE
    assert descriptor.completeness.vocabulary_bindings is SemanticCompleteness.PARTIAL
    assert descriptor.completeness.semantic_aliases is SemanticCompleteness.PARTIAL
    assert descriptor.completeness.terms is SemanticCompleteness.EXTERNAL


def test_explicit_descriptor_completeness_overrides_default(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["descriptor"]["completeness"] = {
        "structure": "COMPLETE",
        "obligations": "COMPLETE",
        "vocabulary_bindings": "COMPLETE",
        "semantic_aliases": "COMPLETE",
        "terms": "COMPLETE",
    }
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    descriptor = adapter.describe()
    assert all(
        getattr(descriptor.completeness, f) is SemanticCompleteness.COMPLETE
        for f in (
            "structure",
            "obligations",
            "vocabulary_bindings",
            "semantic_aliases",
            "terms",
        )
    )


def test_synonyms_upgrade_entity_semantic_aliases_to_complete(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"][0]["properties"][0]["synonyms"] = ["alias"]
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    entity = adapter.load_entity(next(iter(adapter.discover_entities())))
    assert entity.completeness is not None
    assert entity.completeness.semantic_aliases is SemanticCompleteness.COMPLETE


def test_inline_terms_upgrade_entity_terms_to_complete(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["vocabularies"] = [{"name": "GENDER_CV", "source": "INLINE"}]
    payload["terms"] = [{"vocabulary": "GENDER_CV", "code": "M", "display": "Male"}]
    payload["entities"][0]["properties"][0]["vocabulary_binding"] = {
        "vocabulary": "GENDER_CV"
    }
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    entity = adapter.load_entity(next(iter(adapter.discover_entities())))
    assert entity.completeness is not None
    assert entity.completeness.terms is SemanticCompleteness.COMPLETE


def test_table_row_with_endpoints_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"][0]["endpoints"] = {
        "subject": {"target_entity": "x"},
        "object": {"target_entity": "y"},
    }
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestEndpointError, match="forbids endpoints"):
        ManifestTargetAdapter(path)


def test_graph_edge_missing_endpoints_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"] = [
        {
            "qualified_name": "fake.OWNS",
            "kind": "GRAPH_EDGE",
            "properties": [],
            "obligation": {
                "required_fields": ["subject"],
                "primary_key": "NATURAL_KEY",
            },
        }
    ]
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestEndpointError, match="requires endpoints"):
        ManifestTargetAdapter(path)


def test_graph_edge_endpoint_targets_table_row_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"].append(
        {
            "qualified_name": "fake.OWNS",
            "kind": "GRAPH_EDGE",
            "endpoints": {
                "subject": {"target_entity": "fake.person"},
                "object": {"target_entity": "fake.person"},
            },
            "properties": [],
            "obligation": {
                "required_fields": ["subject"],
                "primary_key": "NATURAL_KEY",
            },
        }
    )
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestEndpointError, match="TABLE_ROW"):
        ManifestTargetAdapter(path)


def test_reserved_property_name_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"][0]["properties"].append(
        {"name": "subject", "type": "string", "nullable": False}
    )
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestReservedNameError, match="reserved"):
        ManifestTargetAdapter(path)


def test_missing_card_version_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"][0]["context_card"] = {"description": "Hello"}
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestSchemaError, match="card_version"):
        ManifestTargetAdapter(path)


def test_card_hash_in_manifest_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"][0]["context_card"] = {
        "card_version": "1.0.0",
        "description": "Hello",
        "card_hash": "deadbeef",
    }
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestContextCardError, match="card_hash"):
        ManifestTargetAdapter(path)


def test_empty_card_description_rejected(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["entities"][0]["context_card"] = {
        "card_version": "1.0.0",
        "description": "",
    }
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ManifestSchemaError, match="description"):
        ManifestTargetAdapter(path)


def test_synthesized_card_when_block_omitted(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _minimal_manifest())
    adapter = ManifestTargetAdapter(path)
    ref = next(iter(adapter.discover_entities()))
    card = adapter.load_context_card(ref)
    assert card.card_version == "0.0.0+synthesized"
    assert card.description.startswith("Auto-generated card for fake.person")


def test_graph_edge_endpoints_round_trip(tmp_path: Path) -> None:
    payload = _acris_manifest()
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    refs = {r.qualified_name: r for r in adapter.discover_entities()}
    edge_entity = adapter.load_entity(refs["acris.OWNS"])
    assert edge_entity.endpoints is not None
    assert edge_entity.endpoints.subject.target_entity.qualified_name == "acris.LLC"
    assert edge_entity.endpoints.subject.cardinality == "one"
    assert edge_entity.endpoints.subject.nullable is False


def test_graph_edge_minimum_viable_row_default_is_subject_and_object(
    tmp_path: Path,
) -> None:
    payload = _acris_manifest()
    payload["entities"][2]["obligation"] = {
        "required_fields": ["subject", "object"],
        "primary_key": "NATURAL_KEY",
    }
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    obligation = adapter.load_obligation(
        next(r for r in adapter.discover_entities() if r.qualified_name == "acris.OWNS")
    )
    minimum = obligation.minimum_viable_row
    assert minimum is not None
    fields = sorted(c.field for c in minimum.clauses)
    assert fields == ["object", "subject"]


def test_graph_edge_explicit_temporal_minimum_honored(tmp_path: Path) -> None:
    payload = _acris_manifest()
    payload["entities"][2]["obligation"] = {
        "required_fields": ["subject", "object", "valid_from"],
        "primary_key": "NATURAL_KEY",
        "minimum_viable_row": {
            "op": "AND",
            "clauses": [
                {"kind": "presence", "field": "subject"},
                {"kind": "presence", "field": "object"},
                {"kind": "presence", "field": "valid_from"},
            ],
        },
    }
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    obligation = adapter.load_obligation(
        next(r for r in adapter.discover_entities() if r.qualified_name == "acris.OWNS")
    )
    minimum = obligation.minimum_viable_row
    assert minimum is not None
    fields = sorted(c.field for c in minimum.clauses)
    assert fields == ["object", "subject", "valid_from"]


def test_vocabulary_binding_round_trip(tmp_path: Path) -> None:
    payload = _minimal_manifest()
    payload["vocabularies"] = [{"name": "SNOMED", "source": "EXTERNAL"}]
    payload["entities"][0]["properties"][0]["vocabulary_binding"] = {
        "vocabulary": "SNOMED",
        "domain": "Gender",
        "require_standard": True,
        "allow_zero_default": False,
        "resolver_policy_ref": "policy-A",
    }
    path = _write_yaml(tmp_path, payload)
    adapter = ManifestTargetAdapter(path)
    ref = next(iter(adapter.discover_entities()))
    from sema.models.target.refs import TargetPropertyRef

    bindings = list(
        adapter.load_vocabulary_bindings(
            TargetPropertyRef(entity_ref=ref, property_name="person_id")
        )
    )
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.vocabulary.name == "SNOMED"
    assert binding.vocabulary.source is VocabularySource.EXTERNAL
    assert binding.domain == "Gender"
    assert binding.require_standard is True
    assert binding.resolver_policy_ref == "policy-A"


def test_adapter_satisfies_runtime_protocol(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _minimal_manifest())
    adapter = ManifestTargetAdapter(path)
    assert isinstance(adapter, TargetOntologyAdapter)


def test_load_unknown_entity_raises_keyerror(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _minimal_manifest())
    adapter = ManifestTargetAdapter(path)
    bogus = next(iter(adapter.discover_entities())).model_copy(
        update={"qualified_name": "fake.unknown"}
    )
    with pytest.raises(KeyError, match="fake.unknown"):
        adapter.load_entity(bogus)


def _acris_manifest() -> dict:
    return {
        "manifest_version": 1,
        "descriptor": {
            "target_model_id": "acris-nyc",
            "target_model_version": "1.0.0",
            "display_name": "ACRIS NYC",
        },
        "entities": [
            {
                "qualified_name": "acris.LLC",
                "kind": "GRAPH_NODE",
                "properties": [{"name": "name", "type": "string", "nullable": False}],
                "obligation": {
                    "required_fields": ["name"],
                    "primary_key": "NATURAL_KEY",
                },
            },
            {
                "qualified_name": "acris.Property",
                "kind": "GRAPH_NODE",
                "properties": [{"name": "name", "type": "string", "nullable": False}],
                "obligation": {
                    "required_fields": ["name"],
                    "primary_key": "NATURAL_KEY",
                },
            },
            {
                "qualified_name": "acris.OWNS",
                "kind": "GRAPH_EDGE",
                "endpoints": {
                    "subject": {
                        "target_entity": "acris.LLC",
                        "cardinality": "one",
                        "nullable": False,
                    },
                    "object": {
                        "target_entity": "acris.Property",
                        "cardinality": "one",
                        "nullable": False,
                    },
                },
                "properties": [
                    {"name": "valid_from", "type": "date", "nullable": False}
                ],
                "obligation": {
                    "required_fields": ["subject", "object", "valid_from"],
                    "primary_key": "NATURAL_KEY",
                    "minimum_viable_row": {
                        "op": "AND",
                        "clauses": [
                            {"kind": "presence", "field": "subject"},
                            {"kind": "presence", "field": "object"},
                            {"kind": "presence", "field": "valid_from"},
                        ],
                    },
                },
            },
        ],
    }


def _rich_manifest() -> dict:
    payload = _minimal_manifest()
    payload["entities"].append(
        {
            "qualified_name": "fake.observation",
            "kind": "TABLE_ROW",
            "properties": [
                {"name": "obs_id", "type": "integer", "nullable": False},
            ],
            "obligation": {
                "required_fields": ["obs_id"],
                "primary_key": "NATURAL_KEY",
            },
            "context_card": {
                "card_version": "1.0.0",
                "description": "Observation table.",
            },
        }
    )
    payload["vocabularies"] = [{"name": "SNOMED", "source": "EXTERNAL"}]
    return payload
