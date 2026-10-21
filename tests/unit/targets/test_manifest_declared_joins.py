"""Entity-level declared joins and domains on the manifest (design Decision 7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sema.intake.table_schema import TableSchema, to_manifest_entity
from sema.targets.adapters import manifest_models
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.adapters.manifest_models import ManifestEntity
from sema.targets.adapters.manifest_parser import (
    _load_raw,
    parse_manifest_file,
    parse_manifest_raw,
)
from sema.targets.hashing import SnapshotHasher
from sema.targets.normalizer import TargetModelNormalizer
from tests.unit.intake.test_acris_import import (
    DOMAINS,
    ENTITIES,
    FOREIGN_KEYS,
    JOIN_COLUMNS,
    _Imported,
    imported,
)

pytestmark = pytest.mark.unit

__all__ = ["imported"]

REPO = Path(__file__).parents[3]
GOLDEN_HASHES: dict[str, str | None] = {
    "showcase/cbioportal_to_omop/manifests/omop_condition_slice0.yaml": (
        "dd05541e2a2307b8fe5b47364e0af1fc330e664e4ce761951353a6058e401546"
    ),
    "src/sema/targets/manifests/dim_customer.yaml": (
        "86dc149e10e5571709bbfd853d609c321e88a96c5bbc70c64916a0cc100960e2"
    ),
    "tests/fixtures/intake/semsql/ontology.manifest.yaml": (
        "7ffcdf97e922f1909ee948a1f0fed3b6ab7e1b1ddaa1ca5b5a9f71e7cf16330e"
    ),
    "tests/fixtures/intake/tabular/codes_a2.manifest.yaml": (
        "172bf155fff971e9249c4b8196da45d3f2cea66e6d0a4bf7602dc7128b67916f"
    ),
    "tests/unit/binding/fixtures/golden/syn_library_v1.manifest.yaml": (
        "3a5c3a1f37ea879a71cd5107382d023915c6ab7144c6c7a74487229a99986212"
    ),
    "tests/unit/targets/fixtures/concept_manifest.yaml": (
        "593bc64124d748c3533d172a1d8d5ae2e00f5cbc6af53925b08ffd411586be7d"
    ),
    "tests/unit/targets/fixtures/golden_manifest.yaml": (
        "e8c9bfdc1026f1593686d197047c0d49acbd97db343badb73d0bba847241644b"
    ),
}
NAMESPACE = "ns"
CHILD = f"{NAMESPACE}.child"
PARENT = f"{NAMESPACE}.parent"
DOMAIN_ID = "status_codes"


def _entity(**extra: Any) -> dict[str, Any]:
    return {
        "qualified_name": CHILD,
        "kind": "TABLE_ROW",
        "properties": [
            {"name": "parent_id", "type": "string"},
            {"name": "status", "type": "string"},
        ],
        **extra,
    }


def _join(**overrides: Any) -> dict[str, Any]:
    return {
        "referenced_entity": PARENT,
        "join_keys": [["parent_id", "id"]],
        "evidence": "declared",
        **overrides,
    }


def _domain(**overrides: Any) -> dict[str, Any]:
    return {
        "property_name": "status",
        "domain_id": DOMAIN_ID,
        "evidence": "declared",
        **overrides,
    }


def _schema(*, primary_key: bool, required: bool) -> TableSchema:
    constraints = {"required": True} if required else None
    descriptor: dict[str, Any] = {
        "fields": [
            {"name": "id", "type": "string", "constraints": constraints},
            {"name": "parent_id", "type": "string", "constraints": constraints},
            {"name": "status", "type": "string", "x-sema-domain": DOMAIN_ID},
        ],
        "foreignKeys": [
            {"fields": ["parent_id"], "reference": {"resource": "parent", "fields": ["id"]}}
        ],
    }
    if primary_key:
        descriptor["primaryKey"] = ["id"]
    return TableSchema.model_validate(descriptor)


def _golden_ids() -> list[str]:
    return list(GOLDEN_HASHES)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _acris_entities(imported: _Imported) -> list[ManifestEntity]:
    raw = yaml.safe_load(yaml.safe_dump(imported.result.manifest))
    return parse_manifest_raw(raw).entities


def _short(qualified_name: str) -> str:
    return qualified_name.split(".", 1)[1]


def test_entity_accepts_joins_and_domains_with_evidence() -> None:
    entity = ManifestEntity.model_validate(
        _entity(joins=[_join()], domains=[_domain(evidence="model")])
    )

    [join] = entity.joins
    [domain] = entity.domains
    assert isinstance(join, manifest_models.ManifestDeclaredJoin)
    assert isinstance(domain, manifest_models.ManifestDeclaredDomain)
    assert (join.referenced_entity, join.join_keys, join.evidence) == (
        PARENT,
        [("parent_id", "id")],
        "declared",
    )
    assert (domain.property_name, domain.domain_id, domain.evidence) == (
        "status",
        DOMAIN_ID,
        "model",
    )
    assert entity.obligation is None


def test_joins_and_domains_default_to_empty() -> None:
    entity = ManifestEntity.model_validate(_entity())

    assert entity.joins == []
    assert entity.domains == []


@pytest.mark.parametrize(
    "field,entry",
    [
        ("joins", _join(evidence="inferred")),
        ("joins", {k: v for k, v in _join().items() if k != "evidence"}),
        ("joins", _join(join_keys=[])),
        ("joins", _join(on_delete="cascade")),
        ("domains", _domain(evidence="guessed")),
        ("domains", {k: v for k, v in _domain().items() if k != "evidence"}),
        ("domains", _domain(required=True)),
    ],
    ids=[
        "join-unknown-evidence",
        "join-missing-evidence",
        "join-no-keys",
        "join-extra-key",
        "domain-unknown-evidence",
        "domain-missing-evidence",
        "domain-extra-key",
    ],
)
def test_invalid_join_or_domain_is_rejected(field: str, entry: dict[str, Any]) -> None:
    assert field in ManifestEntity.model_fields
    with pytest.raises(ValidationError):
        ManifestEntity.model_validate(_entity(**{field: [entry]}))


def test_entity_joins_are_independent_of_obligation_keys() -> None:
    obligation = {
        "required_fields": ["parent_id"],
        "primary_key": "NATURAL_KEY",
        "foreign_keys": [{"referenced_entity": PARENT, "join_keys": [["parent_id", "id"]]}],
        "domain_constraints": [{"property_name": "status", "domain_id": DOMAIN_ID}],
    }
    other = f"{NAMESPACE}.other"
    entity = ManifestEntity.model_validate(
        _entity(obligation=obligation, joins=[_join(referenced_entity=other)])
    )

    assert entity.obligation is not None
    assert [fk.referenced_entity for fk in entity.obligation.foreign_keys] == [PARENT]
    assert [dc.domain_id for dc in entity.obligation.domain_constraints] == [DOMAIN_ID]
    assert [join.referenced_entity for join in entity.joins] == [other]


@pytest.mark.parametrize("golden", _golden_ids())
def test_golden_parses_unchanged(golden: str) -> None:
    path = REPO / golden
    raw = _json(_load_raw(path))

    parsed = parse_manifest_file(path)

    dumped = parsed.model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert dumped == raw
    assert parse_manifest_raw(dumped) == parsed
    for entity, raw_entity in zip(parsed.entities, raw["entities"], strict=True):
        obligation = entity.obligation
        assert (
            obligation.model_dump(mode="json", exclude_unset=True)
            if obligation is not None
            else None
        ) == raw_entity.get("obligation")
        assert entity.joins == []
        assert entity.domains == []


@pytest.mark.parametrize(
    "golden", [name for name, digest in GOLDEN_HASHES.items() if digest is not None]
)
def test_golden_normalized_hash_is_unchanged(golden: str) -> None:
    adapter = ManifestTargetAdapter(REPO / golden)

    digest = SnapshotHasher.hash(TargetModelNormalizer.normalize(adapter))

    assert digest == GOLDEN_HASHES[golden]


def test_acris_import_yields_eight_entity_level_joins(imported: _Imported) -> None:
    entities = _acris_entities(imported)

    found = [
        (_short(entity.qualified_name), _short(join.referenced_entity), join)
        for entity in entities
        for join in entity.joins
    ]
    assert len(found) == 8
    assert {(owner, target) for owner, target, _ in found} == FOREIGN_KEYS
    pairs = [(column, column) for column in JOIN_COLUMNS]
    assert all(join.join_keys == pairs for _, _, join in found)
    assert {join.evidence for _, _, join in found} == {"declared"}


def test_acris_import_yields_nine_entity_level_domains(imported: _Imported) -> None:
    entities = _acris_entities(imported)

    found = [
        (_short(entity.qualified_name), domain.property_name, domain.domain_id)
        for entity in entities
        for domain in entity.domains
    ]
    assert len(found) == 9
    assert set(found) == DOMAINS
    evidence = {domain.evidence for entity in entities for domain in entity.domains}
    assert evidence == {"declared"}


def test_acris_entities_have_no_obligation(imported: _Imported) -> None:
    entities = _acris_entities(imported)

    assert len(entities) == ENTITIES
    assert [e.qualified_name for e in entities if e.obligation is not None] == []


def test_obligation_entity_keeps_foreign_keys_in_obligation() -> None:
    raw = to_manifest_entity(
        _schema(primary_key=True, required=True), CHILD, namespace=NAMESPACE
    )

    entity = ManifestEntity.model_validate(raw)

    assert entity.obligation is not None
    assert [
        (fk.referenced_entity, fk.join_keys) for fk in entity.obligation.foreign_keys
    ] == [(PARENT, [("parent_id", "id")])]
    assert [
        (dc.property_name, dc.domain_id) for dc in entity.obligation.domain_constraints
    ] == [("status", DOMAIN_ID)]
    assert entity.joins == []
    assert entity.domains == []


@pytest.mark.parametrize(
    "primary_key,required",
    [(False, True), (True, False)],
    ids=["no-primary-key", "no-required-field"],
)
def test_entity_without_obligation_carries_declared_joins_and_domains(
    primary_key: bool, required: bool
) -> None:
    raw = to_manifest_entity(
        _schema(primary_key=primary_key, required=required), CHILD, namespace=NAMESPACE
    )

    entity = ManifestEntity.model_validate(raw)

    assert entity.obligation is None
    assert [
        (join.referenced_entity, join.join_keys, join.evidence) for join in entity.joins
    ] == [(PARENT, [("parent_id", "id")], "declared")]
    assert [
        (domain.property_name, domain.domain_id, domain.evidence)
        for domain in entity.domains
    ] == [("status", DOMAIN_ID, "declared")]
