"""Entity-level joins and domains survive manifest normalization (Decision 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.hashing import SnapshotHasher
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.unit

CHILD = "ns.child"
PARENT = "ns.parent"


def _manifest(child_extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "descriptor": {
            "target_model_id": "joins-fixture",
            "target_model_version": "1",
            "display_name": "Joins Fixture",
        },
        "entities": [
            {
                "qualified_name": PARENT,
                "kind": "TABLE_ROW",
                "properties": [{"name": "id", "type": "string"}],
            },
            {
                "qualified_name": CHILD,
                "kind": "TABLE_ROW",
                "properties": [
                    {"name": "parent_id", "type": "string"},
                    {"name": "status", "type": "string"},
                ],
                **child_extra,
            },
        ],
    }


def _normalize(tmp_path: Path, child_extra: dict[str, Any]) -> Any:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_manifest(child_extra)))
    return TargetModelNormalizer.normalize(ManifestTargetAdapter(path))


def test_entity_joins_and_domains_survive_normalization(tmp_path: Path) -> None:
    declared = {
        "joins": [
            {
                "referenced_entity": PARENT,
                "join_keys": [["parent_id", "id"]],
                "evidence": "declared",
            }
        ],
        "domains": [
            {"property_name": "status", "domain_id": "status_codes", "evidence": "model"}
        ],
    }

    normalized = _normalize(tmp_path, declared)

    by_name = {e.ref.qualified_name: e for e in normalized.entities}
    child = by_name[CHILD]
    assert [
        (j.referenced_entity, j.join_keys, j.evidence) for j in child.declared_joins
    ] == [(PARENT, [("parent_id", "id")], "declared")]
    assert [
        (d.property_name, d.domain_id, d.evidence) for d in child.declared_domains
    ] == [("status", "status_codes", "model")]
    assert by_name[PARENT].declared_joins == []
    assert by_name[PARENT].declared_domains == []
    assert SnapshotHasher.hash(normalized) != SnapshotHasher.hash(
        _normalize(tmp_path, {})
    )
