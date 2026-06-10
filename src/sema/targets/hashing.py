"""SnapshotHasher and card_hash computation.

Owns `target_schema_snapshot_hash`. Adapters MUST NOT compute or return
snapshot hashes; the normalizer rejects DTOs carrying a literal field
named `snapshot_hash`.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from sema.models.target.context_card import TargetContextCard
from sema.models.target.normalized import NormalizedTargetModel
from sema.targets.exceptions import AdapterContractError
from sema.targets.hashing_utils import canonical_dumps


_DESCRIPTOR_NON_SCHEMA_FIELDS: tuple[str, ...] = ("display_name", "owner")


def _project_descriptor(descriptor: BaseModel) -> dict[str, object]:
    raw = descriptor.model_dump(mode="json")
    for field in _DESCRIPTOR_NON_SCHEMA_FIELDS:
        raw.pop(field, None)
    return raw


def _project_schema_bearing(model: NormalizedTargetModel) -> dict[str, object]:
    return {
        "descriptor": _project_descriptor(model.descriptor),
        "entities": [e.model_dump(mode="json") for e in model.entities],
        "obligations": [o.model_dump(mode="json") for o in model.obligations],
        "vocabularies": [v.model_dump(mode="json") for v in model.vocabularies],
        "vocabulary_bindings": [
            b.model_dump(mode="json") for b in model.vocabulary_bindings
        ],
        "terms": [t.model_dump(mode="json") for t in model.terms],
    }


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scan_for_snapshot_hash_field(value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        if "snapshot_hash" in value:
            raise AdapterContractError(
                f"DTO at {path} carries forbidden field 'snapshot_hash'; "
                f"snapshot hashing is owned by SnapshotHasher"
            )
        for key, item in value.items():
            _scan_for_snapshot_hash_field(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _scan_for_snapshot_hash_field(item, f"{path}[{i}]")


class SnapshotHasher:
    """Deterministic SHA-256 over the schema-bearing projection of a model."""

    @staticmethod
    def hash(model: NormalizedTargetModel) -> str:
        projection = _project_schema_bearing(model)
        _scan_for_snapshot_hash_field(projection)
        return _sha256_hex(canonical_dumps(projection))


_CARD_CONTENT_FIELDS: tuple[str, ...] = (
    "description",
    "examples",
    "obligation_summary",
    "curated_synonyms",
)


def compute_card_hash(card: TargetContextCard) -> str:
    payload = {field: getattr(card, field) for field in _CARD_CONTENT_FIELDS}
    return _sha256_hex(canonical_dumps(payload))
