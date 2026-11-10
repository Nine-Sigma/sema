"""The per-file entity identity is settled by code: it adds no role and moves no call key.

The snapshot was taken on the tree before ``PerFileEntry.identity`` existed (c9ec826).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from sema.binding.spec_sections import EntityModelRoles, PerFileEntry
from sema.intake.output_models import role_map_model

pytestmark = pytest.mark.unit

ROLE_MAP_NAME = "RoleMapEntityModelRoles_v1"
ROLE_MAP_FIELDS = [
    "role_property", "role_type", "role_required", "role_primary_key", "role_fk_entity",
    "role_fk_property", "role_domain", "role_description", "role_entity_description",
    "confidences", "reasons",
]  # fmt: skip
ROLE_MAP_SCHEMA_SHA256 = "32a704ca71e6f79cbc6b42eb011c636750c6cff10e2c713ff6b06ce972ed8c0b"
PER_FILE_DUMP_KEYS = ["domains", "entity", "foreign_keys", "header", "relation"]


def test_the_entity_model_role_map_is_unchanged_by_the_per_file_identity() -> None:
    model = role_map_model({"entity_model": EntityModelRoles})
    schema = json.dumps(model.model_json_schema())
    assert model.__name__ == ROLE_MAP_NAME
    assert list(model.model_fields) == ROLE_MAP_FIELDS
    assert hashlib.sha256(schema.encode()).hexdigest() == ROLE_MAP_SCHEMA_SHA256


def test_a_per_file_entry_without_an_identity_dumps_as_before() -> None:
    dumped = PerFileEntry(relation="land.orders__Columns").model_dump(mode="json", by_alias=True)
    assert sorted(dumped) == PER_FILE_DUMP_KEYS
