"""A per-file entry's identity reaches the target entity the graph loader reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.importer import import_concept_source, write_import
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.relations import RelationProvider
from sema.models.target.identity import EntityIdentity
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.normalizer import TargetModelNormalizer
from tests.unit.binding.test_importer_entity_model import (
    _create,
    _orders_entry,
    _per_file_import,
    _per_file_spec,
)

pytestmark = pytest.mark.unit


def _loaded_entities(result: Any, path: Path) -> dict[str, Any]:
    write_import(result, path, path.with_suffix(".evidence.json"))
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(path))
    return {e.ref.qualified_name: e for e in normalized.entities}


def test_a_per_file_entry_identity_reaches_its_entity(tmp_path: Path) -> None:
    identity = {"key_columns": ["Order ID"], "attributes": [{"column": "Status", "rule": "max"}]}
    entries = [_orders_entry(identity=identity), {"relation": "land.customers__Columns"}]

    result = _per_file_import(tmp_path, _per_file_spec(entries))
    entities = _loaded_entities(result, tmp_path / "manifest.yaml")

    assert entities["ex.customer_orders"].identity == EntityIdentity.model_validate(identity)
    assert entities["ex.customers"].identity is None


LEGALS_COLUMNS = [
    ("Document ID", "Unique identifier of the recorded document", ""),
    ("Good Through Date", "Date the record is current through", ""),
    ("Lot", "Tax lot of the parcel", "Blank when unrecorded"),
]
LEGALS_DOCUMENT = """
relation: land.Real_Property_Legals__Columns
entity: Document
identity:
  key_columns: [Document ID, Good Through Date]
  attributes:
    - {column: Lot, rule: min, name: first_lot}
"""


def test_an_acris_document_identity_from_a_spec_snippet_reaches_its_entity(
    tmp_path: Path,
) -> None:
    entry = yaml.safe_load(LEGALS_DOCUMENT)
    with Staging(str(tmp_path / "legals.duckdb"), schemas=()) as staging:
        staging.execute("CREATE SCHEMA IF NOT EXISTS land")
        _create(staging, "Real_Property_Legals__Columns", ("field", "meaning", "remarks"),
                LEGALS_COLUMNS)  # fmt: skip
        provider = RelationProvider(warehouse=None, staging=staging)
        result = import_concept_source(_per_file_spec([entry]), provider, dataset_map={})

    entity = _loaded_entities(result, tmp_path / "manifest.yaml")["ex.Document"]

    assert entity.identity == EntityIdentity.model_validate(entry["identity"])
    assert [p.name for p in entity.properties] == ["Document ID", "Good Through Date", "Lot"]
