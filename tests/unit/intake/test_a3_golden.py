"""A3: land the semsql fixture and import it with the hand-written spec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_parser import parse_manifest_raw

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[2] / "fixtures" / "intake" / "semsql"
DB = FIXTURES / "ontology.db"
SPEC = FIXTURES / "ontology.spec.yaml"
GOLDEN = FIXTURES / "ontology.manifest.yaml"


def import_a3(tmp_path: Path) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec
    from sema.intake.land.loaders.semsql import land_semsql
    from sema.intake.land.relations import RelationProvider

    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land_semsql(DB, staging, schema="land_a3")
        return import_concept_source(
            load_concept_source_spec(SPEC),
            RelationProvider(warehouse=None, staging=staging),
            dataset_map={},
            staging=staging,
        )


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return import_a3(tmp_path_factory.mktemp("a3"))


def test_sidecar_holds_exactly_the_two_deprecated_classes(result: Any) -> None:
    retirements = [
        (r.kind, r.code, r.replaced_by) for r in result.authored.retirements
    ]

    assert retirements == [("term", "EX:0049", "EX:0002"), ("term", "EX:0050", "EX:0003")]


def test_manifest_holds_neither_retired_term(result: Any) -> None:
    codes = {t["code"] for t in result.manifest["terms"]}

    assert len(codes) == 48
    assert not codes & {"EX:0049", "EX:0050"}


def test_close_match_is_kept_whole_with_its_predicate(result: Any) -> None:
    mappings = {(m.subject_id, m.predicate_id, m.object_id) for m in result.authored.mappings}

    assert ("EX:0004", "skos:closeMatch", "NCIT:C1002") in mappings
    assert ("EX:0002", "skos:exactMatch", "NCIT:C1000") in mappings
    assert ("EX:0005", "oio:hasDbXref", "MESH:D000001") in mappings


def test_exact_match_term_carries_the_external_ref(result: Any) -> None:
    terms = {t["code"]: t for t in result.manifest["terms"]}

    assert terms["EX:0002"]["external_refs"] == [{"system": "NCIT", "id": "C1000"}]
    assert terms["EX:0004"]["external_refs"] == []


def test_subclass_edges_are_hierarchical_relationships(result: Any) -> None:
    edges = result.manifest["relationships"]

    assert len(edges) == 47
    assert all(e["hierarchical"] and e["far_end_loaded"] for e in edges)
    assert {"from": {"system": "EX", "id": "EX:0002"}, "to": {"system": "EX", "id": "EX:0001"},
            "type": "rdfs:subClassOf", "hierarchical": True, "far_end_loaded": True} in edges


def test_manifest_equals_golden(result: Any) -> None:
    assert yaml.safe_dump(result.manifest, sort_keys=False) == GOLDEN.read_text()
    parse_manifest_raw(yaml.safe_load(GOLDEN.read_text()))
