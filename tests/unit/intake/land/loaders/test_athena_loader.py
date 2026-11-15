"""land_athena_bundle: the Athena bundle loader in the core intake package."""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging
from tests.support.athena_bundle import export_schema, make_minimal_bundle
from tests.unit.intake.athena_golden_env import GOLDEN

pytestmark = pytest.mark.unit

SCHEMA = "vocabulary_omop"


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    handle = Staging(str(tmp_path / "vocab.duckdb"), schemas=())
    yield handle
    handle.close()


def _land(bundle: Path, staging: Staging):  # type: ignore[no-untyped-def]
    from sema.intake.land.loaders.athena import land_athena_bundle

    return land_athena_bundle(bundle, staging)


def _tables(staging: Staging) -> list[str]:
    rows = staging.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = ? ORDER BY 1", [SCHEMA]
    ).fetchall()
    return [row[0] for row in rows]


def test_loader_output_equals_the_pre_move_golden(tmp_path: Path, staging: Staging) -> None:
    bundle = tmp_path / "athena"
    make_minimal_bundle(bundle, with_synonyms=True)

    _land(bundle, staging)

    assert export_schema(staging) == GOLDEN.read_text()


def test_tables_carry_no_locator_column(tmp_path: Path, staging: Staging) -> None:
    bundle = tmp_path / "athena"
    make_minimal_bundle(bundle)

    result = _land(bundle, staging)

    for relation in result.relations:
        assert "_sema_locator" not in dict(relation.columns)
        assert "_sema_locator" not in staging.describe(SCHEMA, relation.name).columns


def test_missing_required_file_leaves_no_table(tmp_path: Path, staging: Staging) -> None:
    bundle = tmp_path / "athena"
    make_minimal_bundle(bundle)
    (bundle / "CONCEPT.csv").unlink()

    with pytest.raises(FileNotFoundError, match=r"CONCEPT\.csv"):
        _land(bundle, staging)

    assert _tables(staging) == []


def test_identity_carries_vocabulary_versions(tmp_path: Path, staging: Staging) -> None:
    bundle = tmp_path / "athena"
    make_minimal_bundle(bundle)

    result = _land(bundle, staging)

    assert ("HL7", "2024-01") in result.source_identity
    assert ("bundle", str(bundle.resolve())) in result.source_identity
    assert result.medium == "athena"


def test_relations_have_relation_level_provenance(tmp_path: Path, staging: Staging) -> None:
    bundle = tmp_path / "athena"
    make_minimal_bundle(bundle)

    result = _land(bundle, staging)

    locators = {r.name: r.provenance.locator for r in result.relations}
    assert locators == {
        "concept": "CONCEPT.csv",
        "concept_ancestor": "CONCEPT_ANCESTOR.csv",
        "concept_relationship": "CONCEPT_RELATIONSHIP.csv",
        "domain": "DOMAIN.csv",
        "vocabulary": "VOCABULARY.csv",
    }
    for relation in result.relations:
        assert relation.provenance.locator_kind == "relation"
        assert relation.provenance.loader == "athena"
        assert relation.schema == SCHEMA
    assert result.relation("concept").row_count == 2
