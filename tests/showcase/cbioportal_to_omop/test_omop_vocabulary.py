from __future__ import annotations

from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging
from showcase.cbioportal_to_omop.omop_ingest import ingest_vocabulary
from tests.support.athena_bundle import make_minimal_bundle as _make_minimal_bundle
from tests.support.athena_bundle import write_vocab_csv as _write_vocab_csv


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    return Staging(str(tmp_path / "vocab.duckdb"))


@pytest.mark.unit
class TestIngestVocabulary:
    def test_happy_path_loads_required_tables(self, tmp_path: Path, staging: Staging) -> None:
        bundle = tmp_path / "athena"
        _make_minimal_bundle(bundle)

        ingest_vocabulary(bundle, staging)

        concept = staging.describe("vocabulary_omop", "concept")
        assert "concept_id" in concept.columns

        rel = staging.describe("vocabulary_omop", "concept_relationship")
        assert "relationship_id" in rel.columns

        anc = staging.describe("vocabulary_omop", "concept_ancestor")
        assert "ancestor_concept_id" in anc.columns

    def test_loads_optional_files_when_present(self, tmp_path: Path, staging: Staging) -> None:
        bundle = tmp_path / "athena"
        _make_minimal_bundle(bundle)
        _write_vocab_csv(
            bundle / "CONCEPT_SYNONYM.csv",
            ["concept_id", "concept_synonym_name", "language_concept_id"],
            [["1", "Sex", "4180186"]],
        )

        ingest_vocabulary(bundle, staging)

        syn = staging.describe("vocabulary_omop", "concept_synonym")
        assert "concept_synonym_name" in syn.columns

    def test_missing_required_file_raises(self, tmp_path: Path, staging: Staging) -> None:
        bundle = tmp_path / "incomplete"
        _make_minimal_bundle(bundle)
        (bundle / "CONCEPT.csv").unlink()

        with pytest.raises(FileNotFoundError) as exc:
            ingest_vocabulary(bundle, staging)
        assert "CONCEPT.csv" in str(exc.value)

    def test_missing_bundle_directory_raises(self, tmp_path: Path, staging: Staging) -> None:
        with pytest.raises(FileNotFoundError):
            ingest_vocabulary(tmp_path / "does_not_exist", staging)

    def test_none_path_skips_cleanly(self, staging: Staging) -> None:
        ingest_vocabulary(None, staging)
        with pytest.raises(ValueError):
            staging.describe("vocabulary_omop", "concept")

    def test_partial_failure_does_not_leave_partial_tables(
        self, tmp_path: Path, staging: Staging
    ) -> None:
        bundle = tmp_path / "incomplete"
        bundle.mkdir()
        _write_vocab_csv(
            bundle / "CONCEPT.csv",
            ["concept_id", "concept_name"],
            [["1", "Only concept table"]],
        )

        with pytest.raises(FileNotFoundError):
            ingest_vocabulary(bundle, staging)

        with pytest.raises(ValueError):
            staging.describe("vocabulary_omop", "concept")
