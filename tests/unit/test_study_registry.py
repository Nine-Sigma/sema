from __future__ import annotations

from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging
from sema.ingest.study_registry import (
    StudyCollisionError,
    StudyRegistry,
)


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    db_path = tmp_path / "test.duckdb"
    return Staging(str(db_path))


@pytest.mark.unit
class TestRegistryTable:
    def test_initialises_table_in_sema_schema(self, staging: Staging) -> None:
        StudyRegistry(staging)
        rows = staging.execute(
            "SELECT table_name FROM duckdb_tables() "
            "WHERE schema_name = '_sema' AND table_name = '_sema_study_registry'"
        ).fetchall()
        assert len(rows) == 1

    def test_init_is_idempotent(self, staging: Staging) -> None:
        StudyRegistry(staging)
        StudyRegistry(staging)
        rows = staging.execute(
            "SELECT count(*) FROM duckdb_tables() "
            "WHERE schema_name = '_sema' AND table_name = '_sema_study_registry'"
        ).fetchone()
        assert rows[0] == 1

    def test_register_persists_row(self, staging: Staging) -> None:
        registry = StudyRegistry(staging)
        registry.register(
            schema_name="cbioportal_msk_chord_2024",
            original_study_id="msk_chord_2024",
            source_type="cbioportal",
        )
        rows = staging.execute(
            "SELECT schema_name, original_study_id, source_type "
            "FROM _sema._sema_study_registry"
        ).fetchall()
        assert rows == [("cbioportal_msk_chord_2024", "msk_chord_2024", "cbioportal")]

    def test_register_same_study_twice_is_no_op(self, staging: Staging) -> None:
        registry = StudyRegistry(staging)
        registry.register("cbioportal_x", "X", "cbioportal")
        registry.register("cbioportal_x", "X", "cbioportal")
        rows = staging.execute(
            "SELECT count(*) FROM _sema._sema_study_registry"
        ).fetchone()
        assert rows[0] == 1

    def test_collision_raises(self, staging: Staging) -> None:
        registry = StudyRegistry(staging)
        registry.register("cbioportal_brca_tcga", "BRCA-TCGA", "cbioportal")
        with pytest.raises(StudyCollisionError) as exc_info:
            registry.register("cbioportal_brca_tcga", "BRCA_TCGA", "cbioportal")
        msg = str(exc_info.value)
        assert "BRCA-TCGA" in msg
        assert "BRCA_TCGA" in msg
        assert "cbioportal_brca_tcga" in msg

    def test_collision_does_not_overwrite_existing(self, staging: Staging) -> None:
        registry = StudyRegistry(staging)
        registry.register("cbioportal_brca_tcga", "BRCA-TCGA", "cbioportal")
        with pytest.raises(StudyCollisionError):
            registry.register("cbioportal_brca_tcga", "BRCA_TCGA", "cbioportal")
        rows = staging.execute(
            "SELECT original_study_id FROM _sema._sema_study_registry"
        ).fetchall()
        assert rows == [("BRCA-TCGA",)]

    def test_list_registered_schemas(self, staging: Staging) -> None:
        registry = StudyRegistry(staging)
        registry.register("cbioportal_a", "A", "cbioportal")
        registry.register("cbioportal_b", "B", "cbioportal")
        names = registry.list_schemas()
        assert sorted(names) == ["cbioportal_a", "cbioportal_b"]

    def test_list_empty_when_no_registrations(self, staging: Staging) -> None:
        registry = StudyRegistry(staging)
        assert registry.list_schemas() == []

    def test_register_records_created_at(self, staging: Staging) -> None:
        registry = StudyRegistry(staging)
        registry.register("cbioportal_x", "x", "cbioportal")
        row = staging.execute(
            "SELECT created_at FROM _sema._sema_study_registry"
        ).fetchone()
        assert row[0] is not None
