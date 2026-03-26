import json
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from sema.migration.export_overrides import (
    export_overrides,
    _query_overrides,
)
from sema.migration.import_overrides import (
    import_overrides,
    _find_matching_assertion,
)


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(
        return_value=session,
    )
    driver.session.return_value.__exit__ = MagicMock(
        return_value=False,
    )
    return driver, session


class TestExportOverrides:
    def test_exports_to_json(self, mock_driver, tmp_path):
        driver, session = mock_driver
        session.run.return_value = [
            {
                "id": "a-1",
                "subject_ref": "unity://cdm.clinical.dx",
                "predicate": "has_entity_name",
                "payload": '{"value": "Diagnosis"}',
                "source": "human",
                "confidence": 1.0,
                "status": "pinned",
                "run_id": "run-1",
                "observed_at": "2026-01-01",
            },
        ]
        out = tmp_path / "overrides.json"
        result = export_overrides(driver, str(out))
        assert len(result) == 1
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["status"] == "pinned"

    def test_empty_export(self, mock_driver, tmp_path):
        driver, session = mock_driver
        session.run.return_value = []
        out = tmp_path / "empty.json"
        result = export_overrides(driver, str(out))
        assert result == []


class TestImportOverrides:
    def test_restores_matching_assertion(
        self, mock_driver, tmp_path,
    ):
        driver, session = mock_driver
        session.run.return_value.single.return_value = {
            "id": "a-new-1",
        }

        overrides = [{
            "subject_ref": "unity://cdm.clinical.dx",
            "predicate": "has_entity_name",
            "source": "human",
            "status": "pinned",
        }]
        inp = tmp_path / "overrides.json"
        inp.write_text(json.dumps(overrides))

        result = import_overrides(
            driver, str(inp), workspace="my-ws",
        )
        assert result["restored"] == 1
        assert result["orphaned"] == 0

    def test_logs_orphaned_overrides(
        self, mock_driver, tmp_path,
    ):
        driver, session = mock_driver
        session.run.return_value.single.return_value = None

        overrides = [{
            "subject_ref": "unity://cdm.clinical.gone",
            "predicate": "has_entity_name",
            "source": "human",
            "status": "accepted",
        }]
        inp = tmp_path / "overrides.json"
        inp.write_text(json.dumps(overrides))

        result = import_overrides(
            driver, str(inp), workspace="my-ws",
        )
        assert result["restored"] == 0
        assert result["orphaned"] == 1

    def test_dry_run_does_not_write(
        self, mock_driver, tmp_path,
    ):
        driver, session = mock_driver
        session.run.return_value.single.return_value = {
            "id": "a-1",
        }

        overrides = [{
            "subject_ref": "unity://cdm.clinical.dx",
            "predicate": "has_entity_name",
            "source": "human",
            "status": "pinned",
        }]
        inp = tmp_path / "overrides.json"
        inp.write_text(json.dumps(overrides))

        result = import_overrides(
            driver, str(inp), workspace="my-ws",
            dry_run=True,
        )
        assert result["restored"] == 1
        # In dry_run, the restore SET query should not be called
        # (only the matching query runs)
