"""Tests for FK detection wiring in `orchestrate_utils.run_fk_detection`."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sema.models.config import BuildConfig
from sema.models.extraction import ExtractedColumn
from sema.pipeline.orchestrate_utils import run_fk_detection

pytestmark = pytest.mark.unit

SCHEMA = "cbioportal_msk_chord_2024"


def _columns_with_fk_pair() -> list[ExtractedColumn]:
    return [
        ExtractedColumn(
            name="patient_id", table_name="patient",
            catalog="workspace", schema=SCHEMA, data_type="string",
        ),
        ExtractedColumn(
            name="patient_id", table_name="sample",
            catalog="workspace", schema=SCHEMA, data_type="string",
        ),
    ]


def _build_config(**overrides) -> BuildConfig:
    return BuildConfig(
        catalog="workspace", schemas=[SCHEMA], **overrides,
    )


def _connector() -> MagicMock:
    c = MagicMock()
    c._execute = MagicMock(return_value=[])
    return c


def test_skips_when_disabled():
    loader = MagicMock()
    config = _build_config(enable_fk_detection=False)
    with patch(
        "sema.pipeline.orchestrate_utils.fetch_columns_by_schema"
    ) as fetch:
        run_fk_detection(
            loader, _connector(), config, [SCHEMA], run_id="r-1",
        )
    fetch.assert_not_called()


def test_uses_default_threshold_080():
    loader = MagicMock()
    config = _build_config()
    captured: dict[str, float] = {}
    with patch(
        "sema.pipeline.orchestrate_utils.fetch_columns_by_schema",
        return_value=[],
    ), patch(
        "sema.pipeline.orchestrate_utils.JoinDetector"
    ) as DetectorCls:
        DetectorCls.side_effect = lambda **kw: (
            captured.update(kw) or MagicMock()
        )
        run_fk_detection(
            loader, _connector(), config, [SCHEMA], run_id="r-1",
        )
    assert captured["materialization_threshold"] == 0.80


def test_threshold_drops_to_070_with_structural_opt_in():
    loader = MagicMock()
    config = _build_config(materialize_structural_fk=True)
    captured: dict[str, float] = {}
    with patch(
        "sema.pipeline.orchestrate_utils.fetch_columns_by_schema",
        return_value=[],
    ), patch(
        "sema.pipeline.orchestrate_utils.JoinDetector"
    ) as DetectorCls:
        DetectorCls.side_effect = lambda **kw: (
            captured.update(kw) or MagicMock()
        )
        run_fk_detection(
            loader, _connector(), config, [SCHEMA], run_id="r-1",
        )
    assert captured["materialization_threshold"] == 0.70


def test_filters_by_should_materialize_then_calls_materializer():
    loader = MagicMock()
    config = _build_config()
    columns = _columns_with_fk_pair()

    above = MagicMock(name="above_threshold")
    below = MagicMock(name="below_threshold")
    detector_inst = MagicMock()
    detector_inst.detect.return_value = [above, below]
    detector_inst.should_materialize.side_effect = (
        lambda fk: fk is above
    )

    with patch(
        "sema.pipeline.orchestrate_utils.fetch_columns_by_schema",
        return_value=columns,
    ), patch(
        "sema.pipeline.orchestrate_utils.JoinDetector",
        return_value=detector_inst,
    ), patch(
        "sema.pipeline.orchestrate_utils.to_fk_assertion"
    ) as to_fk, patch(
        "sema.pipeline.orchestrate_utils.materialize_join_paths"
    ) as materialize:
        to_fk.return_value = MagicMock(
            subject_ref="ref-1",
            predicate=MagicMock(value="fk_to"),
        )
        run_fk_detection(
            loader, _connector(), config, [SCHEMA], run_id="r-1",
        )

    to_fk.assert_called_once_with(above, "r-1")
    materialize.assert_called_once()
    _, kwargs = materialize.call_args
    assert kwargs["source_schema"] == SCHEMA


def test_skips_schemas_with_no_columns():
    loader = MagicMock()
    config = _build_config()
    with patch(
        "sema.pipeline.orchestrate_utils.fetch_columns_by_schema",
        return_value=[],
    ), patch(
        "sema.pipeline.orchestrate_utils.materialize_join_paths"
    ) as materialize:
        run_fk_detection(
            loader, _connector(), config, [SCHEMA], run_id="r-1",
        )
    materialize.assert_not_called()


def test_passes_sampler_and_profiles_to_detector():
    loader = MagicMock()
    config = _build_config()
    columns = _columns_with_fk_pair()
    connector = _connector()

    detector_inst = MagicMock()
    detector_inst.detect.return_value = []

    with patch(
        "sema.pipeline.orchestrate_utils.fetch_columns_by_schema",
        return_value=columns,
    ), patch(
        "sema.pipeline.orchestrate_utils.JoinDetector",
        return_value=detector_inst,
    ):
        run_fk_detection(
            loader, connector, config, [SCHEMA], run_id="r-1",
        )

    detect_kwargs = detector_inst.detect.call_args.kwargs
    assert "sampler" in detect_kwargs
    assert detect_kwargs["sampler"] is not None
    assert "profiles" in detect_kwargs
    assert isinstance(detect_kwargs["profiles"], dict)


def test_prebuilds_profiles_only_for_candidate_columns():
    """Profile lookup must run only for columns mentioned in candidates."""
    loader = MagicMock()
    config = _build_config()
    columns = _columns_with_fk_pair()

    profile_calls: list[tuple] = []

    def fake_lookup(key):
        profile_calls.append(key)
        return (5, 10)

    detector_inst = MagicMock()
    detector_inst.detect.return_value = []

    with patch(
        "sema.pipeline.orchestrate_utils.fetch_columns_by_schema",
        return_value=columns,
    ), patch(
        "sema.pipeline.orchestrate_utils.JoinDetector",
        return_value=detector_inst,
    ), patch(
        "sema.pipeline.orchestrate_utils.WarehouseProfileLookup",
        return_value=fake_lookup,
    ):
        run_fk_detection(
            loader, _connector(), config, [SCHEMA], run_id="r-1",
        )

    keys = set(profile_calls)
    assert (SCHEMA, "patient", "patient_id") in keys
    assert (SCHEMA, "sample", "patient_id") in keys
