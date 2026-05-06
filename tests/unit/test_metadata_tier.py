"""Metadata tier classifier (Section 5b)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _evidence(
    *,
    n_cols: int = 4,
    cols_with_comment: int = 0,
    table_comment: str | None = None,
    cols_with_top_values: int = 0,
    sample_rows: list | None = None,
) -> dict:
    cols = []
    for i in range(n_cols):
        c = {"name": f"c{i}", "data_type": "STRING"}
        if i < cols_with_comment:
            c["column_comment"] = f"comment for c{i}"
        if i < cols_with_top_values:
            c["top_values"] = [{"value": "x", "count": 1}]
        cols.append(c)
    return {
        "table_name": "t",
        "table_ref": "unity://cat/sch/t",
        "columns": cols,
        "table_comment": table_comment or "",
        "sample_rows": sample_rows or [],
    }


class TestClassifyMetadataTier:
    def test_full_column_comments_with_table_comment_is_rich(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        ev = _evidence(
            n_cols=10, cols_with_comment=10,
            table_comment="Patient demographic table",
        )
        assert classify_metadata_tier(ev) == "rich"

    def test_zero_evidence_is_name_only(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        ev = _evidence(n_cols=14)
        assert classify_metadata_tier(ev) == "name_only"

    def test_partial_evidence_is_sparse(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        ev = _evidence(
            n_cols=14, cols_with_comment=0,
            cols_with_top_values=8,
        )
        assert classify_metadata_tier(ev) == "sparse"

    def test_some_comments_below_floor_is_sparse(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        # 3/14 = 0.21 — below default rich floor (0.60)
        ev = _evidence(n_cols=14, cols_with_comment=3)
        assert classify_metadata_tier(ev) == "sparse"

    def test_source_agnostic_same_evidence_same_tier(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        ev_omop = _evidence(n_cols=14)
        ev_omop["table_ref"] = "unity://omop_v5/clinical/person"
        ev_cbio = _evidence(n_cols=14)
        ev_cbio["table_ref"] = "unity://cbioportal/brca/patient"
        assert classify_metadata_tier(ev_omop) == "name_only"
        assert classify_metadata_tier(ev_cbio) == "name_only"

    def test_pure_no_io(self):
        """Classifier must be pure — repeated calls produce same output."""
        from sema.engine.metadata_tier import classify_metadata_tier
        ev = _evidence(n_cols=10, cols_with_comment=10, table_comment="x")
        assert classify_metadata_tier(ev) == classify_metadata_tier(ev)

    def test_rich_floor_knob_shifts_threshold(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        # 3/10 = 0.30 column comments
        ev = _evidence(n_cols=10, cols_with_comment=3, table_comment="x")
        # Default 0.60 floor: not enough → sparse
        assert classify_metadata_tier(ev) == "sparse"
        # Lowered to 0.20 floor: passes → rich
        assert classify_metadata_tier(ev, rich_floor=0.20) == "rich"


class TestRichTierAccessoryConditions:
    def test_rich_requires_one_of_table_comment_top_values_or_samples(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        # 100% column comments but no other evidence → still requires
        # table_comment / top_values / sample_rows for rich
        ev = _evidence(n_cols=10, cols_with_comment=10)
        # No table_comment, no top_values, no sample_rows
        assert classify_metadata_tier(ev) == "sparse"

    def test_rich_with_top_values_majority(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        ev = _evidence(
            n_cols=10, cols_with_comment=10,
            cols_with_top_values=6,  # 60% — meets the >=50% floor
        )
        assert classify_metadata_tier(ev) == "rich"

    def test_rich_with_sample_rows(self):
        from sema.engine.metadata_tier import classify_metadata_tier
        ev = _evidence(
            n_cols=10, cols_with_comment=10,
            sample_rows=[{"c0": "a"}],
        )
        assert classify_metadata_tier(ev) == "rich"
