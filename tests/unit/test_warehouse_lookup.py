"""Tests for `WarehouseSampler` and `WarehouseProfileLookup`."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.engine.warehouse_lookup import (
    WarehouseProfileLookup,
    WarehouseSampler,
)

pytestmark = pytest.mark.unit

KEY = ("cbioportal_msk_chord_2024", "sample", "patient_id")


class TestWarehouseSampler:
    def test_returns_set_of_distinct_string_values(self):
        query_fn = MagicMock(
            return_value=[("p1",), ("p2",), ("p3",)],
        )
        sampler = WarehouseSampler(
            query_fn=query_fn, catalog="workspace",
        )
        assert sampler(KEY) == {"p1", "p2", "p3"}

    def test_issues_select_distinct_with_limit(self):
        query_fn = MagicMock(return_value=[])
        sampler = WarehouseSampler(
            query_fn=query_fn, catalog="workspace", sample_cap=500,
        )
        sampler(KEY)
        sql = query_fn.call_args[0][0]
        assert "SELECT DISTINCT" in sql
        assert "LIMIT 500" in sql
        assert "`workspace`.`cbioportal_msk_chord_2024`.`sample`" in sql
        assert "`patient_id`" in sql

    def test_caches_repeat_lookups(self):
        query_fn = MagicMock(return_value=[("p1",)])
        sampler = WarehouseSampler(
            query_fn=query_fn, catalog="workspace",
        )
        sampler(KEY)
        sampler(KEY)
        sampler(KEY)
        query_fn.assert_called_once()

    def test_returns_none_on_query_error(self):
        query_fn = MagicMock(side_effect=RuntimeError("warehouse down"))
        sampler = WarehouseSampler(
            query_fn=query_fn, catalog="workspace",
        )
        assert sampler(KEY) is None

    def test_caches_none_to_avoid_retrying_failed_queries(self):
        query_fn = MagicMock(side_effect=RuntimeError("boom"))
        sampler = WarehouseSampler(
            query_fn=query_fn, catalog="workspace",
        )
        sampler(KEY)
        sampler(KEY)
        query_fn.assert_called_once()

    def test_skips_null_values_in_sample(self):
        query_fn = MagicMock(return_value=[("p1",), (None,), ("p2",)])
        sampler = WarehouseSampler(
            query_fn=query_fn, catalog="workspace",
        )
        assert sampler(KEY) == {"p1", "p2"}


class TestWarehouseProfileLookup:
    def test_returns_distinct_and_row_count(self):
        # First call: row count, second call: distinct
        query_fn = MagicMock(
            side_effect=[[(1000,)], [(742,)]],
        )
        lookup = WarehouseProfileLookup(
            query_fn=query_fn, catalog="workspace",
        )
        result = lookup(KEY)
        assert result == (742, 1000)

    def test_caches_row_count_per_table(self):
        query_fn = MagicMock(
            side_effect=[[(1000,)], [(742,)], [(900,)]],
        )
        lookup = WarehouseProfileLookup(
            query_fn=query_fn, catalog="workspace",
        )
        lookup(("schema_x", "sample", "patient_id"))
        lookup(("schema_x", "sample", "sample_id"))
        # Third call should hit row-count cache → 3 queries total
        # (1 count + 2 distinct), not 4
        assert query_fn.call_count == 3

    def test_caches_full_result_per_column(self):
        query_fn = MagicMock(
            side_effect=[[(1000,)], [(742,)]],
        )
        lookup = WarehouseProfileLookup(
            query_fn=query_fn, catalog="workspace",
        )
        lookup(KEY)
        lookup(KEY)
        assert query_fn.call_count == 2

    def test_returns_none_on_row_count_error(self):
        query_fn = MagicMock(side_effect=RuntimeError("denied"))
        lookup = WarehouseProfileLookup(
            query_fn=query_fn, catalog="workspace",
        )
        assert lookup(KEY) is None

    def test_returns_none_on_distinct_error(self):
        query_fn = MagicMock(
            side_effect=[[(1000,)], RuntimeError("col missing")],
        )
        lookup = WarehouseProfileLookup(
            query_fn=query_fn, catalog="workspace",
        )
        assert lookup(KEY) is None

    def test_issues_count_and_exact_distinct_sql(self):
        """Use exact `COUNT(DISTINCT col)` rather than APPROX.

        APPROX_COUNT_DISTINCT has ~2-7% error on Databricks which
        breaks `verify_cardinality`'s `pk_distinct == pk_rows`
        uniqueness check. Exact distinct is single-pass aggregate
        (NOT an RI scan), so it stays within the design's
        "no unbounded RI scans" rule.
        """
        query_fn = MagicMock(
            side_effect=[[(1000,)], [(742,)]],
        )
        lookup = WarehouseProfileLookup(
            query_fn=query_fn, catalog="workspace",
        )
        lookup(KEY)
        first_sql = query_fn.call_args_list[0][0][0]
        second_sql = query_fn.call_args_list[1][0][0]
        assert "COUNT(*)" in first_sql
        assert "COUNT(DISTINCT" in second_sql
        assert "APPROX_COUNT_DISTINCT" not in second_sql
        assert "`patient_id`" in second_sql
