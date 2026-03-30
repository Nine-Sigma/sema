"""Helper functions for the Databricks connector.

Extracted from databricks.py to keep the module focused on the
DatabricksConnector class and its public interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)

if TYPE_CHECKING:
    from sema.connectors.databricks import (
        DatabricksConnector,
        TableWorkItem,
    )


def _make_table_ref(connector: DatabricksConnector, catalog: str, schema: str, table: str) -> str:
    """Build a canonical table ref using the connector's workspace."""
    workspace = connector._config.host.replace("https://", "").rstrip("/")
    return f"databricks://{workspace}/{catalog}/{schema}/{table}"


def _build_table_assertion(
    connector: DatabricksConnector, work_item: TableWorkItem
) -> list[Assertion]:
    assertions: list[Assertion] = []
    catalog = work_item.catalog
    schema = work_item.schema
    table_name = work_item.table_name
    fqn = work_item.fqn

    assertions.append(
        connector._make_assertion(fqn, AssertionPredicate.TABLE_EXISTS, {"table_type": "TABLE"})
    )

    fks = connector._get_fk_constraints(catalog, schema, table_name)
    for fk in fks:
        to_ref = _make_table_ref(connector, catalog, schema, fk["to_table"])
        assertions.append(
            connector._make_assertion(
                fqn,
                AssertionPredicate.JOINS_TO,
                {"on_column": fk["from_col"], "to_column": fk["to_col"]},
                object_ref=to_ref,
                confidence=0.95,
            )
        )

    tags = connector._get_tags(catalog, schema, table_name)
    for tag in tags:
        assertions.append(
            connector._make_assertion(
                f"{fqn}/{tag['column_name']}",
                AssertionPredicate.HAS_TAG,
                {"tag_key": tag["tag_key"], "tag_value": tag["tag_value"]},
            )
        )

    return assertions


def _build_column_assertions(
    connector: DatabricksConnector, work_item: TableWorkItem, columns: list[dict[str, Any]]
) -> list[Assertion]:
    assertions: list[Assertion] = []
    fqn = work_item.fqn

    for col in columns:
        col_ref = f"{fqn}/{col['name']}"
        assertions.append(
            connector._make_assertion(
                col_ref,
                AssertionPredicate.COLUMN_EXISTS,
                {
                    "data_type": col["data_type"],
                    "nullable": col["nullable"],
                    "comment": col["comment"],
                },
            )
        )
        assertions.append(
            connector._make_assertion(
                col_ref,
                AssertionPredicate.HAS_DATATYPE,
                {"value": col["data_type"]},
            )
        )
        if col["comment"]:
            assertions.append(
                connector._make_assertion(
                    col_ref,
                    AssertionPredicate.HAS_COMMENT,
                    {"value": col["comment"]},
                )
            )

    return assertions


def _build_profiling_assertions(
    connector: DatabricksConnector, work_item: TableWorkItem, columns: list[dict[str, Any]]
) -> list[Assertion]:
    assertions: list[Assertion] = []
    catalog = work_item.catalog
    schema = work_item.schema
    table_name = work_item.table_name
    fqn = work_item.fqn

    for col in columns:
        col_ref = f"{fqn}/{col['name']}"

        if connector._should_skip_profiling(col["data_type"]):
            continue

        try:
            approx_distinct = connector._approx_distinct(catalog, schema, table_name, col["name"])
        except Exception:
            continue

        if approx_distinct <= connector._profiling.categorical_threshold:
            try:
                top_values = connector._top_k_values(
                    catalog, schema, table_name, col["name"], connector._profiling.top_k_values
                )
                assertions.append(
                    connector._make_assertion(
                        col_ref,
                        AssertionPredicate.HAS_TOP_VALUES,
                        {"values": top_values, "approx_distinct": approx_distinct},
                    )
                )
            except Exception:
                pass

    try:
        sample_data, col_names = connector._sample_rows(
            catalog, schema, table_name, connector._profiling.sample_rows
        )
        assertions.append(
            connector._make_assertion(
                fqn,
                AssertionPredicate.HAS_SAMPLE_ROWS,
                {"rows": sample_data, "columns": col_names},
            )
        )
    except Exception:
        pass

    return assertions
