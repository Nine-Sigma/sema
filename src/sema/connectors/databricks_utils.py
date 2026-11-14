"""Helper functions for the Databricks connector.

Extracted from databricks.py to keep the module focused on the
DatabricksConnector class and its public interface.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from databricks.sql import exc as databricks_exc

from sema.log import logger
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)
from sema.models.extraction import ExtractedForeignKey
from sema.models.physical_key import CanonicalRef, PhysicalKey
from sema.models.schema_metadata import (
    ColumnMetadata,
    ObjectMetadata,
    SchemaMetadata,
    Unreadable,
)
from sema.pipeline.normalizer import AssertionNormalizer

if TYPE_CHECKING:
    from sema.connectors.databricks import (
        DatabricksConnector,
        TableWorkItem,
    )


def _make_table_ref(connector: DatabricksConnector, catalog: str, schema: str, table: str) -> str:
    """Build a canonical table ref using the connector's workspace."""
    workspace = connector._config.host.replace("https://", "").rstrip("/")
    return CanonicalRef.build("databricks", PhysicalKey(workspace, catalog, schema, table))


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

    _, platform, workspace = connector.get_datasource_ref()
    normalizer = AssertionNormalizer(workspace, connector._run_id, platform)
    for key in connector._get_fk_constraints(catalog, schema, table_name):
        assertions.extend(normalizer.normalize_foreign_key(key))

    tags = connector._get_tags(catalog, schema, table_name)
    for tag in tags:
        assertions.append(
            connector._make_assertion(
                CanonicalRef.with_column(fqn, tag["column_name"]),
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
        col_ref = CanonicalRef.with_column(fqn, col["name"])
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
    distinct_scanned = columns if connector._profiling.enable_distinct_counts else []

    for col in distinct_scanned:
        col_ref = CanonicalRef.with_column(fqn, col["name"])

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


METADATA_SOURCE = "`system`.`information_schema`"

Row = tuple[Any, ...]


def sql_string_literal(value: str) -> str:
    """Quote a value as a Databricks string literal.

    Databricks reads '' as two adjacent literals, not an escaped quote, so the
    quote is backslash-escaped, after backslashes themselves.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def schema_metadata_queries(catalog: str, schema: str) -> tuple[str, str, str]:
    """Return the objects, columns, and view-text statements for one schema."""
    where = (
        f"WHERE table_catalog = {sql_string_literal(catalog)} "
        f"AND table_schema = {sql_string_literal(schema)}"
    )
    objects = f"SELECT table_name, table_type FROM {METADATA_SOURCE}.`tables` {where}"
    columns = (
        f"SELECT table_name, column_name, full_data_type, comment "
        f"FROM {METADATA_SOURCE}.`columns` {where} "
        f"ORDER BY table_name, ordinal_position"
    )
    views = f"SELECT table_name, view_definition FROM {METADATA_SOURCE}.`views` {where}"
    return objects, columns, views


def build_schema_metadata(
    catalog: str,
    schema: str,
    object_rows: Sequence[Row],
    column_rows: Sequence[Row],
    view_rows: Sequence[Row],
) -> SchemaMetadata:
    columns_by_table: dict[str, list[ColumnMetadata]] = {}
    for table_name, column_name, data_type, comment in column_rows:
        columns_by_table.setdefault(table_name, []).append(
            ColumnMetadata(name=column_name, data_type=data_type, comment=comment or None)
        )
    view_text = {table_name: definition for table_name, definition in view_rows}
    objects = tuple(
        ObjectMetadata(
            catalog=catalog,
            schema_name=schema,
            name=table_name,
            object_type=table_type,
            columns=tuple(columns_by_table.get(table_name, ())),
            view_definition=view_text.get(table_name),
        )
        for table_name, table_type in object_rows
    )
    return SchemaMetadata(catalog=catalog, schema_name=schema, objects=objects)


def read_schema_metadata(
    execute: Callable[[str], list[Row]], catalog: str, schema: str,
) -> SchemaMetadata | Unreadable:
    """Run the three metadata statements; a warehouse error yields Unreadable."""
    try:
        object_rows, column_rows, view_rows = [
            execute(query) for query in schema_metadata_queries(catalog, schema)
        ]
    except databricks_exc.Error as exc:
        first_line = next(iter(str(exc).splitlines()), "")
        reason = f"{type(exc).__name__}: {first_line}"
        logger.warning("Schema metadata unreadable for {}.{}: {}", catalog, schema, reason)
        return Unreadable(catalog=catalog, schema_name=schema, reason=reason)
    return build_schema_metadata(catalog, schema, object_rows, column_rows, view_rows)


def _same_constraint(left: str, right: str, prefix: str = "") -> str:
    return " AND ".join(
        f"{left}.constraint_{part} = {right}.{prefix}constraint_{part}"
        for part in ("catalog", "schema", "name")
    )


def declared_keys_query(catalog: str, schema: str, table: str) -> str:
    """Each foreign key column of one table beside the referenced column it pairs with."""
    return (
        "SELECT kcu.constraint_name, kcu.column_name, "
        "ref.table_catalog, ref.table_schema, ref.table_name, ref.column_name "
        f"FROM {METADATA_SOURCE}.`table_constraints` tc "
        f"JOIN {METADATA_SOURCE}.`referential_constraints` rc "
        f"ON {_same_constraint('rc', 'tc')} "
        f"JOIN {METADATA_SOURCE}.`key_column_usage` kcu "
        f"ON {_same_constraint('kcu', 'tc')} "
        f"JOIN {METADATA_SOURCE}.`key_column_usage` ref "
        f"ON {_same_constraint('ref', 'rc', 'unique_')} "
        "AND ref.ordinal_position = kcu.position_in_unique_constraint "
        "WHERE tc.constraint_type = 'FOREIGN KEY' "
        f"AND tc.table_catalog = {sql_string_literal(catalog)} "
        f"AND tc.table_schema = {sql_string_literal(schema)} "
        f"AND tc.table_name = {sql_string_literal(table)} "
        "ORDER BY kcu.constraint_name, kcu.ordinal_position"
    )


def declared_keys_from_rows(
    catalog: str, schema: str, table: str, rows: Sequence[Row],
) -> list[ExtractedForeignKey]:
    """Group ordinal-ordered key rows into one foreign key per constraint."""
    pairs: dict[tuple[str, str, str, str], list[tuple[str, str]]] = {}
    for name, column, ref_catalog, ref_schema, ref_table, ref_column in rows:
        pairs.setdefault((name, ref_catalog, ref_schema, ref_table), []).append(
            (column, ref_column),
        )
    return [
        ExtractedForeignKey(
            constraint_name=name,
            referencing_catalog=catalog, referencing_schema=schema, referencing_table=table,
            referenced_catalog=ref_catalog, referenced_schema=ref_schema,
            referenced_table=ref_table, column_pairs=tuple(columns),
        )
        for (name, ref_catalog, ref_schema, ref_table), columns in pairs.items()
    ]
