from __future__ import annotations

import fnmatch
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from databricks import sql as databricks_sql

from sema.connectors.base import Connector
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)
from sema.models.config import DatabricksConfig, ProfilingConfig


@dataclass(frozen=True)
class TableWorkItem:
    """Lightweight reference to a table for pipeline processing."""
    catalog: str
    schema: str
    table_name: str
    fqn: str

logger = logging.getLogger(__name__)

# Alias for mocking in tests
sql_connect = databricks_sql.connect


class DatabricksConnector(Connector):
    def __init__(self, config: DatabricksConfig, profiling: ProfilingConfig | None = None) -> None:
        self._config = config
        self._profiling = profiling or ProfilingConfig()
        self._run_id = str(uuid.uuid4())
        try:
            self._connection = sql_connect(
                server_hostname=config.host.replace("https://", ""),
                http_path=config.http_path,
                access_token=config.token.get_secret_value(),
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Databricks: {e}") from e

    def _execute(self, query: str) -> list[tuple[Any, ...]]:
        with self._connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()  # type: ignore[return-value]

    def _execute_with_description(self, query: str) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        with self._connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall(), cursor.description or []  # type: ignore[return-value]

    def get_datasource_ref(self) -> tuple[str, str, str]:
        """Return (ref, platform, workspace) for this Databricks instance."""
        workspace = self._config.host.replace("https://", "").rstrip("/")
        ref = f"databricks://{workspace}"
        return ref, "databricks", workspace

    def list_catalogs(self) -> list[str]:
        rows = self._execute("SHOW CATALOGS")
        return [row[0] for row in rows]

    def _discover_schemas(self, catalog: str) -> list[str]:
        rows = self._execute(f"SHOW SCHEMAS IN `{catalog}`")
        return [row[0] for row in rows]

    def _discover_tables(self, catalog: str, schema: str) -> list[tuple[str, str]]:
        rows = self._execute(f"SHOW TABLES IN `{catalog}`.`{schema}`")
        # SHOW TABLES returns (database, tableName, isTemporary)
        return [(row[1], "TABLE") for row in rows]

    def _get_columns(self, catalog: str, schema: str, table: str) -> list[dict[str, Any]]:
        query = (
            f"SELECT column_name, data_type, is_nullable, comment "
            f"FROM `system`.`information_schema`.`columns` "
            f"WHERE table_catalog = '{catalog}' "
            f"AND table_schema = '{schema}' "
            f"AND table_name = '{table}' "
            f"ORDER BY ordinal_position"
        )
        rows = self._execute(query)
        return [
            {
                "name": row[0],
                "data_type": row[1],
                "nullable": row[2] == "YES",
                "comment": row[3],
            }
            for row in rows
        ]

    def _get_fk_constraints(self, catalog: str, schema: str, table: str) -> list[dict[str, str]]:
        query = (
            f"SELECT tc.table_name, kcu.column_name, "
            f"kcu.referenced_table_name, kcu.referenced_column_name "
            f"FROM `system`.`information_schema`.`table_constraints` tc "
            f"JOIN `system`.`information_schema`.`key_column_usage` kcu "
            f"ON tc.constraint_name = kcu.constraint_name "
            f"WHERE tc.constraint_type = 'FOREIGN KEY' "
            f"AND tc.table_catalog = '{catalog}' "
            f"AND tc.table_schema = '{schema}' "
            f"AND tc.table_name = '{table}'"
        )
        try:
            rows = self._execute(query)
        except Exception:
            return []
        return [
            {
                "from_table": row[0],
                "from_col": row[1],
                "to_table": row[2],
                "to_col": row[3],
            }
            for row in rows
        ]

    def _get_tags(self, catalog: str, schema: str, table: str) -> list[dict[str, str]]:
        query = (
            f"SELECT column_name, tag_name, tag_value "
            f"FROM `system`.`information_schema`.`column_tags` "
            f"WHERE table_catalog = '{catalog}' "
            f"AND table_schema = '{schema}' "
            f"AND table_name = '{table}'"
        )
        try:
            rows = self._execute(query)
        except Exception:
            return []
        return [
            {"column_name": row[0], "tag_key": row[1], "tag_value": row[2]}
            for row in rows
        ]

    def _approx_distinct(self, catalog: str, schema: str, table: str, column: str) -> int:
        query = f"SELECT APPROX_COUNT_DISTINCT(`{column}`) FROM `{catalog}`.`{schema}`.`{table}`"
        rows = self._execute(query)
        return rows[0][0] if rows else 0

    def _top_k_values(
        self, catalog: str, schema: str, table: str, column: str, k: int
    ) -> list[dict[str, Any]]:
        query = (
            f"SELECT `{column}`, COUNT(*) AS freq "
            f"FROM `{catalog}`.`{schema}`.`{table}` "
            f"GROUP BY `{column}` ORDER BY freq DESC LIMIT {k}"
        )
        rows = self._execute(query)
        return [{"value": str(row[0]), "frequency": row[1]} for row in rows]

    def _sample_rows(self, catalog: str, schema: str, table: str, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
        query = f"SELECT * FROM `{catalog}`.`{schema}`.`{table}` LIMIT {limit}"
        rows, description = self._execute_with_description(query)
        col_names = [d[0] for d in description] if description else []
        return [dict(zip(col_names, [str(v) for v in row])) for row in rows], col_names

    def _make_assertion(
        self,
        subject_ref: str,
        predicate: AssertionPredicate,
        payload: dict[str, Any],
        object_ref: str | None = None,
        confidence: float = 1.0,
    ) -> Assertion:
        return Assertion(
            id=str(uuid.uuid4()),
            subject_ref=subject_ref,
            predicate=predicate,
            payload=payload,
            object_ref=object_ref,
            source="unity_catalog",
            confidence=confidence,
            run_id=self._run_id,
            observed_at=datetime.now(timezone.utc),
        )

    def discover_tables(
        self,
        catalog: str,
        schemas: list[str] | None = None,
        table_pattern: str | None = None,
    ) -> list[TableWorkItem]:
        """Discover tables without performing extraction or profiling.

        Returns lightweight TableWorkItem references.
        """
        work_items: list[TableWorkItem] = []

        if schemas is None:
            schemas = self._discover_schemas(catalog)

        for schema in schemas:
            tables = self._discover_tables(catalog, schema)
            for table_name, _table_type in tables:
                if table_pattern and not fnmatch.fnmatch(table_name, table_pattern):
                    continue
                workspace = self._config.host.replace(
                    "https://", ""
                ).rstrip("/")
                fqn = (
                    f"databricks://{workspace}"
                    f"/{catalog}/{schema}/{table_name}"
                )
                work_items.append(TableWorkItem(
                    catalog=catalog,
                    schema=schema,
                    table_name=table_name,
                    fqn=fqn,
                ))

        return work_items

    _TEMPORAL_TYPES = {"DATE", "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ"}
    _NUMERIC_TYPES = {
        "INT", "BIGINT", "SMALLINT", "TINYINT",
        "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC",
    }

    def _should_skip_profiling(self, data_type: str) -> bool:
        dt_upper = data_type.upper().split("(")[0].strip()
        if self._profiling.skip_temporal_profiling:
            if dt_upper in self._TEMPORAL_TYPES:
                return True
        if self._profiling.skip_numeric_profiling:
            if dt_upper in self._NUMERIC_TYPES:
                return True
        return False

    def extract_table(self, work_item: TableWorkItem) -> list[Assertion]:
        """Extract metadata and profiling for a single table."""
        from sema.connectors.databricks_utils import (
            _build_column_assertions,
            _build_profiling_assertions,
            _build_table_assertion,
        )

        catalog = work_item.catalog
        schema = work_item.schema
        table_name = work_item.table_name
        fqn = work_item.fqn

        columns = self._get_columns(catalog, schema, table_name)

        assertions: list[Assertion] = []
        assertions.extend(_build_table_assertion(self, work_item))
        assertions.extend(_build_column_assertions(self, work_item, columns))
        assertions.extend(_build_profiling_assertions(self, work_item, columns))
        return assertions

    def extract(
        self,
        catalog: str,
        schemas: list[str] | None = None,
        table_pattern: str | None = None,
        **kwargs: Any,
    ) -> list[Assertion]:
        work_items = self.discover_tables(catalog, schemas, table_pattern)
        assertions: list[Assertion] = []
        for work_item in work_items:
            try:
                assertions.extend(self.extract_table(work_item))
            except Exception as e:
                logger.warning(f"Failed to extract table {work_item.fqn}: {e}")
                continue
        return assertions
