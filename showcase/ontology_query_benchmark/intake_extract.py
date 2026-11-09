"""Project the Standard Library rows the declared workload needs into a DuckDB file.

Rows are chosen by the data element ids in the CPD v2 product specs, so the projection
is supplied assistance, not automatic discovery. A read that exceeds its row limit,
returns other than the counted rows, or sees the source refresh is incomplete and
writes no database file.

    uv run python -m showcase.ontology_query_benchmark.intake_extract \\
      --spec DEMOGRAPHICS_SPEC --spec CANCER_REGISTRY_SPEC --out DIR
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import duckdb

from sema.ingest.duckdb_staging_utils import build_table_comment_sql, qualified, quote_ident
from showcase.ontology_query_benchmark.intake_extract_utils import (
    CONCEPT,
    DATA_ELEMENT,
    RELATIONS,
    SCHEMA,
    columns_sql,
    count_sql,
    declared_de_ids,
    duckdb_type,
    plain,
    predicates,
    select_sql,
    sha256,
    tables_sql,
)

DUCKDB_NAME = "workload_projection.duckdb"
RECEIPT_NAME = "extract_receipt.json"
DEFAULT_ROW_LIMIT = 10_000
HOST = "https://partner-workspace.cloud.databricks.com"
WAREHOUSE_ID = "feedface00000000"
ASSISTANCE = {
    "kind": "supplied",
    "reason": "rows are selected by the declared workload's data element ids from the "
    "CPD v2 product specs; this projection is not automatic discovery",
}


class Cursor(Protocol):
    def execute(self, sql: str, parameters: Any = ...) -> Any: ...

    def fetchall(self) -> list[Any]: ...

    def close(self) -> None: ...


@dataclass
class _Reader:
    cursor: Cursor
    statements: list[dict[str, Any]] = field(default_factory=list)

    def rows(self, sql: str) -> list[tuple[Any, ...]]:
        self.cursor.execute(sql)
        rows = [tuple(plain(value) for value in row) for row in self.cursor.fetchall()]
        self.statements.append({"sql": sql, "returned_rows": len(rows)})
        return rows


@dataclass(frozen=True)
class _Relation:
    name: str
    columns: tuple[tuple[str, str, str], ...]
    updated_at: str
    comment: str | None
    expected: int
    rows: tuple[tuple[Any, ...], ...] | None

    def ids(self, column: str) -> set[str]:
        index = [name for name, _, _ in self.columns].index(column)
        return {str(row[index]) for row in self.rows or () if row[index] is not None}


def extract(
    connect: Callable[[], Cursor],
    spec_paths: Sequence[Path],
    out_dir: Path,
    *,
    row_limit: int = DEFAULT_ROW_LIMIT,
    origin: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    de_ids = declared_de_ids(spec_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / DUCKDB_NAME
    target.unlink(missing_ok=True)
    reader = _Reader(connect())
    try:
        relations, incomplete = _read(reader, de_ids, row_limit)
    finally:
        reader.cursor.close()
    if not incomplete:
        _write(target, relations)
    receipt = {
        "assistance": ASSISTANCE,
        "origin": dict(origin or {}),
        "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "row_limit": row_limit,
        "complete": not incomplete,
        "incomplete": incomplete,
        "workload": _workload(spec_paths, de_ids, relations),
        "relations": {r.name: _relation_entry(r) for r in relations},
        "statements": reader.statements,
        "duckdb": None if incomplete else {"path": str(target), "sha256": sha256(target)},
    }
    (out_dir / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def _read(
    reader: _Reader, de_ids: Sequence[str], row_limit: int
) -> tuple[list[_Relation], list[str]]:
    columns = _columns(reader.rows(columns_sql()))
    before = {str(name): (when, comment) for name, when, comment in reader.rows(tables_sql())}
    wheres = predicates(de_ids)
    relations: list[_Relation] = []
    incomplete: list[str] = []
    for name in RELATIONS:
        expected = int(reader.rows(count_sql(name, wheres[name]))[0][0])
        rows = None
        if expected > row_limit:
            incomplete.append(f"{name}: {expected} counted rows exceed row_limit {row_limit}")
        else:
            names = [column for column, _, _ in columns[name]]
            rows = tuple(reader.rows(select_sql(name, names, wheres[name], row_limit)))
            if len(rows) != expected:
                incomplete.append(f"{name}: returned_rows {len(rows)} != counted {expected}")
        when, comment = before[name]
        relations.append(_Relation(name, columns[name], _iso(when), comment, expected, rows))
    after = {str(name): when for name, when, _ in reader.rows(tables_sql())}
    incomplete += [
        f"{name}: updated_at moved from {_iso(before[name][0])} to {_iso(after.get(name))}"
        for name in RELATIONS
        if after.get(name) != before[name][0]
    ]
    return relations, incomplete


def _columns(rows: list[tuple[Any, ...]]) -> dict[str, tuple[tuple[str, str, str], ...]]:
    found: dict[str, list[tuple[str, str, str]]] = {}
    for table, column, warehouse_type in rows:
        found.setdefault(str(table), []).append(
            (str(column), str(warehouse_type), duckdb_type(str(warehouse_type)))
        )
    missing = sorted(set(RELATIONS) - set(found))
    if missing:
        raise ValueError(f"source relations not in the catalog: {missing}")
    return {name: tuple(columns) for name, columns in found.items()}


def _iso(when: Any) -> str:
    return when.isoformat() if isinstance(when, datetime) else str(when)


def _write(target: Path, relations: Sequence[_Relation]) -> None:
    partial = target.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    with duckdb.connect(str(partial)) as con:
        con.execute(f"CREATE SCHEMA {quote_ident(SCHEMA)}")
        for relation in relations:
            table = qualified(SCHEMA, relation.name)
            listed = ", ".join(f"{quote_ident(n)} {t}" for n, _, t in relation.columns)
            con.execute(f"CREATE TABLE {table} ({listed})")
            marks = ", ".join("?" for _ in relation.columns)
            con.executemany(f"INSERT INTO {table} VALUES ({marks})", list(relation.rows or ()))
            if relation.comment:
                con.execute(build_table_comment_sql(SCHEMA, relation.name, relation.comment))
    partial.replace(target)


def _workload(
    spec_paths: Sequence[Path], de_ids: Sequence[str], relations: Sequence[_Relation]
) -> dict[str, Any]:
    by_name = {relation.name: relation for relation in relations}
    as_element = set(de_ids) & by_name[DATA_ELEMENT].ids("data_element_prt_id")
    as_concept = (set(de_ids) & by_name[CONCEPT].ids("concept_prt_id")) - as_element
    return {
        "spec_files": [{"path": str(p), "sha256": sha256(p)} for p in spec_paths],
        "de_ids_requested": list(de_ids),
        "found_as_data_element": sorted(as_element),
        "found_as_concept_only": sorted(as_concept),
        "missing": sorted(set(de_ids) - as_element - as_concept),
    }


def _relation_entry(relation: _Relation) -> dict[str, Any]:
    return {
        "source": f"{SCHEMA}.{relation.name}",
        "updated_at": relation.updated_at,
        "comment": relation.comment,
        "columns": [list(column) for column in relation.columns],
        "expected_rows": relation.expected,
        "returned_rows": None if relation.rows is None else len(relation.rows),
    }


def _connect_test() -> Cursor:
    from databricks import sql
    from databricks.sdk import WorkspaceClient

    config = WorkspaceClient(profile="TEST").config
    if (config.host or "").rstrip("/") != HOST or not config.token:
        raise SystemExit(f"TEST profile must be a token profile on {HOST}; got {config.host}")
    connection = sql.connect(
        server_hostname=HOST.removeprefix("https://"),
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=config.token,
    )
    cursor: Cursor = connection.cursor()
    return cursor


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    args = parser.parse_args(argv)
    receipt = extract(
        _connect_test, args.spec, args.out, row_limit=args.row_limit,
        origin={"host": HOST, "warehouse_id": WAREHOUSE_ID},
    )
    summary = {
        "complete": receipt["complete"],
        "incomplete": receipt["incomplete"],
        "rows": {name: e["returned_rows"] for name, e in receipt["relations"].items()},
        "duckdb": receipt["duckdb"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
