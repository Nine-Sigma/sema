"""Project the library rows a declared workload needs into a DuckDB file.

Rows are chosen by the element ids the supplied product specs declare, so the projection
is supplied assistance, not automatic discovery. A read that exceeds its row limit,
returns other than the counted rows, or sees the source refresh is incomplete and
writes no database file.

    uv run python -m showcase.ontology_query_benchmark.intake_extract \\
      --profile PROFILE --spec PRODUCT_SPEC [--spec PRODUCT_SPEC ...] --out DIR
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
from showcase.ontology_query_benchmark.intake_extract_hierarchy import SCHEMA, read_hierarchy
from showcase.ontology_query_benchmark.intake_extract_utils import (
    columns_sql,
    count_sql,
    declared_de_ids,
    duckdb_type,
    duplicate_keys,
    order_keys,
    plain,
    predicates,
    select_sql,
    sha256,
    tables_sql,
)
from showcase.ontology_query_benchmark.profile import Attribution, Extract, Routing, load_profile
from showcase.ontology_query_benchmark.routing import require_host

DUCKDB_NAME = "workload_projection.duckdb"
RECEIPT_NAME = "extract_receipt.json"
ASSISTANCE = {
    "kind": Attribution.SUPPLIED.value,
    "reason": "rows are selected by the element ids the declared workload's product specs "
    "name; this projection is not automatic discovery",
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
    schema: str | None = None
    source: str | None = None

    @property
    def key(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name

    def ids(self, column: str) -> set[str]:
        index = [name for name, _, _ in self.columns].index(column)
        return {str(row[index]) for row in self.rows or () if row[index] is not None}


def extract(
    connect: Callable[[], Cursor],
    spec: Extract,
    spec_paths: Sequence[Path],
    out_dir: Path,
    *,
    row_limit: int | None = None,
    origin: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    row_limit = spec.row_limit if row_limit is None else row_limit
    de_ids = declared_de_ids(spec_paths, spec.id_pattern)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / DUCKDB_NAME
    target.unlink(missing_ok=True)
    reader = _Reader(connect())
    try:
        relations, incomplete = _read(reader, spec, de_ids, row_limit)
    finally:
        reader.cursor.close()
    if not incomplete:
        _write(target, spec.schema_name, relations)
    receipt = {
        "assistance": ASSISTANCE,
        "origin": dict(origin or {}),
        "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "row_limit": row_limit,
        "complete": not incomplete,
        "incomplete": incomplete,
        "workload": _workload(spec, spec_paths, de_ids, relations),
        "relations": {r.key: _relation_entry(spec, r) for r in relations},
        "statements": reader.statements,
        "duckdb": None if incomplete else {"path": target.name, "sha256": sha256(target)},
    }
    (out_dir / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def _read(
    reader: _Reader, spec: Extract, de_ids: Sequence[str], row_limit: int
) -> tuple[list[_Relation], list[str]]:
    columns = _columns(reader.rows(columns_sql(spec)), spec.relations)
    before = {str(name): (when, comment) for name, when, comment in reader.rows(tables_sql(spec))}
    wheres = predicates(spec, de_ids)
    relations: list[_Relation] = []
    incomplete: list[str] = []
    for name in spec.relations:
        expected = int(reader.rows(count_sql(spec, name, wheres[name]))[0][0])
        rows = None
        if expected > row_limit:
            incomplete.append(f"{name}: {expected} counted rows exceed row_limit {row_limit}")
        else:
            names = [column for column, _, _ in columns[name]]
            rows = tuple(reader.rows(select_sql(spec, name, names, wheres[name], row_limit)))
            if len(rows) != expected:
                incomplete.append(f"{name}: returned_rows {len(rows)} != counted {expected}")
            keys = order_keys(spec, name)
            if duplicates := duplicate_keys(rows, names, keys):
                incomplete.append(f"{name}: {duplicates} duplicate order key rows over {keys}")
        when, comment = before[name]
        relations.append(_Relation(name, columns[name], _iso(when), comment, expected, rows))
    after = {str(name): when for name, when, _ in reader.rows(tables_sql(spec))}
    incomplete += [
        f"{name}: updated_at moved from {_iso(before[name][0])} to {_iso(after.get(name))}"
        for name in spec.relations
        if after.get(name) != before[name][0]
    ]
    if spec.concept_hierarchy is not None:
        members = relations[0].ids(spec.elements.member_key)
        slices, short = read_hierarchy(reader.rows, spec.concept_hierarchy, members, row_limit)
        relations += [_Relation(s.name, s.columns, s.updated_at, s.comment, s.expected, s.rows,
                                SCHEMA, s.source) for s in slices]  # fmt: skip
        incomplete += short
    return relations, incomplete


def _columns(
    rows: list[tuple[Any, ...]], relations: Sequence[str]
) -> dict[str, tuple[tuple[str, str, str], ...]]:
    found: dict[str, list[tuple[str, str, str]]] = {}
    for table, column, warehouse_type in rows:
        found.setdefault(str(table), []).append(
            (str(column), str(warehouse_type), duckdb_type(str(warehouse_type)))
        )
    missing = sorted(set(relations) - set(found))
    if missing:
        raise ValueError(f"source relations not in the catalog: {missing}")
    return {name: tuple(columns) for name, columns in found.items()}


def _iso(when: Any) -> str:
    return when.isoformat() if isinstance(when, datetime) else str(when)


def _write(target: Path, schema: str, relations: Sequence[_Relation]) -> None:
    partial = target.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    with duckdb.connect(str(partial)) as con:
        for relation in relations:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(relation.schema or schema)}")
            table = qualified(relation.schema or schema, relation.name)
            listed = ", ".join(f"{quote_ident(n)} {t}" for n, _, t in relation.columns)
            con.execute(f"CREATE TABLE {table} ({listed})")
            marks = ", ".join("?" for _ in relation.columns)
            con.executemany(f"INSERT INTO {table} VALUES ({marks})", list(relation.rows or ()))
            if relation.comment:
                con.execute(
                    build_table_comment_sql(relation.schema or schema, relation.name, relation.comment)
                )
    partial.replace(target)


def _workload(
    spec: Extract,
    spec_paths: Sequence[Path],
    de_ids: Sequence[str],
    relations: Sequence[_Relation],
) -> dict[str, Any]:
    by_name = {relation.name: relation for relation in relations}
    elements, members = spec.elements, spec.members
    as_element = set(de_ids) & by_name[elements.relation].ids(elements.key)
    as_concept = (set(de_ids) & by_name[members.relation].ids(members.key)) - as_element
    return {
        "spec_files": [{"path": str(p), "sha256": sha256(p)} for p in spec_paths],
        "de_ids_requested": list(de_ids),
        "found_as_data_element": sorted(as_element),
        "found_as_concept_only": sorted(as_concept),
        "missing": sorted(set(de_ids) - as_element - as_concept),
    }


def _relation_entry(spec: Extract, relation: _Relation) -> dict[str, Any]:
    return {
        "source": relation.source or f"{spec.catalog}.{spec.schema_name}.{relation.name}",
        "updated_at": relation.updated_at,
        "comment": relation.comment,
        "columns": [list(column) for column in relation.columns],
        "expected_rows": relation.expected,
        "returned_rows": None if relation.rows is None else len(relation.rows),
    }


def _connect(routing: Routing) -> Cursor:
    from databricks import sql
    from databricks.sdk import WorkspaceClient

    config = WorkspaceClient(profile=routing.databricks_profile).config
    host = require_host(config.host or "", routing)
    if not config.token:
        raise SystemExit(f"profile {routing.databricks_profile} must be a token profile")
    connection = sql.connect(
        server_hostname=host.removeprefix("https://"),
        http_path=f"/sql/1.0/warehouses/{routing.sql_warehouse_id}",
        access_token=config.token,
    )
    cursor: Cursor = connection.cursor()
    return cursor


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, help="default: the profile's extract row_limit")
    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    routing = profile.routing
    receipt = extract(
        lambda: _connect(routing), profile.extract, args.spec, args.out,
        row_limit=args.row_limit,
        origin={"host": routing.host, "warehouse_id": routing.sql_warehouse_id},
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
