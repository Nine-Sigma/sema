"""Load public CSV relations into a DuckDB file exactly as a schema declares them, or say why not.

``load`` downloads each relation the profile names (or reads its local file), records the
publisher's row count and the file's sha256 in a manifest beside the database, and lands it
in the default schema so an unqualified name resolves. ``verify`` checks names, types and
counts against the schema and the manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import build_create_table_sql, qualified
from sema.log import logger

from showcase.open_data_load import cast_utils, fetch_utils
from showcase.open_data_load.fetch_utils import FileRecord
from showcase.open_data_load.spec_utils import (
    LoadError,
    Profile,
    Relation,
    map_headers,
    read_profile,
)

SCHEMA = "main"


def row_count(staging: Staging, name: str) -> int:
    count: int = staging.execute(f"SELECT count(*) FROM {qualified(SCHEMA, name)}").fetchone()[0]
    return count


def mismatch(staging: Staging, relation: Relation, expected: int) -> str | None:
    found = staging.execute(
        "SELECT column_name, data_type FROM duckdb_columns() "
        "WHERE schema_name = ? AND table_name = ? ORDER BY column_index",
        [SCHEMA, relation.name],
    ).fetchall()
    if not found:
        return "no table"
    declared = [(column.name, column.type) for column in relation.columns]
    if found != declared:
        return f"columns {found} differ from schema {declared}"
    count = row_count(staging, relation.name)
    return None if count == expected else f"{count} rows, expected {expected}"


def _fetch(
    profile: Profile, relation: Relation, known: FileRecord | None, paths: argparse.Namespace
) -> FileRecord:
    if relation.file is not None:
        local = paths.source_root / relation.file
        return fetch_utils.record(str(local), local, fetch_utils.csv_data_rows(local))
    target = paths.downloads / f"{relation.name}.csv"
    if known and target.exists() and fetch_utils.sha256(target) == known.sha256:
        logger.info(f"{relation.name}: reusing {target}")
        return known
    url = profile.download_url.format(dataset=relation.dataset)
    count = fetch_utils.publisher_count(profile.count_url.format(dataset=relation.dataset))
    fetch_utils.download(url, target)
    return fetch_utils.record(url, target, count)


def _land(staging: Staging, profile: Profile, relation: Relation, file: FileRecord) -> None:
    path = Path(file.path)
    headers = fetch_utils.csv_headers(path)
    targets = map_headers(relation, headers)
    source = cast_utils.read_csv_sql(path, len(targets))
    cast_utils.check_casts(staging, relation, headers, targets, source, profile.date_format)
    columns = {column.name: column.type for column in relation.columns}
    table = qualified(SCHEMA, relation.name)
    staging.drop_table(SCHEMA, relation.name)
    staging.execute(build_create_table_sql(SCHEMA, relation.name, columns))
    staging.execute(cast_utils.insert_sql(relation, table, targets, source, profile.date_format))
    loaded = row_count(staging, relation.name)
    if loaded != file.publisher_count:
        raise LoadError(
            f"{relation.name}: loaded {loaded} rows, publisher count {file.publisher_count}"
        )
    logger.info(f"{relation.name}: loaded {loaded:,} rows")


def run_load(profile: Profile, paths: argparse.Namespace) -> int:
    manifest_path = paths.db.with_suffix(".manifest.json")
    manifest = fetch_utils.read_manifest(manifest_path)
    failed = 0
    with Staging(str(paths.db), schemas=()) as staging:
        for relation in profile.relations:
            known = manifest.get(relation.name)
            if known and mismatch(staging, relation, known.publisher_count) is None:
                logger.info(f"{relation.name}: already loaded and verified, skipped")
                continue
            try:
                manifest[relation.name] = _fetch(profile, relation, known, paths)
                fetch_utils.write_manifest(manifest_path, manifest)
                _land(staging, profile, relation, manifest[relation.name])
            except (LoadError, OSError, duckdb.Error) as error:
                print(error, file=sys.stderr)
                failed += 1
    return 1 if failed else 0


def run_verify(profile: Profile, paths: argparse.Namespace) -> int:
    manifest = fetch_utils.read_manifest(paths.db.with_suffix(".manifest.json"))
    matched = 0
    with Staging(str(paths.db), schemas=()) as staging:
        for relation in profile.relations:
            known = manifest.get(relation.name)
            problem = "no manifest entry" if known is None else mismatch(
                staging, relation, known.publisher_count
            )  # fmt: skip
            if problem is None:
                matched += 1
                print(f"OK {relation.name} rows={row_count(staging, relation.name)}")
            else:
                print(f"FAIL {relation.name}: {problem}")
    print(f"{matched}/{len(profile.relations)} relations match")
    return 0 if matched == len(profile.relations) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("load", "verify"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--db", type=lambda raw: Path(raw).expanduser(), required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."),
                        help="resolves a profile's relative file paths")  # fmt: skip
    paths = parser.parse_args(argv)
    paths.downloads = paths.db.parent / "downloads" / paths.db.stem
    try:
        profile = read_profile(paths.profile, paths.schema)
    except LoadError as error:
        print(error, file=sys.stderr)
        return 1
    run = run_load if paths.command == "load" else run_verify
    return run(profile, paths)


if __name__ == "__main__":
    sys.exit(main())
