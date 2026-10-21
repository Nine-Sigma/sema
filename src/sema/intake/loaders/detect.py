"""Pick the existing loader for a source by its medium, from the smallest evidence.

A SQLite header means a semantic-sql file, a directory holding every required Athena
file an Athena bundle, a ``.sql`` file a DDL dictionary, and any file the tabular loader
reads a tabular source. A warehouse is never a path, so it is not detected here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sema.ingest.duckdb_staging import Staging
from sema.intake.landing import LandResult, Medium
from sema.intake.loaders.athena import REQUIRED_VOCAB_FILES, land_athena_bundle
from sema.intake.loaders.dictionary import land_ddl
from sema.intake.loaders.semsql import land_semsql
from sema.intake.loaders.tabular import land_tabular, supported_files

__all__ = ["NoLoaderError", "detect_medium", "land_source"]

SQLITE_HEADER = b"SQLite format 3\x00"
DDL_SUFFIX = ".sql"

Loader = Callable[..., LandResult]
LOADERS: dict[Medium, Loader] = {
    "semsql": land_semsql,
    "athena": land_athena_bundle,
    "dictionary": land_ddl,
    "tabular": land_tabular,
}


class NoLoaderError(ValueError):
    """No loader reads the source; nothing is landed."""

    def __init__(self, source: Path) -> None:
        self.source = source
        super().__init__(f"no loader reads {source.name} ({source})")


def detect_medium(source: Path) -> Medium | None:
    if source.is_file() and _is_sqlite(source):
        return "semsql"
    if source.is_file() and source.suffix.lower() == DDL_SUFFIX:
        return "dictionary"
    if source.is_dir() and all((source / name).is_file() for name in REQUIRED_VOCAB_FILES):
        return "athena"
    return "tabular" if supported_files(source) else None


def land_source(source: Path, staging: Staging, *, schema: str) -> LandResult:
    medium = detect_medium(source)
    if medium is None:
        raise NoLoaderError(source)
    return LOADERS[medium](source, staging, schema=schema)


def _is_sqlite(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(SQLITE_HEADER)) == SQLITE_HEADER
