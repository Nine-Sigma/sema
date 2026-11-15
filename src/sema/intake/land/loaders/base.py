"""Shared loader pieces: the loader call shape and the staging write helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import pyarrow as pa

from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import quote_ident
from sema.intake.land.landing import LandResult, LocatorKind, Provenance, utc_now

DEFAULT_POLICY_CLASS = "metadata_only"
_DIGEST_CHUNK = 1 << 20


class Loader(Protocol):
    def __call__(self, source: Path, staging: Staging, *, schema: str) -> LandResult: ...


def relation_name(text: str) -> str:
    """A landed relation name the importer's unquoted SQL can name: word chars only."""
    return re.sub(r"\W", "_", text)


def ensure_schema(staging: Staging, schema: str) -> None:
    """``Staging.write_table`` does not create schemas; loaders do."""
    staging.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}")


def provenance(
    source: Path | str,
    loader: str,
    loader_version: str,
    locator_kind: LocatorKind,
    *,
    locator: str | None = None,
    landed_at: str | None = None,
    policy_class: str = DEFAULT_POLICY_CLASS,
) -> Provenance:
    return Provenance(
        source_uri=Path(source).resolve().as_uri(),
        loader=loader,
        loader_version=loader_version,
        landed_at=landed_at or utc_now(),
        policy_class=policy_class,
        locator_kind=locator_kind,
        locator=locator,
    )


def write_arrow(
    staging: Staging, schema: str, name: str, table: pa.Table, types: dict[str, str]
) -> None:
    staging.write_table(schema, name, table, types, {}, None)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DIGEST_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
