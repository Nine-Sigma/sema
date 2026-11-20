"""What to load: relations from a schema file, and where each comes from from a profile."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class LoadError(Exception):
    """A relation cannot be loaded as declared; the message names the cause."""


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    header: str


@dataclass(frozen=True)
class Relation:
    name: str
    columns: tuple[Column, ...]
    dataset: str | None
    file: str | None
    ignore_headers: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    date_format: str
    page_url: str
    page_size: int
    count_url: str
    relations: tuple[Relation, ...]


def normalize_header(header: str) -> str:
    return re.sub(r"[\s_]+", " ", header).strip().casefold()


def _relation(table: dict[str, Any], source: dict[str, Any]) -> Relation:
    if (source.get("dataset") is None) == (source.get("file") is None):
        raise LoadError(f"{table['name']}: the profile needs exactly one of dataset or file")
    headers: dict[str, str] = source.get("headers") or {}
    unknown = set(headers) - {c["name"] for c in table["columns"]}
    if unknown:
        raise LoadError(f"{table['name']}: profile headers name no column: {sorted(unknown)}")
    columns = tuple(
        Column(c["name"], c["type"], headers.get(c["name"], c["dictionary_name"]))
        for c in table["columns"]
    )
    return Relation(
        name=table["name"],
        columns=columns,
        dataset=source.get("dataset"),
        file=source.get("file"),
        ignore_headers=tuple(source.get("ignore_headers") or ()),
    )


def read_profile(profile_path: Path, schema_path: Path) -> Profile:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    tables = json.loads(schema_path.read_text(encoding="utf-8"))["tables"]
    sources: dict[str, dict[str, Any]] = profile["relations"]
    differ = set(sources) ^ {table["name"] for table in tables}
    if differ:
        raise LoadError(f"profile relations differ from schema: {', '.join(sorted(differ))}")
    return Profile(
        date_format=profile["date_format"],
        page_url=profile["page_url"],
        page_size=int(profile["page_size"]),
        count_url=profile["count_url"],
        relations=tuple(_relation(table, sources[table["name"]]) for table in tables),
    )


def map_headers(relation: Relation, headers: list[str]) -> list[str | None]:
    """The column each file position loads into; None for an ignored header."""
    by_header = {normalize_header(column.header): column for column in relation.columns}
    ignored = {normalize_header(header) for header in relation.ignore_headers}
    present = [normalize_header(header) for header in headers]
    problems = [
        f"{relation.name}: column {column.name} has no header {column.header!r}"
        for key, column in by_header.items() if key not in present
    ]  # fmt: skip
    problems += [
        f"{relation.name}: header {header!r} maps to no column"
        for header, key in zip(headers, present, strict=True)
        if key not in by_header and key not in ignored
    ]
    if len(set(present)) != len(present):
        problems.append(f"{relation.name}: the file repeats a header: {headers}")
    if problems:
        raise LoadError("\n".join(problems))
    return [by_header[key].name if key in by_header else None for key in present]
