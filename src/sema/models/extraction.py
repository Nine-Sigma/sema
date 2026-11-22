"""Connector-neutral extraction DTOs.

These types define the contract between connectors and the
normalization layer. Connectors emit DTOs; the normalizer
converts them to canonical assertions. Connectors should
never import assertion types.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractedTable:
    name: str
    catalog: str
    schema: str
    comment: str | None = None
    tags: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedColumn:
    name: str
    table_name: str
    catalog: str
    schema: str
    data_type: str
    nullable: bool = True
    comment: str | None = None
    ref: str | None = None


@dataclass(frozen=True)
class ExtractedForeignKey:
    """One declared constraint; `column_pairs` is (referencing, referenced) in ordinal order."""
    constraint_name: str
    referencing_catalog: str
    referencing_schema: str
    referencing_table: str
    referenced_catalog: str
    referenced_schema: str
    referenced_table: str
    column_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ExtractedTag:
    table_name: str
    column_name: str | None
    tag_key: str
    tag_value: str
    catalog: str = ""
    schema: str = ""


@dataclass(frozen=True)
class ExtractedProfile:
    """Column-level statistics from profiling."""
    column_name: str
    table_name: str
    catalog: str
    schema: str
    approx_distinct: int
    data_type: str


@dataclass(frozen=True)
class ExtractedSampleRows:
    """Sample data from a table."""
    table_name: str
    catalog: str
    schema: str
    rows: list[dict[str, str]]
    column_names: list[str]


@dataclass(frozen=True)
class ExtractedTopValues:
    """Top-K values for a categorical column."""
    column_name: str
    table_name: str
    catalog: str
    schema: str
    values: list[dict[str, str]]
    approx_distinct: int
