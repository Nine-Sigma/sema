from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sema.models.assertions import Assertion
from sema.models.extraction import (
    ExtractedColumn,
    ExtractedForeignKey,
    ExtractedSampleRows,
    ExtractedTable,
    ExtractedTag,
    ExtractedTopValues,
)


class Connector(ABC):
    """Base class for metadata source connectors.

    Each connector extracts metadata from a source and returns
    normalized semantic assertions.

    New connectors should implement the DTO-based extract methods
    (extract_tables_dto, extract_columns_dto, etc.) and use the
    AssertionNormalizer for assertion creation. The legacy extract()
    method is kept for backward compatibility with the Databricks
    connector.
    """

    @abstractmethod
    def extract(
        self,
        catalog: str,
        schemas: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Assertion]:
        """Extract metadata and return source-scoped assertions.

        Legacy method — new connectors should implement the DTO-based
        methods below and use AssertionNormalizer instead.
        """
        ...

    @abstractmethod
    def list_catalogs(self) -> list[str]:
        """Return available catalog names from the source."""
        ...

    @abstractmethod
    def get_datasource_ref(self) -> tuple[str, str, str]:
        """Return (ref, platform, workspace) for this data source.

        ref      — canonical URI, e.g. ``databricks://<workspace>``
        platform — platform name, e.g. ``"databricks"``
        workspace — workspace identifier, e.g. the host name
        """
        ...

    # --- DTO-based extraction (new connector contract) ---

    def extract_tables_dto(
        self, catalog: str, schemas: list[str] | None = None,
    ) -> list[ExtractedTable]:
        """Extract table metadata as DTOs. Override in new connectors."""
        raise NotImplementedError

    def extract_columns_dto(
        self, catalog: str, schema: str, table: str,
    ) -> list[ExtractedColumn]:
        """Extract column metadata as DTOs. Override in new connectors."""
        raise NotImplementedError

    def extract_foreign_keys_dto(
        self, catalog: str, schema: str, table: str,
    ) -> list[ExtractedForeignKey]:
        """Extract FK relationships as DTOs. Override in new connectors."""
        raise NotImplementedError

    def extract_sample_rows_dto(
        self, catalog: str, schema: str, table: str, limit: int = 10,
    ) -> ExtractedSampleRows | None:
        """Extract sample rows as a DTO. Override in new connectors."""
        raise NotImplementedError

    def extract_top_values_dto(
        self, catalog: str, schema: str, table: str, column: str, k: int = 20,
    ) -> ExtractedTopValues | None:
        """Extract top-K values for a column. Override in new connectors."""
        raise NotImplementedError

    def extract_tags_dto(
        self, catalog: str, schema: str, table: str,
    ) -> list[ExtractedTag]:
        """Extract tags/labels. Override in new connectors."""
        raise NotImplementedError
