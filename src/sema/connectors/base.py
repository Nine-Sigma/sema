from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sema.models.assertions import Assertion


class Connector(ABC):
    """Base class for metadata source connectors.

    Each connector extracts metadata from a source and returns
    normalized semantic assertions.
    """

    @abstractmethod
    def extract(
        self,
        catalog: str,
        schemas: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Assertion]:
        """Extract metadata and return source-scoped assertions."""
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
