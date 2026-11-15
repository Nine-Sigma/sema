"""Shared setup for test_sema_build.py and the tests that reuse it."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.graph.physical_catalog import PhysicalColumn, StaticPhysicalCatalogSource
from showcase.ontology_query_benchmark import sema_live
from showcase.ontology_query_benchmark.routing import (
    FORBIDDEN_ENV,
    FORBIDDEN_ENV_PREFIXES,
    PROVIDER_ENV,
)
from tests.unit.ontology_query_benchmark.sema_bindings_env import catalog_fixture


class ClientBuilt(Exception):
    """A live client was about to be built."""


def clean_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process the routing check accepts: no dotenv module and no routing variable."""
    routed = (*FORBIDDEN_ENV, *PROVIDER_ENV, "DATABRICKS_HOST", "DATABRICKS_CONFIG_PROFILE")
    for name in list(os.environ):
        if name in routed or name.startswith(FORBIDDEN_ENV_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.delitem(sys.modules, "dotenv", raising=False)


def no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ClientBuilt

    monkeypatch.setattr(sema_live, "pinned_workspace", refuse)
    monkeypatch.setattr(sema_live, "neo4j_driver", refuse)


def _catalog() -> StaticPhysicalCatalogSource:
    return StaticPhysicalCatalogSource({
        ("cat", "sch", table): [PhysicalColumn(c, t, True) for c, t in cols.items()]
        for table, cols in catalog_fixture().items()
    })


def _embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda texts: [[0.5] * 4 for _ in texts]
    return embedder
