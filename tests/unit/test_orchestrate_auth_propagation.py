"""Retrieval-path catches must re-raise DatabricksProviderAuthError."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sema.cli_factories import DatabricksProviderAuthError
from sema.models.config import EmbeddingConfig, QueryConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def query_config():
    return QueryConfig(
        question="What are our diagnoses?",
        embedding=EmbeddingConfig(
            provider="databricks", model="databricks-bge-large-en",
        ),
    )


class TestOrchestrateRunContextAuthPropagation:
    def test_databricks_auth_error_propagates_from_run_context(
        self, query_config,
    ):
        with patch(
            "sema.pipeline.orchestrate._get_neo4j_driver",
        ) as mock_driver, patch(
            "sema.pipeline.orchestrate._get_embedder",
            side_effect=DatabricksProviderAuthError("missing creds"),
        ):
            mock_driver.return_value = MagicMock()
            from sema.pipeline.orchestrate import run_context
            with pytest.raises(DatabricksProviderAuthError):
                run_context(query_config)

    def test_generic_embedder_exception_still_degrades(self, query_config):
        query_config.embedding = EmbeddingConfig(
            provider="openrouter", model="openai/text-embedding-ada-002",
        )
        with patch(
            "sema.pipeline.orchestrate._get_neo4j_driver",
        ) as mock_driver, patch(
            "sema.pipeline.orchestrate._get_embedder",
            side_effect=RuntimeError("network blip"),
        ), patch(
            "sema.pipeline.retrieval.RetrievalEngine",
        ) as mock_engine_cls, patch(
            "sema.pipeline.context.prune_to_sco",
            return_value=MagicMock(
                model_dump=MagicMock(return_value={}),
            ),
        ):
            mock_driver.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.retrieve.return_value = MagicMock(
                model_dump=MagicMock(return_value={}),
            )
            mock_engine_cls.return_value = mock_engine
            from sema.pipeline.orchestrate import run_context
            run_context(query_config)
            kwargs = mock_engine_cls.call_args.kwargs
            assert kwargs["embedder"] is None


class TestOrchestrateRetrieveContextAuthPropagation:
    def test_databricks_auth_error_propagates(self, query_config):
        with patch(
            "sema.pipeline.orchestrate_utils._get_neo4j_driver",
        ) as mock_driver, patch(
            "sema.cli_factories._get_embedder",
            side_effect=DatabricksProviderAuthError("missing creds"),
        ):
            mock_driver.return_value = MagicMock()
            from sema.pipeline.orchestrate_utils import _retrieve_context
            with pytest.raises(DatabricksProviderAuthError):
                _retrieve_context(query_config)

    def test_generic_embedder_exception_still_degrades(self, query_config):
        query_config.embedding = EmbeddingConfig(
            provider="openrouter", model="openai/text-embedding-ada-002",
        )
        with patch(
            "sema.pipeline.orchestrate_utils._get_neo4j_driver",
        ) as mock_driver, patch(
            "sema.cli_factories._get_embedder",
            side_effect=RuntimeError("network blip"),
        ), patch(
            "sema.pipeline.orchestrate_utils.RetrievalEngine",
        ) as mock_engine_cls, patch(
            "sema.pipeline.orchestrate_utils.prune_to_sco",
            return_value={},
        ):
            mock_driver.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.retrieve.return_value = {}
            mock_engine_cls.return_value = mock_engine
            from sema.pipeline.orchestrate_utils import _retrieve_context
            _retrieve_context(query_config)
            kwargs = mock_engine_cls.call_args.kwargs
            assert kwargs["embedder"] is None
