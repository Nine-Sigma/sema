"""`sema target discover` hands the graph SEMA's embedder only when one is configured.

Pinned (graph follow-up adds ``discover(..., embedder=None)``): with ``EMBEDDING_PROVIDER`` in
the environment, discover passes ``embedder=cli_factories._get_embedder(EmbeddingConfig())``,
resolved through ``sema.cli_factories`` at call time; without it, ``embedder=None`` and the
factory is never called. The real graph runs with the embedder dropped, so no provider is hit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.models.config import EmbeddingConfig
from tests.unit.cli_intake_env import Cli, intake_cli, spy_discover
from tests.unit.intake.graph_env import A2_DESCRIPTOR
from tests.unit.intake.structure_env import A2_WORKBOOK

pytestmark = pytest.mark.unit

FACTORY = "sema.cli_factories._get_embedder"
EMBEDDING_ENV = {"EMBEDDING_PROVIDER": "openrouter", "EMBEDDING_MODEL": "vendor-a/embed-x"}


@pytest.mark.parametrize("configured", [True, False], ids=["provider-set", "provider-unset"])
def test_discover_passes_the_configured_embedder_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    cli: Cli = intake_cli(tmp_path / "runs", monkeypatch)
    for name, value in EMBEDDING_ENV.items():
        if configured:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    built: list[EmbeddingConfig] = []
    embedder = object()

    def _factory(config: EmbeddingConfig) -> Any:
        built.append(config)
        return embedder

    monkeypatch.setattr(FACTORY, _factory)
    seen = spy_discover(monkeypatch, drop=("embedder",))

    cli.discover(A2_WORKBOOK, A2_DESCRIPTOR)

    ((_, kwargs),) = seen
    assert "embedder" in kwargs
    if configured:
        assert kwargs["embedder"] is embedder
        (config,) = built
        assert (config.provider, config.model) == tuple(EMBEDDING_ENV.values())
    else:
        assert kwargs["embedder"] is None
        assert built == []
