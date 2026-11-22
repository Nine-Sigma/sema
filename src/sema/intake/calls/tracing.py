"""Trace an ingestion run to the backend its configuration names (design Decision 14).

``mlflow`` is imported only when a run enters with that backend, so SEMA runs without it.
Autolog traces the LangChain runnables ``LLMClient`` invokes inside the graph's nodes; a span
processor tags every span with the run id and the stage it ran in, joining a trace to the
run ledger.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from types import ModuleType
from typing import Any, Protocol

from sema.log import logger
from sema.models.config import TracingConfig

__all__ = ["Tracer", "TracingUnavailableError", "run_tracing"]

RUN_ATTRIBUTE = "sema.run_id"
STAGE_ATTRIBUTE = "sema.stage"
METADATA_ATTRIBUTE = "metadata"
NODE_METADATA = "langgraph_node"
MLFLOW_PACKAGE = "mlflow-tracing"
MLFLOW_EXTRA = "tracing-mlflow"


class TracingUnavailableError(RuntimeError):
    """The configured tracing backend's package is not installed."""


class Tracer(Protocol):
    def node(self, stage: str) -> AbstractContextManager[None]: ...


class _Untraced:
    def node(self, stage: str) -> AbstractContextManager[None]:
        return nullcontext()


class _MlflowTracer:
    """Tags spans in order: the node a span ran in, LangGraph's own node name, the entry."""

    def __init__(self, run_id: str, entry: str) -> None:
        self._run_id = run_id
        self._entry = entry
        self._stage: ContextVar[str | None] = ContextVar("sema_stage", default=None)

    @contextmanager
    def node(self, stage: str) -> Iterator[None]:
        token = self._stage.set(stage)
        try:
            yield
        finally:
            self._stage.reset(token)

    def tag(self, span: Any) -> None:
        metadata = span.attributes.get(METADATA_ATTRIBUTE)
        node = metadata.get(NODE_METADATA) if isinstance(metadata, dict) else None
        span.set_attribute(RUN_ATTRIBUTE, self._run_id)
        span.set_attribute(STAGE_ATTRIBUTE, self._stage.get() or node or self._entry)


@contextmanager
def run_tracing(config: TracingConfig, run_id: str, *, entry: str) -> Iterator[Tracer]:
    """One trace per entry point; a missing backend package fails before the body runs."""
    if config.backend == "none":
        yield _Untraced()
        return
    mlflow = _import_mlflow()
    tracer = _MlflowTracer(run_id, entry)
    with mlflow.tracing.configure(span_processors=[tracer.tag]):
        mlflow.langchain.autolog()
        logger.debug("Tracing run {} entry {} to mlflow", run_id, entry)
        try:
            with mlflow.start_span(name=entry):
                yield tracer
        finally:
            mlflow.langchain.autolog(disable=True)


def _import_mlflow() -> ModuleType:
    try:
        import mlflow
        import mlflow.langchain  # noqa: F401 - autolog lives in the submodule
    except ImportError as missing:
        raise TracingUnavailableError(
            f"tracing backend 'mlflow' needs the {MLFLOW_PACKAGE} package; "
            f"install SEMA with the {MLFLOW_EXTRA} extra"
        ) from missing
    return mlflow
