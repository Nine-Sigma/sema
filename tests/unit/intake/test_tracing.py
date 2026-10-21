"""Run tracing: the backend comes from configuration, spans carry run id and stage.

Pinned API (task 12.2):

- ``sema.models.config.TracingConfig`` is a ``BaseSettings`` beside ``IntakeModelConfig`` with
  env prefix ``INTAKE_TRACING_`` and one field, ``backend: Literal["none", "mlflow",
  "langfuse"] = "none"``. ``langfuse`` is declared and rejected at validation with a message
  naming it and the slice that adds it (``S4``). Where traces land is MLflow's own setting
  (``MLFLOW_TRACKING_URI`` and the active experiment); SEMA adds no field for it.
- ``sema.intake.tracing.run_tracing(config, run_id, entry=...)`` is a context manager yielding
  a tracer. Each graph entry point (``discover``, ``approve``, ``reject``, ``approve_values``,
  ``resume``) enters it once, before the ``land`` node, so each entry is one trace. ``none`` imports and attaches
  nothing. ``mlflow`` imports ``mlflow`` on enter (never at module import), raises
  ``sema.intake.tracing.TracingUnavailableError`` naming the ``mlflow-tracing`` package and
  the ``tracing-mlflow`` extra when the import fails, calls ``mlflow.langchain.autolog()``,
  and turns autolog off on exit.
- ``tracer.node(stage)`` is a context manager the graph enters around each node body. Every
  span emitted inside the run carries ``sema.run_id`` and ``sema.stage``: spans inside a node,
  including the chat-model span of an ``LLMClient`` call, take the node's stage; LangGraph's
  own span for a node takes the node name; the trace's root span takes the entry name.
- ``sema.intake.graph.discover`` builds ``TracingConfig()`` from the environment when it is
  not handed one, so a missing package fails the run before Land.
- ``pyproject.toml`` declares the extra ``tracing-mlflow = ["mlflow-tracing..."]``.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from sema.llm_client import LLMClient
from tests.unit.intake.graph_env import (
    A2_DESCRIPTOR,
    GraphRun,
    approve,
    code_tables,
    discover,
    graph,
    routed_client,
)
from tests.unit.intake.structure_env import A2_WORKBOOK

pytestmark = pytest.mark.unit

PYPROJECT = Path(__file__).parents[3] / "pyproject.toml"
BACKEND_ENV = "INTAKE_TRACING_BACKEND"
RUN_ATTR = "sema.run_id"
STAGE_ATTR = "sema.stage"
STAGE = "roles"
ENTRY = "discover"
PROMPT = "# prompt: tracing v1\nChoose."
ANSWER = "bound"
PACKAGE = "mlflow-tracing"
EXTRA = "tracing-mlflow"
CHAT_MODEL_SPAN = "CHAT_MODEL"


class Answer(BaseModel):
    answer: str


class NodeState(TypedDict):
    answer: str


def tracing() -> Any:
    import sema.intake.tracing as module

    return module


def tracing_config(**values: Any) -> Any:
    from sema.models.config import TracingConfig

    return TracingConfig(**values)


def fake_client() -> LLMClient:
    chat = FakeListChatModel(responses=[Answer(answer=ANSWER).model_dump_json()] * 4)
    return LLMClient(chat, use_structured_output="off")


def run_one_node(tracer: Any, client: LLMClient) -> str:
    """One LangGraph node that makes one call through a real ``LLMClient``."""

    def roles(state: NodeState) -> NodeState:
        with tracer.node(STAGE):
            return {"answer": client.invoke(PROMPT, Answer, stage_name=STAGE).answer}

    builder = StateGraph(NodeState)
    builder.add_node(STAGE, roles)
    builder.add_edge(START, STAGE)
    builder.add_edge(STAGE, END)
    result = builder.compile().invoke({"answer": ""})
    return str(result["answer"])


def all_traces() -> list[Any]:
    import mlflow

    return list(mlflow.search_traces(return_type="list"))


def root_span(trace: Any) -> Any:
    (root,) = [span for span in trace.data.spans if span.parent_id is None]
    return root


def traces_for(run_id: str) -> list[Any]:
    return [
        trace for trace in all_traces()
        if any(span.attributes.get(RUN_ATTR) == run_id for span in trace.data.spans)
    ]


@pytest.fixture(scope="module")
def mlflow_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A local MLflow store in a temporary directory; no network."""
    import mlflow

    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        yield uri
        mlflow.langchain.autolog(disable=True)
        mlflow.set_tracking_uri(previous)


@pytest.fixture
def without_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes every ``import mlflow`` fail as if the package were not installed."""
    for name in [name for name in sys.modules if name.startswith("mlflow.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mlflow", None)


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BACKEND_ENV, raising=False)


def test_backend_defaults_to_none() -> None:
    assert tracing_config().backend == "none"


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_langfuse_is_rejected_naming_the_slice_that_adds_it(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {"backend": "langfuse"} if source == "argument" else {}
    if source == "environment":
        monkeypatch.setenv(BACKEND_ENV, "langfuse")

    with pytest.raises(ValidationError) as rejected:
        tracing_config(**values)

    assert "langfuse" in str(rejected.value)
    assert "S4" in str(rejected.value)


def test_an_unnamed_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        tracing_config(backend="langsmith")


def test_none_attaches_nothing(mlflow_store: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow.langchain

    autolog = MagicMock()
    monkeypatch.setattr(mlflow.langchain, "autolog", autolog)
    before = len(all_traces())

    with tracing().run_tracing(tracing_config(backend="none"), "run-none", entry=ENTRY) as tracer:
        answer = run_one_node(tracer, fake_client())

    assert answer == ANSWER
    autolog.assert_not_called()
    assert len(all_traces()) == before


@pytest.mark.usefixtures("without_mlflow")
def test_none_runs_without_the_tracing_package() -> None:
    with tracing().run_tracing(tracing_config(backend="none"), "run-bare", entry=ENTRY) as tracer:
        answer = run_one_node(tracer, fake_client())

    assert answer == ANSWER


def test_mlflow_every_span_carries_run_id_and_stage(mlflow_store: str) -> None:
    run_id = "run-mlflow"

    with tracing().run_tracing(tracing_config(backend="mlflow"), run_id, entry=ENTRY) as tracer:
        answer = run_one_node(tracer, fake_client())

    traces = traces_for(run_id)
    assert answer == ANSWER
    assert len(traces) == 1
    spans = traces[0].data.spans
    chat = [span for span in spans if span.span_type == CHAT_MODEL_SPAN]
    assert len(chat) == 1
    assert chat[0].attributes.get(STAGE_ATTR) == STAGE
    assert {span.attributes.get(RUN_ATTR) for span in spans} == {run_id}
    assert all(span.attributes.get(STAGE_ATTR) for span in spans)
    graph_node = [span for span in spans if span.name == STAGE and span.span_type == "CHAIN"]
    assert [span.attributes.get(STAGE_ATTR) for span in graph_node] == [STAGE]
    assert root_span(traces[0]).attributes.get(STAGE_ATTR) == ENTRY


def test_mlflow_tracing_stops_when_the_run_ends(mlflow_store: str) -> None:
    client = fake_client()
    with tracing().run_tracing(tracing_config(backend="mlflow"), "run-ended", entry=ENTRY):
        pass
    before = len(all_traces())

    client.invoke(PROMPT, Answer, stage_name=STAGE)

    assert len(all_traces()) == before


@pytest.mark.usefixtures("without_mlflow")
def test_missing_mlflow_fails_at_run_start_naming_the_package() -> None:
    module = tracing()
    entered: list[bool] = []

    with pytest.raises(module.TracingUnavailableError) as missing:
        with module.run_tracing(tracing_config(backend="mlflow"), "run-missing", entry=ENTRY):
            entered.append(True)

    assert entered == []
    assert PACKAGE in str(missing.value)
    assert EXTRA in str(missing.value)


def test_missing_mlflow_fails_discover_before_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    graph()  # imported before the block, so only the tracing check can hit the missing package
    unavailable = tracing().TracingUnavailableError
    source = code_tables(tmp_path)
    client = MagicMock()
    run_id = "run-before-land"
    monkeypatch.setenv(BACKEND_ENV, "mlflow")
    request.getfixturevalue("without_mlflow")

    with pytest.raises(unavailable, match=PACKAGE):
        discover(tmp_path, source, A2_DESCRIPTOR, client, run_id=run_id)

    assert GraphRun(tmp_path, run_id, client).nodes() == []
    assert client.invoke.call_count == 0


def test_each_graph_entry_is_one_trace_of_the_run(
    tmp_path: Path, mlflow_store: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-entries"
    monkeypatch.setenv(BACKEND_ENV, "mlflow")

    approve(discover(tmp_path, A2_WORKBOOK, A2_DESCRIPTOR, routed_client(), run_id=run_id))

    traces = traces_for(run_id)
    assert sorted(root_span(t).attributes.get(STAGE_ATTR) for t in traces) == ["approve", ENTRY]
    spans = [span for trace in traces for span in trace.data.spans]
    assert {span.attributes.get(RUN_ATTR) for span in spans} == {run_id}
    assert all(span.attributes.get(STAGE_ATTR) for span in spans)


def test_tracing_package_is_an_optional_extra() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras = project.get("optional-dependencies", {})

    assert any(entry.startswith(PACKAGE) for entry in extras.get(EXTRA, []))
    assert not any(entry.startswith(("mlflow", PACKAGE)) for entry in project["dependencies"])
