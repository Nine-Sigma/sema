"""A failed or refused model call ends its ask; only a choice that fails is re-asked.

A call that exhausts the client's chain (``LLMStageError``) or is refused by the budget
(``BudgetExceededError``) is not a choice that failed a check, so the Structure and Roles
resolvers make no second call for that ask: the entries stay open with the failure as
their reason, the error text stays in the ledger event and the reason, and the resolver
returns so the run reaches Review. A verifier failure or a misaligned answer is still
re-asked at most twice. Whatever note reaches a re-ask prompt carries no run id and no
provider or exception text, so every call key is the same in any run.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from pydantic import BaseModel

from sema.intake.ledger_utils import prompt_hash
from sema.intake.output_models import ROLES_CHOICE, STRUCTURE_BATCH
from sema.intake.output_models_utils import ABSTAIN
from sema.llm_client import LLMStageError
from tests.unit.intake.model_calls_env import MODEL, WiredRun, intake_config, stats, wired_run
from tests.unit.intake.resolve_roles_env import (
    CONFLICT_TABLES,
    MEMORY_TABLES,
    Synthetic,
    answering,
    entry,
    prompts,
    relation,
    shared_column,
)
from tests.unit.intake.shapes.shapes_env import Source, warehouse_source
from tests.unit.intake.structure_env import fresh_tools, run
from tests.unit.intake.test_resolve_structure import CONFLICT_SQL, claim_own, open_subjects
from tests.unit.intake.test_resolve_structure import answering as structure_answering

pytestmark = pytest.mark.unit

PROVIDER_TEXT = "Error code: 400 context_length_exceeded request req-4f1a"
OTHER_PROVIDER_TEXT = "Error code: 400 invalid_request_error request req-9c2e"
FAILED_CALL_EVENT = "call_failed"
CONFLICT_ROLES = ("code", "display")


def resolve() -> Any:
    return importlib.import_module("sema.intake.resolve")


def provider_error(text: str = PROVIDER_TEXT) -> LLMStageError:
    return LLMStageError("", "intake", [("plain_invoke", ValueError(text))])


def failing(text: str = PROVIDER_TEXT) -> MagicMock:
    """A mocked `LLMClient` whose every invoke exhausts its chain with a non-transient 400."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    client.invoke.side_effect = provider_error(text)
    return client


def failing_after(client: MagicMock, answered: int) -> MagicMock:
    """``client`` answers its first ``answered`` invokes, then every invoke fails."""
    answer = client.invoke.side_effect
    count = [0]

    def _invoke(*args: Any, **kwargs: Any) -> BaseModel:
        count[0] += 1
        if count[0] > answered:
            raise provider_error()
        return answer(*args, **kwargs)  # type: ignore[no-any-return]

    client.invoke.side_effect = _invoke
    return client


def misaligned() -> MagicMock:
    """A mocked `LLMClient` choosing a member for every field but listing no confidence."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        properties = output_model.model_json_schema()["properties"]
        chosen = {
            name: next(value for value in field["enum"] if value != ABSTAIN)
            for name, field in properties.items()
            if "enum" in field
        }
        return output_model(**chosen, confidences=[], reasons=[])

    client.invoke.side_effect = _invoke
    return client


def refused_config(root: Path) -> Any:
    """A zero-call replay: the recordings directory holds no key and the cap is zero."""
    recordings = root / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    return intake_config(max_calls=0, recordings_dir=recordings)


class Watched:
    """A wired run whose every requested call key and every registry advance are kept."""

    def __init__(self, root: Path, client: MagicMock, config: Any, run_id: str) -> None:
        self.wired: WiredRun = wired_run(root / "calls", client, config, run_id)
        calls, registry = self.wired.calls, self.wired.registry
        self._call = patch.object(calls, "call", wraps=calls.call)
        self._advance = patch.object(registry, "advance", wraps=registry.advance)
        self.call = self._call.start()
        self.advance = self._advance.start()

    def keys(self) -> list[str]:
        calls = self.wired.calls
        return [calls.call_key(c.args[1], c.args[2]) for c in self.call.call_args_list]

    def stop(self) -> None:
        self._call.stop()
        self._advance.stop()


def watched(
    root: Path, client: MagicMock, *, config: Any | None = None, run_id: str = "run-a"
) -> Watched:
    return Watched(root, client, config or intake_config(), run_id)


@pytest.fixture(scope="module")
def tmp_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("resolve_call_failures")


@pytest.fixture(scope="module")
def roles_conflict(tmp_module: Path) -> Iterator[Synthetic]:
    built = Synthetic(CONFLICT_TABLES, tmp_module / "roles_conflict")
    yield built
    built.close()


@pytest.fixture(scope="module")
def roles_memory(tmp_module: Path) -> Iterator[Synthetic]:
    built = Synthetic(MEMORY_TABLES, tmp_module / "roles_memory")
    yield built
    built.close()


@pytest.fixture(scope="module")
def structure_conflict(tmp_module: Path) -> Iterator[Source]:
    con = duckdb.connect(":memory:")
    for statement in CONFLICT_SQL:
        con.execute(statement)
    yield warehouse_source(con, "tc", tmp_module / "structure_conflict" / "landing")
    con.close()


def resolved_roles(built: Synthetic, watch: Watched, root: Path) -> Any:
    tools = fresh_tools(built.source, root / "resolve", stage="roles", structure=built.structure)
    try:
        return resolve().resolve_roles(
            built.roles, built.structure, built.source.land, tools, watch.wired.calls,
            provider=built.source.provider,
        )
    finally:
        watch.stop()


def resolved_structure(source: Source, watch: Watched, root: Path) -> Any:
    structure = run(source, fresh_tools(source, root / "structure"))
    tools = fresh_tools(source, root / "resolve")
    try:
        return structure, resolve().resolve_structure(
            structure, source.land, tools, watch.wired.calls, provider=source.provider
        )
    finally:
        watch.stop()


def test_roles_call_failure_is_asked_once_and_left_open_with_the_error(
    roles_conflict: Synthetic, tmp_path: Path
) -> None:
    client = failing()
    watch = watched(tmp_path, client)

    result = resolved_roles(roles_conflict, watch, tmp_path)

    assert client.invoke.call_count == 1
    (failed,) = watch.wired.events(FAILED_CALL_EVENT)
    assert failed["payload"]["prompt_hash"] == watch.keys()[0]
    assert PROVIDER_TEXT in str(failed["payload"]["step_errors"])
    assert watch.wired.row().model_calls == 1
    for role in CONFLICT_ROLES:
        left = entry(result, relation("zeta"), role)
        assert (left.choice, left.source, left.call_key) == (None, None, None)
        assert left.reason is not None and PROVIDER_TEXT in left.reason


def test_roles_call_failure_makes_one_call_per_ask(
    roles_memory: Synthetic, tmp_path: Path
) -> None:
    client = failing()
    watch = watched(tmp_path, client)

    resolved_roles(roles_memory, watch, tmp_path)

    keys = watch.keys()
    assert len(keys) == len(set(keys)) == 2
    assert client.invoke.call_count == 2


def test_roles_call_failure_on_a_re_ask_ends_the_ask(
    roles_conflict: Synthetic, tmp_path: Path
) -> None:
    client = failing_after(answering(shared_column), answered=1)
    watch = watched(tmp_path, client)

    result = resolved_roles(roles_conflict, watch, tmp_path)

    assert client.invoke.call_count == 2
    for role in CONFLICT_ROLES:
        assert entry(result, relation("zeta"), role).choice is None


def test_roles_budget_refusal_is_refused_once_per_key(
    roles_conflict: Synthetic, tmp_path: Path
) -> None:
    client = answering(shared_column)
    watch = watched(tmp_path, client, config=refused_config(tmp_path))

    result = resolved_roles(roles_conflict, watch, tmp_path)

    assert client.invoke.call_count == 0
    assert watch.keys() == watch.keys()[:1]
    assert watch.advance.call_count == 1
    assert watch.wired.calls.refused
    for role in CONFLICT_ROLES:
        left = entry(result, relation("zeta"), role)
        assert left.choice is None
        assert left.reason is not None and "budget" in left.reason


@pytest.mark.parametrize("failing_choice", ["verifier", "misaligned"])
def test_roles_failed_choice_is_still_re_asked_twice(
    roles_conflict: Synthetic, tmp_path: Path, failing_choice: str
) -> None:
    client = answering(shared_column) if failing_choice == "verifier" else misaligned()
    watch = watched(tmp_path, client)

    resolved_roles(roles_conflict, watch, tmp_path)

    assert client.invoke.call_count == 3
    assert len(set(watch.keys())) == 3


def test_roles_re_ask_keys_do_not_depend_on_the_run_id(
    roles_conflict: Synthetic, tmp_path: Path
) -> None:
    sequences = []
    for run_id in ("run-a", "run-b"):
        root = tmp_path / run_id
        client = answering(shared_column)
        resolved_roles(roles_conflict, watched(root, client, run_id=run_id), root)
        sequences.append([prompt_hash(MODEL, p, ROLES_CHOICE) for p in prompts(client)])

    assert len(sequences[0]) == 3
    assert sequences[0] == sequences[1]


def test_roles_keys_match_across_call_failures_and_budget_refusals(
    roles_conflict: Synthetic, tmp_path: Path
) -> None:
    runs = {
        "live-a": (failing(PROVIDER_TEXT), None),
        "live-b": (failing(OTHER_PROVIDER_TEXT), None),
        "replay-a": (answering(shared_column), refused_config(tmp_path / "replay-a")),
        "replay-b": (answering(shared_column), refused_config(tmp_path / "replay-b")),
    }
    sequences = {}
    for run_id, (client, config) in runs.items():
        root = tmp_path / run_id
        watch = watched(root, client, config=config, run_id=run_id)
        resolved_roles(roles_conflict, watch, root)
        sequences[run_id] = watch.keys()

    first = sequences["live-a"]
    assert first
    assert all(keys == first for keys in sequences.values()), sequences


def test_structure_call_failure_is_asked_once_and_left_open_with_the_error(
    structure_conflict: Source, tmp_path: Path
) -> None:
    client = failing()
    watch = watched(tmp_path, client)

    structure, result = resolved_structure(structure_conflict, watch, tmp_path)

    assert client.invoke.call_count == 1
    (failed,) = watch.wired.events(FAILED_CALL_EVENT)
    assert failed["payload"]["prompt_hash"] == watch.keys()[0]
    assert PROVIDER_TEXT in str(failed["payload"]["step_errors"])
    assert watch.wired.row().model_calls == 1
    assert open_subjects(result) == open_subjects(structure)
    for ambiguity in result.ambiguities:
        assert ambiguity.choice is None
        assert ambiguity.reason is not None and PROVIDER_TEXT in ambiguity.reason


def test_structure_call_failure_on_a_re_ask_ends_the_ask(
    structure_conflict: Source, tmp_path: Path
) -> None:
    client = failing_after(structure_answering(claim_own), answered=1)
    watch = watched(tmp_path, client)

    structure, result = resolved_structure(structure_conflict, watch, tmp_path)

    assert client.invoke.call_count == 2
    assert open_subjects(result) == open_subjects(structure)


def test_structure_budget_refusal_is_refused_once_per_key(
    structure_conflict: Source, tmp_path: Path
) -> None:
    client = structure_answering(claim_own)
    watch = watched(tmp_path, client, config=refused_config(tmp_path))

    structure, result = resolved_structure(structure_conflict, watch, tmp_path)

    assert client.invoke.call_count == 0
    assert watch.keys() == watch.keys()[:1]
    assert watch.advance.call_count == 1
    assert open_subjects(result) == open_subjects(structure)
    for ambiguity in result.ambiguities:
        assert ambiguity.reason is not None and "budget" in ambiguity.reason


@pytest.mark.parametrize("failing_choice", ["verifier", "misaligned"])
def test_structure_failed_choice_is_still_re_asked_twice(
    structure_conflict: Source, tmp_path: Path, failing_choice: str
) -> None:
    client = structure_answering(claim_own) if failing_choice == "verifier" else misaligned()
    watch = watched(tmp_path, client)

    resolved_structure(structure_conflict, watch, tmp_path)

    assert client.invoke.call_count == 3
    assert len(set(watch.keys())) == 3


def test_structure_re_ask_keys_do_not_depend_on_the_run_id(
    structure_conflict: Source, tmp_path: Path
) -> None:
    sequences = []
    for run_id in ("run-a", "run-b"):
        root = tmp_path / run_id
        client = structure_answering(claim_own)
        resolved_structure(structure_conflict, watched(root, client, run_id=run_id), root)
        sequences.append([prompt_hash(MODEL, p, STRUCTURE_BATCH) for p in prompts(client)])

    assert len(sequences[0]) == 3
    assert sequences[0] == sequences[1]


def test_structure_keys_match_across_call_failures_and_budget_refusals(
    structure_conflict: Source, tmp_path: Path
) -> None:
    runs = {
        "live-a": (failing(PROVIDER_TEXT), None),
        "live-b": (failing(OTHER_PROVIDER_TEXT), None),
        "replay-a": (structure_answering(claim_own), refused_config(tmp_path / "replay-a")),
        "replay-b": (structure_answering(claim_own), refused_config(tmp_path / "replay-b")),
    }
    sequences = {}
    for run_id, (client, config) in runs.items():
        root = tmp_path / run_id
        watch = watched(root, client, config=config, run_id=run_id)
        resolved_structure(structure_conflict, watch, root)
        sequences[run_id] = watch.keys()

    first = sequences["live-a"]
    assert first
    assert all(keys == first for keys in sequences.values()), sequences
