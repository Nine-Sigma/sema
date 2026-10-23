"""The recording script: no-hint runs to Review, call records copied, replay verified.

Contract the tests pin (task 14.3; intake-model-calls "Committed recordings are a second
replay source", design Decision 10):

- ``scripts.record_intake_run_utils``: ``RECORDINGS_DIR`` (``tests/fixtures/intake/
  recordings``), ``MANIFEST_FILE = "recording.json"``, ``RECORDING_PROVIDER = "openrouter"``;
  ``RecordedSource(name, path, descriptor, schema=None, build=None)``, frozen, where
  ``descriptor`` is a committed run descriptor path or a ``RunDescriptor`` written at
  record time, and ``build`` (when set) writes the source under a work directory;
  ``prepare(source, workdir) -> (source_path, descriptor_path)``;
  ``check_provider(config)`` raising ``IntakeProviderNotSetError`` when no provider is set
  and ``ProviderRefusedError`` (a ``ValueError``) naming any provider but OpenRouter;
  ``RecordingError`` (a run that stopped anywhere but Review) and ``ReplayMismatchError``.
- ``scripts.record_intake_run``: ``RECORDED_SOURCES``; ``record_source(source, *,
  recordings, workdir, config, client=None) -> Recording(run_id, outcome, directory,
  manifest)`` calls ``sema.intake.graph.discover`` through the module with no hint and no
  decisions, stops at Review, replaces ``<recordings>/<name>/calls/`` with the run ledger's
  ``calls/*.json``, and writes ``<recordings>/<name>/recording.json`` with exactly
  ``source, provider, model, prompt_versions, calls, tokens, cost, approver, value_asks``;
  ``verify_replay(source, *, recordings, workdir, config, client) -> RunOutcome`` runs a
  fresh run with ``recordings_dir`` set and raises unless it reaches Review with zero
  provider calls; ``main(argv) -> int`` with ``--source`` (repeatable), ``--recordings``,
  ``--workdir``, building ``IntakeModelConfig()`` from the environment.

Held-out sources are listed by path only: no test here opens a held-out file.
Modules are imported lazily so a missing one fails each test on its own.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from sema.intake.eligibility_utils import RunDescriptor
from sema.intake.model_calls import IntakeProviderNotSetError
from sema.intake.registry import Registry
from tests.support.athena_bundle import make_minimal_bundle
from tests.unit.intake.graph_env import ACRIS_DESCRIPTOR, REGISTRY_FILE, code_tables, graph
from tests.unit.intake.graph_env import routed_client
from tests.unit.intake.model_calls_env import MODEL, intake_config
from tests.unit.intake.shapes.shapes_env import FIXTURES
from tests.unit.record_intake_run_env import (
    ENV_TEXT,
    HELD_OUT,
    REVIEW_PARTS,
    fake_discover,
    forbid,
    guard_held_out_reads,
    model_env,
)

pytestmark = pytest.mark.unit

SOURCE_NAMES = {
    "a1", "a2", "a3", "a7_csv", "a7_athena", "a10",
    "heldout", "heldout2", "heldout3", "heldout4", "heldout5",
}
COMMITTED_DESCRIPTORS = {
    "a1": FIXTURES / "partner_sl" / "standard_library.run.yaml",
    "a2": FIXTURES / "tabular" / "codes_a2.run.yaml",
    "a3": FIXTURES / "semsql" / "ontology.run.yaml",
    "a10": FIXTURES / "acris" / "acris.run.yaml",
}
SOURCE_PATHS = {
    "a1": FIXTURES / "partner_sl" / "library.duckdb",
    "a2": FIXTURES / "tabular" / "codes_a2.xlsx",
    "a3": FIXTURES / "semsql" / "ontology.db",
    "a7_csv": FIXTURES / "omop" / "OMOP_CDMv5.4_Field_Level.csv",
    "a10": FIXTURES / "acris",
}
MANIFEST_KEYS = {"source", "provider", "model", "prompt_versions", "calls", "tokens", "cost",
                 "approver", "value_asks", "agreement"}
FORBIDDEN_ENTRIES = ("approve", "approve_values", "reject", "resume")
VERSION_PREFIX = "# prompt: "


def utils() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run_utils")


def script() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run")


def sources() -> dict[str, Any]:
    return {source.name: source for source in script().RECORDED_SOURCES}


def code_table_source(root: Path, name: str = "code_tables") -> Any:
    """ACRIS code tables: a small known source whose structure entries reach the model.

    A2 settles every entry by code and makes no model call, so it cannot show a replay.
    """
    return utils().RecordedSource(name=name, path=code_tables(root), descriptor=ACRIS_DESCRIPTOR)


def call_files(directory: Path) -> list[str]:
    return sorted(path.name for path in (directory / "calls").glob("*.json"))


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One recording of the code tables through the real graph, with discover spied on."""
    root = tmp_path_factory.mktemp("record")
    source = code_table_source(root)
    client = routed_client()
    seen: list[dict[str, Any]] = []
    real = graph().discover

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graph(), "discover", spy)
        for name in FORBIDDEN_ENTRIES:
            patch.setattr(graph(), name, forbid(name))
        recording = script().record_source(
            source,
            recordings=root / "recordings",
            workdir=root / "work",
            config=intake_config(max_calls=200),
            client=client,
        )
    yield {"root": root, "source": source, "client": client, "seen": seen, "rec": recording}


# --- the source table -------------------------------------------------------------------


def test_source_table_names_every_recorded_source() -> None:
    assert set(sources()) == SOURCE_NAMES
    assert utils().RECORDINGS_DIR == FIXTURES / "recordings"


def test_known_sources_point_at_their_committed_fixtures() -> None:
    table = sources()

    for name, path in SOURCE_PATHS.items():
        assert table[name].path == path and path.exists(), name
    for name, descriptor in COMMITTED_DESCRIPTORS.items():
        assert table[name].descriptor == descriptor and descriptor.is_file(), name


def test_athena_entry_is_built_by_the_loader_tests_bundle(tmp_path: Path) -> None:
    entry = sources()["a7_athena"]

    assert entry.build is make_minimal_bundle
    path, descriptor = utils().prepare(entry, tmp_path)
    assert (path / "CONCEPT.csv").is_file()
    assert path.resolve().is_relative_to(tmp_path.resolve())
    RunDescriptor.model_validate(yaml.safe_load(descriptor.read_text(encoding="utf-8")))


def test_prepare_writes_an_uncommitted_descriptor_and_keeps_a_committed_one(
    tmp_path: Path,
) -> None:
    a7_path, a7_descriptor = utils().prepare(sources()["a7_csv"], tmp_path)
    a2_path, a2_descriptor = utils().prepare(sources()["a2"], tmp_path / "a2")

    assert a7_path == SOURCE_PATHS["a7_csv"]
    assert a7_descriptor.resolve().is_relative_to(tmp_path.resolve())
    RunDescriptor.model_validate(yaml.safe_load(a7_descriptor.read_text(encoding="utf-8")))
    assert (a2_path, a2_descriptor) == (SOURCE_PATHS["a2"], COMMITTED_DESCRIPTORS["a2"])


@pytest.mark.parametrize("held_out", HELD_OUT)
def test_held_out_entries_are_their_fixture_directories(held_out: str) -> None:
    entry = sources()[held_out]

    assert entry.path == FIXTURES / held_out
    assert Path(entry.path).exists()


# --- recording --------------------------------------------------------------------------


def test_recording_runs_once_with_no_hint_and_stops_at_review(recorded: dict[str, Any]) -> None:
    (kwargs,) = recorded["seen"]
    rec = recorded["rec"]

    assert kwargs.get("hint") is None
    assert not kwargs.get("enrich", False)
    assert rec.outcome.node == "review"
    assert recorded["client"].invoke.call_count >= 1


def test_recording_copies_every_call_record_of_the_run(recorded: dict[str, Any]) -> None:
    rec = recorded["rec"]
    ledger = recorded["root"] / "work" / rec.run_id
    copied = call_files(rec.directory)

    assert rec.directory == recorded["root"] / "recordings" / "code_tables"
    assert copied and copied == call_files(ledger)
    for name in copied:
        original = (ledger / "calls" / name).read_bytes()
        assert (rec.directory / "calls" / name).read_bytes() == original


def test_recording_writes_only_the_calls_and_its_manifest(recorded: dict[str, Any]) -> None:
    directory = recorded["rec"].directory

    assert sorted(path.name for path in directory.iterdir()) == ["calls", "recording.json"]
    assert utils().MANIFEST_FILE == "recording.json"


def test_manifest_states_the_model_provider_versions_and_spend(
    recorded: dict[str, Any],
) -> None:
    rec = recorded["rec"]
    manifest = json.loads((rec.directory / "recording.json").read_text(encoding="utf-8"))
    row = Registry(recorded["root"] / "work" / REGISTRY_FILE).get(rec.run_id)
    prompts = [
        json.loads(path.read_text(encoding="utf-8"))["prompt"].splitlines()[0]
        for path in (rec.directory / "calls").glob("*.json")
    ]

    assert set(manifest) == MANIFEST_KEYS and manifest == rec.manifest
    assert (manifest["source"], manifest["provider"]) == ("code_tables", "openrouter")
    assert (manifest["model"], manifest["agreement"]) == (MODEL, True)
    assert manifest["prompt_versions"] == sorted({p.removeprefix(VERSION_PREFIX) for p in prompts})
    assert "structure v1" in manifest["prompt_versions"]
    assert manifest["calls"] == len(prompts) == row.model_calls
    assert manifest["tokens"] == row.tokens
    assert manifest["cost"] == pytest.approx(row.cost)


def test_re_recording_replaces_the_calls_directory(tmp_path: Path) -> None:
    source = code_table_source(tmp_path)
    recordings = tmp_path / "recordings"
    first = script().record_source(
        source, recordings=recordings, workdir=tmp_path / "w1",
        config=intake_config(max_calls=200), client=routed_client(),
    )
    stale = first.directory / "calls" / "stale.json"
    stale.write_text("{}", encoding="utf-8")
    client = routed_client()

    second = script().record_source(
        source, recordings=recordings, workdir=tmp_path / "w2",
        config=intake_config(max_calls=200, recordings_dir=first.directory), client=client,
    )

    assert client.invoke.call_count >= 1
    assert not stale.exists()
    assert call_files(second.directory) == call_files(tmp_path / "w2" / second.run_id)


def test_a_run_that_stops_short_of_review_is_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph(), "discover", fake_discover(node="value_ask"))
    source = utils().RecordedSource(name="a2", path=SOURCE_PATHS["a2"],
                                    descriptor=COMMITTED_DESCRIPTORS["a2"])

    with pytest.raises(utils().RecordingError, match="value_ask"):
        script().record_source(
            source, recordings=tmp_path / "recordings", workdir=tmp_path / "work",
            config=intake_config(), client=MagicMock(),
        )
    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize("held_out", HELD_OUT)
def test_held_out_source_is_passed_by_path_and_never_opened(
    held_out: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_discover()
    monkeypatch.setattr(graph(), "discover", fake)
    guard_held_out_reads(monkeypatch)

    rec = script().record_source(
        sources()[held_out], recordings=tmp_path / "recordings", workdir=tmp_path / "work",
        config=intake_config(), client=MagicMock(),
    )

    (seen,) = fake.seen
    assert seen["source"] == FIXTURES / held_out
    assert seen.get("hint") is None
    assert len(call_files(rec.directory)) == 1


# --- replay -----------------------------------------------------------------------------


def test_recording_replays_with_zero_provider_calls(recorded: dict[str, Any]) -> None:
    root, rec = recorded["root"], recorded["rec"]
    client = routed_client()

    outcome = script().verify_replay(
        recorded["source"], recordings=root / "recordings", workdir=root / "replay",
        config=intake_config(max_calls=200), client=client,
    )
    row = Registry(root / "replay" / REGISTRY_FILE).get(outcome.run_id)

    assert client.invoke.call_count == 0
    assert outcome.run_id != rec.run_id and outcome.node == "review"
    for part in REVIEW_PARTS:
        assert outcome.payload[part] == rec.outcome.payload[part], part
    assert (row.model_calls, row.tokens, row.cost) == (0, 0, 0.0)
    assert call_files(root / "replay" / outcome.run_id) == call_files(rec.directory)


def test_replay_with_a_missing_record_is_refused(tmp_path: Path) -> None:
    source = code_table_source(tmp_path)
    rec = script().record_source(
        source, recordings=tmp_path / "recordings", workdir=tmp_path / "work",
        config=intake_config(max_calls=200), client=routed_client(),
    )
    next((rec.directory / "calls").glob("*.json")).unlink()

    with pytest.raises(utils().ReplayMismatchError, match="code_tables"):
        script().verify_replay(
            source, recordings=tmp_path / "recordings", workdir=tmp_path / "replay",
            config=intake_config(max_calls=200), client=routed_client(),
        )


# --- the provider guard -----------------------------------------------------------------


def test_check_provider_requires_an_explicit_provider() -> None:
    with pytest.raises(IntakeProviderNotSetError):
        utils().check_provider(intake_config(provider=None))
    utils().check_provider(intake_config(provider="openrouter"))
    assert utils().RECORDING_PROVIDER == "openrouter"


def test_check_provider_refuses_any_other_provider_by_name() -> None:
    with pytest.raises(utils().ProviderRefusedError, match="databricks"):
        utils().check_provider(intake_config(provider="databricks"))
    assert issubclass(utils().ProviderRefusedError, ValueError)


def test_record_source_refuses_another_provider_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph(), "discover", forbid("discover"))
    client = MagicMock()

    with pytest.raises(utils().ProviderRefusedError, match="databricks"):
        script().record_source(
            sources()["a2"], recordings=tmp_path / "recordings", workdir=tmp_path / "work",
            config=intake_config(provider="databricks"), client=client,
        )
    assert client.invoke.call_count == 0
    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize(("provider", "named"), [(None, "INTAKE_MODEL_PROVIDER"),
                                                 ("databricks", "databricks")])
def test_main_refuses_without_openrouter_and_leaves_env_alone(
    provider: str | None, named: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = model_env(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(script(), "record_source", forbid("record_source"))

    code = script().main(["--source", "a2", "--recordings", str(tmp_path / "rec")])

    assert code != 0
    assert named in capsys.readouterr().err
    assert env.read_text(encoding="utf-8") == ENV_TEXT


def test_main_records_then_verifies_each_named_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = model_env(monkeypatch, tmp_path, "openrouter")
    steps: list[tuple[str, str, Any]] = []

    def _record(source: Any, **kwargs: Any) -> Any:
        steps.append(("record", source.name, kwargs))
        return MagicMock(run_id="r")

    def _verify(source: Any, **kwargs: Any) -> Any:
        steps.append(("verify", source.name, kwargs))
        return MagicMock(node="review")

    monkeypatch.setattr(script(), "record_source", _record)
    monkeypatch.setattr(script(), "verify_replay", _verify)
    recordings = tmp_path / "rec"

    code = script().main(["--source", "a2", "--source", "a3", "--recordings", str(recordings),
                          "--workdir", str(tmp_path / "work")])

    assert code == 0
    assert [(step, name) for step, name, _ in steps] == [
        ("record", "a2"), ("verify", "a2"), ("record", "a3"), ("verify", "a3"),
    ]
    for _, _, kwargs in steps:
        assert kwargs["recordings"] == recordings
        assert kwargs["config"].provider == "openrouter"
    assert env.read_text(encoding="utf-8") == ENV_TEXT
