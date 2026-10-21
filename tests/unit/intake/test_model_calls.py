"""Intake model-call path: one configured client, replay by prompt hash, recordings."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sema.intake.ledger_utils import prompt_hash
from sema.llm_client import LLMClient
from sema.models.config import LLMConfig
from tests.unit.intake.model_calls_env import (
    MODEL,
    PROMPT,
    PROMPT_V2,
    STAGE,
    Choice_v1,
    Choice_v2,
    choice,
    fake_client,
    files_under,
    intake_config,
    record_in,
    wired_run,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
INTAKE_ROOT = REPO_ROOT / "src" / "sema" / "intake"
CALL_PATH_FILE = INTAKE_ROOT / "model_calls.py"
PROVIDER_IMPORT_ROOTS = frozenset(
    {
        "openai",
        "anthropic",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_community",
        "langchain_databricks",
        "databricks_langchain",
    }
)
PROVIDER_NAMES = frozenset(
    {"ChatOpenAI", "ChatAnthropic", "ChatDatabricks", "ChatOpenRouter", "init_chat_model"}
)
CALL_PATH_ONLY_NAMES = frozenset({"_get_llm", "LLMClient"})
KEY_V1 = prompt_hash(MODEL, PROMPT, "Choice_v1")


def _build(tmp_path: Path, config: object) -> object:
    from sema.intake.ledger import Ledger
    from sema.intake.model_calls import build_model_calls
    from sema.intake.registry import Registry

    run_id = "run-build"
    registry = Registry(tmp_path / "registry.sqlite")
    registry.register(run_id, source="source-under-test", stage=STAGE)
    ledger = Ledger(tmp_path / "runs").run(run_id)
    return build_model_calls(config, ledger, registry, run_id)


@pytest.mark.parametrize(("provider", "model"), [("openrouter", MODEL), ("databricks", "ep-1")])
def test_provider_is_chosen_by_configuration_only(
    tmp_path: Path, provider: str, model: str
) -> None:
    from sema.intake.model_calls import ModelCalls

    provider_llm = MagicMock(spec=["invoke"])
    with patch("sema.intake.model_calls._get_llm", return_value=provider_llm) as factory:
        calls = _build(tmp_path, intake_config(provider=provider, model=model))

    factory.assert_called_once()
    (llm_config,) = factory.call_args.args
    assert isinstance(llm_config, LLMConfig)
    assert (llm_config.provider, llm_config.model) == (provider, model)
    assert type(calls) is ModelCalls
    assert isinstance(calls.client, LLMClient)


def test_a_provider_not_set_for_intake_fails_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sema.intake.model_calls import IntakeProviderNotSetError
    from sema.models.config import IntakeModelConfig

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("INTAKE_MODEL_PROVIDER", raising=False)
    config = IntakeModelConfig(model=MODEL, max_calls=3, max_cost=1.0)

    assert config.provider is None
    with (
        patch("sema.intake.model_calls._get_llm") as factory,
        pytest.raises(IntakeProviderNotSetError),
    ):
        _build(tmp_path, config)
    factory.assert_not_called()


def _referenced_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    roots: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return roots, names


def test_no_intake_module_constructs_a_provider_client() -> None:
    assert CALL_PATH_FILE.is_file()
    offenders: list[str] = []
    for path in sorted(INTAKE_ROOT.rglob("*.py")):
        roots, names = _referenced_names(ast.parse(path.read_text(encoding="utf-8")))
        banned = (roots & PROVIDER_IMPORT_ROOTS) | (names & PROVIDER_NAMES)
        if path != CALL_PATH_FILE:
            banned |= names & CALL_PATH_ONLY_NAMES
        offenders.extend(f"{path.relative_to(REPO_ROOT)}: {name}" for name in sorted(banned))

    assert offenders == []


def test_a_miss_calls_the_provider_once_and_records_the_call(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client([choice("b")]))

    result = run.calls.call(STAGE, PROMPT, Choice_v1)

    assert result == choice("b")
    run.client.invoke.assert_called_once()
    assert run.client.invoke.call_args.args[:2] == (PROMPT, Choice_v1)
    record = run.call_record(KEY_V1)
    assert (record["model"], record["prompt"], record["schema_name"]) == (
        MODEL,
        PROMPT,
        "Choice_v1",
    )
    assert record["output"] == choice("b").model_dump(mode="json")
    assert record["tokens"] == 1500


def test_a_ledger_hit_replays_without_a_provider_call(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client())
    key = record_in(run.ledger.directory, PROMPT, choice("b"))

    result = run.calls.call(STAGE, PROMPT, Choice_v1)

    assert isinstance(result, Choice_v1)
    assert result == choice("b")
    run.client.invoke.assert_not_called()
    replays = run.events("replay")
    assert [(e["stage"], e["payload"]["prompt_hash"]) for e in replays] == [(STAGE, key)]


def test_re_execution_of_a_stage_replays_its_calls(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client([choice("a")]))

    first = run.calls.call(STAGE, PROMPT, Choice_v1)
    second = run.calls.call(STAGE, PROMPT, Choice_v1)

    assert first == second
    run.client.invoke.assert_called_once()
    assert run.row().model_calls == 1
    assert len(run.events("replay")) == 1


def test_a_recording_hit_is_copied_into_the_run_ledger(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    key = record_in(recordings, PROMPT, choice("b"), tokens=1234)
    recorded_files = files_under(recordings)
    config = intake_config(recordings_dir=recordings)
    run = wired_run(tmp_path, fake_client(), config, run_id="fresh-run")

    result = run.calls.call(STAGE, PROMPT, Choice_v1)

    assert result == choice("b")
    run.client.invoke.assert_not_called()
    copied = run.call_record(key)
    assert copied["output"] == choice("b").model_dump(mode="json")
    assert copied["tokens"] == 1234
    assert [e["payload"]["prompt_hash"] for e in run.events("replay")] == [key]
    row = run.row()
    assert (row.model_calls, row.tokens, row.cost) == (0, 0, 0.0)
    assert files_under(recordings) == recorded_files


def test_a_recording_miss_falls_through_to_the_provider(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    record_in(recordings, PROMPT_V2, choice("b"))
    run = wired_run(tmp_path, fake_client([choice("a")]), intake_config(recordings_dir=recordings))

    assert run.calls.call(STAGE, PROMPT, Choice_v1) == choice("a")
    run.client.invoke.assert_called_once()


def test_a_prompt_version_line_change_is_a_miss(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client([choice("a"), choice("b")]))

    run.calls.call(STAGE, PROMPT, Choice_v1)
    result = run.calls.call(STAGE, PROMPT_V2, Choice_v1)

    assert result == choice("b")
    assert run.client.invoke.call_count == 2
    key_v2 = prompt_hash(MODEL, PROMPT_V2, "Choice_v1")
    assert run.call_files() == sorted([f"{KEY_V1}.json", f"{key_v2}.json"])
    assert run.events("replay") == []


def test_an_output_model_version_change_is_a_miss(tmp_path: Path) -> None:
    second = Choice_v2(choice="b", confidence=0.7, reason="named in the header")
    run = wired_run(tmp_path, fake_client([choice("a"), second]))

    run.calls.call(STAGE, PROMPT, Choice_v1)
    result = run.calls.call(STAGE, PROMPT, Choice_v2)

    assert result == second
    assert run.client.invoke.call_count == 2
    assert len(run.call_files()) == 2


def test_the_same_prompt_under_two_run_ids_hashes_equal(tmp_path: Path) -> None:
    first = wired_run(tmp_path, fake_client([choice("a")]), run_id="run-one")
    second = wired_run(tmp_path, fake_client([choice("a")]), run_id="run-two")

    first.calls.call(STAGE, PROMPT, Choice_v1)
    second.calls.call(STAGE, PROMPT, Choice_v1)

    assert first.call_files() == second.call_files() == [f"{KEY_V1}.json"]
    assert first.call_record(KEY_V1)["prompt"] == second.call_record(KEY_V1)["prompt"] == PROMPT


def test_the_call_record_names_the_client_path_that_served_it(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client([choice("a")], served_by="plain_invoke"))

    run.calls.call(STAGE, PROMPT, Choice_v1)

    (served,) = run.events("call")
    assert served["stage"] == STAGE
    assert served["payload"]["prompt_hash"] == KEY_V1
    assert served["payload"]["path"] == "plain_invoke"


def test_a_fallback_inside_the_client_is_one_call_served_by_the_plain_path(
    tmp_path: Path,
) -> None:
    provider = MagicMock()
    provider.with_structured_output.return_value.invoke.side_effect = ValueError("bad json")
    provider.invoke.return_value = MagicMock(content=choice("b").model_dump_json())
    client = LLMClient(provider, retry_max_attempts=1, use_structured_output="true")
    run = wired_run(tmp_path, fake_client())
    run.calls = type(run.calls)(client, run.ledger, run.registry, run.run_id, intake_config())

    assert run.calls.call(STAGE, PROMPT, Choice_v1) == choice("b")
    assert run.row().model_calls == 1
    assert [e["payload"]["path"] for e in run.events("call")] == ["plain_invoke"]
