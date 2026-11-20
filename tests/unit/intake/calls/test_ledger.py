"""Run ledger: one directory per run with events, SQL, artifacts, and replayable calls."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import BaseModel

from sema.intake.land.landing import LandedRelation, LandResult, Provenance

pytestmark = pytest.mark.unit

FORBIDDEN_IMPORT_ROOTS = ("langgraph", "langchain")
MODEL = "openrouter/gpt-5.4"
PROMPT = "Pick the template for relation r1."
SCHEMA_NAME = "StructureVerdict"


class _Verdict(BaseModel):
    stage: str
    passed: bool
    failures: list[str]


def _run(root: Path, run_id: str | None = None) -> Any:
    from sema.intake.calls.ledger import Ledger

    return Ledger(root).run(run_id or str(uuid.uuid4()))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _land_result() -> LandResult:
    provenance = Provenance(
        source_uri="file:///x/dict.xlsx",
        loader="tabular",
        loader_version="1",
        landed_at="2026-10-20T00:00:00+00:00",
        policy_class="metadata_only",
        locator_kind="row",
    )
    relation = LandedRelation(
        schema="land",
        name="dict__sheet1",
        columns=(("code", "VARCHAR"), ("label", "VARCHAR")),
        row_count=3,
        provenance=provenance,
        header_row=1,
    )
    return LandResult(
        medium="tabular",
        source_identity=(("id", "dict"),),
        relations=(relation,),
        policy_class="metadata_only",
    )


def _expected_hash(model: str, prompt: str, schema_name: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}\n{schema_name}".encode()).hexdigest()


def test_run_gets_its_own_directory_under_the_root(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    run = _run(tmp_path, run_id)
    run.event("land", "started", {"relations": 0})

    assert (tmp_path / run_id).is_dir()
    assert (tmp_path / run_id / "events.jsonl").is_file()


def test_events_append_as_json_lines_with_stage_kind_timestamp_payload(
    tmp_path: Path,
) -> None:
    run_id = str(uuid.uuid4())
    run = _run(tmp_path, run_id)
    run.event("land", "started", {"source": "dict"})
    run.event("structure", "verdict", {"passed": True})

    events = _lines(tmp_path / run_id / "events.jsonl")

    assert [(e["stage"], e["kind"]) for e in events] == [
        ("land", "started"),
        ("structure", "verdict"),
    ]
    assert events[0]["payload"] == {"source": "dict"}
    assert events[1]["payload"] == {"passed": True}
    assert all(isinstance(e["timestamp"], str) and e["timestamp"] for e in events)


def test_event_log_is_append_only_across_ledger_instances(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    _run(tmp_path, run_id).event("land", "started", {})
    _run(tmp_path, run_id).event("land", "finished", {})

    events = _lines(tmp_path / run_id / "events.jsonl")

    assert [e["kind"] for e in events] == ["started", "finished"]


def test_sql_appends_with_value_bearing_and_schema_relation_flags(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    run = _run(tmp_path, run_id)
    run.sql("structure", "staging", 'SELECT count(*) FROM "land"."r1"', False, False)
    run.sql("roles", "warehouse", "SELECT `code` FROM `c`.`s`.`t`", True, False)
    run.sql("structure", "staging", 'SELECT "name" FROM "land"."r1__cols"', True, True)

    rows = _lines(tmp_path / run_id / "sql.jsonl")

    assert [
        (r["stage"], r["provider"], r["value_bearing"], r["schema_relation"]) for r in rows
    ] == [
        ("structure", "staging", False, False),
        ("roles", "warehouse", True, False),
        ("structure", "staging", True, True),
    ]
    assert rows[1]["statement"] == "SELECT `code` FROM `c`.`s`.`t`"


def _artifact_file(run_dir: Path, name: str) -> Path:
    matches = sorted((run_dir / "artifacts").glob(f"{name}.*"))
    assert len(matches) == 1, matches
    assert matches[0].suffix in {".json", ".yaml"}
    return matches[0]


def test_pydantic_artifact_is_written_and_reads_back(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    verdict = _Verdict(stage="structure", passed=False, failures=["no template"])
    _run(tmp_path, run_id).artifact("structure_verdict", verdict)

    path = _artifact_file(tmp_path / run_id, "structure_verdict")
    loaded = yaml.safe_load(path.read_text())

    assert _Verdict.model_validate(loaded) == verdict


def test_dataclass_artifact_is_written_and_reads_back(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    land = _land_result()
    _run(tmp_path, run_id).artifact("land_result", land)

    path = _artifact_file(tmp_path / run_id, "land_result")
    loaded = yaml.safe_load(path.read_text())

    assert loaded == json.loads(json.dumps(dataclasses.asdict(land)))


def test_record_call_writes_under_prompt_hash(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    output = {"template": "concept_list", "relation": "r1"}
    _run(tmp_path, run_id).record_call(MODEL, PROMPT, SCHEMA_NAME, output, 321)

    path = tmp_path / run_id / "calls" / f"{_expected_hash(MODEL, PROMPT, SCHEMA_NAME)}.json"
    recorded = json.loads(path.read_text())

    assert output in recorded.values()
    assert 321 in recorded.values()


def test_replay_hit_returns_output_without_calling_the_model(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    run = _run(tmp_path, run_id)
    output = {"template": "concept_list"}
    run.record_call(MODEL, PROMPT, SCHEMA_NAME, output, 40)
    model = MagicMock()

    replayed = run.replay(MODEL, PROMPT, SCHEMA_NAME)
    answer = replayed if replayed is not None else model.invoke(PROMPT)

    assert answer == output
    model.invoke.assert_not_called()
    replays = [
        e for e in _lines(tmp_path / run_id / "events.jsonl") if e["kind"] == "replay"
    ]
    assert len(replays) == 1
    assert _expected_hash(MODEL, PROMPT, SCHEMA_NAME) in json.dumps(replays[0]["payload"])


def test_replay_miss_when_one_prompt_character_differs(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    run = _run(tmp_path, run_id)
    run.record_call(MODEL, PROMPT, SCHEMA_NAME, {"template": "concept_list"}, 40)

    assert run.replay(MODEL, PROMPT[:-1] + "!", SCHEMA_NAME) is None
    events_path = tmp_path / run_id / "events.jsonl"
    events = _lines(events_path) if events_path.exists() else []
    assert not [e for e in events if e["kind"] == "replay"]


@pytest.mark.parametrize(
    ("model", "schema_name"),
    [("other/model", SCHEMA_NAME), (MODEL, "RolesVerdict")],
)
def test_replay_key_covers_model_and_schema_name(
    tmp_path: Path, model: str, schema_name: str
) -> None:
    run = _run(tmp_path)
    run.record_call(MODEL, PROMPT, SCHEMA_NAME, {"template": "concept_list"}, 40)

    assert run.replay(model, PROMPT, schema_name) is None


def test_provenance_event_round_trips(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    prompt_hash = _expected_hash(MODEL, PROMPT, SCHEMA_NAME)
    fields = {
        "model": MODEL,
        "prompt_hash": prompt_hash,
        "input_locator": "land.r1!7",
        "output_locator": "spec.roles.display",
    }
    _run(tmp_path, run_id).provenance("roles", **fields)

    events = [
        e for e in _lines(tmp_path / run_id / "events.jsonl") if e["kind"] == "provenance"
    ]

    assert len(events) == 1
    assert events[0]["stage"] == "roles"
    assert events[0]["payload"] == fields


def _imported_modules(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_ledger_imports_no_orchestration_framework() -> None:
    roots = {name.split(".")[0] for name in _imported_modules("sema.intake.calls.ledger")}

    assert not roots & set(FORBIDDEN_IMPORT_ROOTS)


def test_ledger_does_not_reference_the_registry() -> None:
    assert "sema.intake.calls.registry" not in _imported_modules("sema.intake.calls.ledger")
