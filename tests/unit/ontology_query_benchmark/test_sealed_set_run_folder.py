"""Done-when 3 (N5): one run, one folder.

`python -m showcase.ontology_query_benchmark.sealed_set` takes `--run-name <name>`: the run
folder is `<set>/sealed/runs/<name>/`; with no name it is the schedule name
(`<arms>x<repetitions>`), as today. A run whose folder already holds a file is refused before
any write, with a non-zero exit and a message that names the folder and holds no question
text. The harness loads before the first file is written, so a run that cannot start leaves no
file. `--dry-run` writes nothing. Runs here use arm A, the toy harness and the scripted model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tests.unit.ontology_query_benchmark.sealed_set_env import (
    SEALED_SET_FIXTURE,
    ScriptedModel,
    captured_logs,
    materialize,
    question_texts,
    sealed_set_module,
)

pytestmark = pytest.mark.unit


@dataclass
class Entry:
    profile: Path
    runs: Path
    report: Path
    models: list[ScriptedModel] = field(default_factory=list)


def files_under(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


@pytest.fixture
def entry(frozen: Path, harness: SimpleNamespace, tmp_path: Path,
          monkeypatch: pytest.MonkeyPatch) -> Entry:  # fmt: skip
    root = tmp_path / "sealed_set"
    made = Entry(materialize(root, frozen), root / "neutral" / "sealed" / "runs",
                 tmp_path / "report.md")  # fmt: skip
    texts = question_texts(SEALED_SET_FIXTURE / "neutral")

    def client(*_: Any, **__: Any) -> ScriptedModel:
        made.models.append(ScriptedModel(harness, texts, arm="A"))
        return made.models[-1]

    module = sealed_set_module()
    monkeypatch.setattr(module, "require_clean_process", lambda *_, **__: None)
    monkeypatch.setattr(module, "model_client", client)
    return made


def run_main(made: Entry, *extra: str) -> tuple[int, str]:
    """The exit code, and the text the run showed: stdout, stderr and any error message."""
    argv = ["--profile", str(made.profile), "--arms", "A", "--repetitions", "1",
            "--report", str(made.report), *extra]  # fmt: skip
    try:
        return sealed_set_module().main(argv), ""
    except SystemExit as exc:
        return (exc.code if isinstance(exc.code, int) else 1), str(exc.code or "")
    except Exception as exc:  # noqa: BLE001 - an uncaught error is a non-zero exit too
        return 1, str(exc)


def calls_after_first(made: Entry) -> int:
    return sum(len(model.calls) for model in made.models[1:])


def test_a_second_run_with_the_same_name_is_refused_and_leaves_the_first_run_unchanged(
    entry: Entry,
) -> None:
    first, _ = run_main(entry, "--run-name", "first-pass")
    assert first == 0
    assert (entry.runs / "first-pass" / "receipts.jsonl").is_file()
    before = files_under(entry.runs)

    second, _ = run_main(entry, "--run-name", "first-pass")

    assert second != 0
    assert calls_after_first(entry) == 0
    assert files_under(entry.runs) == before


def test_a_second_run_with_no_name_is_refused_too(entry: Entry) -> None:
    first, _ = run_main(entry)
    assert first == 0
    before = files_under(entry.runs)

    second, _ = run_main(entry)

    assert second != 0
    assert calls_after_first(entry) == 0
    assert files_under(entry.runs) == before


def test_the_refusal_names_the_folder_and_holds_no_question_text(
    entry: Entry, capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_main(entry, "--run-name", "named-twice")[0] == 0
    capsys.readouterr()

    with captured_logs() as logs:
        code, message = run_main(entry, "--run-name", "named-twice")

    out, err = capsys.readouterr()
    shown = "\n".join((out, err, message, *logs))
    texts = question_texts(SEALED_SET_FIXTURE / "neutral").values()
    assert code != 0
    assert "named-twice" in shown, shown
    assert [t for t in texts if t in shown] == []


def test_a_run_whose_harness_cannot_load_leaves_no_file(entry: Entry, tmp_path: Path) -> None:
    raw = yaml.safe_load(entry.profile.read_text())
    raw["harness"]["frozen"] = str(tmp_path / "no_such_closure")
    entry.profile.write_text(yaml.safe_dump(raw))

    with pytest.raises((ImportError, RuntimeError)):
        sealed_set_module().run_sealed_set(entry.profile, ("A",), 1, model=None, s_provider=None,
                                           report=entry.report)  # fmt: skip

    assert not entry.runs.exists() or files_under(entry.runs) == {}


def test_run_name_sets_the_run_folder(entry: Entry) -> None:
    code, _ = run_main(entry, "--run-name", "chosen-name")

    assert code == 0
    assert sorted(p.name for p in entry.runs.iterdir()) == ["chosen-name"]
    assert (entry.runs / "chosen-name" / "receipts.jsonl").is_file()


def test_with_no_name_the_run_folder_is_the_schedule_name(entry: Entry) -> None:
    code, _ = run_main(entry)

    assert code == 0
    assert sorted(p.name for p in entry.runs.iterdir()) == ["Ax1"]
    assert (entry.runs / "Ax1" / "receipts.jsonl").is_file()


def test_a_dry_run_with_a_run_name_writes_nothing(
    entry: Entry, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    before = files_under(tmp_path)

    code, _ = run_main(entry, "--run-name", "dry-pass", "--dry-run")

    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and "spend 0.00" in lines
    assert files_under(tmp_path) == before
    assert entry.models == []
