"""`sema target discover|approve` write the assembled spec to the spec output folder.

Pinned surface (design decision 10, task 6.1), in ``sema.cli_intake``:

- ``discover`` and ``approve`` take ``--out DIR``; the default is ``specs`` beside the ledger
  root (``.sema/specs`` for the default root ``.sema/intake``).
- A command whose run holds an assembled spec writes it as YAML to
  ``<out>/<descriptor id>/spec.yaml``: discover when it stops at Review (status
  ``waiting_review``), approve when it imports (status ``imported``). The file opens with a
  header comment naming the run id, that status, and the run's source; its body loads with
  ``load_concept_source_spec`` to the spec the ledger's ``spec`` artifact holds.
- The printed JSON carries ``spec_path``: the written file, or null when the run holds no
  assembled spec, in which case nothing is written under the folder.
- A later run of the same descriptor replaces the file.

A2 at ``--tier full`` stops at Review with an assembled spec; ACRIS code tables stop at
Review with open structure entries and no spec. Zero provider calls leave the machine: the
model is the routed mock and ``EMBEDDING_PROVIDER`` is empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.unit.cli_intake_env import (
    SPEC_PATH_KEY,
    Cli,
    decide,
    default_specs_dir,
    files_under,
    header_of,
    intake_cli,
    open_listed,
    printed_path,
    same_spec,
    spec_file,
)
from tests.unit.intake.graph_env import A2_DESCRIPTOR, ACRIS_DESCRIPTOR, GraphRun, code_tables
from tests.unit.intake.structure_env import A2_WORKBOOK

pytestmark = pytest.mark.unit

WAITING, IMPORTED = "waiting_review", "imported"
ASSEMBLING = ("--tier", "full")
NO_CHOICE = "none"
ESCAPING_ID = "../escape"


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")
    return intake_cli(tmp_path / ".sema" / "intake", monkeypatch)


def _discover_a2(cli: Cli, *options: str | Path) -> tuple[GraphRun, dict[str, object]]:
    printed = cli.ok("discover", A2_WORKBOOK, "--descriptor", A2_DESCRIPTOR, *ASSEMBLING, *options)
    assert (printed["stopped_at"], printed["status"]) == ("review", WAITING)
    return cli.graph_run(str(printed["run_id"])), printed


def _assert_written(path: Path, run: GraphRun, status: str, stale: str) -> None:
    header = header_of(path)
    assert run.run_id in header
    assert status in header and stale not in header
    assert run.row().source in header
    assert same_spec(path, run.artifact("spec"))


def test_spec_output_discover_writes_the_assembled_spec_with_its_header(cli: Cli) -> None:
    run, printed = _discover_a2(cli)

    written = spec_file(default_specs_dir(cli), A2_DESCRIPTOR)
    assert written.is_file()
    _assert_written(written, run, WAITING, IMPORTED)
    assert printed_path(printed) == written.resolve()
    assert run.has_artifact("spec")


def test_spec_output_plain_approve_rewrites_the_file_as_imported(cli: Cli) -> None:
    run, _ = _discover_a2(cli)

    printed = cli.approve(run)

    assert printed["status"] == "approved"
    written = spec_file(default_specs_dir(cli), A2_DESCRIPTOR)
    _assert_written(written, run, IMPORTED, WAITING)
    assert printed_path(printed) == written.resolve()
    assert files_under(default_specs_dir(cli)) == [written]


def test_spec_output_approve_after_decisions_rewrites_the_file_as_imported(cli: Cli) -> None:
    run, _ = _discover_a2(cli)
    entries = open_listed(run, "roles")
    assert entries and all(NO_CHOICE in entry.members for entry in entries)
    written = spec_file(default_specs_dir(cli), A2_DESCRIPTOR)

    decided = decide(cli, run, [("roles", e.subject, NO_CHOICE) for e in entries], "d.yaml")

    assert decided["stopped_at"] == "review"
    _assert_written(written, run, WAITING, IMPORTED)
    assert printed_path(decided) == written.resolve()
    printed = cli.approve(run)
    _assert_written(written, run, IMPORTED, WAITING)
    assert printed_path(printed) == written.resolve()


def test_spec_output_out_overrides_the_default_folder(cli: Cli) -> None:
    out = cli.root.parent.parent / "elsewhere"

    run, discovered = _discover_a2(cli, "--out", out)
    approved = cli.approve(run, "--out", out)

    written = spec_file(out, A2_DESCRIPTOR)
    _assert_written(written, run, IMPORTED, WAITING)
    assert printed_path(discovered) == printed_path(approved) == written.resolve()
    assert files_under(out) == [written]
    assert files_under(default_specs_dir(cli)) == []


def test_spec_output_a_later_run_of_the_descriptor_replaces_the_file(cli: Cli) -> None:
    first, _ = _discover_a2(cli)

    second, printed = _discover_a2(cli)

    assert second.run_id != first.run_id
    written = spec_file(default_specs_dir(cli), A2_DESCRIPTOR)
    assert files_under(default_specs_dir(cli)) == [written]
    assert first.run_id not in header_of(written)
    _assert_written(written, second, WAITING, IMPORTED)
    assert printed_path(printed) == written.resolve()


def test_spec_output_a_run_without_a_spec_writes_nothing_and_says_so(cli: Cli) -> None:
    printed = cli.ok("discover", code_tables(cli.root.parent), "--descriptor", ACRIS_DESCRIPTOR)

    run = cli.graph_run(printed["run_id"])
    assert (printed["stopped_at"], printed["status"]) == ("review", WAITING)
    assert open_listed(run, "structure")
    assert not run.has_artifact("spec")
    assert SPEC_PATH_KEY in printed and printed[SPEC_PATH_KEY] is None
    assert files_under(default_specs_dir(cli)) == []


def test_spec_output_refuses_an_edited_spec_id_that_leaves_the_folder(cli: Cli) -> None:
    run, _ = _discover_a2(cli)
    edited = cli.root.parent / "edited.spec.yaml"
    escaping = {**run.artifact("spec"), "id": ESCAPING_ID}
    edited.write_text(yaml.safe_dump({"concept_source": escaping}, sort_keys=False))
    out = default_specs_dir(cli)
    before = files_under(out)

    result = cli.run("approve", run.run_id, "--approver", "reviewer", "--spec", edited)

    assert result.exit_code != 0
    assert ESCAPING_ID in result.output
    assert files_under(out) == before
    assert not (out / ESCAPING_ID).exists()
