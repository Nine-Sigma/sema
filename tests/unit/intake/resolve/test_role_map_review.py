"""What the Roles stage hands Review (S5 decisions 2, 4, 18c; task 2.7 fix round).

- An ``unverifiable`` status (decision 2): a ``status`` binding under ``metadata_only``
  stays bound and its entry in ``RolesResult.ambiguities`` carries the mark and the reason,
  settled, on every path that builds the result; the reader prints it with the mark.
- A contest reaches kin (decision 18c): when the model contests a lexicon claim on one
  relation, the same role on a same-signature kin relation stays open with both columns,
  never ``source: lexicon``, and ``RolesResult.contested`` names both subjects.
- The disagreement count reaches Review (decision 4): with agreement on, two role-map
  answers that differ on one role record the relation's count on the roles ambiguity
  artifact, and the reader prints it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from sema.cli_intake_utils import review_lines
from sema.intake.resolve.output_models import ROLE_PREFIX
from sema.intake.resolve.output_models_utils import NONE
from sema.intake.verify.roles import UNVERIFIABLE
from tests.unit.intake.graph_env import (
    MAX_CALLS, ROLE_MAP, STRUCTURE_BATCH, GraphRun, choose_flat_lists, graph, routed_client,
)
from tests.unit.intake.model_calls_env import intake_config
from tests.unit.intake.review_reader_env import RUN_ID, seeded_root, write_run
from tests.unit.intake.role_map_env import (
    SQL, STATUSED, chosen, code_list, mapped, scenario, subject_entry, table,
)

pytestmark = pytest.mark.unit

STATUS_COLUMN = "item_status"
SPEC_BODY = {"concept_source": {"spec_version": 2}}
GLIMS = ("rm.glim_first", "rm.glim_second")
GLIM_COLUMNS = {
    "item_code": "'M' || lpad(CAST(i AS VARCHAR), 2, '0')",
    "zorb": "'Z' || lpad(CAST(i AS VARCHAR), 2, '0')",
    "item_label": "'Term entry ' || i",
}
FLIM = "flim_items"
FLIM_ROWS = 20
AGREED_ROLES = {"code": "item_code", "display": "item_label", "section": "code_lists"}


def _status_entries(result: Any) -> list[Any]:
    return [
        a for a in result.ambiguities
        if a.subject.startswith(STATUSED) and a.subject.endswith(".status")
    ]


def test_an_unverifiable_status_is_listed_settled_with_the_mark(tmp_path: Path) -> None:
    scene = scenario(SQL[STATUSED], [code_list(STATUSED)], tmp_path)

    resolved, _ = scene.resolve(mapped({"status": STATUS_COLUMN}))

    for result in (scene.roles, resolved):
        (entry,) = _status_entries(result)
        assert UNVERIFIABLE in entry.hints and entry.reason, entry
        assert entry.choice == STATUS_COLUMN and entry.source is not None, entry
    root = seeded_root(tmp_path)
    listed = [a.model_dump(mode="json") for a in resolved.ambiguities]
    write_run(root, [], listed, spec=SPEC_BODY)
    lines = review_lines(root, RUN_ID, settled=True)
    (subject,) = {a.subject for a in _status_entries(resolved)}
    (marked,) = [line for line in lines if line.split(" | ")[0].endswith(f" {subject}")]
    assert UNVERIFIABLE in marked, marked
    scene.con.close()


def test_a_contest_on_one_relation_keeps_the_kin_role_open(tmp_path: Path) -> None:
    statements = [table(name, **GLIM_COLUMNS) for name in GLIMS]
    scene = scenario(statements, [code_list(name) for name in GLIMS], tmp_path)
    client = mapped({"code": "zorb", "display": "item_label"})

    result, _ = scene.resolve(client)

    assert client.invoke.call_count == 1
    for name in GLIMS:
        entry = subject_entry(result, f"{name}.code")
        assert entry.choice is None and entry.source is None, entry
        assert {"item_code", "zorb"} <= set(entry.members), entry
        assert chosen(result, name, "code").source != "lexicon"
    assert {f"{name}.code" for name in GLIMS} <= set(result.contested), result.contested
    scene.con.close()


def _flim_source(root: Path) -> Path:
    directory = root / "inputs"
    directory.mkdir(parents=True)
    rows = [
        f"T{i:02d},Term entry {i},Vell says a long sentence about item number {i}"
        for i in range(1, FLIM_ROWS + 1)
    ]
    text = "\n".join(["item_code,item_label,vell", *rows]) + "\n"
    (directory / f"{FLIM}.csv").write_text(text)
    return directory


def _role_answer(extra: dict[str, str]) -> Any:
    named = {**AGREED_ROLES, **extra}

    def _answer(model: type[BaseModel]) -> BaseModel:
        fields = {
            name: named.get(name.removeprefix(ROLE_PREFIX), NONE)
            for name in model.model_fields
            if name.startswith(ROLE_PREFIX)
        }
        count = len(fields)
        return model(**fields, confidences=[0.8] * count, reasons=["facts fit"] * count)

    return _answer


def test_review_records_and_prints_the_disagreement_count(tmp_path: Path) -> None:
    client = routed_client({
        STRUCTURE_BATCH: [choose_flat_lists],
        ROLE_MAP: [_role_answer({"display_fallback": "vell"}), _role_answer({})],
    })
    descriptor = tmp_path / "flim.run.yaml"
    descriptor.write_text("id: flim\nidentity_system: flim\nrelease: '2026-10-23'\n")
    root = tmp_path / "runs"
    outcome = graph().discover(
        _flim_source(tmp_path), descriptor=descriptor, client=client, root=root,
        config=intake_config(max_calls=MAX_CALLS, agreement=True),
    )
    run = GraphRun(root, outcome.run_id, client, outcome)
    assert outcome.node == "review", outcome.node

    counts = run.artifact("roles_ambiguities")["disagreements"]

    ((relation, count),) = counts.items()
    assert relation.endswith(FLIM) and count == 1, counts
    lines = review_lines(root, run.run_id)
    assert [line for line in lines if relation in line and "disagreements: 1" in line], lines
