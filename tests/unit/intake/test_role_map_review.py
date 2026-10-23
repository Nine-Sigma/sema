"""What the Roles stage hands Review (S5 decisions 2, 4, 18c; task 2.7 fix round).

- An ``unverifiable`` status (decision 2): a ``status`` binding under ``metadata_only``
  stays bound and its entry in ``RolesResult.ambiguities`` carries the mark and the reason,
  settled, on every path that builds the result; the reader prints it with the mark.
- A contest reaches kin (decision 18c): when the model contests a lexicon claim on one
  relation, the same role on a same-signature kin relation stays open with both columns,
  never ``source: lexicon``, and ``RolesResult.contested`` names both subjects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.cli_intake_utils import review_lines
from sema.intake.verify.roles import UNVERIFIABLE
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
