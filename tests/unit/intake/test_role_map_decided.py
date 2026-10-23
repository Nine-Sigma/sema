"""A contested lexicon claim on the decided path (S5 decisions 1, 3, 18c; task 2.2b).

Pinned contract, over one invented CSV relation and a mocked model, through the graph:
``discover`` stops at Review, a decisions file re-enters through ``decide``, and
``graph_stages`` re-runs Roles with the decisions and the model's kept choices.

- A. The role map contests two lexicon claims (``code`` and ``display``): each entry
  opens with both columns as members and a reason naming both sources (guard).
- B. A decisions file choosing the model's column for ``code`` keeps that column with
  ``source: human``; the lexicon's column is not the choice (guard).
- C. The contested ``display`` the file did not decide stays open with both columns and
  both sources; no contested subject re-settles as ``source: lexicon``, and the open
  role count drops only by the one human choice.
- D. The decided re-run makes no model call (guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from sema.intake.ledger import Ledger
from sema.intake.output_models import ROLE_PREFIX
from sema.intake.output_models_utils import NONE
from sema.intake.verify.base import Ambiguity
from tests.unit.intake.decisions_env import ROLES, presented, write_decisions
from tests.unit.intake.graph_env import (
    ROLE_MAP,
    STRUCTURE_BATCH,
    GraphRun,
    approve,
    choose_flat_lists,
    discover,
    routed_client,
)

pytestmark = pytest.mark.unit

RELATION = "glim_tokens"
LEXICON_CODE, MODEL_CODE = "item_code", "zorb"
LEXICON_DISPLAY, MODEL_DISPLAY = "item_label", "vell"
CODE_LISTS = "code_lists"
ROWS = 20
CONFIDENCE = 0.8
REASON = "the column facts fit the role"
HUMAN, LEXICON, MODEL = "human", "lexicon", "model"
CONTESTED_ROLES = ("code", "display")
ANSWERS = {"code": MODEL_CODE, "display": MODEL_DISPLAY, "section": CODE_LISTS}
DESCRIPTOR = "id: glim\nidentity_system: glim\nrelease: '2026-10-23'\n"


def _source(root: Path) -> Path:
    """One code table: the lexicon claims ``item_code`` and ``item_label`` whole-token."""
    directory = root / "inputs"
    directory.mkdir(parents=True)
    header = f"{LEXICON_CODE},{MODEL_CODE},{LEXICON_DISPLAY},{MODEL_DISPLAY}"
    rows = [
        f"T{i:02d},Z{i:02d},Term entry {i},Vell says a long sentence about item number {i}"
        for i in range(1, ROWS + 1)
    ]
    (directory / f"{RELATION}.csv").write_text("\n".join([header, *rows]) + "\n")
    return directory


def _contesting_map(model: type[BaseModel]) -> BaseModel:
    """Names the model's columns for the contested roles and ``none`` for every other."""
    fields = {
        name: ANSWERS.get(name.removeprefix(ROLE_PREFIX), NONE)
        for name in model.model_fields
        if name.startswith(ROLE_PREFIX)
    }
    count = len(fields)
    return model(**fields, confidences=[CONFIDENCE] * count, reasons=[REASON] * count)


@dataclass
class Contested:
    run: GraphRun
    before: list[Ambiguity]

    def listed(self) -> list[Ambiguity]:
        return presented(Ledger(self.run.root).run(self.run.run_id))

    def subject(self, role: str) -> str:
        """The landed subject of ``role``: Land prefixes the relation with its schema."""
        wanted = f".{RELATION}.{role}"
        (found,) = [a.subject for a in self.before if a.subject.endswith(wanted)]
        return found

    def entry(self, role: str) -> Ambiguity:
        (found,) = [a for a in self.listed() if a.subject == self.subject(role)]
        return found


def _discovered(root: Path) -> Contested:
    client = routed_client(
        {STRUCTURE_BATCH: [choose_flat_lists]}, defaults={ROLE_MAP: _contesting_map}
    )
    descriptor = root / "glim.run.yaml"
    descriptor.write_text(DESCRIPTOR)
    run = discover(root / "runs", _source(root), descriptor, client)
    assert run.outcome.node == "review", run.outcome.node
    contested = Contested(run, [])
    contested.before = contested.listed()
    return contested


def _decide_code(contested: Contested, root: Path) -> None:
    rows = [(ROLES, contested.subject("code"), MODEL_CODE)]
    approve(contested.run, decisions=write_decisions(root / "decisions.yaml", rows))
    assert contested.run.outcome.node == "review", contested.run.outcome.node


def _is_open(entry: Ambiguity) -> bool:
    return entry.choice is None and entry.source is None


def _names_both_sources(entry: Ambiguity) -> bool:
    explained = " ".join([entry.reason or "", *map(str, entry.hints)])
    return LEXICON in explained and MODEL in explained


def _open_roles(run: GraphRun) -> int:
    return int(run.artifact("run_metrics")["open_roles"])


def test_a_contest_opens_each_role_with_both_columns_and_sources(tmp_path: Path) -> None:
    """Regression guard: passes on HEAD (S5 2.1 ruling R5)."""
    contested = _discovered(tmp_path)

    for role, pair in zip(
        CONTESTED_ROLES, ((LEXICON_CODE, MODEL_CODE), (LEXICON_DISPLAY, MODEL_DISPLAY))
    ):
        entry = contested.entry(role)
        assert _is_open(entry), entry
        assert set(pair) <= set(entry.members), entry
        assert _names_both_sources(entry), entry
    assert _open_roles(contested.run) == len(CONTESTED_ROLES)


def test_a_human_choice_on_a_contested_role_is_kept_as_human(tmp_path: Path) -> None:
    """Regression guard: passes on HEAD (a stated choice precedes the lexicon claim)."""
    contested = _discovered(tmp_path)

    _decide_code(contested, tmp_path)

    entry = contested.entry("code")
    assert (entry.choice, entry.source) == (MODEL_CODE, HUMAN), entry
    assert entry.choice != LEXICON_CODE


def test_an_undecided_contested_role_stays_open_on_the_decided_path(tmp_path: Path) -> None:
    contested = _discovered(tmp_path)
    open_before = _open_roles(contested.run)

    _decide_code(contested, tmp_path)

    entry = contested.entry("display")
    assert _is_open(entry), (entry.choice, entry.source)
    assert {LEXICON_DISPLAY, MODEL_DISPLAY} <= set(entry.members), entry
    assert _names_both_sources(entry), entry
    subjects = {contested.subject(role) for role in CONTESTED_ROLES}
    listed = contested.listed()
    resettled = [a.subject for a in listed if a.subject in subjects and a.source == LEXICON]
    assert resettled == [], resettled
    assert _open_roles(contested.run) == open_before - 1


def test_every_entry_open_before_and_undecided_is_open_after(tmp_path: Path) -> None:
    contested = _discovered(tmp_path)
    undecided = {a.subject for a in contested.before if _is_open(a)} - {contested.subject("code")}

    _decide_code(contested, tmp_path)

    after = {a.subject: a for a in contested.listed()}
    closed = sorted(s for s in undecided if not _is_open(after[s]))
    assert closed == [], [(s, after[s].choice, after[s].source) for s in closed]


def test_the_decided_rerun_makes_no_model_call(tmp_path: Path) -> None:
    """Regression guard: passes on HEAD."""
    contested = _discovered(tmp_path)
    run = contested.run
    before, calls, mark = run.counters(), run.client.invoke.call_count, run.mark()

    _decide_code(contested, tmp_path)

    assert run.client.invoke.call_count == calls
    assert run.counters() == before
    assert "decide" in run.nodes(mark)
    assert "resolve_roles" not in run.nodes(mark)
