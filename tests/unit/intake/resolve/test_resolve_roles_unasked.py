"""Every open role entry at Review was asked, or carries a reason saying why it was not.

Defect (14.3 second re-record, A1 ``<concept relation>.name``): an open role entry reached
Review with no choice, no source, no call key, and no reason. Three generic paths leave
one so, each pinned below on a synthetic source:

- Section: an unsectioned relation's role map asks the section and the roles of both role
  models in one call (S5 decision 1), and no role settles by member count (decision 3),
  so a section choice opens no role the call did not ask. Pinned: one call, and every
  entry left open carries the model's reason.
- Claimed: an entry whose every member a lexicon claim holds is still a field of the map;
  the claims are stated in the prompt and the model may contest them (decision 18c).
- Kin: a later relation of a signature inherits only a verified role map. When the
  signature's one call leaves entries open, the kin entries are not asked again (one call
  per signature) and each carries a reason naming the relation that was asked.

Reasons are written after the calls, so no prompt and no call key changes: every committed
known-source recording still replays with zero calls (guard below, A1 included), except the
sources whose Roles keys the roles prompt v2 moved (``no_hint_env.KEYS_MOVED``, decision 26).

The mocked model answers every enum field it is handed (``role_entries_env.answering``).
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.intake.resolve.output_models_utils import ABSTAIN
from sema.models.config import ModelPrice
from tests.unit.intake.graph_env import (
    ACRIS_DESCRIPTOR,
    STRUCTURE_BATCH,
    choose_flat_lists,
    code_tables,
    routed_client,
)
from tests.unit.intake.graph_sources_env import discover_with
from tests.unit.intake.model_calls_env import intake_config
from tests.unit.intake.no_hint_env import KEYS_MOVED, keyed_param
from tests.unit.intake.resolve_roles_env import map_claims, map_members
from tests.unit.intake.role_entries_env import (
    KIN,
    KIN_SQL,
    REASON,
    abstaining,
    answering,
    concept_list,
    offered,
    pinned,
    prompts,
    resolved,
    roles_of,
    source_of,
)

pytestmark = pytest.mark.unit

ROLES, CONCEPTS, SECTION_WORD = "roles", "concepts", "section"
WIDE_CAP = 200

# Under the unsectioned role set `code` and `concept_id` claim `entry_code` and
# `entry_key_id`; `name` is asked with every column in the one role map.
SECTIONED = "sx.entries"
SECTIONED_SQL = (
    "CREATE SCHEMA sx",
    f"CREATE TABLE {SECTIONED} AS SELECT"
    " 'K' || lpad(CAST(i AS VARCHAR), 3, '0') AS entry_key_id,"
    " CASE WHEN i % 2 = 0 THEN 'Alpha' ELSE 'Beta' END AS kind_tag,"
    " 'T' || CAST(i * 7 AS VARCHAR) AS entry_code FROM range(1, 41) r(i)",
)
SECTION_SUBJECT = f"{SECTIONED}.section"

# Two columns: the lexicon claims both for `code` and `concept_id`, so `name` and `display`
# have no free member.
CLAIMED = "zz.tags"
CLAIMED_COLUMNS = ("tag_code", "tag_id")
CLAIMED_SQL = (
    "CREATE SCHEMA zz",
    f"CREATE TABLE {CLAIMED} AS SELECT 'C' || lpad(CAST(i AS VARCHAR), 2, '0') AS tag_code,"
    " 'Tag text ' || CAST(i AS VARCHAR) AS tag_id FROM range(1, 31) r(i)",
)

SOURCES = {
    "section": (SECTIONED_SQL, "sx", (SECTIONED,)),
    "claimed": (CLAIMED_SQL, "zz", (CLAIMED,)),
    "kin": (KIN_SQL, "kn", KIN),
}
Built = tuple[Any, Any, Any]


def _built(name: str, root: Path) -> Iterator[Built]:
    statements, schema, relations = SOURCES[name]
    con, source = source_of(statements, schema, root)
    structure = pinned(source, [concept_list(r, None) for r in relations], root / "pinned")
    yield source, structure, roles_of(source, structure, root / "roles")
    con.close()


@pytest.fixture
def built(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Built]:
    yield from _built(request.param, tmp_path)


def _concepts_section(field: str, members: tuple[str, ...]) -> str:
    return CONCEPTS if CONCEPTS in members else ABSTAIN


def _open(result: Any) -> list[Any]:
    return [
        a for a in result.ambiguities
        if a.stage == ROLES and a.choice is None and a.source is None
    ]


def _asked(sent: list[str], subject: str) -> bool:
    pattern = re.compile(rf"^### entry \d+: {re.escape(subject)}$", re.MULTILINE)
    return any(pattern.search(prompt) for prompt in sent)


@pytest.mark.parametrize("built", list(SOURCES), indirect=True)
def test_every_open_role_entry_says_why_it_is_open(built: Built, tmp_path: Path) -> None:
    source, structure, roles = built
    client = answering(_concepts_section)

    result, _ = resolved(source, structure, roles, client, tmp_path)

    silent = [a.subject for a in _open(result) if not a.reason]
    assert silent == [], silent


def test_the_section_fixture_opens_name_beside_the_section_before_any_answer(
    tmp_path: Path,
) -> None:
    """No role settles by member count: ``name`` is open with the section and offers the
    column the unsectioned claims leave free."""
    (source, structure, roles) = next(_built("section", tmp_path))

    opened = {a.subject: a for a in _open(roles)}

    assert {SECTION_SUBJECT, f"{SECTIONED}.name"} <= set(opened)
    assert offered(opened[f"{SECTIONED}.name"])


def test_a_section_and_its_roles_are_asked_in_one_map_call(tmp_path: Path) -> None:
    (source, structure, roles) = next(_built("section", tmp_path))
    client = answering(_concepts_section)

    result, _ = resolved(source, structure, roles, client, tmp_path / "resolve")

    (section,) = [a for a in result.ambiguities if a.subject == SECTION_SUBJECT]
    assert (section.choice, section.source) == (CONCEPTS, "model")
    assert client.invoke.call_count == 1
    (prompt,) = prompts(client)
    fields = map_members(prompt)
    assert any(SECTION_WORD in field for field in fields)
    assert any(field.endswith("_name") for field in fields)
    left = _open(result)
    assert f"{SECTIONED}.name" in {a.subject for a in left}
    for entry in left:
        assert entry.reason, entry.subject


def test_an_entry_whose_members_are_all_claimed_is_still_a_map_field(
    tmp_path: Path,
) -> None:
    (source, structure, roles) = next(_built("claimed", tmp_path))
    unaskable = [a for a in _open(roles) if not offered(a)]
    assert unaskable, "the fixture holds entries every claim excludes"
    client = answering(abstaining)

    result, _ = resolved(source, structure, roles, client, tmp_path / "resolve")

    (prompt,) = prompts(client)
    assert set(CLAIMED_COLUMNS) <= set(map_claims(prompt).values())
    fields = map_members(prompt)
    for listed in unaskable:
        role = listed.subject.rpartition(".")[2]
        assert any(field.endswith(f"_{role}") for field in fields), role
        (entry,) = [a for a in result.ambiguities if a.subject == listed.subject]
        assert entry.reason, entry.subject


def test_kin_left_open_by_their_signatures_call_name_the_relation_asked(
    tmp_path: Path,
) -> None:
    (source, structure, roles) = next(_built("kin", tmp_path))
    first, second = KIN
    client = answering(abstaining)

    result, _ = resolved(source, structure, roles, client, tmp_path / "resolve")

    sent = prompts(client)
    kin = [a for a in _open(result) if a.subject.startswith(f"{second}.")]
    assert kin, "the abstained signature leaves its kin open"
    for entry in kin:
        assert not _asked(sent, entry.subject), entry.subject
        assert entry.reason and first in entry.reason, (entry.subject, entry.reason)


def test_review_lists_no_silent_open_role_entry(tmp_path: Path) -> None:
    """Graph level: the code-tables run the model binds as flat lists, roles abstained."""
    client = routed_client(defaults={STRUCTURE_BATCH: choose_flat_lists})
    run = discover_with(
        tmp_path, code_tables(tmp_path), ACRIS_DESCRIPTOR, client,
        config={"max_calls": WIDE_CAP},
    )

    assert run.outcome.node == "review"
    listed = run.outcome.payload["ambiguities"]
    silent = [
        e["subject"] for e in listed
        if e["stage"] == ROLES and e["choice"] is None and e["source"] is None
        and not e["reason"]
    ]
    assert silent == [], silent


RECORDED = ("a1", "a2", "a3", "a7_csv", "a7_athena", "a10")


def _replayed(name: str, root: Path) -> Any:
    recorder = importlib.import_module("scripts.record_intake_run")
    recordings: Path = importlib.import_module("scripts.record_intake_run_utils").RECORDINGS_DIR
    manifest = json.loads((recordings / name / "recording.json").read_text())
    model = manifest["model"]
    config = intake_config(
        model=model, max_calls=WIDE_CAP,
        prices={model: ModelPrice(input_per_1k=0.0001, output_per_1k=0.0005)},
    )
    (source,) = [s for s in recorder.RECORDED_SOURCES if s.name == name]
    offline = MagicMock()
    offline.invoke.side_effect = AssertionError("a recorded call missed and reached the client")
    return recorder.verify_replay(
        source, recordings=recordings, workdir=root, config=config, client=offline,
        approver=manifest.get("approver"),
    )


@pytest.mark.parametrize("name", [keyed_param(name, KEYS_MOVED) for name in RECORDED])
def test_recordings_replay_with_zero_calls_and_no_silent_open_role(
    name: str, tmp_path: Path
) -> None:
    outcome = _replayed(name, tmp_path)

    assert outcome.node == "review"
    silent = [
        e["subject"] for e in outcome.payload["ambiguities"]
        if e["stage"] == ROLES and e["choice"] is None and e["source"] is None
        and not e["reason"]
    ]
    assert silent == [], silent
