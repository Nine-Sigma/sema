"""Agreement between two asks is the Roles confidence signal (S5 decisions 4, 16, 17).

Pinned contract (task 2.3, rulings A1 to A5), over invented relations and a mocked model:

- With ``IntakeModelConfig.agreement`` on (the default; env ``INTAKE_MODEL_AGREEMENT``) every
  role map and every reference ask is made twice; the second prompt holds the same lines
  with the columns in an order seeded by the relation signature (the same order on every
  run; another signature, another order). The registry counts both calls.
- Two agreeing answers settle the role with ``source: model``, the lower confidence, and
  ``agreement`` ``"2/2"`` on the entry. Differing answers leave it open with both columns
  ranked by confidence and both reasons, and the relation records one disagreement.
- A replay follows its recording manifest: ``agreement: false``, or no field, makes one call
  per ask whatever the environment says; ``verify_replay`` fails naming the flag when the
  environment and the manifest differ.

A recordings directory here holds a manifest and no call record, so every ask misses the
replay and reaches the mocked model, where the calls are counted.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.unit.intake.model_calls_env import intake_config
from tests.unit.intake.resolve_roles_env import DICTIONARY, REFERENCE_TABLES, Synthetic
from tests.unit.intake.resolve.role_agreement_env import (
    AGREEMENT_ENV,
    OTHER,
    OTHER_COLUMNS,
    WIDE,
    WIDE_COLUMNS,
    Ask,
    agreement_of,
    answers_for,
    asking,
    calls_with,
    catalog_source,
    column_order,
    disagreements,
    permutation,
    recordings,
    resolve_under,
    wide,
)
from tests.unit.intake.role_map_env import Scenario, chosen, explained, subject_entry

pytestmark = pytest.mark.unit

AGREED = Ask(answers_for(WIDE_COLUMNS), 0.9, "the key column is unique")
LOWER = Ask(answers_for(WIDE_COLUMNS), 0.7, "the label column reads as names")
DIFFERING = Ask({**answers_for(WIDE_COLUMNS), "name": "vell"}, 0.6, "the text reads as names")


@pytest.fixture(autouse=True)
def agreement_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(AGREEMENT_ENV, raising=False)


@pytest.fixture
def scene(tmp_path: Path) -> Iterator[Scenario]:
    found = wide(tmp_path / "wide")
    yield found
    found.con.close()


def _order(root: Path, name: str, columns: tuple[str, ...]) -> list[list[str]]:
    found = wide(root, name, columns)
    client = asking(Ask(answers_for(columns), 0.9, "fits"))
    resolve_under(found, client)
    found.con.close()
    return [column_order(prompt, columns) for prompt in calls_with(client, "role_")]


def test_agreement_is_on_by_default_and_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert intake_config().agreement is True
    monkeypatch.setenv(AGREEMENT_ENV, "0")
    assert intake_config().agreement is False


def test_a_role_map_is_asked_twice_with_the_columns_reordered(scene: Scenario) -> None:
    client = asking(AGREED)

    resolve_under(scene, client)

    sent = calls_with(client, "role_")
    assert len(sent) == 2, f"{len(sent)} role-map calls"
    first, second = sent
    assert sorted(first.splitlines()) == sorted(second.splitlines())
    assert column_order(first, WIDE_COLUMNS) != column_order(second, WIDE_COLUMNS)


def test_the_reordering_is_seeded_by_the_relation_signature(tmp_path: Path) -> None:
    once = _order(tmp_path / "once", WIDE, WIDE_COLUMNS)
    again = _order(tmp_path / "again", WIDE, WIDE_COLUMNS)
    other = _order(tmp_path / "other", OTHER, OTHER_COLUMNS)

    assert len(once) == len(again) == len(other) == 2, (len(once), len(other))
    assert once == again
    assert permutation(*once) != permutation(*other)


def test_a_reference_ask_is_made_twice(tmp_path: Path) -> None:
    built = Synthetic(REFERENCE_TABLES, tmp_path, dictionaries=frozenset({DICTIONARY}))
    found = Scenario(built.con, built.source, built.structure, built.roles, tmp_path, {})
    client = asking(Ask({"property": "property", "description": "description"}, 0.9, "fits"))

    resolve_under(found, client)

    assert len(calls_with(client, "reference_")) == 2
    built.close()


def test_agreeing_answers_settle_with_the_lower_confidence_and_two_of_two(
    scene: Scenario,
) -> None:
    result, _ = resolve_under(scene, asking(AGREED, LOWER))

    name = chosen(result, WIDE, "name")
    assert (name.choice, name.source, name.confidence) == ("quax", "model", 0.7), name
    assert agreement_of(subject_entry(result, f"{WIDE}.name")) == "2/2"


def test_differing_answers_stay_open_with_both_and_count_one_disagreement(
    scene: Scenario,
) -> None:
    result, _ = resolve_under(scene, asking(AGREED, DIFFERING))

    entry = subject_entry(result, f"{WIDE}.name")
    assert (entry.choice, entry.source) == (None, None), entry
    assert entry.members[:2] == ("quax", "vell"), entry.members
    assert AGREED.reason in explained(entry) and DIFFERING.reason in explained(entry), entry
    assert chosen(result, WIDE, "concept_id").choice == "zorb"
    assert disagreements(result, WIDE) == 1


def test_the_registry_counts_both_calls_of_one_ask(scene: Scenario) -> None:
    client = asking(AGREED)

    _, wired = resolve_under(scene, client)

    assert client.invoke.call_count == 2
    assert wired.row().model_calls == 2


def test_with_agreement_off_one_call_is_made(
    scene: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: holds on HEAD, where the flag does not exist."""
    monkeypatch.setenv(AGREEMENT_ENV, "0")
    client = asking(AGREED)

    resolve_under(scene, client)

    assert client.invoke.call_count == 1


@pytest.mark.parametrize("manifest", [False, None], ids=["false", "absent"])
def test_a_replay_follows_its_manifest_whatever_the_environment_says(
    scene: Scenario, monkeypatch: pytest.MonkeyPatch, manifest: bool | None
) -> None:
    monkeypatch.setenv(AGREEMENT_ENV, "1")
    assert intake_config().agreement is True
    directory = recordings(scene.root, "synthetic", manifest)
    client = asking(AGREED)

    resolve_under(scene, client, recordings_dir=directory, max_calls=50)

    assert client.invoke.call_count == 1


def _raising() -> MagicMock:
    client = MagicMock()
    client.invoke.side_effect = AssertionError("a provider call was made")
    return client


def _verify(root: Path) -> Any:
    script = importlib.import_module("scripts.record_intake_run")
    return script.verify_replay(
        catalog_source(root), recordings=root / "recordings", workdir=root / "replay",
        config=intake_config(max_calls=50), client=_raising(),
    )


@pytest.mark.parametrize("manifest", [False, None], ids=["false", "absent"])
def test_verify_replay_names_the_flag_when_environment_and_manifest_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: bool | None
) -> None:
    monkeypatch.setenv(AGREEMENT_ENV, "1")
    recordings(tmp_path, "synthetic", manifest)

    with pytest.raises(Exception, match="agreement"):
        _verify(tmp_path)


def test_verify_replay_passes_when_the_flag_matches_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: a source that needs no call replays with the flags equal."""
    monkeypatch.setenv(AGREEMENT_ENV, "0")
    recordings(tmp_path, "synthetic", False)

    assert _verify(tmp_path).node == "review"
