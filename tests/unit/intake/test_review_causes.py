"""Review names every cause of a missing spec (S5 decisions 14 and 21; bug-846, bug-849).

Pinned contract, read through what Review presents:

- A run that reaches Review with at least one binding and no assembled spec lists at least
  one open entry; bindings, zero open entries, and no spec never reach Review together.
- When assembly fails, Review lists one open entry per cause, naming the relation, the role or
  section, and the cause: a required role settled ``none`` names ``required_role_none``;
  per-file dictionaries whose role maps differ name the differing role; a section two
  bindings fill names the section. ``run_metrics`` counts each entry as open.
- An entry a section conflict opens carries every member of the Structure ask that settled
  it, the settled choice first, and ``validate_decisions`` admits any of them, so a reviewer
  can choose the member the model rejected.
- No provider call: the facets runs use a client that fails on any call; the books run's
  Roles answers (one whole role map per dictionary since task 2.2) and the conflict run's
  Structure and Roles answers are mocked outputs, and its decided pass makes no call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.intake.decisions import (
    Decision,
    DecisionRejectedError,
    validate_decisions,
)
from sema.intake.verify.base import Ambiguity
from tests.unit.intake.review_causes_env import (
    BOOKS,
    CONCEPT_LIST,
    CONCEPTS_SECTION,
    DIVERGENT_ROLE,
    FACETS,
    IGNORE,
    MASSES,
    REQUIRED_ROLE,
    REQUIRED_ROLE_NONE,
    ROLES,
    SOUNDS,
    STRUCTURE,
    Offered,
    approve_warehouse,
    asks_by_stage,
    books_client,
    dictionary_relation,
    discover_books,
    discover_warehouse,
    facet_pair,
    grouped_client,
    grouped_pair,
    naming,
    open_count,
    open_entries,
    qualified,
    raising_client,
    registry_calls,
    review_lists,
    summary,
    warehouse,
)

pytestmark = pytest.mark.unit

REVIEW = "review"


def _assert_no_spec_lists_a_cause(outcome: Any, *, bound: bool = True) -> None:
    """``bound=False`` where the conflict rule already moved every binding to open."""
    metrics = outcome.payload["run_metrics"]
    assert outcome.node == REVIEW, outcome.node
    assert outcome.payload["spec"] is None, "the run assembled a spec"
    assert metrics["bound"] > 0 or not bound, metrics
    assert open_count(outcome) >= 1, f"bindings, zero open entries, and no spec: {metrics}"
    assert open_entries(outcome), "Review presents no open entry"


def _required_role_none(root: Path) -> tuple[Any, Any]:
    client = raising_client()
    subject = f"{qualified(FACETS)}.{REQUIRED_ROLE}"
    with warehouse(facet_pair) as connector:
        first = discover_warehouse(root, connector, client)
        assert first.payload["spec"] is not None, "the pair must assemble before the decision"
        decided = approve_warehouse(
            root, first.run_id, connector, client, [(ROLES, subject, IGNORE)]
        )
    return decided, client


def test_per_file_dictionaries_with_divergent_role_maps_never_reach_review_unexplained(
    tmp_path: Path,
) -> None:
    client = books_client()

    outcome = discover_books(tmp_path, client)

    _assert_no_spec_lists_a_cause(outcome)
    assert asks_by_stage(client) == (len(BOOKS), 0)
    assert registry_calls(tmp_path, outcome.run_id) == len(BOOKS)


def test_divergent_role_maps_open_an_entry_naming_each_dictionary_and_the_role(
    tmp_path: Path,
) -> None:
    client = books_client()

    outcome = discover_books(tmp_path, client)

    listed = open_entries(outcome)
    for book in BOOKS:
        found = naming(listed, dictionary_relation(book), DIVERGENT_ROLE)
        assert found, f"no open entry names {book} and {DIVERGENT_ROLE}: {summary(listed)}"
    assert asks_by_stage(client) == (len(BOOKS), 0)


def test_a_required_role_settled_none_opens_an_entry_naming_its_cause(tmp_path: Path) -> None:
    outcome, client = _required_role_none(tmp_path)

    _assert_no_spec_lists_a_cause(outcome)
    listed = open_entries(outcome)
    found = naming(listed, qualified(FACETS), REQUIRED_ROLE, REQUIRED_ROLE_NONE)
    assert len(found) == 1, summary(listed)
    assert client.invoke.call_count == 0
    assert registry_calls(tmp_path, outcome.run_id) == 0


def test_a_section_two_bindings_fill_opens_one_entry_per_relation_naming_the_section(
    tmp_path: Path,
) -> None:
    offered = Offered()
    with warehouse(grouped_pair) as connector:
        outcome = discover_warehouse(tmp_path, connector, grouped_client(offered))

    _assert_no_spec_lists_a_cause(outcome, bound=False)
    listed = [e for e in open_entries(outcome) if e["stage"] == STRUCTURE]
    for table in (SOUNDS, MASSES):
        found = [e for e in naming(listed, CONCEPTS_SECTION) if e["subject"] == qualified(table)]
        assert len(found) == 1, summary(listed)


def _conflict(root: Path) -> tuple[Any, Offered, Any]:
    offered = Offered()
    client = grouped_client(offered)
    with warehouse(grouped_pair) as connector:
        outcome = discover_warehouse(root, connector, client)
    return outcome, offered, client


def _structure_entry(outcome: Any, relation: str) -> dict[str, Any]:
    (entry,) = [
        e for e in open_entries(outcome)
        if e["stage"] == STRUCTURE and e["subject"] == relation
    ]
    found: dict[str, Any] = entry
    return found


def test_a_conflict_entry_carries_every_member_the_ask_offered_settled_first(
    tmp_path: Path,
) -> None:
    outcome, offered, _ = _conflict(tmp_path)

    for table in (SOUNDS, MASSES):
        relation = qualified(table)
        members = _structure_entry(outcome, relation)["members"]
        asked = offered.for_relation(relation)
        assert len(asked) > 1, asked
        assert members[0] == offered.choice_for(relation), members
        assert {m for m in members if m != IGNORE} == set(asked), members


def _rejected_member(offered: Offered, relation: str) -> str:
    (found,) = [m for m in offered.for_relation(relation) if m.startswith(CONCEPT_LIST)]
    return found


def test_validate_decisions_admits_the_member_the_model_rejected(tmp_path: Path) -> None:
    outcome, offered, _ = _conflict(tmp_path)
    written = review_lists(tmp_path, outcome.run_id)
    presented = [Ambiguity.model_validate(entry) for entry in written]
    relation = qualified(SOUNDS)
    choice = Decision(stage=STRUCTURE, subject=relation, choice=_rejected_member(offered, relation))

    try:
        checked = validate_decisions([choice], presented)
    except DecisionRejectedError as error:
        pytest.fail(f"rejected; the entry offers only {error.members}", pytrace=False)

    assert checked == (choice,)


def test_a_decisions_file_choosing_the_rejected_member_is_accepted_with_no_call(
    tmp_path: Path,
) -> None:
    outcome, offered, _ = _conflict(tmp_path)
    calls = registry_calls(tmp_path, outcome.run_id)
    kept, dropped = qualified(SOUNDS), qualified(MASSES)
    rows = [
        (STRUCTURE, kept, _rejected_member(offered, kept)),
        (STRUCTURE, dropped, IGNORE),
    ]
    client = raising_client()

    with warehouse(grouped_pair) as connector:
        try:
            decided = approve_warehouse(tmp_path, outcome.run_id, connector, client, rows)
        except DecisionRejectedError as error:
            pytest.fail(f"rejected; the entry offers only {error.members}", pytrace=False)

    assert decided.node == REVIEW, decided.node
    assert client.invoke.call_count == 0
    assert registry_calls(tmp_path, outcome.run_id) == calls
