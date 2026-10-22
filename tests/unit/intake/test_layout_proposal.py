"""Layout proposals for relations no template reads, verified by code (S3 Decision 4).

Pinned API, imported lazily so a missing module fails each test on its own:

- ``sema.intake.proposal.propose_layouts(structure, land, tools, calls, *,
  provider="staging", max_retries=2) -> ProposalResult``. It makes one call per relation in
  ``structure.proposal_targets`` (task 7.2) with the output model ``layout_model(...)`` over
  the closed ``TEMPLATES``, the ``VIEWS`` ids, the relation's columns, and its group.
- ``ProposalResult``: ``structure`` (the updated StructureResult), ``attempts`` (relation ->
  tuple of ``LayoutAttempt(call_key, proposal, failures)`` in call order), ``escalated``
  (relations listed for Review after ``max_retries + 1`` failures), ``findings`` (tuple of
  ``LayoutFinding(relation, rationale)``), ``value_requests`` (tuple of ``NeedsValues``).
- Slot mapping: a ``slot_<slot>`` field names a column of the relation for a column-valued
  slot; ``none`` leaves the slot unset; relation-valued slots (``relation``) are filled by
  code from the proposal's ``relations``, so no ``slot_relation`` field exists.
- Verification order: the view's ``apply`` (when named), the template's ``verify``, then
  ``verify_structure``. A pass is bound as ``BoundEntry(source="model", confidence,
  call_key, view)``; ``view`` is the ``ViewSpec`` the binding was verified over, or None.
- Values: ``needs_values`` is returned as ``NeedsValues(relation, columns)`` with no further
  call; once approved (the run's policy allowlists those columns) one more call is made
  whose prompt carries the values.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from sema.intake.loaders.tabular import land_tabular
from sema.intake.profile_utils import NeedsValues
from sema.intake.shapes import TEMPLATES
from sema.intake.verify.structure import verify_structure
from sema.intake.views import VIEWS, ViewSpec
from sema.llm_client import LLMStageError
from tests.unit.intake.layout_proposal_env import (
    CODEBOOK,
    CODEBOOK_COLUMNS,
    CODEBOOK_CSV,
    CODEBOOK_VALUES,
    CONFIDENCE,
    LISTING,
    NONE,
    PROMPT_HEADER,
    REASON,
    REMARKS,
    SLOT_PREFIX,
    TERMS,
    UNANCHORED_CSV,
    VARIABLE,
    Answer,
    Landed,
    abstain,
    answer,
    approved_tools,
    bound_over,
    checks,
    columns_of,
    enum,
    ignored_over,
    invokes,
    invokes_for,
    needs,
    open_over,
    proposal,
    propose,
    run_proposals,
    scripted,
    tools_for,
    write_book,
    write_codebook,
)
from tests.unit.intake.model_calls_env import WiredRun, wired_run
from tests.unit.intake.structure_env import fresh_tools, land, run

pytestmark = pytest.mark.unit

CONCEPT_LIST = "concept_list"
DENORMALIZED = "concept_with_values_denormalized"
FILL_DOWN = "fill_down"
GROUP_BY_NON_NULL = "group_by_non_null"
ANCHOR = "fill_down_anchor"


def _landed(root: Path, name: str, path: Path) -> Iterator[Landed]:
    staging, source = land(root, name, land_tabular, path)
    structure = run(source, fresh_tools(source, root / "ledger" / "structure"))
    yield Landed(source, structure)
    staging.close()


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("layout_proposal")


@pytest.fixture(scope="module")
def book(root: Path) -> Iterator[Landed]:
    yield from _landed(root / "book", "book", write_book(root / "inputs" / "book"))


@pytest.fixture(scope="module")
def codebook(root: Path) -> Iterator[Landed]:
    path = write_codebook(root / "inputs" / "codebook", CODEBOOK_CSV)
    yield from _landed(root / "codebook", "codebook", path)


@pytest.fixture(scope="module")
def unanchored(root: Path) -> Iterator[Landed]:
    path = write_codebook(root / "inputs" / "unanchored", UNANCHORED_CSV)
    yield from _landed(root / "unanchored", "unanchored", path)


def _propose(
    landed: Landed, script: Mapping[tuple[str, ...], Sequence[Answer]], tmp_path: Path
) -> tuple[Any, MagicMock, WiredRun]:
    client = scripted(script)
    wired = wired_run(tmp_path / "calls", client)
    result = run_proposals(landed, wired, tools_for(landed, tmp_path / "ledger" / "proposal"))
    return result, client, wired


def _verified(landed: Landed, result: Any, tmp_path: Path) -> Any:
    tools = fresh_tools(landed.source, tmp_path / "ledger" / "verify")
    return verify_structure(
        result.structure, landed.source.land, tools, provider=landed.source.provider
    )


# Depends on task 7.2: the targets are read from StructureResult.proposal_targets.
def test_targets_are_the_no_header_and_no_template_relations(
    book: Landed, codebook: Landed, unanchored: Landed
) -> None:
    assert book.structure.proposal_targets == (book.name(LISTING), book.name(REMARKS))
    assert [b.slots["reason"] for b in ignored_over(book.structure, book.name(LISTING))] == [
        "no_header"
    ]
    assert codebook.structure.proposal_targets == (codebook.name(CODEBOOK),)
    assert unanchored.structure.proposal_targets == (unanchored.name(CODEBOOK),)


def test_one_versioned_call_per_target_relation(book: Landed, tmp_path: Path) -> None:
    _, client, _ = _propose(book, {}, tmp_path)

    called = [columns_of(model) for _, model in invokes(client)]
    assert sorted(called) == sorted([book.columns(LISTING), book.columns(REMARKS)])
    assert book.columns(TERMS) not in called
    assert all(prompt.startswith(PROMPT_HEADER) for prompt, _ in invokes(client))


def test_output_model_enumerates_only_the_closed_sets(book: Landed, tmp_path: Path) -> None:
    _, client, _ = _propose(book, {}, tmp_path)
    template_ids = {template.ID for template in TEMPLATES}

    for relation in (LISTING, REMARKS):
        columns = book.columns(relation)
        ((_, model),) = invokes_for(client, columns)
        templates = set(enum(model, "template")) - {NONE}
        assert templates <= template_ids
        assert template_ids - {"ignore"} <= templates
        assert enum(model, "view") == (*(view.ID for view in VIEWS), NONE)
        assert set(enum(model, "view_columns")) == set(columns)
        assert set(enum(model, "needs_values")) == set(columns)
        assert set(enum(model, "relations")) == book.group_of(relation)
        slots = [name for name in model.model_fields if name.startswith(SLOT_PREFIX)]
        assert f"{SLOT_PREFIX}group_by" in slots
        assert f"{SLOT_PREFIX}relation" not in slots
        for slot in slots:
            assert set(enum(model, slot)) == {*columns, NONE}


def test_output_model_rejects_a_view_outside_the_registry(
    book: Landed, tmp_path: Path
) -> None:
    _, client, _ = _propose(book, {}, tmp_path)
    ((_, model), *_) = invokes(client)

    with pytest.raises(ValidationError):
        answer(model, template=CONCEPT_LIST, view="pivot_columns")


def test_passing_proposal_replaces_the_no_header_ignore(book: Landed, tmp_path: Path) -> None:
    listing = book.name(LISTING)
    script = {book.columns(LISTING): [propose(CONCEPT_LIST, listing)]}

    result, client, wired = _propose(book, script, tmp_path)

    ((prompt, model),) = invokes_for(client, book.columns(LISTING))
    (entry,) = bound_over(result.structure, listing)
    assert entry.binding.template == CONCEPT_LIST
    assert entry.binding.relations == (listing,)
    assert entry.binding.slots == {"relation": listing}
    assert (entry.source, entry.confidence) == ("model", CONFIDENCE)
    assert entry.call_key == wired.calls.call_key(prompt, model)
    assert entry.view is None
    assert ignored_over(result.structure, listing) == []
    assert _verified(book, result, tmp_path).status == "pass"


def test_failing_header_less_proposal_stays_ignored(book: Landed, tmp_path: Path) -> None:
    listing = book.name(LISTING)
    columns = book.columns(LISTING)
    failing = propose(DENORMALIZED, listing, group_by=columns[0])
    script = {columns: [failing, failing, failing]}

    result, client, _ = _propose(book, script, tmp_path)

    assert len(invokes_for(client, columns)) == 3
    assert bound_over(result.structure, listing) == []
    assert [b.slots["reason"] for b in ignored_over(result.structure, listing)] == ["no_header"]
    assert listing in result.escalated


def test_view_proposal_is_verified_over_the_view_and_recorded(
    codebook: Landed, tmp_path: Path
) -> None:
    relation = codebook.name(CODEBOOK)
    filled = propose(
        DENORMALIZED, relation, view=FILL_DOWN, view_columns=[VARIABLE], group_by=VARIABLE
    )

    result, _, _ = _propose(codebook, {CODEBOOK_COLUMNS: [filled]}, tmp_path)

    (entry,) = bound_over(result.structure, relation)
    assert entry.binding.template == DENORMALIZED
    assert entry.binding.relations == (relation,)
    assert entry.binding.slots["group_by"] == VARIABLE
    assert entry.source == "model"
    assert entry.view == ViewSpec(view=FILL_DOWN, relation=relation, columns=(VARIABLE,))
    assert open_over(result.structure, relation) == []
    assert _verified(codebook, result, tmp_path).status == "pass"


def test_failing_proposal_names_the_check_and_the_retry_carries_it(
    codebook: Landed, tmp_path: Path
) -> None:
    relation = codebook.name(CODEBOOK)
    unfilled = propose(DENORMALIZED, relation, group_by=VARIABLE)
    filled = propose(
        DENORMALIZED, relation, view=FILL_DOWN, view_columns=[VARIABLE], group_by=VARIABLE
    )

    result, client, _ = _propose(codebook, {CODEBOOK_COLUMNS: [unfilled, filled]}, tmp_path)

    first, second = result.attempts[relation]
    assert GROUP_BY_NON_NULL in checks(first.failures)
    assert second.failures == ()
    (_, (retry, _)) = invokes_for(client, CODEBOOK_COLUMNS)
    assert GROUP_BY_NON_NULL in retry
    assert [entry.view for entry in bound_over(result.structure, relation)] == [
        ViewSpec(view=FILL_DOWN, relation=relation, columns=(VARIABLE,))
    ]


def test_unanchored_fill_down_fails_with_the_anchor_check(
    unanchored: Landed, tmp_path: Path
) -> None:
    relation = unanchored.name(CODEBOOK)
    filled = propose(
        DENORMALIZED, relation, view=FILL_DOWN, view_columns=[VARIABLE], group_by=VARIABLE
    )

    result, _, _ = _propose(unanchored, {CODEBOOK_COLUMNS: [filled] * 3}, tmp_path)

    attempts = result.attempts[relation]
    assert len(attempts) == 3
    assert all(ANCHOR in checks(attempt.failures) for attempt in attempts)
    assert bound_over(result.structure, relation) == []


def test_three_failures_escalate_with_every_attempt(codebook: Landed, tmp_path: Path) -> None:
    relation = codebook.name(CODEBOOK)
    unfilled = propose(DENORMALIZED, relation, group_by=VARIABLE)

    result, client, wired = _propose(codebook, {CODEBOOK_COLUMNS: [unfilled] * 4}, tmp_path)

    calls = invokes_for(client, CODEBOOK_COLUMNS)
    attempts = result.attempts[relation]
    assert len(calls) == len(attempts) == 3
    assert [a.call_key for a in attempts] == [wired.calls.call_key(p, m) for p, m in calls]
    for attempt in attempts:
        assert attempt.proposal.template == DENORMALIZED
        assert GROUP_BY_NON_NULL in checks(attempt.failures)
    assert all(GROUP_BY_NON_NULL in prompt for prompt, _ in calls[1:])
    assert result.escalated == (relation,)
    assert bound_over(result.structure, relation) == []
    assert open_over(result.structure, relation) != []


def _failed_call(model: type[BaseModel]) -> BaseModel:
    raise LLMStageError("", "intake", [("plain_invoke", ValueError("Error code: 400"))])


def test_failed_call_escalates_without_another_call(codebook: Landed, tmp_path: Path) -> None:
    relation = codebook.name(CODEBOOK)

    result, client, _ = _propose(codebook, {CODEBOOK_COLUMNS: [_failed_call] * 3}, tmp_path)

    assert len(invokes_for(client, CODEBOOK_COLUMNS)) == 1
    (attempt,) = result.attempts[relation]
    assert attempt.proposal is None
    assert result.escalated == (relation,)
    assert bound_over(result.structure, relation) == []


def test_inexpressible_layout_records_a_layout_finding(book: Landed, tmp_path: Path) -> None:
    remarks = book.name(REMARKS)
    script = {book.columns(REMARKS): [abstain(REASON)]}

    result, client, _ = _propose(book, script, tmp_path)

    assert len(invokes_for(client, book.columns(REMARKS))) == 1
    found = [finding for finding in result.findings if finding.relation == remarks]
    assert found == [proposal().LayoutFinding(relation=remarks, rationale=REASON)]
    assert bound_over(result.structure, remarks) == []
    assert open_over(result.structure, remarks) != []
    assert remarks not in result.escalated


def test_needs_values_routes_to_the_value_ask_without_another_call(
    codebook: Landed, tmp_path: Path
) -> None:
    relation = codebook.name(CODEBOOK)
    client = scripted({CODEBOOK_COLUMNS: [needs(VARIABLE, "label")]})
    wired = wired_run(tmp_path / "calls", client)
    tools = tools_for(codebook, tmp_path / "ledger" / "proposal")

    result = run_proposals(codebook, wired, tools)
    again = run_proposals(codebook, wired, tools)

    assert client.invoke.call_count == 1
    ((prompt, _),) = invokes(client)
    assert not [value for value in CODEBOOK_VALUES if value in prompt]
    expected = (NeedsValues(relation=relation, columns=(VARIABLE, "label")),)
    assert result.value_requests == again.value_requests == expected
    assert bound_over(result.structure, relation) == []
    assert relation not in result.escalated
    assert [f for f in result.findings if f.relation == relation] == []


def test_approved_values_make_one_more_call_carrying_them(
    codebook: Landed, tmp_path: Path
) -> None:
    relation = codebook.name(CODEBOOK)
    filled = propose(
        DENORMALIZED, relation, view=FILL_DOWN, view_columns=[VARIABLE], group_by=VARIABLE
    )
    client = scripted({CODEBOOK_COLUMNS: [needs(VARIABLE, "label"), filled]})
    wired = wired_run(tmp_path / "calls", client)
    run_proposals(codebook, wired, tools_for(codebook, tmp_path / "ledger" / "asked"))
    approved = approved_tools(codebook, tmp_path / "ledger" / "approved", relation, [VARIABLE])

    result = run_proposals(codebook, wired, approved)

    assert client.invoke.call_count == 2
    (_, (prompt, _)) = invokes(client)
    assert CODEBOOK_VALUES[0] in prompt
    assert result.value_requests == ()
    assert [entry.source for entry in bound_over(result.structure, relation)] == ["model"]
