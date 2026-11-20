"""A structure ``ignore`` decision may name any landed relation, bound or open (Dean, 14.3).

Pinned contract (ingestion-graph "A decisions file settles open entries", scenarios
"Ignoring a bound relation" and "Unknown relation"):

- ``validate_decisions(decisions, ambiguities, *, landed=())``: ``landed`` holds the
  qualified names of the relations Land landed. A ``structure`` decision with choice
  ``IGNORE`` passes when its subject is listed or landed; one naming neither raises
  ``DecisionRejectedError`` naming the subject. Every other decision is checked against the
  entry Review listed, as before: a ``roles`` ``ignore`` naming a relation, or a member
  choice on a relation that has no open entry, is still rejected.
- ``apply_structure_decisions``: ignoring a relation that Structure bound (by code, a model,
  or a proposal) drops the binding that holds it from ``bound``, records it in ``ignored``
  with reason ``human``, and leaves every other binding as it was. A settled structure entry
  for it is no longer reported as the model's choice. Roles over the result lists no
  eligibility or entry on it, and the structure verdict passes.
- ``graph.approve(..., decisions=...)`` and ``sema target approve --decisions`` validate with
  the landed relations of the run, so the ignore re-enters at Decide with zero model calls,
  the spec assembles without the relation, and ``run_metrics`` counts it under ``human``.
- A binding holds its relations together: an ``ignore`` naming some but not all relations of
  one bound binding is rejected (``DecisionRejectedError``) naming the binding's relations,
  before anything re-runs. Ignoring every relation of the binding is accepted.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.intake.run.decisions import IGNORE, DecisionRejectedError, apply_structure_decisions
from sema.intake.run.decisions import validate_decisions
from sema.intake.structure.shapes import ignore
from sema.intake.structure.shapes.base import SlotBinding
from sema.intake.structure.shapes.precedence import member
from sema.intake.structure.structure import BoundEntry
from sema.intake.verify.base import Ambiguity
from sema.intake.verify.structure import verify_structure
from tests.unit.cli_intake_env import Cli, decide, intake_cli, snapshot
from tests.unit.intake.decisions_env import decision, write_decisions
from tests.unit.intake.graph_env import (
    A2_DESCRIPTOR,
    ACRIS_DESCRIPTOR,
    STAGE_NODES,
    STRUCTURE_BATCH,
    GraphRun,
    approve,
    choose_flat_lists,
    code_tables,
    routed_client,
    through_values,
)
from tests.unit.intake.graph_sources_env import discover_with
from tests.unit.intake.model_calls_env import intake_config
from tests.unit.intake.role_entries_env import (
    LOOSE,
    PLAIN,
    TERMS,
    TERMS_SQL,
    concept_list,
    roles_of,
    source_of,
)
from tests.unit.intake.structure_env import A2_WORKBOOK, fresh_tools, run

pytestmark = pytest.mark.unit

STRUCTURE, ROLES, HUMAN, MODEL = "structure", "roles", "human", "model"
MODEL_KEY = "a-recorded-structure-call"
WIDE_CAP = 200
NOT_LANDED = "never_landed"
PAIRED, CONCEPTS_SECTION = "concept_and_value_set", "concepts"


# --- unit: validation and application -------------------------------------------------


def _bound(relation: str, source: str) -> BoundEntry:
    key = MODEL_KEY if source == MODEL else None
    binding = concept_list(relation, None)
    return BoundEntry(binding=binding, confidence=0.9, source=source, call_key=key)


def _settled(relation: str) -> Ambiguity:
    chosen = member(concept_list(relation, None))
    return Ambiguity(
        stage=STRUCTURE, subject=relation, members=(chosen,), choice=chosen,
        confidence=0.9, source=MODEL, call_key=MODEL_KEY,
    )


@pytest.fixture
def terms(tmp_path: Path) -> Iterator[tuple[Any, Any]]:
    """Three concept lists: ``LOOSE`` bound by a model, ``TERMS`` and ``PLAIN`` by code.

    Each relation is contested, so each carries a settled entry for the verifier."""
    con, source = source_of(TERMS_SQL, "cc", tmp_path)
    found = run(source, fresh_tools(source, tmp_path / "structure"))
    bound = (_bound(TERMS, "code"), _bound(LOOSE, MODEL), _bound(PLAIN, "code"))
    structure = found.model_copy(
        update={
            "bound": bound,
            "ambiguities": (),
            "settled": tuple(_settled(r) for r in (TERMS, LOOSE, PLAIN)),
        }
    )
    yield source, structure
    con.close()


def _landed(source: Any) -> tuple[str, ...]:
    return tuple(source.ref(relation.name) for relation in source.land.relations)


@pytest.mark.parametrize("relation", [PLAIN, LOOSE], ids=["code-bound", "model-bound"])
def test_ignoring_a_bound_relation_drops_its_binding_and_role_entries(
    terms: tuple[Any, Any], tmp_path: Path, relation: str
) -> None:
    source, structure = terms
    kept = [e for e in structure.bound if relation not in e.binding.relations]
    tools = fresh_tools(source, tmp_path / "decide")

    result = apply_structure_decisions(
        structure, [decision(STRUCTURE, relation, IGNORE)], source.land, tools,
        provider=source.provider,
    )

    assert [e for e in result.bound if relation in e.binding.relations] == []
    assert list(result.bound) == kept
    (ignored,) = [b for b in result.ignored if relation in b.relations]
    assert (ignored.template, ignored.slots[ignore.REASON]) == (ignore.ID, HUMAN)
    assert relation not in result.proposal_targets
    reported = [a for a in (*result.ambiguities, *result.settled) if a.subject == relation]
    assert all(a.source == HUMAN and a.choice is None for a in reported), reported
    roles = roles_of(source, result, tmp_path / "roles")
    assert [e for e in roles.eligibilities if e.relation == relation] == []
    assert [a for a in roles.ambiguities if a.subject.startswith(f"{relation}.")] == []
    verdict = verify_structure(result, source.land, tools, provider=source.provider)
    assert verdict.status == "pass", verdict.failures


def test_an_ignore_on_a_landed_bound_relation_passes_validation(
    terms: tuple[Any, Any],
) -> None:
    source, _ = terms
    assert PLAIN in _landed(source)
    chosen = [decision(STRUCTURE, PLAIN, IGNORE)]

    assert validate_decisions(chosen, [], landed=_landed(source)) == tuple(chosen)


def test_an_ignore_naming_a_relation_that_was_not_landed_is_rejected_naming_it(
    terms: tuple[Any, Any],
) -> None:
    source, _ = terms
    subject = f"{PLAIN.rpartition('.')[0]}.{NOT_LANDED}"

    with pytest.raises(DecisionRejectedError) as raised:
        validate_decisions([decision(STRUCTURE, subject, IGNORE)], [], landed=_landed(source))

    assert subject in str(raised.value)


@pytest.mark.parametrize("stage", [STRUCTURE, ROLES])
def test_non_ignore_or_role_choices_on_a_bound_relation_are_still_rejected(
    terms: tuple[Any, Any], stage: str
) -> None:
    source, _ = terms
    choice = member(concept_list(PLAIN, None)) if stage == STRUCTURE else IGNORE

    with pytest.raises(DecisionRejectedError) as raised:
        validate_decisions([decision(stage, PLAIN, choice)], [], landed=_landed(source))

    assert PLAIN in str(raised.value)


def test_a_member_outside_an_open_entry_is_still_rejected_naming_its_members(
    terms: tuple[Any, Any],
) -> None:
    source, _ = terms
    listed = Ambiguity(stage=STRUCTURE, subject=PLAIN, members=("first", "second"))

    with pytest.raises(DecisionRejectedError) as raised:
        validate_decisions(
            [decision(STRUCTURE, PLAIN, "third")], [listed], landed=_landed(source)
        )

    assert all(name in str(raised.value) for name in ("first", "second", "third"))


def _paired(relations: tuple[str, ...]) -> BoundEntry:
    binding = SlotBinding(
        template=PAIRED, section=CONCEPTS_SECTION, relations=relations,
        slots={"concepts": relations[0], "value_sets": relations[1]},
    )
    return BoundEntry(binding=binding, confidence=0.9, source="code")


def test_apply_rejects_an_ignore_of_one_relation_of_a_multi_relation_binding(
    terms: tuple[Any, Any], tmp_path: Path
) -> None:
    source, structure = terms
    pair = (TERMS, LOOSE)
    held = structure.model_copy(update={"bound": (_paired(pair), _bound(PLAIN, "code"))})
    tools = fresh_tools(source, tmp_path / "decide")

    with pytest.raises(DecisionRejectedError) as raised:
        apply_structure_decisions(
            held, [decision(STRUCTURE, TERMS, IGNORE)], source.land, tools,
            provider=source.provider,
        )

    assert all(relation in str(raised.value) for relation in pair), str(raised.value)


# --- graph: Decide re-runs with zero calls ---------------------------------------------


def _a2(root: Path) -> GraphRun:
    found = discover_with(
        root, A2_WORKBOOK, A2_DESCRIPTOR, routed_client(), config={"max_calls": WIDE_CAP}
    )
    through_values(found, config=intake_config(max_calls=WIDE_CAP))
    assert found.outcome.node == "review"
    return found


def _single_bound(found: GraphRun, source: str) -> str:
    """The first relation bound alone by ``source``: its ignore changes no other relation."""
    bound = found.artifact("structure")["bound"]
    alone = [e["binding"]["relations"] for e in bound if e["source"] == source]
    (relation, *_) = [names[0] for names in alone if len(names) == 1]
    return str(relation)


def _ignore(found: GraphRun, relation: str) -> tuple[int, int]:
    calls, mark = found.client.invoke.call_count, found.mark()
    path = write_decisions(found.root / "ignore.yaml", [(STRUCTURE, relation, IGNORE)])
    approve(found, decisions=path)
    return calls, mark


def _assert_ignored(found: GraphRun, relation: str, others: list[Any]) -> None:
    structure = found.artifact("structure")
    assert [e for e in structure["bound"] if relation in e["binding"]["relations"]] == []
    assert [e for e in structure["bound"] if relation not in e["binding"]["relations"]] == others
    (ignored,) = [b for b in structure["ignored"] if relation in b["relations"]]
    assert ignored["slots"][ignore.REASON] == HUMAN
    roles = found.artifact("roles_ambiguities")["ambiguities"]
    assert [e for e in roles if e["subject"].startswith(f"{relation}.")] == []
    assert found.artifact("run_metrics")["ignored"].get(HUMAN) == 1
    listed = [e for e in found.outcome.payload["ambiguities"] if e["subject"] == relation]
    assert all(e["source"] == HUMAN for e in listed), listed


def test_graph_ignore_of_a_code_bound_relation_reruns_with_zero_calls(tmp_path: Path) -> None:
    found = _a2(tmp_path)
    relation = _single_bound(found, "code")
    others = [
        e for e in found.artifact("structure")["bound"]
        if relation not in e["binding"]["relations"]
    ]

    calls, mark = _ignore(found, relation)

    assert found.outcome.node == "review"
    assert found.client.invoke.call_count == calls
    assert found.nodes(mark)[0] == "decide"
    _assert_ignored(found, relation, others)
    spec = found.outcome.payload["spec"]
    assert spec is not None
    assert relation not in json.dumps(spec)
    assert found.has_artifact("decisions")


def test_graph_ignore_of_a_model_bound_relation_reruns_with_zero_calls(tmp_path: Path) -> None:
    client = routed_client(defaults={STRUCTURE_BATCH: choose_flat_lists})
    found = discover_with(
        tmp_path, code_tables(tmp_path), ACRIS_DESCRIPTOR, client,
        config={"max_calls": WIDE_CAP},
    )
    relation = _single_bound(found, MODEL)
    others = [
        e for e in found.artifact("structure")["bound"]
        if relation not in e["binding"]["relations"]
    ]

    calls, mark = _ignore(found, relation)

    assert found.outcome.node == "review"
    assert found.client.invoke.call_count == calls
    assert found.nodes(mark)[0] == "decide"
    _assert_ignored(found, relation, others)
    spec = found.outcome.payload["spec"]
    assert spec is None or relation not in json.dumps(spec)


def test_graph_ignore_of_an_unlanded_relation_is_rejected_and_nothing_reruns(
    tmp_path: Path,
) -> None:
    found = _a2(tmp_path)
    schema = _single_bound(found, "code").rpartition(".")[0]
    subject = f"{schema}.{NOT_LANDED}"
    calls, mark = found.client.invoke.call_count, found.mark()
    path = write_decisions(tmp_path / "unknown.yaml", [(STRUCTURE, subject, IGNORE)])

    with pytest.raises(DecisionRejectedError) as raised:
        approve(found, decisions=path)

    assert subject in str(raised.value)
    assert not set(STAGE_NODES) & set(found.nodes(mark))
    assert found.client.invoke.call_count == calls
    assert not found.has_artifact("decisions")
    row = found.row()
    assert (row.stage, row.status) == ("review", "waiting_review")


def _paired_relations(found: GraphRun) -> tuple[str, ...]:
    bound = found.artifact("structure")["bound"]
    (relations, *_) = [e["binding"]["relations"] for e in bound if len(e["binding"]["relations"]) > 1]
    return tuple(relations)


def test_graph_ignore_of_part_of_a_binding_is_rejected_naming_its_relations(
    tmp_path: Path,
) -> None:
    found = _a2(tmp_path)
    relations = _paired_relations(found)
    calls, mark = found.client.invoke.call_count, found.mark()
    path = write_decisions(tmp_path / "part.yaml", [(STRUCTURE, relations[0], IGNORE)])

    with pytest.raises(DecisionRejectedError) as raised:
        approve(found, decisions=path)

    assert all(relation in str(raised.value) for relation in relations), str(raised.value)
    assert not set(STAGE_NODES) & set(found.nodes(mark))
    assert found.client.invoke.call_count == calls
    assert not found.has_artifact("decisions")


def test_graph_ignore_of_every_relation_of_a_binding_is_accepted(tmp_path: Path) -> None:
    found = _a2(tmp_path)
    relations = _paired_relations(found)
    calls = found.client.invoke.call_count
    rows = [(STRUCTURE, relation, IGNORE) for relation in relations]

    approve(found, decisions=write_decisions(tmp_path / "all.yaml", rows))

    assert found.outcome.node == "review"
    assert found.client.invoke.call_count == calls
    structure = found.artifact("structure")
    assert [e for e in structure["bound"] if set(relations) & set(e["binding"]["relations"])] == []
    reasons = {
        name: b["slots"][ignore.REASON] for b in structure["ignored"] for name in b["relations"]
    }
    assert all(reasons.get(relation) == HUMAN for relation in relations), reasons


# --- CLI: approve --decisions validates first ------------------------------------------


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    return intake_cli(tmp_path / "runs", monkeypatch)


def test_cli_ignore_of_a_bound_relation_settles_through_decide(cli: Cli) -> None:
    found = cli.reviewed(A2_WORKBOOK, A2_DESCRIPTOR)
    relation = _single_bound(found, "code")

    printed = decide(cli, found, [(STRUCTURE, relation, IGNORE)], "ignore.yaml")

    assert printed["stopped_at"] == "review"
    ignored = found.artifact("structure")["ignored"]
    assert [b["slots"][ignore.REASON] for b in ignored if relation in b["relations"]] == [HUMAN]
    assert printed["run_metrics"]["ignored"].get(HUMAN) == 1


def test_cli_ignore_of_an_unlanded_relation_fails_naming_it_and_changes_nothing(
    cli: Cli,
) -> None:
    found = cli.reviewed(A2_WORKBOOK, A2_DESCRIPTOR)
    schema = _single_bound(found, "code").rpartition(".")[0]
    subject = f"{schema}.{NOT_LANDED}"
    path = write_decisions(cli.root.parent / "unknown.yaml", [(STRUCTURE, subject, IGNORE)])
    before = snapshot(found)

    result = cli.run("approve", found.run_id, "--approver", "reviewer", "--decisions", path)

    assert result.exit_code != 0
    assert subject in result.output
    assert snapshot(found) == before
    assert not found.has_artifact("decisions")
