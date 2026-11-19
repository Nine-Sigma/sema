"""Done-when 3 (N5): the entity a relation's rows describe, its attribute rules and the
relations that share it, from one mocked proposal per relation and counts on the data.

The identity step runs over the identity fixture in DuckDB with a mocked model that answers
each relation's proposal call with the fixture's `Proposal`. An entity fact is read back from
the `has_identity` assertions the step stores: `entity`, `key_columns`, `owner` (the owning
relation's table reference), `identity` (an `EntityIdentity` dump), `verification`,
`evidence {rows, groups, null_rows, conflict_share}` and `alternatives`.
"""

from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.models.target.identity import EntityIdentity
from tests.support.connector_fixture_identity import (
    IDENTITY_SCHEMA,
    PROPOSALS,
    TABLES,
    conflict_shares,
    distinct_count,
    null_row_count,
)
from tests.support.connector_fixture_identity_runs import (
    STEP_RUN_ID,
    StepRun,
    prepare_step,
    proposal_for,
    relation_named,
    relation_of,
    run_step,
)

pytestmark = pytest.mark.unit

ACCEPTED = ("proven", "supported")


def proposing_client() -> MagicMock:
    """A model that answers each proposal call with the fixture's proposal for its relation."""
    client = MagicMock()

    def invoke(prompt: str, schema: Any, **kwargs: Any) -> Any:
        assert getattr(schema, "__name__", "") == "IdentityProposal", schema
        return proposal_for(schema, prompt, str(kwargs.get("table_ref", "")))

    client.invoke.side_effect = invoke
    return client


def run(directory: Path, client: Any) -> StepRun:
    def identity_step(prepared: StepRun) -> None:
        run_identity = importlib.import_module("sema.pipeline.identity").run_identity
        run_identity(prepared.graph, prepared.connector, prepared.config, [IDENTITY_SCHEMA],
                     STEP_RUN_ID, client=client)

    return run_step(prepare_step(directory), identity_step)


@pytest.fixture(scope="module")
def client() -> MagicMock:
    return proposing_client()


@pytest.fixture(scope="module")
def proposed(tmp_path_factory: pytest.TempPathFactory, client: MagicMock) -> StepRun:
    return run(tmp_path_factory.mktemp("entities"), client)


def identities(step: StepRun, table: str) -> list[dict[str, Any]]:
    return [a.payload for a in step.built().asserted("has_identity")
            if relation_of(a.subject_ref) == table]


def coarser(step: StepRun, table: str) -> dict[str, Any] | None:
    """The accepted entity coarser than the relation's row, if the step states one."""
    found = [p for p in identities(step, table)
             if p.get("verification") in ACCEPTED and p.get("identity")]
    assert len(found) <= 1, f"{table}: several coarser entities {found}"
    return found[0] if found else None


def owner_relation(value: str) -> str:
    return relation_of(value) if "://" in value else value


def shared_with(step: StepRun, owner: str) -> list[str]:
    """Relations other than the owner whose accepted entity the owner owns."""
    return sorted(
        relation_of(a.subject_ref) for a in step.built().asserted("has_identity")
        if a.payload.get("verification") in ACCEPTED and a.payload.get("owner")
        and owner_relation(str(a.payload["owner"])) == owner
        and relation_of(a.subject_ref) != owner
    )


def rules(payload: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    identity = EntityIdentity.model_validate(payload["identity"])
    return {a.column: (a.rule.value, a.by) for a in identity.attributes}


def test_the_proposal_type_has_the_contract_fields() -> None:
    proposal = importlib.import_module("sema.llm_client").IdentityProposal
    made = proposal(entity="Hull", entity_key=["hull_no"], latest_by="rev_no", row_key=None)
    assert (made.entity, made.entity_key, made.latest_by, made.row_key) == (
        "Hull", ["hull_no"], "rev_no", None)


def test_every_relation_gets_one_model_call(proposed: StepRun, client: MagicMock) -> None:
    proposed.built()
    called = Counter(relation_named(str(c.args[0] if c.args else c.kwargs.get("prompt", "")),
                                    str(c.kwargs.get("table_ref", "")))
                     for c in client.invoke.call_args_list)
    assert called == Counter(TABLES)
    assert called["gate_tally"] == 1, "a relation with no key is asked too"


def test_an_entity_coarser_than_its_row_is_accepted_with_the_proposals_name(
    proposed: StepRun,
) -> None:
    hull = coarser(proposed, "hull_rev")

    assert hull is not None
    assert (hull["entity"], hull["key_columns"], hull["verification"]) == (
        PROPOSALS["hull_rev"].entity, ["hull_no"], "proven")
    assert EntityIdentity.model_validate(hull["identity"]).key_columns == ("hull_no",)
    evidence = hull["evidence"]
    assert (evidence["rows"], evidence["groups"], evidence["null_rows"]) == (
        len(TABLES["hull_rev"].rows), distinct_count("hull_rev", ["hull_no"]), 0)
    assert evidence["conflict_share"] == pytest.approx(conflict_shares("hull_rev", ["hull_no"]))


def test_a_changing_attribute_takes_the_ordering_column_and_a_fixed_one_is_single(
    proposed: StepRun,
) -> None:
    hull = coarser(proposed, "hull_rev")
    assert hull is not None

    assert rules(hull)["paint_band"] == ("max_by", "rev_no")
    assert "keel_class" not in rules(hull)
    assert hull["evidence"]["conflict_share"]["keel_class"] == 0
    assert hull["evidence"]["conflict_share"]["paint_band"] == 0.5


def test_a_versioned_relation_with_a_unique_surrogate_still_gets_its_entity(
    proposed: StepRun,
) -> None:
    crane = coarser(proposed, "crane_rev")

    assert crane is not None
    assert (crane["entity"], crane["key_columns"]) == ("Crane", ["crane_no"])
    assert crane["evidence"]["null_rows"] == null_row_count("crane_rev", ["crane_no"]) == 2
    assert rules(crane)["lift_tons"] == ("max_by", "rev_no")


def test_an_uncorroborated_entity_key_is_supported_never_proven(proposed: StepRun) -> None:
    crane, vessel = coarser(proposed, "crane_rev"), coarser(proposed, "berth_call")

    assert crane is not None and crane["verification"] == "supported"
    assert vessel is not None and vessel["verification"] == "supported"


def test_an_ordering_column_with_ties_leaves_the_rule_open(proposed: StepRun) -> None:
    assert distinct_count("buoy_check", ["buoy_no", "check_day"]) < len(TABLES["buoy_check"].rows)

    buoy = coarser(proposed, "buoy_check")

    assert buoy is not None
    assert buoy["evidence"]["conflict_share"]["lamp_state"] > 0
    assert "lamp_state" not in rules(buoy)
    assert buoy["evidence"]["conflict_share"]["buoy_zone"] == 0


def test_a_proposed_key_equal_to_the_row_key_states_no_coarser_entity(proposed: StepRun) -> None:
    assert PROPOSALS["berth"].entity_key == TABLES["berth"].primary_key

    assert coarser(proposed, "berth") is None
    assert coarser(proposed, "hull_rev") is not None


def test_a_key_with_a_redundant_column_is_rejected(proposed: StepRun) -> None:
    assert distinct_count("deck_plan", ["deck_no"]) == distinct_count(
        "deck_plan", ["deck_no", "deck_class"])

    assert coarser(proposed, "deck_plan") is None
    grades = [p.get("verification") for p in identities(proposed, "deck_plan")]
    assert "rejected" in grades


def test_a_proposed_row_key_that_repeats_is_rejected(proposed: StepRun) -> None:
    graded = {tuple(sorted(p["columns"])): p["verification"]
              for p in (a.payload for a in proposed.built().asserted("has_key")
                        if relation_of(a.subject_ref) == "gate_tally")}

    assert graded.get(("gate_code", "tally")) == "rejected"


def test_a_shared_entity_needs_containment_at_the_floor(proposed: StepRun) -> None:
    assert shared_with(proposed, "hull_rev") == ["hull_cargo", "hull_survey"]
    assert "hull_quote" not in shared_with(proposed, "hull_rev")


def test_two_relations_keyed_by_a_same_named_dense_column_share_no_entity(
    proposed: StepRun,
) -> None:
    assert shared_with(proposed, "pump") == []
    assert shared_with(proposed, "valve") == []
    assert shared_with(proposed, "hull_rev"), "the positive case shares"


def _key_facts(step: StepRun) -> dict[str, list[str]]:
    return {relation_of(a.subject_ref): a.payload["columns"]
            for a in step.built().asserted("has_key")
            if a.payload.get("verification") in ACCEPTED}


@pytest.mark.parametrize("answer", ["nothing", "failure"])
def test_a_model_that_returns_nothing_or_fails_leaves_the_keys_and_states_no_entity(
    tmp_path: Path, proposed: StepRun, answer: str,
) -> None:
    client = MagicMock()
    if answer == "nothing":
        client.invoke.return_value = None
    else:
        client.invoke.side_effect = RuntimeError("the model is down")

    quiet = run(tmp_path, client)

    assert quiet.built().asserted("has_identity") == []
    assert _key_facts(quiet) == _key_facts(proposed)
    assert _key_facts(quiet)["berth"] == ["berth_no"]
