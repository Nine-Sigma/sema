"""Only claims in force make a single-relation binding fail ``precedence_applied``.

A claim precedence dropped (it is contested elsewhere) leaves the relations it alone touched
open with every binding touching them; choosing a single-relation member there must pass.
A claim that is bound still wins over a single-relation binding on its relations.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.intake.loaders.tabular import land_tabular
from sema.intake.resolve import resolve_structure
from sema.intake.shapes.base import SlotBinding, profile_relations
from sema.intake.shapes.precedence import member
from sema.intake.structure import BoundEntry
from sema.intake.structure_utils import group_candidates
from sema.intake.verify.structure import verify_structure
from tests.unit.intake.model_calls_env import wired_run
from tests.unit.intake.shapes.shapes_env import Source, warehouse_source
from tests.unit.intake.structure_env import (
    GROUPED_CODE_TABLE,
    copy_code_tables,
    fresh_tools,
    land,
    ref,
    run,
)
from tests.unit.intake.test_resolve_structure import FLAT_LIST, answering, starting

pytestmark = pytest.mark.unit

SCHEMA = "tg"
HEADER = f"{SCHEMA}.lib__first"
DICTIONARY = f"{SCHEMA}.lib__values"
LONE = f"{SCHEMA}.lib__more"
CONFIDENCE = 0.9
SQL = (
    f"CREATE SCHEMA {SCHEMA}",
    f"CREATE TABLE {HEADER} AS SELECT * FROM (VALUES ('K1', 'Alder grove'),"
    " ('K2', 'Birch hollow'), ('K3', 'Cedar ridge'), ('K4', 'Dune flat')) t(key, label)",
    f"CREATE TABLE {DICTIONARY} AS SELECT * FROM (VALUES ('K1', 'V1', 'Ivy one'),"
    " ('K1', 'V2', 'Juniper two'), ('K2', 'V3', 'Kelp three'), ('K3', 'V4', 'Larch four'),"
    " ('K3', 'V5', 'Maple five')) t(key, value, label)",
    f"CREATE TABLE {LONE} AS SELECT * FROM (VALUES ('V1', 'Ivy one'), ('V2', 'Juniper two'),"
    " ('V3', 'Kelp three'), ('V4', 'Larch four'), ('V5', 'Maple five')) t(value, label)",
)
SINGLE = f"concept_list(relation={LONE})"
CLAIM = f"entity_model(dictionary={DICTIONARY}, header={LONE}, layout=per_file)"


@pytest.fixture(scope="module")
def contested(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    con = duckdb.connect(":memory:")
    for statement in SQL:
        con.execute(statement)
    yield warehouse_source(con, SCHEMA, tmp_path_factory.mktemp("in_force") / "ledger")
    con.close()


@pytest.fixture(scope="module")
def structure(contested: Source, tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("in_force_structure")
    return run(contested, fresh_tools(contested, root / "ledger" / "structure"))


def _binding(source: Source, name: str) -> SlotBinding:
    profiles = profile_relations(source.land, source.tools, source.provider)
    (found,) = [b for b in group_candidates(profiles, source.tools) if member(b) == name]
    return found


def _chosen(structure: Any, picks: dict[str, SlotBinding]) -> Any:
    ambiguities = tuple(
        a.model_copy(
            update={"choice": member(picks[a.subject]), "source": "model", "confidence": 0.9}
        )
        if a.subject in picks
        else a
        for a in structure.ambiguities
    )
    chosen = {member(binding): binding for binding in picks.values()}
    bound = tuple(
        BoundEntry(binding=binding, confidence=CONFIDENCE, source="model")
        for binding in chosen.values()
    )
    return structure.model_copy(
        update={"ambiguities": ambiguities, "bound": (*structure.bound, *bound)}
    )


def _verify(proposal: Any, source: Source) -> Any:
    return verify_structure(proposal, source.land, source.tools, provider=source.provider)


def test_precondition_the_lone_relation_is_open_beside_a_dropped_claim(
    structure: Any,
) -> None:
    by_subject = {a.subject: a for a in structure.ambiguities}

    assert set(by_subject[LONE].members) == {SINGLE, CLAIM}
    assert DICTIONARY in by_subject
    assert structure.bound == ()


def test_single_member_chosen_beside_a_dropped_claim_passes(
    contested: Source, structure: Any
) -> None:
    proposal = _chosen(structure, {LONE: _binding(contested, SINGLE)})

    verdict = _verify(proposal, contested)

    assert verdict.status == "pass", verdict.failures


def test_single_member_beside_a_bound_claim_still_fails_precedence(
    contested: Source, structure: Any
) -> None:
    claim = _binding(contested, CLAIM)
    proposal = _chosen(structure, {DICTIONARY: claim})
    single = BoundEntry(binding=_binding(contested, SINGLE), source="model")
    proposal = proposal.model_copy(update={"bound": (*proposal.bound, single)})

    verdict = _verify(proposal, contested)

    failing = [f for f in verdict.failures if f.check == "precedence_applied"]
    assert [(f.relation, f.evidence["template"]) for f in failing] == [(LONE, "concept_list")]


@pytest.fixture(scope="module")
def codes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    """The A10 code tables without their dictionaries, so each one stays ambiguous."""
    root = tmp_path_factory.mktemp("in_force_codes")
    staging, source = land(root, "codes", land_tabular, copy_code_tables(root / "csv"))
    yield source
    staging.close()


def test_code_table_resolved_to_a_flat_list_passes_on_the_first_call(
    codes: Source, tmp_path: Path
) -> None:
    subject = ref(codes, GROUPED_CODE_TABLE)
    found = run(codes, fresh_tools(codes, tmp_path / "ledger" / "structure"))
    assert subject in {a.subject for a in found.ambiguities}
    client = answering(starting(FLAT_LIST))
    wired = wired_run(tmp_path / "calls", client)
    tools = fresh_tools(codes, tmp_path / "ledger" / "resolve")

    result = resolve_structure(found, codes.land, tools, wired.calls, provider=codes.provider)

    assert client.invoke.call_count == 1
    (entry,) = [e for e in result.bound if subject in e.binding.relations]
    assert (entry.binding.template, entry.source) == ("concept_list", "model")
    assert _verify(result, codes).status == "pass"
