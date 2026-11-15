"""Settled structure entries leave the open list (fix: move settled entries out of it).

Pinned rule (relation-structure: a model-resolved table "is bound with source: model, it
leaves the ambiguity list"; ambiguity-resolution: the choice is written onto the entry):

- ``StructureResult.ambiguities`` holds open entries only: no choice, no source.
- ``StructureResult.settled: tuple[Ambiguity, ...] = ()`` holds each settled entry with its
  ``choice``, ``confidence``, ``source`` (``model`` or ``human``), and ``call_key`` (None for
  a human decision). ``resolve_structure`` and ``apply_structure_decisions`` both move
  entries there, and a later step keeps what an earlier one settled.
- Spec assembly checks the open list only, so a structure the model settled assembles.
- Review still presents each settled structure entry in its payload's ``ambiguities``
  beside the open ones; the ``structure_ambiguities`` artifact, which decisions are
  validated against, holds the open entries only.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.intake.run.decisions import Decision, apply_structure_decisions, stated_choices
from sema.intake.roles.eligibility import assemble_spec
from sema.intake.calls.ledger_utils import prompt_hash
from sema.intake.land.loaders.tabular import land_tabular
from sema.intake.resolve.output_models import STRUCTURE_BATCH
from sema.intake.structure.structure import StructureResult
from sema.intake.verify.base import Ambiguity
from tests.unit.intake.decisions_env import flat_list, write_decisions
from tests.unit.intake.eligibility_env import descriptor, run_roles
from tests.unit.intake.graph_env import (
    A2_DESCRIPTOR,
    ACRIS_DESCRIPTOR,
    APPROVER,
    STRUCTURE_BATCH as STRUCTURE_BATCH_MODEL,
    choose_flat_lists,
    graph,
    routed_client,
)
from tests.unit.intake.graph_sources_env import (
    KEYED_HEADER,
    discover_with,
    ranked_source,
    role_entry,
)
from tests.unit.intake.model_calls_env import MODEL, intake_config, wired_run
from tests.unit.intake.structure.shapes.shapes_env import Source
from tests.unit.intake.structure_env import (
    CODE_TABLES,
    GROUPED_CODE_TABLE,
    copy_code_tables,
    fresh_tools,
    land,
    ref,
    run,
)
from tests.unit.intake.resolve.resolve_structure_env import (
    CONFIDENCE,
    FLAT_LIST,
    answering,
    prompts,
    resolve,
    starting,
)

pytestmark = pytest.mark.unit

STRUCTURE, ROLES, MODEL_SOURCE, HUMAN = "structure", "roles", "model", "human"
SECTION_CHOICE = "concepts"
WIDE_CAP = 200


@pytest.fixture(scope="module")
def codes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    root = tmp_path_factory.mktemp("structure_settled")
    staging, source = land(root / "codes", "codes", land_tabular, copy_code_tables(root / "csv"))
    yield source
    staging.close()


@pytest.fixture(scope="module")
def codes_structure(codes: Source, tmp_path_factory: pytest.TempPathFactory) -> StructureResult:
    ledger = tmp_path_factory.mktemp("structure_settled_ledger")
    return run(codes, fresh_tools(codes, ledger / "structure"))


def _resolved(
    codes: Source, structure: StructureResult, root: Path, pick: Any
) -> tuple[StructureResult, Any]:
    client = answering(pick)
    wired = wired_run(root / "calls", client)
    tools = fresh_tools(codes, root / "ledger" / "resolve")
    result = resolve().resolve_structure(
        structure, codes.land, tools, wired.calls, provider=codes.provider
    )
    return result, client


def _settled(result: StructureResult, subject: str) -> Ambiguity:
    (found,) = [a for a in getattr(result, "settled", ()) if a.subject == subject]
    return found


def _open_subjects(result: StructureResult) -> set[str]:
    return {a.subject for a in result.ambiguities}


def test_model_settled_entry_leaves_the_open_list_for_settled(
    codes: Source, codes_structure: StructureResult, tmp_path: Path
) -> None:
    result, client = _resolved(codes, codes_structure, tmp_path, starting(FLAT_LIST))

    (prompt,) = prompts(client)
    key = prompt_hash(MODEL, prompt, STRUCTURE_BATCH)
    assert result.ambiguities == ()
    for name in CODE_TABLES:
        entry = _settled(result, ref(codes, name))
        assert entry.choice is not None and entry.choice.startswith(FLAT_LIST)
        assert (entry.source, entry.confidence, entry.call_key) == (
            MODEL_SOURCE, CONFIDENCE, key,
        )


def test_model_settled_structure_assembles_a_spec(
    codes: Source, codes_structure: StructureResult, tmp_path: Path
) -> None:
    result, _ = _resolved(codes, codes_structure, tmp_path, starting(FLAT_LIST))
    first = run_roles(codes, result, tmp_path / "ledger" / "roles")
    open_roles = [a for a in first.ambiguities if a.choice is None and a.source is None]
    stated = [Decision(stage=ROLES, subject=a.subject, choice=a.members[-1]) for a in open_roles]

    roles = run_roles(
        codes, result, tmp_path / "ledger" / "decided", choices=stated_choices(stated)
    )

    assert [a for a in roles.ambiguities if a.choice is None and a.source is None] == []
    assert assemble_spec(descriptor(ACRIS_DESCRIPTOR), roles) is not None


def test_human_decision_moves_the_entry_to_settled(
    codes: Source, codes_structure: StructureResult
) -> None:
    subject = ref(codes, GROUPED_CODE_TABLE)
    (entry,) = [a for a in codes_structure.ambiguities if a.subject == subject]
    made = Decision(stage=STRUCTURE, subject=subject, choice=flat_list(entry))

    result = apply_structure_decisions(
        codes_structure, [made], codes.land, codes.tools, provider=codes.provider
    )

    assert subject not in _open_subjects(result)
    settled = _settled(result, subject)
    assert settled.choice == flat_list(entry)
    assert (settled.source, settled.confidence, settled.call_key) == (HUMAN, 1.0, None)


def test_decisions_keep_what_the_model_settled(
    codes: Source, codes_structure: StructureResult, tmp_path: Path
) -> None:
    grouped = ref(codes, GROUPED_CODE_TABLE)
    result, _ = _resolved(
        codes, codes_structure, tmp_path, starting(FLAT_LIST, mentioning=grouped)
    )
    (other,) = [a for a in result.ambiguities if a.subject != grouped][:1]
    made = Decision(stage=STRUCTURE, subject=other.subject, choice=flat_list(other))

    decided = apply_structure_decisions(
        result, [made], codes.land, codes.tools, provider=codes.provider
    )

    assert _settled(decided, grouped).source == MODEL_SOURCE
    assert _settled(decided, other.subject).source == HUMAN
    assert not {grouped, other.subject} & _open_subjects(decided)


def test_graph_run_with_a_model_settled_structure_reaches_review_with_a_spec(
    tmp_path: Path,
) -> None:
    client = routed_client(defaults={STRUCTURE_BATCH_MODEL: choose_flat_lists})
    source = ranked_source(tmp_path, KEYED_HEADER)
    run = discover_with(tmp_path, source, A2_DESCRIPTOR, client)
    listed = run.artifact("structure_ambiguities")
    assert (listed["count"], listed["ambiguities"]) == (0, [])
    section = role_entry(run, "section")
    path = write_decisions(
        tmp_path / "decisions.yaml", [(ROLES, section["subject"], SECTION_CHOICE)]
    )
    calls = client.invoke.call_count

    outcome = graph().approve(
        run.run_id, root=tmp_path, config=intake_config(max_calls=WIDE_CAP, agreement=False),
        approver=APPROVER, client=client, decisions=path,
    )

    assert outcome.node == "review"
    assert outcome.payload["spec"] is not None
    assert client.invoke.call_count == calls
    presented = [e for e in outcome.payload["ambiguities"] if e["stage"] == STRUCTURE]
    assert [(e["source"], e["choice"] is not None) for e in presented] == [(MODEL_SOURCE, True)]
    assert run.artifact("structure_ambiguities")["count"] == 0
