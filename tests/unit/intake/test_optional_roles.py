"""Optional column roles can be answered "no column" (design 14.3 pause, Dean 2026-10-21).

Pinned rules:

- A role is optional for a target when no role model of that target declares it required
  (``FieldInfo.is_required()``); a relation whose section is still open targets both
  concept and code-list roles, so a role either requires stays required.
- Eligibility adds ``lexicon.NO_REFERENCE`` (``none``) as the last member of every optional
  column role, never of a required one. Every column is a member (S5 decision 3). The
  role-map prompt states each role optional or required, so ``none`` is answerable for an
  optional role only.
- No role settles by member count. Free columns plus ``none`` are, with no hint, an open
  ambiguity entry; a lexicon hit still settles it (hints run in the fixed order). An
  optional role whose every column another role holds settles ``none`` with
  ``source: code`` (``test_identity_roles``). A text column is a date and an array
  candidate: whether it parses is a value check of the verifier.
- A ``none`` choice (model or human) closes the entry, leaves the role unbound in the
  assembled spec, and passes the Roles verifier.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.intake.eligibility_text_utils import CONCEPTS
from sema.intake.lexicon import NO_REFERENCE
from sema.intake.output_models import role_field
from sema.intake.shapes.base import SlotBinding
from sema.intake.verify.roles import verify_roles
from tests.unit.intake.decisions_env import HUMAN, ROLES, decision, decisions
from tests.unit.intake.eligibility_env import by_role, eligibility, eligibility_utils
from tests.unit.intake.eligibility_env import run_roles as roles_over
from tests.unit.intake.model_calls_env import wired_run
from tests.unit.intake.resolve_roles_env import (
    Synthetic,
    Tables,
    answering,
    distinct,
    entry,
    lines_with,
    prompts,
    relation,
)
from tests.unit.intake.structure_env import fresh_tools

pytestmark = pytest.mark.unit

NONE = NO_REFERENCE
LONE = "theta"
NAMED = "iota"
# `pell` is the only low-cardinality column and carries no status word; `row_status` does.
LONE_TABLES: Tables = {
    LONE: (
        ("kelt", "wim", "pell"),
        [(f"K{i:02d}", f"Wim entry {i}", "A" if i % 2 else "B") for i in range(1, 21)],
    ),
}
NAMED_TABLES: Tables = {
    NAMED: (
        ("kelt", "wim", "row_status"),
        [(f"K{i:02d}", f"Wim entry {i}", "A" if i % 2 else "B") for i in range(1, 21)],
    ),
}
CHOSEN = {"code": "kelt", "display": "wim"}
UNCANDIDATED = ("valid_from", "valid_to", "synonyms")
LONE_COLUMNS = ("kelt", "wim", "pell")


def resolve() -> Any:
    return importlib.import_module("sema.intake.resolve")


@pytest.fixture(scope="module")
def tmp_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("optional_roles")


@pytest.fixture(scope="module")
def lone(tmp_module: Path) -> Iterator[Synthetic]:
    built = Synthetic(LONE_TABLES, tmp_module / "lone")
    yield built
    built.close()


@pytest.fixture(scope="module")
def named(tmp_module: Path) -> Iterator[Synthetic]:
    built = Synthetic(NAMED_TABLES, tmp_module / "named")
    yield built
    built.close()


def _descriptor() -> Any:
    return eligibility_utils().RunDescriptor(id="optional", identity_system="optional", release="1")


def _verdict(built: Synthetic, result: Any, spec: Any, root: Path) -> Any:
    tools = fresh_tools(built.source, root / "verify", stage=ROLES, structure=built.structure)
    return verify_roles(result, tools, spec=spec)


def _resolved(built: Synthetic, client: MagicMock, root: Path) -> Any:
    wired = wired_run(root / "calls", client)
    tools = fresh_tools(
        built.source, root / "ledger" / "resolve", stage=ROLES, structure=built.structure
    )
    return resolve().resolve_roles(
        built.roles, built.structure, built.source.land, tools, wired.calls,
        provider=built.source.provider,
    )


def _assert_unbound_status(built: Synthetic, result: Any, root: Path) -> None:
    spec = eligibility().assemble_spec(_descriptor(), result)
    assert spec is not None, [(a.subject, a.members, a.choice) for a in result.ambiguities]
    (listed,) = spec.code_lists
    roles = listed.roles
    assert (roles.code, roles.display, roles.status) == ("kelt", "wim", None)
    assert all(getattr(roles, name) is None for name in UNCANDIDATED)
    assert NONE not in roles.model_dump(exclude_none=True).values()
    verdict = _verdict(built, result, spec, root)
    assert verdict.status == "pass", verdict.failures


def test_optional_code_list_roles_offer_none_and_required_roles_do_not(
    lone: Synthetic,
) -> None:
    roles = by_role(lone.roles.eligibilities, relation(LONE))

    assert roles["status"].members == (*LONE_COLUMNS, NONE)
    for name in UNCANDIDATED:
        assert NONE in roles[name].members, name
    for name in ("code", "display"):
        assert NONE not in roles[name].members, name


def test_optional_concept_roles_offer_none_and_required_roles_do_not(lone: Synthetic) -> None:
    ref = relation(LONE)
    binding = SlotBinding(
        template="concept_list", slots={"relation": ref}, relations=(ref,), section=CONCEPTS
    )
    roles = by_role(eligibility().compute(binding, lone.source.profiles()))

    for name in ("definition", "status", "synonyms"):
        assert NONE in roles[name].members, name
    for name in ("concept_id", "name"):
        assert NONE not in roles[name].members, name


def test_free_columns_plus_none_are_an_open_ambiguity(lone: Synthetic) -> None:
    subject = f"{relation(LONE)}.status"
    status = by_role(lone.roles.eligibilities, relation(LONE))["status"]

    assert (status.choice, status.source) == (None, None)
    (listed,) = [a for a in lone.roles.ambiguities if a.subject == subject]
    assert listed.members == (*LONE_COLUMNS, NONE)
    assert (listed.choice, listed.source) == (None, None)


def test_a_lexicon_hit_still_settles_an_optional_role(named: Synthetic) -> None:
    subject = f"{relation(NAMED)}.status"
    status = by_role(named.roles.eligibilities, relation(NAMED))["status"]

    assert status.members == ("kelt", "wim", "row_status", NONE)
    assert (status.choice, status.source) == ("row_status", "lexicon")
    (listed,) = [a for a in named.roles.ambiguities if a.subject == subject]
    assert (listed.choice, listed.source) == ("row_status", "lexicon")


def test_a_free_text_column_keeps_date_and_array_roles_open(lone: Synthetic) -> None:
    roles = by_role(lone.roles.eligibilities, relation(LONE))
    listed = {a.subject for a in lone.roles.ambiguities}

    for name in UNCANDIDATED:
        assert roles[name].members == (*LONE_COLUMNS, NONE), name
        assert (roles[name].choice, roles[name].source) == (None, None), name
        assert f"{relation(LONE)}.{name}" in listed, name


def test_the_role_map_states_none_answerable_for_an_optional_role_only(
    lone: Synthetic, tmp_path: Path
) -> None:
    client = answering(distinct({**CHOSEN, "status": NONE}))

    _resolved(lone, client, tmp_path)

    (prompt,) = prompts(client)
    (status,) = lines_with(prompt, f"`{role_field('status')}`:")
    assert "; optional" in status, status
    for name in ("code", "display"):
        (line,) = lines_with(prompt, f"`{role_field(name)}`:")
        assert "; required" in line, line


def test_a_model_none_closes_the_entry_and_the_spec_assembles(
    lone: Synthetic, tmp_path: Path
) -> None:
    client = answering(distinct({**CHOSEN, "status": NONE}))

    result = _resolved(lone, client, tmp_path)

    listed = entry(result, relation(LONE), "status")
    assert (listed.choice, listed.source) == (NONE, "model")
    assert listed.call_key is not None
    status = by_role(result.eligibilities, relation(LONE))["status"]
    assert (status.choice, status.source) == (NONE, "model")
    _assert_unbound_status(lone, result, tmp_path)


def test_a_human_none_closes_the_entry_and_the_spec_assembles(
    lone: Synthetic, tmp_path: Path
) -> None:
    ref = relation(LONE)
    made = [decision(ROLES, f"{ref}.{role}", column) for role, column in CHOSEN.items()]
    made.append(decision(ROLES, f"{ref}.status", NONE))
    valid = decisions().validate_decisions(made, lone.roles.ambiguities)
    stated = decisions().stated_choices(valid)

    result = roles_over(lone.source, lone.structure, tmp_path / "roles", choices=stated)

    (listed,) = [a for a in result.ambiguities if a.subject == f"{ref}.status"]
    assert (listed.choice, listed.source, listed.confidence) == (NONE, HUMAN, 1.0)
    status = by_role(result.eligibilities, ref)["status"]
    assert (status.choice, status.source) == (NONE, HUMAN)
    _assert_unbound_status(lone, result, tmp_path)
