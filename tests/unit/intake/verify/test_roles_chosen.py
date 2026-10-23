"""The Roles verifier over a relation's chosen roles, with no assembled spec.

``verify_roles(roles, tools, spec=None, relations=[...])`` re-proves the roles each named
relation has chosen, whatever settled them, with the checks a spec binding gets: one role
per column and the per-role value checks. Every such failure names the source of the
failing entry in ``evidence["source"]``. Stated choices enter Roles through
``run_roles(..., choices={subject: StatedChoice(...)})`` and settle before any hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.intake.eligibility import run_roles
from sema.intake.eligibility_utils import StatedChoice
from sema.intake.verify.roles import verify_roles
from tests.unit.intake.eligibility_env import by_role
from tests.unit.intake.resolve_roles_env import (
    CONFLICT_TABLES,
    VALUE_TABLES,
    Synthetic,
    relation,
)
from tests.unit.intake.structure_env import fresh_tools

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def conflict(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Synthetic]:
    built = Synthetic(CONFLICT_TABLES, tmp_path_factory.mktemp("chosen_conflict"))
    yield built
    built.close()


@pytest.fixture(scope="module")
def blanks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Synthetic]:
    built = Synthetic(VALUE_TABLES, tmp_path_factory.mktemp("chosen_blanks"))
    yield built
    built.close()


def _stated(built: Synthetic, root: Path, choices: dict[str, StatedChoice]) -> Any:
    tools = fresh_tools(built.source, root / "stated", stage="roles", structure=built.structure)
    return run_roles(built.structure, built.source.profiles(), tools, choices=choices)


def _tools(built: Synthetic, root: Path) -> Any:
    return fresh_tools(built.source, root / "verify", stage="roles", structure=built.structure)


def test_stated_choices_settle_before_hints_with_their_source(
    conflict: Synthetic, tmp_path: Path
) -> None:
    zeta = relation("zeta")
    stated = StatedChoice(choice="quax", source="model", confidence=0.7)

    roles = _stated(conflict, tmp_path, {f"{zeta}.display": stated})

    display = by_role(roles.eligibilities, zeta)["display"]
    assert (display.choice, display.source, display.confidence) == ("quax", "model", 0.7)


def test_two_model_roles_on_one_column_fail_naming_both_and_the_source(
    conflict: Synthetic, tmp_path: Path
) -> None:
    zeta = relation("zeta")
    chosen = StatedChoice(choice="zorb", source="model", confidence=0.8)
    roles = _stated(conflict, tmp_path, {f"{zeta}.code": chosen, f"{zeta}.display": chosen})

    verdict = verify_roles(roles, _tools(conflict, tmp_path), spec=None, relations=[zeta])

    (found,) = [f for f in verdict.failures if f.check == "one_role_per_column"]
    assert (found.relation, found.column) == (zeta, "zorb")
    assert sorted(found.evidence["roles"]) == ["code", "display"]
    assert found.evidence["source"] == "model"


def test_a_human_display_over_blanks_passes_with_the_null_rate_finding(
    blanks: Synthetic, tmp_path: Path
) -> None:
    eta = relation("eta")
    decided = StatedChoice(choice="wim", source="human", confidence=1.0)
    roles = _stated(blanks, tmp_path, {f"{eta}.display": decided})

    verdict = verify_roles(roles, _tools(blanks, tmp_path), spec=None, relations=[eta])

    assert not [f for f in verdict.failures if f.check == "display_null_rate"]
    (found,) = [f for f in verdict.findings if f.name == "display_null_rate"]
    assert found.subject == f"{eta}.wim"


def test_without_named_relations_no_chosen_role_is_checked(
    blanks: Synthetic, tmp_path: Path
) -> None:
    eta = relation("eta")
    decided = StatedChoice(choice="wim", source="human", confidence=1.0)
    roles = _stated(blanks, tmp_path, {f"{eta}.display": decided})

    verdict = verify_roles(roles, _tools(blanks, tmp_path), spec=None)

    assert not [f for f in verdict.failures if f.check == "display_null_rate"]
