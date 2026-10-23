"""Shared setup for the decisions tests: decision builders, ledger readers, a reviewer.

``sema.intake.decisions`` is imported lazily, so a missing module fails each test on its own.
An entry is open when it carries neither a choice nor a source; an entry a human closed
with ``ignore`` and no member keeps ``choice: None`` with ``source: human``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

from sema.binding.spec import ConceptSourceSpec
from sema.intake.eligibility import assemble_spec, run_roles
from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.eligibility_text_utils import VOCABULARY_ROLE
from sema.intake.ledger import RunLedger
from sema.intake.lexicon import NO_REFERENCE
from sema.intake.shapes.base import profile_relations
from sema.intake.structure import StructureResult, run_structure
from sema.intake.verify.base import Ambiguity
from sema.llm_client import LLMClient
from tests.unit.intake.eligibility_env import ACRIS_RUN, descriptor
from tests.unit.intake.pipeline_env import (
    PROVIDER,
    Landed,
    read_ambiguity_list,
    write_ambiguity_lists,
)
from tests.unit.intake.structure_env import ACRIS_SPEC

STRUCTURE, ROLES, HUMAN = "structure", "roles", "human"
FLAT_LIST = "concept_list("
UNRESOLVED = "unresolved_reference"


def decisions() -> Any:
    return importlib.import_module("sema.intake.decisions")


def decision(stage: str, subject: str, choice: str) -> Any:
    return decisions().Decision(stage=stage, subject=subject, choice=choice)


def open_entries(listed: Sequence[Ambiguity], stage: str) -> list[Ambiguity]:
    return [a for a in listed if a.stage == stage and a.choice is None and a.source is None]


def open_subjects(listed: Sequence[Ambiguity], stage: str) -> set[str]:
    return {a.subject for a in open_entries(listed, stage)}


def flat_list(entry: Ambiguity) -> str:
    (found,) = [m for m in entry.members if m.startswith(FLAT_LIST)]
    return found


def presented(ledger: RunLedger) -> list[Ambiguity]:
    """The ambiguity lists Review presented, as the ledger holds them."""
    return [
        a for stage in (STRUCTURE, ROLES) for a in read_ambiguity_list(ledger, stage).ambiguities
    ]


def write_decisions(path: Path, chosen: Sequence[tuple[str, str, str]]) -> Path:
    rows = [{"stage": st, "subject": subject, "choice": choice} for st, subject, choice in chosen]
    path.write_text(yaml.safe_dump(rows, sort_keys=False))
    return path


def hand_written_roles() -> dict[str, dict[str, str]]:
    """Code list roles by relation as the hand-written spec states them: a reviewer's reading."""
    raw = yaml.safe_load(ACRIS_SPEC.read_text())["concept_source"]
    return {entry["relation"]: entry["roles"] for entry in raw["code_lists"]}


def hand_written_domains() -> dict[str, str]:
    """Reference choices by ambiguity subject, from the hand-written spec's entity ``domains``."""
    raw = yaml.safe_load(ACRIS_SPEC.read_text())["concept_source"]
    return {
        f"{entry['relation']}.{VOCABULARY_ROLE}.{prop}": vocabulary
        for entry in raw["entity_model"]["entities"]
        for prop, vocabulary in (entry.get("domains") or {}).items()
    }


def reviewer_choice(
    entry: Ambiguity, stated: dict[str, dict[str, str]], domains: dict[str, str]
) -> str | None:
    if UNRESOLVED in entry.hints:
        return str(decisions().IGNORE)
    if entry.subject in domains:
        return domains[entry.subject]
    relation, _, role = entry.subject.rpartition(".")
    if relation in stated and role in stated[relation]:
        return stated[relation][role]
    return NO_REFERENCE if NO_REFERENCE in entry.members else None


@dataclass
class Decided:
    """A no-hint run taken through two Review rounds of decisions, then re-run from Structure."""

    landed: Landed
    first_roles: RolesResult
    rounds: tuple[tuple[Any, ...], ...]
    structure: StructureResult
    roles: RolesResult
    spec: ConceptSourceSpec | None
    invoke: MagicMock


def _stages(landed: Landed, chosen: Sequence[Any]) -> tuple[StructureResult, RolesResult]:
    tools = landed.tools(STRUCTURE)
    found = run_structure(landed.land, tools, provider=PROVIDER)
    apply = decisions().apply_structure_decisions
    structure = apply(found, chosen, landed.land, tools, provider=PROVIDER)
    roles_tools = landed.tools(ROLES, structure)
    profiles = profile_relations(landed.land, roles_tools, PROVIDER)
    stated = decisions().stated_choices(chosen)
    roles = run_roles(structure, profiles, roles_tools, choices=stated)
    write_ambiguity_lists(landed.ledger, structure.ambiguities, roles.ambiguities)
    return structure, roles


def _review_round(landed: Landed, stage: str, path: Path) -> tuple[Any, ...]:
    """Structure first: role entries over unbound code tables would read as unresolved."""
    listed, stated = open_entries(presented(landed.ledger), stage), hand_written_roles()
    domains = hand_written_domains()
    rows = [(STRUCTURE, a.subject, flat_list(a)) for a in listed if stage == STRUCTURE]
    choices = [(a.subject, reviewer_choice(a, stated, domains)) for a in listed if stage == ROLES]
    rows += [(ROLES, subject, choice) for subject, choice in choices if choice is not None]
    loaded = decisions().load_decisions(write_decisions(path, rows))
    return tuple(decisions().validate_decisions(loaded, presented(landed.ledger)))


def decided(landed: Landed, root: Path) -> Decided:
    with patch.object(LLMClient, "invoke") as invoke:
        _, first_roles = _stages(landed, ())
        first = _review_round(landed, STRUCTURE, root / "first.yaml")
        _stages(landed, first)
        second = _review_round(landed, ROLES, root / "second.yaml")
        structure, roles = _stages(landed, (*first, *second))
    spec = assemble_spec(descriptor(ACRIS_RUN), roles)
    return Decided(landed, first_roles, (first, second), structure, roles, spec, invoke)
