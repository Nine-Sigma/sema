"""Shared setup for the stage verifier tests: landed sources, stage outputs, verdict readers.

The verifier modules are imported lazily, so a missing module fails each test on its own
instead of failing collection.

The contract the tests pin (stage-verifiers spec, design Decision 16):

- ``verify.land.verify_land(land, tools)``: the run's evidence tier is ``tools.policy``.
- ``verify.structure.verify_structure(proposal, land, tools, *, provider="staging")``.
- ``verify.roles.verify_roles(roles, tools, *, spec)``: ``roles`` is the Roles stage output
  (``eligibility.RolesResult``); ``spec`` is the assembled or edited spec, or ``None`` when
  none was assembled.
- ``verify.dry_run.verify_dry_run(spec, imported, *, structure, tools)``: ``imported`` is
  the importer's ``ConceptImport``.
- Each module exposes ``CHECKS``, the documented check names; every failure names one.
- Each returns a ``verify.base.Verdict`` whose status is ``pass`` or ``fail``, never
  ``escalate``. Land lists header-less relations, positional, empty-named, and all-null
  columns as ``land`` ambiguities whose subject is the relation's ``<schema>.<name>``.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.intake.ledger import Ledger
from sema.intake.profile import EvidencePolicy, EvidenceTier, ProfileTools
from tests.unit.intake.shapes.shapes_env import Source

LAND_CHECKS = frozenset(
    {
        "all_landed",
        "row_counts_match",
        "header_row_valid",
        "types_parse",
        "locator_present",
        "policy_class_permitted",
    }
)
STRUCTURE_CHECKS = frozenset(
    {
        "partition_complete",
        "unknown_template",
        "in_eligibility_set",
        "template_verifies",
        "precedence_applied",
        "ignore_reason_known",
        "ignore_profile_holds",
        "ambiguity_list_complete",
    }
)
ROLES_CHECKS = frozenset(
    {
        "required_roles_bound",
        "column_exists",
        "in_eligibility_set",
        "one_role_per_column",
        "section_resolved",
        "ambiguity_list_complete",
        "concept_id_unique",
        "code_unique",
        "display_null_rate",
        "synonyms_shape",
        "status_values",
        "valid_dates",
        "replaced_by_resolves",
        "lang_tags",
        "external_refs_resolve",
    }
)
DRY_RUN_CHECKS = frozenset(
    {
        "import_completed",
        "counts_in_range",
        "unique_active_code",
        "manifest_parses",
        "inline_under_cap",
        "replaced_by_active",
    }
)
VERDICT_STATUSES = {"pass", "fail"}


def verifier(name: str) -> ModuleType:
    return importlib.import_module(f"sema.intake.verify.{name}")


def tools_for(
    source: Source,
    ledger_dir: Path,
    *,
    tier: EvidenceTier = "metadata_only",
    stage: str = "verify",
    structure: Any = None,
) -> ProfileTools:
    ledger = Ledger(ledger_dir.parent).run(ledger_dir.name)
    policy = EvidencePolicy(tier=tier)
    return ProfileTools(
        source.tools.provider,
        policy,
        ledger,
        stage=stage,
        structure=structure,
        land=source.tools.land,
    )


def failures(verdict: Any, check: str) -> list[Any]:
    return [failure for failure in verdict.failures if failure.check == check]


def checks(verdict: Any) -> set[str]:
    return {failure.check for failure in verdict.failures}


def assert_named(verdict: Any, documented: Iterable[str]) -> None:
    """The scenario 'Failure names the check' holds for every verdict a test reads."""
    assert verdict.status in VERDICT_STATUSES
    assert checks(verdict) <= set(documented), checks(verdict) - set(documented)
    if verdict.failures:
        assert verdict.status == "fail"


def assert_passes(verdict: Any, documented: Iterable[str]) -> None:
    assert_named(verdict, documented)
    described = [(f.check, f.relation, f.column, f.evidence) for f in verdict.failures]
    assert verdict.status == "pass", described


def edited_spec(path: Path, directory: Path, edit: Any) -> ConceptSourceSpec:
    """A copy of a hand-written spec with ``edit(section_mapping)`` applied."""
    raw = yaml.safe_load(path.read_text())
    edit(raw["concept_source"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    target.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_concept_source_spec(target)


def without_relation(result: Any, relation: str) -> dict[str, Any]:
    """The structure fields with every bound entry, ignore, and ambiguity over a relation
    removed, as ``model_copy`` updates."""
    return {
        "bound": tuple(e for e in result.bound if relation not in e.binding.relations),
        "ignored": tuple(i for i in result.ignored if relation not in i.relations),
        "ambiguities": tuple(a for a in result.ambiguities if a.subject != relation),
    }
