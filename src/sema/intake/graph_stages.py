"""Stage recomputation for the ingestion graph: every input re-executed from Land.

A node never reads a stage object back from the ledger or the checkpoint. It asks
``Stages`` for what it needs, and ``Stages`` runs the plain stage functions over the run's
staging file. Model calls inside them replay by key, so a recomputation reaches the
provider only for a prompt the run has not recorded. Results are memoized for the life of
one process, so a node does not repeat what an earlier node of the same call computed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.binding.spec_sections import Provider
from sema.connectors.base import Connector
from sema.ingest.duckdb_staging import Staging
from sema.intake.decisions import Decision, apply_structure_decisions, stated_choices
from sema.intake.eligibility import assemble_spec, run_roles
from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.eligibility_utils import (
    EligibilitySource,
    Embedder,
    RunDescriptor,
    StatedChoice,
)
from sema.intake.landing import LandResult
from sema.intake.ledger import Ledger
from sema.intake.loaders.detect import NoLoaderError, land_source
from sema.intake.loaders.warehouse import land_warehouse
from sema.intake.model_calls import IntakeClient, model_calls_for
from sema.intake.profile import EvidencePolicy, EvidenceTier, ProfileTools
from sema.intake.profile_utils import NeedsValues
from sema.intake.proposal import ProposalResult, propose_layouts
from sema.intake.registry import Registry
from sema.intake.relations import RelationProvider
from sema.intake.report import DryRunReport, dry_run
from sema.intake.resolve import resolve_roles, resolve_structure
from sema.intake.shapes.base import RelationProfile, profile_relations
from sema.intake.structure import StructureResult, run_structure
from sema.intake.structure_conflict_utils import open_section_conflicts
from sema.intake.verify.base import Verdict
from sema.intake.verify.dry_run import verify_dry_run
from sema.models.config import IntakeModelConfig

T = TypeVar("T")

PROVIDER: Provider = "staging"
WAREHOUSE_PROVIDER: Provider = "warehouse"
REGISTRY_FILE = "registry.sqlite"
STAGING_FILE = "staging.duckdb"
VALUE_TIER: EvidenceTier = "bounded_values"
NO_VALUES_TIER: EvidenceTier = "metadata_only"
MODEL_SOURCE: EligibilitySource = "model"
LAND, STRUCTURE, RESOLVE_STRUCTURE = "land", "structure", "resolve_structure"
PROPOSE, ROLES, RESOLVE_ROLES = "propose_layouts", "roles", "resolve_roles"
DECIDE, DRY_RUN = "decide", "dry_run"
FAILED, NO_LOADER = "failed", "no_loader"
LAND_ARTIFACT = "land_result"

RECOMPUTED = frozenset({PROPOSE, ROLES, RESOLVE_ROLES, RESOLVE_STRUCTURE})

Approvals = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RunInputs:
    """``catalog`` is set for a warehouse run, whose ``schema`` is then the warehouse's."""

    run_id: str
    source: Path | None
    descriptor: Path
    schema: str
    tier: EvidenceTier
    hint: Path | None
    enrich: bool
    catalog: str | None = None


@dataclass(frozen=True)
class Reviewed:
    """What value asks and Review supplied, and which nodes the run has entered."""

    approvals: Approvals = ()
    decisions: tuple[Decision, ...] = ()
    spec_path: Path | None = None
    reached: frozenset[str] = field(default_factory=frozenset)

    def relevant(self) -> Reviewed:
        """The same inputs with only the nodes that change what Review sees."""
        return replace(self, reached=self.reached & RECOMPUTED)


@dataclass(frozen=True)
class Current:
    """The run as Review sees it: the latest structure, roles, and spec."""

    land: LandResult
    structure: StructureResult
    proposal: ProposalResult | None
    roles: RolesResult | None
    spec: ConceptSourceSpec | None


class Stages:
    def __init__(
        self,
        inputs: RunInputs,
        root: Path,
        config: IntakeModelConfig,
        client: IntakeClient | None,
        embedder: Embedder | None = None,
        warehouse: Connector | None = None,
    ) -> None:
        self.inputs = inputs
        self.embedder = embedder
        self.warehouse = warehouse
        self.provider_name = WAREHOUSE_PROVIDER if inputs.catalog else PROVIDER
        self.registry = Registry(root / REGISTRY_FILE)
        self.ledger = Ledger(root).run(inputs.run_id)
        self.calls = model_calls_for(
            config, self.ledger, self.registry, inputs.run_id, client
        )
        self._staging: Staging | None = None
        self._memo: dict[tuple[object, ...], object] = {}

    @property
    def staging(self) -> Staging:
        if self._staging is None:
            path = self.ledger.directory / STAGING_FILE
            self._staging = Staging(str(path), schemas=())
        return self._staging

    @property
    def provider(self) -> RelationProvider:
        query = self.warehouse.execute_query if self.warehouse is not None else None
        return RelationProvider(warehouse=query, staging=self.staging)

    def close(self) -> None:
        if self._staging is not None:
            self._staging.close()
            self._staging = None

    def land(self) -> LandResult:
        return self._once((LAND,), self._land)

    def descriptor(self) -> RunDescriptor:
        raw: Any = yaml.safe_load(self.inputs.descriptor.read_text())
        return RunDescriptor.model_validate(raw)

    def hint(self) -> ConceptSourceSpec | None:
        path = self.inputs.hint
        return self._once(("hint",), lambda: load_concept_source_spec(path) if path else None)

    def tools(
        self, stage: str, structure: StructureResult | None = None, approvals: Approvals = ()
    ) -> ProfileTools:
        return ProfileTools(
            self.provider, self._policy(approvals), self.ledger,
            stage=stage, structure=structure, land=self.land(),
        )

    def profiles(self, stage: str) -> list[RelationProfile]:
        return profile_relations(self.land(), self.tools(stage), self.provider_name)

    def structure(self) -> StructureResult:
        return self._once((STRUCTURE,), self._structure)

    def resolved(self) -> StructureResult:
        return self._once((RESOLVE_STRUCTURE,), self._resolved)

    def proposals(self, approvals: Approvals) -> ProposalResult:
        return self._once((PROPOSE, approvals), lambda: self._proposals(approvals))

    def roles(self, approvals: Approvals) -> RolesResult:
        return self._once((ROLES, approvals), lambda: self._roles(approvals))

    def resolved_roles(self, approvals: Approvals) -> RolesResult:
        return self._once((RESOLVE_ROLES, approvals), lambda: self._resolved_roles(approvals))

    def value_requests(self, node: str, approvals: Approvals) -> tuple[NeedsValues, ...]:
        """The value reads ``node`` asked for: a layout proposal's, or any Roles tool's."""
        if node != ROLES:
            return self.proposals(approvals).value_requests
        self.roles(approvals)
        return tuple(dict.fromkeys(self._role_tools(approvals).value_requests))

    def current(self, reviewed: Reviewed) -> Current:
        relevant = reviewed.relevant()
        return self._once((DECIDE, relevant), lambda: self._current(relevant))

    def dry_run(self, reviewed: Reviewed) -> tuple[Verdict, DryRunReport] | None:
        relevant = reviewed.relevant()
        return self._once((DRY_RUN, relevant), lambda: self._dry_run(relevant))

    def _dry_run(self, reviewed: Reviewed) -> tuple[Verdict, DryRunReport] | None:
        current = self.current(reviewed)
        if current.spec is None:
            return None
        spec = current.spec
        ran = dry_run(spec, self.provider, self.staging, self.ledger)
        structure = current.roles.structure if current.roles else current.structure
        verdict = verify_dry_run(
            spec, ran.imported, structure=structure, tools=self.tools(DRY_RUN), staged=ran.staged
        )
        return verdict, ran.report

    def _current(self, reviewed: Reviewed) -> Current:
        reached, approvals = reviewed.reached, reviewed.approvals
        proposal = self.proposals(approvals) if PROPOSE in reached else None
        structure = self._base_structure(reviewed)
        roles: RolesResult | None = None
        if reviewed.decisions:
            structure, roles = self._decided(reviewed)
        elif RESOLVE_ROLES in reached:
            roles = self.resolved_roles(approvals)
        elif ROLES in reached:
            roles = self.roles(approvals)
        spec = self._spec(reviewed.spec_path, roles, decided=bool(reviewed.decisions))
        return Current(self.land(), structure, proposal, roles, spec)

    def _spec(
        self, edited: Path | None, roles: RolesResult | None, *, decided: bool = False
    ) -> ConceptSourceSpec | None:
        """An edited spec wins; beside decisions, only where it sets a value."""
        assembled = assemble_spec(self.descriptor(), roles) if roles is not None else None
        if edited is None:
            return assembled
        spec = load_concept_source_spec(edited)
        if not decided or assembled is None:
            return spec
        merged = overlay(_dumped(assembled), _dumped(spec))
        return ConceptSourceSpec.model_validate(merged, context={"base_dir": edited.parent})

    def _base_structure(self, reviewed: Reviewed) -> StructureResult:
        if PROPOSE in reviewed.reached:
            base = self.proposals(reviewed.approvals).structure
        else:
            base = self.resolved() if RESOLVE_STRUCTURE in reviewed.reached else self.structure()
        return open_section_conflicts(base)

    def _decided(self, reviewed: Reviewed) -> tuple[StructureResult, RolesResult]:
        """Decisions over the structure Review saw; the model's role choices are kept, and
        a lexicon claim the model contested stays open unless a decision chose it."""
        tools = self.tools(DECIDE)
        base = self._base_structure(reviewed)
        land, decisions = self.land(), reviewed.decisions
        structure = open_section_conflicts(
            apply_structure_decisions(base, decisions, land, tools, provider=self.provider_name)
        )
        resolved = (
            self.resolved_roles(reviewed.approvals) if RESOLVE_ROLES in reviewed.reached else None
        )
        kept = model_choices(resolved) if resolved is not None else {}
        contested = resolved.contested if resolved is not None else {}
        choices = {**kept, **stated_choices(decisions)}
        roles = run_roles(
            structure, self.profiles(DECIDE), self.tools(DECIDE, structure, reviewed.approvals),
            hint=self.hint(), embedder=self.embedder, choices=choices, contested=contested,
        )
        return structure, with_contested_reasons(roles, contested)

    def _land(self) -> LandResult:
        inputs = self.inputs
        if inputs.source is None:
            land = land_warehouse(
                self._connector(), catalog=str(inputs.catalog), schema=inputs.schema,
                ledger=self.ledger, relations=self.descriptor().relations,
            )
        else:
            try:
                land = land_source(inputs.source, self.staging, schema=inputs.schema)
            except NoLoaderError:
                self.registry.advance(inputs.run_id, LAND, FAILED, reason=NO_LOADER)
                raise
        self.ledger.artifact(LAND_ARTIFACT, land)
        return land

    def _connector(self) -> Connector:
        if self.warehouse is None:
            raise ValueError(
                f"run {self.inputs.run_id} lands warehouse schema "
                f"{self.inputs.catalog}.{self.inputs.schema}; pass its warehouse connector"
            )
        return self.warehouse

    def _structure(self) -> StructureResult:
        tools = self.tools(STRUCTURE)
        return run_structure(self.land(), tools, provider=self.provider_name, hint=self.hint())

    def _resolved(self) -> StructureResult:
        tools = self.tools(RESOLVE_STRUCTURE)
        return resolve_structure(
            self.structure(), self.land(), tools, self.calls, provider=self.provider_name
        )

    def _proposals(self, approvals: Approvals) -> ProposalResult:
        tools = self.tools(PROPOSE, approvals=approvals)
        return propose_layouts(
            self.resolved(), self.land(), tools, self.calls, provider=self.provider_name
        )

    def _role_tools(self, approvals: Approvals) -> ProfileTools:
        structure = self.proposals(approvals).structure
        return self._once(
            ("role_tools", approvals), lambda: self.tools(ROLES, structure, approvals)
        )

    def _roles(self, approvals: Approvals) -> RolesResult:
        structure = self.proposals(approvals).structure
        tools = self._role_tools(approvals)
        return run_roles(
            structure, self.profiles(ROLES), tools, hint=self.hint(), embedder=self.embedder
        )

    def _resolved_roles(self, approvals: Approvals) -> RolesResult:
        structure = self.proposals(approvals).structure
        tools = self.tools(RESOLVE_ROLES, structure, approvals)
        return resolve_roles(
            self.roles(approvals), structure, self.land(), tools, self.calls,
            provider=self.provider_name, hint=self.hint(), embedder=self.embedder,
        )

    def _policy(self, approvals: Approvals) -> EvidencePolicy:
        tier = self.inputs.tier
        if not approvals:
            return EvidencePolicy(tier=tier)
        allowed = VALUE_TIER if tier == NO_VALUES_TIER else tier
        return EvidencePolicy(tier=allowed, allowlist=frozenset(approvals))

    def _once(self, key: tuple[object, ...], compute: Callable[[], T]) -> T:
        if key not in self._memo:
            self._memo[key] = compute()
        return cast(T, self._memo[key])


def overlay(base: dict[str, Any], edited: dict[str, Any]) -> dict[str, Any]:
    """``base`` with every value ``edited`` sets; a null in ``edited`` keeps ``base``'s."""
    merged = dict(base)
    for key, value in edited.items():
        below = merged.get(key)
        if isinstance(value, dict) and isinstance(below, dict):
            merged[key] = overlay(below, value)
        elif value is not None:
            merged[key] = value
    return merged


def _dumped(spec: ConceptSourceSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json", by_alias=True)


def model_choices(roles: RolesResult) -> dict[str, StatedChoice]:
    """The model's settled role choices, restated so a re-run of Roles keeps them."""
    return {
        entry.subject: StatedChoice(
            choice=entry.choice, source=MODEL_SOURCE, confidence=entry.confidence or 0.0
        )
        for entry in roles.ambiguities
        if entry.source == MODEL_SOURCE and entry.choice is not None
    }


def with_contested_reasons(roles: RolesResult, contested: dict[str, str]) -> RolesResult:
    """A re-run of Roles drops the contest's reason; an entry still open carries it again."""
    entries = tuple(
        entry.model_copy(update={"reason": contested[entry.subject]})
        if entry.choice is None and entry.subject in contested
        else entry
        for entry in roles.ambiguities
    )
    return replace(roles, ambiguities=entries)
