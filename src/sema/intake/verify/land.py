"""Land verifier: the land result re-checked against the landed relations and the run's tier.

Every check reads the catalog or an aggregate, so the verifier issues no value-bearing
statement. Header-less relations and positional, empty-named, and all-null columns are
listed as ``land`` ambiguities; a listing is never a failure.
"""

from __future__ import annotations

from typing import get_args

from sema.binding.spec_sections import Provider, RelationRef
from sema.intake.landing import LOCATOR_COLUMN, LandedRelation, LandResult
from sema.intake.profile import EvidenceTier, ProfileTools
from sema.intake.profile_utils import CatalogColumn, RelationProfile
from sema.intake.shapes.base_utils import failure
from sema.intake.verify.base import Ambiguity, Failure, Verdict
from sema.intake.verify.land_utils import (
    HEADERLESS,
    header_failures,
    listed_columns,
    locator_condition,
    type_failures,
)

ALL_LANDED = "all_landed"
ROW_COUNTS_MATCH = "row_counts_match"
HEADER_ROW_VALID = "header_row_valid"
TYPES_PARSE = "types_parse"
LOCATOR_PRESENT = "locator_present"
POLICY_CLASS_PERMITTED = "policy_class_permitted"
CHECKS = (
    ALL_LANDED,
    ROW_COUNTS_MATCH,
    HEADER_ROW_VALID,
    TYPES_PARSE,
    LOCATOR_PRESENT,
    POLICY_CLASS_PERMITTED,
)
STAGE = "land"
TIERS: tuple[str, ...] = get_args(EvidenceTier)
ROW_LOCATOR = "row"


def verify_land(
    land: LandResult, tools: ProfileTools, *, provider: Provider | None = None
) -> Verdict:
    chosen = provider or landed_provider(tools)
    failures = policy_failures(land, tools.policy.tier)
    ambiguities: list[Ambiguity] = []
    names = {landed.name for landed in land.relations}
    unlanded = [name for name in land.headerless if name not in names]
    failures += [failure(ALL_LANDED, name, listed_as=HEADERLESS) for name in unlanded]
    headerless = set(land.headerless)
    for landed in land.relations:
        ref = RelationRef(provider=chosen, name=f"{landed.schema}.{landed.name}")
        catalog = tools.catalog(ref)
        if not catalog:
            failures.append(failure(ALL_LANDED, ref.name, recorded_rows=landed.row_count))
            continue
        described = tools.describe_relation(ref)
        failures += _relation_failures(landed, ref, catalog, described, tools)
        listed = listed_columns(landed, described, landed.name in headerless)
        if listed:
            members, hints = listed
            ambiguities.append(
                Ambiguity(stage=STAGE, subject=ref.name, members=members, hints=hints)
            )
    return Verdict.of(failures=failures, ambiguities=ambiguities, floors={})


def landed_provider(tools: ProfileTools) -> Provider:
    return "staging" if tools.provider.staging is not None else "warehouse"


def policy_failures(land: LandResult, tier: EvidenceTier) -> list[Failure]:
    allowed = set(TIERS[: TIERS.index(tier) + 1])
    failures = []
    if land.policy_class not in allowed:
        failures.append(failure(POLICY_CLASS_PERMITTED, "", declared=land.policy_class, run=tier))
    for landed in land.relations:
        declared = landed.provenance.policy_class
        if declared not in allowed:
            relation = f"{landed.schema}.{landed.name}"
            failures.append(failure(POLICY_CLASS_PERMITTED, relation, declared=declared, run=tier))
    return failures


def _relation_failures(
    landed: LandedRelation,
    ref: RelationRef,
    catalog: list[CatalogColumn],
    described: RelationProfile,
    tools: ProfileTools,
) -> list[Failure]:
    failures = []
    if described.row_count != landed.row_count:
        failures.append(failure(
            ROW_COUNTS_MATCH, ref.name, recorded=landed.row_count, landed=described.row_count
        ))
    failures += [
        failure(TYPES_PARSE, ref.name, column, **evidence)
        for column, evidence in type_failures(landed, catalog)
    ]
    names = [column.name for column in catalog]
    row_locator = landed.provenance.locator_kind == ROW_LOCATOR
    if row_locator and LOCATOR_COLUMN not in names:
        failures.append(failure(LOCATOR_PRESENT, ref.name, LOCATOR_COLUMN, missing="column"))
        row_locator = False
    locator = [locator_condition(tools, ref)] if row_locator else []
    header, conditions = header_failures(landed, names, tools, ref)
    failures += [failure(HEADER_ROW_VALID, ref.name, column, **ev) for column, ev in header]
    if not conditions and not locator:
        return failures
    _, *counts = tools.count_where(ref, [*conditions.values(), *locator])
    echoed = dict(zip(conditions, counts, strict=False))
    failures += [
        failure(HEADER_ROW_VALID, ref.name, column, header_row=landed.header_row, body_rows=n)
        for column, n in echoed.items()
        if n
    ]
    if locator and counts[-1]:
        failures.append(failure(LOCATOR_PRESENT, ref.name, LOCATOR_COLUMN, rows=counts[-1]))
    return failures
