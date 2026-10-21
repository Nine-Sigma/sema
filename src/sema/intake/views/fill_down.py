"""fill_down: a continuation-row layout read as one row per value (design Decision 4).

Each null in the named columns takes the last non-null value above it in landed row
order. A named column whose first landed row is null has nothing to fill from, so the
spec is refused with ``fill_down_anchor`` instead of inventing a value.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.landing import LOCATOR_COLUMN
from sema.intake.profile import ProfileTools
from sema.intake.shapes.base import RelationProfile, relation_profile
from sema.intake.shapes.base_utils import COLUMN_EXISTS, IN_GROUP, failure, find, verdict
from sema.intake.verify.base import Failure, Verdict
from sema.intake.views.base import ViewRefusedError, ViewSpec
from sema.intake.views.fill_down_utils import fill_down_sql, first_row_where, view_ref

ID = "fill_down"
ANCHOR = "fill_down_anchor"
STAGING_ONLY = "fill_down_staging"
LOCATED = "fill_down_locator"


def verify(spec: ViewSpec, group: Sequence[RelationProfile], tools: ProfileTools) -> Verdict:
    source = find(group, spec.relation)
    if source is None:
        return verdict([failure(IN_GROUP, spec.relation)], {})
    return verdict(_failures(spec, source, tools), {})


def apply(
    spec: ViewSpec, group: Sequence[RelationProfile], tools: ProfileTools
) -> RelationProfile:
    source = find(group, spec.relation)
    if source is None:
        raise ViewRefusedError(spec, [failure(IN_GROUP, spec.relation)])
    if failures := _failures(spec, source, tools):
        raise ViewRefusedError(spec, failures)
    view = view_ref(spec)
    statement = fill_down_sql(tools.provider, source.ref, view, spec.columns)
    tools.ledger.sql(tools.stage, view.provider, statement, False, False)
    tools.provider.query(view)(statement)
    return relation_profile(view, source.landed, tools, headerless=source.headerless)


def _failures(spec: ViewSpec, source: RelationProfile, tools: ProfileTools) -> list[Failure]:
    relation = source.ref.name
    if source.ref.provider != "staging":
        return [failure(STAGING_ONLY, relation, provider=source.ref.provider)]
    if not any(name == LOCATOR_COLUMN for name, _ in source.landed.columns):
        return [failure(LOCATED, relation, locator=LOCATOR_COLUMN)]
    missing = [c for c in spec.columns if source.column(c) is None]
    if missing:
        return [failure(COLUMN_EXISTS, relation, column) for column in missing]
    return _unanchored(spec, source, tools)


def _unanchored(spec: ViewSpec, source: RelationProfile, tools: ProfileTools) -> list[Failure]:
    quoted = [tools.provider.quote(source.ref, column) for column in spec.columns]
    where = first_row_where(tools.provider, source.ref)
    rows, *filled = tools.count_where(
        source.ref, [f"{column} IS NOT NULL" for column in quoted], where=where
    )
    return [
        failure(ANCHOR, source.ref.name, column)
        for column, count in zip(spec.columns, filled, strict=True)
        if rows == 0 or count == 0
    ]
