"""SQL for the fill_down view: its deterministic name, the first-row scope, the view body."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from sema.binding.spec_sections import RelationRef
from sema.intake.land.landing import LOCATOR_COLUMN, locator_order_sql
from sema.intake.land.profile_utils import schema_and_table
from sema.intake.land.relations import RelationProvider
from sema.intake.structure.views.base import ViewSpec

DIGEST_LENGTH = 12
WINDOW = "landed_order"


def view_ref(spec: ViewSpec) -> RelationRef:
    """Beside the landed relation, named by the spec, so a re-apply replaces the same view."""
    schema, table = schema_and_table(RelationRef(provider="staging", name=spec.relation))
    digest = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:DIGEST_LENGTH]
    return RelationRef(provider="staging", name=f"{schema}.{table}_{spec.view}_{digest}")


def first_row_where(provider: RelationProvider, source: RelationRef) -> str:
    """Scopes a count to the first landed row, so the anchor check reads no value."""
    locator = provider.quote(source, LOCATOR_COLUMN)
    order = locator_order_sql(locator)
    relation = provider.relation_sql(source)
    return f"{locator} = (SELECT {locator} FROM {relation} ORDER BY {order} LIMIT 1)"


def fill_down_sql(
    provider: RelationProvider, source: RelationRef, view: RelationRef, columns: Sequence[str]
) -> str:
    """Each null in ``columns`` takes the last non-null above it in landed row order."""
    order = locator_order_sql(provider.quote(source, LOCATOR_COLUMN))
    fills = ", ".join(
        f"last_value({quoted} IGNORE NULLS) OVER {WINDOW} AS {quoted}"
        for quoted in (provider.quote(source, column) for column in columns)
    )
    return (
        f"CREATE OR REPLACE VIEW {provider.relation_sql(view)} AS"
        f" SELECT * REPLACE ({fills}) FROM {provider.relation_sql(source)}"
        f" WINDOW {WINDOW} AS (ORDER BY {order}"
        " ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
    )
