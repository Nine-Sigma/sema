"""Roles verifier containment: a reference role's column holds only its target's values.

``fk_entity`` must be contained in the entity column and ``fk_property`` in the property
column (``lexicon.CONTAINED_IN``). Each check is a count over the relation, so no value is
read and the check runs under ``metadata_only``. Values compare trimmed and case-folded,
because identifier case varies between a dictionary's own columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sema.binding.spec_sections import RelationRef
from sema.intake.roles import lexicon
from sema.intake.land.profile import ProfileTools
from sema.intake.land.profile_utils import TEXT_TYPE
from sema.intake.structure.shapes.base_utils import failure
from sema.intake.verify.base import Failure

CONTAINED = "contained"
# Measured 2026-10-23 on DuckDB over a 1000-column relation: one statement with a condition
# per column fails at 1000 (expression depth) and runs out of memory at 600; statements of
# 16, 32, and 64 conditions take 0.62 s, 0.87 s, and 1.79 s in all.
CONTAINMENT_BATCH = 16


def containment_failures(
    ref: RelationRef, bound: Mapping[str, str], tools: ProfileTools
) -> list[Failure]:
    """One ``contained`` failure per bound reference role with values outside its target."""
    failures = []
    for role in lexicon.ENTITY_REFERENCE_ROLES:
        column, target = bound.get(role), _target(role, bound)
        if column is None or target is None or column == target:
            continue
        (outside,) = outside_counts(tools, ref, target, [column])
        if outside:
            failures.append(
                failure(CONTAINED, ref.name, column, role=role, into=target, outside=outside)
            )
    return failures


def outside_counts(
    tools: ProfileTools, ref: RelationRef, target: str, columns: Sequence[str]
) -> list[int]:
    """Rows of each column holding a value ``target`` does not, in batches of statements."""
    if not columns:
        return []
    quoted_target = tools.provider.quote(ref, target)
    values = (
        f"SELECT {_folded(quoted_target)} FROM {tools.provider.relation_sql(ref)}"
        f" WHERE {quoted_target} IS NOT NULL"
    )
    outside: list[int] = []
    for start in range(0, len(columns), CONTAINMENT_BATCH):
        conditions = []
        for name in columns[start : start + CONTAINMENT_BATCH]:
            quoted = tools.provider.quote(ref, name)
            conditions.append(f"{quoted} IS NOT NULL AND {_folded(quoted)} NOT IN ({values})")
        outside += tools.count_where(ref, conditions)[1:]
    return outside


def _target(role: str, bound: Mapping[str, str]) -> str | None:
    return next((bound[h] for h in lexicon.CONTAINED_IN[role] if h in bound), None)


def _folded(quoted: str) -> str:
    return f"LOWER(TRIM(CAST({quoted} AS {TEXT_TYPE})))"
