"""Lifecycle roles: status evaluation, valid dates, and sidecar retirement rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sema.binding.importer_utils import Record, as_text, first_text
from sema.binding.models import Retirement
from sema.binding.spec_sections import LifecycleRoles, StatusRole


@dataclass(frozen=True)
class Lifecycle:
    retired: bool
    replaced_by: str | None
    valid_from: str | None
    valid_to: str | None


def lifecycle_of(rows: list[Record], roles: LifecycleRoles, relation: str) -> Lifecycle:
    status = row_status(rows, roles.status, relation) if roles.status else None
    valid_from, valid_to = valid_dates(rows, roles)
    return Lifecycle(
        retired=is_retired(status, roles.status),
        replaced_by=first_text(rows, roles.replaced_by),
        valid_from=valid_from,
        valid_to=valid_to,
    )


def lifecycle_columns(roles: LifecycleRoles) -> list[str | None]:
    status = roles.status.column if roles.status else None
    return [status, roles.valid_from, roles.valid_to, roles.replaced_by, roles.lang]


def row_status(rows: list[Record], role: StatusRole, relation: str) -> str | None:
    """The group's status value; a null status is active, an unlisted one fails."""
    known = {*role.active_values, *role.retired_values}
    for row in rows:
        value = as_text(row[role.column])
        if value is not None and value not in known:
            raise ValueError(
                f"relation {relation} column {role.column} has status value "
                f"{value!r}, which is in neither active_values nor retired_values"
            )
    return first_text(rows, role.column)


def is_retired(status: str | None, role: StatusRole | None) -> bool:
    return role is not None and status is not None and status in role.retired_values


def valid_dates(rows: list[Record], roles: LifecycleRoles) -> tuple[str | None, str | None]:
    return first_text(rows, roles.valid_from), first_text(rows, roles.valid_to)


def retirement_row(
    kind: Literal["term", "concept"],
    vocabulary: str | None,
    code: str,
    lifecycle: Lifecycle,
) -> Retirement:
    return Retirement(
        kind=kind,
        vocabulary=vocabulary,
        code=code,
        replaced_by=lifecycle.replaced_by,
        valid_from=lifecycle.valid_from,
        valid_to=lifecycle.valid_to,
    )
