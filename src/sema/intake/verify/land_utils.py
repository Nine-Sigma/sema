"""Land verifier helpers: header predicates, column listings, and recorded types."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from sema.binding.spec_sections import RelationRef
from sema.intake.land.landing import LOCATOR_COLUMN, LandedRelation
from sema.intake.land.loaders.tabular_utils import EMPTY_COLUMN_PREFIX, POSITIONAL_COLUMN_PREFIX
from sema.intake.land.profile import ProfileTools
from sema.intake.land.profile_utils import TEXT_TYPE, CatalogColumn, RelationProfile, sql_string

HEADERLESS = "headerless"
POSITIONAL = "positional"
EMPTY_NAMED = "empty_named"
ALL_NULL = "all_null"
GENERATED_PREFIXES = (POSITIONAL_COLUMN_PREFIX, EMPTY_COLUMN_PREFIX)
MISSING = "missing"

Evidence = dict[str, Any]


def header_failures(
    landed: LandedRelation, names: Sequence[str], tools: ProfileTools, ref: RelationRef
) -> tuple[list[tuple[str | None, Evidence]], dict[str, str]]:
    """Failures read from the header alone, and per header value the body-echo condition."""
    if landed.header_row is None:
        return [], {}
    header = [n for n in names if n != LOCATOR_COLUMN and not n.startswith(GENERATED_PREFIXES)]
    if not header:
        return [], {}
    found: list[tuple[str | None, Evidence]] = []
    if all(_is_number(name) for name in header):
        found.append((None, {"header_row": landed.header_row, "numeric": len(header)}))
    counts = Counter(name.casefold() for name in header)
    found += [
        (name, {"header_row": landed.header_row, "duplicate": counts[name.casefold()]})
        for name in header
        if counts[name.casefold()] > 1
    ]
    conditions = {
        name: f"CAST({tools.provider.quote(ref, name)} AS {TEXT_TYPE}) = {sql_string(name)}"
        for name in header
    }
    return found, conditions


def locator_condition(tools: ProfileTools, ref: RelationRef) -> str:
    quoted = tools.provider.quote(ref, LOCATOR_COLUMN)
    return f"({quoted} IS NULL OR CAST({quoted} AS {TEXT_TYPE}) = '')"


def listed_columns(
    landed: LandedRelation, described: RelationProfile, headerless: bool
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Positional, empty-named, and all-null columns with one hint each, or None."""
    members: list[str] = []
    hints = [HEADERLESS] if headerless else []
    for column in described.columns:
        kind = _listing_kind(column.name, column.null_rate, described.row_count)
        if kind is not None:
            members.append(column.name)
            hints.append(f"{kind}: {column.name}")
    return (tuple(members), tuple(hints)) if members else None


def type_failures(
    landed: LandedRelation, catalog: Sequence[CatalogColumn]
) -> list[tuple[str, Evidence]]:
    """Columns whose recorded type is not the type the relation holds."""
    recorded = dict(landed.columns)
    actual = {column.name: column.type for column in catalog}
    found = []
    for name in dict.fromkeys([*recorded, *actual]):
        kinds = recorded.get(name, MISSING), actual.get(name, MISSING)
        if kinds[0].upper() != kinds[1].upper():
            found.append((name, {"recorded": kinds[0], "landed": kinds[1]}))
    return found


def _listing_kind(name: str, null_rate: float, row_count: int) -> str | None:
    if name == LOCATOR_COLUMN:
        return None
    if name.startswith(POSITIONAL_COLUMN_PREFIX):
        return POSITIONAL
    if name.startswith(EMPTY_COLUMN_PREFIX):
        return EMPTY_NAMED
    if row_count > 0 and null_rate == 1.0:
        return ALL_NULL
    return None


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
