"""Roles verifier helpers: the spec's bound sections and each role's value predicate.

Every value check is a count or a join rate over the bound relation, so it runs under
``metadata_only`` without reading a value. A display column's empty rate is a finding,
reported beside the verdict and never a failure (S5 decision 18). Role names come from
the spec role models.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import RelationRef, StatusRole
from sema.intake.roles import lexicon
from sema.intake.roles.eligibility_spec_utils import VALUE_SETS
from sema.intake.roles.eligibility_text_utils import CODE_LISTS, CONCEPTS, STAGE
from sema.intake.roles.eligibility_utils import (
    ARRAY_TYPE_MARKERS,
    DISPLAY_NULL_RATE,
    RoleEligibility,
    scope_of,
    subject_of,
)
from sema.intake.land.profile import ProfileTools
from sema.intake.land.profile_utils import TEXT_TYPE, RelationProfile, sql_string
from sema.intake.structure.shapes import entity_model
from sema.intake.structure.shapes.base import JOIN_RESOLVE_FLOOR, is_text
from sema.intake.structure.shapes.base_utils import failure
from sema.intake.structure.structure_hint_utils import KEY_ROLE, SECTION_SEPARATOR
from sema.intake.verify.base import Ambiguity, Failure, Finding

IDENTITY_UNIQUE = f"{KEY_ROLE}_unique"
CODE_UNIQUE = "code_unique"
SYNONYMS_SHAPE = "synonyms_shape"
STATUS_VALUES = "status_values"
VALID_DATES = "valid_dates"
REPLACED_BY_RESOLVES = "replaced_by_resolves"
LANG_TAGS = "lang_tags"
UNVERIFIABLE = "unverifiable"
UNVERIFIABLE_STATUS = "status values are unverifiable: {}"
VALUE_CHECKS = (
    IDENTITY_UNIQUE,
    CODE_UNIQUE,
    SYNONYMS_SHAPE,
    STATUS_VALUES,
    VALID_DATES,
    REPLACED_BY_RESOLVES,
    LANG_TAGS,
)
TIMESTAMP_TYPE = "TIMESTAMP"
# The lexicon's date roles in settling order: the start, then the end.
DATE_ROLES = tuple(entry.name for entry in lexicon.ROLES if entry.predicate == "date")
BCP47_PATTERN = "^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*$"
# A concept key and its name may share a column that is the group's constant key.
GROUP_KEY_ROLES = frozenset({KEY_ROLE, "name"})


@dataclass(frozen=True)
class BoundSection:
    """``vocabulary`` scopes code uniqueness; ``identity`` is what ``replaced_by`` names."""

    path: str
    relation: RelationRef
    roles: BaseModel
    filter: str | None = None
    group_by: str | None = None
    vocabulary: str | None = None
    identity: str | None = None


def bound_sections(spec: ConceptSourceSpec) -> list[BoundSection]:
    found = []
    if spec.concepts is not None:
        concepts = spec.concepts
        identity = getattr(concepts.roles, KEY_ROLE)
        found.append(BoundSection(
            CONCEPTS, concepts.relation, concepts.roles, concepts.filter,
            group_by=concepts.group_by, identity=identity,
        ))
    if spec.value_sets is not None:
        values = spec.value_sets
        vocabulary = getattr(values.roles, KEY_ROLE)
        found.append(BoundSection(
            VALUE_SETS, values.relation, values.roles,
            vocabulary=vocabulary, identity=values.roles.code,
        ))
    for entry in spec.code_lists or ():
        found.append(BoundSection(
            f"{CODE_LISTS}{SECTION_SEPARATOR}{entry.vocabulary_id}", entry.relation,
            entry.roles, entry.filter, vocabulary=entry.roles.vocabulary_id,
            identity=entry.roles.code,
        ))
    model = spec.entity_model
    if model is None:
        return found
    entries = model.entities or ()
    relations = [model.relation] if model.relation else [entry.relation for entry in entries]
    return found + [
        BoundSection(entity_model.SECTION, relation, model.roles, model.filter)
        for relation in relations
    ]


ValueCheck = Callable[[BoundSection, dict[str, str], ProfileTools], list[Failure]]


def role_columns(roles: BaseModel) -> dict[str, str]:
    """Each role bound to one column; a status role binds its column."""
    found = {}
    for name in type(roles).model_fields:
        value = getattr(roles, name)
        if isinstance(value, StatusRole):
            found[name] = value.column
        elif isinstance(value, str):
            found[name] = value
    return found


def constant_group_key(section: BoundSection, column: str, tools: ProfileTools) -> bool:
    """The column is the section's ``group_by`` key and holds one value on every row."""
    if section.group_by != column:
        return False
    ref, where = section.relation, section.filter
    is_null = f"{tools.provider.quote(ref, column)} IS NULL"
    rows, nulls = tools.count_where(ref, [is_null], where=where)
    return rows > 0 and not nulls and tools.count_distinct(ref, [column], where=where) == 1


def value_failures(
    section: BoundSection, profile: RelationProfile, tools: ProfileTools
) -> list[Failure]:
    present = {column.name for column in profile.columns}
    roles = role_columns(section.roles)
    bound = {role: column for role, column in roles.items() if column in present}
    checks: tuple[ValueCheck, ...] = (
        _identity_unique, _code_unique, _status_values, _valid_dates, _replaced_by, _lang_tags,
    )
    failures = [found for check in checks for found in check(section, bound, tools)]
    return failures + _synonyms_shape(section, bound, profile)


def value_findings(
    section: BoundSection, profile: RelationProfile, tools: ProfileTools
) -> list[Finding]:
    present = {column.name for column in profile.columns}
    display = role_columns(section.roles).get("display")
    return _display_nulls(section, display, tools) if display in present else []


def _text(tools: ProfileTools, ref: RelationRef, column: str) -> str:
    return f"CAST({tools.provider.quote(ref, column)} AS {TEXT_TYPE})"


def _count(section: BoundSection, tools: ProfileTools, condition: str) -> tuple[int, int]:
    rows, matched = tools.count_where(section.relation, [condition], where=section.filter)
    return rows, matched


def _identity_unique(
    section: BoundSection, bound: dict[str, str], tools: ProfileTools
) -> list[Failure]:
    identity, group_by = bound.get(KEY_ROLE), section.group_by
    if section.path != CONCEPTS or identity is None or group_by is None:
        return []
    ref, where = section.relation, section.filter
    counts = [tools.count_distinct(ref, cols, where=where)
              for cols in ([group_by], [identity], [group_by, identity])]
    if len(set(counts)) == 1:
        return []
    groups, identities, pairs = counts
    return [failure(IDENTITY_UNIQUE, ref.name, identity, groups=groups,
                    identities=identities, pairs=pairs)]


def _code_unique(
    section: BoundSection, bound: dict[str, str], tools: ProfileTools
) -> list[Failure]:
    code = bound.get("code")
    if code is None:
        return []
    columns = [c for c in (section.vocabulary, code) if c is not None]
    ref = section.relation
    present = " AND ".join(f"{tools.provider.quote(ref, c)} IS NOT NULL" for c in columns)
    _, keyed = _count(section, tools, present)
    distinct = tools.count_distinct(ref, columns, where=section.filter)
    if distinct >= keyed:
        return []
    return [failure(CODE_UNIQUE, ref.name, code, rows=keyed, distinct=distinct)]


def _display_nulls(
    section: BoundSection, display: str | None, tools: ProfileTools
) -> list[Finding]:
    if display is None:
        return []
    ref = section.relation
    quoted, text = tools.provider.quote(ref, display), _text(tools, ref, display)
    rows, empty = _count(section, tools, f"({quoted} IS NULL OR TRIM({text}) = '')")
    rate = empty / rows if rows else 0.0
    if rate < lexicon.DISPLAY_NULL_RATE_FLOOR:
        return []
    return [Finding(name=DISPLAY_NULL_RATE, subject=f"{ref.name}.{display}", value=rate)]


def _status_values(
    section: BoundSection, bound: dict[str, str], tools: ProfileTools
) -> list[Failure]:
    status = getattr(section.roles, lexicon.STATUS_ROLE, None)
    if not isinstance(status, StatusRole) or lexicon.STATUS_ROLE not in bound:
        return []
    ref = section.relation
    declared = [*status.active_values, *status.retired_values]
    listed = ", ".join(sql_string(value) for value in declared) or "NULL"
    quoted = tools.provider.quote(ref, status.column)
    condition = f"{quoted} IS NOT NULL AND {_text(tools, ref, status.column)} NOT IN ({listed})"
    _, undeclared = _count(section, tools, condition)
    if not undeclared:
        return []
    return [failure(STATUS_VALUES, ref.name, status.column, undeclared_rows=undeclared)]


def _valid_dates(
    section: BoundSection, bound: dict[str, str], tools: ProfileTools
) -> list[Failure]:
    ref = section.relation
    dated = {role: bound[role] for role in DATE_ROLES if role in bound}
    failures = []
    for role, column in dated.items():
        quoted, parsed = tools.provider.quote(ref, column), _parsed(tools, ref, column)
        _, bad = _count(section, tools, f"{quoted} IS NOT NULL AND {parsed} IS NULL")
        if bad:
            failures.append(failure(VALID_DATES, ref.name, column, role=role, unparseable=bad))
    if failures or len(dated) < len(DATE_ROLES):
        return failures
    start, end = (dated[role] for role in DATE_ROLES)
    condition = f"{_parsed(tools, ref, start)} > {_parsed(tools, ref, end)}"
    _, inverted = _count(section, tools, condition)
    return [failure(VALID_DATES, ref.name, start, inverted_rows=inverted)] if inverted else []


def _parsed(tools: ProfileTools, ref: RelationRef, column: str) -> str:
    return f"TRY_CAST({_text(tools, ref, column)} AS {TIMESTAMP_TYPE})"


def _replaced_by(
    section: BoundSection, bound: dict[str, str], tools: ProfileTools
) -> list[Failure]:
    replaced_by, identity = bound.get("replaced_by"), section.identity
    if replaced_by is None or identity is None:
        return []
    ref = section.relation
    _, named = _count(section, tools, f"{tools.provider.quote(ref, replaced_by)} IS NOT NULL")
    if not named:
        return []
    rate = tools.check_join(ref, replaced_by, ref, identity).resolve_a_to_b
    if rate >= JOIN_RESOLVE_FLOOR:
        return []
    return [failure(REPLACED_BY_RESOLVES, ref.name, replaced_by, into=identity,
                    resolve_rate=rate, floor=JOIN_RESOLVE_FLOOR)]


def _lang_tags(
    section: BoundSection, bound: dict[str, str], tools: ProfileTools
) -> list[Failure]:
    lang = bound.get("lang")
    if lang is None:
        return []
    ref = section.relation
    text, pattern = _text(tools, ref, lang), sql_string(BCP47_PATTERN)
    # regexp_extract shares one signature on DuckDB and Databricks; RLIKE is Databricks-only.
    # The pattern never matches the empty string, so a non-empty extract means a match.
    matched = f"regexp_extract({text}, {pattern}, 0) <> ''"
    quoted = tools.provider.quote(ref, lang)
    _, bad = _count(section, tools, f"{quoted} IS NOT NULL AND NOT ({matched})")
    return [failure(LANG_TAGS, ref.name, lang, rows=bad)] if bad else []


def _synonyms_shape(
    section: BoundSection, bound: dict[str, str], profile: RelationProfile
) -> list[Failure]:
    synonyms = bound.get("synonyms")
    column = next((c for c in profile.columns if c.name == synonyms), None)
    if column is None:
        return []
    kind = column.type.upper()
    if is_text(column) or any(marker in kind for marker in ARRAY_TYPE_MARKERS):
        return []
    return [failure(SYNONYMS_SHAPE, section.relation.name, column.name, type=column.type)]


def marked_unverifiable(
    listed: Sequence[Ambiguity],
    eligibilities: Iterable[RoleEligibility],
    unread: Mapping[str, str],
) -> list[Ambiguity]:
    """Each bound status whose values the policy kept unread carries the mark and the reason
    on its entry, settled; code settled it with no entry, so one is added. Idempotent."""
    marks = {
        subject_of(e): (e, UNVERIFIABLE_STATUS.format(unread[scope_of(e.relation, e.section)]))
        for e in eligibilities
        if e.role == lexicon.STATUS_ROLE and e.subject is None and e.column is not None
        and scope_of(e.relation, e.section) in unread
    }
    found = [_marked(a, marks[a.subject][1]) if a.subject in marks else a for a in listed]
    present = {a.subject for a in listed}
    found += [
        Ambiguity(
            stage=STAGE, subject=subject, members=(str(e.column),), hints=(UNVERIFIABLE,),
            choice=e.column, confidence=e.confidence, source=e.source or "code", reason=reason,
        )
        for subject, (e, reason) in marks.items()
        if subject not in present
    ]
    return found


def _marked(entry: Ambiguity, reason: str) -> Ambiguity:
    hints = entry.hints if UNVERIFIABLE in entry.hints else (*entry.hints, UNVERIFIABLE)
    if entry.reason is None or reason in entry.reason:
        joined = entry.reason or reason
    else:
        joined = f"{entry.reason}; {reason}"
    return entry.model_copy(update={"hints": hints, "reason": joined})
