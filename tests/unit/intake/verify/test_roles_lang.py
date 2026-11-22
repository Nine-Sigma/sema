"""``lang_tags`` runs on every landing dialect (S5 decision 2, task 2.6).

Pinned over one invented code list whose ``lang`` role (``LifecycleRoles.lang``, so
``CodeListRoles`` carries it) is bound to a column of language tags:

- On the warehouse provider (a DuckDB stand-in for the warehouse) ``verify_roles`` raises
  nothing; DuckDB has no ``RLIKE``.
- Every tag valid: no ``lang_tags`` failure.
- One invalid tag: a ``lang_tags`` failure naming the relation and column with ``rows`` 1,
  or an ``unverifiable`` entry naming ``lang_tags`` and a reason (decision 2 allows either
  on the stand-in).
- On the staging provider the check is unchanged (regression guard).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec
from sema.intake.roles.eligibility import assemble_spec
from sema.intake.roles.eligibility_utils import RunDescriptor
from sema.intake.land.loaders.tabular import land_tabular
from sema.intake.verify.roles import UNVERIFIABLE, verify_roles
from sema.intake.verify.roles_utils import LANG_TAGS
from tests.unit.intake.catalog_env import write_descriptor
from tests.unit.intake.role_entries_env import concept_list, pinned, roles_of, source_of
from tests.unit.intake.structure.shapes.shapes_env import Source
from tests.unit.intake.structure_env import land
from tests.unit.intake.verify.verify_env import failures, tools_for

pytestmark = pytest.mark.unit

SCHEMA, TABLE = "lx", "phrases"
HEADER = ("phrase_code", "phrase_label", "phrase_lang")
LANG = "phrase_lang"
VALID = ("en", "en-GB", "fr", "pt-BR")
INVALID = "english!"


def _rows(tags: tuple[str, ...]) -> list[tuple[str, str, str]]:
    return [(f"P{i:02d}", f"Phrase number {i}", tag) for i, tag in enumerate(tags, 1)]


@contextmanager
def _warehouse(root: Path, tags: tuple[str, ...]) -> Iterator[Source]:
    values = ", ".join(f"('{code}', '{label}', '{tag}')" for code, label, tag in _rows(tags))
    statements = (
        f"CREATE SCHEMA {SCHEMA}",
        f"CREATE TABLE {SCHEMA}.{TABLE} AS SELECT * FROM (VALUES {values}) t({', '.join(HEADER)})",
    )
    con, source = source_of(statements, SCHEMA, root)
    try:
        yield source
    finally:
        con.close()


@contextmanager
def _staging(root: Path, tags: tuple[str, ...]) -> Iterator[Source]:
    directory = root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{TABLE}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(_rows(tags))
    staging, source = land(root, SCHEMA, land_tabular, directory)
    try:
        yield source
    finally:
        staging.close()


def _verdict(source: Source, root: Path) -> tuple[Any, str]:
    """The roles verdict over the assembled spec with ``lang`` bound to the tag column."""
    (relation,) = [f"{r.schema}.{r.name}" for r in source.land.relations]
    structure = pinned(source, [concept_list(relation, f"code_lists.{TABLE}")], root / "pin")
    roles = roles_of(source, structure, root / "roles")
    raw = yaml.safe_load(write_descriptor(root / "descriptor").read_text())
    spec = assemble_spec(RunDescriptor.model_validate(raw), roles)
    assert spec is not None, "the code list did not assemble"
    dumped = spec.model_dump(mode="json", by_alias=True)
    dumped["code_lists"][0]["roles"]["lang"] = LANG
    edited = ConceptSourceSpec.model_validate(dumped)
    tools = tools_for(source, root / "verify", stage="roles", structure=roles.structure)
    return verify_roles(roles, tools, spec=edited), relation


def _lang_failures(verdict: Any) -> list[Any]:
    return failures(verdict, LANG_TAGS)


def _unverifiable_lang(verdict: Any) -> list[Any]:
    return [
        a for a in verdict.ambiguities
        if UNVERIFIABLE in a.hints and LANG_TAGS in " ".join([a.subject, *a.hints])
        and a.reason
    ]


def test_valid_tags_pass_on_the_warehouse_stand_in(tmp_path: Path) -> None:
    with _warehouse(tmp_path, VALID) as source:
        verdict, _ = _verdict(source, tmp_path)

    assert _lang_failures(verdict) == [], verdict.failures


def test_an_invalid_tag_fails_or_is_unverifiable_on_the_warehouse_stand_in(
    tmp_path: Path,
) -> None:
    with _warehouse(tmp_path, (*VALID, INVALID)) as source:
        verdict, relation = _verdict(source, tmp_path)

    found = [
        (f.relation, f.column, f.evidence.get("rows")) for f in _lang_failures(verdict)
    ]
    assert found == [(relation, LANG, 1)] or _unverifiable_lang(verdict), (
        found, verdict.ambiguities,
    )


def test_an_invalid_tag_fails_on_staging(tmp_path: Path) -> None:
    """Regression guard: the staging dialect's check passes on HEAD."""
    with _staging(tmp_path, (*VALID, INVALID)) as source:
        verdict, relation = _verdict(source, tmp_path)

    found = [(f.relation, f.column, f.evidence.get("rows")) for f in _lang_failures(verdict)]
    assert found == [(relation, LANG, 1)], verdict.failures
