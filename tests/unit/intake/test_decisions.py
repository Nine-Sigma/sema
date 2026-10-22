"""A review decisions file settles open entries as human choices, with no model call.

Pinned API (``sema.intake.decisions``, imported lazily so a missing module fails each test):

- ``Decision``: frozen pydantic model, ``extra="forbid"``, fields ``stage``, ``subject``,
  ``choice``; ``choice`` is a member of the recorded entry or ``IGNORE = "ignore"``.
- ``load_decisions(path) -> tuple[Decision, ...]``: the file is a YAML list of
  ``{stage, subject, choice}`` mappings.
- ``validate_decisions(decisions, ambiguities) -> tuple[Decision, ...]``: checks each
  decision against the ambiguity list Review presented (read from the ledger's
  ``<stage>_ambiguities`` artifacts) by ``(stage, subject)``. A decision naming no listed
  entry, or a choice that is neither a member nor ``IGNORE``, raises
  ``DecisionRejectedError`` (a ``ValueError``) with ``subject``, ``choice``, and ``members``
  attributes; its message names the subject, the choice, and every member.
- ``apply_structure_decisions(structure, decisions, land, tools, *, provider="staging")
  -> StructureResult``: applies the ``structure`` decisions only. A member choice binds that
  member's binding with ``source: human`` and confidence 1.0, and the entry is no longer
  open; ``IGNORE`` moves the relation to ``ignored`` with slot ``reason: human``.
- ``stated_choices(decisions) -> dict[str, StatedChoice]``: every ``roles`` decision (role,
  reference, and section entries) by subject, ``source: human``, confidence 1.0; ``IGNORE``
  states ``lexicon.NO_REFERENCE``. Structure decisions are not included. The result is
  passed to ``run_roles(..., choices=...)``, which honours section subjects too.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from sema.binding.spec import load_concept_source_spec
from sema.intake.eligibility_text_utils import CODE_LISTS, CONCEPTS
from sema.intake.ledger import Ledger
from sema.intake.lexicon import NO_REFERENCE
from sema.intake.loaders.tabular import land_tabular
from sema.intake.shapes import ignore
from sema.intake.structure import StructureResult
from sema.intake.verify.base import Ambiguity
from sema.intake.verify.roles import REQUIRED_ROLES_BOUND, verify_roles
from sema.intake.verify.roles_utils import DISPLAY_NULL_RATE
from sema.intake.verify.structure import verify_structure
from tests.unit.intake.decisions_env import (
    HUMAN,
    ROLES,
    STRUCTURE,
    UNRESOLVED,
    Decided,
    decided,
    decision,
    decisions,
    flat_list,
    open_entries,
    open_subjects,
    presented,
    write_decisions,
)
from tests.unit.intake.eligibility_env import (
    COLOURS_SQL,
    GROUPED_CODE_TABLE,
    MISSPELLED_PROPERTY,
    by_role,
    copy_without,
    memory_source,
    normalized,
)
from tests.unit.intake.eligibility_env import run_roles as roles_over
from tests.unit.intake.pipeline_env import landed_run, write_ambiguity_lists
from tests.unit.intake.resolve_roles_env import VALUE_TABLES, Synthetic, relation
from tests.unit.intake.shapes.shapes_env import Source
from tests.unit.intake.structure_env import (
    ACRIS,
    ACRIS_SPEC,
    CODE_TABLES,
    copy_code_tables,
    fresh_tools,
    land,
    ref,
    run,
)

pytestmark = pytest.mark.unit

SCHEMA, IGNORED_TABLE = "land_acris", "Country_Codes"
UNRESOLVED_COUNT = 2


@pytest.fixture(scope="module")
def tmp_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("decisions")


@pytest.fixture(scope="module")
def codes(tmp_module: Path) -> Iterator[Source]:
    directory = copy_code_tables(tmp_module / "codes_only" / "csv")
    staging, source = land(tmp_module / "codes", "codes", land_tabular, directory)
    yield source
    staging.close()


@pytest.fixture(scope="module")
def codes_structure(codes: Source, tmp_module: Path) -> StructureResult:
    return run(codes, fresh_tools(codes, tmp_module / "ledger" / "structure"))


@pytest.fixture
def codes_listed(codes_structure: StructureResult, tmp_path: Path) -> list[Ambiguity]:
    ledger = Ledger(tmp_path).run("review")
    write_ambiguity_lists(ledger, codes_structure.ambiguities, ())
    return presented(ledger)


def _decide_structure(source: Source, structure: StructureResult, chosen: Sequence[Any]) -> Any:
    apply = decisions().apply_structure_decisions
    return apply(structure, chosen, source.land, source.tools, provider=source.provider)


def test_decisions_file_loads_as_a_list_of_stage_subject_choice(tmp_path: Path) -> None:
    rows = [(STRUCTURE, "s.a", "concept_list(relation=s.a)"), (ROLES, "s.d.domain.Slant", "ignore")]
    path = write_decisions(tmp_path / "decisions.yaml", rows)

    loaded = decisions().load_decisions(path)

    assert loaded == tuple(decision(*row) for row in rows)


def test_a_choice_outside_the_entrys_members_is_rejected_naming_them(
    codes: Source, codes_listed: list[Ambiguity]
) -> None:
    subject = ref(codes, GROUPED_CODE_TABLE)
    (entry,) = [a for a in codes_listed if a.subject == subject]
    outsider = f"concept_list(relation={ref(codes, IGNORED_TABLE)})"
    assert outsider not in entry.members

    with pytest.raises(decisions().DecisionRejectedError) as raised:
        decisions().validate_decisions([decision(STRUCTURE, subject, outsider)], codes_listed)

    error = raised.value
    assert isinstance(error, ValueError)
    assert (error.subject, error.choice, tuple(error.members)) == (subject, outsider, entry.members)
    assert subject in str(error) and outsider in str(error)
    assert all(member in str(error) for member in entry.members)


@pytest.mark.parametrize("stage", [STRUCTURE, ROLES])
def test_a_decision_naming_no_listed_entry_is_rejected(
    codes: Source, codes_listed: list[Ambiguity], stage: str
) -> None:
    subject = ref(codes, GROUPED_CODE_TABLE) if stage == ROLES else f"{SCHEMA}.not_landed"

    with pytest.raises(decisions().DecisionRejectedError) as raised:
        decisions().validate_decisions([decision(stage, subject, "anything")], codes_listed)

    assert subject in str(raised.value)


def test_member_and_ignore_choices_pass_validation(
    codes: Source, codes_listed: list[Ambiguity]
) -> None:
    grouped = next(a for a in codes_listed if a.subject == ref(codes, GROUPED_CODE_TABLE))
    chosen = [
        decision(STRUCTURE, grouped.subject, flat_list(grouped)),
        decision(STRUCTURE, ref(codes, IGNORED_TABLE), decisions().IGNORE),
    ]

    assert decisions().validate_decisions(chosen, codes_listed) == tuple(chosen)


def test_structure_decision_pins_the_member_binding_as_human(
    codes: Source, codes_structure: StructureResult
) -> None:
    subject = ref(codes, GROUPED_CODE_TABLE)
    (entry,) = [a for a in codes_structure.ambiguities if a.subject == subject]

    made = decision(STRUCTURE, subject, flat_list(entry))

    result = _decide_structure(codes, codes_structure, [made])

    (bound,) = [e for e in result.bound if subject in e.binding.relations]
    assert bound.binding.template == "concept_list"
    assert (bound.source, bound.confidence) == (HUMAN, 1.0)
    assert bound.call_key is None
    assert subject not in {a.subject for a in result.ambiguities}
    others = {ref(codes, name) for name in CODE_TABLES} - {subject}
    assert {a.subject for a in result.ambiguities} == others == open_subjects(
        result.ambiguities, STRUCTURE
    )
    verdict = verify_structure(result, codes.land, codes.tools, provider=codes.provider)
    assert verdict.status == "pass", verdict.failures


def test_structure_ignore_records_reason_human_and_the_verifier_passes(
    codes: Source, codes_structure: StructureResult
) -> None:
    subject = ref(codes, IGNORED_TABLE)

    result = _decide_structure(
        codes, codes_structure, [decision(STRUCTURE, subject, decisions().IGNORE)]
    )

    (ignored,) = [b for b in result.ignored if b.relations == (subject,)]
    assert ignored.template == ignore.ID
    assert ignored.slots[ignore.REASON] == HUMAN
    assert not [e for e in result.bound if subject in e.binding.relations]
    assert subject not in {a.subject for a in result.ambiguities}
    verdict = verify_structure(result, codes.land, codes.tools, provider=codes.provider)
    assert verdict.status == "pass", verdict.failures


def test_stated_choices_carry_source_human_and_confidence_one() -> None:
    chosen = [
        decision(STRUCTURE, "s.a", "concept_list(relation=s.a)"),
        decision(ROLES, "s.b.code", "kelt"),
        decision(ROLES, "s.b.section", CODE_LISTS),
        decision(ROLES, "s.d.domain.Slant", decisions().IGNORE),
    ]

    stated = decisions().stated_choices(chosen)

    choices = {subject: found.choice for subject, found in stated.items()}
    want = {"s.b.code": "kelt", "s.b.section": CODE_LISTS, "s.d.domain.Slant": NO_REFERENCE}
    assert choices == want
    assert {(s.source, s.confidence) for s in stated.values()} == {(HUMAN, 1.0)}


@pytest.mark.parametrize("section", [CODE_LISTS, CONCEPTS])
def test_section_decision_settles_the_concept_list_section(
    tmp_path: Path, section: str
) -> None:
    con, source = memory_source(COLOURS_SQL, "tc", tmp_path)
    try:
        structure = run(source, fresh_tools(source, tmp_path / "structure"))
        subject = f"{ref(source, 'colours')}.section"
        before = roles_over(source, structure, tmp_path / "r0")
        assert subject in open_subjects(before.ambiguities, ROLES)
        stated = decisions().stated_choices([decision(ROLES, subject, section)])

        result = roles_over(source, structure, tmp_path / "r1", choices=stated)
    finally:
        con.close()

    (entry,) = [a for a in result.ambiguities if a.subject == subject]
    assert (entry.choice, entry.source, entry.confidence) == (section, HUMAN, 1.0)
    (bound,) = [e.binding for e in result.structure.bound if e.binding.template == "concept_list"]
    assert bound.section is not None
    assert bound.section.partition(".")[0] == section


@pytest.fixture(scope="module")
def blanks(tmp_module: Path) -> Iterator[Synthetic]:
    built = Synthetic(VALUE_TABLES, tmp_module / "blanks")
    yield built
    built.close()


def _decided_roles(built: Synthetic, root: Path, made: Sequence[Any]) -> tuple[Any, Any]:
    """Roles over ``made`` as stated choices, and the Roles verdict re-proving ``eta``."""
    stated = decisions().stated_choices(made)
    result = roles_over(built.source, built.structure, root / "roles", choices=stated)
    tools = fresh_tools(built.source, root / "verify", stage=ROLES, structure=built.structure)
    return result, verify_roles(result, tools, spec=None, relations=(relation("eta"),))


def test_ignore_on_a_required_column_role_closes_it_unbound_and_fails_the_verifier(
    blanks: Synthetic, tmp_path: Path
) -> None:
    eta = relation("eta")
    subject = f"{eta}.display"
    assert subject in open_subjects(blanks.roles.ambiguities, ROLES)
    (before,) = [a for a in blanks.roles.ambiguities if a.subject == subject]
    assert NO_REFERENCE not in before.members
    made = [decision(ROLES, subject, decisions().IGNORE)]

    result, verdict = _decided_roles(blanks, tmp_path, made)

    (entry,) = [a for a in result.ambiguities if a.subject == subject]
    assert (entry.choice, entry.source) == (None, HUMAN)
    assert subject not in open_subjects(result.ambiguities, ROLES)
    display = by_role(result.eligibilities, eta)["display"]
    assert (display.choice, display.source) == (None, HUMAN)
    failing = [f for f in verdict.failures if f.check == REQUIRED_ROLES_BOUND]
    assert failing, verdict.failures
    named = {(f.relation, f.evidence.get("role"), f.evidence.get("source")) for f in failing}
    assert (eta, "display", HUMAN) in named


def test_human_display_choice_failing_a_value_check_is_listed_with_source_human(
    blanks: Synthetic, tmp_path: Path
) -> None:
    eta = relation("eta")
    subject = f"{eta}.display"

    result, verdict = _decided_roles(blanks, tmp_path, [decision(ROLES, subject, "wim")])

    assert by_role(result.eligibilities, eta)["display"].source == HUMAN
    (found,) = [f for f in verdict.failures if f.check == DISPLAY_NULL_RATE]
    assert (found.relation, found.column, found.evidence["source"]) == (eta, "wim", HUMAN)
    (listed,) = [a for a in verdict.ambiguities if a.subject == subject]
    assert (listed.choice, listed.source) == ("wim", HUMAN)


@pytest.fixture(scope="module")
def a10(tmp_module: Path) -> Iterator[Decided]:
    with landed_run(ACRIS, tmp_module / "a10", SCHEMA) as landed:
        yield decided(landed, tmp_module / "a10")


@pytest.fixture(scope="module")
def dropped(tmp_module: Path) -> Iterator[Decided]:
    files = copy_without(tmp_module / "nodcc" / "files", GROUPED_CODE_TABLE)
    with landed_run(files, tmp_module / "nodcc", SCHEMA) as landed:
        yield decided(landed, tmp_module / "nodcc")


def test_role_decision_enters_run_roles_as_a_stated_human_choice(a10: Decided) -> None:
    subject = f"{a10.landed.qualified('Property_Types_Codes')}.code"
    (before,) = [a for a in a10.first_roles.ambiguities if a.subject == subject]
    assert (before.choice, before.source) == (None, None)
    (made,) = [d for d in a10.rounds[1] if d.subject == subject]

    (entry,) = [a for a in a10.roles.ambiguities if a.subject == subject]
    (role,) = [e for e in a10.roles.eligibilities if f"{e.relation}.{e.role}" == subject]

    assert (entry.choice, entry.source, entry.confidence) == (made.choice, HUMAN, 1.0)
    assert (role.choice, role.source, role.confidence) == (made.choice, HUMAN, 1.0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "no zero-call path derives a dictionary's self domain and the hinted path does not "
    "read a hint's `domains`; S4 honours hint domains as `source: human`"
))
def test_a10_no_hint_run_assembles_after_decisions_with_zero_model_calls(a10: Decided) -> None:
    first, second = a10.rounds
    code_tables = {a10.landed.qualified(name) for name in CODE_TABLES}
    # Since the described-data rule (2026-10-21) code binds the code tables, so the
    # structure round has nothing to decide.
    assert first == ()
    code_bound = {
        name
        for entry in a10.structure.bound
        if (entry.binding.template, entry.source) == ("concept_list", "code")
        for name in entry.binding.relations
    }
    assert code_bound == code_tables
    assert second and {d.stage for d in second} == {ROLES}

    still_open = [(a.subject, a.members) for a in open_entries(a10.roles.ambiguities, ROLES)]
    assert a10.spec is not None, still_open
    a10.invoke.assert_not_called()
    assert a10.landed.model_calls() == []
    assert a10.landed.record().model_calls == 0
    assert a10.structure.ambiguities == ()
    roles_tools = a10.landed.tools(ROLES, a10.structure)
    verdict = verify_roles(a10.roles, roles_tools, spec=a10.spec)
    assert verdict.status == "pass", verdict.failures
    assert normalized(a10.spec) == normalized(load_concept_source_spec(ACRIS_SPEC))


def test_ignore_on_an_unresolved_reference_drops_the_domain(dropped: Decided) -> None:
    ignored = [d for d in dropped.rounds[1] if d.choice == decisions().IGNORE]
    subjects = {d.subject for d in ignored}
    assert len(subjects) == UNRESOLVED_COUNT
    assert all(MISSPELLED_PROPERTY in subject for subject in subjects)

    settled = [a for a in dropped.roles.ambiguities if a.subject in subjects]
    assert {(a.choice, a.source, a.confidence) for a in settled} == {(NO_REFERENCE, HUMAN, 1.0)}
    assert not [a for a in open_entries(dropped.roles.ambiguities, ROLES) if UNRESOLVED in a.hints]
    assert dropped.spec is not None
    assert dropped.spec.entity_model is not None
    domains = [entity.domains for entity in dropped.spec.entity_model.entities]
    assert all(MISSPELLED_PROPERTY not in found for found in domains)
    dropped.invoke.assert_not_called()
