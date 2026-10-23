"""`sema target review <run>` prints a run's open entries (S5 decisions 14 and 19).

Pinned contract (generality-gate spec, "Open entries have a human-readable view"):

- The command reads ``<root>/<run>/artifacts/structure_ambiguities.json`` and
  ``roles_ambiguities.json`` as plain JSON, not through ``Ambiguity``, and prints one line
  per open entry (``choice`` and ``source`` both null; a human ``none`` is closed):
  structure entries in ledger order, then roles entries. A settled entry prints nothing.
- Each line carries the entry's ``stage``, ``subject``, every member in the writer's rank
  (the reader never re-ranks), every exclusion fact, every hint (lexicon and embedding
  hits), and the ``reason`` when it is not null.
- Keys later tasks add print when present and are ignored when absent: ``agreement``
  (decision 4) on its entry's line, ``findings`` after the member each belongs to, and
  entries carrying ``cause`` (decision 14) after every other open entry, one line each
  with the cause and the subject.
- Text only, read-only: no JSON, no provider call, the registry record unchanged.
- A run with no open entry and an assembled ``artifacts/spec.json`` prints that path and
  nothing else; with no open entry and no spec (decision 14 forbids it at Review) it exits
  non-zero naming the run id and ``no spec``. A run id with no ledger directory exits
  non-zero naming the run id.

"Line" means a non-blank output line. Every name is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sema.intake.registry import Registry
from tests.unit.intake.review_reader_env import (
    LIST_MEMBER,
    MISSING_RUN_ID,
    REGISTRY_FILE,
    ROLES,
    RUN_ID,
    STRUCTURE,
    VALUES_MEMBER,
    alpha_structure,
    assert_in_order,
    assert_line_shows,
    assert_settled_line_shows,
    assert_succeeded,
    beta_structure,
    entry,
    forbid_provider_calls,
    gamma_role,
    human_none,
    invoke,
    invoke_settled,
    lexicon_settled_role,
    line_naming,
    marks_settled,
    model_settled_structure,
    printed_lines,
    seeded_root,
    settled_entry,
    settled_structure,
    write_run,
)

pytestmark = pytest.mark.unit

AGREEMENT = "1/2"
FINDING = "denormalized group ratio 0.4"
REQUIRED_NONE_CAUSE = "required role label settled none"
DIVERGENT_CAUSE = "per-file dictionaries map the role code differently"
SPEC_BODY = {"concept_source": {"spec_version": 2}}
NO_SPEC = "no spec"
HUMAN_NONE_SUBJECT = "zeta_table.label"
AGREED, CONFIDENCE = "2/2", 0.83


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return seeded_root(tmp_path)


def test_each_open_entry_prints_one_line_structure_first_settled_absent(root: Path) -> None:
    opened = [alpha_structure(), beta_structure(), gamma_role()]
    write_run(
        root, [opened[0], opened[1]], [opened[2]], structure_settled=[settled_structure()]
    )

    result = invoke(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == len(opened), lines
    for line, shown in zip(lines, opened, strict=True):
        assert_line_shows(line, shown)
    assert not [line for line in lines if "settled_table" in line], lines


def test_agreement_prints_on_its_entry_and_its_absence_does_not_fail(root: Path) -> None:
    agreed = {**alpha_structure(), "agreement": AGREEMENT}
    write_run(root, [agreed], [gamma_role()])

    result = invoke(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 2, lines
    assert AGREEMENT in line_naming(lines, agreed["subject"])
    assert_line_shows(line_naming(lines, gamma_role()["subject"]), gamma_role())


def test_a_finding_prints_after_the_member_it_belongs_to(root: Path) -> None:
    found = {**beta_structure(), "findings": {LIST_MEMBER: [FINDING]}}
    write_run(root, [found], [])

    result = invoke(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 1, lines
    assert_line_shows(lines[0], found)
    assert_in_order(lines[0], [LIST_MEMBER, FINDING])


def test_no_spec_causes_print_after_every_other_open_entry(root: Path) -> None:
    required_none = entry(
        STRUCTURE, "delta_table", (LIST_MEMBER, VALUES_MEMBER), cause=REQUIRED_NONE_CAUSE
    )
    divergent = entry(ROLES, "epsilon_table.code", ("concept_code",), cause=DIVERGENT_CAUSE)
    write_run(root, [required_none, alpha_structure()], [divergent, gamma_role()])

    result = invoke(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 4, lines
    assert_line_shows(lines[0], alpha_structure())
    assert_line_shows(lines[1], gamma_role())
    for caused in (required_none, divergent):
        line = line_naming(lines[2:], caused["subject"])
        assert caused["cause"] in line, line


def test_reading_makes_no_call_and_leaves_the_registry_record_unchanged(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_run(root, [alpha_structure(), beta_structure()], [gamma_role()])
    registry = Registry(root / REGISTRY_FILE)
    before = registry.get(RUN_ID)
    forbid_provider_calls(monkeypatch)

    result = invoke(root)

    assert_succeeded(result)
    after = registry.get(RUN_ID)
    counters = (after.model_calls, after.tokens, after.cost)
    assert counters == (before.model_calls, before.tokens, before.cost)
    assert after == before


def test_output_is_text_not_json(root: Path) -> None:
    write_run(root, [alpha_structure()], [gamma_role()])

    result = invoke(root)

    assert_succeeded(result)
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)
    assert not [line for line in result.output.splitlines() if line.lstrip().startswith("{")]


def test_a_run_with_no_open_entry_prints_only_its_spec_path(root: Path) -> None:
    spec_path = write_run(root, [], [], spec=SPEC_BODY)

    result = invoke(root)

    assert_succeeded(result)
    assert spec_path.is_file()
    assert printed_lines(result) == [str(spec_path)]


def test_a_run_without_a_ledger_exits_non_zero_naming_the_run(root: Path) -> None:
    result = invoke(root, MISSING_RUN_ID)

    assert result.exit_code != 0
    assert "No such command" not in result.output, result.output
    assert MISSING_RUN_ID in result.output, result.output


def test_a_run_with_no_open_entry_and_no_spec_exits_non_zero_naming_the_run(
    root: Path,
) -> None:
    spec_path = write_run(root, [], [], structure_settled=[settled_structure()])

    result = invoke(root)

    assert not spec_path.exists()
    assert result.exit_code != 0, result.output
    assert RUN_ID in result.output, result.output
    assert NO_SPEC in result.output.lower(), result.output


def test_a_role_a_human_settled_none_is_closed_and_not_printed(root: Path) -> None:
    closed = human_none(ROLES, HUMAN_NONE_SUBJECT, ("concept_label", "none"))
    write_run(root, [], [closed, gamma_role()])

    result = invoke(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 1, lines
    assert_line_shows(lines[0], gamma_role())
    assert HUMAN_NONE_SUBJECT not in result.output, result.output


def _open_and_settled(root: Path) -> None:
    write_run(
        root,
        [beta_structure()],
        [lexicon_settled_role(), gamma_role()],
        structure_settled=[model_settled_structure()],
    )


def test_settled_prints_settled_entries_after_open_ones_marked_with_their_source(
    root: Path,
) -> None:
    _open_and_settled(root)

    result = invoke_settled(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 4, lines
    assert_line_shows(lines[0], beta_structure())
    assert_line_shows(lines[1], gamma_role())
    assert not [line for line in lines[:2] if marks_settled(line)], lines
    assert_settled_line_shows(lines[2], model_settled_structure())
    assert_settled_line_shows(lines[3], lexicon_settled_role())


def test_settled_prints_the_confidence_and_agreement_when_present(root: Path) -> None:
    agreed = settled_entry(
        ROLES, "iota_table.i_code", ("concept_code", "concept_label"), source="model",
        choice="concept_label", confidence=CONFIDENCE, agreement=AGREED,
    )
    write_run(root, [beta_structure()], [agreed])

    result = invoke_settled(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 2, lines
    assert_settled_line_shows(lines[1], agreed)


def test_a_human_none_entry_is_settled_and_printed_only_with_the_flag(root: Path) -> None:
    closed = human_none(ROLES, HUMAN_NONE_SUBJECT, ("concept_label", "concept_note"))
    write_run(root, [beta_structure()], [closed])

    plain, flagged = invoke(root), invoke_settled(root)

    assert_succeeded(plain)
    assert_succeeded(flagged)
    assert not [line for line in printed_lines(plain) if marks_settled(line)], plain.output
    lines = printed_lines(flagged)
    assert len(lines) == 2, lines
    assert_line_shows(lines[0], beta_structure())
    assert_settled_line_shows(lines[1], closed)


def test_without_the_flag_output_is_unchanged_by_settled_entries(root: Path) -> None:
    _open_and_settled(root)

    result = invoke(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 2, lines
    assert_line_shows(lines[0], beta_structure())
    assert_line_shows(lines[1], gamma_role())


def test_settled_on_a_run_with_a_spec_prints_the_spec_path_then_the_settled_lines(
    root: Path,
) -> None:
    spec_path = write_run(
        root, [], [lexicon_settled_role()], spec=SPEC_BODY,
        structure_settled=[model_settled_structure()],
    )

    result = invoke_settled(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 3, lines
    assert lines[0] == str(spec_path)
    assert_settled_line_shows(lines[1], model_settled_structure())
    assert_settled_line_shows(lines[2], lexicon_settled_role())


def test_settled_never_masks_a_run_with_no_open_entry_and_no_spec(root: Path) -> None:
    spec_path = write_run(
        root, [], [lexicon_settled_role()], structure_settled=[model_settled_structure()]
    )

    result = invoke_settled(root)

    assert not spec_path.exists()
    assert result.exit_code != 0, result.output
    assert RUN_ID in result.output, result.output
    assert NO_SPEC in result.output.lower(), result.output


def test_settled_reads_structure_choices_from_the_structure_artifact(root: Path) -> None:
    write_run(
        root, [beta_structure()], [lexicon_settled_role()],
        structure_settled=[model_settled_structure()],
    )

    flagged, plain = invoke_settled(root), invoke(root)

    assert_succeeded(flagged)
    lines = printed_lines(flagged)
    assert len(lines) == 3, lines
    assert_line_shows(lines[0], beta_structure())
    assert_settled_line_shows(lines[1], model_settled_structure())
    assert_settled_line_shows(lines[2], lexicon_settled_role())
    assert_succeeded(plain)
    assert len(printed_lines(plain)) == 1, plain.output


def test_a_run_without_a_structure_artifact_still_prints_with_settled(root: Path) -> None:
    write_run(root, [beta_structure()], [lexicon_settled_role()])

    result = invoke_settled(root)

    assert_succeeded(result)
    lines = printed_lines(result)
    assert len(lines) == 2, lines
    assert_line_shows(lines[0], beta_structure())
    assert_settled_line_shows(lines[1], lexicon_settled_role())
