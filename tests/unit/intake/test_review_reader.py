"""`sema target review <run>` prints a run's open entries (S5 decisions 14 and 19).

Pinned contract (generality-gate spec, "Open entries have a human-readable view"):

- The command reads ``<root>/<run>/artifacts/structure_ambiguities.json`` and
  ``roles_ambiguities.json`` as plain JSON, not through ``Ambiguity``, and prints one line
  per open entry (``choice`` null): structure entries in ledger order, then roles entries.
  A settled entry prints nothing.
- Each line carries the entry's ``stage``, ``subject``, every member in the writer's rank
  (the reader never re-ranks), every exclusion fact, every hint (lexicon and embedding
  hits), and the ``reason`` when it is not null.
- Keys later tasks add print when present and are ignored when absent: ``agreement``
  (decision 4) on its entry's line, ``findings`` after the member each belongs to, and
  entries carrying ``cause`` (decision 14) after every other open entry, one line each
  with the cause and the subject.
- Text only, read-only: no JSON, no provider call, the registry record unchanged.
- A run with no open entry and an assembled ``artifacts/spec.json`` prints that path and
  nothing else. A run id with no ledger directory exits non-zero naming the run id.

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
    assert_succeeded,
    beta_structure,
    entry,
    forbid_provider_calls,
    gamma_role,
    invoke,
    line_naming,
    printed_lines,
    seeded_root,
    settled_structure,
    write_run,
)

pytestmark = pytest.mark.unit

AGREEMENT = "1/2"
FINDING = "denormalized group ratio 0.4"
REQUIRED_NONE_CAUSE = "required role label settled none"
DIVERGENT_CAUSE = "per-file dictionaries map the role code differently"
SPEC_BODY = {"concept_source": {"spec_version": 2}}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return seeded_root(tmp_path)


def test_each_open_entry_prints_one_line_structure_first_settled_absent(root: Path) -> None:
    opened = [alpha_structure(), beta_structure(), gamma_role()]
    write_run(root, [opened[0], settled_structure(), opened[1]], [opened[2]])

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
