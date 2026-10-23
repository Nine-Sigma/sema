"""Builders for the `sema target review <run>` reader tests.

A synthetic run ledger: a registry record, the two open-entry artifacts written as plain
JSON (so keys later tasks add, such as ``agreement``, ``findings``, and ``cause``, survive;
the ``Ambiguity`` model drops them and ``RunLedger.artifact`` refuses a dict), ``run_metrics``,
and an optional spec.
Every name is invented; nothing here reads a fixture or a held-out source.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from sema.cli_target import target_group
from sema.intake.registry import Registry

Entry = dict[str, Any]

STRUCTURE, ROLES = "structure", "roles"
RUN_ID, MISSING_RUN_ID = "review-run-1", "review-run-absent"
SOURCE = "invented-source"
REGISTRY_FILE, ARTIFACTS_DIR, SPEC_FILE = "registry.sqlite", "artifacts", "spec.json"
SEEDED_TOKENS, SEEDED_COST = 120, 0.002
PROVIDER_ENTRY_POINTS = (
    "sema.cli_intake._intake_client",
    "sema.intake.model_calls.build_intake_client",
)
OPTIONAL_ENTRY_POINTS = (
    "sema.cli_intake.build_intake_client",
    "sema.cli_intake.configured_embedder",
)
PROVIDER_CALLED = "a provider call was made"

LIST_MEMBER = "concept_list(code=c1, label=l1)"
VALUES_MEMBER = "concept_with_values(code=c1, value=v1)"
DENORMALIZED_MEMBER = "concept_with_values_denormalized(group_by=g1, value=v1)"
CODE_ROLE, LABEL_ROLE = "concept_code", "concept_label"


def entry(
    stage: str,
    subject: str,
    members: Sequence[str],
    *,
    hints: Sequence[str] = (),
    exclusions: dict[str, str] | None = None,
    reason: str | None = None,
    **extra: Any,
) -> Entry:
    """An open entry (``choice`` null) as the ledger writes it; ``extra`` adds later keys."""
    return {
        "stage": stage,
        "subject": subject,
        "members": list(members),
        "hints": list(hints),
        "exclusions": dict(exclusions or {}),
        "choice": None,
        "confidence": None,
        "source": None,
        "call_key": None,
        "reason": reason,
        **extra,
    }


def settled(stage: str, subject: str, members: Sequence[str]) -> Entry:
    return {
        **entry(stage, subject, members),
        "choice": members[0],
        "confidence": 0.9,
        "source": "model",
        "call_key": "invented-call-key",
    }


def human_none(stage: str, subject: str, members: Sequence[str]) -> Entry:
    """A required role a human settled ``none``: ``choice`` stays null, ``source`` names
    the human (as the eligibility writers close it)."""
    return {**entry(stage, subject, members), "source": "human"}


def alpha_structure() -> Entry:
    return entry(
        STRUCTURE,
        "alpha_table",
        (LIST_MEMBER, VALUES_MEMBER),
        hints=("lexicon: code, label", "embedding: label margin 0.312", "settled_by: none"),
        exclusions={VALUES_MEMBER: "value column v1 is empty in every row"},
        reason="two shapes fit the alpha header",
    )


def beta_structure() -> Entry:
    return entry(
        STRUCTURE,
        "beta_table",
        (DENORMALIZED_MEMBER, LIST_MEMBER, VALUES_MEMBER),
        hints=("lexicon: group, value",),
        exclusions={
            LIST_MEMBER: "column g1 repeats across 40 rows",
            VALUES_MEMBER: "column g1 is not a code column",
        },
    )


def gamma_role() -> Entry:
    return entry(
        ROLES,
        "gamma_table.c_key",
        (CODE_ROLE, LABEL_ROLE),
        hints=("lexicon: c_key", "embedding: margin 0.120"),
        exclusions={LABEL_ROLE: "c_key values are 6-character tokens"},
        reason="the key reads as a code or a label",
    )


def settled_structure() -> Entry:
    return settled(STRUCTURE, "settled_table", (LIST_MEMBER, VALUES_MEMBER))


def seeded_root(tmp_path: Path, run_id: str = RUN_ID) -> Path:
    """A ledger root whose registry holds one run that has made one counted call."""
    root = tmp_path / "ledger"
    registry = Registry(root / REGISTRY_FILE)
    registry.register(run_id, source=SOURCE, stage="review")
    registry.count_call(run_id, SEEDED_TOKENS, SEEDED_COST)
    return root


def write_run(
    root: Path,
    structure: Sequence[Entry],
    roles: Sequence[Entry],
    *,
    run_id: str = RUN_ID,
    spec: dict[str, Any] | None = None,
) -> Path:
    """Write the two lists, ``run_metrics``, and the spec when given; return the spec path."""
    artifacts = root / run_id / ARTIFACTS_DIR
    for stage, listed in ((STRUCTURE, structure), (ROLES, roles)):
        payload = {"stage": stage, "count": len(listed), "ambiguities": list(listed)}
        _write_json(artifacts / f"{stage}_ambiguities.json", payload)
    _write_json(artifacts / "run_metrics.json", _metrics(structure, roles))
    spec_path = artifacts / SPEC_FILE
    if spec is not None:
        _write_json(spec_path, spec)
    return spec_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _metrics(structure: Sequence[Entry], roles: Sequence[Entry]) -> dict[str, Any]:
    def _open(listed: Sequence[Entry]) -> int:
        return sum(1 for listed_entry in listed if listed_entry["choice"] is None)

    return {
        "relations": len(structure) + len(roles),
        "ignored": {},
        "bound": 0,
        "bound_fraction": 0.0,
        "open_structure": _open(structure),
        "open_roles": _open(roles),
        "proposals_made": 0,
        "proposals_bound": 0,
        "model_calls": 1,
        "tokens": SEEDED_TOKENS,
        "cost": SEEDED_COST,
        "value_asks": 0,
    }


def forbid_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(PROVIDER_CALLED)

    for target in PROVIDER_ENTRY_POINTS:
        monkeypatch.setattr(target, _refuse)
    for target in OPTIONAL_ENTRY_POINTS:
        monkeypatch.setattr(target, _refuse, raising=False)


def invoke(root: Path, run_id: str = RUN_ID) -> Result:
    return CliRunner().invoke(target_group, ["review", run_id, "--root", str(root)])


def printed_lines(result: Result) -> list[str]:
    """Non-blank output lines; a trailing newline or blank separator is not a line."""
    return [line for line in result.output.splitlines() if line.strip()]


def assert_succeeded(result: Result) -> None:
    assert result.exit_code == 0, (result.exit_code, result.output, result.exception)


def assert_in_order(line: str, parts: Sequence[str]) -> None:
    """Each part is on the line, and each starts after the one before it."""
    positions = [line.find(part) for part in parts]
    missing = [part for part, at in zip(parts, positions, strict=True) if at < 0]
    assert not missing, (missing, line)
    assert positions == sorted(positions), (list(zip(parts, positions, strict=True)), line)


def assert_line_shows(line: str, shown: Entry) -> None:
    """The line carries the entry's stage, subject, ranked members, facts, hints, reason."""
    assert_in_order(line, shown["members"])
    texts = [shown["stage"], shown["subject"], *shown["exclusions"].values(), *shown["hints"]]
    if shown["reason"] is not None:
        texts.append(shown["reason"])
    absent = [text for text in texts if text not in line]
    assert not absent, (absent, line)


def line_naming(lines: Sequence[str], subject: str) -> str:
    found = [line for line in lines if subject in line]
    assert len(found) == 1, (subject, lines)
    return found[0]
