"""Row-ordered samples on an approved value ask (S5 decision 5; task 3.4, rulings R1 to R4).

Pinned contract, over invented CSVs landed by the tabular loader (so rows carry a landed
order that differs from their sorted order):

- A. ``ProfileTools.sample_rows(ref, columns, k)`` is gated like ``sample_values``: a
  sensitive column is refused naming it; under ``metadata_only``, or for a column outside
  a ``bounded_values`` allowlist, it returns a ``NeedsValues`` for the relation and columns
  and records it in ``value_requests``; once allowed it returns the first ``k`` landed rows
  (at most 20) of the columns in the order asked, logged value-bearing.
- B. On the layout re-ask after approval the prompt holds a ``First rows`` block after
  ``Values`` with the approved columns' rows in landed order; before approval it holds none.
- C. The Roles prompt over approved columns holds the same block (needs task 2.2's
  role-map prompt to be meaningful).
- D. A second ``needs_values`` for approved columns is a ``LayoutFinding`` naming them, not
  a second ``NeedsValues`` (pinned on the proposal result, not through the graph).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from sema.binding.spec_sections import Provider, RelationRef
from sema.intake.calls.ledger import Ledger
from sema.intake.land.loaders.tabular import land_tabular
from sema.intake.land.profile import (
    EvidencePolicy,
    EvidenceTier,
    NeedsValues,
    ProfileRefusedError,
    ProfileTools,
)
from sema.intake.resolve.resolve import resolve_roles
from tests.unit.intake.layout_proposal_env import (
    CODEBOOK,
    CODEBOOK_COLUMNS,
    CODEBOOK_CSV,
    VARIABLE,
    Landed,
    abstain,
    approved_tools,
    invokes,
    needs,
    run_proposals,
    scripted,
    tools_for,
    write_codebook,
)
from tests.unit.intake.model_calls_env import wired_run
from tests.unit.intake.role_entries_env import concept_list, pinned, roles_of
from tests.unit.intake.role_map_env import mapped, prompts
from tests.unit.intake.structure.shapes.shapes_env import Source
from tests.unit.intake.structure_env import fresh_tools, land, run

pytestmark = pytest.mark.unit

STEPS = "steps"
STEP_COLUMNS = ("step_code", "step_name", "step_note")
ROWS = 25
CAP = 20
FIRST_ROWS, VALUES = "First rows", "Values"
# Landed descending, so the first landed rows are not the first sorted ones.
STEP_ROWS = [
    (f"S{i:02d}", f"Step name {i:02d}", f"Note for step {i:02d}") for i in range(ROWS, 0, -1)
]
CODEBOOK_FIRST_LABELS = ("red", "green", "blue", "small")


def _write_steps(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{STEPS}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(STEP_COLUMNS)
        writer.writerows(STEP_ROWS)
    return directory


@pytest.fixture
def steps(tmp_path: Path) -> Iterator[Source]:
    staging, source = land(tmp_path / "steps", "sr", land_tabular, _write_steps(tmp_path / "in"))
    yield source
    staging.close()


def _ref(source: Source, table: str) -> RelationRef:
    relation = source.land.relation(table)
    return RelationRef(provider=cast(Provider, source.provider), name=f"{relation.schema}.{table}")


def _tools(
    source: Source, root: Path, policy: EvidencePolicy, stage: str = "structure", **kwargs: Any
) -> ProfileTools:
    ledger = Ledger(root / "ledger").run("profile")
    return ProfileTools(
        source.tools.provider, policy, ledger, stage=stage, land=source.land, **kwargs
    )


def _value_bearing(root: Path) -> list[dict[str, Any]]:
    path = root / "ledger" / "profile" / "sql.jsonl"
    lines = path.read_text().splitlines() if path.is_file() else []
    return [found for line in lines if (found := json.loads(line))["value_bearing"]]


def _rows(sample: Any) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in getattr(sample, "rows", sample)]


def _sample_rows(tools: ProfileTools, ref: RelationRef, columns: list[str], k: int) -> Any:
    """Read by name so the missing method fails its test, not the type check."""
    return getattr(tools, "sample_rows")(ref, columns, k)


def _allowing(ref: RelationRef, columns: Sequence[str]) -> EvidencePolicy:
    allowlist = frozenset((ref.name, column) for column in columns)
    return EvidencePolicy(tier="bounded_values", allowlist=allowlist)


def test_rows_come_in_landed_order_in_the_columns_asked(steps: Source, tmp_path: Path) -> None:
    ref = _ref(steps, STEPS)
    tools = _tools(steps, tmp_path, _allowing(ref, STEP_COLUMNS))

    sample = _sample_rows(tools, ref, ["step_name", "step_code"], 3)

    assert _rows(sample) == [(name, code) for code, name, _ in STEP_ROWS[:3]]
    assert _value_bearing(tmp_path)


def test_rows_are_capped_at_twenty(steps: Source, tmp_path: Path) -> None:
    ref = _ref(steps, STEPS)
    tools = _tools(steps, tmp_path, EvidencePolicy(tier="full"))

    sample = _sample_rows(tools, ref, ["step_code"], 50)

    assert _rows(sample) == [(code,) for code, _, _ in STEP_ROWS[:CAP]]


@pytest.mark.parametrize("tier", ["metadata_only", "bounded_values"])
def test_withheld_columns_return_a_value_request_and_read_nothing(
    steps: Source, tmp_path: Path, tier: EvidenceTier
) -> None:
    ref = _ref(steps, STEPS)
    tools = _tools(steps, tmp_path, EvidencePolicy(tier=tier))

    marker = _sample_rows(tools, ref, ["step_code", "step_note"], 5)

    assert marker == NeedsValues(relation=ref.name, columns=("step_code", "step_note"))
    assert marker in tools.value_requests
    assert _value_bearing(tmp_path) == []


def test_a_sensitive_column_is_refused_like_a_value_sample(
    steps: Source, tmp_path: Path
) -> None:
    ref = _ref(steps, STEPS)
    policy = EvidencePolicy(tier="full", sensitive=frozenset({(ref.name, "step_note")}))
    tools = _tools(steps, tmp_path, policy)

    with pytest.raises(ProfileRefusedError) as rows_error:
        _sample_rows(tools, ref, ["step_code", "step_note"], 5)
    with pytest.raises(ProfileRefusedError) as values_error:
        tools.sample_values(ref, "step_note", 5)

    for error in (rows_error.value, values_error.value):
        assert "step_note" in str(error) and "sensitive" in str(error), error
    assert _value_bearing(tmp_path) == []


@pytest.fixture
def codebook(tmp_path: Path) -> Iterator[Landed]:
    directory = write_codebook(tmp_path / "codebook_in", CODEBOOK_CSV)
    staging, source = land(tmp_path / "codebook", "sr", land_tabular, directory)
    structure = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))
    yield Landed(source, structure)
    staging.close()


def _after(prompt: str, heading: str) -> str:
    assert heading in prompt, f"no {heading!r} block"
    return prompt[prompt.index(heading):]


def _in_order(text: str, words: Sequence[str]) -> bool:
    places = [text.find(word) for word in words]
    return -1 not in places and places == sorted(places)


def _asked_then_approved(codebook: Landed, root: Path, second: Any) -> tuple[Any, Any]:
    relation = codebook.name(CODEBOOK)
    client = scripted({CODEBOOK_COLUMNS: [needs(VARIABLE, "label"), second]})
    wired = wired_run(root / "calls", client)
    run_proposals(codebook, wired, tools_for(codebook, root / "ledger" / "asked"))
    approved = approved_tools(codebook, root / "ledger" / "approved", relation, [VARIABLE, "label"])
    return run_proposals(codebook, wired, approved), client


def test_the_layout_re_ask_shows_first_rows_after_values(
    codebook: Landed, tmp_path: Path
) -> None:
    _, client = _asked_then_approved(codebook, tmp_path, abstain())

    (asked, _), (reasked, _) = invokes(client)

    assert FIRST_ROWS not in asked and "green" not in asked
    rows = _after(_after(reasked, VALUES), FIRST_ROWS)
    assert _in_order(rows, CODEBOOK_FIRST_LABELS), rows[:400]


def test_a_repeated_value_ask_for_approved_columns_is_a_layout_finding(
    codebook: Landed, tmp_path: Path
) -> None:
    result, client = _asked_then_approved(codebook, tmp_path, needs(VARIABLE, "label"))

    relation = codebook.name(CODEBOOK)
    assert client.invoke.call_count == 2
    assert result.value_requests == (), result.value_requests
    found = [f for f in result.findings if f.relation == relation]
    assert len(found) == 1, result.findings
    assert VARIABLE in found[0].rationale and "label" in found[0].rationale, found[0]


def test_the_roles_prompt_shows_first_rows_after_values_once_approved(
    steps: Source, tmp_path: Path
) -> None:
    ref = _ref(steps, STEPS)
    structure = pinned(steps, [concept_list(ref.name, "concepts")], tmp_path / "pinned")
    roles = roles_of(steps, structure, tmp_path / "roles")
    client = mapped({"concept_id": "step_code", "name": "step_name"})
    wired = wired_run(tmp_path / "calls", client)
    policy = _allowing(ref, STEP_COLUMNS)
    tools = _tools(steps, tmp_path, policy, stage="roles", structure=structure)

    resolve_roles(roles, structure, steps.land, tools, wired.calls, provider=ref.provider)

    sent = prompts(client)
    assert sent, "no Roles call was made"
    rows = _after(_after(sent[0], VALUES), FIRST_ROWS)
    assert _in_order(rows, [code for code, _, _ in STEP_ROWS[:3]]), rows[:400]
