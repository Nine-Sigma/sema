"""How a layout proposal's fields become a slot binding, and how a finding is recorded.

Each template declares its slots once (``SLOTS``) with their kind: relation slots are filled
by code from the proposal's ``relations`` in declared order, column slots from
``slot_<slot>``, literal slots from ``literal_<slot>``; ``none`` leaves a slot unset. A slot
declared ``when`` another literal holds a value applies only then.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from sema.intake.land.loaders.tabular import land_tabular
from sema.intake.structure.shapes import (
    TEMPLATES,
    concept_and_value_set,
    entity_model,
    ignore,
    path_codes,
)
from sema.intake.structure.shapes.base import SlotBinding
from tests.unit.intake.layout_proposal_env import (
    LITERAL_PREFIX,
    NONE,
    REASON,
    REMARKS,
    Landed,
    abstain,
    enum,
    invokes,
    run_proposals,
    scripted,
    tools_for,
    write_book,
)
from tests.unit.intake.model_calls_env import wired_run
from tests.unit.intake.structure_env import fresh_tools, land, run

pytestmark = pytest.mark.unit

CONCEPTS, VALUES, ROWS = "s.concepts", "s.values", "s.rows"


def _binding(
    template: str,
    relations: tuple[str, ...],
    columns: dict[str, str] | None = None,
    literals: dict[str, str] | None = None,
) -> SlotBinding:
    from sema.intake.structure.proposal_utils import proposed_binding

    return proposed_binding(template, relations, columns or {}, literals or {})


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    root = tmp_path_factory.mktemp("layout_slots")
    staging, source = land(root, "book", land_tabular, write_book(root / "inputs"))
    yield Landed(source, run(source, fresh_tools(source, root / "ledger" / "structure")))
    staging.close()


def test_every_proposable_template_declares_its_slots() -> None:
    kinds = {"relation", "column", "literal"}
    for template in TEMPLATES:
        if template.ID == ignore.ID:
            assert not getattr(template, "SLOTS", ())
            continue
        slots = getattr(template, "SLOTS")
        assert slots, template.ID
        assert {slot.kind for slot in slots} <= kinds
        assert any(slot.kind == "relation" for slot in slots), template.ID


def test_multi_relation_slots_fill_in_declared_order() -> None:
    columns = {"concept_key": "key", "value_key": "key"}

    binding = _binding(concept_and_value_set.ID, (CONCEPTS, VALUES), columns)

    assert binding.template == concept_and_value_set.ID
    assert binding.relations == (CONCEPTS, VALUES)
    assert binding.slots == {
        "concept_relation": CONCEPTS,
        "value_relation": VALUES,
        "concept_key": "key",
        "value_key": "key",
    }


def test_literal_slot_is_read_from_its_own_field() -> None:
    binding = _binding(path_codes.ID, (ROWS,), {"code": "code"}, {"separator": "."})

    assert binding.slots == {"relation": ROWS, "code": "code", "separator": "."}


def test_unset_slots_are_left_out() -> None:
    binding = _binding(path_codes.ID, (ROWS,), {"code": NONE}, {"separator": NONE})

    assert binding.slots == {"relation": ROWS}


def test_slots_apply_only_under_their_literal() -> None:
    columns = {"entity": "table", "property": "field"}

    single = _binding(entity_model.ID, (ROWS,), columns, {"layout": "single_relation"})
    per_file = _binding(entity_model.ID, (ROWS, CONCEPTS), columns, {"layout": "per_file"})

    assert single.slots == {
        "layout": "single_relation", "relation": ROWS, "entity": "table", "property": "field"
    }
    assert per_file.slots == {"layout": "per_file", "dictionary": ROWS, "header": CONCEPTS}
    assert per_file.relations == (ROWS, CONCEPTS)


def test_output_model_offers_each_literal_slot_with_its_values(
    book: Landed, tmp_path: Path
) -> None:
    client = scripted({})
    run_proposals(book, wired_run(tmp_path / "calls", client), tools_for(book, tmp_path / "l"))

    for _, model in invokes(client):
        assert enum(model, f"{LITERAL_PREFIX}separator") == (*path_codes.SEPARATORS, NONE)
        layouts = (entity_model.PER_FILE, entity_model.SINGLE_RELATION, NONE)
        assert set(enum(model, f"{LITERAL_PREFIX}layout")) == set(layouts)
        assert not [name for name in model.model_fields if name.startswith("slot_level_0")]


def test_a_finding_is_written_as_a_layout_finding_event(book: Landed, tmp_path: Path) -> None:
    remarks = book.name(REMARKS)
    client = scripted({book.columns(REMARKS): [abstain(REASON)]})
    ledger_dir = tmp_path / "ledger"

    run_proposals(book, wired_run(tmp_path / "calls", client), tools_for(book, ledger_dir))

    events = [
        json.loads(line)
        for line in (ledger_dir / "events.jsonl").read_text("utf-8").splitlines()
    ]
    findings = [e["payload"] for e in events if e["kind"] == "layout_finding"]
    assert {"relation": remarks, "rationale": REASON} in [
        {k: p[k] for k in ("relation", "rationale")} for p in findings
    ]
