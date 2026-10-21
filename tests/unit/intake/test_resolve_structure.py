"""Structure ambiguities settled by a mocked model: batched, recorded, verified, or left open.

Pinned API: ``sema.intake.resolve.resolve_structure(structure, land, tools, calls, *,
provider="staging", max_retries=2) -> StructureResult``. A choice is written onto its
ambiguity entry (``choice``, ``confidence``, ``source: model``, ``call_key``) and bound with
``source: model``; an entry with no choice is open. ``sema.intake.resolve`` is imported
lazily so a missing module fails each test on its own.
"""

from __future__ import annotations

import csv
import importlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from pydantic import BaseModel

from sema.intake.ledger_utils import prompt_hash
from sema.intake.loaders.tabular import land_tabular
from sema.intake.output_models import MAX_STRUCTURE_BATCH, STRUCTURE_BATCH
from sema.intake.output_models_utils import ABSTAIN
from sema.intake.shapes.base import SlotBinding, profile_relations
from sema.intake.shapes.precedence import member
from sema.intake.structure import BoundEntry
from sema.intake.structure_utils import group_candidates
from sema.intake.verify.structure import verify_structure
from tests.unit.intake.model_calls_env import MODEL, WiredRun, stats, wired_run
from tests.unit.intake.shapes.shapes_env import Source, warehouse_source
from tests.unit.intake.structure_env import (
    ACRIS,
    CODE_TABLES,
    GROUPED_CODE_TABLE,
    copy_code_tables,
    fresh_tools,
    land,
    ref,
    run,
    sql_lines,
)

pytestmark = pytest.mark.unit

PROMPT_VERSION = "# prompt: structure v1"
CONFIDENCE = 0.9
REASON = "the label column reads as one flat list"
ABSTAIN_REASON = "both readings fit the profile equally"
OUTLINES = MAX_STRUCTURE_BATCH + 1
OUTLINE_ROWS = (
    "('1', 'Zephyr part'), ('1.1', 'Zephyr part, item one'), ('1.2', 'Zephyr part, item two'),"
    " ('2', 'Quillon part'), ('2.1', 'Quillon part, item one'), ('3', 'Marrow part')"
)
CONFLICT_SQL = (
    "CREATE SCHEMA tc",
    "CREATE TABLE tc.lib__first AS SELECT * FROM (VALUES ('K1', 'Alder grove'),"
    " ('K2', 'Birch hollow'), ('K3', 'Cedar ridge'), ('K4', 'Dune flat')) t(key, label)",
    "CREATE TABLE tc.lib__second AS SELECT * FROM (VALUES ('K1', 'Elm spire'),"
    " ('K2', 'Fir basin'), ('K3', 'Gorse moor'), ('K4', 'Heath knoll')) t(key, label)",
    "CREATE TABLE tc.lib__values AS SELECT * FROM (VALUES ('K1', 'V1', 'Ivy one'),"
    " ('K1', 'V2', 'Juniper two'), ('K2', 'V3', 'Kelp three'), ('K3', 'V4', 'Larch four'),"
    " ('K3', 'V5', 'Maple five')) t(key, value, label)",
)
SHARED = "tc.lib__values"
CLAIMING = "concept_and_value_set("
FLAT_LIST = "concept_list("

Picker = Callable[[tuple[str, ...]], str]


def resolve() -> Any:
    return importlib.import_module("sema.intake.resolve")


def entry_fields(output_model: type[BaseModel]) -> list[str]:
    names = [name for name in output_model.model_fields if name.startswith("entry_")]
    return sorted(names, key=lambda name: int(name.removeprefix("entry_")))


def entry_members(output_model: type[BaseModel], field: str) -> tuple[str, ...]:
    enum = output_model.model_json_schema()["properties"][field]["enum"]
    return tuple(value for value in enum if value != ABSTAIN)


def answering(pick: Picker, reason: str = REASON) -> MagicMock:
    """A mocked `LLMClient` that answers every entry of a batch with `pick(members)`."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        fields = entry_fields(output_model)
        chosen = {field: pick(entry_members(output_model, field)) for field in fields}
        count = len(fields)
        return output_model(**chosen, confidences=[CONFIDENCE] * count, reasons=[reason] * count)

    client.invoke.side_effect = _invoke
    return client


def starting(prefix: str, *, mentioning: str = "") -> Picker:
    def _pick(members: tuple[str, ...]) -> str:
        found = [m for m in members if m.startswith(prefix) and mentioning in m]
        return found[0] if found else ABSTAIN

    return _pick


def claim_own(members: tuple[str, ...]) -> str:
    """Each concept relation claims the shared value relation; the shared entry abstains."""
    for found in members:
        if found.startswith(CLAIMING) and f"concept_relation={SHARED}" not in found:
            subject = found.split("concept_relation=")[1].split(",")[0]
            if all(subject in m for m in members):
                return found
    return ABSTAIN


def prompts(client: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in client.invoke.call_args_list]


def output_models(client: MagicMock) -> list[type[BaseModel]]:
    return [call.args[1] for call in client.invoke.call_args_list]


def open_subjects(result: Any) -> set[str]:
    return {a.subject for a in result.ambiguities if a.choice is None}


def chosen(result: Any, subject: str) -> Any:
    (found,) = [a for a in result.ambiguities if a.subject == subject and a.choice is not None]
    return found


def entry_over(result: Any, subject: str) -> BoundEntry:
    (found,) = [e for e in result.bound if subject in e.binding.relations]
    return found


def model_bound(result: Any) -> list[BoundEntry]:
    return [entry for entry in result.bound if entry.source == "model"]


@pytest.fixture(scope="module")
def tmp_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("resolve_structure")


@pytest.fixture(scope="module")
def codes(tmp_module: Path) -> Iterator[Source]:
    directory = copy_code_tables(tmp_module / "codes_only" / "csv")
    staging, source = land(tmp_module / "codes", "codes", land_tabular, directory)
    yield source
    staging.close()


@pytest.fixture(scope="module")
def codes_structure(codes: Source, tmp_module: Path) -> Any:
    return run(codes, fresh_tools(codes, tmp_module / "ledger" / "structure"))


def _warehouse(statements: tuple[str, ...], schema: str, root: Path) -> Iterator[Source]:
    con = duckdb.connect(":memory:")
    for statement in statements:
        con.execute(statement)
    yield warehouse_source(con, schema, root / "ledger" / "landing")
    con.close()


@pytest.fixture(scope="module")
def outlines(tmp_module: Path) -> Iterator[Source]:
    statements = ("CREATE SCHEMA tw",) + tuple(
        f"CREATE TABLE tw.outline_{i} AS SELECT * FROM (VALUES {OUTLINE_ROWS})"
        f" t(code_{i}, label_{i})"
        for i in range(OUTLINES)
    )
    yield from _warehouse(statements, "tw", tmp_module / "outlines")


@pytest.fixture(scope="module")
def conflict(tmp_module: Path) -> Iterator[Source]:
    yield from _warehouse(CONFLICT_SQL, "tc", tmp_module / "conflict")


def _structure_of(source: Source, root: Path) -> Any:
    return run(source, fresh_tools(source, root / "ledger" / "structure"))


def _resolved(
    source: Source, structure: Any, client: MagicMock, root: Path, **kwargs: Any
) -> tuple[Any, WiredRun]:
    wired = wired_run(root / "calls", client)
    tools = fresh_tools(source, root / "ledger" / "resolve")
    result = resolve().resolve_structure(
        structure, source.land, tools, wired.calls, provider=source.provider, **kwargs
    )
    return result, wired


def test_entries_are_sent_one_call_per_batch_of_at_most_fifty(
    outlines: Source, tmp_path: Path
) -> None:
    structure = _structure_of(outlines, tmp_path)
    assert len(open_subjects(structure)) == OUTLINES
    client = answering(starting(FLAT_LIST))

    result, _ = _resolved(outlines, structure, client, tmp_path)

    sizes = [len(entry_fields(model)) for model in output_models(client)]
    assert sizes == [MAX_STRUCTURE_BATCH, OUTLINES - MAX_STRUCTURE_BATCH]
    assert all(model.__name__ == STRUCTURE_BATCH for model in output_models(client))
    assert open_subjects(result) == set()
    assert len(model_bound(result)) == OUTLINES


def test_choice_is_bound_with_source_model_confidence_and_call_key(
    codes: Source, codes_structure: Any, tmp_path: Path
) -> None:
    subject = ref(codes, GROUPED_CODE_TABLE)
    assert open_subjects(codes_structure) == {ref(codes, name) for name in CODE_TABLES}
    client = answering(starting(FLAT_LIST))

    result, wired = _resolved(codes, codes_structure, client, tmp_path)

    (prompt,) = prompts(client)
    key = prompt_hash(MODEL, prompt, STRUCTURE_BATCH)
    entry = entry_over(result, subject)
    assert entry.binding.template == "concept_list"
    assert entry.source == "model"
    assert entry.confidence == CONFIDENCE
    settled = chosen(result, subject)
    assert settled.choice == member(entry.binding)
    assert settled.source == "model"
    assert settled.confidence == CONFIDENCE
    assert settled.call_key == key
    assert f"{key}.json" in wired.call_files()
    assert subject not in open_subjects(result)


def test_code_bound_entries_are_kept_unchanged(
    codes: Source, codes_structure: Any, tmp_path: Path
) -> None:
    result, _ = _resolved(codes, codes_structure, answering(starting(FLAT_LIST)), tmp_path)

    kept = [entry for entry in result.bound if entry.source != "model"]
    assert kept == list(codes_structure.bound)
    assert result.ignored == codes_structure.ignored
    assert len(result.bound) == len(CODE_TABLES)


def test_structure_verifier_reruns_and_passes_on_the_model_bound_result(
    codes: Source, codes_structure: Any, tmp_path: Path
) -> None:
    with patch.object(resolve(), "verify_structure", wraps=verify_structure) as verifier:
        result, _ = _resolved(codes, codes_structure, answering(starting(FLAT_LIST)), tmp_path)

    assert verifier.called
    verdict = verify_structure(result, codes.land, codes.tools, provider=codes.provider)
    assert verdict.status == "pass", verdict.failures


def _claims_over_shared(source: Source) -> list[SlotBinding]:
    profiles = profile_relations(source.land, source.tools, source.provider)
    found = group_candidates(profiles, source.tools)
    return [b for b in found if b.template == "concept_and_value_set" and SHARED in b.relations]


def test_two_model_choices_claiming_one_relation_fail_precedence_applied(
    conflict: Source, tmp_path: Path
) -> None:
    structure = _structure_of(conflict, tmp_path)
    claims = _claims_over_shared(conflict)
    assert len(claims) == 2
    bound = tuple(
        BoundEntry(binding=binding, confidence=CONFIDENCE, source="model")
        for binding in claims
    )
    shared_open = tuple(a for a in structure.ambiguities if a.subject == SHARED)
    proposal = structure.model_copy(update={"bound": bound, "ambiguities": shared_open})

    verdict = verify_structure(proposal, conflict.land, conflict.tools, provider=conflict.provider)

    failing = [f for f in verdict.failures if f.check == "precedence_applied"]
    assert failing, verdict.failures
    assert {f.relation for f in failing} == {SHARED}
    assert all(f.evidence.get("source") == "model" for f in failing)


def test_conflicting_batch_is_retried_twice_with_the_failure_then_left_open(
    conflict: Source, tmp_path: Path
) -> None:
    structure = _structure_of(conflict, tmp_path)
    subjects = open_subjects(structure)
    assert subjects == {"tc.lib__first", "tc.lib__second", SHARED}
    client = answering(claim_own)

    result, _ = _resolved(conflict, structure, client, tmp_path)

    sent = prompts(client)
    assert len(sent) == 3
    for retry in sent[1:]:
        assert retry != sent[0]
        assert "precedence_applied" in retry
        assert SHARED in retry
        assert all(member(claim) in retry for claim in _claims_over_shared(conflict))
    assert model_bound(result) == []
    assert open_subjects(result) == subjects
    verdict = verify_structure(result, conflict.land, conflict.tools, provider=conflict.provider)
    assert verdict.status == "pass", verdict.failures


def test_retries_stop_at_max_retries(conflict: Source, tmp_path: Path) -> None:
    structure = _structure_of(conflict, tmp_path)
    client = answering(claim_own)

    _resolved(conflict, structure, client, tmp_path, max_retries=0)

    assert client.invoke.call_count == 1


def test_abstain_leaves_the_entry_open_with_the_reason(
    codes: Source, codes_structure: Any, tmp_path: Path
) -> None:
    subject = ref(codes, GROUPED_CODE_TABLE)
    client = answering(lambda _members: ABSTAIN, reason=ABSTAIN_REASON)

    result, _ = _resolved(codes, codes_structure, client, tmp_path)

    assert client.invoke.call_count == 1
    assert model_bound(result) == []
    (entry,) = [a for a in result.ambiguities if a.subject == subject]
    assert entry.choice is None
    assert entry.source is None
    assert entry.reason == ABSTAIN_REASON
    assert open_subjects(result) == {ref(codes, name) for name in CODE_TABLES}


def _code_table_values(codes: Source) -> set[str]:
    path = ACRIS / f"{GROUPED_CODE_TABLE}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    names = " ".join([*header, *(r.name for r in codes.land.relations)]).lower()
    cells = {cell.strip() for row in body for cell in row}
    return {cell for cell in cells if len(cell) >= 5 and cell.lower() not in names}


def test_metadata_only_prompt_holds_no_value_from_a_non_schema_relation(
    codes: Source, codes_structure: Any, tmp_path: Path
) -> None:
    tools = fresh_tools(codes, tmp_path / "ledger" / "resolve")
    assert tools.policy.tier == "metadata_only"
    assert ref(codes, GROUPED_CODE_TABLE) not in codes_structure.schema_relations
    client = answering(starting(FLAT_LIST))

    _resolved(codes, codes_structure, client, tmp_path)

    (prompt,) = prompts(client)
    assert prompt.splitlines()[0] == PROMPT_VERSION
    values = _code_table_values(codes)
    assert len(values) > 50
    leaked = sorted(value for value in values if value in prompt)
    assert leaked == []
    lines = sql_lines(tmp_path / "ledger" / "resolve")
    assert [line for line in lines if line["value_bearing"]] == []
