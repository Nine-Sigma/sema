"""Synthetic relations and a field-keyed mocked model for the role-map tests (S5 task 2.1).

Every relation, column, and value is invented. Each scenario lands one in-memory warehouse
schema, pins its structure by hand, and runs Roles under ``metadata_only``. The mocked
model answers a role field from a per-call mapping keyed by the role, so a test states
which column the model names; under the S4 enum model a name outside the enum abstains.
Names that do not exist before task 2.2 are imported lazily, so each test fails alone.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock

import duckdb
from pydantic import BaseModel

from sema.binding.spec_sections import Provider

from sema.intake.output_models import ENTRY_PREFIX, ROLE_PREFIX
from sema.intake.output_models_utils import ABSTAIN, NONE
from sema.intake.resolve import resolve_roles
from sema.intake.shapes import entity_model
from sema.intake.shapes.base import SlotBinding
from tests.unit.intake.model_calls_env import WiredRun, stats, wired_run
from tests.unit.intake.role_entries_env import concept_list, pinned, roles_of, source_of
from tests.unit.intake.shapes.shapes_env import Source, make_tools
from tests.unit.intake.structure_env import fresh_tools, run

SCHEMA = "rm"
ROWS = 20
CONFIDENCE = 0.8
REASON = "the column facts fit the role"
REASONING = ("confidences", "reasons")
ROLES_STAGE = "roles"
CONCEPTS = "concepts"
GHOST = "ghost_column"
PLANTED = "Qwyxzorbel"
VELL_COMMENT = "Free words a curator wrote about the tally"

PLAIN, CODES, TOKENS, SPARSE = "rm.plain", "rm.codes", "rm.tokens", "rm.sparse"
STATUSED, TALLIES, TALLY_FIELDS, DICT = "rm.statused", "rm.tallies", "rm.tally_fields", "rm.dict"
NAMED = "rm.named"
KIN = ("rm.kin_first", "rm.kin_second")
CLAIMED = "rm.lib__values"
OWNERS = ("rm.lib__first", "rm.lib__second")

Answer = Mapping[str, str]
Picker = Callable[[tuple[str, ...]], str]


def module(name: str) -> ModuleType:
    return importlib.import_module(name)


def output_models() -> ModuleType:
    return module("sema.intake.output_models")


def _pad(prefix: str) -> str:
    return f"'{prefix}' || lpad(CAST(i AS VARCHAR), 2, '0')"


def table(name: str, rows: int = ROWS, **columns: str) -> str:
    selected = ", ".join(f"{expression} AS {column}" for column, expression in columns.items())
    return f"CREATE TABLE {name} AS SELECT {selected} FROM range(1, {rows + 1}) r(i)"


LONG = "'{} says a long sentence about item number ' || CAST(i AS VARCHAR)"
SQL: dict[str, tuple[str, ...]] = {
    PLAIN: (table(PLAIN, zorb=_pad("Z"), quax="'Quax entry ' || i", vell=LONG.format("Vell")),),
    CODES: (
        table(
            CODES, zorb=_pad("Z"), quax="'Quax entry ' || i",
            pell=f"CASE WHEN i <= 4 THEN NULL ELSE {_pad('P')} END",
        ),
    ),
    TOKENS: (
        table(
            TOKENS, item_code=_pad("M"), zorb=_pad("Z"), item_label="'Term entry ' || i",
            vell=LONG.format("Vell"),
        ),
    ),
    SPARSE: (
        table(
            SPARSE, kelt=_pad("B"), wim="CASE WHEN i <= 4 THEN '' ELSE 'Marrow ' || i END",
            rusk="'Rusk ' || (i % 12)",
        ),
    ),
    STATUSED: (
        table(
            STATUSED, item_code=_pad("S"), item_label="'Term entry ' || i",
            item_status="CASE WHEN i % 2 = 0 THEN 'active' ELSE 'retired' END",
        ),
    ),
    TALLIES: (
        table(
            TALLIES, zorb=_pad("Z"), flag="'Y'", vell=LONG.format("Vell"),
            quax_name=f"CASE WHEN i = 7 THEN '{PLANTED}' ELSE 'Quax name ' || i END",
        ),
    ),
    DICT: (
        table(
            DICT, table_label="'Tbl ' || (i % 3)", field_label="'Field ' || i",
            key_flag="CASE WHEN i % 3 = 0 THEN 'Y' ELSE 'N' END", field_text=LONG.format("Note"),
        ),
    ),
    NAMED: (table(NAMED, item_code=_pad("N"), item_label="'Term entry ' || i"),),
    KIN[0]: (table(KIN[0], zorb=_pad("Z"), quax="'Quax entry ' || i"),),
    KIN[1]: (table(KIN[1], zorb=_pad("Z"), quax="'Quax entry ' || i"),),
}
CONFLICT_SQL = (
    "CREATE TABLE rm.lib__first AS SELECT * FROM (VALUES ('K1', 'Alder grove'),"
    " ('K2', 'Birch hollow'), ('K3', 'Cedar ridge'), ('K4', 'Dune flat')) t(key, label)",
    "CREATE TABLE rm.lib__second AS SELECT * FROM (VALUES ('K1', 'Elm spire'),"
    " ('K2', 'Fir basin'), ('K3', 'Gorse moor'), ('K4', 'Heath knoll')) t(key, label)",
    "CREATE TABLE rm.lib__values AS SELECT * FROM (VALUES ('K1', 'V1', 'Ivy one'),"
    " ('K1', 'V2', 'Juniper two'), ('K2', 'V3', 'Kelp three'), ('K3', 'V4', 'Larch four'),"
    " ('K3', 'V5', 'Maple five')) t(key, value, label)",
)


def schema_rows() -> int:
    """More dictionary rows than the role-map prompt may show."""
    return int(module("sema.intake.resolve_roles_utils").SCHEMA_TEXT_SAMPLE_ROWS) + 5


def tally_fields_sql() -> str:
    return table(
        TALLY_FIELDS, rows=schema_rows(), field_name="'tally_field_' || i",
        field_description=LONG.format("Field"),
    )


def code_list(relation: str) -> SlotBinding:
    return concept_list(relation, f"code_lists.{relation.rpartition('.')[2]}")


def per_file(dictionary: str) -> SlotBinding:
    slots = {entity_model.LAYOUT: entity_model.PER_FILE, entity_model.DICTIONARY: dictionary}
    return SlotBinding(
        template=entity_model.ID, slots=slots, relations=(dictionary,),
        section=entity_model.SECTION,
    )


def single_relation(relation: str, entity: str, prop: str) -> SlotBinding:
    slots = {
        entity_model.LAYOUT: entity_model.SINGLE_RELATION, entity_model.RELATION: relation,
        entity_model.ENTITY: entity, entity_model.PROPERTY: prop,
    }
    return SlotBinding(
        template=entity_model.ID, slots=slots, relations=(relation,),
        section=entity_model.SECTION,
    )


def commented(source: Source, relation: str, column: str, text: str, root: Path) -> Source:
    """``source`` with ``text`` as the landed comment of ``relation.column``."""
    name = relation.rpartition(".")[2]
    relations = tuple(
        dataclasses.replace(r, comments=((column, text),)) if r.name == name else r
        for r in source.land.relations
    )
    land = dataclasses.replace(source.land, relations=relations)
    ledger = root / "ledger" / "commented"
    return Source(land, make_tools(source.tools.provider, ledger, land), source.provider, ledger)


@dataclasses.dataclass
class Scenario:
    con: duckdb.DuckDBPyConnection
    source: Source
    structure: Any
    roles: Any
    root: Path
    kwargs: dict[str, Any]

    def resolve(self, client: MagicMock, **options: Any) -> tuple[Any, WiredRun]:
        wired = wired_run(self.root / "calls", client)
        tools = fresh_tools(
            self.source, self.root / "resolve", stage=ROLES_STAGE, structure=self.structure
        )
        result = resolve_roles(
            self.roles, self.structure, self.source.land, tools, wired.calls,
            provider=cast(Provider, self.source.provider), **self.kwargs, **options,
        )
        return result, wired

    def tools(self) -> Any:
        return fresh_tools(
            self.source, self.root / "verify", stage=ROLES_STAGE, structure=self.structure
        )


def scenario(
    statements: Sequence[str], bindings: Sequence[SlotBinding], root: Path,
    comment: tuple[str, str, str] | None = None, **kwargs: Any,
) -> Scenario:
    """``kwargs`` (an embedder) reach both ``run_roles`` and ``resolve_roles``."""
    con, source = source_of((f"CREATE SCHEMA {SCHEMA}", *statements), SCHEMA, root)
    if comment is not None:
        source = commented(source, *comment, root)
    structure = pinned(source, bindings, root / "pinned")
    roles = roles_of(source, structure, root / "roles", **kwargs)
    return Scenario(con, source, structure, roles, root, kwargs)


def conflict_structure(root: Path) -> tuple[duckdb.DuckDBPyConnection, Source, Any]:
    con, source = source_of((f"CREATE SCHEMA {SCHEMA}", *CONFLICT_SQL), SCHEMA, root)
    return con, source, run(source, fresh_tools(source, root / "ledger" / "structure"))


def claim_own(members: tuple[str, ...]) -> str:
    """Each concept relation claims the shared value relation, so two bindings span it."""
    for found in members:
        concept = found.partition("concept_relation=")[2].partition(",")[0]
        if found.startswith("concept_and_value_set(") and concept and concept != CLAIMED:
            if all(concept in other for other in members):
                return found
    return ABSTAIN


def _enum(output_model: type[BaseModel], name: str) -> tuple[str, ...] | None:
    schema = output_model.model_json_schema()["properties"][name]
    enum = schema.get("enum")
    return None if enum is None else tuple(enum)


def _value(wanted: str | None, enum: tuple[str, ...] | None) -> str:
    if enum is None:
        return wanted or NONE
    return wanted if wanted in enum else ABSTAIN


def mapped(*answers: Answer, entries: Picker | None = None) -> MagicMock:
    """Call ``n`` answers each role field from ``answers[n]`` (the last one repeats), keyed by
    the field without its prefix; an unnamed role answers ``none``. Structure entry fields
    answer through ``entries``. The calls' models are kept on ``client.models``."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    client.models = []

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        client.models.append(output_model)
        answer = answers[min(len(client.models), len(answers)) - 1] if answers else {}
        chosen: dict[str, str] = {}
        for name in output_model.model_fields:
            enum = None if name in REASONING else _enum(output_model, name)
            if name.startswith(ROLE_PREFIX):
                chosen[name] = _value(answer.get(name.removeprefix(ROLE_PREFIX)), enum)
            elif name.startswith(ENTRY_PREFIX) and enum is not None:
                chosen[name] = (entries or (lambda _: ABSTAIN))(
                    tuple(m for m in enum if m != ABSTAIN)
                )
            elif name not in REASONING and enum is not None:
                chosen[name] = ABSTAIN
        count = len(chosen)
        return output_model(**chosen, confidences=[CONFIDENCE] * count, reasons=[REASON] * count)

    client.invoke.side_effect = _invoke
    return client


def prompts(client: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in client.invoke.call_args_list]


def model_names(client: MagicMock) -> list[str]:
    return [model.__name__ for model in client.models]


def lines_with(text: str, *words: str) -> list[str]:
    return [line for line in text.splitlines() if all(word in line for word in words)]


def subject_entry(result: Any, subject: str) -> Any:
    (found,) = [a for a in result.ambiguities if a.subject == subject]
    return found


def chosen(result: Any, relation: str, role: str) -> Any:
    """The relation-level eligibility of ``role``, or ``None`` when Roles asked no such role."""
    found = [
        e for e in result.eligibilities
        if e.relation == relation and e.role == role and e.subject is None
    ]
    assert len(found) <= 1, found
    return found[0] if found else None


def explained(entry: Any) -> str:
    """Where an open entry names why it is open: its reason and its hints."""
    return " ".join([entry.reason or "", *map(str, entry.hints)])
