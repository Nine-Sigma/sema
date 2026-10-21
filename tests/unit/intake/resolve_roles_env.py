"""Shared setup for the role resolution tests: synthetic sources, a mocked model, readers.

Each synthetic source pins its structure by hand (every relation a code list with its
section stated, ``source: human``), so only roles are open when Roles runs.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import duckdb
from pydantic import BaseModel

from sema.intake.output_models import REFERENCE_PREFIX, ROLE_PREFIX
from sema.intake.output_models_utils import ABSTAIN
from sema.intake.resolve_utils import text_columns
from sema.intake.shapes import entity_model
from sema.intake.shapes.base import SlotBinding
from sema.intake.structure import BoundEntry
from tests.unit.intake.eligibility_env import Env, run_roles
from tests.unit.intake.model_calls_env import stats
from tests.unit.intake.shapes.shapes_env import warehouse_source
from tests.unit.intake.structure_env import ACRIS, CODE_TABLES, fresh_tools, run

CONFIDENCE = 0.85
REASON = "the column profile fits the role"
SCHEMA = "tq"
ENTITY_ROLES = ("property", "description")
SETTLED_BY_NONE = "settled_by: none"

Rows = list[tuple[str, ...]]
Tables = Mapping[str, tuple[tuple[str, ...], Rows]]
Picker = Callable[[str, tuple[str, ...], set[str]], str]
ReferencePicker = Callable[[tuple[str, ...]], str]


def _same_rows() -> Rows:
    return [(f"Z{i:02d}", f"Quillon item {i}") for i in range(1, 21)]


MEMORY_TABLES: Tables = {
    "alpha": (("zorb", "quax"), _same_rows()),
    "beta": (("zorb", "quax"), _same_rows()),
    "gamma": (("zorb", "quax"), _same_rows()),
    "epsilon": (
        ("tarn", "vell", "mox"),
        [(f"T{i:02d}", f"Vell entry {i}", f"Mox {i % 15}") for i in range(1, 21)],
    ),
}
# `pell` is the only low-cardinality column, so code settles `status` on it.
CONFLICT_TABLES: Tables = {
    "zeta": (
        ("zorb", "quax", "pell"),
        [(f"C{i:02d}", f"Quill {i}", "active" if i % 2 else "retired") for i in range(1, 21)],
    ),
}
# `kelt` is the only key, so code settles `code` and excludes it from `display`; `wim`
# holds four empty strings in twenty rows, above the display null-rate floor.
VALUE_TABLES: Tables = {
    "eta": (
        ("kelt", "wim", "rusk"),
        [(f"B{i:02d}", "" if i <= 4 else f"Marrow {i}", f"Rusk {i % 12}") for i in range(1, 21)],
    ),
}

# `lexa` is a dictionary whose `Slant` text names both code lists, so its domain stays open.
DICTIONARY = "lexa"
SLANT = "Slant"
REFERENCE_TABLES: Tables = {
    "ribs": (("code", "label"), [(f"R{i:02d}", f"Rib entry {i}") for i in range(1, 21)]),
    "fens": (("code", "label"), [(f"F{i:02d}", f"Fen entry {i}") for i in range(1, 21)]),
    DICTIONARY: (
        ("property", "description"),
        [(SLANT, "Coded by the ribs list or by the fens list")]
        + [(f"Heft{i}", f"A plain amount number {i}") for i in range(1, 12)],
    ),
}


def relation(name: str) -> str:
    return f"{SCHEMA}.{name}"


def _create(con: duckdb.DuckDBPyConnection, name: str, columns: tuple[str, ...], rows: Rows) -> None:
    values = ", ".join("(" + ", ".join(f"'{cell}'" for cell in row) + ")" for row in rows)
    con.execute(
        f"CREATE TABLE {relation(name)} AS SELECT * FROM (VALUES {values})"
        f" t({', '.join(columns)})"
    )


def _code_list(name: str) -> BoundEntry:
    binding = SlotBinding(
        template="concept_list",
        slots={"relation": relation(name)},
        relations=(relation(name),),
        section=f"code_lists.{name}",
    )
    return BoundEntry(binding=binding, confidence=1.0, source="human")


def _dictionary(name: str) -> BoundEntry:
    binding = SlotBinding(
        template=entity_model.ID,
        slots={
            entity_model.LAYOUT: entity_model.PER_FILE,
            entity_model.DICTIONARY: relation(name),
        },
        relations=(relation(name),),
        section=entity_model.SECTION,
    )
    return BoundEntry(binding=binding, confidence=1.0, source="human")


def _pinned(name: str, dictionaries: frozenset[str]) -> BoundEntry:
    return _dictionary(name) if name in dictionaries else _code_list(name)


class Synthetic:
    """A warehouse source, its hand-pinned structure, and its Roles result."""

    def __init__(
        self, tables: Tables, root: Path, dictionaries: frozenset[str] = frozenset()
    ) -> None:
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"CREATE SCHEMA {SCHEMA}")
        for name, (columns, rows) in tables.items():
            _create(self.con, name, columns, rows)
        self.source = warehouse_source(self.con, SCHEMA, root / "ledger" / "landing")
        found = run(self.source, fresh_tools(self.source, root / "ledger" / "structure"))
        bound = tuple(_pinned(name, dictionaries) for name in tables)
        self.structure = found.model_copy(update={"bound": bound, "ambiguities": ()})
        self.roles = run_roles(self.source, self.structure, root / "ledger" / "roles")

    def close(self) -> None:
        self.con.close()


def role_fields(output_model: type[BaseModel]) -> list[str]:
    return [name for name in output_model.model_fields if name.startswith(ROLE_PREFIX)]


def reference_fields(output_model: type[BaseModel]) -> list[str]:
    return [name for name in output_model.model_fields if name.startswith(REFERENCE_PREFIX)]


def enum_of(output_model: type[BaseModel], field: str) -> tuple[str, ...]:
    enum = output_model.model_json_schema()["properties"][field]["enum"]
    return tuple(value for value in enum if value != ABSTAIN)


def first_member(members: tuple[str, ...]) -> str:
    return members[0]


def answering(pick: Picker, reference: ReferencePicker = first_member) -> MagicMock:
    """A mocked `LLMClient` answering every role field of a call with `pick` and every
    reference field with `reference`; each answer is kept on `client.answers` by field."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    client.answers = []

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        used: set[str] = set()
        chosen = {}
        for field in role_fields(output_model):
            choice = pick(field.removeprefix(ROLE_PREFIX), enum_of(output_model, field), used)
            chosen[field] = choice
            used.add(choice)
        for field in reference_fields(output_model):
            chosen[field] = reference(enum_of(output_model, field))
        count = len(chosen)
        client.answers.append(chosen)
        return output_model(**chosen, confidences=[CONFIDENCE] * count, reasons=[REASON] * count)

    client.invoke.side_effect = _invoke
    return client


def distinct(preferred: Mapping[str, str] | None = None) -> Picker:
    """A role's preferred column when offered, else the first column no role took yet."""

    def _pick(role: str, members: tuple[str, ...], used: set[str]) -> str:
        wanted = (preferred or {}).get(role)
        if wanted in members:
            return str(wanted)
        free = [member for member in members if member not in used]
        return free[0] if free else ABSTAIN

    return _pick


def shared_column(role: str, members: tuple[str, ...], used: set[str]) -> str:
    """Every role takes a column already taken, so two roles name one column."""
    taken = [member for member in members if member in used]
    return taken[0] if taken else members[0]


def prompts(client: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in client.invoke.call_args_list]


def output_models(client: MagicMock) -> list[type[BaseModel]]:
    return [call.args[1] for call in client.invoke.call_args_list]


def entry(result: Any, name: str, role: str) -> Any:
    (found,) = [a for a in result.ambiguities if a.subject == f"{name}.{role}"]
    return found


def lines_with(prompt: str, *words: str) -> list[str]:
    return [line for line in prompt.splitlines() if all(word in line for word in words)]


def opened(roles: Any, relations: set[str]) -> Any:
    """The roles with every listed relation-level entity role of ``relations`` open, as a
    source whose column names carry no lexicon token would leave them."""

    def _target(e: Any) -> bool:
        listed = e.relation in relations and e.role in ENTITY_ROLES and e.subject is None
        return listed and len(e.members) > 1

    clear = dict(choice=None, confidence=None, source=None)
    eligibilities = tuple(
        e.model_copy(update={**clear, "hints": ()}) if _target(e) else e
        for e in roles.eligibilities
    )
    subjects = {f"{e.relation}.{e.role}" for e in roles.eligibilities if _target(e)}
    ambiguities = tuple(
        a.model_copy(update={**clear, "hints": (SETTLED_BY_NONE,)}) if a.subject in subjects
        else a
        for a in roles.ambiguities
    )
    return replace(roles, eligibilities=eligibilities, ambiguities=ambiguities)


def schema_text(env: Env, dictionary: str, root: Path) -> tuple[str, str, str]:
    """All text of ``dictionary`` and its header, then one description from each."""
    source = env.acris.source
    (binding,) = [e.binding for e in env.a10.bound if dictionary in e.binding.relations]
    header = next(name for name in binding.relations if name != dictionary)
    tools = fresh_tools(source, root / "schema_text", stage="roles", structure=env.a10)
    by_name = {profile.ref.name: profile for profile in source.profiles()}
    rows = [
        row
        for name in (dictionary, header)
        for row in tools.read_schema_text(by_name[name].ref, text_columns(by_name[name]))
    ]
    text = "\n".join(str(cell) for row in rows for cell in row if cell is not None)
    described = tools.read_schema_text(by_name[dictionary].ref, ["Column Description"])
    column_text = next(str(row[0]) for row in described if row[0] and len(str(row[0])) >= 5)
    dataset_text = env.a10_roles.headers[header].description
    assert dataset_text, header
    return text, column_text, str(dataset_text)


def code_table_values(env: Env) -> set[str]:
    """Cells of the code tables, which are data, not schema, long enough to be telling."""
    relations = env.acris.source.land.relations
    names = " ".join(
        name for r in relations for name in (r.name, *(column for column, _ in r.columns))
    ).lower()
    cells: set[str] = set()
    for table in CODE_TABLES:
        with (ACRIS / f"{table}.csv").open(newline="", encoding="utf-8") as handle:
            cells |= {cell.strip() for row in csv.reader(handle) for cell in row}
    return {cell for cell in cells if len(cell) >= 5 and cell.lower() not in names}
