"""Shared setup for the prompt budget tests: synthetic wide relations built in SQL.

The structure is pinned by hand, so the structure stage's pairwise join checks never run
over thousands of columns. Every relation is generated with ``range``, so its row order on
disk is set by the test (ascending or descending) and a schema-text order is observable.
Every name is generic (``col_0000``); nothing here is taken from a fixture.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb
from pydantic import BaseModel

from sema.intake.output_models import ROLE_PREFIX
from sema.intake.output_models_utils import ABSTAIN
from sema.intake.shapes import entity_model
from sema.intake.shapes.base import SlotBinding
from sema.intake.structure import STEPS, BoundEntry, StructureResult, stage_floors
from sema.llm_client_utils import approx_tokens
from tests.unit.intake.eligibility_env import run_roles
from tests.unit.intake.model_calls_env import WiredRun, intake_config, stats, wired_run
from tests.unit.intake.shapes.shapes_env import Source, warehouse_source
from tests.unit.intake.structure_env import fresh_tools

SCHEMA = "pb"
ROWS = 20
REASON = "the evidence does not separate the members"
CONFIDENCE = 0.5
REFUSED_EVENT = "call_refused"
LABELS = 3

Statement = str


def relation(name: str) -> str:
    return f"{SCHEMA}.{name}"


def column_name(index: int) -> str:
    return f"col_{index:04d}"


def budget() -> int:
    """The policy constant; imported here so its absence fails each test on its own."""
    module = importlib.import_module("sema.intake.model_calls")
    value = module.MAX_PROMPT_TOKENS
    assert isinstance(value, int) and value > 0, value
    return value


def schema_json(output_model: type[BaseModel]) -> str:
    return json.dumps(output_model.model_json_schema())


def size(prompt: str, output_model: type[BaseModel]) -> int:
    """What reaches the provider, the prompt and its output schema, at the client's own
    four-characters-per-token estimate."""
    return approx_tokens(len(prompt) + len(schema_json(output_model)))


def chars(token_count: int) -> int:
    return token_count * 4


def text_table(name: str, width: int, rows: int = ROWS, start: int = 0) -> Statement:
    """``width`` unique text columns: every one a code and a display candidate."""
    cells = ", ".join(
        f"'v{i:04d}r' || lpad(r::VARCHAR, 3, '0') AS {column_name(i)}"
        for i in range(start, start + width)
    )
    return f"CREATE TABLE {relation(name)} AS SELECT {cells} FROM range({rows}) t(r)"


def mixed_table(name: str, numeric_width: int, rows: int = ROWS) -> Statement:
    """One key (code settles on it and it stays in force), three label columns offered
    for display, then ``numeric_width`` non-unique numbers no open role offers."""
    key = [f"'v' || lpad(r::VARCHAR, 3, '0') AS {column_name(0)}"]
    labels = [
        f"'a generic text value ' || ((r + {i}) % 15)::VARCHAR AS {column_name(i)}"
        for i in range(1, LABELS + 1)
    ]
    numbers = [
        f"(r % 15)::DOUBLE + 0.5 AS {column_name(i)}"
        for i in range(LABELS + 1, LABELS + 1 + numeric_width)
    ]
    return (
        f"CREATE TABLE {relation(name)} AS SELECT {', '.join(key + labels + numbers)}"
        f" FROM range({rows}) t(r)"
    )


def dictionary_table(name: str, rows: int, descending: bool = False) -> Statement:
    """A three-column dictionary: a unique key, a label, and a sentence per row."""
    order = "DESC" if descending else "ASC"
    return (
        f"CREATE TABLE {relation(name)} AS SELECT"
        f" 'k' || lpad(r::VARCHAR, 6, '0') AS {column_name(0)},"
        f" 'generic label ' || lpad(r::VARCHAR, 6, '0') AS {column_name(1)},"
        f" 'a generic sentence describing entry ' || lpad(r::VARCHAR, 6, '0')"
        f" AS {column_name(2)}"
        f" FROM range({rows}) t(r) ORDER BY r {order}"
    )


def long_dictionary_table(name: str, rows: int, sentence: int) -> Statement:
    """A three-column dictionary whose sentence column holds ``sentence`` characters."""
    return (
        f"CREATE TABLE {relation(name)} AS SELECT"
        f" 'k' || lpad(r::VARCHAR, 6, '0') AS {column_name(0)},"
        f" 'generic label ' || lpad(r::VARCHAR, 6, '0') AS {column_name(1)},"
        f" repeat('generic words ', {sentence // 14 + 1}) AS {column_name(2)}"
        f" FROM range({rows}) t(r)"
    )


def dictionary_key(row: int) -> str:
    return f"k{row:06d}"


def code_list(name: str) -> BoundEntry:
    binding = SlotBinding(
        template="concept_list",
        slots={"relation": relation(name)},
        relations=(relation(name),),
        section=f"code_lists.{name}",
    )
    return BoundEntry(binding=binding, confidence=1.0, source="human")


def dictionary(name: str) -> BoundEntry:
    binding = SlotBinding(
        template=entity_model.ID,
        slots={entity_model.LAYOUT: entity_model.PER_FILE, entity_model.DICTIONARY: relation(name)},
        relations=(relation(name),),
        section=entity_model.SECTION,
    )
    return BoundEntry(binding=binding, confidence=1.0, source="human")


class Built:
    """A warehouse source over ``statements``, its hand-pinned structure, and its Roles."""

    def __init__(
        self,
        statements: Sequence[Statement],
        root: Path,
        bound: Callable[[], Sequence[BoundEntry]],
    ) -> None:
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"CREATE SCHEMA {SCHEMA}")
        for statement in statements:
            self.con.execute(statement)
        self.root = root
        self.source: Source = warehouse_source(self.con, SCHEMA, root / "ledger" / "landing")
        profiles = self.source.profiles()
        self.structure = StructureResult(
            order=STEPS,
            groups=tuple((p.ref.name,) for p in profiles),
            signatures={p.ref.name: p.signature for p in profiles},
            bound=tuple(bound()),
            ignored=(),
            ambiguities=(),
            edges=(),
            floors=stage_floors(),
        )
        self.roles = run_roles(self.source, self.structure, root / "ledger" / "roles")

    def close(self) -> None:
        self.con.close()


def abstaining() -> MagicMock:
    """A mocked `LLMClient` that abstains on every choice and role-map field, so no
    verifier re-asks."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        properties = output_model.model_json_schema()["properties"]
        chosen = {
            name: ABSTAIN for name, field in properties.items()
            if "enum" in field or name.startswith(ROLE_PREFIX)
        }
        count = len(chosen)
        return output_model(**chosen, confidences=[CONFIDENCE] * count, reasons=[REASON] * count)

    client.invoke.side_effect = _invoke
    return client


def prompts(client: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in client.invoke.call_args_list]


def output_models(client: MagicMock) -> list[type[BaseModel]]:
    return [call.args[1] for call in client.invoke.call_args_list]


def enum_of(output_model: type[BaseModel], field: str) -> tuple[str, ...]:
    enum = output_model.model_json_schema()["properties"][field]["enum"]
    return tuple(value for value in enum if value != ABSTAIN)


class Watched:
    """A wired run whose every requested call key is kept."""

    def __init__(self, root: Path, client: MagicMock, run_id: str, **config: Any) -> None:
        self.wired: WiredRun = wired_run(root / "calls", client, intake_config(**config), run_id)
        calls = self.wired.calls
        self._call = patch.object(calls, "call", wraps=calls.call)
        self.call = self._call.start()

    def keys(self) -> list[str]:
        calls = self.wired.calls
        return [calls.call_key(c.args[1], c.args[2]) for c in self.call.call_args_list]

    def stop(self) -> None:
        self._call.stop()


def resolve() -> Any:
    return importlib.import_module("sema.intake.resolve")


def resolved_roles(built: Built, roles: Any, watch: Watched, root: Path) -> Any:
    tools = fresh_tools(
        built.source, root / "resolve", stage="roles", structure=built.structure
    )
    try:
        return resolve().resolve_roles(
            roles, built.structure, built.source.land, tools, watch.wired.calls,
            provider=built.source.provider,
        )
    finally:
        watch.stop()


def lines_naming(prompt: str, name: str) -> list[str]:
    return [line for line in prompt.splitlines() if name in line]


Builder = Callable[[int, Path], Built]


def wide(width: int, root: Path) -> Built:
    return Built([text_table("wide", width)], root, lambda: [code_list("wide")])


def mixed(numeric_width: int, root: Path) -> Built:
    return Built([mixed_table("wide", numeric_width)], root, lambda: [code_list("wide")])


def measured(build: Builder, root: Path, widths: tuple[int, int] = (4, 8)) -> tuple[int, int]:
    """The fixed and per-column characters of a Roles prompt and its output schema small
    enough to fit, where every column of a width adds the same text."""
    lengths = []
    for width in widths:
        built = build(width, root / f"calibrate_{width}")
        client = abstaining()
        resolved_roles(built, built.roles, Watched(built.root, client, "calibration"), built.root)
        built.close()
        (prompt,), (output_model,) = prompts(client), output_models(client)
        lengths.append(len(prompt) + len(schema_json(output_model)))
    step = widths[1] - widths[0]
    per_column, rest = divmod(lengths[1] - lengths[0], step)
    assert rest == 0, lengths
    return lengths[0] - widths[0] * per_column, per_column


def width_over(fixed: int, per_column: int, share: float) -> int:
    """The fewest columns whose full rendering is ``share`` times the budget or more."""
    target = chars(budget()) * share - fixed
    return max(1, int(-(-target // per_column)))
