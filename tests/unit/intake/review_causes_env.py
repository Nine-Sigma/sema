"""Synthetic runs that reach Review with bindings and no spec, one per assembly cause.

Three sources, all invented: two workbooks whose per-file dictionaries read the same roles
from differently named columns; a warehouse concept and value set pair that assembles until a
reviewer states ``none`` for its required ``name`` role; and two warehouse tables a mocked
Structure answer binds into the same ``concepts`` section. A raising client stands in for the
provider wherever the run needs no model answer; the books run's Roles answers keep every claim
in force and answer ``none`` elsewhere, so its divergence is the dictionaries' own; the conflict
run's answers are mocked outputs, and the members each Structure ask offered are recorded as
the mock reads them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import duckdb
import openpyxl
from pydantic import BaseModel

from sema.intake.output_models import ROLE_PREFIX
from sema.intake.output_models_utils import ABSTAIN
from sema.intake.registry import Registry
from sema.models.schema_metadata import SchemaMetadata
from tests.unit.intake.catalog_env import (
    WAREHOUSE,
    connector_over,
    warehouse_tables,
    write_descriptor,
)
from tests.unit.intake.decisions_env import write_decisions
from tests.unit.intake.graph_env import (
    APPROVER,
    REGISTRY_FILE,
    ROLE_MAP,
    STRUCTURE_BATCH,
    Answer,
    batch_answer,
    graph,
    model_names,
    routed_client,
    settings,
)
from tests.unit.intake.model_calls_env import stats
from tests.unit.intake.warehouse_env import CATALOG, SCHEMA, FakeConnector

STRUCTURE, ROLES, IGNORE, NONE = "structure", "roles", "ignore", "none"
CLAIMS_HEADING, CLAIMED_ROLE = "## Claims in force", re.compile(r"^- `([^`]+)`:", re.MULTILINE)
REQUIRED_ROLE_NONE = "required_role_none"
CONCEPTS_SECTION = "concepts"
CONCEPT_LIST = "concept_list("
DENORMALIZED = "concept_with_values_denormalized("

BOOKS = ("alpha_book", "beta_book")
DICTIONARY_SHEET, HEADER_SHEET = "Attributes", "About"
DIVERGENT_ROLE = "description"
DICTIONARY_HEADERS = {
    "alpha_book": ("field", "description", "kind"),
    "beta_book": ("field", "meaning", "kind"),
}

FACETS, FACET_VALUES = "facets", "facet_values"
FACET_ROWS: tuple[tuple[Any, ...], ...] = (
    ("facet_code", "facet_name", "definition"),
    ("HUE", "Hue", "Hue of a thing"),
    ("GRIT", "Grit", "Grit of a thing"),
)
FACET_VALUE_ROWS: tuple[tuple[Any, ...], ...] = (
    ("facet_code", "code", "label"),
    ("HUE", "H1", "Pale"),
    ("HUE", "H2", "Deep"),
    ("GRIT", "G1", "Fine"),
)
REQUIRED_ROLE = "name"

SOUNDS, MASSES = "sounds", "masses"
GROUPED_HEADER = ("set_key", "set_label", "set_note", "member_code", "member_label")
SOUND_ROWS: tuple[tuple[Any, ...], ...] = (
    GROUPED_HEADER,
    ("P1", "Pitch", "How high a sound is", "P01", "Low"),
    ("P1", "Pitch", "How high a sound is", "P02", "High"),
    ("P2", "Speed", "How fast a thing moves", "V01", "Slow"),
    ("P2", "Speed", "How fast a thing moves", "V02", "Fast"),
)
MASS_ROWS: tuple[tuple[Any, ...], ...] = (
    GROUPED_HEADER,
    ("Q1", "Weight", "How heavy a thing is", "W01", "Light"),
    ("Q1", "Weight", "How heavy a thing is", "W02", "Heavy"),
    ("Q2", "Depth", "How deep a thing goes", "D01", "Shallow"),
    ("Q2", "Depth", "How deep a thing goes", "D02", "Deep"),
)

Entry = dict[str, Any]
Warehouse = tuple[duckdb.DuckDBPyConnection, SchemaMetadata]


def qualified(table: str) -> str:
    return f"{WAREHOUSE}.{table}"


def raising_client() -> MagicMock:
    """A client any call fails on: the run must reach Review on code and replays alone."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    client.invoke.side_effect = AssertionError("a provider call was made")
    return client


def _claimed(prompt: str) -> set[str]:
    """The roles a Roles prompt states as claims in force, read as a model would."""
    section = prompt.partition(CLAIMS_HEADING)[2].partition("\n## ")[0]
    return set(CLAIMED_ROLE.findall(section))


def _optional_none(model: type[BaseModel], prompt: str) -> BaseModel:
    """Keeps every claim in force and answers ``none`` for every other role it is asked."""
    claimed = _claimed(prompt)
    fields = [name for name in model.model_fields if name.startswith(ROLE_PREFIX)]
    values: dict[str, Any] = {
        name: ABSTAIN if name.removeprefix(ROLE_PREFIX) in claimed else NONE for name in fields
    }
    values |= {"confidences": [0.6] * len(fields), "reasons": ["read from names"] * len(fields)}
    return model(**values)


def _structure_unused(model: type[BaseModel]) -> BaseModel:
    raise AssertionError(f"a Structure call was made: {model.__name__}")


def books_client() -> MagicMock:
    """Mocked Roles answers for the books run (decision 21 allows mocked answers, not calls):
    since task 2.2 each dictionary's whole role map is asked; Structure is never asked."""
    prompts: list[str] = []
    client = routed_client(defaults={
        STRUCTURE_BATCH: _structure_unused,
        ROLE_MAP: lambda model: _optional_none(model, prompts[-1]),
    })
    route = client.invoke.side_effect

    def _invoke(prompt: str, output_model: type[BaseModel], **kwargs: Any) -> BaseModel:
        prompts.append(prompt)
        return route(prompt, output_model, **kwargs)

    client.invoke.side_effect = _invoke
    return client


def asks_by_stage(client: MagicMock) -> tuple[int, int]:
    """``(roles, structure)`` asks the mocked client answered."""
    names = model_names(client)
    return sum(1 for name in names if name.startswith(ROLE_MAP)), names.count(STRUCTURE_BATCH)


def _attribute_rows(book: str) -> list[tuple[Any, ...]]:
    stem = book.partition("_")[0]
    return [
        DICTIONARY_HEADERS[book],
        (f"{stem}_id", f"The {stem} key", "integer"),
        (f"{stem}_name", f"The {stem} name", "text"),
        (f"{stem}_day", f"The {stem} date", "date"),
    ]


def _write_book(path: Path, sheets: Mapping[str, Sequence[tuple[Any, ...]]]) -> None:
    workbook = openpyxl.Workbook()
    for at, (title, rows) in enumerate(sheets.items()):
        sheet = workbook.active if at == 0 else workbook.create_sheet(title)
        assert sheet is not None
        sheet.title = title
        for row in rows:
            sheet.append(list(row))
    workbook.save(path)


def divergent_books(root: Path) -> tuple[Path, Path]:
    """Two workbooks, each a dictionary sheet beside a key and value sheet; the dictionaries
    hold ``description`` under different column names. Returns the source and descriptor."""
    source = root / "inputs" / "books"
    source.mkdir(parents=True, exist_ok=True)
    for book in BOOKS:
        about = [("key", "value"), ("title", f"The {book}"), ("owner", "A team")]
        sheets = {DICTIONARY_SHEET: _attribute_rows(book), HEADER_SHEET: about}
        _write_book(source / f"{book}.xlsx", sheets)
    return source, write_descriptor(root / "inputs")


def dictionary_relation(book: str) -> str:
    """The landed name's tail; the schema comes from the source directory."""
    return f"{book}__{DICTIONARY_SHEET}"


@contextmanager
def warehouse(build: Callable[[], Warehouse]) -> Iterator[FakeConnector]:
    con, metadata = build()
    try:
        yield connector_over(con, metadata)
    finally:
        con.close()


def facet_pair() -> Warehouse:
    return warehouse_tables({FACETS: FACET_ROWS, FACET_VALUES: FACET_VALUE_ROWS})


def grouped_pair() -> Warehouse:
    return warehouse_tables({SOUNDS: SOUND_ROWS, MASSES: MASS_ROWS})


@dataclass
class Offered:
    """Every Structure ask's members, in the order the mock read them, and its choices."""

    asks: list[tuple[str, ...]] = field(default_factory=list)
    chosen: list[str] = field(default_factory=list)

    def for_relation(self, relation: str) -> tuple[str, ...]:
        (found,) = [ask for ask in self.asks if all(relation in m for m in ask)]
        return found

    def choice_for(self, relation: str) -> str:
        (found,) = [choice for choice in self.chosen if relation in choice]
        return found


def choosing_grouped(offered: Offered) -> Answer:
    """Settles every Structure entry on its first grouped member, recording what it saw."""

    def _pick(members: tuple[str, ...]) -> str:
        listed = tuple(member for member in members if member != ABSTAIN)
        offered.asks.append(listed)
        choice = next((m for m in listed if m.startswith(DENORMALIZED)), ABSTAIN)
        offered.chosen.append(choice)
        return choice

    def _answer(model: type[BaseModel]) -> BaseModel:
        return batch_answer(model, _pick)

    return _answer


def grouped_client(offered: Offered) -> MagicMock:
    return routed_client(defaults={STRUCTURE_BATCH: choosing_grouped(offered)})


def discover_warehouse(root: Path, connector: FakeConnector, client: MagicMock) -> Any:
    return graph().discover(
        None, warehouse=connector, catalog=CATALOG, schema=SCHEMA,
        descriptor=write_descriptor(root / "inputs"), client=client, **settings(root),
    )


def approve_warehouse(
    root: Path, run_id: str, connector: FakeConnector, client: MagicMock,
    rows: Sequence[tuple[str, str, str]],
) -> Any:
    decisions = write_decisions(root / "decisions.yaml", rows)
    return graph().approve(
        run_id, approver=APPROVER, client=client, warehouse=connector,
        decisions=decisions, **settings(root),
    )


def discover_books(root: Path, client: MagicMock) -> Any:
    source, descriptor = divergent_books(root)
    return graph().discover(source, descriptor=descriptor, client=client, **settings(root))


def registry_calls(root: Path, run_id: str) -> int:
    return int(Registry(root / REGISTRY_FILE).get(run_id).model_calls)


def review_lists(root: Path, run_id: str) -> list[Entry]:
    """The ambiguity lists Review wrote to the ledger: what a decisions file is checked against."""
    artifacts = root / run_id / "artifacts"
    return [
        entry
        for stage in (STRUCTURE, ROLES)
        for entry in json.loads((artifacts / f"{stage}_ambiguities.json").read_text())[
            "ambiguities"
        ]
    ]


def open_entries(outcome: Any) -> list[Entry]:
    return [
        entry for entry in outcome.payload["ambiguities"]
        if entry["choice"] is None and entry["source"] is None
    ]


def open_count(outcome: Any) -> int:
    metrics = outcome.payload["run_metrics"]
    return int(metrics["open_structure"]) + int(metrics["open_roles"])


def text_of(entry: Entry) -> str:
    """Where an entry can name its relation, role or section, and cause."""
    return " ".join(str(entry.get(key) or "") for key in ("subject", "reason", "cause"))


def naming(entries: Sequence[Entry], *parts: str) -> list[Entry]:
    return [entry for entry in entries if all(part in text_of(entry) for part in parts)]


def summary(entries: Sequence[Entry]) -> list[tuple[str, str, Any]]:
    return [(e["stage"], e["subject"], e.get("reason")) for e in entries]
