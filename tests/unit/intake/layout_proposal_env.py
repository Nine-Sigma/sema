"""Shared setup for the layout proposal tests: synthetic landings, a scripted model, readers.

``sema.intake.proposal`` is imported lazily, so a missing module fails each test on its own
instead of failing collection.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from sema.intake.landing import LOCATOR_COLUMN
from sema.intake.profile import EvidencePolicy, ProfileTools
from sema.intake.verify.base import Failure
from tests.unit.intake.model_calls_env import WiredRun, stats
from tests.unit.intake.shapes.shapes_env import Source, write_workbook
from tests.unit.intake.structure_env import fresh_tools, ref

NONE = "none"
SLOT_PREFIX = "slot_"
LITERAL_PREFIX = "literal_"
CONFIDENCE = 0.85
REASON = "the relation reads as one flat list of labelled codes"
UNSCRIPTED = "no template over any view reads this relation"
PROMPT_HEADER = "# prompt: layout v2"

LISTING, REMARKS, TERMS = "book__listing", "book__remarks", "book__terms"
# Integer ids first: no row passes the header test, so the sheet lands with positional
# columns and is ignored as no_header, while concept_list would read it.
LISTING_ROWS: list[list[object]] = [
    [1, "Alder grove"], [2, "Birch hollow"], [3, "Cedar ridge"], [4, "Dune flat"],
]
# One repeating text column: no ignore reason and no template, so it is left no_template.
REMARK_ROWS: list[list[object]] = [["remark"], ["alpha"], ["alpha"], ["beta"], ["beta"]]
# A headed concept list code binds on its own, so it is never a proposal target.
TERM_ROWS: list[list[object]] = [
    ["id", "label"], [10, "Elm spire"], [11, "Fir basin"], [12, "Gorse moor"],
]

CODEBOOK = "codebook"
CODEBOOK_COLUMNS = ("variable", "description", "code", "label")
VARIABLE = "variable"
# One variable per first row and its codes on continuation rows with the variable empty.
# A label shared by two codes defeats level_columns, so no template reads it as landed.
CONTINUATION_ROWS = (
    ",,2,green\n,,3,blue\n"
    "size,overall size,1,small\n,,2,medium\n,,3,large\n,,9,unknown\n"
    "shape,outline shape,1,round\n,,2,square\n,,3,unknown\n,,9,unknown\n"
)
CODEBOOK_CSV = ",".join(CODEBOOK_COLUMNS) + "\ncolour,surface colour,1,red\n" + CONTINUATION_ROWS
# The first landed row has no variable, so fill_down has nothing to fill from.
UNANCHORED_CSV = (
    ",".join(CODEBOOK_COLUMNS)
    + "\n,surface colour,1,red\ncolour,,2,green\n"
    + CONTINUATION_ROWS.split("\n", 1)[1]
)
# Cell values only a value read can put into a prompt.
CODEBOOK_VALUES = ("colour", "medium")

Answer = Callable[[type[BaseModel]], BaseModel]


def proposal() -> ModuleType:
    return importlib.import_module("sema.intake.proposal")


def write_book(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sheets = {"listing": LISTING_ROWS, "remarks": REMARK_ROWS, "terms": TERM_ROWS}
    return write_workbook(directory / "book.xlsx", sheets)


def write_codebook(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{CODEBOOK}.csv").write_text(text, encoding="utf-8")
    return directory


@dataclass
class Landed:
    source: Source
    structure: Any

    def name(self, relation: str) -> str:
        return ref(self.source, relation)

    def columns(self, relation: str) -> tuple[str, ...]:
        landed = self.source.land.relation(relation).columns
        return tuple(name for name, _ in landed if name != LOCATOR_COLUMN)

    def group_of(self, relation: str) -> set[str]:
        name = self.name(relation)
        (group,) = [group for group in self.structure.groups if name in group]
        return set(group)


def answer(
    model: type[BaseModel],
    *,
    kind: str = "propose",
    template: str = NONE,
    view: str = NONE,
    view_columns: Sequence[str] = (),
    slots: Mapping[str, str] | None = None,
    literals: Mapping[str, str] | None = None,
    relations: Sequence[str] = (),
    needs_values: Sequence[str] = (),
    reason: str = REASON,
) -> BaseModel:
    """A LayoutProposal_v1 instance with every slot and literal field unset unless named."""
    prefixes = (SLOT_PREFIX, LITERAL_PREFIX)
    fields = {name: NONE for name in model.model_fields if name.startswith(prefixes)}
    fields |= {f"{SLOT_PREFIX}{slot}": column for slot, column in (slots or {}).items()}
    fields |= {f"{LITERAL_PREFIX}{slot}": value for slot, value in (literals or {}).items()}
    return model(
        answer=kind,
        template=template,
        view=view,
        view_columns=list(view_columns),
        relations=list(relations),
        needs_values=list(needs_values),
        confidence=CONFIDENCE,
        reason=reason,
        **fields,
    )


def propose(
    template: str,
    relation: str,
    *,
    view: str = NONE,
    view_columns: Sequence[str] = (),
    **slots: str,
) -> Answer:
    def _answer(model: type[BaseModel]) -> BaseModel:
        return answer(
            model,
            template=template,
            relations=[relation],
            view=view,
            view_columns=view_columns,
            slots=slots,
        )

    return _answer


def abstain(reason: str = UNSCRIPTED) -> Answer:
    return lambda model: answer(model, kind="abstain", reason=reason)


def needs(*columns: str) -> Answer:
    return lambda model: answer(model, kind="needs_values", needs_values=columns)


def enum(model: type[BaseModel], field: str) -> tuple[str, ...]:
    schema = model.model_json_schema()["properties"][field]
    return tuple(schema.get("items", schema)["enum"])


def columns_of(model: type[BaseModel]) -> tuple[str, ...]:
    """The relation a proposal call is for, read from the columns its model enumerates."""
    return enum(model, "view_columns")


def scripted(script: Mapping[tuple[str, ...], Sequence[Answer]]) -> MagicMock:
    """A mocked `LLMClient` answering each relation's calls in order; unscripted calls abstain."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    pending = {columns: list(answers) for columns, answers in script.items()}

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        queue = pending.get(columns_of(output_model), [])
        respond = queue.pop(0) if queue else abstain()
        return respond(output_model)

    client.invoke.side_effect = _invoke
    return client


def invokes(client: MagicMock) -> list[tuple[str, type[BaseModel]]]:
    return [(str(call.args[0]), call.args[1]) for call in client.invoke.call_args_list]


def invokes_for(client: MagicMock, columns: tuple[str, ...]) -> list[tuple[str, Any]]:
    return [(prompt, model) for prompt, model in invokes(client) if columns_of(model) == columns]


def run_proposals(landed: Landed, wired: WiredRun, tools: ProfileTools) -> Any:
    return proposal().propose_layouts(
        landed.structure, landed.source.land, tools, wired.calls, provider=landed.source.provider
    )


def tools_for(landed: Landed, ledger_dir: Path) -> ProfileTools:
    return fresh_tools(landed.source, ledger_dir)


def approved_tools(
    landed: Landed, ledger_dir: Path, relation: str, columns: Iterable[str]
) -> ProfileTools:
    """The run's tools once the value ask for ``columns`` of ``relation`` is approved."""
    base = fresh_tools(landed.source, ledger_dir)
    allowlist = frozenset((relation, column) for column in columns)
    policy = EvidencePolicy(tier="bounded_values", allowlist=allowlist)
    return ProfileTools(base.provider, policy, base.ledger, stage=base.stage, land=base.land)


def checks(failures: Iterable[Failure]) -> set[str]:
    """Each failure's check, and the template check a Structure verifier failure wraps."""
    found: set[str] = set()
    for failure in failures:
        found.add(failure.check)
        if "template_check" in failure.evidence:
            found.add(str(failure.evidence["template_check"]))
    return found


def bound_over(structure: Any, relation: str) -> list[Any]:
    return [entry for entry in structure.bound if relation in entry.binding.relations]


def ignored_over(structure: Any, relation: str) -> list[Any]:
    return [binding for binding in structure.ignored if relation in binding.relations]


def open_over(structure: Any, relation: str) -> list[Any]:
    return [a for a in structure.ambiguities if a.subject == relation and a.choice is None]
