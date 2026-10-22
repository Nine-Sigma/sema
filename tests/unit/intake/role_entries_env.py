"""Shared setup for the role-entry tests: synthetic sources, a schema-driven mocked model.

The mocked model answers every enum field of whatever output model it is handed, so the
tests do not depend on how a field is named, only on the members it enumerates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from sema.intake.decisions import Decision, apply_structure_decisions
from sema.intake.eligibility_utils import subject_of
from sema.intake.lexicon import NO_REFERENCE
from sema.intake.output_models_utils import ABSTAIN
from sema.intake.resolve import resolve_roles
from sema.intake.shapes.base import SlotBinding
from sema.intake.shapes.precedence import member
from sema.intake.structure import BoundEntry
from tests.unit.intake.eligibility_env import memory_source
from tests.unit.intake.eligibility_env import run_roles as roles_over
from tests.unit.intake.model_calls_env import WiredRun, stats, wired_run
from tests.unit.intake.shapes.shapes_env import Source
from tests.unit.intake.structure_env import fresh_tools, run

CONFIDENCE = 0.85
REASON = "the evidence fits this member"
ROLES = "roles"
DETERMINISTIC = ("code", "lexicon")
NONE = NO_REFERENCE

Pick = Callable[[str, tuple[str, ...]], str]
EnumSet = tuple[str, ...]

# A denormalized catalog: three groups of three rows. `group_band` is a function of the
# group, `item_band` varies inside a group, so each side holds one open optional `status`.
DENORMALIZED = "concept_with_values_denormalized"
CATALOG = "dn.catalog"
GROUP_KEY, GROUP_BAND, ITEM_BAND = "group_key", "group_band", "item_band"
_GROUPS = (
    ("Colour", "The colour a surface shows under daylight", "P"),
    ("Size", "The overall extent of an object in its longest axis", "Q"),
    ("Shape", "The outline an object traces when seen from above", "R"),
)
_CATALOG_ROWS = ", ".join(
    f"('G{g}', '{label}', '{text}', '{band}', 'I{g}{j}', 'Item {g}{j} name',"
    f" '{'A' if j % 2 else 'B'}')"
    for g, (label, text, band) in enumerate(_GROUPS, 1)
    for j in range(1, 4)
)
CATALOG_SQL = (
    "CREATE SCHEMA dn",
    f"CREATE TABLE {CATALOG} AS SELECT * FROM (VALUES {_CATALOG_ROWS})"
    " t(group_key, group_label, group_description, group_band, item_code, item_label,"
    " item_band)",
)

# A code list whose two key columns both carry the `code` token, so `code` stays open,
# while `concept_code` and `display` settle by the lexicon on columns `code` also offers.
TERMS = "cc.terms"
ITEM_CODE, ITEM_CONCEPT_CODE, ITEM_DISPLAY = "item_code", "item_concept_code", "item_display"
# No column name of `loose` carries a role token, so its column roles stay open. Every
# column role of `plain` settles without the model, so only its section stays open.
LOOSE, PLAIN = "cc.loose", "cc.plain"
TERMS_SQL = (
    "CREATE SCHEMA cc",
    f"CREATE TABLE {TERMS} AS SELECT 'M' || lpad(CAST(i AS VARCHAR), 2, '0') AS item_code,"
    " 'X' || CAST(i AS VARCHAR) AS item_concept_code,"
    " 'Term entry ' || CAST(i AS VARCHAR) AS item_display FROM range(1, 21) r(i)",
    f"CREATE TABLE {LOOSE} AS SELECT 'Z' || lpad(CAST(i AS VARCHAR), 2, '0') AS zorb,"
    " 'Quillon item ' || CAST(i AS VARCHAR) AS quax FROM range(1, 21) r(i)",
    f"CREATE TABLE {PLAIN} AS SELECT 'P' || lpad(CAST(i AS VARCHAR), 2, '0') AS plain_code,"
    " 'Lb ' || CAST(i % 5 AS VARCHAR) AS plain_label FROM range(1, 21) r(i)",
)

# Two relations with one signature, neither section stated.
KIN = ("kn.first", "kn.second")
KIN_SQL = (
    "CREATE SCHEMA kn",
    *(
        f"CREATE TABLE {name} AS SELECT 'Z' || lpad(CAST(i AS VARCHAR), 2, '0') AS zorb,"
        " 'Quillon item ' || CAST(i AS VARCHAR) AS quax FROM range(1, 21) r(i)"
        for name in KIN
    ),
)


def source_of(statements: Sequence[str], schema: str, root: Path) -> tuple[Any, Source]:
    return memory_source(statements, schema, root)


def pinned(source: Source, bindings: Iterable[SlotBinding], root: Path) -> Any:
    """Structure with exactly ``bindings`` bound by a human and nothing left open."""
    found = run(source, fresh_tools(source, root))
    bound = tuple(BoundEntry(binding=b, confidence=1.0, source="human") for b in bindings)
    return found.model_copy(update={"bound": bound, "ambiguities": ()})


def concept_list(relation: str, section: str | None) -> SlotBinding:
    return SlotBinding(
        template="concept_list", slots={"relation": relation}, relations=(relation,),
        section=section,
    )


def denormalized(relation: str, group_by: str) -> SlotBinding:
    return SlotBinding(
        template=DENORMALIZED, section="concepts",
        slots={"relation": relation, "group_by": group_by}, relations=(relation,),
    )


def decided(source: Source, structure: Any, choose: Callable[[Any], str], root: Path) -> Any:
    """Structure with each open entry settled by a human: a member string or ``ignore``."""
    decisions = [
        Decision(stage="structure", subject=a.subject, choice=choose(a))
        for a in structure.ambiguities
    ]
    tools = fresh_tools(source, root)
    return apply_structure_decisions(
        structure, decisions, source.land, tools, provider=source.provider
    )


def member_of(binding: SlotBinding) -> str:
    return member(binding)


def roles_of(source: Source, structure: Any, root: Path, **kwargs: Any) -> Any:
    return roles_over(source, structure, root, **kwargs)


def enum_fields(output_model: type[BaseModel]) -> dict[str, EnumSet]:
    """Each enum field of an output model and its members, without ``abstain``."""
    properties = output_model.model_json_schema()["properties"]
    return {
        name: tuple(v for v in spec["enum"] if v != ABSTAIN)
        for name, spec in properties.items()
        if "enum" in spec
    }


def answering(pick: Pick) -> MagicMock:
    """A mocked `LLMClient` answering every enum field with ``pick(field, members)``."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    client.answers = []

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        fields = enum_fields(output_model)
        chosen = {field: pick(field, members) for field, members in fields.items()}
        client.answers.append(chosen)
        count = len(chosen)
        return output_model(**chosen, confidences=[CONFIDENCE] * count, reasons=[REASON] * count)

    client.invoke.side_effect = _invoke
    return client


def abstaining(field: str, members: tuple[str, ...]) -> str:
    return ABSTAIN


def prompts(client: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in client.invoke.call_args_list]


def output_models(client: MagicMock) -> list[type[BaseModel]]:
    return [call.args[1] for call in client.invoke.call_args_list]


def resolved(
    source: Source, structure: Any, roles: Any, client: MagicMock, root: Path
) -> tuple[Any, WiredRun]:
    wired = wired_run(root / "calls", client)
    tools = fresh_tools(source, root / "resolve", stage=ROLES, structure=structure)
    result = resolve_roles(
        roles, structure, source.land, tools, wired.calls, provider=source.provider
    )
    return result, wired


def entry(result: Any, subject: str) -> Any:
    (found,) = [a for a in result.ambiguities if a.subject == subject]
    return found


def offered(ambiguity: Any) -> EnumSet:
    return tuple(m for m in ambiguity.members if m not in ambiguity.exclusions)


def open_column_entries(roles: Any) -> list[Any]:
    """Open role entries with something to choose: not a per-property reference entry."""
    per_property = {subject_of(e) for e in roles.eligibilities if e.subject is not None}
    return [
        a for a in roles.ambiguities
        if a.stage == ROLES and a.choice is None and a.source is None
        and a.subject not in per_property and offered(a)
    ]


def enum_counter(output_model: type[BaseModel]) -> Counter[frozenset[str]]:
    return Counter(frozenset(members) for members in enum_fields(output_model).values())


def claims(roles: Any) -> dict[tuple[str, str | None, str], str]:
    """Column-role claims settled without the model, by (relation, section, role)."""
    return {
        (e.relation, e.section, e.role): str(e.choice)
        for e in roles.eligibilities
        if e.subject is None and e.source in DETERMINISTIC and e.choice not in (None, NONE)
    }


def eligibility(result: Any, relation: str, section: str | None, role: str) -> Any:
    (found,) = [
        e for e in result.eligibilities
        if e.relation == relation and e.section == section and e.role == role
        and e.subject is None
    ]
    return found


def field_lines(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if line.startswith("field:")]
