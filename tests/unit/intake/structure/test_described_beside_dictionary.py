"""Done-when 1 (N3): a data relation beside its describer is not read as a dictionary.

Inside one `entity_model` candidate, when the relation in the `header` slot has a key column
that lists exactly the column names of the relation in the `dictionary` slot, that relation is
the data its header describes. The candidate is dropped, so Structure does not bind the data
relation as a dictionary by code, and `verify_structure` fails a pinned or a proposed binding of
that shape with the existing described-data failure (`in_eligibility_set` naming its
describer). The match is `described_by`'s, unchanged: exact names after the lexicon's
normalization. Decoys keep today's result: a header whose key column lists other names, one
that lists only some of the columns, and a relation a bound dictionary describes.

The stand-in mirrors `harness/n1-helpers/bug1137_repro.py` with neutral names: a data relation
and a two-column description relation in one stem group of an in-memory DuckDB schema.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.intake.structure.shapes.base import SlotBinding
from sema.intake.structure.structure import BoundEntry
from tests.unit.intake.structure.shapes.shapes_env import Source, warehouse_source
from tests.unit.intake.structure.shapes.test_described_data import (  # noqa: F401 - a fixture
    DESCRIBED,
    DICTIONARY,
    HEADER,
    catalog,
)
from tests.unit.intake.structure_env import entry_over, fresh_tools, run
from tests.unit.intake.verify.verify_env import tools_for
from tests.unit.intake.verify.verify_env import verifier as verifier_module

pytestmark = pytest.mark.unit

SCHEMA = "sb"
DATA = "shelf__items"
DESCRIBER = "shelf__fields"
DATA_COLUMNS = ("item_code", "item_label", "item_note")
DATA_ROWS = (
    ("I1", "Anchor", "Holds fast"), ("I2", "Bell", "Rings twice"), ("I3", "Candle", "Burns slow"),
    ("I4", "Drum", "Beats loud"), ("I5", "Easel", "Stands tall"),
)
OTHER_NAMES = ("part_code", "part_label", "part_note")
SOME_NAMES = DATA_COLUMNS[:2]
TODAY_SLOTS = {"layout": "per_file", "dictionary": f"{SCHEMA}.{DATA}",
               "header": f"{SCHEMA}.{DESCRIBER}"}  # fmt: skip

Build = Callable[..., Source]


def _table(name: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    values = ", ".join("(" + ", ".join(f"'{cell}'" for cell in row) + ")" for row in rows)
    quoted = ", ".join(f'"{c}"' for c in columns)
    return f"CREATE TABLE {SCHEMA}.{name} AS SELECT * FROM (VALUES {values}) t({quoted})"


@pytest.fixture
def shelf(tmp_path: Path) -> Iterator[Build]:
    connections: list[duckdb.DuckDBPyConnection] = []

    def build(listed: Sequence[str] = DATA_COLUMNS) -> Source:
        con = duckdb.connect(":memory:")
        connections.append(con)
        con.execute(f"CREATE SCHEMA {SCHEMA}")
        con.execute(_table(DATA, DATA_COLUMNS, DATA_ROWS))
        con.execute(_table(DESCRIBER, ("Column", "Description"),
                           [(name, f"Describes {name}") for name in listed]))
        return warehouse_source(con, SCHEMA, tmp_path / "ledger" / "landing")

    yield build
    for con in connections:
        con.close()


def _structure(source: Source, root: Path) -> Any:
    return run(source, fresh_tools(source, root / "ledger" / "structure"))


def _reads_as_dictionary(result: Any, relation: str) -> list[str]:
    bound = [f"bound {e.binding.template}" for e in result.bound
             if e.binding.slots.get("dictionary") == relation]
    members = [m for a in result.ambiguities for m in a.members if f"dictionary={relation}" in m]
    return bound + members


def _verify(source: Source, result: Any, root: Path) -> Any:
    tools = tools_for(source, root / "ledger" / "verify", stage="structure")
    return verifier_module("structure").verify_structure(
        result, source.land, tools, provider=source.provider)


def _stated(result: Any, source_kind: str) -> Any:
    """The result with the data relation and its describer bound in today's shape."""
    names = set(TODAY_SLOTS.values()) - {"per_file"}
    entry = {"binding": SlotBinding(template="entity_model", slots=dict(TODAY_SLOTS),
                                    relations=(f"{SCHEMA}.{DATA}", f"{SCHEMA}.{DESCRIBER}")),
             "confidence": 1.0, "source": source_kind}
    kept = tuple(e for e in result.bound if not names & set(e.binding.relations))
    if source_kind == "model":
        entry["call_key"] = "structure-proposal"
    return result.model_copy(update={
        "bound": (*kept, BoundEntry(**entry)),
        "ignored": tuple(b for b in result.ignored if not names & set(b.relations)),
        "ambiguities": tuple(a for a in result.ambiguities if a.subject not in names),
        "proposal_targets": (f"{SCHEMA}.{DATA}",) if source_kind == "model" else (),
    })


def _described_failures(verdict: Any) -> list[tuple[str, str]]:
    return [(f.relation, f.evidence["described_by"]) for f in verdict.failures
            if f.check == "in_eligibility_set" and "described_by" in f.evidence]


def test_the_data_relation_beside_its_describer_is_not_bound_as_a_dictionary(
    shelf: Build, tmp_path: Path,
) -> None:
    result = _structure(shelf(), tmp_path)

    assert _reads_as_dictionary(result, f"{SCHEMA}.{DATA}") == []


@pytest.mark.parametrize("source_kind", ["human", "model"], ids=["pinned", "proposed"])
def test_the_verifier_fails_a_binding_that_reads_the_described_data_as_a_dictionary(
    shelf: Build, tmp_path: Path, source_kind: str,
) -> None:
    source = shelf()
    stated = _stated(_structure(source, tmp_path), source_kind)

    verdict = _verify(source, stated, tmp_path)

    assert _described_failures(verdict) == [(f"{SCHEMA}.{DATA}", f"{SCHEMA}.{DESCRIBER}")]


@pytest.mark.parametrize("listed", [OTHER_NAMES, SOME_NAMES], ids=["other_names", "some_names"])
def test_decoy_a_header_not_listing_exactly_the_columns_keeps_todays_binding(
    shelf: Build, tmp_path: Path, listed: tuple[str, ...],
) -> None:
    source = shelf(listed)
    result = _structure(source, tmp_path)

    entry = entry_over(result, f"{SCHEMA}.{DATA}")
    assert (entry.binding.template, entry.binding.slots, entry.source) == (
        "entity_model", TODAY_SLOTS, "code")
    assert _described_failures(_verify(source, _stated(result, "human"), tmp_path)) == []


def test_decoy_a_relation_a_bound_dictionary_describes_resolves_as_today(
    catalog: Build, tmp_path: Path,  # noqa: F811 - the fixture
) -> None:
    source = catalog()
    result = _structure(source, tmp_path)

    listed = entry_over(result, f"dd.{DESCRIBED}")
    dictionary = entry_over(result, f"dd.{DICTIONARY}")
    assert (listed.binding.template, listed.source) == ("concept_list", "code")
    assert dictionary.binding.slots == {"layout": "per_file", "dictionary": f"dd.{DICTIONARY}",
                                        "header": f"dd.{HEADER}"}
    assert _described_failures(_verify(source, result, tmp_path)) == []
