"""Shared setup for the structure stage tests: landed sources, hints, and result readers.

``sema.intake.structure`` is imported lazily, so a missing module fails each test on its own
instead of failing collection.
"""

from __future__ import annotations

import importlib
import json
import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.ingest.duckdb_staging import Staging
from sema.intake.landing import LandResult
from sema.intake.profile import ProfileTools
from tests.unit.intake.shapes.shapes_env import FIXTURES, Source, make_tools, staging_source

A2_WORKBOOK = FIXTURES / "tabular" / "codes_a2.xlsx"
A2_SPEC = FIXTURES / "tabular" / "codes_a2.spec.yaml"
A3_DB = FIXTURES / "semsql" / "ontology.db"
ACRIS = FIXTURES / "acris"
ACRIS_SPEC = ACRIS / "acris.spec.yaml"
HELDOUT = FIXTURES / "heldout"

STEPS = ("stem_groups", "profiles", "ignore", "candidates", "join_graph", "binding")
CODE_TABLES = (
    "Country_Codes",
    "Document_Control_Codes",
    "Property_Types_Codes",
    "State_Codes",
    "UCC_Collateral_Codes",
)
GROUPED_CODE_TABLE = "Document_Control_Codes"
# Real and Personal twins whose Column Name lists are identical; measured: Legals, Parties,
# Remarks resolve 1.0 both ways, Master 0.79/0.65 and References 0.44/0.67 do not.
TWIN_SUBJECTS = ("Legals", "Parties", "Remarks")
A2_SHAPE = [("concepts", "concept_and_value_set"), ("hierarchy", "level_columns")]
A2_SHAPE_CONTESTED = (
    "the six steps as written leave A2 ambiguous: edge_list_hierarchy has six edge-relation"
    " bindings over Concepts with Hierarchy as the edge relation (each of l1, l2, l3 resolves"
    " into Concepts.concept_code at 1.0 and every child/parent permutation is acyclic), each a"
    " multi-relation claim on Concepts and Hierarchy; Concepts is then claimed by both"
    " concept_and_value_set and edge_list_hierarchy, Hierarchy by six edge_list_hierarchy"
    " bindings, so level_columns is dropped as a single-relation candidate over a claimed"
    " relation and precedence may not prefer a template by identity; no code-bound shape"
    " [concepts/concept_and_value_set, hierarchy/level_columns] follows without a hint"
)
# Dotted codes beside labels: eligible for concept_list and for path_codes, nothing else.
OUTLINE_SQL = (
    "CREATE SCHEMA tw",
    "CREATE TABLE tw.outline AS SELECT * FROM (VALUES ('1', 'First part'),"
    " ('1.1', 'First part, item one'), ('1.2', 'First part, item two'),"
    " ('2', 'Second part'), ('2.1', 'Second part, item one'), ('3', 'Third part'))"
    " t(code, label)",
)

Lander = Callable[..., LandResult]


def structure() -> ModuleType:
    return importlib.import_module("sema.intake.structure")


def land(root: Path, name: str, lander: Lander, path: Path) -> tuple[Staging, Source]:
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = lander(path, staging, schema=f"land_{name}")
    return staging, staging_source(staging, landed, root / "ledger" / "landing")


def fresh_tools(source: Source, ledger_dir: Path, **kwargs: Any) -> ProfileTools:
    tools = make_tools(source.tools.provider, ledger_dir)
    if not kwargs:
        return tools
    return ProfileTools(tools.provider, tools.policy, tools.ledger, **kwargs)


def run(source: Source, tools: ProfileTools, hint: ConceptSourceSpec | None = None) -> Any:
    return structure().run_structure(source.land, tools, provider=source.provider, hint=hint)


def copy_code_tables(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in CODE_TABLES:
        shutil.copy(ACRIS / f"{name}.csv", target / f"{name}.csv")
    return target


def a2_hint(directory: Path, *, value_key: str | None = None) -> ConceptSourceSpec:
    """The A2 hand-written spec with its shape stated; ``value_key`` rebinds the join key."""
    raw = yaml.safe_load(A2_SPEC.read_text())
    section = raw["concept_source"]
    section["shape"] = [{"section": s, "template": t} for s, t in A2_SHAPE]
    if value_key is not None:
        roles = section["value_sets"]["roles"]
        roles.pop(value_key, None)
        roles["concept_id"] = value_key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "hint.spec.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_concept_source_spec(path)


def namespace(source: Source) -> set[str]:
    return {f"{r.schema}.{r.name}" for r in source.land.relations}


def bound_relations(result: Any) -> list[str]:
    return [name for entry in result.bound for name in entry.binding.relations]


def ignored_relations(result: Any) -> list[str]:
    return [name for ignored in result.ignored for name in ignored.relations]


def ambiguous_relations(result: Any) -> list[str]:
    return [ambiguity.subject for ambiguity in result.ambiguities]


def entries(result: Any, template: str) -> list[Any]:
    return [entry for entry in result.bound if entry.binding.template == template]


def entry_over(result: Any, relation: str) -> Any:
    (found,) = [entry for entry in result.bound if relation in entry.binding.relations]
    return found


def resolving(edges: Iterable[Any], floor: float) -> list[Any]:
    return [edge for edge in edges if edge.resolve_rate >= floor]


def joins(edge: Any, a: str, b: str, column: str | None = None) -> bool:
    ends = {edge.from_relation, edge.to_relation}
    named = column is None or edge.from_column == edge.to_column == column
    return ends == {a, b} and named


def reasons(result: Any) -> dict[str, str]:
    return {ignored.relations[0]: ignored.slots["reason"] for ignored in result.ignored}


def sql_lines(ledger_dir: Path) -> list[dict[str, Any]]:
    path = ledger_dir / "sql.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ref(source: Source, name: str) -> str:
    return f"{source.land.relation(name).schema}.{name}"


def sheet(source: Source, suffix: str) -> list[str]:
    return [ref(source, r.name) for r in source.land.relations if r.name.endswith(f"__{suffix}")]


def stems(source: Source) -> list[str]:
    return [name.rpartition("__")[0] for name in sheet(source, "Column_Info")]


def assert_partition(source: Source, result: Any) -> None:
    bound, ignored = bound_relations(result), ignored_relations(result)
    ambiguous = ambiguous_relations(result)
    for listed in (bound, ignored, ambiguous):
        assert len(listed) == len(set(listed)), Counter(listed).most_common(3)
    assert not set(bound) & set(ignored)
    assert not set(bound) & set(ambiguous)
    assert not set(ignored) & set(ambiguous)
    assert set(bound) | set(ignored) | set(ambiguous) == namespace(source)
