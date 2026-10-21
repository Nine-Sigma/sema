"""Shared setup for the role eligibility tests: sources, fake embedders, readers.

``sema.intake.eligibility`` and ``eligibility_utils`` are imported lazily, so a missing
module fails each test on its own instead of failing collection.

The contract the tests pin (design Decisions 5, 6, 7, 20):

- ``eligibility.compute(binding, profiles, *, memory=None, embedder=None,
  margin_floor=None)`` returns one ``RoleEligibility`` per role of the binding's template.
- ``RoleEligibility``: ``relation``, ``role``, ``subject`` (the property of a per-property
  role such as ``domain``, else ``None``), ``members`` (the predicate set, before any
  narrowing), ``hints`` (``Hint`` with ``source``, ``matches``, ``rank`` as ``(column,
  score)`` pairs best first, ``margin``), ``excluded_by``, ``choice``, ``confidence``,
  ``source``.
- ``eligibility.ambiguities(eligibilities)`` is the stage ``roles`` ambiguity list. A
  subject starts with the relation name and names the property of a per-property role;
  a reference phrase naming nothing landed carries the hint ``unresolved_reference``.
- ``eligibility.run_roles(structure, profiles, tools, *, hint=None, embedder=None,
  margin_floor=None)`` returns ``eligibilities``, ``ambiguities``, ``joins``
  (``DeclaredJoin``), ``headers`` (by header relation: ``name``, ``description``,
  ``values`` keyed by the landed key text, ``prose`` keys) and ``structure`` (the input
  with every section Roles resolved set on its bindings).
- ``eligibility.assemble_spec(descriptor, roles)`` returns a ``ConceptSourceSpec`` or
  ``None``; ``eligibility_utils.RunDescriptor`` validates a ``*.run.yaml`` mapping.
"""

from __future__ import annotations

import importlib
import json
import math
import shutil
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import duckdb
import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.tabular import land_tabular
from sema.intake.shapes.base import SlotBinding
from tests.unit.intake.shapes.shapes_env import FIXTURES, Source, warehouse_source
from tests.unit.intake.structure_env import (
    A2_WORKBOOK,
    ACRIS,
    ACRIS_SPEC,
    fresh_tools,
    land,
    run,
)

A8_CSV = FIXTURES / "tabular" / "ambiguity_a8.csv"
A8_BUILDER = FIXTURES / "tabular" / "build_ambiguity_a8.py"
A2_RUN = FIXTURES / "tabular" / "codes_a2.run.yaml"
A2_GOLDEN = FIXTURES / "tabular" / "codes_a2.manifest.yaml"
ACRIS_RUN = ACRIS / "acris.run.yaml"
GROUPED_CODE_TABLE = "Document_Control_Codes"
MISSPELLED_PROPERTY = "Docoument Type"
REFERENCES = {
    "Property Type": 2,
    MISSPELLED_PROPERTY: 2,
    "UCC Collateral": 1,
    "Country": 2,
    "State": 2,
}
REAL_MASTER = "ACRIS - Real Property Master"
PERSONAL_MASTER = "ACRIS - Personal Property Master"
JOIN_COLUMNS = ["Document ID", "Good Through Date"]
DECLARED_JOINS = {
    (f"ACRIS - {kind} Property {subject}", master)
    for kind, master in (("Real", REAL_MASTER), ("Personal", PERSONAL_MASTER))
    for subject in ("Legals", "Parties", "Remarks", "References")
}
LEGALS_DESCRIPTION = (
    "The properties under which real property documents recorded in ACRIS are indexed."
)
PROMPT_KEY_START = "List any additional information"
CONCEPT_LIST = "concept_list"
SPEC_TOKENS = {
    "concept_id": {"id", "concept_id", "identifier"},
    "code": {"code", "value"},
    "name": {"name", "label", "title"},
    "display": {"display", "label", "name"},
    "definition": {"definition", "description", "desc", "meaning"},
    "status": {"status", "active", "deprecated", "retired", "invalid"},
    "parent": {"parent", "parent_id", "broader"},
}
SECTION_MEMBERS = {"concepts", "code_lists"}
LONG_TAIL = "' written out at length for the embedding rank case'"
# Text columns longer than any definition floor, repeating so neither is unique.
RANKED_SQL = (
    "CREATE SCHEMA tw",
    "CREATE TABLE tw.ranked AS SELECT i AS id, 'Lbl ' || CAST(i AS VARCHAR) AS label,"
    f" 'Zorbit text ' || CAST(i % 15 AS VARCHAR) || {LONG_TAIL} AS zorbit_text,"
    f" 'Quaxle body ' || CAST(i % 15 AS VARCHAR) || {LONG_TAIL} AS quaxle_body"
    " FROM range(1, 21) r(i)",
    "CREATE TABLE tw.flagged AS SELECT i AS id, 'Name ' || CAST(i AS VARCHAR) AS label,"
    " 'Y' AS flag FROM range(1, 21) r(i)",
)
COLOURS_SQL = (
    "CREATE SCHEMA tc",
    "CREATE TABLE tc.colours AS SELECT * FROM (VALUES ('RED', 'Red'), ('GREEN', 'Green'),"
    " ('BLUE', 'Blue'), ('AMBER', 'Amber'), ('WHITE', 'White')) t(code, label)",
)


def eligibility() -> ModuleType:
    return importlib.import_module("sema.intake.eligibility")


def eligibility_utils() -> ModuleType:
    return importlib.import_module("sema.intake.eligibility_utils")


def lexicon() -> ModuleType:
    return importlib.import_module("sema.intake.lexicon")


class KeywordEmbedder:
    """Columns named by a keyword get that keyword's vector; every other text, role
    descriptions included, gets ``default``."""

    def __init__(self, vectors: dict[str, list[float]], default: list[float]) -> None:
        self.vectors = vectors
        self.default = default
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        found = [v for word, v in self.vectors.items() if word in text.casefold()]
        return found[0] if found else self.default


def clear_embedder() -> KeywordEmbedder:
    return KeywordEmbedder({"zorbit": [1.0, 0.0], "quaxle": [0.0, 1.0]}, [1.0, 0.0])


def close_embedder() -> KeywordEmbedder:
    return KeywordEmbedder({"zorbit": [1.0, 0.9], "quaxle": [0.9, 1.0]}, [1.0, 0.0])


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.hypot(*a) * math.hypot(*b))


def memory_source(
    statements: Sequence[str], schema: str, tmp: Path
) -> tuple[duckdb.DuckDBPyConnection, Source]:
    con = duckdb.connect(":memory:")
    for statement in statements:
        con.execute(statement)
    return con, warehouse_source(con, schema, tmp / "ledger" / "landing")


def concept_list_over(ref: str) -> SlotBinding:
    return SlotBinding(template=CONCEPT_LIST, slots={"relation": ref}, relations=(ref,))


def by_role(eligibilities: Iterable[Any], relation: str | None = None) -> dict[str, Any]:
    return {
        e.role: e
        for e in eligibilities
        if e.subject is None and (relation is None or e.relation == relation)
    }


def per_property(eligibilities: Iterable[Any], role: str) -> list[Any]:
    return [e for e in eligibilities if e.role == role and e.subject is not None]


def hint_sources(eligibility_entry: Any) -> list[str]:
    return [hint.source for hint in eligibility_entry.hints]


def copy_without(target: Path, name: str) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for path in ACRIS.iterdir():
        if path.suffix in {".xlsx", ".csv"} and path.stem != name:
            shutil.copy(path, target / path.name)
    return target


def write_hint(directory: Path, raw: dict[str, Any]) -> ConceptSourceSpec:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "hint.spec.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_concept_source_spec(path)


def grouped_table_hint(directory: Path) -> ConceptSourceSpec:
    """Pins only the grouped code table as a code list, so the others stay code-found."""
    raw = yaml.safe_load(ACRIS_SPEC.read_text())
    section = raw["concept_source"]
    keep = {"spec_version", "id", "identity_system", "release", "default_provider"}
    trimmed = {key: value for key, value in section.items() if key in keep}
    trimmed["shape"] = [
        {"section": f"code_lists.{GROUPED_CODE_TABLE}", "template": CONCEPT_LIST}
    ]
    trimmed["code_lists"] = [
        entry for entry in section["code_lists"] if entry["vocabulary_id"] == GROUPED_CODE_TABLE
    ]
    return write_hint(directory, {"concept_source": trimmed})


def derived_free_hint(directory: Path) -> ConceptSourceSpec:
    """The hand-written spec without what the rules derive: joins, domains, notes."""
    raw = yaml.safe_load(ACRIS_SPEC.read_text())
    model = raw["concept_source"]["entity_model"]
    model["roles"].pop("notes")
    for entry in model["entities"]:
        entry.pop("foreign_keys", None)
        entry.pop("domains", None)
    return write_hint(directory, raw)


def normalized(spec: ConceptSourceSpec) -> Any:
    """YAML round trip with every list of mappings sorted; lists of strings keep order."""
    dumped = yaml.safe_load(yaml.safe_dump(spec.model_dump(mode="json", by_alias=True)))
    return _sorted_mappings(dumped)


def _sorted_mappings(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _sorted_mappings(value) for key, value in node.items()}
    if not isinstance(node, list):
        return node
    items = [_sorted_mappings(item) for item in node]
    if items and all(isinstance(item, dict) for item in items):
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    return items


def model_calls(ledger_dir: Path) -> list[Path]:
    calls = ledger_dir / "calls"
    return sorted(calls.iterdir()) if calls.is_dir() else []


def run_roles(source: Source, structure: Any, ledger_dir: Path, **kwargs: Any) -> Any:
    tools = fresh_tools(source, ledger_dir, stage="roles", structure=structure)
    return eligibility().run_roles(structure, source.profiles(), tools, **kwargs)


def descriptor(path: Path) -> Any:
    return eligibility_utils().RunDescriptor.model_validate(yaml.safe_load(path.read_text()))


class Landed(NamedTuple):
    staging: Staging
    source: Source


class Env:
    """Module-wide sources and stage results, each computed on first use."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._closers: list[Callable[[], None]] = []

    def close(self) -> None:
        for closer in reversed(self._closers):
            closer()

    def _land(self, name: str, path: Path) -> Landed:
        staging, source = land(self.root / name, name, land_tabular, path)
        self._closers.append(staging.close)
        return Landed(staging, source)

    def structured(self, landed: Landed, name: str, hint: Any = None) -> Any:
        ledger = self.root / name / "structure"
        return run(landed.source, fresh_tools(landed.source, ledger), hint)

    @cached_property
    def a8(self) -> Landed:
        return self._land("a8", A8_CSV)

    @cached_property
    def a2(self) -> Landed:
        return self._land("a2", A2_WORKBOOK)

    @cached_property
    def acris(self) -> Landed:
        return self._land("acris", ACRIS)

    @cached_property
    def dropped(self) -> Landed:
        files = copy_without(self.root / "nodcc_files", GROUPED_CODE_TABLE)
        return self._land("nodcc", files)

    @cached_property
    def ranked(self) -> Source:
        con, source = memory_source(RANKED_SQL, "tw", self.root / "ranked")
        self._closers.append(con.close)
        return source

    @cached_property
    def a10(self) -> Any:
        return self.structured(self.acris, "a10")

    @cached_property
    def a10_roles(self) -> Any:
        return run_roles(self.acris.source, self.a10, self.root / "a10" / "roles")

    @cached_property
    def pinned_roles(self) -> Any:
        hint = grouped_table_hint(self.root / "pinned")
        structured = self.structured(self.acris, "pinned", hint)
        return run_roles(self.acris.source, structured, self.root / "pinned" / "roles")

    @cached_property
    def hinted_roles(self) -> Any:
        hint = derived_free_hint(self.root / "hinted")
        structured = self.structured(self.acris, "hinted", hint)
        ledger = self.root / "hinted" / "roles"
        return run_roles(self.acris.source, structured, ledger, hint=hint)

    @cached_property
    def dropped_roles(self) -> Any:
        structured = self.structured(self.dropped, "nodcc")
        return run_roles(self.dropped.source, structured, self.root / "nodcc" / "roles")
