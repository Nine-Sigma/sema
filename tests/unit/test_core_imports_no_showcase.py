"""S2-01b / S2-04 structural guard: agnosticism beyond the R29 literal grep.

Two assertions R29's string denylist cannot make:
  1. No core `src/sema/` module imports a ``showcase.`` symbol at *module
     scope*. Lazy, function-scoped imports (the intended plugin-wiring seam in
     `cli` / `resolve.policies`) are allowed; a top-level import would ossify a
     showcase dependency into the always-loaded core.
  2. The new generic graph ports/loaders name no target-domain literal — so
     OMOP semantics cannot leak in as method names or assumptions.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[2] / "src" / "sema"

_GENERIC_GRAPH_MODULES = (
    "graph/concept_source.py",
    "graph/concept_value_set.py",
    "graph/cross_layer_bridge.py",
    "graph/physical_catalog.py",
    "graph/target_graph_build.py",
    "graph/target_join_paths.py",
    "graph/target_lifecycle.py",
    "graph/target_retrieval_loader.py",
    "graph/target_retrieval_loader_utils.py",
)

_DOMAIN_LITERALS = (
    "OncoTree", "oncotree", "ONCOTREE", "cBioPortal", "cbioportal", "cBio",
    "cbio", "Maps to", "standard_concept", "condition_occurrence",
    "concept_id", "OMOP", "SNOMED",
)


def _module_level_import_names(tree: ast.Module) -> list[str]:
    """Names imported at module scope (not inside a def/class)."""
    names: list[str] = []

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            for attr in ("body", "orelse", "finalbody"):
                walk(getattr(node, attr, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                walk(handler.body)

    walk(tree.body)
    return names


def test_no_core_module_imports_showcase_at_module_scope() -> None:
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _module_level_import_names(tree):
            if name == "showcase" or name.startswith("showcase."):
                offenders.append(path.relative_to(_SRC).as_posix())
    assert not offenders, f"module-scope showcase imports: {offenders}"


def test_generic_graph_modules_name_no_domain_literal() -> None:
    offenders: list[str] = []
    for rel in _GENERIC_GRAPH_MODULES:
        text = (_SRC / rel).read_text(encoding="utf-8")
        for literal in _DOMAIN_LITERALS:
            if literal in text:
                offenders.append(f"{rel}: {literal!r}")
    assert not offenders, f"domain literals leaked into core: {offenders}"
