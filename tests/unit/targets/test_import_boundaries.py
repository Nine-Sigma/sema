"""Module import boundary enforcement for `sema.targets`.

Adapters, normalizer, hashing, and registry MUST NOT import from
`sema.engine`, `sema.pipeline`, or `sema.graph`. The materializer MAY
import from `sema.graph` but MUST NOT import from `sema.engine` or
`sema.pipeline`.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_FOR_NON_MATERIALIZER = ("sema.engine", "sema.pipeline", "sema.graph")
_FORBIDDEN_FOR_MATERIALIZER = ("sema.engine", "sema.pipeline")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TARGETS_ROOT = _REPO_ROOT / "src" / "sema" / "targets"
_MATERIALIZER_SUFFIX = "materializer.py"


def _iter_target_modules() -> list[Path]:
    return sorted(p for p in _TARGETS_ROOT.rglob("*.py") if p.name != "__init__.py")


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.add(node.module)
    return out


def _is_materializer(path: Path) -> bool:
    return path.name == _MATERIALIZER_SUFFIX


def _forbidden_for(path: Path) -> tuple[str, ...]:
    return _FORBIDDEN_FOR_MATERIALIZER if _is_materializer(path) else _FORBIDDEN_FOR_NON_MATERIALIZER


def _violation(import_name: str, forbidden: tuple[str, ...]) -> str | None:
    for prefix in forbidden:
        if import_name == prefix or import_name.startswith(prefix + "."):
            return prefix
    return None


def test_direct_imports_respect_boundaries() -> None:
    violations: list[str] = []
    for path in _iter_target_modules():
        forbidden = _forbidden_for(path)
        for imp in _module_imports(path):
            offender = _violation(imp, forbidden)
            if offender is not None:
                violations.append(f"{path.relative_to(_REPO_ROOT)} imports {imp} (forbidden prefix: {offender})")
    assert not violations, "import-boundary violations:\n" + "\n".join(violations)


def _resolve_module_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None or spec.origin == "built-in":
        return None
    return Path(spec.origin)


def _is_inside_repo(path: Path) -> bool:
    try:
        path.relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return True


def _walk_transitive(start: Path, max_depth: int = 5) -> set[str]:
    seen_paths: set[Path] = set()
    seen_modules: set[str] = set()

    def _walk(path: Path, depth: int) -> None:
        if depth > max_depth or path in seen_paths or not path.exists():
            return
        seen_paths.add(path)
        try:
            imports = _module_imports(path)
        except SyntaxError:
            return
        for imp in imports:
            seen_modules.add(imp)
            if not imp.startswith("sema."):
                continue
            resolved = _resolve_module_path(imp)
            if resolved is None or not _is_inside_repo(resolved):
                continue
            _walk(resolved, depth + 1)

    _walk(start, 0)
    return seen_modules


def test_transitive_imports_respect_boundaries() -> None:
    violations: list[str] = []
    for path in _iter_target_modules():
        forbidden = _forbidden_for(path)
        for imp in _walk_transitive(path):
            offender = _violation(imp, forbidden)
            if offender is not None:
                violations.append(
                    f"{path.relative_to(_REPO_ROOT)} transitively reaches {imp} "
                    f"(forbidden prefix: {offender})"
                )
    assert not violations, "transitive import-boundary violations:\n" + "\n".join(violations)


def test_fixture_adapter_importing_engine_is_detected(tmp_path: Path) -> None:
    bad = tmp_path / "fake_adapter.py"
    bad.write_text("from sema.engine.semantic import infer_type\n")
    imports = _module_imports(bad)
    assert any(imp.startswith("sema.engine") for imp in imports)
    assert _violation("sema.engine.semantic", _FORBIDDEN_FOR_NON_MATERIALIZER) == "sema.engine"


def test_fixture_normalizer_importing_graph_is_detected(tmp_path: Path) -> None:
    bad = tmp_path / "normalizer.py"
    bad.write_text("from sema.graph.loader import GraphLoader\n")
    assert _violation("sema.graph.loader", _FORBIDDEN_FOR_NON_MATERIALIZER) == "sema.graph"


def test_fixture_materializer_importing_graph_is_allowed(tmp_path: Path) -> None:
    materializer = tmp_path / "materializer.py"
    materializer.write_text("from sema.graph.loader import GraphLoader\n")
    assert _is_materializer(materializer)
    assert _violation("sema.graph.loader", _FORBIDDEN_FOR_MATERIALIZER) is None


def test_fixture_materializer_importing_engine_is_rejected(tmp_path: Path) -> None:
    materializer = tmp_path / "materializer.py"
    materializer.write_text("from sema.engine.semantic import infer_type\n")
    assert _is_materializer(materializer)
    assert _violation("sema.engine.semantic", _FORBIDDEN_FOR_MATERIALIZER) == "sema.engine"


def test_modules_inspected_so_no_silent_skip() -> None:
    modules = _iter_target_modules()
    paths = [p.name for p in modules]
    expected = {"base.py", "registry.py", "registry_utils.py", "exceptions.py"}
    missing = expected - set(paths)
    assert not missing, f"target package missing expected modules: {missing}"


def test_walking_an_unknown_module_is_safe(tmp_path: Path) -> None:
    f = tmp_path / "nonexistent.py"
    f.write_text("import does_not_exist\n")
    seen = _walk_transitive(f)
    assert "does_not_exist" in seen
    assert importlib.util.find_spec("does_not_exist") is None
    assert "sys" in sys.modules
