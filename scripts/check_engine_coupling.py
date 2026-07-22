"""R29 guard: no OMOP/OncoTree literal may leak into the engine core or the
generic mapping spine.

The 'any ontology' promise (R29) is enforced mechanically here, not in review.
A single policy constant (:data:`R29_POLICY`) owns the denylist, the scanned
core paths, and the allowlist of locations where domain literals legitimately
live (per-vocabulary policy modules, manifest adapters, authored manifests,
eval, tests). The same scan runs as a unit test (``tests/unit/
test_engine_coupling.py``) so ``uv run pytest`` fails on a leak, and as a CLI:

    uv run python scripts/check_engine_coupling.py

Configured core paths that do not exist yet (e.g. ``src/sema/resolve/engine.py``,
``src/sema/compile/``) are skipped so the guard is green today and tightens
automatically as later Slice-0 stories create them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence


@dataclass(frozen=True)
class EngineCouplingPolicy:
    """Frozen R29 policy: what is forbidden, where it is checked, what is exempt.

    ``core_paths`` and ``allowlist`` are repo-relative POSIX paths. A path is a
    file (scanned directly) or a directory (scanned recursively for ``*.py``).
    A scanned file is exempt when its repo-relative path is, or lives under, an
    allowlist entry.
    """

    denylist: tuple[str, ...]
    core_paths: tuple[str, ...]
    allowlist: tuple[str, ...]


# Case-explicit: a casing variant must not bypass the US-000 migration.
R29_POLICY = EngineCouplingPolicy(
    denylist=(
        "OncoTree",
        "oncotree",
        "ONCOTREE",
        "cBioPortal",
        "cbioportal",
        "cBio",
        "cbio",
        "Maps to",
        "standard_concept",
        "condition_occurrence",
        "concept_id",
    ),
    core_paths=(
        "src/sema/engine",
        "src/sema/resolve/engine.py",
        "src/sema/resolve/assembler.py",
        "src/sema/resolve/identity_registry.py",
        "src/sema/resolve/identity_registry_utils.py",
        "src/sema/resolve/identity_resolver.py",
        "src/sema/resolve/identity_collapse.py",
        "src/sema/resolve/identity_collapse_utils.py",
        "src/sema/compile",
        "src/sema/targets",
    ),
    allowlist=(
        "src/sema/resolve/policy.py",
        "src/sema/resolve/policies",
        "src/sema/targets/adapters",
        "src/sema/eval",
        "tests",
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class Violation(NamedTuple):
    path: str
    lineno: int
    literal: str


def _is_allowlisted(rel_path: str, allowlist: Sequence[str]) -> bool:
    return any(
        rel_path == entry or rel_path.startswith(f"{entry}/") for entry in allowlist
    )


def _iter_scanned_files(
    root: Path, policy: EngineCouplingPolicy
) -> Iterator[Path]:
    """Yield ``*.py`` files under the core paths, skipping allowlisted ones."""
    for core in policy.core_paths:
        target = root / core
        if not target.exists():
            continue
        candidates = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for candidate in candidates:
            if candidate.suffix != ".py":
                continue
            rel = candidate.relative_to(root).as_posix()
            if not _is_allowlisted(rel, policy.allowlist):
                yield candidate


def scan(root: Path, policy: EngineCouplingPolicy) -> list[Violation]:
    """Return every denylist hit across the policy's core paths under ``root``."""
    violations: list[Violation] = []
    for py_file in _iter_scanned_files(root, policy):
        rel = py_file.relative_to(root).as_posix()
        lines = py_file.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            for literal in policy.denylist:
                if literal in line:
                    violations.append(Violation(rel, lineno, literal))
    return violations


def find_violations(
    root: Path = REPO_ROOT, policy: EngineCouplingPolicy = R29_POLICY
) -> list[Violation]:
    """Scan the live repository tree with the default R29 policy."""
    return scan(root, policy)


def main(argv: Sequence[str] | None = None) -> int:
    violations = find_violations()
    if not violations:
        print("R29 engine-coupling guard: clean")
        return 0
    print("R29 engine-coupling guard: VIOLATIONS")
    for v in violations:
        print(f"  {v.path}:{v.lineno}: {v.literal!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
