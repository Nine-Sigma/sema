"""Make the repo-root ``showcase/`` package importable from an installed CLI.

``showcase/`` lives at the repository root, which the editable install does not
place on ``sys.path`` (only ``src/`` is installed). It is therefore importable
solely by accident of the current working directory. This helper inserts the
repo root explicitly when — and only when — a source checkout is present, so the
showcase commands register regardless of where the console script runs.

Contract: source checkout -> showcase importable; wheel / ``uv tool install`` /
pipx -> the repo root has no ``showcase/`` package, so this is a no-op and the
showcase commands are intentionally absent.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_showcase_importable() -> None:
    root = Path(__file__).resolve().parents[2]
    if not (root / "showcase" / "__init__.py").exists():
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
