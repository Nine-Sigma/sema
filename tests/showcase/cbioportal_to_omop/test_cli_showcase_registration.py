"""F1 regression: showcase commands survive the console-entry-point boundary.

The bug: ``_register_showcase_commands`` imports ``showcase.*`` inside a bare
``except ImportError: return``. ``showcase/`` lives at the repo root, which the
editable install does NOT put on ``sys.path`` — only the current working
directory does. So running the installed ``sema`` console script from anywhere
but the repo root silently dropped ``fit`` / ``fit-omop-shape`` /
``collapse-omop-identities``.

This drives the real console script in a subprocess from a scratch directory
with a clean environment (no repo root on the path, no ``PYTHONPATH``), which is
the faithful reproduction of how an installed CLI is invoked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SHOWCASE_COMMANDS = ("fit", "fit-omop-shape", "collapse-omop-identities")


def _sema_console_script() -> Path:
    return Path(sys.executable).with_name("sema")


def test_console_script_registers_showcase_commands_outside_repo(tmp_path) -> None:
    script = _sema_console_script()
    if not script.exists():
        pytest.skip("sema console script not installed in this environment")
    result = subprocess.run(
        [str(script), "--help"],
        cwd=str(tmp_path),
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in _SHOWCASE_COMMANDS:
        assert name in result.stdout, (
            f"showcase command {name!r} missing from console-script --help; "
            f"got:\n{result.stdout}"
        )
