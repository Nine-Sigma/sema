"""Profile values may name environment variables as ``${NAME}``, so values stay out of git."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def profile_env(dotenv: Path = Path(".env")) -> dict[str, str]:
    """The KEY=VALUE lines of ``dotenv`` under the process environment; nothing is exported.

    Parsed here, not with python-dotenv: the routing guard refuses that module once loaded.
    """
    pairs: dict[str, str] = {}
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() and not key.lstrip().startswith("#"):
                pairs[key.strip()] = value.strip().strip("'\"")
    return pairs | dict(os.environ)


def expand_env(raw: Any, env: Mapping[str, str]) -> Any:
    """Replace every ``${NAME}`` in the string values of ``raw``; an unset name is an error."""
    if isinstance(raw, dict):
        return {key: expand_env(value, env) for key, value in raw.items()}
    if isinstance(raw, list):
        return [expand_env(value, env) for value in raw]
    if not isinstance(raw, str):
        return raw
    return re.sub(r"\$\{([A-Z][A-Z0-9_]*)\}", lambda match: _lookup(match.group(1), env), raw)


def _lookup(name: str, env: Mapping[str, str]) -> str:
    if name not in env:
        raise ValueError(f"${{{name}}} is set in neither the environment nor .env")
    return env[name]
