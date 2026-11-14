"""The loader's code and tests hold no profile value: every source value lives in a profile.

The literals are read from the profile files at test time, so a profile added later is
covered without editing this test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "showcase" / "open_data_load"
PROFILES = PACKAGE / "profiles"
TESTS = Path(__file__).parent
MIN_LITERAL_LENGTH = 3


def _values(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _values(item)
    elif node is not None:
        yield str(node)


def profile_literals(profiles: Path = PROFILES) -> set[str]:
    found: set[str] = set()
    for path in sorted(profiles.glob("*.yaml")):
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        found |= set(profile.get("relations", {})) | set(_values(profile))
    return {literal for literal in found if len(literal.strip()) >= MIN_LITERAL_LENGTH}


def scanned_files(package: Path = PACKAGE) -> list[Path]:
    tests = [path for path in TESTS.glob("*.py") if path != Path(__file__)]
    return sorted([*package.glob("*.py"), *tests])


def literal_hits(paths: list[Path], literals: set[str]) -> list[str]:
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8").casefold()
        hits += [f"{path.name}: {literal}" for literal in sorted(literals)
                 if literal.casefold() in text]  # fmt: skip
    return hits


def test_the_scan_covers_the_package_and_its_tests() -> None:
    names = {path.name for path in scanned_files()}
    assert {"load.py", "test_load.py", "conftest.py"} <= names


def test_profile_literals_include_relation_names_and_leaf_values(tmp_path: Path) -> None:
    (tmp_path / "p.yaml").write_text(
        "download_url: https://host.example/{dataset}\n"
        "relations:\n  widgets:\n    dataset: abcd-1234\n    headers: {size: WIDGET SIZE}\n"
    )
    assert {"https://host.example/{dataset}", "widgets", "abcd-1234", "WIDGET SIZE"} <= (
        profile_literals(tmp_path)
    )


def test_the_guard_reports_a_planted_profile_value(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text('URL = "generic"\nDATASET = "ABCD-1234"\n')
    assert literal_hits([planted], {"abcd-1234", "zzzz"}) == ["planted.py: abcd-1234"]


def test_a_profile_exists_to_guard() -> None:
    assert profile_literals()


def test_loader_code_and_tests_hold_no_profile_value() -> None:
    hits = literal_hits(scanned_files(), profile_literals())
    assert not hits, f"{len(hits)} source-specific literals:\n" + "\n".join(hits)
