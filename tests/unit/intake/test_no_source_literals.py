"""Generality gate: no fixture literal appears in the general intake code.

The literals are read from the fixture files at test time, so a fixture added
later is covered without editing this test.
"""

from __future__ import annotations

import ast
import csv
import inspect
import io
import json
import keyword
import re
import tokenize
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import openpyxl
import pytest
from pydantic import BaseModel

from sema.binding import spec_sections
from sema.intake.loaders.base import relation_name

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "intake"
SCAN_ROOT = REPO_ROOT / "src" / "sema" / "intake"
UNSCANNED_PACKAGES = ("loaders",)
MIN_LITERAL_LENGTH = 3
TABULAR_SUFFIXES = (".xlsx", ".csv")
SEE_PHRASE = re.compile(r"\bsee\s+(.+?)\s+dataset\b", re.IGNORECASE | re.DOTALL)
TOKEN = re.compile(r"\w+(?:-\w+)*")
KEY_VALUE_WIDTH = 2
TABLE_SCHEMA = FIXTURE_ROOT / "frictionless" / "table-schema.json"


class Hit(NamedTuple):
    path: str
    lineno: int
    literal: str


def _collapse(text: str) -> str:
    return " ".join(text.split()).casefold()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN.findall(text.casefold()))


def _texts(cells: tuple[object, ...]) -> list[str]:
    return [str(cell) for cell in cells if isinstance(cell, str) and cell.strip()]


def _see_phrases(text: str) -> Iterator[str]:
    for match in SEE_PHRASE.finditer(text):
        yield match.group(0)
        yield match.group(1)


def _filled(row: tuple[object, ...]) -> int:
    return sum(cell not in (None, "") for cell in row)


def _header_cells(rows: Sequence[tuple[object, ...]]) -> Iterator[str]:
    """Every text cell down to the first row holding two or more texts."""
    for row in rows:
        texts = _texts(row)
        yield from texts
        if len(texts) >= 2:
            return


def _all_see_phrases(rows: Sequence[tuple[object, ...]]) -> Iterator[str]:
    for row in rows:
        for text in _texts(row):
            yield from _see_phrases(text)


def _grid_literals(rows: Sequence[tuple[object, ...]]) -> Iterator[str]:
    yield from _header_cells(rows)
    yield from _all_see_phrases(rows)
    if all(_filled(row) <= KEY_VALUE_WIDTH for row in rows):
        yield from (str(row[0]) for row in rows if _texts(row[:1]))


def _xlsx_literals(path: Path) -> Iterator[str]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            yield sheet.title
            yield relation_name(f"{path.stem}__{sheet.title}")
            yield from _grid_literals(list(sheet.iter_rows(values_only=True)))
    finally:
        workbook.close()


def _csv_literals(path: Path) -> Iterator[str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(text[:4096])
    except csv.Error:
        dialect = csv.excel
    rows = [tuple(row) for row in csv.reader(io.StringIO(text), dialect)]
    yield relation_name(path.stem)
    yield from _header_cells(rows)
    yield from _all_see_phrases(rows)


def fixture_files(root: Path = FIXTURE_ROOT) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in TABULAR_SUFFIXES)


def literals_of(path: Path) -> set[str]:
    reader = _xlsx_literals if path.suffix.lower() == ".xlsx" else _csv_literals
    return {_collapse(text) for text in reader(path)}


def _table_schema_vocabulary(node: object) -> Iterator[str]:
    """Property names and enumerated values of the vendored Table Schema."""
    if isinstance(node, list):
        for item in node:
            yield from _table_schema_vocabulary(item)
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            yield from value
        if key == "enum" and isinstance(value, list):
            yield from (item for item in value if isinstance(item, str))
        yield from _table_schema_vocabulary(value)


def excluded_names() -> set[str]:
    names = set(keyword.kwlist) | set(keyword.softkwlist)
    for _, model in inspect.getmembers(spec_sections, inspect.isclass):
        if issubclass(model, BaseModel) and model.__name__.endswith("Roles"):
            for field_name, field in model.model_fields.items():
                names.add(field_name)
                names.update(alias for alias in (field.alias,) if alias)
    schema = json.loads(TABLE_SCHEMA.read_text(encoding="utf-8"))
    names.update(_table_schema_vocabulary(schema))
    return {name.casefold() for name in names}


def fixture_literals(root: Path = FIXTURE_ROOT) -> dict[tuple[str, ...], str]:
    excluded = excluded_names()
    found: dict[tuple[str, ...], str] = {}
    for path in fixture_files(root):
        for literal in literals_of(path):
            tokens = _tokens(literal)
            if len(literal) >= MIN_LITERAL_LENGTH and tokens and literal not in excluded:
                found.setdefault(tokens, literal)
    return found


def _docstring_nodes(tree: ast.Module) -> set[int]:
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, owners)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }


def source_texts(source: str) -> Iterator[tuple[int, str]]:
    """String constants other than docstrings, then comments, with line numbers."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.lineno, node.value
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string.lstrip("#")


def _matches(text: str, literals: Mapping[tuple[str, ...], str]) -> Iterator[str]:
    tokens = _tokens(_collapse(text))
    for width in {len(key) for key in literals if len(key) <= len(tokens)}:
        for start in range(len(tokens) - width + 1):
            literal = literals.get(tokens[start : start + width])
            if literal is not None:
                yield literal


def scanned_modules(root: Path = SCAN_ROOT) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.relative_to(root).parts[0] not in UNSCANNED_PACKAGES
    )


def find_hits(root: Path, literals: Mapping[tuple[str, ...], str]) -> list[Hit]:
    hits: list[Hit] = []
    for path in scanned_modules(root):
        rel = path.relative_to(root).as_posix()
        for lineno, text in source_texts(path.read_text(encoding="utf-8")):
            hits.extend(Hit(rel, lineno, literal) for literal in _matches(text, literals))
    return sorted(set(hits))


def test_every_fixture_file_contributes_literals() -> None:
    files = fixture_files()
    assert files, f"no tabular fixture under {FIXTURE_ROOT}"
    for path in files:
        assert literals_of(path), f"{path} yielded no literal"


def test_role_names_and_keywords_are_excluded() -> None:
    excluded = excluded_names()
    assert "from" in excluded
    assert set(spec_sections.ValueSetRoles.model_fields) <= excluded


def test_table_schema_vocabulary_is_read_from_the_vendored_schema() -> None:
    schema = {
        "properties": {"fooKey": {"enum": ["barValue", 3]}},
        "items": [{"enum": ["baz"]}],
    }
    assert set(_table_schema_vocabulary(schema)) == {"fooKey", "barValue", "baz"}
    assert {"foreignkeys", "primarykey", "date", "string"} <= excluded_names()


def test_key_value_sheet_yields_its_keys_and_wide_sheet_its_header() -> None:
    narrow = [("Title", None), ("Key One", "value"), ("Key Two", None), (None, "more")]
    wide = [("Name", "Kind", "Note"), ("row one", "x", "see Other Thing dataset")]
    assert set(_grid_literals(narrow)) == {"Title", "Key One", "value", "Key Two"}
    assert set(_grid_literals(wide)) == {
        "Name", "Kind", "Note", "see Other Thing dataset", "Other Thing"
    }


def test_planted_fixture_literal_is_flagged(tmp_path: Path) -> None:
    literals = fixture_literals()
    planted = max(
        (text for text in literals.values() if re.fullmatch(r"[a-z0-9_ ]+", text)), key=len
    )
    module = tmp_path / "planted.py"
    module.write_text(
        f'"""{planted}"""\n\nRELATION = {planted!r}\n# reads {planted.upper()}\n',
        encoding="utf-8",
    )
    hits = find_hits(tmp_path, literals)
    assert {(hit.lineno, hit.literal) for hit in hits} >= {(3, planted), (4, planted)}
    assert all(hit.lineno != 1 for hit in hits)


def test_loader_package_is_not_scanned(tmp_path: Path) -> None:
    literals = fixture_literals()
    planted = next(iter(literals.values()))
    (tmp_path / "loaders").mkdir()
    (tmp_path / "loaders" / "reader.py").write_text(f"NAME = {planted!r}\n", "utf-8")
    assert find_hits(tmp_path, literals) == []


def test_partial_token_overlap_is_not_a_hit(tmp_path: Path) -> None:
    literals: dict[tuple[str, ...], str] = {("alpha", "beta"): "alpha beta"}
    (tmp_path / "near.py").write_text(
        "A = 'alphabet beta'\nB = 'alpha'\nC = 'x-alpha beta'\n", "utf-8"
    )
    (tmp_path / "exact.py").write_text("C = 'ALPHA\\n   Beta gamma'\n", "utf-8")
    assert find_hits(tmp_path, literals) == [Hit("exact.py", 1, "alpha beta")]


def test_general_intake_code_names_no_fixture_literal() -> None:
    hits = find_hits(SCAN_ROOT, fixture_literals())
    assert hits == [], "fixture literals in general intake code:\n" + "\n".join(
        f"  {hit.path}:{hit.lineno}: {hit.literal!r}" for hit in hits
    )
