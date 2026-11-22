"""Both sides of the source-literal guards: the literals a source file holds, and the text
of product code that is scanned for them.

Literals come from tabular fixtures (header cells, "see ... dataset" phrases, key-value
sheet keys) and from structured profiles (relation names and leaf values). Python modules
yield string constants (docstrings excluded) and comments; prompt files under ``prompts/``
yield every line as prose.
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
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import openpyxl
from pydantic import BaseModel

from sema.binding import spec_sections
from sema.intake.roles import lexicon
from sema.intake.land.loaders.base import relation_name

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = REPO_ROOT / "src" / "sema" / "intake"
UNSCANNED_PACKAGES = ("loaders",)
PROMPTS_DIR = "prompts"
TOKEN = re.compile(r"\w+(?:-\w+)*")


class Hit(NamedTuple):
    path: str
    lineno: int
    literal: str


class SourceText(NamedTuple):
    lineno: int
    text: str
    is_constant: bool


def collapse(text: str) -> str:
    return " ".join(text.split()).casefold()


def tokens_of(text: str) -> tuple[str, ...]:
    return tuple(TOKEN.findall(text.casefold()))


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


def source_texts(source: str) -> Iterator[SourceText]:
    """String constants other than docstrings, then comments, with line numbers."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield SourceText(node.lineno, node.value, True)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield SourceText(token.start[0], token.string.lstrip("#"), False)


def prompt_texts(text: str) -> Iterator[SourceText]:
    """Every line of a prompt file, read as prose: only multi-token literals hit."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        yield SourceText(lineno, line, False)


def matches(text: SourceText, literals: Mapping[tuple[str, ...], str]) -> Iterator[str]:
    """A multi-token literal hits anywhere; a single token only as a whole constant.

    A lone common word ("values", "source") names a source only when code uses it as
    an entire name; inside SQL or prose it is ordinary language.
    """
    tokens = tokens_of(collapse(text.text))
    if text.is_constant and len(tokens) == 1 and (whole := literals.get(tokens)):
        yield whole
    widths = {len(key) for key in literals if 1 < len(key) <= len(tokens)}
    for width in widths:
        for start in range(len(tokens) - width + 1):
            literal = literals.get(tokens[start : start + width])
            if literal is not None:
                yield literal


def _scanned(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(pattern)
        if not set(path.relative_to(root).parts[:-1]) & set(UNSCANNED_PACKAGES)
    )


def scanned_modules(root: Path = SCAN_ROOT) -> list[Path]:
    return _scanned(root, "*.py")


def scanned_prompts(root: Path = SCAN_ROOT) -> list[Path]:
    return _scanned(root, f"{PROMPTS_DIR}/*.md")


def find_hits(root: Path, literals: Mapping[tuple[str, ...], str]) -> list[Hit]:
    readers: list[tuple[Path, Callable[[str], Iterator[SourceText]]]] = [
        (path, source_texts) for path in scanned_modules(root)
    ]
    readers += [(path, prompt_texts) for path in scanned_prompts(root)]
    hits: list[Hit] = []
    for path, read in readers:
        rel = path.relative_to(root).as_posix()
        for text in read(path.read_text(encoding="utf-8")):
            hits.extend(Hit(rel, text.lineno, literal) for literal in matches(text, literals))
    return sorted(set(hits))


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "intake"
MIN_LITERAL_LENGTH = 3
TABULAR_SUFFIXES = (".xlsx", ".csv")
SEE_PHRASE = re.compile(r"\bsee\s+(.+?)\s+dataset\b", re.IGNORECASE | re.DOTALL)
KEY_VALUE_WIDTH = 2
TABLE_SCHEMA = FIXTURE_ROOT / "frictionless" / "table-schema.json"


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
    return {collapse(text) for text in reader(path)}


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
    names.update(word for group in _lexicon_token_groups() for word in group)
    return {name.casefold() for name in names}


def literal_index(literals: Iterable[str], excluded: Collection[str]) -> dict[tuple[str, ...], str]:
    """Each kept literal under its token tuple, the form ``matches`` looks up."""
    found: dict[tuple[str, ...], str] = {}
    for literal in map(collapse, literals):
        tokens = tokens_of(literal)
        if len(literal) >= MIN_LITERAL_LENGTH and tokens and literal not in excluded:
            found.setdefault(tokens, literal)
    return found


def fixture_literals(root: Path = FIXTURE_ROOT) -> dict[tuple[str, ...], str]:
    paths = fixture_files(root)
    return literal_index((text for path in paths for text in literals_of(path)), excluded_names())


def _lexicon_token_groups() -> list[Collection[str]]:
    groups: list[Collection[str]] = [entry.tokens for entry in lexicon.ROLES]
    return [*groups, *lexicon.HEADER_KEY_TOKENS.values(), *lexicon.STATUS_VALUE_WORDS.values()]


def leaf_values(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from leaf_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from leaf_values(item)
    elif node is not None:
        yield str(node)


def structured_literals(document: object) -> set[str]:
    """A profile's relation names (mapping keys) and every leaf value."""
    relations = document.get("relations", {}) if isinstance(document, dict) else {}
    return set(relations) | set(leaf_values(document))
