"""The scan side of the source-literal guard: what text in intake code and prompts is read.

Python modules yield string constants (docstrings excluded) and comments; prompt files
under ``prompts/`` yield every line as prose.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[3]
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
