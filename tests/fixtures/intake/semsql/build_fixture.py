"""Write ``ontology.db``, the A3 semsql fixture, from ``ontology.ttl``.

Run: ``uv run python tests/fixtures/intake/semsql/build_fixture.py``.

The base tables follow the semsql DDL (``statements``, ``entailed_edge``,
``prefix``); the views come verbatim from ``semsql_views.sql`` (upstream commit
recorded there). The Turtle reader handles only this fixture's constructs:
``@prefix`` lines and one triple per line. ``entailed_edge`` holds the
transitive closure of named ``rdfs:subClassOf`` edges, standing in for
relation-graph. Writes go through the standard library ``sqlite3`` module, the
same module the loader reads with.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
TTL = HERE / "ontology.ttl"
VIEWS = HERE / "semsql_views.sql"
OUT = HERE / "ontology.db"

BASE_DDL = """
CREATE TABLE prefix (prefix TEXT, base TEXT);
CREATE TABLE statements (
    stanza TEXT, subject TEXT, predicate TEXT, object TEXT,
    value TEXT, datatype TEXT, language TEXT
);
CREATE TABLE entailed_edge (subject TEXT, predicate TEXT, object TEXT);
"""
PREFIX_LINE = re.compile(r"^@prefix (\w+): <([^>]+)> \.$")
TRIPLE_LINE = re.compile(r"^(\S+) (\S+) (.+) \.$")
LITERAL = re.compile(r'^"(.*)"(?:@(\w+)|\^\^(\S+))?$')
SUBCLASS = "rdfs:subClassOf"

Row = tuple[str, str, str, str | None, str | None, str | None, str | None]


def read_turtle(text: str) -> tuple[list[tuple[str, str]], list[Row]]:
    prefixes: list[tuple[str, str]] = []
    statements: list[Row] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if match := PREFIX_LINE.match(line):
            prefixes.append((match.group(1), match.group(2)))
            continue
        match = TRIPLE_LINE.match(line)
        if match is None:
            raise ValueError(f"unsupported Turtle line: {line}")
        subject, predicate, obj = match.groups()
        statements.append(_statement(_term(subject), _term(predicate), obj))
    return prefixes, statements


def _term(token: str) -> str:
    return token[1:-1] if token.startswith("<") else token


def _statement(subject: str, predicate: str, obj: str) -> Row:
    literal = LITERAL.match(obj)
    if literal is None:
        return (subject, subject, predicate, _term(obj), None, None, None)
    value, language, datatype = literal.groups()
    return (subject, subject, predicate, None, value, datatype, language)


def closure(statements: list[Row]) -> list[tuple[str, str, str]]:
    parents: dict[str, set[str]] = {}
    for _, subject, predicate, obj, *_ in statements:
        if predicate == SUBCLASS and obj is not None:
            parents.setdefault(subject, set()).add(obj)
    edges: set[tuple[str, str, str]] = set()
    for child in parents:
        stack, seen = list(parents[child]), set()
        while stack:
            ancestor = stack.pop()
            if ancestor in seen:
                continue
            seen.add(ancestor)
            edges.add((child, SUBCLASS, ancestor))
            stack.extend(parents.get(ancestor, ()))
    return sorted(edges)


def build() -> None:
    prefixes, statements = read_turtle(TTL.read_text())
    OUT.unlink(missing_ok=True)
    with sqlite3.connect(OUT) as con:
        con.executescript(BASE_DDL)
        con.executemany("INSERT INTO prefix VALUES (?, ?)", prefixes)
        con.executemany("INSERT INTO statements VALUES (?, ?, ?, ?, ?, ?, ?)", statements)
        con.executemany("INSERT INTO entailed_edge VALUES (?, ?, ?)", closure(statements))
        con.executescript(VIEWS.read_text())
    con.close()


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}")
