"""Generality gate: no fixture literal appears in the general intake code or its prompts.

The literals are read from the fixture files at test time, so a fixture added
later is covered without editing this test.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import io
import json
import keyword
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from pathlib import Path

import openpyxl
import pytest
from pydantic import BaseModel

from sema.binding import spec_sections
from sema.intake import lexicon
from sema.intake.loaders.base import relation_name
from tests.unit.intake.source_literals_env import (
    REPO_ROOT,
    SCAN_ROOT,
    TOKEN,
    Hit,
    collapse,
    find_hits,
    scanned_prompts,
    tokens_of,
)

pytestmark = pytest.mark.unit

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


def fixture_literals(root: Path = FIXTURE_ROOT) -> dict[tuple[str, ...], str]:
    excluded = excluded_names()
    found: dict[tuple[str, ...], str] = {}
    for path in fixture_files(root):
        for literal in literals_of(path):
            tokens = tokens_of(literal)
            if len(literal) >= MIN_LITERAL_LENGTH and tokens and literal not in excluded:
                found.setdefault(tokens, literal)
    return found


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


def _lexicon_token_groups() -> list[Collection[str]]:
    groups: list[Collection[str]] = [entry.tokens for entry in lexicon.ROLES]
    return [*groups, *lexicon.HEADER_KEY_TOKENS.values(), *lexicon.STATUS_VALUE_WORDS.values()]


def _lexicon_words() -> set[str]:
    groups = _lexicon_token_groups()
    return {w for g in groups for w in (g.split() if isinstance(g, str) else g)}


def test_lexicon_role_tokens_are_tuples_of_single_words() -> None:
    for group in _lexicon_token_groups():
        assert not isinstance(group, str), f"space-joined token string {group!r}"
        assert all(TOKEN.fullmatch(word) for word in group), group
    for entry in lexicon.ROLES:
        values = [getattr(entry, field.name) for field in dataclasses.fields(entry)]
        assert " ".join(entry.tokens) not in values, f"{entry.name} spells its tokens as one string"


def test_lexicon_single_word_role_tokens_are_excluded() -> None:
    assert _lexicon_words() - excluded_names() == set()


def test_multi_word_fixture_literal_of_lexicon_words_is_still_a_hit(tmp_path: Path) -> None:
    literals = fixture_literals()
    words = _lexicon_words()
    phrase = min(text for key, text in literals.items() if 1 < len(key) and set(key) <= words)
    assert set(tokens_of(phrase)) <= excluded_names()
    (tmp_path / "words.py").write_text(f"PHRASE = {phrase!r}\nWORDS = {tuple(phrase.split())!r}\n")
    assert find_hits(tmp_path, literals) == [Hit("words.py", 1, phrase)]


def _longest_plain_literal(literals: Mapping[tuple[str, ...], str], multi: bool) -> str:
    return max(
        (
            text
            for tokens, text in literals.items()
            if re.fullmatch(r"[a-z0-9_ ]+", text) and (len(tokens) > 1) == multi
        ),
        key=len,
    )


def test_planted_multi_token_fixture_literal_is_flagged(tmp_path: Path) -> None:
    literals = fixture_literals()
    planted = _longest_plain_literal(literals, multi=True)
    module = tmp_path / "planted.py"
    module.write_text(
        f'"""{planted}"""\n\nRELATION = {planted!r}\n# reads {planted.upper()}\n'
        f"NOTE = {('reads the ' + planted + ' column')!r}\n",
        encoding="utf-8",
    )
    hits = find_hits(tmp_path, literals)
    assert {(hit.lineno, hit.literal) for hit in hits} >= {
        (3, planted), (4, planted), (5, planted)
    }
    assert all(hit.lineno != 1 for hit in hits)


def test_planted_single_token_fixture_literal_is_flagged_only_as_a_whole_constant(
    tmp_path: Path,
) -> None:
    literals = fixture_literals()
    planted = _longest_plain_literal(literals, multi=False)
    module = tmp_path / "planted.py"
    module.write_text(
        f'"""{planted}"""\n\nRELATION = {planted.upper()!r}\n# reads {planted}\n'
        f"QUERY = {('SELECT * FROM ' + planted)!r}\n",
        encoding="utf-8",
    )
    assert find_hits(tmp_path, literals) == [Hit("planted.py", 3, planted)]


def test_single_token_literal_as_an_entire_constant_is_a_hit(tmp_path: Path) -> None:
    literals: dict[tuple[str, ...], str] = {("values",): "values"}
    (tmp_path / "whole.py").write_text(
        "A = 'Values'\nB = '  VALUES \\n'\nC = ('val' 'ues')\n", "utf-8"
    )
    assert find_hits(tmp_path, literals) == [
        Hit("whole.py", 1, "values"), Hit("whole.py", 2, "values"), Hit("whole.py", 3, "values")
    ]


def test_single_token_literal_inside_sql_or_a_comment_is_not_a_hit(tmp_path: Path) -> None:
    literals: dict[tuple[str, ...], str] = {("values",): "values", ("source",): "source"}
    (tmp_path / "sql.py").write_text(
        "INSERT = 'INSERT INTO runs (source_identity) VALUES (?)'\n"
        "# Values\n"
        "# the source of these values\n"
        "LABEL = 'source identity'\n",
        "utf-8",
    )
    assert find_hits(tmp_path, literals) == []


def test_multi_token_literal_inside_a_longer_constant_or_comment_is_a_hit(
    tmp_path: Path,
) -> None:
    literals: dict[tuple[str, ...], str] = {
        ("good", "through", "date"): "good through date",
        ("column", "info"): "column info",
    }
    (tmp_path / "multi.py").write_text(
        "A = 'reads the Good Through Date field'\n"
        "# a Column  Info sheet\n"
        "B = 'Good Through'\n",
        "utf-8",
    )
    assert find_hits(tmp_path, literals) == [
        Hit("multi.py", 1, "good through date"), Hit("multi.py", 2, "column info")
    ]


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


def test_empty_prompts_directory_has_no_hit(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    assert find_hits(tmp_path, fixture_literals()) == []


def test_planted_fixture_literal_in_a_prompt_file_is_flagged(tmp_path: Path) -> None:
    literals = fixture_literals()
    planted = _longest_plain_literal(literals, multi=True)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "probe_v1.md").write_text(
        f"# prompt: probe v1\n\nChoose a member.\nRead the {planted.upper()} relation.\n",
        encoding="utf-8",
    )
    hits = find_hits(tmp_path, literals)
    assert Hit("prompts/probe_v1.md", 4, planted) in hits
    assert {hit.lineno for hit in hits} == {4}


def test_single_token_literal_in_prompt_prose_is_not_a_hit(tmp_path: Path) -> None:
    literals: dict[tuple[str, ...], str] = {("values",): "values"}
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "probe_v1.md").write_text("Values\nthe values shown\n", "utf-8")
    assert find_hits(tmp_path, literals) == []


def test_intake_prompt_files_are_scanned() -> None:
    names = {path.name for path in scanned_prompts()}
    assert "structure_v1.md" in names


def test_general_intake_code_names_no_fixture_literal() -> None:
    hits = find_hits(SCAN_ROOT, fixture_literals())
    assert hits == [], "fixture literals in general intake code:\n" + "\n".join(
        f"  {hit.path}:{hit.lineno}: {hit.literal!r}" for hit in hits
    )
