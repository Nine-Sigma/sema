"""Generality gate: no fixture literal appears in the general intake code or its prompts.

The literals are read from the fixture files at test time, so a fixture added
later is covered without editing this test.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from sema.binding import spec_sections
from sema.intake.roles import lexicon
from tests.unit.source_literals_env import (
    FIXTURE_ROOT,
    SCAN_ROOT,
    TOKEN,
    Hit,
    _grid_literals,
    _lexicon_token_groups,
    _table_schema_vocabulary,
    excluded_names,
    find_hits,
    fixture_files,
    fixture_literals,
    literals_of,
    scanned_prompts,
    tokens_of,
)

pytestmark = pytest.mark.unit


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
