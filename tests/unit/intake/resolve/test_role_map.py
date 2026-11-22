"""One role map per relation signature, verified after the answer (S5 decisions 1, 2, 3, 6, 18).

Pinned contract (task 2.1, rulings R1 to R9), over invented relations and a mocked model:

- A. ``output_models.role_map_model({section: role model})`` is a flat class named
  ``RoleMap..._v1``: one string field ``role_<field>`` per field of every role model except
  ``notes`` and ``entity``, a repeat qualified by section (``role_value_sets_status``), plus
  ``confidences`` and ``reasons``; at most 16 keys for one model and 26 for two sections.
- B. The role-map prompt (the mock's ``invoke``) renders every column's name, type, comment,
  aggregates, lexicon token hits, embedding rank, and findings; a lexicon-settled role as a
  claim the model may contest; at most ``SCHEMA_TEXT_SAMPLE_ROWS`` schema rows and a
  rows-not-shown line; no value of a non-schema relation under ``metadata_only``.
- C. A failing answer is re-asked once (two calls) with the check and the role named, then
  left open: an invented name never becomes a member; shared or failing columns rank first.
  ``resolve.DEFAULT_MAX_RETRIES`` is 1 and both stages default to it.
- D. A finding never fails a choice; E. a contested lexicon claim opens the role with both
  sources; F. nothing open, no call; G. ``status`` under ``metadata_only`` is listed
  ``unverifiable`` and kept; H. kin inherit with ``source: memory``; I. token hits are facts.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from sema.binding.spec_sections import (
    Provider,
    CodeListRoles,
    ConceptRoles,
    DetailRoles,
    EntityModelRoles,
    ValueSetRoles,
)
from sema.intake.roles.eligibility_spec_utils import roles_of
from sema.intake.resolve.resolve import resolve_structure
from sema.intake.verify.roles import (
    COLUMN_EXISTS,
    ONE_ROLE_PER_COLUMN,
    UNVERIFIABLE,
    verify_roles,
)
from sema.intake.verify.roles_utils import DISPLAY_NULL_RATE
from tests.unit.intake.eligibility_env import KeywordEmbedder
from tests.unit.intake.model_calls_env import wired_run
from tests.unit.intake.role_map_env import (
    CODES, CONCEPTS, DICT, GHOST, KIN, NAMED, OWNERS, PLAIN, PLANTED, SPARSE, SQL, STATUSED,
    TALLIES, TALLY_FIELDS, TOKENS, VELL_COMMENT, CLAIMED, Scenario, chosen, claim_own,
    code_list, conflict_structure, explained, lines_with, mapped, model_names,
    module, output_models, per_file, prompts, scenario, schema_rows, single_relation,
    subject_entry, tally_fields_sql,
)
from tests.unit.intake.role_entries_env import concept_list
from tests.unit.intake.structure_env import fresh_tools

pytestmark = pytest.mark.unit

KEY_UNIQUE_NON_NULL = "key_unique_non_null"
SKIPPED = frozenset({"notes", "entity"})
SINGLE_MODELS: dict[str, type[BaseModel]] = {
    CONCEPTS: ConceptRoles, "code_lists": CodeListRoles, "value_sets": ValueSetRoles,
    "details": DetailRoles, "entity_model": EntityModelRoles,
}
IDENTITY = {CONCEPTS: {"role_title"}, "code_lists": {"role_vocabulary_id", "role_concept_code"}}
FLAT_BREAKERS = ("anyOf", "oneOf", "$ref", "$defs")
PLAIN_MAP = {"concept_id": "zorb", "name": "quax"}


def _role_fields(model: type[BaseModel]) -> set[str]:
    return {f"role_{name}" for name in model.model_fields if name not in SKIPPED}


def _assert_flat_role_map(built: type[BaseModel]) -> None:
    schema = built.model_json_schema()
    assert built.__name__.startswith("RoleMap") and built.__name__.endswith("_v1"), built
    assert not [word for word in FLAT_BREAKERS if word in json.dumps(schema)], schema
    for name, spec in schema["properties"].items():
        if name.startswith("role_"):
            assert spec.get("type") == "string" and "enum" not in spec, (name, spec)


@pytest.mark.parametrize("section", sorted(SINGLE_MODELS))
def test_role_map_model_has_one_string_field_per_role(section: str) -> None:
    model = SINGLE_MODELS[section]

    built = output_models().role_map_model({section: model})

    _assert_flat_role_map(built)
    assert set(built.model_fields) == _role_fields(model) | {"confidences", "reasons"}
    assert IDENTITY.get(section, set()) <= set(built.model_fields)
    assert len(built.model_fields) <= 16


def test_two_section_role_map_qualifies_each_repeated_role() -> None:
    sections: dict[str, type[BaseModel]] = {CONCEPTS: ConceptRoles, "value_sets": ValueSetRoles}
    built = output_models().role_map_model(sections)

    _assert_flat_role_map(built)
    keys = set(built.model_fields)
    repeated = set(ConceptRoles.model_fields) & set(ValueSetRoles.model_fields)
    assert len(keys) == 26 and "role_value_sets_status" in keys, sorted(keys)
    for name in repeated:
        assert f"role_value_sets_{name}" in keys and len([k for k in keys if k.endswith(name)]) == 2
    single = (set(ConceptRoles.model_fields) ^ set(ValueSetRoles.model_fields)) - SKIPPED
    assert {f"role_{name}" for name in single} <= keys


def _open(root: Path, *keys: str, **kwargs: Any) -> Scenario:
    bindings = {
        PLAIN: concept_list(PLAIN, CONCEPTS), TALLIES: concept_list(TALLIES, CONCEPTS),
        CODES: code_list(CODES), TOKENS: code_list(TOKENS), SPARSE: code_list(SPARSE),
        STATUSED: code_list(STATUSED), NAMED: code_list(NAMED), KIN[0]: code_list(KIN[0]),
        KIN[1]: code_list(KIN[1]), DICT: single_relation(DICT, "table_label", "field_label"),
    }
    statements = [line for key in keys for line in SQL[key]]
    chosen_bindings = [bindings[key] for key in keys]
    return scenario(statements, chosen_bindings, root, **kwargs)


@pytest.fixture
def tallies(tmp_path: Path) -> Iterator[tuple[Scenario, Any]]:
    embedder = KeywordEmbedder({"vell": [1.0, 0.0], "explaining": [1.0, 0.0]}, [0.0, 1.0])
    found = _open(tmp_path, TALLIES, comment=(TALLIES, "vell", VELL_COMMENT), embedder=embedder)
    client = mapped({**PLAIN_MAP, "name": "quax_name"})
    found.resolve(client)
    yield found, client
    found.con.close()


def _role_prompts(client: Any) -> list[str]:
    names = model_names(client)
    assert names and all(name.startswith("RoleMap") for name in names), names
    return prompts(client)


def test_the_prompt_renders_every_column_with_its_facts(tallies: tuple[Scenario, Any]) -> None:
    (prompt, *_) = _role_prompts(tallies[1])

    for column in ("zorb", "flag", "vell", "quax_name"):
        assert lines_with(prompt, column, "VARCHAR", "distinct", "null rate"), column
    assert lines_with(prompt, "vell", "mean length", VELL_COMMENT)
    assert lines_with(prompt, "flag", "constant")


def test_token_hits_and_embedding_rank_are_facts_in_the_prompt(
    tallies: tuple[Scenario, Any],
) -> None:
    (prompt, *_) = _role_prompts(tallies[1])

    hits = lines_with(prompt, "quax_name", "name")
    assert [line for line in hits if "lexicon" in line or "token" in line], hits
    assert lines_with(prompt, "embedding", "vell", "definition")


def test_a_lexicon_settled_role_is_a_claim_the_map_still_asks(
    tallies: tuple[Scenario, Any],
) -> None:
    scene, client = tallies
    (prompt, *_) = _role_prompts(client)

    assert lines_with(prompt, "name", "quax_name", "lexicon")
    assert "role_name" in client.models[0].model_fields


def test_no_value_of_a_non_schema_relation_reaches_the_prompt(
    tallies: tuple[Scenario, Any],
) -> None:
    sent = _role_prompts(tallies[1])

    assert not [prompt for prompt in sent if PLANTED in prompt]


def test_schema_text_is_a_fixed_sample_with_the_rows_not_shown(tmp_path: Path) -> None:
    scene = scenario([tally_fields_sql()], [per_file(TALLY_FIELDS)], tmp_path)
    client = mapped({"property": "field_name", "description": "field_description"})

    scene.resolve(client)

    cap = int(module("sema.intake.resolve.resolve_roles_utils").SCHEMA_TEXT_SAMPLE_ROWS)
    (prompt, *_) = _role_prompts(client)
    assert 0 < len(lines_with(prompt, "tally_field_")) <= cap
    assert lines_with(prompt, "not shown", str(schema_rows() - cap))
    scene.con.close()


def _assert_reasked_once(client: Any, *named: str) -> None:
    sent = prompts(client)
    assert len(sent) == 2, f"{len(sent)} calls"
    assert all(word in sent[1] for word in named), [w for w in named if w not in sent[1]]


def test_a_name_outside_the_relation_is_reasked_once_then_left_open(tmp_path: Path) -> None:
    scene = _open(tmp_path, PLAIN)
    client = mapped({**PLAIN_MAP, "name": GHOST})

    result, _ = scene.resolve(client)

    _assert_reasked_once(client, COLUMN_EXISTS, "name", GHOST)
    entry = subject_entry(result, f"{PLAIN}.name")
    assert entry.choice is None and entry.members and GHOST not in entry.members, entry
    assert set(entry.members) <= {"zorb", "quax", "vell", "none"}, entry.members
    assert GHOST in explained(entry) and COLUMN_EXISTS in explained(entry), entry
    scene.con.close()


def test_two_roles_on_one_column_are_reasked_once_then_left_open_ranked_first(
    tmp_path: Path,
) -> None:
    scene = _open(tmp_path, PLAIN)
    client = mapped({**PLAIN_MAP, "definition": "quax"})

    result, _ = scene.resolve(client)

    _assert_reasked_once(client, ONE_ROLE_PER_COLUMN, "name", "definition")
    for role in ("name", "definition"):
        entry = subject_entry(result, f"{PLAIN}.{role}")
        assert entry.choice is None and entry.members[:1] == ("quax",), entry
    scene.con.close()


def test_a_failed_data_fact_is_reasked_once_then_left_open_with_the_check(
    tmp_path: Path,
) -> None:
    scene = _open(tmp_path, CODES)
    client = mapped({"code": "pell", "display": "quax"})

    result, _ = scene.resolve(client)

    _assert_reasked_once(client, KEY_UNIQUE_NON_NULL, "code")
    entry = subject_entry(result, f"{CODES}.code")
    assert entry.choice is None and entry.members[:1] == ("pell",), entry
    assert KEY_UNIQUE_NON_NULL in explained(entry), entry
    scene.con.close()


def test_a_failing_structure_choice_is_reasked_once_then_left_open(tmp_path: Path) -> None:
    con, source, structure = conflict_structure(tmp_path)
    client = mapped(entries=claim_own)
    wired = wired_run(tmp_path / "calls", client)
    tools = fresh_tools(source, tmp_path / "ledger" / "resolve")

    provider = cast(Provider, source.provider)
    result = resolve_structure(structure, source.land, tools, wired.calls, provider=provider)

    assert client.invoke.call_count == 2
    assert {a.subject for a in result.ambiguities if a.choice is None} >= {*OWNERS, CLAIMED}
    con.close()


def test_both_stages_retry_once_by_one_constant() -> None:
    resolve = module("sema.intake.resolve.resolve")

    defaults = {
        name: inspect.signature(getattr(resolve, name)).parameters["max_retries"].default
        for name in ("resolve_structure", "resolve_roles")
    }

    assert resolve.DEFAULT_MAX_RETRIES == 1
    assert defaults == {"resolve_structure": 1, "resolve_roles": 1}


def test_a_finding_on_a_chosen_column_never_fails_it(tmp_path: Path) -> None:
    scene = _open(tmp_path, SPARSE)
    client = mapped({"code": "kelt", "display": "wim"})

    result, _ = scene.resolve(client)

    assert client.invoke.call_count == 1
    display = chosen(result, SPARSE, "display")
    assert (display.choice, display.source) == ("wim", "model"), display
    assert DISPLAY_NULL_RATE in explained(subject_entry(result, f"{SPARSE}.display"))
    verdict = verify_roles(result, scene.tools(), spec=None, relations=(SPARSE,))
    assert not [f for f in verdict.failures if f.check == DISPLAY_NULL_RATE], verdict.failures
    assert DISPLAY_NULL_RATE in json.dumps(verdict.model_dump(mode="json", exclude={"failures"}))
    scene.con.close()


def test_a_contested_lexicon_claim_opens_the_role_with_both_answers(tmp_path: Path) -> None:
    scene = _open(tmp_path, TOKENS)
    client = mapped({"code": "zorb", "display": "item_label"})

    result, _ = scene.resolve(client)

    entry = subject_entry(result, f"{TOKENS}.code")
    assert entry.choice is None and {"item_code", "zorb"} <= set(entry.members), entry
    assert "lexicon" in explained(entry) and "model" in explained(entry), entry
    assert chosen(result, TOKENS, "code").choice is None
    scene.con.close()


def test_a_relation_with_every_role_settled_makes_no_call(tmp_path: Path) -> None:
    """Regression guard: passes on HEAD (R6)."""
    scene = _open(tmp_path, NAMED)
    client = mapped()

    _, wired = scene.resolve(client)

    assert client.invoke.call_count == 0
    assert wired.row().model_calls == 0
    scene.con.close()


def test_status_under_metadata_only_is_unverifiable_and_kept(tmp_path: Path) -> None:
    scene = _open(tmp_path, STATUSED)

    status = chosen(scene.roles, STATUSED, "status")
    verdict = verify_roles(scene.roles, scene.tools(), spec=None, relations=(STATUSED,))

    assert status is not None and status.choice == "item_status", status
    listed = [
        a for a in verdict.ambiguities
        if UNVERIFIABLE in a.hints and a.subject.startswith(STATUSED) and "status" in a.subject
    ]
    assert listed and all(a.reason for a in listed), verdict.ambiguities
    assembled = roles_of(STATUSED, CodeListRoles, scene.roles)
    assert assembled is not None and assembled.status is not None, assembled
    assert assembled.status.column == "item_status"
    scene.con.close()


def test_kin_inherit_the_role_map_with_source_memory(tmp_path: Path) -> None:
    """Regression guard: passes on HEAD."""
    scene = _open(tmp_path, *KIN)
    client = mapped({"code": "zorb", "display": "quax"})

    result, _ = scene.resolve(client)

    assert client.invoke.call_count == 1
    inherited = chosen(result, KIN[1], "display")
    assert (inherited.choice, inherited.source) == ("quax", "memory"), inherited
    scene.con.close()


def test_a_column_with_no_token_hit_may_be_named_for_primary_key(tmp_path: Path) -> None:
    scene = _open(tmp_path, DICT)
    client = mapped({"property": "field_label", "primary_key": "key_flag"})

    result, _ = scene.resolve(client)

    assert [prompt for prompt in prompts(client) if lines_with(prompt, "key_flag")]
    found = chosen(result, DICT, "primary_key")
    assert found is not None and (found.choice, found.source) == ("key_flag", "model"), found
    scene.con.close()
