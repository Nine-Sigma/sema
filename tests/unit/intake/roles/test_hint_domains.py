"""A hint's per-file ``domains`` are human reference choices, judged by the Roles verifier."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.intake.roles.eligibility_text_utils import VOCABULARY_ROLE
from sema.intake.verify.roles import IN_ELIGIBILITY_SET, verify_roles
from tests.unit.intake.eligibility_env import (
    ACRIS_RUN, Env, derived_free_hint, descriptor, eligibility, model_calls, write_hint,
)
from tests.unit.intake.structure_env import ACRIS_SPEC, fresh_tools

pytestmark = pytest.mark.unit

UNLOADED_SUFFIX = "_unloaded"


class Domain(NamedTuple):
    relation: str
    prop: str
    vocabulary: str


class Hinted(NamedTuple):
    roles: Any
    spec: ConceptSourceSpec | None
    verdict: Any
    calls: list[Path]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Env]:
    held = Env(tmp_path_factory.mktemp("hint_domains"))
    yield held
    held.close()


def _oracle_domains() -> list[Domain]:
    oracle = load_concept_source_spec(ACRIS_SPEC)
    assert oracle.entity_model is not None and oracle.entity_model.entities
    return [
        Domain(entry.relation.name, prop, vocabulary)
        for entry in oracle.entity_model.entities
        for prop, vocabulary in entry.domains.items()
    ]


def _reference(roles: Any, domain: Domain) -> Any:
    (entry,) = [
        e for e in roles.eligibilities
        if (e.relation, e.role, e.subject) == (domain.relation, VOCABULARY_ROLE, domain.prop)
    ]
    return entry


def _hint_stating(directory: Path, domain: Domain | None) -> ConceptSourceSpec:
    """The derived-free hint with no ``domains`` but ``domain``, on its one per-file entry."""
    base = derived_free_hint(directory / "base")
    raw = {"concept_source": base.model_dump(mode="json", by_alias=True, exclude_unset=True)}
    entities = raw["concept_source"]["entity_model"]["entities"]
    for entity in entities:
        entity.pop("domains", None)
    if domain is not None:
        (entry,) = [e for e in entities if _relation_name(e["relation"]) == domain.relation]
        entry["domains"] = {domain.prop: domain.vocabulary}
    return write_hint(directory, raw)


def _relation_name(raw: Any) -> str:
    return str(raw["name"]) if isinstance(raw, dict) else str(raw)


def _hinted_run(env: Env, name: str, domain: Domain | None) -> Hinted:
    hint = _hint_stating(env.root / name / "hint", domain)
    source = env.acris.source
    structured = env.structured(env.acris, name, hint)
    ledger = env.root / name / "roles"
    tools = fresh_tools(source, ledger, stage="roles", structure=structured)
    roles = eligibility().run_roles(structured, source.profiles(), tools, hint=hint)
    spec = eligibility().assemble_spec(descriptor(ACRIS_RUN), roles)
    verdict = verify_roles(roles, tools, spec=spec)
    return Hinted(roles, spec, verdict, model_calls(ledger))


def _membership_failures(verdict: Any) -> list[Any]:
    return [f for f in verdict.failures if f.check == IN_ELIGIBILITY_SET]


def _spec_domains(spec: ConceptSourceSpec, relation: str) -> dict[str, str]:
    assert spec.entity_model is not None and spec.entity_model.entities
    (entry,) = [e for e in spec.entity_model.entities if e.relation.name == relation]
    return dict(entry.domains)


def test_a_hint_domain_on_an_unloaded_vocabulary_fails_membership(env: Env) -> None:
    first = _oracle_domains()[0]
    unloaded = Domain(first.relation, first.prop, first.vocabulary + UNLOADED_SUFFIX)

    run = _hinted_run(env, "unloaded", unloaded)

    assert unloaded.vocabulary not in _reference(run.roles, unloaded).members
    assert run.calls == []
    assert run.spec is not None, [(a.subject, a.members) for a in run.roles.ambiguities]
    failures = _membership_failures(run.verdict)
    assert len(failures) == 1, run.verdict.failures
    (found,) = failures
    assert (found.relation, found.column) == (unloaded.relation, unloaded.prop)
    assert found.evidence["role"] == VOCABULARY_ROLE
    assert unloaded.vocabulary not in found.evidence["members"]
    assert unloaded.vocabulary in {v for v in found.evidence.values() if isinstance(v, str)}


def test_a_hint_domain_on_an_open_reference_reaches_the_spec(env: Env) -> None:
    """The positive leg: a stated member the rules left open becomes the spec's domain."""
    plain = _hinted_run(env, "domainless", None).roles
    open_domains = [d for d in _oracle_domains() if _reference(plain, d).choice is None]
    assert open_domains, "no oracle domain is left open by the rules"
    stated = open_domains[0]

    run = _hinted_run(env, "loaded", stated)

    assert run.calls == []
    assert run.spec is not None, [(a.subject, a.members) for a in run.roles.ambiguities]
    assert _spec_domains(run.spec, stated.relation).get(stated.prop) == stated.vocabulary
    entry = _reference(run.roles, stated)
    assert (entry.choice, entry.source) == (stated.vocabulary, "human")
    assert _membership_failures(run.verdict) == []
