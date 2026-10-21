"""A verified layout proposal is re-proved against what it was proposed over (task 7.6).

A model-bound entry over a proposal target is not one the predicates produced, so the
verifier runs its template's ``verify`` over the relation, or over the view it names,
instead of checking it against the recomputed candidates. Code-bound entries keep the
eligibility check.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sema.intake.loaders.tabular import land_tabular
from sema.intake.shapes.base import SlotBinding
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.verify.structure import verify_structure
from sema.intake.views import ViewSpec
from tests.unit.intake.layout_proposal_env import (
    CODEBOOK,
    CODEBOOK_CSV,
    VARIABLE,
    Landed,
    checks,
    write_codebook,
)
from tests.unit.intake.structure_env import fresh_tools, land, run

pytestmark = pytest.mark.unit

DENORMALIZED = "concept_with_values_denormalized"
FILL_DOWN = "fill_down"
CALL_KEY = "a" * 64


@pytest.fixture(scope="module")
def codebook(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    root = tmp_path_factory.mktemp("structure_proposed")
    path = write_codebook(root / "inputs", CODEBOOK_CSV)
    staging, source = land(root, "codebook", land_tabular, path)
    yield Landed(source, run(source, fresh_tools(source, root / "ledger" / "structure")))
    staging.close()


def _bound(landed: Landed, entry: BoundEntry) -> StructureResult:
    relation = landed.name(CODEBOOK)
    structure = landed.structure
    ambiguities = tuple(a for a in structure.ambiguities if a.subject != relation)
    update = {"bound": (*structure.bound, entry), "ambiguities": ambiguities}
    result: StructureResult = structure.model_copy(update=update)
    return result


def _entry(landed: Landed, *, source: str, view: ViewSpec | None) -> BoundEntry:
    relation = landed.name(CODEBOOK)
    binding = SlotBinding(
        template=DENORMALIZED,
        slots={"relation": relation, "group_by": VARIABLE},
        relations=(relation,),
    )
    key = CALL_KEY if source == "model" else None
    return BoundEntry(binding=binding, confidence=0.8, source=source, call_key=key, view=view)


def _verify(landed: Landed, proposal: StructureResult, tmp_path: Path) -> object:
    tools = fresh_tools(landed.source, tmp_path / "ledger" / "verify")
    return verify_structure(proposal, landed.source.land, tools, provider=landed.source.provider)


def _spec(landed: Landed) -> ViewSpec:
    return ViewSpec(view=FILL_DOWN, relation=landed.name(CODEBOOK), columns=(VARIABLE,))


def test_codebook_is_a_proposal_target(codebook: Landed) -> None:
    assert codebook.structure.proposal_targets == (codebook.name(CODEBOOK),)


def test_model_proposal_over_a_view_verifies_over_the_view(
    codebook: Landed, tmp_path: Path
) -> None:
    entry = _entry(codebook, source="model", view=_spec(codebook))

    verdict = _verify(codebook, _bound(codebook, entry), tmp_path)

    assert verdict.status == "pass"  # type: ignore[attr-defined]


def test_model_proposal_without_its_view_fails_the_template_not_eligibility(
    codebook: Landed, tmp_path: Path
) -> None:
    entry = _entry(codebook, source="model", view=None)

    verdict = _verify(codebook, _bound(codebook, entry), tmp_path)

    found = checks(verdict.failures)  # type: ignore[attr-defined]
    assert "template_verifies" in found
    assert "group_by_non_null" in found
    assert "in_eligibility_set" not in found
    sources = {f.evidence["source"] for f in verdict.failures}  # type: ignore[attr-defined]
    assert sources == {"model"}


def test_code_bound_entry_on_a_target_still_fails_eligibility(
    codebook: Landed, tmp_path: Path
) -> None:
    entry = _entry(codebook, source="code", view=None)

    verdict = _verify(codebook, _bound(codebook, entry), tmp_path)

    assert "in_eligibility_set" in checks(verdict.failures)  # type: ignore[attr-defined]


def test_refused_view_fails_with_the_view_check(codebook: Landed, tmp_path: Path) -> None:
    spec = ViewSpec(view=FILL_DOWN, relation=codebook.name(CODEBOOK), columns=("absent",))
    entry = _entry(codebook, source="model", view=spec)

    verdict = _verify(codebook, _bound(codebook, entry), tmp_path)

    assert "column_exists" in checks(verdict.failures)  # type: ignore[attr-defined]
