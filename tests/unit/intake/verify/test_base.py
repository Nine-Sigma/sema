"""One verdict shape for every stage verifier: status, failures, ambiguities, floors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

pytestmark = pytest.mark.unit

FLOORS: dict[str, Any] = {
    "ID_DISTINCT_RATIO_FLOOR": 0.98,
    "LABEL_MEAN_LENGTH_BOUNDS": (3.0, 120.0),
    "EMBEDDING_MARGIN_FLOOR": None,
}


def _failure(**overrides: Any) -> Any:
    from sema.intake.verify.base import Failure

    fields: dict[str, Any] = {
        "check": "display_null_rate",
        "relation": "land.dict__values",
        "column": "label",
        "evidence": {"null_rate": 0.4, "floor": 0.05},
    }
    return Failure(**(fields | overrides))


def _ambiguity(**overrides: Any) -> Any:
    from sema.intake.verify.base import Ambiguity

    fields: dict[str, Any] = {
        "stage": "roles",
        "subject": "land.dict__values.concept_id",
        "members": ("id", "code"),
        "hints": ("lexicon:id",),
        "exclusions": {"code": "code"},
        "choice": "id",
        "confidence": 1.0,
        "source": "lexicon",
    }
    return Ambiguity(**(fields | overrides))


def test_failure_names_check_relation_column_and_evidence() -> None:
    failure = _failure()

    assert (failure.check, failure.relation, failure.column) == (
        "display_null_rate",
        "land.dict__values",
        "label",
    )
    assert failure.evidence == {"null_rate": 0.4, "floor": 0.05}


def test_failure_column_is_optional() -> None:
    failure = _failure(column=None, check="relation_count")

    assert failure.column is None


def test_ambiguity_carries_every_listed_field() -> None:
    ambiguity = _ambiguity()

    assert ambiguity.stage == "roles"
    assert ambiguity.subject == "land.dict__values.concept_id"
    assert tuple(ambiguity.members) == ("id", "code")
    assert tuple(ambiguity.hints) == ("lexicon:id",)
    assert dict(ambiguity.exclusions) == {"code": "code"}
    assert (ambiguity.choice, ambiguity.confidence, ambiguity.source) == ("id", 1.0, "lexicon")


def test_open_ambiguity_has_no_choice_confidence_or_source() -> None:
    ambiguity = _ambiguity(choice=None, confidence=None, source=None, hints=(), exclusions={})

    assert ambiguity.choice is None
    assert ambiguity.confidence is None
    assert ambiguity.source is None


def test_choice_outside_the_members_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _ambiguity(choice="label")


def test_choice_without_a_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _ambiguity(source=None)


def test_verifier_status_is_fail_when_any_check_fails() -> None:
    from sema.intake.verify.base import Verdict

    verdict = Verdict.of(failures=[_failure()], ambiguities=[], floors=FLOORS)

    assert verdict.status == "fail"
    assert [f.check for f in verdict.failures] == ["display_null_rate"]


def test_verifier_status_is_pass_with_no_failures_even_with_ambiguities() -> None:
    from sema.intake.verify.base import Verdict

    verdict = Verdict.of(failures=[], ambiguities=[_ambiguity()], floors=FLOORS)

    assert verdict.status == "pass"
    assert len(verdict.ambiguities) == 1


def test_verifier_helper_never_produces_escalate() -> None:
    from sema.intake.verify.base import Verdict

    statuses = {
        Verdict.of(failures=failures, ambiguities=[], floors={}).status
        for failures in ([], [_failure()], [_failure(), _failure(check="other")])
    }

    assert statuses == {"pass", "fail"}


def test_escalate_is_constructible_for_the_orchestrator() -> None:
    from sema.intake.verify.base import Verdict

    verdict = Verdict(status="escalate", failures=[_failure()], ambiguities=[], floors=FLOORS)

    assert verdict.status == "escalate"


def test_unknown_status_is_rejected() -> None:
    from sema.intake.verify.base import Verdict

    with pytest.raises(ValidationError):
        Verdict(status="retry", failures=[], ambiguities=[], floors={})


def test_pass_with_a_failure_is_rejected() -> None:
    from sema.intake.verify.base import Verdict

    with pytest.raises(ValidationError):
        Verdict(status="pass", failures=[_failure()], ambiguities=[], floors={})


def test_fail_without_a_failure_is_rejected() -> None:
    from sema.intake.verify.base import Verdict

    with pytest.raises(ValidationError):
        Verdict(status="fail", failures=[], ambiguities=[], floors={})


def test_verdict_records_the_floors_in_force() -> None:
    from sema.intake.verify.base import Verdict

    verdict = Verdict.of(failures=[], ambiguities=[], floors=FLOORS)

    assert verdict.floors["ID_DISTINCT_RATIO_FLOOR"] == 0.98
    assert tuple(verdict.floors["LABEL_MEAN_LENGTH_BOUNDS"]) == (3.0, 120.0)
    assert verdict.floors["EMBEDDING_MARGIN_FLOOR"] is None


def test_verdict_json_round_trip_is_lossless() -> None:
    from sema.intake.verify.base import Verdict

    verdict = Verdict.of(
        failures=[_failure(), _failure(column=None, check="relation_count", evidence={})],
        ambiguities=[_ambiguity(), _ambiguity(choice=None, confidence=None, source=None)],
        floors=FLOORS,
    )

    restored = Verdict.model_validate_json(verdict.model_dump_json())

    assert restored == verdict
    assert isinstance(json.loads(verdict.model_dump_json()), dict)


def test_ambiguity_and_failure_round_trip_on_their_own() -> None:
    from sema.intake.verify.base import Ambiguity, Failure

    failure, ambiguity = _failure(), _ambiguity()

    assert Failure.model_validate_json(failure.model_dump_json()) == failure
    assert Ambiguity.model_validate_json(ambiguity.model_dump_json()) == ambiguity


def test_verdict_is_a_ledger_artifact(tmp_path: Path) -> None:
    from sema.intake.calls.ledger import Ledger
    from sema.intake.verify.base import Verdict

    verdict = Verdict.of(failures=[_failure()], ambiguities=[_ambiguity()], floors=FLOORS)
    assert isinstance(verdict, BaseModel)

    path = Ledger(tmp_path).run("run-1").artifact("structure_verdict", verdict)

    assert Verdict.model_validate_json(path.read_text()) == verdict

