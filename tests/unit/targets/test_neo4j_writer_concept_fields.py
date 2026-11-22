"""Neo4j writer params for the concept-binding node properties (design 5.2, 5.9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.loader import load_target
from sema.targets.neo4j_writer import Neo4jGraphWriter

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"
PROPERTY_CONCEPT_KEYS = (
    "concept_ref_system",
    "concept_ref_id",
    "concept_ref_version",
    "definition",
    "external_refs",
)
TERM_CONCEPT_KEYS = ("definition", "synonyms", "external_refs")

Call = tuple[str, dict[str, Any]]


def _mock_driver_capturing(calls: list[Call]) -> MagicMock:
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.side_effect = lambda cypher, **kwargs: calls.append((cypher, kwargs))
    driver.session.return_value = session
    return driver


def _load_and_capture(manifest: str) -> list[Call]:
    calls: list[Call] = []
    writer = Neo4jGraphWriter(_mock_driver_capturing(calls))
    load_target(ManifestTargetAdapter(FIXTURES / manifest), writer=writer)
    return calls


def _merges(calls: list[Call], label: str) -> list[Call]:
    return [c for c in calls if c[0].startswith(f"MERGE (n:{label} ")]


def _concept_params(call: Call, keys: tuple[str, ...]) -> dict[str, Any]:
    cypher, params = call
    missing = [k for k in keys if f"${k}" not in cypher or k not in params]
    assert not missing, f"concept params not written: {missing}"
    return {k: params[k] for k in keys}


def test_concept_manifest_sends_encoded_property_params() -> None:
    properties = {
        c[1]["name"]: c for c in _merges(_load_and_capture("concept_manifest.yaml"), "Property")
    }
    assert _concept_params(properties["race"], PROPERTY_CONCEPT_KEYS) == {
        "concept_ref_system": "partner",
        "concept_ref_id": "PRT4013886",
        "concept_ref_version": "2026-09",
        "definition": "A self-identified social construct of ancestral grouping.",
        "external_refs": ["omop:4013886", "SNOMED:103579009"],
    }
    assert _concept_params(properties["record_note"], PROPERTY_CONCEPT_KEYS) == {
        "concept_ref_system": None,
        "concept_ref_id": None,
        "concept_ref_version": None,
        "definition": None,
        "external_refs": [],
    }


def test_concept_manifest_sends_encoded_term_params() -> None:
    terms = {c[1]["code"]: c for c in _merges(_load_and_capture("concept_manifest.yaml"), "Term")}
    white = _concept_params(terms["PRT1000001"], TERM_CONCEPT_KEYS)
    assert white["definition"] is not None
    assert white["synonyms"] == ["Caucasian"]
    assert white["external_refs"] == ["omop:8527"]
    assert _concept_params(terms["PRT1000002"], TERM_CONCEPT_KEYS) == {
        "definition": None,
        "synonyms": [],
        "external_refs": [],
    }


def test_golden_manifest_sends_null_and_empty_concept_params() -> None:
    calls = _load_and_capture("golden_manifest.yaml")
    empty_property = dict.fromkeys(PROPERTY_CONCEPT_KEYS[:-1]) | {"external_refs": []}
    empty_term = {"definition": None, "synonyms": [], "external_refs": []}
    for call in _merges(calls, "Property"):
        assert _concept_params(call, PROPERTY_CONCEPT_KEYS) == empty_property
    for call in _merges(calls, "Term"):
        assert _concept_params(call, TERM_CONCEPT_KEYS) == empty_term
