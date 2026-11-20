"""Shared setup for test_acris_import.py and the tests that reuse it."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.ingest.duckdb_staging import Staging

FIXTURES = Path(__file__).parents[2] / "fixtures" / "intake"
ACRIS = FIXTURES / "acris"
SPEC = ACRIS / "acris.spec.yaml"

SCHEMA = "land_acris"
ENTITIES = 15

JOIN_COLUMNS = ["Document ID", "Good Through Date"]
REAL_MASTER = "acris_real_property_master"
PERSONAL_MASTER = "acris_personal_property_master"
FOREIGN_KEYS = {
    ("acris_real_property_legals", REAL_MASTER),
    ("acris_real_property_parties", REAL_MASTER),
    ("acris_real_property_remarks", REAL_MASTER),
    ("acris_real_property_references", REAL_MASTER),
    ("acris_personal_property_legals", PERSONAL_MASTER),
    ("acris_personal_property_parties", PERSONAL_MASTER),
    ("acris_personal_property_remarks", PERSONAL_MASTER),
    ("acris_personal_property_references", PERSONAL_MASTER),
}
DOMAINS = {
    ("acris_real_property_legals", "Property Type", "Property_Types_Codes"),
    ("acris_personal_property_legals", "Property Type", "Property_Types_Codes"),
    ("acris_real_property_master", "Docoument Type", "Document_Control_Codes"),
    ("acris_personal_property_master", "Docoument Type", "Document_Control_Codes"),
    ("acris_personal_property_master", "UCC Collateral", "UCC_Collateral_Codes"),
    ("acris_real_property_parties", "Country", "Country_Codes"),
    ("acris_personal_property_parties", "Country", "Country_Codes"),
    ("acris_real_property_parties", "State", "State_Codes"),
    ("acris_personal_property_parties", "State", "State_Codes"),
    ("acris_country_codes", "COUNTRY CODE", "Country_Codes"),
    ("acris_property_type_codes", "PROPERTY TYPE", "Property_Types_Codes"),
    ("acris_state_codes", "State Code", "State_Codes"),
    ("acris_ucc_collateral_codes", "UCC COLLATERAL CODE", "UCC_Collateral_Codes"),
}


@dataclass(frozen=True)
class _Imported:
    spec: Any
    landed: Any
    result: Any


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Imported]:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec
    from sema.intake.land.loaders.tabular import land_tabular
    from sema.intake.land.relations import RelationProvider

    spec = load_concept_source_spec(SPEC)
    path = tmp_path_factory.mktemp("acris_import") / "staging.duckdb"
    with Staging(str(path), schemas=()) as staging:
        landed = land_tabular(ACRIS, staging, schema=SCHEMA)
        provider = RelationProvider(warehouse=None, staging=staging)
        result = import_concept_source(spec, provider, dataset_map={})
        yield _Imported(spec=spec, landed=landed, result=result)
