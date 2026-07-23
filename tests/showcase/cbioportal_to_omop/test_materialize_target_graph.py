"""S2-09 showcase unit tests: value-mapping → specs, OmopConceptSource over an
in-memory OMOP vocabulary, and CLI command registration."""

from __future__ import annotations

from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from sema.cli import cli
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping
from showcase.cbioportal_to_omop.materialize_target_graph import (
    OmopConceptSource,
    value_mappings_to_specs,
)

pytestmark = pytest.mark.unit


def _mapping(value: str, concept_id: int | None, status: ResolutionStatus) -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=value,
        target_property_ref="omop.condition_occurrence.condition_concept_id",
        target_field="condition_concept_id",
        vocab_binding="OMOP-Condition",
        concept_id=concept_id,
        vocab_release="omop-vocab-2024",
        valid_start=None,
        valid_end=None,
        resolution_status=status,
        no_map_reason=None if status is ResolutionStatus.RESOLVED else "no crosswalk",
        confidence=1.0,
        status=None,
        resolver_policy_ref="omop.oncotree_condition",
        run_id="run-1",
    )


def test_value_mappings_to_specs_excludes_no_map() -> None:
    mappings = [
        _mapping("GBM", 4001458, ResolutionStatus.RESOLVED),
        _mapping("ASTR", 4001459, ResolutionStatus.RESOLVED),
        _mapping("PANEC", None, ResolutionStatus.NO_MAP),
    ]
    fields, bridges = value_mappings_to_specs(mappings)
    assert len(fields) == 1
    assert set(fields[0].concept_codes) == {"4001458", "4001459"}
    assert {(b.source_code, b.concept_code) for b in bridges} == {
        ("GBM", "4001458"), ("ASTR", "4001459"),
    }
    assert all(b.concept_code != "None" for b in bridges)


def _omop_conn() -> Any:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA v")
    conn.execute(
        "CREATE TABLE v.concept (concept_id BIGINT, concept_name VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE v.concept_synonym "
        "(concept_id BIGINT, concept_synonym_name VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE v.concept_ancestor "
        "(ancestor_concept_id BIGINT, descendant_concept_id BIGINT)"
    )
    conn.execute(
        "INSERT INTO v.concept VALUES "
        "(4001458, 'Glioblastoma multiforme'), (444, 'Malignant neoplasm')"
    )
    conn.execute("INSERT INTO v.concept_synonym VALUES (4001458, 'GBM')")
    conn.execute("INSERT INTO v.concept_ancestor VALUES (444, 4001458)")
    return conn


def test_omop_concept_source_name_synonyms_and_ancestors() -> None:
    source = OmopConceptSource(_omop_conn(), "v")
    detail = source.name_synonyms("4001458")
    assert detail is not None
    assert detail.name == "Glioblastoma multiforme"
    assert detail.synonyms == ("GBM",)
    ancestors = source.ancestors("4001458")
    assert [(a.code, a.name) for a in ancestors] == [("444", "Malignant neoplasm")]


def test_omop_concept_source_unknown_code() -> None:
    assert OmopConceptSource(_omop_conn(), "v").name_synonyms("999") is None


def test_command_is_registered() -> None:
    result = CliRunner().invoke(cli, ["materialize-target-graph", "--help"])
    assert result.exit_code == 0
    assert "materialize" in result.output.lower()
