"""Done-when item 3 on the neutral fixture: the showcase build command, driven by a profile.

One build feeds every test; each test names one clause of the report.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from neo4j import GraphDatabase

from sema.models.config import Neo4jConfig
from showcase.open_data_load.build import main
from tests.support.connector_fixture import (
    CATALOG,
    EXPECTED_DETECTED_JOINS,
    MAIN_SCHEMA,
    FixtureJoin,
    fake_model_environment,
    write_fixture_database,
)

pytestmark = pytest.mark.integration


def _wipe(driver: Any) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    directory = tmp_path_factory.mktemp("profile_build")
    database = write_fixture_database(directory)
    profile = directory / "fixture.yaml"
    profile.write_text(yaml.safe_dump(
        {"build": {"database": str(database), "catalog": CATALOG, "schema": MAIN_SCHEMA}},
    ))
    neo4j = Neo4jConfig()
    driver = GraphDatabase.driver(neo4j.uri, auth=(neo4j.user, neo4j.password.get_secret_value()))
    _wipe(driver)
    stdout = io.StringIO()
    try:
        with fake_model_environment(), contextlib.redirect_stdout(stdout):
            exit_code = main(["--profile", str(profile), "--skip-embeddings"])
        assert exit_code == 0
        yield json.loads(stdout.getvalue().strip().splitlines()[-1])
    finally:
        _wipe(driver)
        driver.close()


def _predicate(report: dict[str, Any], join: FixtureJoin) -> dict[str, Any] | None:
    for found in report["joins"]:
        for predicate in found["predicates"]:
            if (predicate["referenced_table"], predicate["referenced_column"],
                    predicate["referencing_table"], predicate["referencing_column"]) == (
                    join.referenced_table, join.referenced_column,
                    join.referencing_table, join.referencing_column):
                return predicate
    return None


def test_the_build_finds_joins(report: dict[str, Any]) -> None:
    assert report["joins"], "the report lists no JoinPath"
    assert report["join_count"] == len(report["joins"])


def test_every_join_uses_both_its_tables_and_links_entities(report: dict[str, Any]) -> None:
    gaps = []
    for join in report["joins"]:
        tables = {p[side] for p in join["predicates"]
                  for side in ("referenced_table", "referencing_table")}  # fmt: skip
        if not tables <= set(join["uses_tables"]):
            gaps.append(f"{join['name']}: USES {sorted(join['uses_tables'])}, needs {tables}")
        if not join["from_entities"] or not join["to_entities"]:
            gaps.append(f"{join['name']}: entity links missing")
    assert not gaps, gaps


def test_every_stored_reference_parses(report: dict[str, Any]) -> None:
    for join in report["joins"]:
        assert join["refs"], f"{join['name']}: no stored reference"
        assert join["refs_parse"], f"{join['name']}: {join['unparsed_refs']}"


def test_the_recheck_confirms_each_expected_detected_join(report: dict[str, Any]) -> None:
    coverage = report["thresholds"]["coverage"]
    for join in EXPECTED_DETECTED_JOINS:
        predicate = _predicate(report, join)
        assert predicate is not None, f"no JoinPath predicate for {join}"
        recheck = predicate["recheck"]
        referenced = recheck["referenced"]
        assert referenced["distinct_non_null"] == referenced["non_null"] == referenced["rows"]
        assert referenced["rows"] > 0, join
        assert recheck["referencing_distinct_non_null"] > 0, join
        assert recheck["containment_ratio"] >= coverage, (join, recheck)


def test_the_detector_candidates_are_listed_with_counts(report: dict[str, Any]) -> None:
    candidates = report["candidates"]
    assert candidates
    assert report["candidate_count"] == len(candidates)
    for candidate in candidates:
        recheck = candidate["recheck"]
        assert recheck is not None, candidate
        assert recheck["referenced"]["rows"] > 0, candidate
        assert recheck["referenced"]["distinct_non_null"] >= 0, candidate
