"""The coded-column fact kinds: cells from the build graph's bindings, and their data rechecks.

A binding is read as the build graph holds it (`BindingRead`): the column, the code source
its containment assertion names (a code relation's key column ref, or a dictionary
vocabulary), its verification, its verified label column, whether its members carry the
dictionary's labels, its listed row share and its members' observed rows. The reads here are
built from the coded fixture's claims, so the expected cells are the claims themselves; the
rechecks run through the DuckDB connector on the coded fixture's database.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.models.config import DuckDBConfig
from sema.models.physical_key import CanonicalRef, PhysicalKey
from tests.support.connector_fixture_codes import (
    CODE_LISTS,
    CODED_TABLES,
    CODES_CATALOG,
    CODES_SCHEMA,
    LOADED_LIST_TABLES,
    ROWS,
    write_coded_database,
)
from tests.support.connector_fixture_scorecard import (
    BINDING_CLAIMS,
    COVERAGE_CLAIMS,
    Key,
    card_rows,
    coded_claims,
    listed_codes,
    present_values,
)

pytestmark = pytest.mark.unit

CODED_CLAIMS = coded_claims()
LABEL_CLAIMS = CODED_CLAIMS["label_source"]
DICTIONARY_ONLY = "dictionary_only"
VERIFIED, UNVERIFIED = "verified", "unverified"
# Decoys the build leaves unverified: data-only containment, containment by chance, a tie.
UNVERIFIED_DECOYS = {("panel", "shift"): "shift_slot", ("panel", "crew_size"): "bay",
                     ("panel", "rack_no"): "rack"}
LABELLED_COLUMNS = sum(isinstance(value, list) for value in LABEL_CLAIMS.values())


def _scorecard() -> Any:
    return importlib.import_module("sema.eval.discovery_scorecard")


def _key_column(relation: str) -> str:
    (table,) = (t for t in CODED_TABLES if t.name == relation)
    return table.column_names[0]


def _code_source(code_list: str) -> str:
    if code_list in CODE_LISTS:
        return code_list
    key = PhysicalKey("ds", CODES_CATALOG, CODES_SCHEMA, code_list, _key_column(code_list))
    return CanonicalRef.build("duckdb", key)


def _read(key: Key, code_list: str, verification: str = VERIFIED,
          label: object = None) -> Any:
    relation, column = key
    present = present_values(key)
    members = {**dict.fromkeys(listed_codes(code_list), 0), **Counter(present)}
    return _scorecard().BindingRead(
        schema=CODES_SCHEMA, relation=relation, column=column,
        code_source=_code_source(code_list), verification=verification,
        label=(CODES_SCHEMA, *label) if isinstance(label, list) else None,
        dictionary_label=label == DICTIONARY_ONLY,
        listed_row_share=sum(v in listed_codes(code_list) for v in present) / len(present),
        member_rows=members,
    )


def _claimed_reads() -> list[Any]:
    return [_read(key, name, label=LABEL_CLAIMS.get(key)) for key, name in BINDING_CLAIMS.items()]


def _decoy_reads() -> list[Any]:
    return [_read(key, name, UNVERIFIED) for key, name in UNVERIFIED_DECOYS.items()]


def test_the_fixture_reading_behind_the_claims() -> None:
    assert (len(BINDING_CLAIMS), LABELLED_COLUMNS, len(LABEL_CLAIMS)) == (10, 4, 8)
    assert COVERAGE_CLAIMS[("panel", "coat_kind")] < 1.0
    assert not set(UNVERIFIED_DECOYS) & set(BINDING_CLAIMS)


def test_each_coded_kinds_cells_equal_the_fixture_claims() -> None:
    produced = _scorecard().coded_facts(_claimed_reads())

    assert produced == CODED_CLAIMS


def test_an_unverified_binding_is_no_produced_fact() -> None:
    produced = _scorecard().coded_facts(_claimed_reads() + _decoy_reads())

    assert produced == CODED_CLAIMS
    assert _scorecard().coded_facts(_decoy_reads()) == {kind: {} for kind in CODED_CLAIMS}


def test_a_dictionary_code_list_is_named_in_lowercase() -> None:
    read = _read(("panel", "seal_kind"), "seal_kinds", label=DICTIONARY_ONLY)

    produced = _scorecard().coded_facts([read._replace(code_source="Seal_Kinds")])

    assert produced["column_to_concept_binding"] == {("panel", "seal_kind"): "seal_kinds"}


def test_a_planted_wrong_label_relation_counts_as_wrong() -> None:
    scorecard = _scorecard()
    reads = [
        read._replace(label=(CODES_SCHEMA, "glaze", "glaze_title"))
        if (read.relation, read.column) == ("panel", "finish_code") else read
        for read in _claimed_reads()
    ]
    produced = scorecard.coded_facts(reads)
    card = scorecard.render(scorecard.score_discovery(produced, CODED_CLAIMS, {}))

    row = card_rows(card)["label_source"]
    assert (row["equal"], row["wrong"]) == (str(len(LABEL_CLAIMS) - 1), "1")
    assert (row["missing"], row["extra"]) == ("0", "0")


def test_a_kind_no_shape_states_prints_not_produced() -> None:
    scorecard = _scorecard()
    produced = scorecard.coded_facts(_claimed_reads())
    rows = card_rows(scorecard.render(scorecard.score_discovery(produced, {}, {})))

    for kind in ("candidate_key", "hierarchy", "time_roles", "measure_semantics"):
        assert rows[kind]["produced"] == "not produced"
    for kind, claims in CODED_CLAIMS.items():
        assert rows[kind]["produced"] == str(len(claims))


@pytest.fixture
def connector(tmp_path: Path) -> Iterator[DuckDBConnector]:
    opened = DuckDBConnector(DuckDBConfig(path=str(write_coded_database(tmp_path))))
    yield opened
    opened.close()


def _rechecked(connector: DuckDBConnector, reads: list[Any]) -> dict[str, int]:
    return dict(_scorecard().recheck_coded(connector, CODES_CATALOG, reads, CODE_LISTS))


def test_every_claimed_coded_fact_rechecks_on_the_data(connector: DuckDBConnector) -> None:
    assert _rechecked(connector, _claimed_reads()) == {
        "column_to_concept_binding": len(BINDING_CLAIMS), "label_source": LABELLED_COLUMNS,
        "mapping_coverage": len(COVERAGE_CLAIMS),
        "observed_values": len(CODED_CLAIMS["observed_values"]),
    }


def test_a_binding_the_data_refutes_does_not_recheck(connector: DuckDBConnector) -> None:
    refuted = _read(("fitting", "grit_code"), "grit")._replace(listed_row_share=1.0)

    rechecked = _rechecked(connector, [refuted])

    assert (rechecked["column_to_concept_binding"], rechecked["mapping_coverage"]) == (0, 0)


def test_a_label_the_code_does_not_determine_does_not_recheck(
    connector: DuckDBConnector,
) -> None:
    undetermined = _read(("fitting", "tone_code"), "tone", label=["fitting", "sheen_name"])

    assert _rechecked(connector, [undetermined])["label_source"] == 0


def test_a_miscounted_observed_value_does_not_recheck(connector: DuckDBConnector) -> None:
    read = _read(("panel", "seal_kind"), "seal_kinds", label=DICTIONARY_ONLY)
    miscounted = read._replace(member_rows={code: rows + 1 for code, rows in
                                            read.member_rows.items() if rows})

    assert _rechecked(connector, [miscounted])["observed_values"] == 0


@pytest.fixture
def loaded_connector(tmp_path: Path) -> Iterator[DuckDBConnector]:
    path = write_coded_database(tmp_path, (*CODED_TABLES, *LOADED_LIST_TABLES))
    opened = DuckDBConnector(DuckDBConfig(path=str(path)))
    yield opened
    opened.close()


def _loaded_bindings(connector: DuckDBConnector) -> dict[Key, str]:
    scorecard = _scorecard()
    reads = _claimed_reads() + _decoy_reads()
    loaded = scorecard.loaded_code_lists(connector, CODES_CATALOG, [CODES_SCHEMA], reads,
                                         CODE_LISTS)
    return dict(scorecard.coded_facts(reads, loaded)["column_to_concept_binding"])


def test_a_dictionary_code_list_loaded_as_a_relation_is_named_by_that_relation(
    loaded_connector: DuckDBConnector,
) -> None:
    assert {str(row[0]) for row in ROWS["seal_kind_codes"]} == listed_codes("seal_kinds")

    bindings = _loaded_bindings(loaded_connector)

    assert bindings == {
        key: "seal_kind_codes" if name == "seal_kinds" else name
        for key, name in BINDING_CLAIMS.items()
    }


def test_a_key_column_holding_more_than_the_codes_or_repeating_them_is_not_the_code_list(
    loaded_connector: DuckDBConnector,
) -> None:
    keys, repeats = ([str(row[i]) for row in ROWS["bevel_superset"]] for i in (0, 1))
    assert set(present_values(("panel", "bevel_kind"))) < set(keys)
    assert set(repeats) == listed_codes("bevel_kinds") and len(repeats) > len(set(repeats))

    assert _loaded_bindings(loaded_connector)[("panel", "bevel_kind")] == "bevel_kinds"


def test_a_code_list_loaded_nowhere_keeps_its_vocabulary_name(
    loaded_connector: DuckDBConnector,
) -> None:
    assert _loaded_bindings(loaded_connector)[("panel", "coat_kind")] == "coat_kinds"
