"""Synthetic concept library in DuckDB, shaped like the design 5.2 relations.

Four concepts: Race (three values, each with an omop crosswalk) and three
siblings with two values each. The data-element relation holds one row per
(concept, permissible value), so an importer that does not group by concept
would emit one property per value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import duckdb
import yaml

from sema.intake.relations import RelationProvider

FIXTURES_DIR = Path(__file__).parent
SPEC_PATH = FIXTURES_DIR / "concept_source.yaml"
DATASET_MAP_PATH = FIXTURES_DIR / "dataset_map.yaml"

RACE_ID = "SYN_RACE"
RACE_REF = "syn:SYN_RACE"
ETHNICITY_REF = "syn:SYN_ETHNICITY"
CONCEPT_IDS = ("SYN_RACE", "SYN_ETHNICITY", "SYN_SEX", "SYN_VITAL_STATUS")
RACE_VALUE_CODES = ("SYN_RACE_WHITE", "SYN_RACE_BLACK", "SYN_RACE_ASIAN")
RESOLVED_PHYSICAL = "syn_cat.syn_schema.patient"
MULTI_CODES = ["SYN_RACE_WHITE", "SYN_RACE_BLACK"]

QueryFn = Callable[[str], list[tuple[Any, ...]]]

_CONCEPTS = (
    ("SYN_RACE", "Race", "Patient race", "Self-identified ancestral grouping."),
    ("SYN_ETHNICITY", "Ethnicity", "Patient ethnicity", "Hispanic or Latino origin."),
    ("SYN_SEX", "Sex", "Patient sex", "Biological sex recorded at registration."),
    (
        "SYN_VITAL_STATUS",
        "Vital status",
        "Vital status",
        "Whether the patient is alive.",
    ),
)

_VALUES = (
    ("SYN_RACE", "SYN_RACE_WHITE", "White", ["Caucasian"]),
    ("SYN_RACE", "SYN_RACE_BLACK", "Black or African American", ["Black"]),
    ("SYN_RACE", "SYN_RACE_ASIAN", "Asian", []),
    ("SYN_ETHNICITY", "SYN_ETH_HISPANIC", "Hispanic or Latino", ["Latino"]),
    ("SYN_ETHNICITY", "SYN_ETH_NOT_HISPANIC", "Not Hispanic or Latino", []),
    ("SYN_SEX", "SYN_SEX_FEMALE", "Female", ["F"]),
    ("SYN_SEX", "SYN_SEX_MALE", "Male", ["M"]),
    ("SYN_VITAL_STATUS", "SYN_VS_ALIVE", "Alive", ["Living"]),
    ("SYN_VITAL_STATUS", "SYN_VS_DECEASED", "Deceased", ["Dead"]),
)

_CONCEPT_DETAILS = (
    ("SYN_RACE", "900001", "SYNVOCAB", "C0001", "Race concept."),
    ("SYN_ETHNICITY", "900002", "SYNVOCAB", "C0002", "Ethnicity concept."),
    ("SYN_SEX", "900003", "SYNVOCAB", "C0003", "Sex concept."),
    ("SYN_VITAL_STATUS", "900004", "SYNVOCAB", "C0004", "Vital status concept."),
    ("SYN_RACE_WHITE", "900101", "SYNVOCAB", "C0101", "Origins in Europe."),
    ("SYN_RACE_BLACK", "900102", "SYNVOCAB", "C0102", "Origins in Africa."),
    ("SYN_RACE_ASIAN", "900103", "SYNVOCAB", "C0103", "Origins in Asia."),
)

# (syn id 1, athena id 1, syn id 2, athena id 2, relationship, hierarchical)
# The library stores the hierarchical flag as the strings '1' / '0'.
_RELATIONSHIPS = (
    ("SYN_RACE", "900001", None, "900999", "Maps to", "0"),
    ("SYN_ETHNICITY", "900002", "SYN_RACE", "900001", "Is a", "1"),
    ("SYN_OTHER", "900500", None, "900501", "Maps to", "0"),
)

_COLUMN_BINDINGS = (
    (
        "SYN_RACE",
        "Synthetic Patients",
        "ds-0001",
        '["race_label", "race_cd"]',
        "Synthetic Project",
    ),
    ("SYN_ETHNICITY", "Unmapped Visits", "ds-0002", '["ethnicity"]', None),
)

_VALUE_MAPPINGS = (
    ("SYN_RACE", "white", "SYN_RACE_WHITE", "Reviewed"),
    ("SYN_RACE", "caucasian", "SYN_RACE_WHITE", "Awaiting Review - LLM Prediction"),
    ("SYN_RACE", "asian american", "SYN_RACE_ASIAN", "Pending Ontologist Review"),
)


def build_library() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA lib")
    _create_data_element(con)
    _create_concept(con)
    _create_relationships(con)
    _create_authored(con)
    return con


def query_fn(con: duckdb.DuckDBPyConnection) -> QueryFn:
    # Same shape as Connector.execute_query: positional tuples, no column names.
    def run(sql: str) -> list[tuple[Any, ...]]:
        return con.execute(sql).fetchall()

    return run


def provider_for(
    con: duckdb.DuckDBPyConnection, query: QueryFn | None = None
) -> RelationProvider:
    """Warehouse-only provider over the fixture, optionally through ``query``."""
    return RelationProvider(warehouse=query or query_fn(con), staging=None)


def with_array_codes(con: duckdb.DuckDBPyConnection) -> None:
    """Mapping codes become ARRAY<string>, plus one row carrying two codes."""
    con.execute(
        "ALTER TABLE lib.value_mappings ALTER mapped_value_id"
        " SET DATA TYPE VARCHAR[] USING [mapped_value_id]"
    )
    con.execute(
        "INSERT INTO lib.value_mappings VALUES (?,?,?,?)",
        ["SYN_RACE", "mixed", MULTI_CODES, "Reviewed"],
    )


def load_dataset_map() -> dict[str, str]:
    loaded: dict[str, str] = yaml.safe_load(DATASET_MAP_PATH.read_text())
    return loaded


def _create_data_element(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE lib.data_element ("
        " data_element_id VARCHAR, data_element_label VARCHAR,"
        " data_element_title VARCHAR, data_element_definition VARCHAR,"
        " data_element_data_type VARCHAR, data_element_synonyms VARCHAR[],"
        " value_id VARCHAR, value_label VARCHAR, value_title VARCHAR,"
        " value_vocabulary_id VARCHAR, value_concept_code VARCHAR,"
        " value_synonyms VARCHAR[])"
    )
    concepts = {c[0]: c for c in _CONCEPTS}
    for concept_id, code, label, synonyms in _VALUES:
        _, name, title, definition = concepts[concept_id]
        con.execute(
            "INSERT INTO lib.data_element VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                concept_id,
                name,
                title,
                definition,
                "string",
                [title],
                code,
                label,
                label,
                "SYNVOCAB",
                f"V_{code}",
                synonyms,
            ],
        )


def _create_concept(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE lib.concept (concept_id VARCHAR, athena_id VARCHAR,"
        " vocabulary_id VARCHAR, concept_code VARCHAR, definition VARCHAR)"
    )
    con.executemany("INSERT INTO lib.concept VALUES (?,?,?,?,?)", _CONCEPT_DETAILS)


def _create_relationships(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE lib.concept_relationship (concept_id_1 VARCHAR,"
        " athena_id_1 VARCHAR, concept_id_2 VARCHAR, athena_id_2 VARCHAR,"
        " relationship_id VARCHAR, is_hierarchical VARCHAR)"
    )
    con.executemany(
        "INSERT INTO lib.concept_relationship VALUES (?,?,?,?,?,?)", _RELATIONSHIPS
    )


def _create_authored(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE lib.column_bindings (data_element_id VARCHAR,"
        " dataset_label VARCHAR, dataset_id VARCHAR, source_columns VARCHAR,"
        " project_label VARCHAR)"
    )
    con.executemany(
        "INSERT INTO lib.column_bindings VALUES (?,?,?,?,?)", _COLUMN_BINDINGS
    )
    con.execute(
        "CREATE TABLE lib.value_mappings (data_element_id VARCHAR,"
        " given_value VARCHAR, mapped_value_id VARCHAR, mapping_status VARCHAR)"
    )
    con.executemany("INSERT INTO lib.value_mappings VALUES (?,?,?,?)", _VALUE_MAPPINGS)
