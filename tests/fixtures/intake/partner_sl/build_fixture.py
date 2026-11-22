"""Write ``library.duckdb``, the A1 offline standard-library fixture.

Run: ``uv run python tests/fixtures/intake/partner_sl/build_fixture.py``.

Relation and column names are the verified ones from the concept-binding design
(section 5.2 and the section 10 live log); the twelve data element ids and labels
are the Race domain set that log records. Every other value is synthesized here:
permissible values, definitions, synonyms, crosswalk ids (except Race's recorded
Athena and SNOMED codes), relationships, bindings, and value mappings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

OUT = Path(__file__).with_name("library.duckdb")
SCHEMA = "standard_library"
DOMAIN = "PRTD2140001103"
RACE = "PRT4013886"
RACE_CATEGORY = "PRT4216292"
ETHNICITY = "PRT44803968"
ETHNICITY_CATEGORY = "PRTL2140007288"
ADMINISTRATIVE_GENDER = "PRTL2140000129"
CONCEPTS = (
    (ADMINISTRATIVE_GENDER, "Administrative gender"),
    ("PRT4179218", "Birth sex"),
    (ETHNICITY, "Ethnicity"),
    (ETHNICITY_CATEGORY, "Ethnicity category"),
    ("PRT4110772", "Gender identity"),
    ("PRTL2140007289", "Hispanic or Latino ethnicity group"),
    ("PRT4181605", "Preferred language"),
    (RACE, "Race"),
    (RACE_CATEGORY, "Race category"),
    ("PRT35917103", "Race, cancer registry"),
    ("PRT35918387", "Spanish/Hispanic Origin, cancer registry"),
    ("PRT4031696", "Spiritual or religious belief"),
)
RECORDED_CROSSWALK = {RACE: ("4013886", "SNOMED", "103579009")}
SYNTHETIC_VOCABULARY = "SYNVOCAB"
BLOCK_SIZE = 16384
TABLES = {
    "standard_library_data_element": (
        "domain_prt_id VARCHAR, data_element_prt_id VARCHAR,"
        " data_element_prt_preferred_label VARCHAR, data_element_title VARCHAR,"
        " data_element_definition VARCHAR, data_element_data_type_id VARCHAR,"
        " data_element_phi VARCHAR, data_element_synonym_name VARCHAR[],"
        " permissible_value_prt_id VARCHAR,"
        " permissible_value_prt_preferred_label VARCHAR,"
        " permissible_value_title VARCHAR, permissible_value_vocabulary_id VARCHAR,"
        " permissible_value_concept_code VARCHAR,"
        " permissible_value_synonym_name VARCHAR[]"
    ),
    "standard_library_concept": (
        "concept_prt_id VARCHAR, concept_athena_id VARCHAR, vocabulary_id VARCHAR,"
        " concept_code VARCHAR, definition VARCHAR"
    ),
    "standard_library_concept_relationship": (
        "concept_prt_id_1 VARCHAR, concept_athena_id_1 VARCHAR,"
        " concept_prt_id_2 VARCHAR, concept_athena_id_2 VARCHAR,"
        " relationship_id VARCHAR, is_hierarchical VARCHAR, defines_ancestry VARCHAR"
    ),
    "data_element_mappings": (
        "unified_column_data_element_prt_id VARCHAR,"
        " mapped_source_dataset_label VARCHAR, mapped_source_dataset_rid VARCHAR,"
        " mapped_source_columns VARCHAR, mapped_project_label VARCHAR,"
        " unified_dataset_location VARCHAR"
    ),
    "permissible_value_mappings_edited": (
        "data_element_prt_id VARCHAR, given_value VARCHAR,"
        " mapped_permissible_value_prt_ids VARCHAR[], mapping_status VARCHAR"
    ),
}

Row = tuple[Any, ...]


def value_codes(index: int) -> list[str]:
    """Two to four synthesized permissible values per concept."""
    return [f"SYNPV{index:02d}{k}" for k in range(1, 3 + index % 3)]


def concept_athena_id(index: int, concept_id: str) -> str:
    recorded = RECORDED_CROSSWALK.get(concept_id)
    return recorded[0] if recorded else str(9900000 + index)


def data_element_rows() -> list[Row]:
    rows: list[Row] = []
    for index, (concept_id, label) in enumerate(CONCEPTS):
        element = (
            DOMAIN,
            concept_id,
            label,
            f"Patient {label.lower()}",
            f"Synthesized definition of {label.lower()}.",
            "string",
            "false",
            [f"{label} (synthesized)"],
        )
        for k, code in enumerate(value_codes(index), start=1):
            value = (
                code,
                f"{label} option {k}",
                f"{label} option {k} title",
                SYNTHETIC_VOCABULARY,
                f"SYN-{index:02d}-{k}",
                [f"opt {k}"] if k % 2 else [],
            )
            rows.append(element + value)
    return rows


def concept_rows() -> list[Row]:
    rows: list[Row] = []
    for index, (concept_id, _) in enumerate(CONCEPTS):
        _, system, code = RECORDED_CROSSWALK.get(
            concept_id, ("", SYNTHETIC_VOCABULARY, f"SYN-C{index:02d}")
        )
        rows.append((concept_id, concept_athena_id(index, concept_id), system, code, None))
        for k, value in enumerate(value_codes(index), start=1):
            athena = str(9910000 + index * 10 + k) if k % 2 else None
            definition = f"Synthesized definition of option {k}." if k == 1 else None
            rows.append(
                (value, athena, SYNTHETIC_VOCABULARY, f"SYN-{index:02d}-{k}", definition)
            )
    return rows


def relationship_rows() -> list[Row]:
    athena = {cid: concept_athena_id(i, cid) for i, (cid, _) in enumerate(CONCEPTS)}
    return [
        (RACE, athena[RACE], None, "9990001", "Maps to", "0", "0"),
        (RACE_CATEGORY, athena[RACE_CATEGORY], RACE, athena[RACE], "Is a", "1", "1"),
        (ETHNICITY, athena[ETHNICITY], ETHNICITY_CATEGORY, athena[ETHNICITY_CATEGORY],
         "Maps to", "0", "0"),
    ]


def binding_rows() -> list[Row]:
    return [
        (RACE, "Synthetic patient dataset", "syn.dataset.0001",
         '["race_desc", "race_cd"]', "Synthetic project", None),
        (ETHNICITY, "Synthetic visit dataset", "syn.dataset.0002",
         '["ethnicity_desc"]', None, None),
    ]


def value_mapping_rows() -> list[Row]:
    ids = [concept_id for concept_id, _ in CONCEPTS]
    race = value_codes(ids.index(RACE))
    gender = value_codes(ids.index(ADMINISTRATIVE_GENDER))
    return [
        (RACE, "option one", [race[0]], "Reviewed"),
        (RACE, "opt 2", [race[1], race[0]], "Awaiting Review - LLM Prediction"),
        (ADMINISTRATIVE_GENDER, "free text", [gender[0]], "Unmapped"),
    ]


def build(path: Path = OUT) -> None:
    path.unlink(missing_ok=True)
    rows = {
        "standard_library_data_element": data_element_rows(),
        "standard_library_concept": concept_rows(),
        "standard_library_concept_relationship": relationship_rows(),
        "data_element_mappings": binding_rows(),
        "permissible_value_mappings_edited": value_mapping_rows(),
    }
    with duckdb.connect() as con:
        con.execute(f"ATTACH '{path}' AS fixture (BLOCK_SIZE {BLOCK_SIZE})")
        con.execute("USE fixture")
        con.execute(f"CREATE SCHEMA {SCHEMA}")
        for table, columns in TABLES.items():
            con.execute(f"CREATE TABLE {SCHEMA}.{table} ({columns})")
            marks = ", ".join("?" * columns.count(" VARCHAR"))
            con.executemany(f"INSERT INTO {SCHEMA}.{table} VALUES ({marks})", rows[table])
        con.execute("CHECKPOINT")
        con.execute("USE memory")
        con.execute("DETACH fixture")


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}")
