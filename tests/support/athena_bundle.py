"""A minimal OMOP Athena vocabulary bundle: tab-delimited CSVs with a header row."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from sema.ingest.duckdb_staging import Staging

VOCABULARY_SCHEMA = "vocabulary_omop"


def write_vocab_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["\t".join(header)]
    lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_minimal_bundle(target: Path, *, with_synonyms: bool = False) -> None:
    target.mkdir(parents=True, exist_ok=True)
    write_vocab_csv(
        target / "CONCEPT.csv",
        ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id"],
        [["1", "Gender", "Meas", "HL7", "Class"], ["2", "Person", "Obs", "HL7", "Class"]],
    )
    write_vocab_csv(
        target / "CONCEPT_RELATIONSHIP.csv",
        ["concept_id_1", "concept_id_2", "relationship_id"],
        [["1", "2", "maps_to"]],
    )
    write_vocab_csv(
        target / "CONCEPT_ANCESTOR.csv",
        ["ancestor_concept_id", "descendant_concept_id", "min_levels", "max_levels"],
        [["1", "2", "1", "1"]],
    )
    write_vocab_csv(
        target / "VOCABULARY.csv",
        ["vocabulary_id", "vocabulary_name", "vocabulary_version"],
        [["HL7", "HL7 V3", "2024-01"]],
    )
    write_vocab_csv(
        target / "DOMAIN.csv", ["domain_id", "domain_name"], [["Meas", "Measurement"]]
    )
    if with_synonyms:
        write_vocab_csv(
            target / "CONCEPT_SYNONYM.csv",
            ["concept_id", "concept_synonym_name", "language_concept_id"],
            [["1", "Sex", "4180186"]],
        )


def export_schema(staging: Staging, schema: str = VOCABULARY_SCHEMA) -> str:
    """Every table of ``schema`` as typed header plus sorted rows, in name order."""
    tables = staging.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = ? ORDER BY 1",
        [schema],
    ).fetchall()
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for (table,) in tables:
        info = staging.describe(schema, table)
        writer.writerow([f"## {table}"])
        writer.writerow([f"{name}:{col.type}" for name, col in info.columns.items()])
        rows = staging.execute(f'SELECT * FROM "{schema}"."{table}" ORDER BY ALL')
        writer.writerows(rows.fetchall())
    return out.getvalue()
