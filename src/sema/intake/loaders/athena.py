"""Land an OMOP Athena vocabulary bundle into DuckDB staging.

Moved from the showcase with the loading SQL unchanged, so the staging tables
are byte-identical. Because of that contract the loader adds no row locator
column: each relation's provenance is ``locator_kind: relation`` with the
source file name as its locator. Allowlisted in the R29 guard because it names
the Athena file set.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from sema.binding.spec_sections import (
    ConceptRoles,
    ConceptsSpec,
    EndpointRole,
    RelationRef,
    RelationshipRoles,
    RelationshipsSpec,
    ValueSetRoles,
    ValueSetsSpec,
)
from sema.ingest.duckdb_staging import Staging
from sema.intake.landing import LandedRelation, LandResult, utc_now
from sema.intake.loaders.base import DEFAULT_POLICY_CLASS, ensure_schema, provenance
from sema.log import logger

LOADER = "athena"
LOADER_VERSION = "1"
VOCABULARY_SCHEMA = "vocabulary_omop"

REQUIRED_VOCAB_FILES: tuple[str, ...] = (
    "CONCEPT.csv",
    "CONCEPT_RELATIONSHIP.csv",
    "CONCEPT_ANCESTOR.csv",
    "VOCABULARY.csv",
    "DOMAIN.csv",
)

OPTIONAL_VOCAB_FILES: tuple[str, ...] = (
    "CONCEPT_SYNONYM.csv",
    "CONCEPT_CLASS.csv",
    "RELATIONSHIP.csv",
    "DRUG_STRENGTH.csv",
)


def vocab_table(filename: str) -> str:
    return Path(filename).stem.lower()


REQUIRED_TABLES = tuple(vocab_table(filename) for filename in REQUIRED_VOCAB_FILES)


def fixed_sections(
    relations: Mapping[str, RelationRef], identity_system: str
) -> dict[str, BaseModel]:
    """Each vocabulary is a concept, its concepts the values, related by their numeric ids."""
    def endpoint(column: str) -> EndpointRole:
        return EndpointRole(system=identity_system, id=column)

    return {
        "concepts": ConceptsSpec(
            relation=relations["vocabulary"],
            group_by="vocabulary_id",
            roles=ConceptRoles(concept_id="vocabulary_id", name="vocabulary_name"),
        ),
        "value_sets": ValueSetsSpec(
            relation=relations["concept"],
            roles=ValueSetRoles(
                concept_id="vocabulary_id", code="concept_id", display="concept_name"
            ),
        ),
        "relationships": RelationshipsSpec(
            relation=relations["concept_relationship"],
            roles=RelationshipRoles.model_validate({
                "from": endpoint("concept_id_1"),
                "to": endpoint("concept_id_2"),
                "type": "relationship_id",
            }),
        ),
    }


def land_athena_bundle(
    bundle_dir: Path | str, staging: Staging, *, schema: str = VOCABULARY_SCHEMA
) -> LandResult:
    vocab_dir = Path(bundle_dir).expanduser()
    _assert_required_vocab_files(vocab_dir)
    ensure_schema(staging, schema)
    landed_at = utc_now()
    relations: list[LandedRelation] = []
    for filename in REQUIRED_VOCAB_FILES + OPTIONAL_VOCAB_FILES:
        path = vocab_dir / filename
        if path.exists():
            table = _load_vocab_csv_into_duckdb(path, staging, schema)
            relations.append(_landed(staging, schema, table, path, landed_at))
    logger.info("Vocabulary ingestion complete from {}", vocab_dir)
    identity = (("bundle", str(vocab_dir.resolve())), *_vocabulary_versions(staging, schema))
    return LandResult(
        medium="athena",
        source_identity=identity,
        relations=tuple(sorted(relations, key=lambda r: r.name)),
        policy_class=DEFAULT_POLICY_CLASS,
    )


def _assert_required_vocab_files(vocab_dir: Path) -> None:
    if not vocab_dir.exists():
        raise FileNotFoundError(f"Vocabulary path does not exist: {vocab_dir}")
    missing = [f for f in REQUIRED_VOCAB_FILES if not (vocab_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required OMOP vocabulary file(s) in {vocab_dir}: {', '.join(missing)}"
        )


def _load_vocab_csv_into_duckdb(path: Path, staging: Staging, schema: str) -> str:
    table_name = vocab_table(path.name)
    logger.info("Loading vocabulary CSV {} into {}.{}", path.name, schema, table_name)
    staging.drop_table(schema, table_name)
    sanitized = str(path).replace("'", "''")
    sql = (
        f'CREATE TABLE "{schema}"."{table_name}" AS '
        f"SELECT * FROM read_csv_auto('{sanitized}', "
        "delim='\\t', header=true, all_varchar=true, strict_mode=false)"
    )
    staging.execute(sql)
    return table_name


def _landed(
    staging: Staging, schema: str, table: str, path: Path, landed_at: str
) -> LandedRelation:
    info = staging.describe(schema, table)
    count = staging.execute(f'SELECT count(*) FROM "{schema}"."{table}"').fetchone()
    return LandedRelation(
        schema=schema,
        name=table,
        columns=tuple((name, column.type) for name, column in info.columns.items()),
        row_count=int(count[0]) if count else 0,
        provenance=provenance(
            path, LOADER, LOADER_VERSION, "relation", locator=path.name, landed_at=landed_at
        ),
    )


def _vocabulary_versions(staging: Staging, schema: str) -> list[tuple[str, str]]:
    columns = staging.describe(schema, "vocabulary").columns
    version = "vocabulary_version" if "vocabulary_version" in columns else "NULL"
    rows = staging.execute(
        f"SELECT vocabulary_id, coalesce({version}, '') "
        f'FROM "{schema}"."vocabulary" ORDER BY 1'
    ).fetchall()
    return [(str(vocabulary), str(release)) for vocabulary, release in rows]
