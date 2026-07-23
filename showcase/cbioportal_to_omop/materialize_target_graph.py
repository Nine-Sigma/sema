"""Showcase wiring for ``sema materialize-target-graph`` (Slice-2, S2-09).

Thin adapter that feeds the *generic* target-retrieval build
(:func:`sema.graph.target_graph_build.build_target_retrieval_graph`) with the
OMOP manifest + ``omop_stage_a`` physical schema, the concept codes and
source→concept crosswalks read from the DuckDB value-mapping store, an OMOP
concept source (names/synonyms/ancestors from the OMOP vocabulary), and a live
Databricks catalog reader. All OMOP specifics live here, behind the R29
boundary; core stays agnostic.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click

from sema.graph.concept_source import ConceptAncestor, ConceptDetail
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import PhysicalColumn
from sema.graph.target_graph_build import (
    ConceptFieldSpec,
    ValueBridgeSpec,
    build_target_retrieval_graph,
)
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
)
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.normalizer import TargetModelNormalizer

_CONCEPT_VOCABULARY = "OMOP"
_CONCEPT_FIELD = "condition_concept_id"
_CONDITION_ENTITY = "omop.condition_occurrence"
_PERSON_ENTITY = "omop.person"


def value_mappings_to_specs(
    mappings: Sequence[ValueMapping],
    *,
    entity_qualified_name: str = _CONDITION_ENTITY,
    property_name: str = _CONCEPT_FIELD,
) -> tuple[list[ConceptFieldSpec], list[ValueBridgeSpec]]:
    """Split resolved value-mapping rows into concept-field + bridge specs.

    NO_MAP rows (``concept_id is None``) are excluded — an unmapped value is
    absent from the index, never a null placeholder (Slice-2 principle).
    """
    resolved = [
        m for m in mappings
        if m.resolution_status is not ResolutionStatus.NO_MAP
        and m.concept_id is not None
    ]
    codes = tuple(dict.fromkeys(str(m.concept_id) for m in resolved))
    bridges = [
        ValueBridgeSpec(
            source_vocabulary=m.source_vocabulary,
            source_code=m.normalized_source_value,
            concept_code=str(m.concept_id),
        )
        for m in resolved
    ]
    field = ConceptFieldSpec(
        entity_qualified_name=entity_qualified_name,
        property_name=property_name,
        concept_codes=codes,
    )
    return [field], bridges


class OmopConceptSource:
    """`ConceptSource` over the OMOP vocabulary (concept + concept_ancestor).

    ``code`` is a stringified OMOP ``concept_id``. Works against any DB-API
    connection holding the OMOP vocabulary (DuckDB or a Databricks cursor).
    """

    def __init__(self, connection: Any, vocab_schema: str) -> None:
        self._conn = connection
        self._schema = vocab_schema

    def name_synonyms(self, code: str) -> ConceptDetail | None:
        rows = self._query(
            f"SELECT concept_name FROM {self._schema}.concept "
            "WHERE concept_id = ?",
            [int(code)],
        )
        if not rows:
            return None
        synonyms = self._query(
            f"SELECT concept_synonym_name FROM {self._schema}.concept_synonym "
            "WHERE concept_id = ?",
            [int(code)],
        )
        return ConceptDetail(
            code=code,
            name=str(rows[0][0]),
            synonyms=tuple(str(r[0]) for r in synonyms),
        )

    def ancestors(self, code: str) -> list[ConceptAncestor]:
        rows = self._query(
            "SELECT c.concept_id, c.concept_name "
            f"FROM {self._schema}.concept_ancestor a "
            f"JOIN {self._schema}.concept c "
            "ON c.concept_id = a.ancestor_concept_id "
            "WHERE a.descendant_concept_id = ? "
            "AND a.ancestor_concept_id <> a.descendant_concept_id",
            [int(code)],
        )
        return [ConceptAncestor(code=str(r[0]), name=str(r[1])) for r in rows]

    def _query(self, sql: str, params: list[Any]) -> list[tuple[Any, ...]]:
        cur = self._conn.execute(sql, params)
        return list(cur.fetchall())


class DatabricksCatalogSource:
    """`PhysicalCatalogSource` over a warehouse ``information_schema``."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def columns(
        self, *, catalog: str, schema: str, table: str
    ) -> list[PhysicalColumn]:
        self._cursor.execute(
            f"SELECT column_name, data_type, is_nullable "
            f"FROM {catalog}.information_schema.columns "
            "WHERE table_schema = ? AND table_name = ? "
            "ORDER BY ordinal_position",
            [schema, table],
        )
        return [
            PhysicalColumn(
                name=str(r[0]), data_type=str(r[1]),
                nullable=str(r[2]).upper() == "YES",
            )
            for r in self._cursor.fetchall()
        ]


def _neo4j_driver() -> Any:
    import neo4j

    return neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "graphrag"),
        ),
    )


@click.command("materialize-target-graph")
@click.option("--manifest", "manifest_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--catalog", required=True, help="Physical catalog of omop_stage_a.")
@click.option("--schema", "schema", default="omop_stage_a", show_default=True)
@click.option("--study-schema", "study_schema", required=True, help="Source study schema (bridge scope).")
@click.option("--duckdb", "duckdb_path", type=click.Path(path_type=Path), required=True, help="Value-mapping + OMOP vocab DuckDB.")
@click.option("--vocab-schema", default="omop_vocab", show_default=True)
def materialize_target_graph_cmd(
    manifest_path: Path,
    catalog: str,
    schema: str,
    study_schema: str,
    duckdb_path: Path,
    vocab_schema: str,
) -> None:
    """Materialize the OMOP target semantic layer into Neo4j for GraphRAG."""
    import duckdb

    from sema.cli_fit_utils import open_databricks_cursor
    from sema.models.config import DatabricksConfig

    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(manifest_path))
    binding = PhysicalSchemaBinding(
        catalog=catalog, schema=schema,
        entity_tables=(
            EntityTableBinding(_PERSON_ENTITY, "person"),
            EntityTableBinding(_CONDITION_ENTITY, "condition_occurrence"),
        ),
    )
    conn = duckdb.connect(str(duckdb_path))
    mappings = [
        m for m in _read_value_mappings(conn)
        if m.normalized_source_value  # scope: this store's resolved rows
    ]
    concept_fields, value_bridges = value_mappings_to_specs(mappings)

    cursor = open_databricks_cursor(DatabricksConfig(), catalog=catalog)
    driver = _neo4j_driver()
    try:
        build_target_retrieval_graph(
            GraphLoader(driver), normalized, binding,
            DatabricksCatalogSource(cursor),
            concepts=OmopConceptSource(conn, vocab_schema),
            concept_vocabulary=_CONCEPT_VOCABULARY,
            concept_fields=concept_fields,
            value_bridges=value_bridges,
            bridge_source_schema=study_schema,
        )
    finally:
        driver.close()
    click.echo(
        f"Materialized target graph: {len(concept_fields)} concept field(s), "
        f"{len(value_bridges)} bridge edge(s)."
    )


def _read_value_mappings(conn: Any) -> list[ValueMapping]:
    from sema.resolve.value_mapping_store import ValueMappingStore

    return ValueMappingStore(conn).read_all()


__all__ = [
    "DatabricksCatalogSource",
    "OmopConceptSource",
    "materialize_target_graph_cmd",
    "value_mappings_to_specs",
]
