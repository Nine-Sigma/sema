"""Build step: each join the dictionary declares, accepted only when a count on the data holds it.

Runs after the coded columns and before the joins are materialized. Each declared join names two
entities; `bind_declared` binds them to loaded relations and checks the join on the data. An
accepted join becomes one `JOINS_TO` assertion in the shape a declared constraint takes.
"""
from __future__ import annotations

from collections.abc import Sequence

from sema.connectors.base import Connector
from sema.engine.declared_join_utils import DeclaredJoin, bind_declared
from sema.engine.identity_utils import Counts, Relation
from sema.engine.join_detector_utils import normalized_name
from sema.engine.warehouse_lookup import RelationEvidence
from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import fetch_columns_by_schema
from sema.models.assertions import Assertion
from sema.models.config import BuildConfig
from sema.models.extraction import ExtractedForeignKey
from sema.pipeline.dictionary_import_utils import read_dictionary
from sema.pipeline.normalizer import AssertionNormalizer


def run_declared_joins(
    loader: GraphLoader, connector: Connector, config: BuildConfig, schemas: list[str],
    run_id: str,
) -> dict[str, list[Assertion]]:
    """The accepted declared joins, keyed by the referencing relation's schema."""
    if not config.enable_fk_detection:
        return {}
    dictionary = read_dictionary(config.concept_spec, config.concept_staging)
    if dictionary is None:
        return {}
    columns: dict[Relation, list[str]] = {}
    for schema in schemas:
        for column in fetch_columns_by_schema(loader, schema):
            columns.setdefault((column.schema, column.table_name), []).append(column.name)
    counts = Counts(RelationEvidence(connector, config.catalog))
    bindings = bind_declared(dictionary.entities, dictionary.declared_joins, columns, counts)
    _, platform, workspace = connector.get_datasource_ref()
    normalizer = AssertionNormalizer(workspace, run_id, platform)
    joins: dict[str, list[Assertion]] = {}
    for join, referencing, referenced, _ in bindings.accepted:
        key = _foreign_key(config.catalog, join, (referencing, columns[referencing]),
                           (referenced, columns[referenced]))
        joins.setdefault(referencing[0], []).extend(
            a.model_copy(update=dict(source="dictionary_declared"))
            for a in normalizer.normalize_foreign_key(key)
        )
    return joins


def _foreign_key(
    catalog: str, join: DeclaredJoin, referencing: tuple[Relation, Sequence[str]],
    referenced: tuple[Relation, Sequence[str]],
) -> ExtractedForeignKey:
    """The join as a constraint, each column spelled as its relation spells it."""
    (child, child_names), (parent, parent_names) = referencing, referenced
    pairs = tuple(
        (_spelled(local, child_names), _spelled(remote, parent_names))
        for local, remote in join.pairs
    )
    return ExtractedForeignKey("dictionary_declared", catalog, child[0], child[1], catalog,
                               parent[0], parent[1], pairs)


def _spelled(name: str, names: Sequence[str]) -> str:
    """As `bind_declared` spells it: of two columns that normalize alike, the last."""
    return {normalized_name(n): n for n in names}[normalized_name(name)]
