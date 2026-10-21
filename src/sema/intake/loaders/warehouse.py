"""Register a warehouse schema as landed relations from one connector catalog read.

A warehouse source copies no rows: each object the connector reports becomes a
relation with its columns, types, and comments, and provenance
``locator_kind: relation`` names the catalog and schema. Row counts are not in the
catalog read, so each relation is counted with one aggregate through the profiling
tools, which log it to the run ledger like every other statement.
"""

from __future__ import annotations

from sema.binding.spec_sections import RelationRef
from sema.connectors.base import Connector
from sema.intake.landing import LandedRelation, LandResult, Provenance, utc_now
from sema.intake.ledger import RunLedger
from sema.intake.loaders.base import DEFAULT_POLICY_CLASS
from sema.intake.profile import EvidencePolicy, ProfileTools
from sema.intake.relations import RelationProvider
from sema.models.schema_metadata import ObjectMetadata, Unreadable

LOADER = "warehouse"
LOADER_VERSION = "1"
STAGE = "land"


class UnreadableSchemaError(ValueError):
    """The connector reported the schema as unreadable; Land stops with its reason."""


def land_warehouse(
    connector: Connector,
    *,
    catalog: str,
    schema: str,
    ledger: RunLedger,
    policy_class: str = DEFAULT_POLICY_CLASS,
) -> LandResult:
    metadata = connector.read_schema_metadata(catalog, schema)
    if isinstance(metadata, Unreadable):
        raise UnreadableSchemaError(
            f"warehouse schema {catalog}.{schema} is unreadable: {metadata.reason}"
        )
    namespace = f"{catalog}.{schema}"
    source_uri = connector.get_datasource_ref()[0]
    record = Provenance(
        source_uri=source_uri,
        loader=LOADER,
        loader_version=LOADER_VERSION,
        landed_at=utc_now(),
        policy_class=policy_class,
        locator_kind="relation",
        locator=namespace,
    )
    provider = RelationProvider(warehouse=connector.execute_query, staging=None)
    tools = ProfileTools(provider, EvidencePolicy(), ledger, stage=STAGE)
    return LandResult(
        medium="warehouse",
        source_identity=((source_uri, namespace),),
        relations=tuple(
            _relation(namespace, item, record, tools) for item in metadata.objects
        ),
        policy_class=policy_class,
    )


def _relation(
    namespace: str, item: ObjectMetadata, record: Provenance, tools: ProfileTools
) -> LandedRelation:
    ref = RelationRef(provider="warehouse", name=f"{namespace}.{item.name}")
    return LandedRelation(
        schema=namespace,
        name=item.name,
        columns=tuple((column.name, column.data_type) for column in item.columns),
        row_count=tools.count_rows(ref),
        provenance=record,
        comments=tuple(
            (column.name, column.comment) for column in item.columns if column.comment
        ),
    )
