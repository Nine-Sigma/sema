"""Read-only, logged profiling tools over landed relations (design Decisions 2 and 18).

Describes and structural checks return aggregates only. Values come back only through
``sample_values`` when the evidence policy allows them, and schema text only through
``read_schema_text`` for a relation the structure output binds as a dictionary or header.
Every statement is written to the run ledger before it is issued.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sema.binding.spec_sections import Provider, RelationRef
from sema.intake.ledger import RunLedger
from sema.intake.profile_utils import (
    ExternalRefCheck,
    HierarchyCheck,
    JoinCheck,
    NeedsValues,
    RelationListing,
    RelationProfile,
    ValueSample,
    aggregate_sql,
    catalog_columns,
    column_profiles,
    distinct_count_sql,
    hierarchy_sql,
    join_check,
    join_sql,
    sample_sql,
    staging_columns_sql,
    staging_tables_sql,
    text_sql,
    warehouse_tables_sql,
)
from sema.intake.relations import RelationProvider
from sema.models.config import BuildConfig

__all__ = [
    "EvidencePolicy",
    "ExternalRefCheck",
    "HierarchyCheck",
    "JoinCheck",
    "NeedsValues",
    "ProfileRefusedError",
    "ProfileTools",
    "RelationListing",
    "RelationProfile",
    "ValueSample",
]

EvidenceTier = Literal["metadata_only", "bounded_values", "full"]
ColumnKey = tuple[str, str]
SAMPLE_CAP = 20
DEFAULT_MAX_DISTINCT_VALUES_READ: int = BuildConfig.model_fields["max_distinct_values_read"].default


class ProfileRefusedError(ValueError):
    """A request the policy, the providers, or the structure output does not allow."""


class EvidencePolicy(BaseModel):
    """The run's evidence tier; column keys are ``(RelationRef.name, column)``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: EvidenceTier
    allowlist: frozenset[ColumnKey] = frozenset()
    max_distinct_values_read: int = Field(default=DEFAULT_MAX_DISTINCT_VALUES_READ, ge=0)
    sensitive: frozenset[ColumnKey] = frozenset()


class SchemaRelations(Protocol):
    """What ``read_schema_text`` needs from a structure output."""

    @property
    def schema_relations(self) -> Collection[str]: ...


class CodeLookup(Protocol):
    def concept_by_code(self, vocabulary: str, code: str) -> object | None: ...


class ProfileTools:
    def __init__(
        self,
        provider: RelationProvider,
        policy: EvidencePolicy,
        ledger: RunLedger,
        *,
        stage: str,
        structure: SchemaRelations | None = None,
        vocabulary: CodeLookup | None = None,
        loaded_systems: frozenset[str] = frozenset(),
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.ledger = ledger
        self.stage = stage
        self.structure = structure
        self.vocabulary = vocabulary
        self.loaded_systems = loaded_systems

    def list_relations(self, provider_name: Provider, namespace: str) -> list[RelationListing]:
        catalog_ref = RelationRef(provider=provider_name, name=namespace)
        listed = self._run(catalog_ref, self._tables_sql(catalog_ref))
        listings = []
        for table, comment in ((str(row[0]), row[1]) for row in listed):
            ref = RelationRef(provider=provider_name, name=f"{namespace}.{table}")
            count = self._run(ref, f"SELECT COUNT(*) FROM {self.provider.relation_sql(ref)}")
            listings.append(RelationListing(ref.name, int(count[0][0]), comment or None))
        return listings

    def describe_relation(self, ref: RelationRef) -> RelationProfile:
        catalog = catalog_columns(self._run(ref, self._columns_sql(ref)))
        quoted = [self.provider.quote(ref, column.name) for column in catalog]
        row = self._run(ref, aggregate_sql(self.provider.relation_sql(ref), quoted))[0]
        row_count, columns = column_profiles(catalog, row)
        return RelationProfile(ref.name, row_count, columns)

    def check_join(self, a: RelationRef, a_col: str, b: RelationRef, b_col: str) -> JoinCheck:
        if a.provider != b.provider:
            raise ProfileRefusedError(
                f"cannot join {a.name} on the {a.provider} provider with {b.name} on the "
                f"{b.provider} provider; a join check runs inside one provider"
            )
        statement = join_sql(
            self.provider.relation_sql(a),
            self.provider.quote(a, a_col),
            self.provider.relation_sql(b),
            self.provider.quote(b, b_col),
        )
        return join_check(self._run(a, statement)[0])

    def check_hierarchy(self, ref: RelationRef, id_col: str, parent_col: str) -> HierarchyCheck:
        statement = hierarchy_sql(
            self.provider.relation_sql(ref),
            self.provider.quote(ref, id_col),
            self.provider.quote(ref, parent_col),
        )
        cycles, roots, depth = (int(value) for value in self._run(ref, statement)[0])
        return HierarchyCheck(cycle_count=cycles, root_count=roots, max_depth=depth)

    def resolve_external_refs(self, system: str, codes: Sequence[str]) -> ExternalRefCheck:
        vocabulary = self.vocabulary
        if vocabulary is None or system not in self.loaded_systems:
            return ExternalRefCheck(system, resolve_rate=None, unverifiable=True)
        if not codes:
            return ExternalRefCheck(system, resolve_rate=None, unverifiable=False)
        found = sum(vocabulary.concept_by_code(system, code) is not None for code in codes)
        return ExternalRefCheck(system, resolve_rate=found / len(codes), unverifiable=False)

    def sample_values(
        self, ref: RelationRef, column: str, n: int
    ) -> ValueSample | NeedsValues:
        key = (ref.name, column)
        if key in self.policy.sensitive:
            raise ProfileRefusedError(
                f"column {column} of {ref.name} is bound to sensitive and is never sampled"
            )
        if not self._values_allowed(ref, column):
            return NeedsValues(relation=ref.name, columns=(column,))
        limit = max(0, min(n, SAMPLE_CAP))
        relation_sql, quoted = self.provider.relation_sql(ref), self.provider.quote(ref, column)
        rows = self._run(ref, sample_sql(relation_sql, quoted, limit), value_bearing=True)
        return ValueSample(ref.name, column, tuple(row[0] for row in rows))

    def read_schema_text(
        self, ref: RelationRef, columns: Sequence[str]
    ) -> list[tuple[Any, ...]]:
        bound = self.structure.schema_relations if self.structure is not None else ()
        if ref.name not in bound:
            raise ProfileRefusedError(
                f"relation {ref.name} is not bound as an entity_model dictionary or header,"
                " so its text is data, not schema"
            )
        quoted = [self.provider.quote(ref, column) for column in columns]
        statement = text_sql(self.provider.relation_sql(ref), quoted)
        return self._run(ref, statement, value_bearing=True, schema_relation=True)

    def _values_allowed(self, ref: RelationRef, column: str) -> bool:
        tier = self.policy.tier
        if tier == "full":
            return True
        if tier != "bounded_values" or (ref.name, column) not in self.policy.allowlist:
            return False
        statement = distinct_count_sql(
            self.provider.relation_sql(ref), self.provider.quote(ref, column)
        )
        distinct = int(self._run(ref, statement)[0][0])
        return distinct <= self.policy.max_distinct_values_read

    def _columns_sql(self, ref: RelationRef) -> str:
        if ref.provider == "staging":
            return staging_columns_sql(ref)
        return f"DESCRIBE TABLE {self.provider.relation_sql(ref)}"

    def _tables_sql(self, catalog_ref: RelationRef) -> str:
        if catalog_ref.provider == "staging":
            return staging_tables_sql(catalog_ref.name)
        catalog, _, schema = catalog_ref.name.rpartition(".")
        quoted = self.provider.quote(catalog_ref, catalog) if catalog else None
        return warehouse_tables_sql(quoted, schema)

    def _run(
        self,
        ref: RelationRef,
        statement: str,
        *,
        value_bearing: bool = False,
        schema_relation: bool = False,
    ) -> list[tuple[Any, ...]]:
        query = self.provider.query(ref)
        self.ledger.sql(self.stage, ref.provider, statement, value_bearing, schema_relation)
        return query(statement)
