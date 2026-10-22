"""Read-only, logged profiling tools over landed relations (design Decisions 2 and 18).

Describes and structural checks return aggregates only. Values come back only through
``sample_values`` when the evidence policy allows them, and schema text only through
``read_schema_text`` for a relation the structure output binds as a dictionary or header.
Every statement is written to the run ledger before it is issued.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sema.binding.spec_sections import Provider, RelationRef
from sema.intake.landing import LandedRelation, LandResult
from sema.intake.ledger import RunLedger
from sema.intake.profile_utils import (
    ExternalRefCheck,
    HierarchyCheck,
    JoinCheck,
    LevelCheck,
    NeedsValues,
    PathCheck,
    RelationListing,
    RelationProfile,
    ValueSample,
    CatalogColumn,
    aggregate_sql,
    catalog_columns,
    column_profiles,
    count_where_sql,
    distinct_count_sql,
    distinct_tuples_sql,
    hierarchy_sql,
    join_check,
    join_sql,
    levels_sql,
    overlap_sql,
    paths_sql,
    sample_sql,
    staging_columns_sql,
    staging_tables_sql,
    text_sql,
)
from sema.intake.relations import RelationProvider
from sema.models.config import BuildConfig, EvidenceTier

__all__ = [
    "EvidencePolicy",
    "EvidenceTier",
    "ExternalRefCheck",
    "HierarchyCheck",
    "JoinCheck",
    "LevelCheck",
    "NeedsValues",
    "PathCheck",
    "ProfileRefusedError",
    "ProfileTools",
    "RelationListing",
    "RelationProfile",
    "ValueSample",
]

ColumnKey = tuple[str, str]
SAMPLE_CAP = 20
DEFAULT_MAX_DISTINCT_VALUES_READ: int = BuildConfig.model_fields["max_distinct_values_read"].default


class ProfileRefusedError(ValueError):
    """A request the policy, the providers, or the structure output does not allow."""


class EvidencePolicy(BaseModel):
    """The run's evidence tier; column keys are ``(RelationRef.name, column)``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: EvidenceTier = "metadata_only"
    allowlist: frozenset[ColumnKey] = frozenset()
    max_distinct_values_read: int = Field(default=DEFAULT_MAX_DISTINCT_VALUES_READ, ge=0)
    sensitive: frozenset[ColumnKey] = frozenset()

    @classmethod
    def from_build_config(cls, config: BuildConfig) -> EvidencePolicy:
        return cls(
            tier=config.evidence_policy,
            max_distinct_values_read=config.max_distinct_values_read,
        )


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
        land: LandResult | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.ledger = ledger
        self.stage = stage
        self.structure = structure
        self.vocabulary = vocabulary
        self.loaded_systems = loaded_systems
        self.land = land
        self.value_requests: list[NeedsValues] = []

    def list_relations(self, provider_name: Provider, namespace: str) -> list[RelationListing]:
        catalog_ref = RelationRef(provider=provider_name, name=namespace)
        listings = []
        for table, comment in self._tables(catalog_ref):
            ref = RelationRef(provider=provider_name, name=f"{namespace}.{table}")
            listings.append(RelationListing(ref.name, self.count_rows(ref), comment or None))
        return listings

    def count_rows(self, ref: RelationRef) -> int:
        count = self._run(ref, f"SELECT COUNT(*) FROM {self.provider.relation_sql(ref)}")
        return int(count[0][0])

    def catalog(self, ref: RelationRef) -> list[CatalogColumn]:
        if ref.provider == "staging":
            return catalog_columns(self._run(ref, staging_columns_sql(ref)))
        landed = self._landed(ref.name)
        comments = dict(landed.comments)
        return [CatalogColumn(name, kind, comments.get(name)) for name, kind in landed.columns]

    def describe_relation(self, ref: RelationRef) -> RelationProfile:
        catalog = self.catalog(ref)
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

    def overlapping_ranges(
        self, ref: RelationRef, typed_groups: Sequence[Sequence[str]]
    ) -> set[frozenset[str]]:
        """Pairs of columns, each group of one type, whose value ranges overlap."""
        paired = [group for group in typed_groups if len(group) > 1]
        if not paired:
            return set()
        names = [column for group in paired for column in group]
        quoted = [[self.provider.quote(ref, column) for column in group] for group in paired]
        rows = self._run(ref, overlap_sql(self.provider.relation_sql(ref), quoted))
        return {frozenset((names[int(a)], names[int(b)])) for a, b in rows}

    def check_hierarchy(self, ref: RelationRef, id_col: str, parent_col: str) -> HierarchyCheck:
        statement = hierarchy_sql(
            self.provider.relation_sql(ref),
            self.provider.quote(ref, id_col),
            self.provider.quote(ref, parent_col),
        )
        cycles, roots, depth = (int(value) for value in self._run(ref, statement)[0])
        return HierarchyCheck(cycle_count=cycles, root_count=roots, max_depth=depth)

    def count_distinct(
        self, ref: RelationRef, columns: Sequence[str], *, where: str | None = None
    ) -> int:
        quoted = [self.provider.quote(ref, column) for column in columns]
        statement = distinct_tuples_sql(self.provider.relation_sql(ref), quoted, where)
        return int(self._run(ref, statement)[0][0])

    def count_where(
        self, ref: RelationRef, conditions: Sequence[str], *, where: str | None = None
    ) -> tuple[int, ...]:
        """The row count, then the rows meeting each condition.

        Conditions are boolean SQL over identifiers quoted through ``provider``; each is
        wrapped in a count, so a value never comes back.
        """
        statement = count_where_sql(self.provider.relation_sql(ref), conditions, where)
        return tuple(int(value) for value in self._run(ref, statement)[0])

    def check_levels(self, ref: RelationRef, columns: Sequence[str]) -> LevelCheck:
        if len(columns) < 2:
            raise ProfileRefusedError(f"a level check over {ref.name} needs two or more columns")
        quoted = [self.provider.quote(ref, column) for column in columns]
        row = self._run(ref, levels_sql(self.provider.relation_sql(ref), quoted))[0]
        return LevelCheck(misplaced_nulls=int(row[0]), multi_parent=int(row[1]))

    def check_paths(self, ref: RelationRef, column: str, separator: str) -> PathCheck:
        if len(separator) != 1:
            raise ProfileRefusedError(f"path separator {separator!r} is not one character")
        statement = paths_sql(
            self.provider.relation_sql(ref), self.provider.quote(ref, column), separator
        )
        codes, separated, malformed, implicit, spaced = (
            int(value) for value in self._run(ref, statement)[0]
        )
        return PathCheck(codes, separated, malformed, implicit, spaced)

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
            request = NeedsValues(relation=ref.name, columns=(column,))
            self.value_requests.append(request)
            return request
        limit = max(0, min(n, SAMPLE_CAP))
        relation_sql, quoted = self.provider.relation_sql(ref), self.provider.quote(ref, column)
        rows = self._run(ref, sample_sql(relation_sql, quoted, limit), value_bearing=True)
        return ValueSample(ref.name, column, tuple(row[0] for row in rows))

    def read_schema_text(
        self, ref: RelationRef, columns: Sequence[str], limit: int | None = None
    ) -> list[tuple[Any, ...]]:
        bound = self.structure.schema_relations if self.structure is not None else ()
        if ref.name not in bound:
            raise ProfileRefusedError(
                f"relation {ref.name} is not bound as an entity_model dictionary or header,"
                " so its text is data, not schema"
            )
        if sensitive := [c for c in columns if (ref.name, c) in self.policy.sensitive]:
            raise ProfileRefusedError(
                f"columns {', '.join(sensitive)} of {ref.name} are bound to sensitive,"
                " so their text is never read"
            )
        quoted = [self.provider.quote(ref, column) for column in columns]
        statement = text_sql(self.provider.relation_sql(ref), quoted, limit)
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

    def _tables(self, catalog_ref: RelationRef) -> list[tuple[str, str | None]]:
        if catalog_ref.provider == "staging":
            listed = self._run(catalog_ref, staging_tables_sql(catalog_ref.name))
            return [(str(row[0]), row[1] or None) for row in listed]
        relations = self._warehouse_land().relations
        names = sorted(r.name for r in relations if r.schema == catalog_ref.name)
        return [(name, None) for name in names]

    def _landed(self, name: str) -> LandedRelation:
        landed = next(
            (r for r in self._warehouse_land().relations if f"{r.schema}.{r.name}" == name), None
        )
        if landed is None:
            raise ProfileRefusedError(
                f"warehouse relation {name} is not in the warehouse Land, so its catalog"
                " facts are unknown"
            )
        return landed

    def _warehouse_land(self) -> LandResult:
        if self.land is None:
            raise ProfileRefusedError(
                "warehouse catalog facts come only from the landed relations, and no Land"
                " was given"
            )
        return self.land

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
