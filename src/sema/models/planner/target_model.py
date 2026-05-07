"""target-model capability: target-side schema graph + obligations."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from sema.models.planner._enums import ModelRole, PrimaryKeyStrategy
from sema.models.planner._role_validation import require_role_identifier


class Constraint(BaseModel):
    id: str
    name: str
    rule_kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    model_role: ModelRole = ModelRole.SOURCE
    source_id: str | None = None
    target_model_id: str | None = None

    @model_validator(mode="after")
    def _validate_role(self) -> Self:
        require_role_identifier(self.model_role, self.source_id, self.target_model_id)
        return self


class ForeignKeyObligation(BaseModel):
    referenced_entity: str = Field(min_length=1)
    join_keys: list[tuple[str, str]] = Field(min_length=1)
    same_build_required: bool = True


class DomainConstraint(BaseModel):
    property_name: str = Field(min_length=1)
    domain_id: str = Field(min_length=1)


class FieldPresence(BaseModel):
    kind: Literal["presence"] = "presence"
    field: str = Field(min_length=1)


class FieldEquality(BaseModel):
    kind: Literal["equality"] = "equality"
    field: str = Field(min_length=1)
    value: Any


RowClause = FieldPresence | FieldEquality


class RowPredicate(BaseModel):
    op: Literal["AND", "OR"]
    clauses: list[RowClause] = Field(min_length=1)

    def evaluate(
        self,
        present_fields: set[str],
        values: dict[str, Any] | None = None,
    ) -> bool:
        results = [self._evaluate_clause(c, present_fields, values) for c in self.clauses]
        return all(results) if self.op == "AND" else any(results)

    @staticmethod
    def _evaluate_clause(
        clause: RowClause,
        present_fields: set[str],
        values: dict[str, Any] | None,
    ) -> bool:
        if isinstance(clause, FieldPresence):
            return clause.field in present_fields
        if values is None or clause.field not in values:
            return False
        return bool(values[clause.field] == clause.value)


class ExternalSequenceMappingTable(BaseModel):
    mapping_table_name: str = Field(min_length=1)
    canonical_identity_column: str = Field(min_length=1)
    sequence_column: str = Field(min_length=1)


class TargetObligation(BaseModel):
    target_entity: str = Field(min_length=1)
    required_fields: list[str] = Field(min_length=1)
    nullable_fields: list[str] = Field(default_factory=list)
    primary_key: PrimaryKeyStrategy
    external_sequence: ExternalSequenceMappingTable | None = None
    foreign_keys: list[ForeignKeyObligation] = Field(default_factory=list)
    domain_constraints: list[DomainConstraint] = Field(default_factory=list)
    allowed_defaults: dict[str, Any] = Field(default_factory=dict)
    minimum_viable_row: RowPredicate | None = None

    @model_validator(mode="after")
    def _validate_pk(self) -> Self:
        is_external = self.primary_key is PrimaryKeyStrategy.EXTERNAL_SEQUENCE
        if is_external and self.external_sequence is None:
            raise ValueError("EXTERNAL_SEQUENCE requires external_sequence")
        if not is_external and self.external_sequence is not None:
            raise ValueError(
                f"primary_key={self.primary_key.value} rejects external_sequence"
            )
        return self
