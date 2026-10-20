"""Authored-evidence records written by the concept source importer.

A library's existing column bindings and value mappings are kept as judge
evidence only: they have no approver, time, or status, so they can never be
mistaken for a ``HumanPin``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sema.binding.mappings_utils import SssomMapping


class AuthoredColumnBinding(BaseModel):
    concept_ref: str
    dataset_label: str | None
    dataset_id: str | None
    physical: str | None
    columns: list[str]
    resolved: bool
    project: str | None


class AuthoredValueMapping(BaseModel):
    concept_ref: str
    source_value: str | None
    codes: list[str]
    status: str | None
    approved: bool


class Retirement(BaseModel):
    """A retired concept or inline term, kept out of the manifest but not lost."""

    kind: Literal["term", "concept"]
    vocabulary: str | None
    code: str
    replaced_by: str | None
    valid_from: str | None
    valid_to: str | None


class AuthoredEvidence(BaseModel):
    column_bindings: list[AuthoredColumnBinding]
    value_mappings: list[AuthoredValueMapping]
    mappings: list[SssomMapping] = Field(default_factory=list)
    retirements: list[Retirement] = Field(default_factory=list)
