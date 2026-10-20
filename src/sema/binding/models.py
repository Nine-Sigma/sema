"""Authored-evidence records written by the concept source importer.

A library's existing column bindings and value mappings are kept as judge
evidence only: they have no approver, time, or status, so they can never be
mistaken for a ``HumanPin``.
"""

from __future__ import annotations

from pydantic import BaseModel


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


class AuthoredEvidence(BaseModel):
    column_bindings: list[AuthoredColumnBinding]
    value_mappings: list[AuthoredValueMapping]
