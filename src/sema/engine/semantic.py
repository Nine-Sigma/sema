from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from sema.engine.semantic_utils import (
    entity_assertions,
    property_assertions,
    run_property_pass,
    run_summary_pass,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)

logger = logging.getLogger(__name__)


class DecodedValue(BaseModel):
    raw: str
    label: str


class PropertyInterpretation(BaseModel):
    column: str
    name: str
    description: str | None = None
    semantic_type: str
    vocabulary_guess: str | None = None
    confidence: float = 0.75
    synonyms: list[str] | None = None
    decoded_values: list[dict[str, str]] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.synonyms is None:
            self.synonyms = []
        if self.decoded_values is None:
            self.decoded_values = []


class TableInterpretation(BaseModel):
    entity_name: str
    entity_description: str | None = None
    synonyms: list[str] | None = None
    properties: list[PropertyInterpretation] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.synonyms is None:
            self.synonyms = []
        if self.properties is None:
            self.properties = []


def build_interpretation_prompt(
    table_metadata: dict[str, Any],
    max_sample_values: int = 10,
) -> str:
    """Build the LLM prompt for interpreting a table's metadata."""
    parts = [
        f"Table: {table_metadata['table_name']}",
    ]
    if table_metadata.get("comment"):
        parts.append(f"Comment: {table_metadata['comment']}")

    parts.append("\nColumns:")
    for col in table_metadata.get("columns", []):
        line = f"  {col['name']} ({col['data_type']})"
        if col.get("comment"):
            line += f" - {col['comment']}"
        if col.get("top_values"):
            values = [v["value"] for v in col["top_values"][:max_sample_values]]
            line += f"\n    top values: {', '.join(values)}"
        parts.append(line)

    if table_metadata.get("sample_rows"):
        parts.append("\nSample rows:")
        for row in table_metadata["sample_rows"][:5]:
            parts.append(f"  {json.dumps(row)}")

    parts.append("""
Generate a JSON object with:
1. "entity_name": human-readable concept name for this table
2. "entity_description": what this table represents
3. "synonyms": alternative names someone might search for
4. "properties": array, one per column, each with:
   - "column": exact column name
   - "name": human-readable property name
   - "description": what this column means
   - "semantic_type": one of "identifier", "categorical", "temporal", "numeric", "free_text"
   - "vocabulary_guess": if categorical, what standard vocabulary (e.g., ICD-10, OncoTree, AJCC)
   - "confidence": 0.0-1.0 how confident you are
   - "synonyms": alternative names for this property
   - "decoded_values": if categorical, array of {"raw": "code", "label": "human name"}

Return ONLY valid JSON, no markdown.
""")

    return "\n".join(parts)


def build_simplified_interpretation_prompt(
    table_metadata: dict[str, Any],
) -> str:
    """Build a simplified prompt requesting TableInterpretation with minimal data.

    Sends only column names and types (no values or comments).
    Requests entity_name, entity_description, synonyms, and properties
    with column, name, and semantic_type only.
    """
    parts = [f"Table: {table_metadata['table_name']}"]

    parts.append("\nColumns (name : type):")
    for col in table_metadata.get("columns", []):
        parts.append(f"  {col['name']} : {col['data_type']}")

    parts.append("""
Return ONLY valid JSON with:
- "entity_name": human-readable concept name
- "entity_description": one-sentence description
- "synonyms": alternative names
- "properties": array, one per column, each with:
  - "column": exact column name
  - "name": human-readable property name
  - "semantic_type": one of "identifier", "categorical", "temporal", "numeric", "free_text"
""")
    return "\n".join(parts)


def build_summary_prompt(table_metadata: dict[str, Any]) -> str:
    """Build a lightweight prompt for the table summary pass.

    Sends only column names and types — no values, no descriptions.
    Asks for entity name, description, and synonyms only.
    """
    parts = [f"Table: {table_metadata['table_name']}"]
    if table_metadata.get("comment"):
        parts.append(f"Comment: {table_metadata['comment']}")

    parts.append("\nColumns (name : type):")
    for col in table_metadata.get("columns", []):
        parts.append(f"  {col['name']} : {col['data_type']}")

    parts.append("""
Based on the table name and column names above, determine what business
entity this table represents.

Return ONLY valid JSON with:
- "entity_name": human-readable concept name
- "entity_description": one-sentence description
- "synonyms": alternative names someone might search for
""")
    return "\n".join(parts)


def build_property_prompt(
    table_metadata: dict[str, Any],
    columns: list[dict[str, Any]],
    entity_name: str,
    max_sample_values: int = 10,
) -> str:
    """Build a prompt for a chunked property extraction pass."""
    parts = [
        f"Table: {table_metadata['table_name']}",
        f"This table represents: {entity_name}",
        "\nColumns to interpret:",
    ]
    for col in columns:
        line = f"  {col['name']} ({col['data_type']})"
        if col.get("comment"):
            line += f" - {col['comment']}"
        if col.get("top_values"):
            values = [
                v["value"] for v in col["top_values"][:max_sample_values]
            ]
            line += f"\n    top values: {', '.join(values)}"
        parts.append(line)

    parts.append("""
For each column above, return a JSON object with:
- "properties": array, one per column, each with:
  - "column": exact column name
  - "name": human-readable property name
  - "description": what this column means
  - "semantic_type": one of "identifier", "categorical", "temporal", "numeric", "free_text"
  - "vocabulary_guess": if categorical, what standard vocabulary
  - "confidence": 0.0-1.0
  - "synonyms": alternative names
  - "decoded_values": if categorical, array of {"raw": "code", "label": "human name"}

Return ONLY valid JSON, no markdown.
""")
    return "\n".join(parts)


class _PropertyBatchResult(BaseModel):
    properties: list[PropertyInterpretation] = []


class SemanticEngine:
    """L2: LLM-assisted semantic interpretation of table metadata."""

    def __init__(
        self,
        llm: Any = None,
        run_id: str | None = None,
        llm_client: Any = None,
        column_batch_size: int = 25,
    ) -> None:
        self._llm = llm
        self._llm_client = llm_client
        self._run_id = run_id or str(uuid.uuid4())
        self._column_batch_size = column_batch_size

    def _make_assertion(
        self,
        subject_ref: str,
        predicate: AssertionPredicate,
        payload: dict[str, Any],
        object_ref: str | None = None,
        confidence: float = 0.75,
    ) -> Assertion:
        return Assertion(
            id=str(uuid.uuid4()),
            subject_ref=subject_ref,
            predicate=predicate,
            payload=payload,
            object_ref=object_ref,
            source="llm_interpretation",
            confidence=confidence,
            run_id=self._run_id,
            observed_at=datetime.now(timezone.utc),
        )

    def _interpret_via_llm_client(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> TableInterpretation:
        prompt = build_interpretation_prompt(table_metadata)
        simplified = build_simplified_interpretation_prompt(table_metadata)
        return self._llm_client.invoke(  # type: ignore[no-any-return]
            prompt,
            TableInterpretation,
            table_ref=table_ref,
            stage_name="L2 semantic",
            simplified_prompt=simplified,
        )

    def _interpret_via_raw_llm(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> TableInterpretation:
        prompt = build_interpretation_prompt(table_metadata)
        try:
            response = self._llm.invoke(prompt)
            raw_content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            raw_content = raw_content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                lines = [
                    line
                    for line in lines
                    if not line.strip().startswith("```")
                ]
                raw_content = "\n".join(lines).strip()
            return TableInterpretation.model_validate_json(raw_content)
        except Exception as e:
            logger.warning(
                f"LLM interpretation failed for {table_ref}: {e}"
            )
            return None  # type: ignore[return-value]

    def _needs_two_pass(self, table_metadata: dict[str, Any]) -> bool:
        columns = table_metadata.get("columns", [])
        threshold = self._column_batch_size * 2
        return len(columns) >= threshold and self._llm_client is not None

    def _run_summary_pass(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> tuple[list[Assertion], Any]:
        return run_summary_pass(self, table_metadata, table_ref)

    def _run_property_pass(
        self, table_metadata: dict[str, Any], table_ref: str, entity_name: str
    ) -> list[Assertion]:
        return run_property_pass(self, table_metadata, table_ref, entity_name)

    def _interpret_two_pass(
        self, table_metadata: dict[str, Any], table_ref: str
    ) -> list[Assertion]:
        summary_assertions, summary = self._run_summary_pass(table_metadata, table_ref)
        property_assertions_list = self._run_property_pass(table_metadata, table_ref, summary.entity_name)
        return summary_assertions + property_assertions_list

    def interpret_table(
        self, table_metadata: dict[str, Any]
    ) -> list[Assertion]:
        """Interpret a single table's metadata via LLM.

        Uses two-pass strategy for wide tables (>= 2*column_batch_size cols).
        When using LLMClient: LLMStageError propagates to caller (no catch).
        When using raw LLM (legacy): errors are swallowed and empty list returned.
        """
        table_ref = table_metadata.get(
            "table_ref",
            f"unity://{table_metadata.get('table_name', 'unknown')}",
        )

        if self._needs_two_pass(table_metadata):
            return self._interpret_two_pass(table_metadata, table_ref)

        if self._llm_client:
            # New path: LLMStageError propagates — no catch
            interpretation = self._interpret_via_llm_client(
                table_metadata, table_ref
            )
        elif self._llm:
            # Legacy path: swallows errors
            interpretation = self._interpret_via_raw_llm(
                table_metadata, table_ref
            )
            if interpretation is None:
                return []
        else:
            return []

        return self._interpretation_to_assertions(interpretation, table_ref)

    def _entity_assertions(
        self, interpretation: TableInterpretation, table_ref: str
    ) -> list[Assertion]:
        return entity_assertions(self, interpretation, table_ref)

    def _property_assertions(
        self, interpretation: TableInterpretation, table_ref: str
    ) -> list[Assertion]:
        return property_assertions(self, interpretation, table_ref)

    def _interpretation_to_assertions(
        self, interpretation: TableInterpretation, table_ref: str
    ) -> list[Assertion]:
        return (
            self._entity_assertions(interpretation, table_ref)
            + self._property_assertions(interpretation, table_ref)
        )

    def interpret_tables(
        self, tables_metadata: list[dict[str, Any]]
    ) -> list[Assertion]:
        """Interpret multiple tables, returning all candidate assertions."""
        all_assertions: list[Assertion] = []
        for table in tables_metadata:
            assertions = self.interpret_table(table)
            all_assertions.extend(assertions)
        return all_assertions
