"""Legacy helpers for the single-pass and two-pass L2 code paths.

These functions support the old SemanticEngine.interpret_table() pipeline.
Kept intact through rollout step 6 for comparison, removed in step 7.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)

if TYPE_CHECKING:
    from sema.engine.semantic import (
        SemanticEngine,
        TableInterpretation,
    )

logger = logging.getLogger(__name__)


def _alias_payload(
    value: str,
    is_preferred: bool,
    description: str | None = None,
) -> dict[str, Any]:
    """Build a HAS_ALIAS assertion payload."""
    payload: dict[str, Any] = {
        "value": value, "is_preferred": is_preferred,
    }
    if description is not None:
        payload["description"] = description
    return payload


def run_summary_pass(
    engine: SemanticEngine, table_metadata: dict[str, Any], table_ref: str
) -> tuple[list[Assertion], Any]:
    from sema.engine.semantic import build_summary_prompt
    from sema.llm_client import TableSummary

    summary_prompt = build_summary_prompt(table_metadata)
    summary = engine._llm_client.invoke(
        summary_prompt,
        TableSummary,
        table_ref=table_ref,
        stage_name="L2 semantic",
    )

    assertions: list[Assertion] = []
    assertions.append(engine._make_assertion(
        table_ref,
        AssertionPredicate.HAS_ENTITY_NAME,
        {
            "value": summary.entity_name,
            "description": summary.entity_description,
        },
    ))
    for i, syn in enumerate(summary.synonyms):
        assertions.append(engine._make_assertion(
            table_ref,
            AssertionPredicate.HAS_ALIAS,
            _alias_payload(syn, is_preferred=(i == 0)),
        ))

    return assertions, summary


def _prop_alias_assertions(
    engine: SemanticEngine,
    col_ref: str,
    synonyms: list[str],
    confidence: float,
) -> list[Assertion]:
    """Emit HAS_ALIAS assertions for property synonyms."""
    return [
        engine._make_assertion(
            col_ref,
            AssertionPredicate.HAS_ALIAS,
            _alias_payload(syn, is_preferred=(i == 0)),
            confidence=confidence,
        )
        for i, syn in enumerate(synonyms)
    ]


def run_property_pass(
    engine: SemanticEngine,
    table_metadata: dict[str, Any],
    table_ref: str,
    entity_name: str,
) -> list[Assertion]:
    from sema.engine.semantic import (
        _PropertyBatchResult,
        build_property_prompt,
    )

    assertions: list[Assertion] = []
    columns = table_metadata.get("columns", [])
    for i in range(0, len(columns), engine._column_batch_size):
        batch = columns[i:i + engine._column_batch_size]
        prop_prompt = build_property_prompt(
            table_metadata, batch, entity_name,
        )
        batch_result = engine._llm_client.invoke(
            prop_prompt,
            _PropertyBatchResult,
            table_ref=table_ref,
            stage_name="L2 semantic",
        )
        for prop in batch_result.properties:
            col_ref = f"{table_ref}.{prop.column}"
            assertions.append(engine._make_assertion(
                col_ref,
                AssertionPredicate.HAS_PROPERTY_NAME,
                {"value": prop.name, "description": prop.description},
                confidence=prop.confidence,
            ))
            assertions.append(engine._make_assertion(
                col_ref,
                AssertionPredicate.HAS_SEMANTIC_TYPE,
                {"value": prop.semantic_type},
                confidence=prop.confidence,
            ))
            assertions.extend(_prop_alias_assertions(
                engine, col_ref,
                prop.synonyms or [], prop.confidence,
            ))
            for dv in prop.decoded_values:
                assertions.append(engine._make_assertion(
                    col_ref,
                    AssertionPredicate.HAS_DECODED_VALUE,
                    {
                        "raw": dv.get("raw", dv.get("code", "")),
                        "label": dv.get(
                            "label", dv.get("name", dv.get("raw", "")),
                        ),
                    },
                    confidence=prop.confidence,
                ))
            if prop.vocabulary_guess:
                assertions.append(engine._make_assertion(
                    col_ref,
                    AssertionPredicate.VOCABULARY_MATCH,
                    {"value": prop.vocabulary_guess},
                    confidence=prop.confidence,
                ))

    return assertions


def entity_assertions(
    engine: SemanticEngine,
    interpretation: TableInterpretation,
    table_ref: str,
) -> list[Assertion]:
    assertions: list[Assertion] = []

    assertions.append(engine._make_assertion(
        table_ref,
        AssertionPredicate.HAS_ENTITY_NAME,
        {
            "value": interpretation.entity_name,
            "description": interpretation.entity_description,
        },
    ))

    for i, syn in enumerate(interpretation.synonyms or []):
        assertions.append(engine._make_assertion(
            table_ref,
            AssertionPredicate.HAS_ALIAS,
            _alias_payload(syn, is_preferred=(i == 0)),
        ))

    return assertions


def property_assertions(
    engine: SemanticEngine,
    interpretation: TableInterpretation,
    table_ref: str,
) -> list[Assertion]:
    assertions: list[Assertion] = []

    for prop in (interpretation.properties or []):
        col_ref = f"{table_ref}.{prop.column}"

        assertions.append(engine._make_assertion(
            col_ref,
            AssertionPredicate.HAS_PROPERTY_NAME,
            {"value": prop.name, "description": prop.description},
            confidence=prop.confidence,
        ))

        assertions.append(engine._make_assertion(
            col_ref,
            AssertionPredicate.HAS_SEMANTIC_TYPE,
            {"value": prop.semantic_type},
            confidence=prop.confidence,
        ))

        assertions.extend(_prop_alias_assertions(
            engine, col_ref,
            prop.synonyms or [], prop.confidence,
        ))

        for dv in (prop.decoded_values or []):
            assertions.append(engine._make_assertion(
                col_ref,
                AssertionPredicate.HAS_DECODED_VALUE,
                {
                    "raw": dv.get("raw", dv.get("code", "")),
                    "label": dv.get(
                        "label",
                        dv.get("name", dv.get("raw", "")),
                    ),
                },
                confidence=prop.confidence,
            ))

        if prop.vocabulary_guess:
            assertions.append(engine._make_assertion(
                col_ref,
                AssertionPredicate.VOCABULARY_MATCH,
                {"value": prop.vocabulary_guess},
                confidence=prop.confidence,
            ))

    return assertions
