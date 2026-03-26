"""Helper functions for the per-table build pipeline.

Extracted from build.py to keep the module focused on
public entry points and data classes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from sema.log import logger
from sema.engine.vocabulary import VocabularyEngine
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

if TYPE_CHECKING:
    from sema.connectors.databricks import (
        DatabricksConnector,
        TableWorkItem,
    )
    from sema.engine.semantic import SemanticEngine
    from sema.graph.loader import GraphLoader
    from sema.llm_client import LLMClient


def _build_table_metadata(
    assertions: list[Assertion],
    table_ref: str,
) -> dict[str, Any]:
    from sema.models.constants import parse_unity_ref_strict as _parse_ref

    meta: dict[str, Any] = {
        "table_ref": table_ref,
        "columns": [],
        "sample_rows": [],
        "comment": None,
    }

    for a in assertions:
        if a.predicate == AssertionPredicate.TABLE_EXISTS:
            cat, sch, tbl, _ = _parse_ref(a.subject_ref)
            meta.update({
                "table_name": tbl,
                "schema_name": sch,
                "catalog": cat,
            })
        elif a.predicate == AssertionPredicate.HAS_COMMENT:
            if a.subject_ref == table_ref:
                meta["comment"] = a.payload.get("value")
        elif a.predicate == AssertionPredicate.COLUMN_EXISTS:
            col_name = a.subject_ref.split(".")[-1]
            col_entry: dict[str, Any] = {
                "name": col_name,
                "data_type": a.payload.get("data_type", "UNKNOWN"),
                "nullable": a.payload.get("nullable", True),
                "comment": a.payload.get("comment"),
                "top_values": None,
            }
            meta["columns"].append(col_entry)
        elif a.predicate == AssertionPredicate.HAS_TOP_VALUES:
            col_name = a.subject_ref.split(".")[-1]
            for col in meta["columns"]:
                if col["name"] == col_name:
                    col["top_values"] = a.payload.get("values", [])
        elif a.predicate == AssertionPredicate.HAS_SAMPLE_ROWS:
            meta["sample_rows"] = a.payload.get("rows", [])

    return meta


def _run_extraction(
    work_item: TableWorkItem,
    connector: DatabricksConnector,
) -> tuple[list[Assertion], int]:
    logger.info(
        f"[{work_item.table_name}] Extracting metadata..."
    )
    extraction_assertions = connector.extract_table(work_item)
    col_count = sum(
        1 for a in extraction_assertions
        if a.predicate == AssertionPredicate.COLUMN_EXISTS
    )
    logger.info(
        f"[{work_item.table_name}] Extracted "
        f"{len(extraction_assertions)} assertions "
        f"({col_count} columns)"
    )
    return extraction_assertions, col_count


def _run_semantic_interpretation(
    table_meta: dict[str, Any],
    work_item: TableWorkItem,
    llm_client: LLMClient,
    run_id: str,
    column_batch_size: int,
) -> list[Assertion]:
    from sema.engine.semantic import SemanticEngine

    col_count = len(table_meta.get("columns", []))
    logger.info(
        f"[{work_item.table_name}] L2 semantic "
        f"interpretation ({col_count} columns)..."
    )
    semantic = SemanticEngine(
        llm_client=llm_client,
        run_id=run_id,
        column_batch_size=column_batch_size,
    )
    semantic_assertions = semantic.interpret_table(table_meta)
    logger.info(
        f"[{work_item.table_name}] L2 produced "
        f"{len(semantic_assertions)} assertions"
    )
    return semantic_assertions


def _build_vocab_work_items(
    extraction_assertions: list[Assertion],
    semantic_assertions: list[Assertion],
) -> list[tuple[str, list[str], list[dict[str, Any]] | None]]:
    items: list[tuple[str, list[str], list[dict[str, Any]] | None]] = []
    for a in extraction_assertions:
        if a.predicate != AssertionPredicate.HAS_TOP_VALUES:
            continue
        values = [
            v["value"]
            for v in a.payload.get("values", [])
        ]
        decoded = [
            da.payload
            for da in semantic_assertions
            if (
                da.predicate
                == AssertionPredicate.HAS_DECODED_VALUE
                and da.subject_ref == a.subject_ref
            )
        ]
        items.append(
            (a.subject_ref, values, decoded or None)
        )
    return items


def _run_vocabulary_alignment(
    extraction_assertions: list[Assertion],
    semantic_assertions: list[Assertion],
    work_item: TableWorkItem,
    llm_client: LLMClient,
    run_id: str,
    vocab_workers: int = 8,
) -> list[Assertion]:
    vocab = VocabularyEngine(
        llm_client=llm_client, run_id=run_id
    )
    work_items = _build_vocab_work_items(
        extraction_assertions, semantic_assertions
    )
    if not work_items:
        return []

    vocab_assertions: list[Assertion] = []

    with ThreadPoolExecutor(
        max_workers=vocab_workers
    ) as executor:
        future_to_ref = {
            executor.submit(
                vocab.process_column, ref, vals, dec
            ): ref
            for ref, vals, dec in work_items
        }
        for future in as_completed(future_to_ref):
            ref = future_to_ref[future]
            try:
                col_assertions = future.result()
                vocab_assertions.extend(col_assertions)
            except Exception as exc:
                logger.warning(
                    f"[{work_item.table_name}] L3 failed "
                    f"for {ref}: {exc}"
                )

    if vocab_assertions:
        logger.info(
            f"[{work_item.table_name}] L3 produced "
            f"{len(vocab_assertions)} vocabulary assertions"
        )
    return vocab_assertions


def _commit_and_materialize(
    all_assertions: list[Assertion],
    work_item: TableWorkItem,
    loader: GraphLoader,
) -> None:
    logger.info(
        f"[{work_item.table_name}] Committing "
        f"{len(all_assertions)} assertions..."
    )
    loader.commit_table_assertions(all_assertions)
    logger.info(
        f"[{work_item.table_name}] Materializing graph..."
    )
    loader.materialize_table_graph(all_assertions)
    logger.info(
        f"[{work_item.table_name}] Done"
    )


def _count_results(
    all_assertions: list[Assertion],
) -> tuple[int, int, int]:
    entity_count = sum(
        1 for a in all_assertions
        if a.predicate == AssertionPredicate.HAS_ENTITY_NAME
    )
    prop_count = sum(
        1 for a in all_assertions
        if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME
    )
    term_count = sum(
        1 for a in all_assertions
        if a.predicate == AssertionPredicate.HAS_DECODED_VALUE
    )
    return entity_count, prop_count, term_count


def _reconstruct_assertions(
    raw_dicts: list[dict[str, Any]],
) -> list[Assertion]:
    import json
    from datetime import datetime, timezone

    assertions: list[Assertion] = []
    for row in raw_dicts:
        payload_raw = row.get("payload", "{}")
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
        observed_raw = row.get("observed_at", "")
        if isinstance(observed_raw, str):
            observed = datetime.fromisoformat(observed_raw)
        else:
            observed = observed_raw
        assertions.append(
            Assertion(
                id=row["id"],
                subject_ref=row["subject_ref"],
                predicate=AssertionPredicate(row["predicate"]),
                payload=payload,
                object_ref=row.get("object_ref"),
                source=row.get("source", "unknown"),
                confidence=float(row.get("confidence", 0.0)),
                status=AssertionStatus(row.get("status", "auto")),
                run_id=row.get("run_id", ""),
                observed_at=observed,
            )
        )
    return assertions


def _run_pipeline_stages(
    work_item: TableWorkItem,
    connector: DatabricksConnector,
    llm_client: LLMClient,
    loader: GraphLoader,
    run_id: str,
    column_batch_size: int,
    vocab_workers: int = 8,
) -> list[Assertion] | Any:
    """Run all pipeline stages for a single table.

    Returns either a list of assertions on success or a TableResult
    if the table should be skipped.
    """
    from sema.pipeline.build import TableResult

    all_assertions: list[Assertion] = []

    extraction_assertions, col_count = _run_extraction(
        work_item, connector
    )
    all_assertions.extend(extraction_assertions)

    table_meta = _build_table_metadata(
        extraction_assertions, work_item.fqn
    )
    if "table_name" not in table_meta:
        return TableResult.skipped(
            work_item.fqn, "no table metadata"
        )

    semantic_assertions = _run_semantic_interpretation(
        table_meta, work_item, llm_client, run_id,
        column_batch_size,
    )
    all_assertions.extend(semantic_assertions)

    vocab_assertions = _run_vocabulary_alignment(
        extraction_assertions, semantic_assertions,
        work_item, llm_client, run_id,
        vocab_workers=vocab_workers,
    )
    all_assertions.extend(vocab_assertions)

    _commit_and_materialize(all_assertions, work_item, loader)

    return all_assertions
