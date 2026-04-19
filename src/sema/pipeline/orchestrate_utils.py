"""Helper functions for pipeline orchestration.

Extracted from orchestrate.py to keep the module focused on the
public run_build / run_context / run_query entry points.
"""
from __future__ import annotations

from typing import Any

import click

import sema.cli_factories as _factories
from sema.cli_factories import (
    _get_embedder,
    _get_neo4j_driver,
)
from sema.models.config import (
    BuildConfig,
    QueryConfig,
)
from sema.models.domain import DomainContext
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine


def _register_datasource(connector: Any, loader: Any) -> None:
    """Create or update the DataSource node for the given connector."""
    import uuid
    ref, platform, workspace = connector.get_datasource_ref()
    loader.upsert_datasource(
        id=str(uuid.uuid4()),
        ref=ref,
        platform=platform,
        workspace=workspace,
    )


def _discover_tables(
    connector: Any,
    config: BuildConfig,
) -> list[Any]:
    if config.verbose:
        click.echo("Step 1: Discovering tables...")
    work_items = connector.discover_tables(
        catalog=config.catalog,
        schemas=config.schemas or None,
        table_pattern=config.table_pattern,
    )
    if config.slice_tables:
        work_items = _filter_work_items_to_slice(
            work_items, config.slice_tables,
        )
    if config.verbose:
        click.echo(f"  Found {len(work_items)} tables")
    return work_items  # type: ignore[no-any-return]


def _filter_work_items_to_slice(
    work_items: list[Any], slice_tables: list[str],
) -> list[Any]:
    if not slice_tables:
        return work_items
    allowed = set(slice_tables)
    return [w for w in work_items if w.table_name in allowed]


def _log_result(result: Any, label: str, verbose: bool) -> None:
    if not verbose:
        return
    status = result.status
    if status == "failed":
        click.echo(
            f"  {label}: FAILED at {result.failed_stage}: "
            f"{result.error_message}"
        )
    else:
        click.echo(f"  {label}: {status}")


def _build_prompt_layers(config: BuildConfig) -> Any:
    """Build PromptLayers from BuildConfig flags."""
    from sema.engine.stage_utils import PromptLayers
    return PromptLayers(
        enable_domain_bias=config.enable_domain_bias,
        enable_type_inventory=config.enable_type_inventory,
        enable_vocab_hints=config.enable_vocab_hints,
        enable_few_shot=config.enable_few_shot,
        enable_stage_c=config.enable_stage_c,
    )


def _spawn_workers_parallel(
    work_items: list[Any],
    config: BuildConfig,
    connector_factory: Any,
    llm_factory: Any,
    loader: Any,
    run_id: str,
    domain_context: DomainContext | None = None,
) -> list[Any]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sema.pipeline.build import process_table

    results: list[Any] = []
    layers = _build_prompt_layers(config)

    def _process_worker(work_item: Any) -> Any:
        worker_connector = connector_factory.create()
        worker_llm_client = llm_factory.create()
        return process_table(
            work_item, worker_connector, worker_llm_client, loader,
            run_id=run_id,
            column_batch_size=config.column_batch_size,
            vocab_workers=config.vocab_workers,
            resume=config.resume,
            domain_context=domain_context,
            use_staged=config.use_staged,
            prompt_layers=layers,
            eval_dump_dir=config.eval_dump_dir,
            eval_config_label=config.eval_config_label,
        )

    with ThreadPoolExecutor(
        max_workers=config.table_workers
    ) as executor:
        future_to_item = {
            executor.submit(_process_worker, wi): wi
            for wi in work_items
        }
        for future in as_completed(future_to_item):
            wi = future_to_item[future]
            try:
                result = future.result()
            except Exception as e:
                from sema.pipeline.build import (
                    TableResult,
                )
                result = TableResult.failed(
                    wi.fqn, "executor", str(e)
                )
            _log_result(result, wi.table_name, config.verbose)
            results.append(result)

    return results


def _spawn_workers(
    work_items: list[Any],
    config: BuildConfig,
    connector_factory: Any,
    llm_factory: Any,
    loader: Any,
    run_id: str,
    domain_context: DomainContext | None = None,
) -> list[Any]:
    from sema.pipeline.build import process_table

    if config.verbose:
        click.echo(
            f"Step 2: Processing tables "
            f"(table_workers={config.table_workers})..."
        )

    if config.table_workers > 1:
        return _spawn_workers_parallel(
            work_items, config, connector_factory,
            llm_factory, loader, run_id,
            domain_context=domain_context,
        )

    results: list[Any] = []
    connector = connector_factory.create()
    llm_client = llm_factory.create()
    layers = _build_prompt_layers(config)
    for i, work_item in enumerate(work_items):
        if config.verbose:
            click.echo(
                f"  [{i+1}/{len(work_items)}] {work_item.table_name}"
            )
        result = process_table(
            work_item, connector, llm_client, loader,
            run_id=run_id,
            column_batch_size=config.column_batch_size,
            vocab_workers=config.vocab_workers,
            resume=config.resume,
            domain_context=domain_context,
            use_staged=config.use_staged,
            prompt_layers=layers,
            eval_dump_dir=config.eval_dump_dir,
            eval_config_label=config.eval_config_label,
        )
        _log_result(result, f"    ", config.verbose)
        results.append(result)

    return results


def _collect_results(
    results: list[Any],
) -> dict[str, Any]:
    from sema.pipeline.build import aggregate_report

    return aggregate_report(results)


def _compute_embeddings(
    config: BuildConfig, loader: Any, *, skip_embeddings: bool = False,
) -> None:
    from sema.engine.embeddings import (
        EMBEDDING_KEY_MAP,
        EmbeddingEngine,
        build_embedding_text,
    )

    if config.verbose:
        click.echo("Step 3: Computing embeddings...")
    try:
        embeddable_labels = config.embedding.embeddable_labels
        embedder = _get_embedder(config.embedding)
        emb_engine = EmbeddingEngine(
            model=embedder, loader=loader,
            embeddable_labels=embeddable_labels,
        )
        if hasattr(embedder, "get_sentence_embedding_dimension"):
            dimensions: int = embedder.get_sentence_embedding_dimension()
        else:
            probe = emb_engine.embed_batch(["dimension probe"])
            dimensions = len(probe[0]) if probe and probe[0] else 768
        emb_engine.create_all_indexes(dimensions=dimensions)
        if config.verbose:
            click.echo(f"  Vector indexes created (dimensions={dimensions})")

        if skip_embeddings or config.skip_embeddings:
            if config.verbose:
                click.echo("  Skipping embed_and_store (skip_embeddings=True)")
            return

        for label in embeddable_labels:
            nodes = loader.query_nodes_by_label(label)
            if not nodes:
                continue
            key_spec = EMBEDDING_KEY_MAP.get(label, "name")
            _embed_label_nodes(
                emb_engine, loader, label, key_spec, nodes,
            )
            if config.verbose:
                click.echo(f"  {label}: embedded {len(nodes)} nodes")
    except Exception as e:
        if config.verbose:
            click.echo(f"  Skipping embeddings: {e}")


def _embed_label_nodes(
    engine: Any,
    loader: Any,
    label: str,
    key_spec: str | tuple[str, ...],
    nodes: list[dict[str, Any]],
) -> None:
    from sema.engine.embeddings import build_embedding_text

    max_batch = 64
    texts = [
        build_embedding_text(label.lower(), **node) for node in nodes
    ]

    for start in range(0, len(nodes), max_batch):
        batch_texts = texts[start : start + max_batch]
        batch_nodes = nodes[start : start + max_batch]
        embeddings = engine.embed_batch(batch_texts)

        for node, embedding in zip(batch_nodes, embeddings):
            emb_list = list(embedding)
            if isinstance(key_spec, tuple):
                loader.set_property_embedding(
                    name=node[key_spec[0]],
                    entity_name=node[key_spec[1]],
                    embedding=emb_list,
                )
            else:
                loader.set_embedding(
                    label=label,
                    match_prop=key_spec,
                    match_value=node[key_spec],
                    embedding=emb_list,
                )


def _retrieve_context(config: QueryConfig) -> Any:
    driver = _get_neo4j_driver(config.neo4j)

    try:
        embedder = _factories._get_embedder(config.embedding)
    except Exception:
        embedder = None

    engine = RetrievalEngine(driver=driver, embedder=embedder)
    candidate_set = engine.retrieve(config.question)
    sco = prune_to_sco(candidate_set, consumer=config.consumer)

    driver.close()
    return sco


