from __future__ import annotations

from typing import Any

import click

from sema.cli_factories import (
    DatabricksProviderAuthError,
    _get_embedder,
    _get_llm,
    _get_neo4j_driver,
)
from sema.models.config import (
    BuildConfig,
    QueryConfig,
)
from sema.models.domain import DomainContext, resolve_domain_context
from sema.models.extraction import ExtractedTable
from sema.pipeline.orchestrate_utils import (
    _collect_results,
    _compute_embeddings,
    _discover_tables,
    _log_result,
    _register_datasource,
    _retrieve_context,
    _spawn_workers,
    _spawn_workers_parallel,
)


def run_build(config: BuildConfig) -> dict[str, Any]:
    """Run the full build pipeline.

    Uses the new vertical per-table processing model:
    discover -> enqueue -> process (per table) -> commit -> materialize.

    Each table is processed through all stages on one thread.
    table_workers controls parallelism (default 1 = sequential).
    """
    from sema.circuit_breaker import CircuitBreaker
    from sema.connectors.databricks import DatabricksConnector
    from sema.graph.loader import GraphLoader
    from sema.pipeline.build import (
        DatabricksConnectorFactory,
        LLMClientFactory,
        aggregate_report,
    )
    import uuid

    run_id = str(uuid.uuid4())
    driver = _get_neo4j_driver(config.neo4j)
    loader = GraphLoader(driver)

    discovery_connector = DatabricksConnector(
        config=config.databricks, profiling=config.profiling,
    )
    _register_datasource(discovery_connector, loader)
    work_items = _discover_tables(discovery_connector, config)

    if not work_items:
        driver.close()
        return aggregate_report([])

    circuit_breaker = CircuitBreaker(
        failure_threshold=config.circuit_breaker_threshold,
        recovery_timeout=float(config.circuit_breaker_timeout),
        success_threshold=config.circuit_breaker_success_threshold,
    )

    connector_factory = DatabricksConnectorFactory(
        config=config.databricks, profiling=config.profiling,
    )
    llm_factory = LLMClientFactory(
        llm_factory=lambda: _get_llm(config.llm),
        retry_max_attempts=config.retry_max_attempts,
        use_structured_output=config.llm.use_structured_output,
        circuit_breaker=circuit_breaker,
    )

    from sema.pipeline.profiler import WarehouseProfiler

    extracted_tables = [
        ExtractedTable(
            name=wi.table_name, catalog=wi.catalog, schema=wi.schema,
        )
        for wi in work_items
    ]
    profiler = WarehouseProfiler()
    datasource_ref, _, _ = discovery_connector.get_datasource_ref()
    profile = profiler.profile(
        tables=extracted_tables, columns=[],
        datasource_id=datasource_ref, run_id=run_id,
    )

    cli_domain = config.domain if config.domain_from_cli else None
    config_domain = config.domain if not config.domain_from_cli else None
    domain_context = resolve_domain_context(
        cli_domain=cli_domain,
        config_domain=config_domain,
        profile=profile,
    )

    results = _spawn_workers(
        work_items, config, connector_factory, llm_factory,
        loader, run_id,
        domain_context=domain_context,
    )

    report = _collect_results(results)

    _compute_embeddings(config, loader, skip_embeddings=config.skip_embeddings)

    driver.close()
    return report


def run_context(config: QueryConfig) -> dict[str, Any]:
    """Run retrieval and produce an SCO."""
    from sema.pipeline.retrieval import RetrievalEngine
    from sema.pipeline.context import prune_to_sco

    driver = _get_neo4j_driver(config.neo4j)
    try:
        embedder = _get_embedder(config.embedding)
    except DatabricksProviderAuthError:
        raise
    except Exception:
        embedder = None

    engine = RetrievalEngine(
        driver=driver,
        embedder=embedder,
        embedder_model_name=config.embedding.model,
    )
    candidate_set = engine.retrieve(config.question)
    sco = prune_to_sco(candidate_set, consumer=config.consumer)

    driver.close()
    return sco.model_dump(mode="json")


def run_query(config: QueryConfig) -> dict[str, Any]:
    """Run retrieval + consumer dispatch."""
    from sema.consumers import resolve_consumer
    from sema.consumers.base import ConsumerDeps, ConsumerRequest
    from sema.runtimes.databricks import DatabricksRuntime

    consumer = resolve_consumer(config.consumer)
    if config.operation not in consumer.capabilities:
        raise ValueError(
            f"Operation {config.operation!r} not supported by "
            f"{consumer.name!r}. "
            f"Supported: {sorted(consumer.capabilities)}"
        )

    llm = _get_llm(config.llm)
    sco = _retrieve_context(config)

    if config.verbose:
        click.echo(
            f"SCO: {len(sco.entities)} entities, "
            f"{len(sco.physical_assets)} tables, "
            f"{len(sco.join_paths)} joins"
        )

    needs_runtime = config.operation in ("explain", "execute")
    runtime = (
        DatabricksRuntime(config.databricks)
        if needs_runtime else None
    )

    try:
        request = ConsumerRequest(
            question=config.question,
            operation=config.operation,
        )
        deps = ConsumerDeps(llm=llm, sql_runtime=runtime)
        result = consumer.run(request, sco, deps)
    finally:
        if runtime is not None:
            runtime.close()

    response: dict[str, Any] = {
        "operation": config.operation,
        "sql": result.artifact,
        "validation": {
            "valid": result.valid,
            "errors": result.errors,
        },
    }

    if result.data:
        response.update(result.data)
    if result.summary:
        response["summary"] = result.summary
    if config.verbose:
        response["sco"] = sco.model_dump(mode="json")

    return response
