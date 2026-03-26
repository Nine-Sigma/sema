from __future__ import annotations

from typing import Any

import click

from sema.cli_factories import (
    _get_embedder,
    _get_llm,
    _get_neo4j_driver,
)
from sema.models.config import (
    BuildConfig,
    QueryConfig,
)
from sema.pipeline.orchestrate_utils import (
    _collect_results,
    _compute_embeddings,
    _discover_tables,
    _execute_by_mode,
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

    results = _spawn_workers(
        work_items, config, connector_factory, llm_factory,
        loader, run_id,
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
    except Exception:
        embedder = None

    engine = RetrievalEngine(driver=driver, embedder=embedder)
    candidate_set = engine.retrieve(config.question)
    sco = prune_to_sco(candidate_set, consumer_hint=config.consumer_hint)

    driver.close()
    return sco.model_dump(mode="json")


def run_query(config: QueryConfig) -> dict[str, Any]:
    """Run retrieval + NL2SQL consumer."""
    from sema.pipeline.sql_gen import SQLGenerator

    llm = _get_llm(config.llm)

    sco = _retrieve_context(config)

    if config.verbose:
        click.echo(f"SCO: {len(sco.entities)} entities, "
                    f"{len(sco.physical_assets)} tables, "
                    f"{len(sco.join_paths)} joins")

    gen = SQLGenerator(llm=llm)
    result = gen.generate(sco, config.question)

    response: dict[str, Any] = {
        "mode": config.mode.value,
        "sql": result.get("sql", ""),
        "validation": {
            "valid": result.get("valid", False),
            "errors": result.get("errors", []),
        },
    }

    if config.verbose:
        response["sco"] = sco.model_dump(mode="json")

    _execute_by_mode(config, result, response, llm)

    return response
