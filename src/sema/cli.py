from __future__ import annotations

import json
import sys
from typing import Any

import click
from pydantic import SecretStr

from sema.models.config import (
    BuildConfig,
    EmbeddingConfig,
    LLMConfig,
    Neo4jConfig,
    QueryConfig,
)
from sema.pipeline.orchestrate import (
    run_build,
    run_context,
    run_query,
)


@click.group()
def cli() -> None:
    """GraphRAG Semantic Ontology — Knowledge graph and semantic retrieval platform."""
    import logging
    from dotenv import load_dotenv
    load_dotenv()

    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            from sema.log import logger
            level: str | int = logger.level(record.levelname).name if record.levelname in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") else record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("databricks.sql").setLevel(logging.WARNING)


def _build_config_from_args(
    *,
    source: str | None,
    catalog: str | None,
    schemas: str | None,
    table_pattern: str | None,
    domain: str | None,
    table_workers: int | None,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    llm_timeout: int | None,
    config_file: str | None,
    skip_embeddings: bool,
    resume: bool,
    verbose: bool,
) -> BuildConfig:
    """Assemble a :class:`BuildConfig` from CLI arguments."""
    overrides: dict[str, Any] = {}
    if source is not None:
        overrides["source"] = source
    if catalog is not None:
        overrides["catalog"] = catalog
    if schemas is not None:
        overrides["schemas"] = [s.strip() for s in schemas.split(",")]
    if table_pattern is not None:
        overrides["table_pattern"] = table_pattern
    if domain is not None:
        overrides["domain"] = domain
        overrides["domain_from_cli"] = True
    if table_workers is not None:
        overrides["table_workers"] = table_workers
    if skip_embeddings:
        overrides["skip_embeddings"] = True
    if resume:
        overrides["resume"] = True
    if verbose:
        overrides["verbose"] = True

    if config_file:
        build_config = BuildConfig.from_file(config_file, overrides=overrides)
    else:
        build_config = BuildConfig(**overrides)

    if neo4j_uri is not None:
        build_config.neo4j = Neo4jConfig(
            uri=neo4j_uri,
            user=neo4j_user or build_config.neo4j.user,
            password=neo4j_password or build_config.neo4j.password,  # type: ignore[arg-type]
        )

    llm_overrides: dict[str, Any] = {}
    if llm_provider is not None:
        llm_overrides["provider"] = llm_provider
    if llm_model is not None:
        llm_overrides["model"] = llm_model
    if llm_timeout is not None:
        llm_overrides["request_timeout"] = llm_timeout
    if llm_overrides:
        build_config.llm = build_config.llm.model_copy(
            update=llm_overrides,
        )

    return build_config


@cli.command()
@click.option("--source", default="databricks", help="Data source connector type")
@click.option("--catalog", default=None, help="Catalog name to extract from")
@click.option("--schemas", default=None, help="Comma-separated schema names")
@click.option("--table-pattern", default=None, help="Glob pattern to filter table names")
@click.option("--domain", default=None, help="Warehouse domain hint (e.g. healthcare, financial)")
@click.option("--table-workers", default=None, type=int, help="Parallel table workers (default 1)")
@click.option("--neo4j-uri", default=None, help="Neo4j bolt URI")
@click.option("--neo4j-user", default=None, help="Neo4j username")
@click.option("--neo4j-password", default=None, help="Neo4j password")
@click.option("--llm-provider", default=None, help="LLM provider (anthropic, openai)")
@click.option("--llm-model", default=None, help="LLM model name")
@click.option("--llm-timeout", default=None, type=int, help="LLM request timeout in seconds (default: 120)")
@click.option("--config", "config_file", default=None, help="Path to config YAML file")
@click.option("--skip-embeddings", is_flag=True, default=False, help="Create indexes only, skip embedding computation")
@click.option("--resume", is_flag=True, default=False, help="Skip tables that already have assertions in the graph")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output")
def build(
    source: str | None,
    catalog: str | None,
    schemas: str | None,
    table_pattern: str | None,
    domain: str | None,
    table_workers: int | None,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    llm_timeout: int | None,
    config_file: str | None,
    skip_embeddings: bool,
    resume: bool,
    verbose: bool,
) -> None:
    """Build the knowledge graph from a data source."""
    build_config = _build_config_from_args(
        source=source, catalog=catalog, schemas=schemas,
        table_pattern=table_pattern, domain=domain,
        table_workers=table_workers,
        neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
        neo4j_password=neo4j_password, llm_provider=llm_provider,
        llm_model=llm_model, llm_timeout=llm_timeout,
        config_file=config_file,
        skip_embeddings=skip_embeddings, resume=resume, verbose=verbose,
    )
    try:
        report = run_build(build_config)
        click.echo("\nBuild Report")
        click.echo("=" * 40)
        for key, value in report.items():
            label = key.replace("_", " ").title()
            if key == "failed_tables" and isinstance(value, list):
                if value:
                    click.echo(f"  {label}:")
                    for ft in value:
                        click.echo(
                            f"    {ft['table']} — "
                            f"{ft['stage']}: {ft['error']}"
                        )
            elif isinstance(value, dict):
                click.echo(f"  {label}:")
                for k, v in value.items():
                    click.echo(f"    {k}: {v}")
            else:
                click.echo(f"  {label}: {value}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--question", required=True, help="Natural language question")
@click.option("--neo4j-uri", default=None, help="Neo4j bolt URI")
@click.option("--neo4j-user", default=None, help="Neo4j username")
@click.option("--neo4j-password", default=None, help="Neo4j password")
@click.option("--consumer", default="nl2sql", help="Consumer for SCO pruning")
@click.option("--embedding-provider", default=None, help="Embedding provider")
@click.option("--embedding-model", default=None, help="Embedding model name")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output")
def context(
    question: str,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
    consumer: str,
    embedding_provider: str | None,
    embedding_model: str | None,
    verbose: bool,
) -> None:
    """Produce a Semantic Context Object for a question."""
    query_config = QueryConfig(
        question=question,
        consumer=consumer,
        verbose=verbose,
    )

    if neo4j_uri is not None:
        query_config.neo4j = Neo4jConfig(
            uri=neo4j_uri,
            user=neo4j_user or query_config.neo4j.user,
            password=neo4j_password or query_config.neo4j.password,  # type: ignore[arg-type]
        )

    if embedding_provider is not None or embedding_model is not None:
        query_config.embedding = EmbeddingConfig(
            provider=embedding_provider or query_config.embedding.provider,
            model=embedding_model or query_config.embedding.model,
            api_key=query_config.embedding.api_key,
        )

    try:
        sco = run_context(query_config)
        click.echo(json.dumps(sco, indent=2, default=str))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--threshold", default=0.85, type=float, help="Confidence threshold for review")
@click.option("--neo4j-uri", default=None, help="Neo4j bolt URI")
@click.option("--neo4j-user", default=None, help="Neo4j username")
@click.option("--neo4j-password", default=None, help="Neo4j password")
@click.option("--output", default=None, help="Output file path (default: stdout)")
def review(
    threshold: float,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
    output: str | None,
) -> None:
    """Export low-confidence assertions for human review."""
    from sema.models.config import Neo4jConfig

    neo4j = Neo4jConfig(
        uri=neo4j_uri or "bolt://localhost:7687",
        user=neo4j_user or "neo4j",
        password=SecretStr(neo4j_password or "password"),
    )

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            neo4j.uri, auth=(neo4j.user, neo4j.password.get_secret_value()),
        )
        with driver.session() as session:
            result = session.run(
                "MATCH (a:Assertion) "
                "WHERE a.confidence < $threshold "
                "RETURN a.id AS id, a.subject_ref AS subject_ref, "
                "a.predicate AS predicate, a.payload AS payload, "
                "a.source AS source, a.confidence AS confidence "
                "ORDER BY a.confidence ASC",
                threshold=threshold,
            )
            assertions = [dict(r) for r in result]
        driver.close()

        # Group by subject_ref column
        grouped: dict[str, list[dict[str, Any]]] = {}
        for a in assertions:
            ref = a.get("subject_ref", "unknown")
            grouped.setdefault(ref, []).append(a)

        review_data = {
            "threshold": threshold,
            "total_assertions": len(assertions),
            "columns": grouped,
        }

        output_str = json.dumps(review_data, indent=2, default=str)
        if output:
            with open(output, "w") as f:
                f.write(output_str)
            click.echo(f"Exported {len(assertions)} assertions to {output}")
        else:
            click.echo(output_str)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--question", required=True, help="Natural language question")
@click.option("--consumer", default="nl2sql", help="Consumer name")
@click.option("--operation", default=None, help="Operation (e.g. plan, explain, execute)")
@click.option("--mode", default=None, help="Alias for --operation")
@click.option("--neo4j-uri", default=None, help="Neo4j bolt URI")
@click.option("--neo4j-user", default=None, help="Neo4j username")
@click.option("--neo4j-password", default=None, help="Neo4j password")
@click.option("--databricks-host", default=None, help="Databricks host")
@click.option("--databricks-token", default=None, help="Databricks token")
@click.option("--databricks-http-path", default=None, help="Databricks HTTP path")
@click.option("--llm-provider", default=None, help="LLM provider")
@click.option("--llm-model", default=None, help="LLM model name")
@click.option("--llm-timeout", default=None, type=int, help="LLM request timeout in seconds (default: 120)")
@click.option("--embedding-provider", default=None, help="Embedding provider")
@click.option("--embedding-model", default=None, help="Embedding model name")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output")
def query(
    question: str,
    consumer: str,
    operation: str | None,
    mode: str | None,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
    databricks_host: str | None,
    databricks_token: str | None,
    databricks_http_path: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    llm_timeout: int | None,
    embedding_provider: str | None,
    embedding_model: str | None,
    verbose: bool,
) -> None:
    """Query the knowledge graph with natural language."""
    resolved_operation = operation or mode or "plan"
    query_config = QueryConfig(
        question=question,
        consumer=consumer,
        operation=resolved_operation,
        verbose=verbose,
    )

    if neo4j_uri is not None:
        query_config.neo4j = Neo4jConfig(
            uri=neo4j_uri,
            user=neo4j_user or query_config.neo4j.user,
            password=neo4j_password or query_config.neo4j.password,  # type: ignore[arg-type]
        )

    llm_overrides: dict[str, Any] = {}
    if llm_provider is not None:
        llm_overrides["provider"] = llm_provider
    if llm_model is not None:
        llm_overrides["model"] = llm_model
    if llm_timeout is not None:
        llm_overrides["request_timeout"] = llm_timeout
    if llm_overrides:
        query_config.llm = query_config.llm.model_copy(
            update=llm_overrides,
        )

    if embedding_provider is not None or embedding_model is not None:
        query_config.embedding = EmbeddingConfig(
            provider=embedding_provider or query_config.embedding.provider,
            model=embedding_model or query_config.embedding.model,
            api_key=query_config.embedding.api_key,
        )

    try:
        result = run_query(query_config)
        if verbose:
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            if "sql" in result:
                click.echo(result["sql"])
            if "results" in result:
                click.echo(json.dumps(result["results"], indent=2, default=str))
            if "explain" in result:
                click.echo(result["explain"])
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
