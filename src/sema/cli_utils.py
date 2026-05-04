"""Helpers for CLI command implementations.

Extracted from ``cli.py`` to keep that module under the size budget.
"""
from __future__ import annotations

from typing import Any

from sema.models.config import BuildConfig, Neo4jConfig


def build_config_from_args(
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
    enable_fk_detection: bool = True,
    materialize_structural_fk: bool = False,
) -> BuildConfig:
    """Assemble a :class:`BuildConfig` from CLI arguments."""
    overrides = _gather_overrides(
        source=source, catalog=catalog, schemas=schemas,
        table_pattern=table_pattern, domain=domain,
        table_workers=table_workers,
        skip_embeddings=skip_embeddings, resume=resume, verbose=verbose,
        enable_fk_detection=enable_fk_detection,
        materialize_structural_fk=materialize_structural_fk,
    )

    if config_file:
        build_config = BuildConfig.from_file(
            config_file, overrides=overrides,
        )
    else:
        build_config = BuildConfig(**overrides)

    if neo4j_uri is not None:
        build_config.neo4j = Neo4jConfig(
            uri=neo4j_uri,
            user=neo4j_user or build_config.neo4j.user,
            password=neo4j_password or build_config.neo4j.password,  # type: ignore[arg-type]
        )

    _apply_llm_overrides(
        build_config,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
    )
    return build_config


def _gather_overrides(
    *,
    source: str | None,
    catalog: str | None,
    schemas: str | None,
    table_pattern: str | None,
    domain: str | None,
    table_workers: int | None,
    skip_embeddings: bool,
    resume: bool,
    verbose: bool,
    enable_fk_detection: bool,
    materialize_structural_fk: bool,
) -> dict[str, Any]:
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
    overrides["enable_fk_detection"] = enable_fk_detection
    overrides["materialize_structural_fk"] = materialize_structural_fk
    return overrides


def _apply_llm_overrides(
    build_config: BuildConfig,
    *,
    llm_provider: str | None,
    llm_model: str | None,
    llm_timeout: int | None,
) -> None:
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
