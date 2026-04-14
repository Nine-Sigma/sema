from __future__ import annotations

from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabricksConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATABRICKS_")

    host: str = ""
    token: SecretStr = SecretStr("")
    http_path: str = ""


class Neo4jConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("graphrag")


class LLMConfig(BaseSettings):
    """LLM provider configuration.

    Supported providers:
        openrouter  - OpenRouter API (default, routes to any model)
        anthropic   - Direct Anthropic API
        openai      - Direct OpenAI API
        databricks  - Mosaic AI / Databricks Model Serving
        custom      - Any OpenAI-compatible endpoint (set base_url)
    """

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-4"
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    use_structured_output: str = "auto"
    request_timeout: int = 120


class EmbeddingConfig(BaseSettings):
    """Embedding provider configuration.

    Supported providers:
        openrouter          - OpenRouter API (default)
        openai              - Direct OpenAI API
        sentence-transformers - Local sentence-transformers model
        databricks          - Mosaic AI / Databricks Model Serving
        custom              - Any OpenAI-compatible endpoint (set base_url)
    """

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: str = "openrouter"
    model: str = "google/gemini-embedding-001"
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    embeddable_labels: list[str] = [
        "Entity", "Property", "Alias", "Term", "Metric",
    ]


class ProfilingConfig(BaseSettings):
    categorical_threshold: int = 500
    top_k_values: int = 100
    sample_rows: int = 5
    max_sample_values: int = 5
    skip_temporal_profiling: bool = True
    skip_numeric_profiling: bool = False


class BuildConfig(BaseSettings):
    source: str = "databricks"
    catalog: str = ""
    schemas: list[str] = []
    table_pattern: str | None = None
    domain: str | None = None
    domain_from_cli: bool = False
    verbose: bool = False
    skip_embeddings: bool = False
    resume: bool = False

    use_staged: bool = True
    enable_domain_bias: bool = True
    enable_type_inventory: bool = True
    enable_vocab_hints: bool = True
    enable_few_shot: bool = True
    enable_stage_c: bool = True

    table_workers: int = 4
    vocab_workers: int = 8
    column_batch_size: int = 25
    retry_max_attempts: int = 3

    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    circuit_breaker_success_threshold: int = 2

    databricks: DatabricksConfig = Field(default_factory=DatabricksConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)

    @classmethod
    def from_file(cls, path: str, overrides: dict[str, Any] | None = None) -> BuildConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class QueryConfig(BaseSettings):
    question: str
    consumer: str = "nl2sql"
    operation: str = "plan"
    verbose: bool = False
    session_id: str | None = None

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    databricks: DatabricksConfig = Field(default_factory=DatabricksConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
