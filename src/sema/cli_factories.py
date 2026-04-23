from __future__ import annotations

from typing import Any

from sema.models.config import (
    EmbeddingConfig,
    LLMConfig,
    Neo4jConfig,
)

DATABRICKS_UNSUPPORTED_ENDPOINT_SUBSTRINGS: tuple[str, ...] = (
    "gpt-oss-",
    "-codex",
)


class DatabricksProviderAuthError(RuntimeError):
    """Raised when Databricks SDK credential resolution fails at factory time.

    Carries a narrow provider-specific type so retrieval-path `except Exception`
    catches can re-raise this while still silencing unrelated degradations.
    """


def _get_neo4j_driver(neo4j_config: Neo4jConfig) -> Any:
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        neo4j_config.uri,
        auth=(neo4j_config.user, neo4j_config.password.get_secret_value()),
    )


def _reject_unsupported_databricks_endpoint(model: str) -> None:
    for substring in DATABRICKS_UNSUPPORTED_ENDPOINT_SUBSTRINGS:
        if substring in model:
            raise ValueError(
                f"Databricks endpoint '{model}' is not supported by sema: "
                f"endpoints matching '{substring}' use response shapes "
                "(reasoning-block content or the OpenAI Responses API) that "
                "LLMClient cannot currently consume. Pick a chat-completions "
                "endpoint (e.g., databricks-llama-4-maverick, "
                "databricks-gemma-3-12b)."
            )


def _force_databricks_auth() -> None:
    """Force Databricks SDK credential resolution at factory-construction time.

    Without this, `ChatDatabricks` resolves auth lazily at first `invoke`,
    and `LLMClient._probe_structured_output` catches Exception — a deferred
    auth failure would be swallowed into "structured output not supported"
    instead of surfacing.
    """
    from databricks.sdk import WorkspaceClient
    try:
        WorkspaceClient().current_user.me()
    except Exception as exc:
        raise DatabricksProviderAuthError(
            "Databricks SDK could not resolve credentials. Set "
            "DATABRICKS_HOST and DATABRICKS_TOKEN, or configure "
            "DATABRICKS_CONFIG_PROFILE to select a ~/.databrickscfg profile. "
            "See the Databricks SDK default auth chain for other options "
            f"(OAuth, service principal). Underlying error: {exc}"
        ) from exc


def _build_databricks_llm(llm_config: LLMConfig) -> Any:
    _reject_unsupported_databricks_endpoint(llm_config.model)
    _force_databricks_auth()
    from databricks_langchain import ChatDatabricks
    return ChatDatabricks(endpoint=llm_config.model)


def _build_databricks_embedder(embedding_config: EmbeddingConfig) -> Any:
    _force_databricks_auth()
    from databricks_langchain import DatabricksEmbeddings
    return DatabricksEmbeddings(endpoint=embedding_config.model)


def _get_llm(llm_config: LLMConfig) -> Any:
    provider = llm_config.provider.lower()
    if provider == "databricks":
        return _build_databricks_llm(llm_config)

    api_key = llm_config.api_key.get_secret_value()
    timeout = llm_config.request_timeout

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            api_key=api_key,  # type: ignore[arg-type]
            base_url="https://openrouter.ai/api/v1",
            request_timeout=timeout,  # type: ignore[call-arg]
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=llm_config.model,
            api_key=api_key,  # type: ignore[call-arg, arg-type]
            timeout=float(timeout),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            api_key=api_key,  # type: ignore[arg-type]
            request_timeout=timeout,  # type: ignore[call-arg]
        )
    if provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=llm_config.base_url,
            request_timeout=timeout,  # type: ignore[call-arg]
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


def _get_embedder(embedding_config: EmbeddingConfig) -> Any:
    provider = embedding_config.provider.lower()

    if provider == "databricks":
        return _build_databricks_embedder(embedding_config)
    if provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(embedding_config.model)
    if provider in ("openrouter", "openai", "custom"):
        from langchain_openai import OpenAIEmbeddings
        base_url = embedding_config.base_url
        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        return OpenAIEmbeddings(
            model=embedding_config.model,
            api_key=embedding_config.api_key.get_secret_value(),  # type: ignore[arg-type]
            base_url=base_url,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")
