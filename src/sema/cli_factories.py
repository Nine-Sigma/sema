from __future__ import annotations

from typing import Any

from sema.models.config import (
    EmbeddingConfig,
    LLMConfig,
    Neo4jConfig,
)


def _get_neo4j_driver(neo4j_config: Neo4jConfig) -> Any:
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        neo4j_config.uri,
        auth=(neo4j_config.user, neo4j_config.password.get_secret_value()),
    )


def _get_llm(llm_config: LLMConfig) -> Any:
    provider = llm_config.provider.lower()
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
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=llm_config.model,
            api_key=api_key,  # type: ignore[call-arg, arg-type]
            timeout=float(timeout),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            api_key=api_key,  # type: ignore[arg-type]
            request_timeout=timeout,  # type: ignore[call-arg]
        )
    elif provider in ("databricks", "custom"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=llm_config.base_url,
            request_timeout=timeout,  # type: ignore[call-arg]
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _get_embedder(embedding_config: EmbeddingConfig) -> Any:
    provider = embedding_config.provider.lower()

    if provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(embedding_config.model)
    elif provider in ("openrouter", "openai", "databricks", "custom"):
        from langchain_openai import OpenAIEmbeddings
        base_url = embedding_config.base_url
        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        return OpenAIEmbeddings(
            model=embedding_config.model,
            api_key=embedding_config.api_key.get_secret_value(),  # type: ignore[arg-type]
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
