"""Fail-closed checks against a profile's pinned routing.

Every check raises RoutingError instead of falling back to another host, provider or tool.
"""

import os
from collections.abc import Collection, Mapping

from showcase.ontology_query_benchmark.profile import Routing, Timbr

FORBIDDEN_ENV = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL", "LLM_BASE_URL", "LLM_API_KEY", "INTAKE_MODEL_BASE_URL",
    "INTAKE_MODEL_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "MLFLOW_TRACKING_URI",
)  # fmt: skip
FORBIDDEN_ENV_PREFIXES = ("OPENROUTER_",)
PROVIDER_ENV = ("LLM_PROVIDER", "INTAKE_MODEL_PROVIDER", "EMBEDDING_PROVIDER")
DATABRICKS = "databricks"
ALLOWED_PROVIDERS = frozenset({"", DATABRICKS})
PROVIDER_KEY_ENV = ("LLM_API_KEY", "INTAKE_MODEL_API_KEY", "EMBEDDING_API_KEY")
# The run rules' provider-free setting: it points the Databricks SDK at no config at all.
INERT_DATABRICKS_ENV = ("DATABRICKS_CONFIG_FILE", os.devnull)


class RoutingError(RuntimeError):
    pass


def require_host(host: str, routing: Routing) -> str:
    text = (host or "").strip().rstrip("/").lower()
    normalized = text if "://" in text else f"https://{text}"
    if normalized != routing.host:
        raise RoutingError(f"resolved host {host!r} is not the pinned host {routing.host}")
    return routing.host


def require_chat_endpoint(name: str, routing: Routing) -> str:
    if name != routing.chat_endpoint:
        raise RoutingError(f"chat endpoint {name!r} is not pinned")
    return name


def require_embedding_endpoint(name: str, routing: Routing) -> str:
    if name != routing.embedding_endpoint:
        raise RoutingError(f"embedding endpoint {name!r} is not pinned")
    return name


def require_mcp_tool(name: str, timbr: Timbr) -> str:
    if name not in timbr.allowed_tools:
        raise RoutingError(f"MCP tool {name!r} is not allowed for arm B")
    return name


def require_timbr_mcp_url(url: str, timbr: Timbr) -> str:
    if url.rstrip("/") != timbr.mcp_url:
        raise RoutingError(f"Timbr MCP url {url!r} is not {timbr.mcp_url}")
    return url


def require_clean_process(
    environ: Mapping[str, str], modules: Collection[str], routing: Routing | None,
    provider: str = DATABRICKS,
) -> None:
    """Refuse a loaded dotenv module or any setting that would route a call away from the
    profile's provider; Databricks settings are checked against the pinned routing."""
    if "dotenv" in modules:
        raise RoutingError("python-dotenv is loaded; the benchmark never reads a .env file")
    if provider != DATABRICKS:
        _require_provider_only(environ, provider)
        return
    if routing is None:
        raise RoutingError("a Databricks run needs the profile's pinned routing")
    for name, value in environ.items():
        if name in FORBIDDEN_ENV or name.startswith(FORBIDDEN_ENV_PREFIXES):
            raise RoutingError(f"{name} is set; only the pinned serving host is allowed")
        if name in PROVIDER_ENV and value.strip().lower() not in ALLOWED_PROVIDERS:
            raise RoutingError(f"{name}={value!r} routes outside Databricks")
    host = environ.get("DATABRICKS_HOST")
    if host is not None:
        try:
            require_host(host, routing)
        except RoutingError as exc:
            raise RoutingError(f"DATABRICKS_HOST: {exc}") from exc
    profile = environ.get("DATABRICKS_CONFIG_PROFILE")
    if profile is not None and profile != routing.databricks_profile:
        raise RoutingError(
            f"DATABRICKS_CONFIG_PROFILE={profile!r} is not {routing.databricks_profile}"
        )


def _require_provider_only(environ: Mapping[str, str], provider: str) -> None:
    for name, value in environ.items():
        # The pinned provider's own key is allowed; a base URL or another vendor's key is not.
        foreign = name in FORBIDDEN_ENV and name not in PROVIDER_KEY_ENV
        databricks = name.startswith("DATABRICKS_") and (name, value) != INERT_DATABRICKS_ENV
        if foreign or databricks:
            raise RoutingError(f"{name} is set; only provider {provider} is allowed")
        if name in PROVIDER_ENV and value.strip().lower() not in {"", provider}:
            raise RoutingError(f"{name}={value!r} is not the profile's provider {provider}")


def intake_env(environ: Mapping[str, str], routing: Routing) -> dict[str, str]:
    """The variables that pin SEMA intake's model and embedding calls to the profile."""
    embedder_off = environ.get("EMBEDDING_PROVIDER", "databricks").strip() == ""
    return {
        "PYTHON_DOTENV_DISABLED": "1",
        "DATABRICKS_HOST": routing.host,
        "DATABRICKS_CONFIG_PROFILE": routing.databricks_profile,
        "INTAKE_MODEL_PROVIDER": "databricks",
        "INTAKE_MODEL_MODEL": routing.chat_endpoint,
        # An explicit empty provider is the zero-call replay's embedder off switch.
        "EMBEDDING_PROVIDER": "" if embedder_off else "databricks",
        "EMBEDDING_MODEL": routing.embedding_endpoint,
        "INTAKE_TRACING_BACKEND": "none",
    }
