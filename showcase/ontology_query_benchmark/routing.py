"""Pinned partner TEST routing for the benchmark (CONTRACT.md) and fail-closed checks.

Every check raises RoutingError instead of falling back to another host, provider or tool.
"""

from collections.abc import Collection, Mapping

TEST_PROFILE = "TEST"
TEST_HOST = "https://partner-workspace.cloud.databricks.com"
SQL_WAREHOUSE_ID = "feedface00000000"
CHAT_ENDPOINTS = frozenset({"databricks-claude-sonnet-5"})
EMBEDDING_ENDPOINTS = frozenset({"databricks-bge-large-en"})
TIMBR_URL = "https://timbr.partner.example"
TIMBR_MCP_URL = f"{TIMBR_URL}/timbr/api/mcp"
TIMBR_ONTOLOGY = "cancer_plus_demographics_v2"
B_MCP_TOOLS = frozenset({"execute_sql", "describe_knowledge_graph"})
FORBIDDEN_ENV = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL", "LLM_BASE_URL", "LLM_API_KEY", "INTAKE_MODEL_BASE_URL",
    "INTAKE_MODEL_API_KEY",
)  # fmt: skip
FORBIDDEN_ENV_PREFIXES = ("OPENROUTER_",)
PROVIDER_ENV = ("LLM_PROVIDER", "INTAKE_MODEL_PROVIDER", "EMBEDDING_PROVIDER")
ALLOWED_PROVIDERS = frozenset({"", "databricks"})


class RoutingError(RuntimeError):
    pass


def require_test_host(host: str) -> str:
    text = (host or "").strip().rstrip("/").lower()
    normalized = text if "://" in text else f"https://{text}"
    if normalized != TEST_HOST:
        raise RoutingError(f"resolved host {host!r} is not the TEST host {TEST_HOST}")
    return TEST_HOST


def require_chat_endpoint(name: str) -> str:
    if name not in CHAT_ENDPOINTS:
        raise RoutingError(f"chat endpoint {name!r} is not pinned")
    return name


def require_embedding_endpoint(name: str) -> str:
    if name not in EMBEDDING_ENDPOINTS:
        raise RoutingError(f"embedding endpoint {name!r} is not pinned")
    return name


def require_mcp_tool(name: str) -> str:
    if name not in B_MCP_TOOLS:
        raise RoutingError(f"MCP tool {name!r} is not allowed for arm B")
    return name


def require_timbr_mcp_url(url: str) -> str:
    if url.rstrip("/") != TIMBR_MCP_URL:
        raise RoutingError(f"Timbr MCP url {url!r} is not {TIMBR_MCP_URL}")
    return url


def require_clean_process(environ: Mapping[str, str], modules: Collection[str]) -> None:
    """Refuse a loaded dotenv module or any setting that would route a call elsewhere."""
    if "dotenv" in modules:
        raise RoutingError("python-dotenv is loaded; the benchmark never reads a .env file")
    for name, value in environ.items():
        if name in FORBIDDEN_ENV or name.startswith(FORBIDDEN_ENV_PREFIXES):
            raise RoutingError(f"{name} is set; only partner TEST serving is allowed")
        if name in PROVIDER_ENV and value.strip().lower() not in ALLOWED_PROVIDERS:
            raise RoutingError(f"{name}={value!r} routes outside Databricks")
    host = environ.get("DATABRICKS_HOST")
    if host is not None:
        try:
            require_test_host(host)
        except RoutingError as exc:
            raise RoutingError(f"DATABRICKS_HOST: {exc}") from exc
    profile = environ.get("DATABRICKS_CONFIG_PROFILE")
    if profile is not None and profile != TEST_PROFILE:
        raise RoutingError(f"DATABRICKS_CONFIG_PROFILE={profile!r} is not {TEST_PROFILE}")
