import pytest

from showcase.ontology_query_benchmark import routing
from showcase.ontology_query_benchmark.routing import RoutingError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "host",
    ["https://partner-workspace.cloud.databricks.com", "partner-workspace.cloud.databricks.com/"],
)
def test_test_host_is_accepted(host: str) -> None:
    assert routing.require_test_host(host) == routing.TEST_HOST


@pytest.mark.parametrize(
    "host",
    [
        "https://partner-workspace-prod.cloud.databricks.com",
        "https://dbc-6395977d-5b1c.cloud.databricks.com",
        "http://partner-workspace.cloud.databricks.com",
        "",
    ],
)
def test_any_other_host_is_refused(host: str) -> None:
    with pytest.raises(RoutingError, match="host"):
        routing.require_test_host(host)


def test_only_pinned_chat_and_embedding_endpoints_pass() -> None:
    assert routing.require_chat_endpoint("databricks-claude-sonnet-5")
    assert routing.require_embedding_endpoint("databricks-bge-large-en")
    with pytest.raises(RoutingError, match="chat endpoint"):
        routing.require_chat_endpoint("databricks-gpt-5-4")
    with pytest.raises(RoutingError, match="embedding endpoint"):
        routing.require_embedding_endpoint("text-embedding-3-small")


@pytest.mark.parametrize("tool", ["execute_sql", "describe_knowledge_graph"])
def test_allowed_mcp_tools_pass(tool: str) -> None:
    assert routing.require_mcp_tool(tool) == tool


@pytest.mark.parametrize(
    "tool", ["query_data", "ask_question", "generate_sql", "identify_concept", ""]
)
def test_timbr_side_llm_tools_are_refused(tool: str) -> None:
    with pytest.raises(RoutingError, match="MCP tool"):
        routing.require_mcp_tool(tool)


def test_timbr_url_must_be_the_pinned_deployment() -> None:
    assert routing.require_timbr_mcp_url("https://timbr.partner.example/timbr/api/mcp")
    with pytest.raises(RoutingError, match="Timbr"):
        routing.require_timbr_mcp_url("https://timbr.example.com/timbr/api/mcp")


def test_clean_process_passes() -> None:
    environ = {"HOME": "/x", "DATABRICKS_CONFIG_PROFILE": "TEST", "LLM_PROVIDER": "databricks"}
    routing.require_clean_process(environ, modules=("json", "requests"))


def test_loaded_dotenv_is_refused() -> None:
    with pytest.raises(RoutingError, match="dotenv"):
        routing.require_clean_process({}, modules=("json", "dotenv"))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENROUTER_API_KEY", "sk-or-x"),
        ("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ("OPENAI_API_KEY", "sk-x"),
        ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ("LLM_BASE_URL", "https://gateway.example.com"),
        ("LLM_PROVIDER", "openrouter"),
        ("INTAKE_MODEL_PROVIDER", "custom"),
        ("EMBEDDING_PROVIDER", "openai"),
        ("DATABRICKS_HOST", "https://dbc-6395977d-5b1c.cloud.databricks.com"),
        ("DATABRICKS_CONFIG_PROFILE", "DEFAULT"),
    ],
)
def test_settings_that_route_elsewhere_are_refused(name: str, value: str) -> None:
    with pytest.raises(RoutingError, match=name):
        routing.require_clean_process({name: value}, modules=())
