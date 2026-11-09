import pytest

from showcase.ontology_query_benchmark import routing
from showcase.ontology_query_benchmark.profile import Profile
from showcase.ontology_query_benchmark.routing import RoutingError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("host", ["https://toy.example.test", "toy.example.test/"])
def test_pinned_host_is_accepted(profile: Profile, host: str) -> None:
    assert routing.require_host(host, profile.routing) == profile.routing.host


@pytest.mark.parametrize(
    "host",
    ["https://other.example.test", "http://toy.example.test", "https://toy.example.test.evil", ""],
)
def test_any_other_host_is_refused(profile: Profile, host: str) -> None:
    with pytest.raises(RoutingError, match="host"):
        routing.require_host(host, profile.routing)


def test_only_pinned_chat_and_embedding_endpoints_pass(profile: Profile) -> None:
    assert routing.require_chat_endpoint("toy-chat", profile.routing)
    assert routing.require_embedding_endpoint("toy-embed", profile.routing)
    with pytest.raises(RoutingError, match="chat endpoint"):
        routing.require_chat_endpoint("other-chat", profile.routing)
    with pytest.raises(RoutingError, match="embedding endpoint"):
        routing.require_embedding_endpoint("text-embedding-3-small", profile.routing)


@pytest.mark.parametrize("tool", ["execute_sql", "describe_knowledge_graph"])
def test_allowed_mcp_tools_pass(profile: Profile, tool: str) -> None:
    assert routing.require_mcp_tool(tool, profile.timbr) == tool


@pytest.mark.parametrize(
    "tool", ["query_data", "ask_question", "generate_sql", "identify_concept", ""]
)
def test_engine_side_llm_tools_are_refused(profile: Profile, tool: str) -> None:
    with pytest.raises(RoutingError, match="MCP tool"):
        routing.require_mcp_tool(tool, profile.timbr)


def test_mcp_url_must_be_the_pinned_deployment(profile: Profile) -> None:
    assert routing.require_timbr_mcp_url("https://ontology.example.test/api/mcp", profile.timbr)
    with pytest.raises(RoutingError, match="Timbr"):
        routing.require_timbr_mcp_url("https://timbr.example.com/api/mcp", profile.timbr)


def test_clean_process_passes(profile: Profile) -> None:
    environ = {"HOME": "/x", "DATABRICKS_CONFIG_PROFILE": "TOY", "LLM_PROVIDER": "databricks"}
    routing.require_clean_process(environ, ("json", "requests"), profile.routing)


def test_loaded_dotenv_is_refused(profile: Profile) -> None:
    with pytest.raises(RoutingError, match="dotenv"):
        routing.require_clean_process({}, ("json", "dotenv"), profile.routing)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENROUTER_API_KEY", "sk-or-x"),
        ("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ("OPENAI_API_KEY", "sk-x"),
        ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ("LLM_BASE_URL", "https://gateway.example.com"),
        ("EMBEDDING_BASE_URL", "https://gateway.example.com"),
        ("EMBEDDING_API_KEY", "sk-x"),
        ("LLM_PROVIDER", "openrouter"),
        ("INTAKE_MODEL_PROVIDER", "custom"),
        ("EMBEDDING_PROVIDER", "openai"),
        ("DATABRICKS_HOST", "https://personal.example.test"),
        ("DATABRICKS_CONFIG_PROFILE", "DEFAULT"),
    ],
)
def test_settings_that_route_elsewhere_are_refused(profile: Profile, name: str, value: str) -> None:
    with pytest.raises(RoutingError, match=name):
        routing.require_clean_process({name: value}, (), profile.routing)
