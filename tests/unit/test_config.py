import pytest

pytestmark = pytest.mark.unit

from sema.models.config import (
    BuildConfig,
    QueryConfig,
    DatabricksConfig,
    Neo4jConfig,
    LLMConfig,
    EmbeddingConfig,
    ProfilingConfig,
)


class TestDatabricksConfig:
    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi123")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
        config = DatabricksConfig()
        assert config.host == "https://example.databricks.com"
        assert config.token.get_secret_value() == "dapi123"
        assert config.http_path == "/sql/1.0/warehouses/abc"

    def test_explicit_values_override_env(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://env.databricks.com")
        config = DatabricksConfig(host="https://explicit.databricks.com")
        assert config.host == "https://explicit.databricks.com"

    def test_missing_required_fields(self):
        with pytest.raises(Exception):
            DatabricksConfig(host=None, token=None, http_path=None)


class TestNeo4jConfig:
    def test_defaults(self):
        config = Neo4jConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.password.get_secret_value() == "graphrag"

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://custom:7687")
        monkeypatch.setenv("NEO4J_USER", "admin")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")
        config = Neo4jConfig()
        assert config.uri == "bolt://custom:7687"
        assert config.user == "admin"
        assert config.password.get_secret_value() == "secret"


class TestLLMConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        config = LLMConfig()
        assert config.provider == "openrouter"
        assert config.model == "anthropic/claude-sonnet-4"
        assert config.base_url is None

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-123")
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key.get_secret_value() == "sk-123"

    def test_anthropic_direct(self):
        config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        assert config.provider == "anthropic"

    def test_databricks_provider(self):
        config = LLMConfig(
            provider="databricks",
            model="databricks-meta-llama-3-1-70b-instruct",
            base_url="https://my-workspace.databricks.com/serving-endpoints",
        )
        assert config.provider == "databricks"
        assert config.base_url is not None

    def test_custom_provider_with_base_url(self):
        config = LLMConfig(
            provider="custom",
            model="my-model",
            base_url="https://my-llm.example.com/v1",
        )
        assert config.provider == "custom"
        assert config.base_url == "https://my-llm.example.com/v1"


class TestEmbeddingConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        config = EmbeddingConfig()
        assert config.provider == "openrouter"
        assert config.model == "google/gemini-embedding-001"
        assert config.base_url is None

    def test_openai_provider(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        config = EmbeddingConfig()
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"

    def test_sentence_transformers_local(self):
        config = EmbeddingConfig(provider="sentence-transformers", model="all-MiniLM-L6-v2")
        assert config.provider == "sentence-transformers"

    def test_embeddable_labels_default(self):
        config = EmbeddingConfig()
        assert config.embeddable_labels == [
            "Entity", "Property", "Alias", "Term", "Metric",
        ]

    def test_embeddable_labels_custom(self):
        config = EmbeddingConfig(embeddable_labels=["Entity", "Alias"])
        assert config.embeddable_labels == ["Entity", "Alias"]

    def test_databricks_provider(self):
        config = EmbeddingConfig(
            provider="databricks",
            model="databricks-bge-large-en",
            base_url="https://my-workspace.databricks.com/serving-endpoints",
        )
        assert config.provider == "databricks"


class TestProfilingConfig:
    def test_defaults(self):
        config = ProfilingConfig()
        assert config.categorical_threshold == 500
        assert config.top_k_values == 100
        assert config.sample_rows == 5

    def test_custom_threshold(self):
        config = ProfilingConfig(categorical_threshold=1000)
        assert config.categorical_threshold == 1000


class TestBuildConfig:
    def test_full_config(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi123")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
        config = BuildConfig(
            source="databricks",
            catalog="cdm",
            schemas=["clinical", "staging"],
        )
        assert config.source == "databricks"
        assert config.catalog == "cdm"
        assert config.schemas == ["clinical", "staging"]
        assert config.neo4j is not None
        assert config.llm is not None
        assert config.embedding is not None
        assert config.profiling is not None

    def test_from_config_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "source: databricks\n"
            "catalog: cdm\n"
            "schemas:\n"
            "  - clinical\n"
            "  - staging\n"
        )
        config = BuildConfig.from_file(str(config_file))
        assert config.source == "databricks"
        assert config.catalog == "cdm"
        assert config.schemas == ["clinical", "staging"]

    def test_cli_flags_override_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi123")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("source: databricks\ncatalog: file_catalog\n")
        config = BuildConfig.from_file(str(config_file), overrides={"catalog": "override_catalog"})
        assert config.catalog == "override_catalog"


class TestQueryConfig:
    def test_defaults(self):
        config = QueryConfig(question="stage 3 colorectal patients")
        assert config.question == "stage 3 colorectal patients"
        assert config.operation == "plan"
        assert config.consumer == "nl2sql"
        assert config.verbose is False

    def test_execute_operation(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi123")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
        config = QueryConfig(
            question="stage 3 colorectal patients",
            operation="execute",
            verbose=True,
        )
        assert config.operation == "execute"
        assert config.verbose is True

    def test_freeform_operation(self):
        config = QueryConfig(question="test", operation="custom_op")
        assert config.operation == "custom_op"
