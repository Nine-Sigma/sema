import pytest

pytestmark = pytest.mark.unit

from sema.graph.context_store import ContextStore
from sema.models.context import SemanticContextObject


@pytest.fixture
def store():
    return ContextStore()


@pytest.fixture
def sample_sco():
    return SemanticContextObject(
        entities=[], physical_assets=[], join_paths=[],
        governed_values=[], consumer_hint="nl2sql",
    )


class TestContextStore:
    def test_cache_miss(self, store):
        result = store.get("sess-1", "stage 3 patients", "nl2sql", "v1")
        assert result is None

    def test_cache_hit(self, store, sample_sco):
        store.put("sess-1", "stage 3 patients", "nl2sql", "v1", sample_sco)
        result = store.get("sess-1", "stage 3 patients", "nl2sql", "v1")
        assert result is not None
        assert result.consumer_hint == "nl2sql"

    def test_different_sessions_dont_share(self, store, sample_sco):
        store.put("sess-1", "stage 3 patients", "nl2sql", "v1", sample_sco)
        result = store.get("sess-2", "stage 3 patients", "nl2sql", "v1")
        assert result is None

    def test_different_consumer_hints_dont_share(self, store, sample_sco):
        store.put("sess-1", "stage 3 patients", "nl2sql", "v1", sample_sco)
        result = store.get("sess-1", "stage 3 patients", "discovery", "v1")
        assert result is None

    def test_different_graph_versions_dont_share(self, store, sample_sco):
        store.put("sess-1", "stage 3 patients", "nl2sql", "v1", sample_sco)
        result = store.get("sess-1", "stage 3 patients", "nl2sql", "v2")
        assert result is None

    def test_invalidate_clears_all(self, store, sample_sco):
        store.put("sess-1", "q1", "nl2sql", "v1", sample_sco)
        store.put("sess-2", "q2", "discovery", "v1", sample_sco)
        store.invalidate()
        assert store.get("sess-1", "q1", "nl2sql", "v1") is None
        assert store.get("sess-2", "q2", "discovery", "v1") is None

    def test_query_normalization(self, store, sample_sco):
        store.put("sess-1", "  Stage 3  Patients  ", "nl2sql", "v1", sample_sco)
        result = store.get("sess-1", "stage 3 patients", "nl2sql", "v1")
        assert result is not None
