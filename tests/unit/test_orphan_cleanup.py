"""US-008: study deletion must not strand orphaned graph elements.

`delete_study_scoped` deletes a study's edges and its `:Assertion` /
`:JoinPath` nodes, but historically left behind (J) `:Alias` nodes whose
only `REFERS_TO` edge was just deleted, and (H) shared concept nodes whose
every referencing study is gone. Both must be garbage-collected — and ONLY
when they have no remaining relationships, so nodes still referenced by a
surviving study are untouched.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import delete_orphaned_nodes

pytestmark = pytest.mark.unit

SCHEMA = "cbioportal_brca_tcga_pan_can_atlas_2018"
CONCEPT_LABELS = ("Entity", "Property", "Term", "Vocabulary", "ValueSet")


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def loader(mock_driver):
    driver, _ = mock_driver
    return GraphLoader(driver)


def _cyphers(session) -> list[str]:
    return [c[0][0] for c in session.run.call_args_list]


class TestDeleteOrphanedAliases:
    def test_alias_cleanup_guarded_by_missing_refers_to(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.delete_study_scoped(SCHEMA)
        assert any(
            ":Alias" in c
            and "NOT (a)-[:REFERS_TO]->()" in c
            and "DELETE a" in c
            for c in _cyphers(session)
        )

    def test_alias_cleanup_never_deletes_unconditionally(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.delete_study_scoped(SCHEMA)
        for c in _cyphers(session):
            if ":Alias" in c and "DELETE a" in c:
                assert "NOT (a)-[:REFERS_TO]->()" in c


class TestDeleteOrphanedConceptNodes:
    def test_each_concept_label_swept_for_orphans(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.delete_study_scoped(SCHEMA)
        joined = "\n".join(_cyphers(session))
        for label in CONCEPT_LABELS:
            assert (
                f"MATCH (n:{label}) WHERE NOT (n)--() DELETE n" in joined
            )

    def test_concept_deletion_always_guarded(self, loader, mock_driver):
        """No concept node is deleted unless it is edge-less (two-study
        safety: shared nodes still referenced by a survivor have edges)."""
        _, session = mock_driver
        loader.delete_study_scoped(SCHEMA)
        for c in _cyphers(session):
            for label in CONCEPT_LABELS:
                if f"(n:{label})" in c and "DELETE n" in c:
                    assert "NOT (n)--()" in c


class TestCleanupOrdering:
    def test_orphan_cleanup_runs_after_edge_sweep(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        loader.delete_study_scoped(SCHEMA)
        cyphers = _cyphers(session)
        edge_sweep = next(
            i for i, c in enumerate(cyphers)
            if "[r {source_schema: $schema}]" in c and "DELETE r" in c
        )
        alias_clean = next(
            i for i, c in enumerate(cyphers)
            if ":Alias" in c and "DELETE a" in c
        )
        concept_clean = next(
            i for i, c in enumerate(cyphers)
            if "(n:Entity)" in c and "DELETE n" in c
        )
        assert edge_sweep < alias_clean
        assert edge_sweep < concept_clean
        # aliases dropped before concept sweep so a freed target can orphan
        assert alias_clean < concept_clean


class TestHelperEntryPoint:
    def test_helper_issues_alias_then_concept_sweeps(self):
        loader = MagicMock()
        delete_orphaned_nodes(loader)
        cyphers = [c[0][0] for c in loader._run.call_args_list]
        assert any(":Alias" in c for c in cyphers)
        for label in CONCEPT_LABELS:
            assert any(f"(n:{label})" in c for c in cyphers)
