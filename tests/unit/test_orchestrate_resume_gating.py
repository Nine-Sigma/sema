"""Resume cleanup (bug-368): the destructive schema-wide wipe must NOT run
on a --resume build. It used to run for every schema even on resume, deleting
all edges and orphan-GC'ing every structural node; because resume then SKIPS
tables that already have assertions, those wiped nodes were never rebuilt (or
were lost if the run was interrupted mid-rebuild). Corrected behavior:

- Fresh build (resume=False): keep the per-schema `delete_study_scoped`
  (full wipe) — it also handles tables dropped from the source.
- Resume (resume=True): clear ONLY tables actually being re-run (no cached
  assertions) via the table-scoped `delete_table_scoped`. Tables with
  assertions are left intact and re-materialized from the cache.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _build_config(resume: bool, schemas: list[str]):
    from sema.models.config import (
        BuildConfig,
        DatabricksConfig,
        LLMConfig,
        Neo4jConfig,
    )
    return BuildConfig(
        catalog="cat",
        schemas=schemas,
        skip_embeddings=True,
        resume=resume,
        databricks=DatabricksConfig(),
        neo4j=Neo4jConfig(),
        llm=LLMConfig(),
    )


def _work_items(table_names: list[str], schema: str = "sch1"):
    from sema.pipeline.build import TableWorkItem

    return [
        TableWorkItem(
            catalog="cat",
            schema=schema,
            table_name=name,
            fqn=f"unity://cat.{schema}.{name}",
        )
        for name in table_names
    ]


@pytest.fixture
def loader_and_driver():
    driver = MagicMock()
    loader = MagicMock()
    return driver, loader


def _results_for(work_items):
    from sema.pipeline.build import TableResult

    return [
        TableResult.success(wi.fqn, entities=1, properties=1, terms=0)
        for wi in work_items
    ]


def _run_build_with(cfg, work_items, loader, driver, spawn_side_effect=None):
    """Drive run_build with heavy collaborators patched out so we can
    observe only the cleanup gating behavior."""
    results = _results_for(work_items)

    def _default_spawn(*args, **kwargs):
        return results

    with patch(
        "sema.pipeline.orchestrate._get_neo4j_driver",
        return_value=driver,
    ), patch(
        "sema.graph.loader.GraphLoader", return_value=loader,
    ), patch(
        "sema.pipeline.build.DatabricksConnectorFactory",
    ) as factory_cls, patch(
        "sema.pipeline.orchestrate._register_datasource",
    ), patch(
        "sema.pipeline.orchestrate._discover_tables",
        return_value=work_items,
    ), patch(
        "sema.pipeline.orchestrate._spawn_workers",
        side_effect=spawn_side_effect or _default_spawn,
    ), patch(
        "sema.pipeline.orchestrate.run_fk_detection",
    ), patch(
        "sema.pipeline.orchestrate._compute_embeddings",
    ), patch(
        "sema.pipeline.profiler.WarehouseProfiler",
    ) as profiler_cls, patch(
        "sema.pipeline.orchestrate.resolve_domain_context",
        return_value=__import__(
            "sema.models.domain", fromlist=["DomainContext"],
        ).DomainContext(),
    ):
        conn = MagicMock()
        conn.get_datasource_ref.return_value = ("unity://cat", "cat", "src")
        factory = MagicMock()
        factory.create.return_value = conn
        factory_cls.return_value = factory
        profiler = MagicMock()
        profiler.profile.return_value = MagicMock()
        profiler_cls.return_value = profiler

        from sema.pipeline.orchestrate import run_build
        run_build(cfg)


class TestResumeGating:
    def test_resume_clears_only_rerun_tables(self, loader_and_driver):
        driver, loader = loader_and_driver
        work_items = _work_items(["kept_a", "rerun_b", "kept_c"])
        # rerun_b has no cached assertions -> must be re-run and cleared.
        has = {
            "unity://cat.sch1.kept_a": True,
            "unity://cat.sch1.rerun_b": False,
            "unity://cat.sch1.kept_c": True,
        }
        loader.has_assertions.side_effect = lambda ref: has[ref]

        cfg = _build_config(resume=True, schemas=["sch1"])
        _run_build_with(cfg, work_items, loader, driver)

        # Never the schema-wide wipe on resume.
        loader.delete_study_scoped.assert_not_called()
        # Exactly the one re-run table cleared, table-scoped.
        assert loader.delete_table_scoped.call_count == 1
        call = loader.delete_table_scoped.call_args
        assert call.args[:4] == (
            "cat", "sch1", "rerun_b", "unity://cat.sch1.rerun_b",
        )

    def test_resume_all_cached_clears_nothing(self, loader_and_driver):
        driver, loader = loader_and_driver
        work_items = _work_items(["kept_a", "kept_b"])
        loader.has_assertions.return_value = True

        cfg = _build_config(resume=True, schemas=["sch1"])
        _run_build_with(cfg, work_items, loader, driver)

        loader.delete_study_scoped.assert_not_called()
        loader.delete_table_scoped.assert_not_called()

    def test_resume_cleanup_runs_before_workers(self, loader_and_driver):
        driver, loader = loader_and_driver
        work_items = _work_items(["rerun_b"])
        loader.has_assertions.return_value = False
        observed = {"cleared_before_workers": None}

        def _spawn(*args, **kwargs):
            observed["cleared_before_workers"] = (
                loader.delete_table_scoped.called
            )
            return _results_for(work_items)

        cfg = _build_config(resume=True, schemas=["sch1"])
        _run_build_with(
            cfg, work_items, loader, driver, spawn_side_effect=_spawn,
        )

        assert observed["cleared_before_workers"] is True

    def test_fresh_build_wipes_each_schema_once(self, loader_and_driver):
        driver, loader = loader_and_driver
        work_items = (
            _work_items(["t1"], schema="sch1")
            + _work_items(["t2"], schema="sch2")
        )

        cfg = _build_config(resume=False, schemas=["sch1", "sch2"])
        _run_build_with(cfg, work_items, loader, driver)

        called_schemas = sorted(
            c.args[0] for c in loader.delete_study_scoped.call_args_list
        )
        assert called_schemas == ["sch1", "sch2"]
        for c in loader.delete_study_scoped.call_args_list:
            assert c.kwargs.get("preserve_assertions") is False
        # Fresh builds do not use the table-scoped primitive.
        loader.delete_table_scoped.assert_not_called()


class TestProcessTableResumeWithAssertions:
    """Worker-level: with resume=True and existing assertions,
    process_table SHALL return TableResult.skipped without invoking LLM."""

    def test_returns_skipped_when_assertions_exist(self):
        from sema.pipeline.build import (
            TableWorkItem, process_table,
        )

        loader = MagicMock()
        loader.has_assertions.return_value = True
        loader.load_assertions.return_value = []
        connector = MagicMock()
        llm_client = MagicMock()

        wi = TableWorkItem(
            catalog="cat", schema="sch1",
            table_name="patient",
            fqn="unity://cat.sch1.patient",
        )
        with patch(
            "sema.graph.materializer.materialize_unified",
        ) as mat:
            result = process_table(
                wi, connector, llm_client, loader,
                run_id="rid", resume=True,
            )

        assert result.status == "skipped"
        assert "resume" in (result.skip_reason or "")
        connector.assert_not_called()
        llm_client.invoke.assert_not_called()
        # materialize_unified MUST receive source_schema — Neo4j rejects
        # MERGE on relationships with null source_schema property.
        assert mat.called
        kwargs = mat.call_args.kwargs
        assert kwargs.get("source_schema") == "sch1"

    def test_returns_full_pipeline_when_no_assertions(self):
        """When resume=True but no assertions exist, full pipeline runs."""
        from sema.pipeline.build import (
            TableWorkItem, process_table,
        )

        loader = MagicMock()
        loader.has_assertions.return_value = False
        connector = MagicMock()
        llm_client = MagicMock()

        wi = TableWorkItem(
            catalog="cat", schema="sch1",
            table_name="patient",
            fqn="unity://cat.sch1.patient",
        )
        with patch(
            "sema.pipeline.build._run_pipeline_stages",
            side_effect=Exception("forced exit after gate"),
        ):
            result = process_table(
                wi, connector, llm_client, loader,
                run_id="rid", resume=True,
            )

        # _run_pipeline_stages was reached, then raised — so resume
        # short-circuit was NOT taken.
        assert result.status == "failed"
