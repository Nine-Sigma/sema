"""Resume cleanup (US-009): the schema-wipe loop runs for BOTH a fresh
build and a --resume build, so a resumed study's prior graph writes are
cleared before re-materialization (finding L). A resume build preserves
:Assertion nodes — they are the resume cache process_table reads, so
deleting them would silently turn every resume into a full rebuild."""
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


def _patch_run_build(work_items_for_schemas: list[str]):
    from sema.pipeline.build import TableResult, TableWorkItem

    work_items = [
        TableWorkItem(
            catalog="cat",
            schema=s,
            table_name=f"tbl_{s}",
            fqn=f"unity://cat.{s}.tbl_{s}",
        )
        for s in work_items_for_schemas
    ]
    return work_items


@pytest.fixture
def loader_and_driver():
    driver = MagicMock()
    loader = MagicMock()
    return driver, loader


def _common_patches(work_items, results=None):
    """Patch the heavy collaborators of run_build so we can isolate
    the schema-wipe gating behavior."""
    if results is None:
        from sema.pipeline.build import TableResult
        results = [
            TableResult.success(wi.fqn, entities=1, properties=1, terms=0)
            for wi in work_items
        ]
    return results


class TestResumeGating:
    def test_resume_true_wipes_each_schema_before_workers(
        self, loader_and_driver,
    ):
        driver, loader = loader_and_driver
        work_items = _patch_run_build(["sch1", "sch2", "sch1"])
        results = _common_patches(work_items)

        observed = {"deleted_before_workers": None}

        def _spawn_side_effect(*args, **kwargs):
            observed["deleted_before_workers"] = (
                loader.delete_study_scoped.called
            )
            return results

        cfg = _build_config(resume=True, schemas=["sch1", "sch2"])
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
            side_effect=_spawn_side_effect,
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
            conn.get_datasource_ref.return_value = (
                "unity://cat", "cat", "src",
            )
            factory = MagicMock()
            factory.create.return_value = conn
            factory_cls.return_value = factory
            profiler = MagicMock()
            profiler.profile.return_value = MagicMock()
            profiler_cls.return_value = profiler

            from sema.pipeline.orchestrate import run_build
            run_build(cfg)

        called_schemas = sorted(
            c.args[0] for c in loader.delete_study_scoped.call_args_list
        )
        assert called_schemas == ["sch1", "sch2"]
        assert observed["deleted_before_workers"] is True
        # Resume cleanup must NOT delete the :Assertion resume cache.
        for c in loader.delete_study_scoped.call_args_list:
            assert c.kwargs.get("preserve_assertions") is True

    def test_resume_false_wipes_each_schema_once(self, loader_and_driver):
        driver, loader = loader_and_driver
        work_items = _patch_run_build(["sch1", "sch2", "sch1"])
        results = _common_patches(work_items)

        cfg = _build_config(resume=False, schemas=["sch1", "sch2"])
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
            return_value=results,
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
            conn.get_datasource_ref.return_value = (
                "unity://cat", "cat", "src",
            )
            factory = MagicMock()
            factory.create.return_value = conn
            factory_cls.return_value = factory
            profiler = MagicMock()
            profiler.profile.return_value = MagicMock()
            profiler_cls.return_value = profiler

            from sema.pipeline.orchestrate import run_build
            run_build(cfg)

        called_schemas = sorted(
            c.args[0] for c in loader.delete_study_scoped.call_args_list
        )
        assert called_schemas == ["sch1", "sch2"]
        # Fresh builds delete :Assertion nodes too (full wipe).
        for c in loader.delete_study_scoped.call_args_list:
            assert c.kwargs.get("preserve_assertions") is False

    def test_resume_true_absent_study_cleanup_is_no_op(
        self, loader_and_driver,
    ):
        """Fresh resume: an absent study still gets a cleanup call, which
        is a harmless no-op (delete matches nothing) — run completes."""
        driver, loader = loader_and_driver
        loader.has_assertions.return_value = False
        loader.delete_study_scoped.return_value = None
        work_items = _patch_run_build(["sch1"])
        results = _common_patches(work_items)

        cfg = _build_config(resume=True, schemas=["sch1"])
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
            return_value=results,
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
            conn.get_datasource_ref.return_value = (
                "unity://cat", "cat", "src",
            )
            factory = MagicMock()
            factory.create.return_value = conn
            factory_cls.return_value = factory
            profiler = MagicMock()
            profiler.profile.return_value = MagicMock()
            profiler_cls.return_value = profiler

            from sema.pipeline.orchestrate import run_build
            run_build(cfg)

        loader.delete_study_scoped.assert_called_once_with(
            "sch1", preserve_assertions=True,
        )


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
