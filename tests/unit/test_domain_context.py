from __future__ import annotations

from typing import Any

import pytest

from sema.models.config import BuildConfig
from sema.models.domain import DomainCandidate, DomainContext

pytestmark = pytest.mark.unit


class TestDomainCandidate:
    def test_create_with_required_fields(self) -> None:
        candidate = DomainCandidate(domain="healthcare", confidence=0.85)
        assert candidate.domain == "healthcare"
        assert candidate.confidence == 0.85

    def test_confidence_clamped_to_valid_range(self) -> None:
        with pytest.raises(ValueError):
            DomainCandidate(domain="healthcare", confidence=1.5)
        with pytest.raises(ValueError):
            DomainCandidate(domain="healthcare", confidence=-0.1)


class TestDomainContext:
    def test_defaults(self) -> None:
        ctx = DomainContext()
        assert ctx.declared_domain is None
        assert ctx.detected_domain is None
        assert ctx.domain_confidence == 0.0
        assert ctx.alternate_domains == []
        assert ctx.domain_source == "default"

    def test_user_declared(self) -> None:
        ctx = DomainContext(
            declared_domain="healthcare",
            domain_source="user",
            domain_confidence=1.0,
        )
        assert ctx.declared_domain == "healthcare"
        assert ctx.domain_source == "user"

    def test_profiler_detected(self) -> None:
        ctx = DomainContext(
            detected_domain="healthcare",
            domain_confidence=0.72,
            domain_source="profiler",
            alternate_domains=[
                DomainCandidate(domain="financial", confidence=0.15),
            ],
        )
        assert ctx.detected_domain == "healthcare"
        assert len(ctx.alternate_domains) == 1
        assert ctx.alternate_domains[0].domain == "financial"

    def test_effective_domain_prefers_declared(self) -> None:
        ctx = DomainContext(
            declared_domain="healthcare",
            detected_domain="financial",
            domain_confidence=0.8,
            domain_source="user",
        )
        assert ctx.effective_domain == "healthcare"

    def test_effective_domain_falls_back_to_detected(self) -> None:
        ctx = DomainContext(
            detected_domain="financial",
            domain_confidence=0.7,
            domain_source="profiler",
        )
        assert ctx.effective_domain == "financial"

    def test_effective_domain_none_when_unknown(self) -> None:
        ctx = DomainContext()
        assert ctx.effective_domain is None

    def test_domain_source_literal_validation(self) -> None:
        with pytest.raises(ValueError):
            DomainContext(domain_source="invalid")

    def test_serialization_roundtrip(self) -> None:
        ctx = DomainContext(
            declared_domain="healthcare",
            detected_domain="healthcare",
            domain_confidence=0.9,
            domain_source="user",
            alternate_domains=[
                DomainCandidate(domain="financial", confidence=0.1),
            ],
        )
        data = ctx.model_dump()
        restored = DomainContext(**data)
        assert restored == ctx


class TestBuildConfigDomain:
    def test_domain_defaults_to_none(self) -> None:
        config = BuildConfig()
        assert config.domain is None

    def test_domain_set_directly(self) -> None:
        config = BuildConfig(domain="healthcare")
        assert config.domain == "healthcare"

    def test_domain_from_yaml(self, tmp_path: Any) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("domain: healthcare\ncatalog: test\n")
        config = BuildConfig.from_file(str(yaml_file))
        assert config.domain == "healthcare"

    def test_domain_from_yaml_absent(self, tmp_path: Any) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("catalog: test\n")
        config = BuildConfig.from_file(str(yaml_file))
        assert config.domain is None


class TestDomainPrecedence:
    def test_cli_overrides_all(self) -> None:
        from sema.models.domain import resolve_domain_context
        from sema.models.warehouse_profile import WarehouseProfile
        from datetime import datetime, timezone

        profile = WarehouseProfile(
            profile_id="p1", run_id="r1", datasource_id="ds1",
            domains={"financial": 0.9}, evidence=["keywords"],
            confidence=0.9, profiled_at=datetime.now(timezone.utc),
        )
        ctx = resolve_domain_context(
            cli_domain="healthcare", config_domain="logistics", profile=profile,
        )
        assert ctx.effective_domain == "healthcare"
        assert ctx.domain_source == "user"
        # Profiler confidence preserved for conflict handling
        assert ctx.domain_confidence == 0.9
        assert ctx.detected_domain == "financial"

    def test_config_overrides_profiler(self) -> None:
        from sema.models.domain import resolve_domain_context
        from sema.models.warehouse_profile import WarehouseProfile
        from datetime import datetime, timezone

        profile = WarehouseProfile(
            profile_id="p1", run_id="r1", datasource_id="ds1",
            domains={"financial": 0.8}, evidence=["keywords"],
            confidence=0.8, profiled_at=datetime.now(timezone.utc),
        )
        ctx = resolve_domain_context(
            cli_domain=None, config_domain="logistics", profile=profile,
        )
        assert ctx.effective_domain == "logistics"
        assert ctx.domain_source == "config"
        # Profiler confidence preserved for conflict handling
        assert ctx.domain_confidence == 0.8
        assert ctx.detected_domain == "financial"

    def test_profiler_used_when_no_override(self) -> None:
        from sema.models.domain import resolve_domain_context
        from sema.models.warehouse_profile import WarehouseProfile
        from datetime import datetime, timezone

        profile = WarehouseProfile(
            profile_id="p1", run_id="r1", datasource_id="ds1",
            domains={"healthcare": 0.7, "financial": 0.2},
            evidence=["patient keywords"], confidence=0.7,
            profiled_at=datetime.now(timezone.utc),
        )
        ctx = resolve_domain_context(
            cli_domain=None, config_domain=None, profile=profile,
        )
        assert ctx.effective_domain == "healthcare"
        assert ctx.domain_source == "profiler"
        assert ctx.domain_confidence == 0.7
        assert len(ctx.alternate_domains) == 1
        assert ctx.alternate_domains[0].domain == "financial"

    def test_default_when_nothing_provided(self) -> None:
        from sema.models.domain import resolve_domain_context

        ctx = resolve_domain_context(
            cli_domain=None, config_domain=None, profile=None,
        )
        assert ctx.effective_domain is None
        assert ctx.domain_source == "default"
        assert ctx.domain_confidence == 0.0

    def test_profiler_with_empty_domains(self) -> None:
        from sema.models.domain import resolve_domain_context
        from sema.models.warehouse_profile import WarehouseProfile
        from datetime import datetime, timezone

        profile = WarehouseProfile(
            profile_id="p1", run_id="r1", datasource_id="ds1",
            domains={}, evidence=[], confidence=0.0,
            profiled_at=datetime.now(timezone.utc),
        )
        ctx = resolve_domain_context(
            cli_domain=None, config_domain=None, profile=profile,
        )
        assert ctx.effective_domain is None
        assert ctx.domain_source == "default"


class TestVocabColumnContextEnrichment:
    def test_legacy_fields_still_work(self) -> None:
        from sema.engine.vocabulary import VocabColumnContext
        ctx = VocabColumnContext(
            column_name="patient_id",
            entity_name="Patient",
            semantic_type="identifier",
        )
        assert ctx.column_name == "patient_id"
        assert ctx.entity_name == "Patient"

    def test_new_fields_raise_at_version_zero(self) -> None:
        from sema.engine.vocabulary import VocabColumnContext
        ctx = VocabColumnContext(column_name="test")
        with pytest.raises(AttributeError):
            _ = ctx.candidate_vocab_families
        with pytest.raises(AttributeError):
            _ = ctx.grain_hypothesis
        with pytest.raises(AttributeError):
            _ = ctx.ambiguity_notes
        with pytest.raises(AttributeError):
            _ = ctx.entity_role
        with pytest.raises(AttributeError):
            _ = ctx.domain_context

    def test_new_fields_accessible_at_version_one(self) -> None:
        from sema.engine.vocabulary import VocabColumnContext
        ctx = VocabColumnContext(
            column_name="test",
            _enrichment_version=1,
            _candidate_vocab_families=["diagnosis coding system"],
            _grain_hypothesis="patient-level",
            _ambiguity_notes=["could be gene or cancer type"],
            _entity_role="primary_key",
            _domain_context=DomainContext(
                declared_domain="healthcare",
            ),
        )
        assert ctx.candidate_vocab_families == ["diagnosis coding system"]
        assert ctx.grain_hypothesis == "patient-level"
        assert ctx.ambiguity_notes == ["could be gene or cancer type"]
        assert ctx.entity_role == "primary_key"
        assert ctx.domain_context.declared_domain == "healthcare"

    def test_default_enrichment_version_is_zero(self) -> None:
        from sema.engine.vocabulary import VocabColumnContext
        ctx = VocabColumnContext()
        assert ctx._enrichment_version == 0


class TestProfilerIntegration:
    def test_profiler_called_in_resolve_domain_context(self) -> None:
        from sema.models.domain import resolve_domain_context
        from sema.models.warehouse_profile import WarehouseProfile
        from datetime import datetime, timezone

        profile = WarehouseProfile(
            profile_id="p1", run_id="r1", datasource_id="ds1",
            domains={"healthcare": 0.8, "financial": 0.1},
            evidence=["patient keywords"],
            confidence=0.8,
            profiled_at=datetime.now(timezone.utc),
        )
        ctx = resolve_domain_context(
            cli_domain=None, config_domain=None, profile=profile,
        )
        assert ctx.detected_domain == "healthcare"
        assert ctx.domain_source == "profiler"
        assert ctx.domain_confidence == 0.8
        assert len(ctx.alternate_domains) == 1
        assert ctx.alternate_domains[0].domain == "financial"

    def test_cli_overrides_profiler_detection(self) -> None:
        from sema.models.domain import resolve_domain_context
        from sema.models.warehouse_profile import WarehouseProfile
        from datetime import datetime, timezone

        profile = WarehouseProfile(
            profile_id="p1", run_id="r1", datasource_id="ds1",
            domains={"financial": 0.9},
            evidence=["financial keywords"],
            confidence=0.9,
            profiled_at=datetime.now(timezone.utc),
        )
        ctx = resolve_domain_context(
            cli_domain="healthcare", config_domain=None, profile=profile,
        )
        assert ctx.effective_domain == "healthcare"
        assert ctx.domain_source == "user"


class TestIsolation:
    def test_domain_context_none_produces_identical_output(self) -> None:
        """Explicit domain_context=None must match the default (not passed).

        Both engines should produce identical assertions from the same
        mocked staged responses.
        """
        from unittest.mock import MagicMock

        from sema.engine.semantic import SemanticEngine
        from sema.llm_client import LLMClient
        from sema.models.stages import (
            StageAResult,
            StageBBatchResult,
            StageBColumnResult,
        )

        metadata = {
            "table_ref": "unity://cat.sch.patient",
            "table_name": "patient",
            "columns": [{"name": "id", "data_type": "STRING"}],
            "sample_rows": [],
            "comment": None,
        }

        def _make_client() -> MagicMock:
            client = MagicMock(spec=LLMClient)
            client.invoke.side_effect = [
                StageAResult(
                    primary_entity="Patient",
                    grain_hypothesis="one row per patient",
                    confidence=0.9,
                ),
                StageBBatchResult(columns=[
                    StageBColumnResult(
                        column="id",
                        canonical_property_label="patient id",
                        semantic_type="identifier",
                        entity_role="primary_key",
                        needs_stage_c=False,
                    ),
                ]),
            ]
            return client

        engine_default = SemanticEngine(
            llm_client=_make_client(), run_id="isolation-test",
        )
        engine_explicit_none = SemanticEngine(
            llm_client=_make_client(), run_id="isolation-test",
            domain_context=None,
        )

        a_default = engine_default.interpret_table(metadata)
        a_explicit = engine_explicit_none.interpret_table(metadata)

        assert len(a_default) == len(a_explicit)
        for a1, a2 in zip(a_default, a_explicit):
            assert a1.predicate == a2.predicate
            assert a1.subject_ref == a2.subject_ref
            assert a1.payload == a2.payload
            assert a1.confidence == a2.confidence


class TestEngineAcceptsDomainContext:
    def test_semantic_engine_stores_domain_context(self) -> None:
        from sema.engine.semantic import SemanticEngine
        ctx = DomainContext(declared_domain="healthcare", domain_source="user")
        engine = SemanticEngine(domain_context=ctx)
        assert engine._domain_context is ctx

    def test_semantic_engine_defaults_to_none(self) -> None:
        from sema.engine.semantic import SemanticEngine
        engine = SemanticEngine()
        assert engine._domain_context is None

    def test_vocabulary_engine_stores_domain_context(self) -> None:
        from sema.engine.vocabulary import VocabularyEngine
        ctx = DomainContext(declared_domain="healthcare", domain_source="user")
        engine = VocabularyEngine(domain_context=ctx)
        assert engine._domain_context is ctx

    def test_vocabulary_engine_defaults_to_none(self) -> None:
        from sema.engine.vocabulary import VocabularyEngine
        engine = VocabularyEngine()
        assert engine._domain_context is None


class TestCLIDomainFlag:
    def test_domain_flag_wired_into_config(self) -> None:
        from unittest.mock import patch
        from click.testing import CliRunner
        from sema.cli import build

        runner = CliRunner()
        with patch("sema.cli.run_build") as mock_run:
            mock_run.return_value = {"tables_processed": 0}
            result = runner.invoke(build, ["--domain", "healthcare"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.domain == "healthcare"

    def test_domain_flag_defaults_to_none(self) -> None:
        from unittest.mock import patch
        from click.testing import CliRunner
        from sema.cli import build

        runner = CliRunner()
        with patch("sema.cli.run_build") as mock_run:
            mock_run.return_value = {"tables_processed": 0}
            result = runner.invoke(build, [])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.domain is None

    def test_cli_flag_sets_domain_from_cli(self) -> None:
        from unittest.mock import patch
        from click.testing import CliRunner
        from sema.cli import build

        runner = CliRunner()
        with patch("sema.cli.run_build") as mock_run:
            mock_run.return_value = {"tables_processed": 0}
            result = runner.invoke(build, ["--domain", "healthcare"])
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.domain_from_cli is True

    def test_yaml_domain_not_marked_as_cli(self, tmp_path: Any) -> None:
        from unittest.mock import patch
        from click.testing import CliRunner
        from sema.cli import build

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("domain: financial\n")
        runner = CliRunner()
        with patch("sema.cli.run_build") as mock_run:
            mock_run.return_value = {"tables_processed": 0}
            result = runner.invoke(
                build, ["--config", str(yaml_file)],
            )
            assert result.exit_code == 0
            config = mock_run.call_args[0][0]
            assert config.domain == "financial"
            assert config.domain_from_cli is False
