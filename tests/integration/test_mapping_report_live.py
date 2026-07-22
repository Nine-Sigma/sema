"""US-012 live test: the mapping eval report over real resolved decisions.

Skip-guarded on ~/.sema/poc.duckdb. Resolves every gold-set ONCOTREE_CODE with
the real §4 resolver into a temp value-mapping store (US-005/US-006), then runs
the §1.5(f) report (US-012) against the gold set.

Acceptance is asserted ONLY at 100% human-labelled gold coverage. The gold set
ships UNLABELLED (the US-002 human gate is open), so this run records metrics
and asserts the verdict is ``provisional — not accepted`` with the gap
surfaced — never self-certifying precision. Once the gold set is fully
hand-labelled, the same test asserts the >=95% precision / >=70%
auto-resolution thresholds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from sema.eval.mapping_report import build_mapping_report, decisions_from_store
from sema.eval.mapping_report_utils import AcceptanceVerdict
from sema.eval.mapping_goldset import GoldSet, load_gold_set
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.resolve.engine import VocabularyResolver
from sema.resolve.engine_utils import ResolveContext
from showcase.cbioportal_to_omop.omop_policy import (
    OMOP_ONCOTREE_CONDITION_REF,
    OMOP_VOCAB_SCHEMA,
    make_omop_oncotree_condition_policy,
)
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.vocab_store import open_duckdb_vocab_store
from tests.integration._omop_binding import build_condition_binding

pytestmark = pytest.mark.integration

_DB = Path.home() / ".sema" / "poc.duckdb"
_GOLD = Path(__file__).resolve().parents[1] / "data" / "gold" / "oncotree_condition_slice0.jsonl"
_VOCAB_RELEASE = "omop-vocab-2024"
_POLICY_REF = OMOP_ONCOTREE_CONDITION_REF
_TARGET_PROPERTY_REF = "target.stage.condition_concept_id"


def _context() -> ResolveContext:
    prov = Provenance(
        run=RunProvenance(
            run_id="us012-live",
            target_model_version="omop-cdm-5.4",
            target_schema_snapshot_hash="t",
            vocab_release=_VOCAB_RELEASE,
            context_card_version="cc",
            prompt_template_version="pt",
            few_shot_set_version="fs",
            constraint_version="cv",
            llm_model="none",
        ),
        source=SourceScope(
            source_id="study", source_schema_hash="s", source_profile_hash="p"
        ),
        timestamp=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )
    return ResolveContext(
        source_field_ref="source.sample.ONCOTREE_CODE",
        source_value_ref="source.sample.ONCOTREE_CODE",
        target_property_ref=_TARGET_PROPERTY_REF,
        target_field="condition_concept_id",
        domain_constraint_ref="target.stage.domain=Condition",
        vocabulary_ref="vocab.snomed",
        vocab_binding="binding.condition",
        vocab_release=_VOCAB_RELEASE,
        resolver_policy_ref=_POLICY_REF,
        run_id="us012-live",
        provenance=prov,
    )


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_mapping_report_over_real_decisions(tmp_path: Path) -> None:
    policy = make_omop_oncotree_condition_policy(build_condition_binding())
    vstore = open_duckdb_vocab_store(str(_DB), schema=OMOP_VOCAB_SCHEMA)
    resolver = VocabularyResolver(vstore, policy)

    gold = GoldSet(load_gold_set(_GOLD))
    codes = sorted({r.oncotree_code for r in gold.rows})

    vm_conn = duckdb.connect(str(tmp_path / "value_mapping.duckdb"))
    store = ValueMappingStore(vm_conn)
    resolver.resolve_and_store(codes, store, _context())

    decisions = decisions_from_store(
        store,
        target_property_ref=_TARGET_PROPERTY_REF,
        resolver_policy_ref=_POLICY_REF,
        vocab_release=_VOCAB_RELEASE,
    )
    assert len(decisions) == len(codes)

    report = build_mapping_report(gold, decisions)
    print(report.human_summary())

    if report.coverage_fraction >= 1.0:
        # Human-label gate complete: assert the §1.5(f) acceptance thresholds.
        m = report.score.distinct_code
        assert m.mapped_precision is not None and m.mapped_precision >= 0.95
        assert m.auto_resolution_rate is not None and m.auto_resolution_rate >= 0.70
        assert report.verdict is AcceptanceVerdict.ACCEPTED
    else:
        # Gate still open: never self-certify; surface the unlabelled gap.
        assert report.verdict is AcceptanceVerdict.PROVISIONAL_NOT_ACCEPTED
        assert report.unlabelled_codes  # the gap is surfaced, not hidden
        assert "human-label gate" in report.verdict_reason

    store.close()
