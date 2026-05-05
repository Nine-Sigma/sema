"""MSK CHORD healthcare few-shot expansion: Stage A/B/C examples for labs, procedures,
biomarker states, performance status, CNA matrices."""
from __future__ import annotations

import pytest

from sema.engine.few_shot import compose_examples, get_examples
from sema.engine.few_shot_healthcare import (
    HEALTHCARE_STAGE_A,
    HEALTHCARE_STAGE_B,
    HEALTHCARE_STAGE_C,
)

pytestmark = pytest.mark.unit


def _table_names_in_stage(stage: list[dict[str, object]]) -> set[str]:
    names: set[str] = set()
    for ex in stage:
        inp = ex["input"]
        assert isinstance(inp, dict)
        name = inp.get("table_name")
        if isinstance(name, str):
            names.add(name)
    return names


class TestStageAHealthcareBreadth:
    def test_includes_lab_timeline(self) -> None:
        names = _table_names_in_stage(HEALTHCARE_STAGE_A)
        assert "timeline_labtest" in names

    def test_includes_procedure_event(self) -> None:
        names = _table_names_in_stage(HEALTHCARE_STAGE_A)
        assert any(n.startswith("timeline_") and "surg" in n.lower() for n in names) or \
               "procedure" in names or "timeline_surgery" in names

    def test_includes_performance_status(self) -> None:
        names = _table_names_in_stage(HEALTHCARE_STAGE_A)
        assert "timeline_performance_status" in names

    def test_includes_cna_segmented(self) -> None:
        names = _table_names_in_stage(HEALTHCARE_STAGE_A)
        assert "cna_segmented" in names


class TestStageBHealthcareBreadth:
    def _by_col(self) -> dict[tuple[str, str], dict[str, object]]:
        index: dict[tuple[str, str], dict[str, object]] = {}
        for ex in HEALTHCARE_STAGE_B:
            inp = ex["input"]
            assert isinstance(inp, dict)
            tbl = inp.get("table_name")
            col = inp.get("column")
            assert isinstance(tbl, str) and isinstance(col, str)
            index[(tbl, col)] = ex["output"]  # type: ignore[assignment]
        return index

    def test_lab_value_with_units_present(self) -> None:
        index = self._by_col()
        assert ("timeline_labtest", "VALUE") in index
        assert ("timeline_labtest", "UNITS") in index

    def test_pdl1_biomarker_state(self) -> None:
        index = self._by_col()
        keys = list(index.keys())
        assert any("pd_l1" in c.lower() or "pdl1" in c.lower() for _t, c in keys)

    def test_mmr_status(self) -> None:
        index = self._by_col()
        assert any("mmr" in c.lower() for _t, c in index)

    def test_ecog_performance_score(self) -> None:
        index = self._by_col()
        assert any("ecog" in c.lower() for _t, c in index)

    def test_karnofsky_performance_score(self) -> None:
        index = self._by_col()
        assert any("karnofsky" in c.lower() for _t, c in index)

    def test_procedure_code(self) -> None:
        index = self._by_col()
        assert any(
            "procedure" in t.lower() or "procedure_code" in c.lower()
            for t, c in index
        )


class TestStageCHealthcareDecodings:
    def _columns(self) -> set[str]:
        cols: set[str] = set()
        for ex in HEALTHCARE_STAGE_C:
            inp = ex["input"]
            assert isinstance(inp, dict)
            col = inp.get("column")
            if isinstance(col, str):
                cols.add(col.lower())
        return cols

    def test_pdl1_decoding(self) -> None:
        assert any("pd_l1" in c or "pdl1" in c for c in self._columns())

    def test_mmr_decoding(self) -> None:
        assert any("mmr" in c for c in self._columns())

    def test_gleason_decoding(self) -> None:
        assert any("gleason" in c for c in self._columns())

    def test_ecog_decoding(self) -> None:
        assert any("ecog" in c for c in self._columns())

    def test_karnofsky_decoding(self) -> None:
        assert any("karnofsky" in c for c in self._columns())


class TestSynonymsCoverageOnNewExamples:
    def test_lab_biomarker_procedure_examples_have_two_or_more_synonyms(self) -> None:
        targets: list[tuple[str, str]] = []
        for ex in HEALTHCARE_STAGE_B:
            inp = ex["input"]
            out = ex["output"]
            assert isinstance(inp, dict) and isinstance(out, dict)
            tbl = inp.get("table_name", "")
            col = inp.get("column", "")
            assert isinstance(tbl, str) and isinstance(col, str)
            sem_type = out.get("semantic_type", "")
            assert isinstance(sem_type, str)
            is_lab = "lab" in tbl.lower() or col.upper() in {"VALUE", "UNITS", "TEST"}
            is_biomarker = (
                "biomarker" in sem_type.lower()
                or "pd_l1" in col.lower()
                or "pdl1" in col.lower()
                or "mmr" in col.lower()
                or "gleason" in col.lower()
            )
            is_procedure = "procedure" in tbl.lower() or "procedure_code" in col.lower()
            if is_lab or is_biomarker or is_procedure:
                synonyms = out.get("synonyms", [])
                assert isinstance(synonyms, list)
                if not synonyms or len(synonyms) < 2:
                    targets.append((tbl, col))
        assert not targets, (
            f"Lab/biomarker/procedure B examples missing ≥2 synonyms: {targets}"
        )


class TestRegistryFallbackBehavior:
    def test_healthcare_domain_returns_healthcare_examples(self) -> None:
        composed_b = compose_examples("healthcare", "B")
        # composed is generic + healthcare
        healthcare_only = get_examples("healthcare", "B")
        assert healthcare_only, "healthcare B examples should be registered"
        # All healthcare entries appear in composed
        for hc_ex in healthcare_only:
            assert hc_ex in composed_b

    def test_unknown_domain_falls_back_to_generic_only(self) -> None:
        composed = compose_examples("nonexistent_domain", "B")
        from sema.engine.few_shot_generic import GENERIC_STAGE_B
        assert composed == list(GENERIC_STAGE_B)

    def test_zero_shot_when_neither_registered(self) -> None:
        # Stage Z is not registered for any domain
        composed = compose_examples("healthcare", "Z")
        assert composed == []
