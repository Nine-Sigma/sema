"""Healthcare few-shot examples for staged L2 prompts.

Examples sourced from oncology warehouse analyst questions; cover the
clinical-genomics shapes (patient, sample, mutation, treatment, structural
variant) plus MSK CHORD-introduced shapes (lab timelines, procedures,
performance status, biomarker states, segmented CNA). Stage arrays live in
sibling modules to keep each file under the per-file line cap.
"""
from __future__ import annotations

from sema.engine.few_shot_healthcare_stage_a import HEALTHCARE_STAGE_A
from sema.engine.few_shot_healthcare_stage_b import HEALTHCARE_STAGE_B
from sema.engine.few_shot_healthcare_stage_c import HEALTHCARE_STAGE_C

__all__ = [
    "HEALTHCARE_STAGE_A",
    "HEALTHCARE_STAGE_B",
    "HEALTHCARE_STAGE_C",
]
