"""Stage A healthcare few-shot examples (table-level entity classification)."""
from __future__ import annotations

from typing import Any

HEALTHCARE_STAGE_A: list[dict[str, Any]] = [
    {
        "input": {
            "table_name": "patient",
            "columns": "patient_id (STRING), gender (STRING), "
            "current_age (INT), os_status (STRING), os_months (DOUBLE), "
            "dfs_status (STRING), dfs_months (DOUBLE), "
            "smoking_status (STRING), stage_highest (STRING)",
        },
        "output": {
            "primary_entity": "Patient",
            "grain_hypothesis": "one row per patient",
            "secondary_entity_hints": [
                "cancer diagnosis", "survival outcome",
            ],
            "ambiguity_flags": [],
            "confidence": 0.95,
        },
    },
    {
        "input": {
            "table_name": "sample",
            "columns": "sample_id (STRING), patient_id (STRING), "
            "cancer_type (STRING), cancer_type_detailed (STRING), "
            "sample_type (STRING), tmb (DOUBLE), msi_type (STRING), "
            "oncotree_code (STRING), sample_class (STRING)",
        },
        "output": {
            "primary_entity": "Biospecimen/Sample",
            "grain_hypothesis": "one row per tumor sample "
            "(multiple samples per patient)",
            "secondary_entity_hints": ["tumor characterization"],
            "ambiguity_flags": [],
            "confidence": 0.9,
        },
    },
    {
        "input": {
            "table_name": "mutation",
            "columns": "sample_id (STRING), hugo_symbol (STRING), "
            "variant_classification (STRING), hgvsp_short (STRING), "
            "chromosome (STRING), start_position (INT), "
            "end_position (INT), reference_allele (STRING), "
            "tumor_seq_allele2 (STRING), mutation_status (STRING)",
        },
        "output": {
            "primary_entity": "Somatic Mutation",
            "grain_hypothesis": "one row per variant call per sample",
            "secondary_entity_hints": [
                "gene", "protein change",
            ],
            "ambiguity_flags": [],
            "confidence": 0.9,
        },
    },
    {
        "input": {
            "table_name": "treatment",
            "columns": "patient_id (STRING), treatment_subtype (STRING), "
            "agent (STRING), start_date (INT), stop_date (INT)",
        },
        "output": {
            "primary_entity": "Treatment Event",
            "grain_hypothesis": "one row per treatment event "
            "(multiple events per patient)",
            "secondary_entity_hints": ["drug/agent", "regimen"],
            "ambiguity_flags": [],
            "confidence": 0.85,
        },
    },
    {
        "input": {
            "table_name": "structural_variant",
            "columns": "sample_id (STRING), site1_gene (STRING), "
            "site2_gene (STRING), sv_class (STRING), "
            "event_info (STRING), annotation (STRING)",
        },
        "output": {
            "primary_entity": "Structural Variant",
            "grain_hypothesis": "one row per structural variant "
            "call per sample",
            "secondary_entity_hints": ["fusion partner genes"],
            "ambiguity_flags": [],
            "confidence": 0.85,
        },
    },
    {
        "input": {
            "table_name": "timeline_labtest",
            "columns": "PATIENT_ID (STRING), START_DATE (INT), "
            "STOP_DATE (INT), EVENT_TYPE (STRING), TEST (STRING), "
            "VALUE (DOUBLE), UNITS (STRING)",
        },
        "output": {
            "primary_entity": "Lab Measurement",
            "grain_hypothesis": "one row per lab measurement event "
            "(many per patient over time)",
            "secondary_entity_hints": [
                "analyte", "unit of measure", "longitudinal observation",
            ],
            "ambiguity_flags": [],
            "confidence": 0.9,
        },
    },
    {
        "input": {
            "table_name": "timeline_surgery",
            "columns": "PATIENT_ID (STRING), START_DATE (INT), "
            "STOP_DATE (INT), EVENT_TYPE (STRING), "
            "PROCEDURE (STRING), PROCEDURE_CODE (STRING), "
            "SURGERY_DETAILS (STRING)",
        },
        "output": {
            "primary_entity": "Procedure Event",
            "grain_hypothesis": "one row per procedure (surgery, "
            "radiation, specimen collection) per patient over time",
            "secondary_entity_hints": [
                "procedure code", "clinical event timeline",
            ],
            "ambiguity_flags": [],
            "confidence": 0.85,
        },
    },
    {
        "input": {
            "table_name": "timeline_performance_status",
            "columns": "PATIENT_ID (STRING), START_DATE (INT), "
            "EVENT_TYPE (STRING), ECOG_SCORE (STRING), "
            "KARNOFSKY_SCORE (STRING)",
        },
        "output": {
            "primary_entity": "Performance Status Assessment",
            "grain_hypothesis": "one row per performance status "
            "assessment per patient over time",
            "secondary_entity_hints": [
                "functional status score", "longitudinal observation",
            ],
            "ambiguity_flags": [],
            "confidence": 0.85,
        },
    },
    {
        "input": {
            "table_name": "cna_segmented",
            "columns": "sample_id (STRING), chrom (STRING), "
            "loc_start (BIGINT), loc_end (BIGINT), "
            "num_mark (BIGINT), seg_mean (DOUBLE)",
        },
        "output": {
            "primary_entity": "Copy Number Segment",
            "grain_hypothesis": "one row per genomic segment per sample "
            "(many segments per sample, segment-level CNA calls)",
            "secondary_entity_hints": [
                "genomic interval", "copy number signal",
            ],
            "ambiguity_flags": [],
            "confidence": 0.85,
        },
    },
]
