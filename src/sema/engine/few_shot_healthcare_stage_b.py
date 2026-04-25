"""Stage B healthcare few-shot examples (column-level semantic typing)."""
from __future__ import annotations

from typing import Any

HEALTHCARE_STAGE_B: list[dict[str, Any]] = [
    {
        "input": {
            "table_name": "patient",
            "column": "patient_id",
            "data_type": "STRING",
            "entity_context": "Patient",
        },
        "output": {
            "canonical_property_label": "patient identifier",
            "semantic_type": "patient identifier",
            "synonyms": ["subject id", "case id", "participant id"],
            "candidate_vocab_families": [],
            "entity_role": "primary_key",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "mutation",
            "column": "sample_id",
            "data_type": "STRING",
            "entity_context": "Somatic Mutation",
        },
        "output": {
            "canonical_property_label": "sample identifier",
            "semantic_type": "specimen/sample identifier",
            "synonyms": ["specimen id", "biospecimen id", "tumor sample id"],
            "candidate_vocab_families": [],
            "entity_role": "foreign_key",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "patient",
            "column": "gender",
            "data_type": "STRING",
            "top_values": "Male, Female, Other",
            "entity_context": "Patient",
        },
        "output": {
            "canonical_property_label": "biological sex",
            "semantic_type": "demographic",
            "synonyms": ["sex", "biological sex"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "treatment",
            "column": "start_date",
            "data_type": "INT",
            "entity_context": "Treatment Event",
        },
        "output": {
            "canonical_property_label": "treatment start date",
            "semantic_type": "temporal field",
            "candidate_vocab_families": [
                "days-from-epoch encoding",
            ],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "sample",
            "column": "cancer_type",
            "data_type": "STRING",
            "top_values": "Non-Small Cell Lung Cancer, "
            "Colorectal Cancer, Breast Cancer",
            "entity_context": "Biospecimen/Sample",
        },
        "output": {
            "canonical_property_label": "cancer type",
            "semantic_type": "diagnosis/condition",
            "candidate_vocab_families": [
                "cancer classification system",
            ],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "sample",
            "column": "cancer_type_detailed",
            "data_type": "STRING",
            "entity_context": "Biospecimen/Sample",
        },
        "output": {
            "canonical_property_label": "cancer subtype",
            "semantic_type": "diagnosis/condition",
            "candidate_vocab_families": [
                "cancer subtype classification",
            ],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "sample",
            "column": "tmb",
            "data_type": "DOUBLE",
            "entity_context": "Biospecimen/Sample",
        },
        "output": {
            "canonical_property_label": "tumor mutational burden",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["tmb", "mutations per megabase", "mutation burden"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "sample",
            "column": "msi_type",
            "data_type": "STRING",
            "top_values": "Instable, Stable",
            "entity_context": "Biospecimen/Sample",
        },
        "output": {
            "canonical_property_label": "microsatellite instability",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["MSI status", "MSI type", "MSI"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "mutation",
            "column": "hugo_symbol",
            "data_type": "STRING",
            "top_values": "TP53, KRAS, EGFR, PIK3CA",
            "entity_context": "Somatic Mutation",
        },
        "output": {
            "canonical_property_label": "gene symbol",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["gene name", "HGNC symbol", "gene"],
            "candidate_vocab_families": [
                "gene symbol namespace",
            ],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "mutation",
            "column": "variant_classification",
            "data_type": "STRING",
            "top_values": "Missense_Mutation, Silent, "
            "Frame_Shift_Del, Nonsense_Mutation",
            "entity_context": "Somatic Mutation",
        },
        "output": {
            "canonical_property_label": "variant effect",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["mutation type", "variant type", "mutation effect"],
            "candidate_vocab_families": [
                "variant effect classification",
            ],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "treatment",
            "column": "agent",
            "data_type": "STRING",
            "top_values": "PACLITAXEL, CAPECITABINE, LETROZOLE",
            "entity_context": "Treatment Event",
        },
        "output": {
            "canonical_property_label": "drug/agent name",
            "semantic_type": "therapy/drug/regimen",
            "synonyms": ["drug", "therapeutic agent", "medication"],
            "candidate_vocab_families": [
                "drug naming system",
            ],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "patient",
            "column": "stage_highest",
            "data_type": "STRING",
            "top_values": "I, II, III, IV, IA, IIB",
            "entity_context": "Patient",
        },
        "output": {
            "canonical_property_label": "highest cancer stage",
            "semantic_type": "diagnosis/condition",
            "synonyms": ["overall stage", "pathologic stage", "tumor stage"],
            "candidate_vocab_families": [
                "cancer staging system",
            ],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "timeline_labtest",
            "column": "VALUE",
            "data_type": "DOUBLE",
            "top_values": "13.5, 8.2, 1.1, 110.0",
            "entity_context": "Lab Measurement",
        },
        "output": {
            "canonical_property_label": "lab value",
            "semantic_type": "measurement",
            "synonyms": ["result value", "test result", "measurement value"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "timeline_labtest",
            "column": "UNITS",
            "data_type": "STRING",
            "top_values": "g/dL, mg/dL, mmol/L, %",
            "entity_context": "Lab Measurement",
        },
        "output": {
            "canonical_property_label": "unit of measure",
            "semantic_type": "unit of measure",
            "synonyms": ["units", "uom", "measurement unit"],
            "candidate_vocab_families": ["unit of measure system"],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "timeline_labtest",
            "column": "TEST",
            "data_type": "STRING",
            "top_values": "Hemoglobin, Creatinine, A1c, Sodium",
            "entity_context": "Lab Measurement",
        },
        "output": {
            "canonical_property_label": "lab test name",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["analyte", "lab test", "assay name"],
            "candidate_vocab_families": ["clinical observation naming system"],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "biomarker",
            "column": "pd_l1_status",
            "data_type": "STRING",
            "top_values": "POSITIVE, NEGATIVE, EQUIVOCAL",
            "entity_context": "Biomarker State",
        },
        "output": {
            "canonical_property_label": "PD-L1 expression class",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["pd-l1 expression", "pdl1 status", "pd-l1 class"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "biomarker",
            "column": "mmr_status",
            "data_type": "STRING",
            "top_values": "Proficient, Deficient",
            "entity_context": "Biomarker State",
        },
        "output": {
            "canonical_property_label": "mismatch repair status",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["mmr proficiency", "mismatch repair", "mmr"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "biomarker",
            "column": "gleason_score",
            "data_type": "STRING",
            "top_values": "6, 7, 8, 9, 10",
            "entity_context": "Biomarker State",
        },
        "output": {
            "canonical_property_label": "gleason grade",
            "semantic_type": "biomarker/gene/variant",
            "synonyms": ["gleason sum", "gleason grade", "prostate grade"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "timeline_performance_status",
            "column": "ECOG_SCORE",
            "data_type": "STRING",
            "top_values": "0, 1, 2, 3, 4",
            "entity_context": "Performance Status Assessment",
        },
        "output": {
            "canonical_property_label": "ECOG performance score",
            "semantic_type": "clinical assessment score",
            "synonyms": ["ECOG", "ECOG PS", "performance status"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "timeline_performance_status",
            "column": "KARNOFSKY_SCORE",
            "data_type": "STRING",
            "top_values": "100, 90, 80, 70, 60",
            "entity_context": "Performance Status Assessment",
        },
        "output": {
            "canonical_property_label": "Karnofsky performance score",
            "semantic_type": "clinical assessment score",
            "synonyms": ["KPS", "karnofsky index", "performance index"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "timeline_surgery",
            "column": "PROCEDURE_CODE",
            "data_type": "STRING",
            "top_values": "0DTJ0ZZ, 0DBJ0ZZ, 0FTG0ZZ",
            "entity_context": "Procedure Event",
        },
        "output": {
            "canonical_property_label": "procedure code",
            "semantic_type": "procedure code",
            "synonyms": ["surgery code", "intervention code", "procedure code"],
            "candidate_vocab_families": ["procedure code system"],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
]
