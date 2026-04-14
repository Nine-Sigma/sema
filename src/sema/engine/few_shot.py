"""Few-shot example library for staged L2 prompts.

Examples are structured data per domain per stage, selectable by
domain+stage key. Healthcare-first implementation for cBioPortal POC.
"""
from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------
# Healthcare Stage A examples (task 8.2)
# --------------------------------------------------------------------------

_HEALTHCARE_STAGE_A: list[dict[str, Any]] = [
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
]

# --------------------------------------------------------------------------
# Healthcare Stage B column examples (task 8.3)
# --------------------------------------------------------------------------

_HEALTHCARE_STAGE_B: list[dict[str, Any]] = [
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
            "candidate_vocab_families": [
                "cancer staging system",
            ],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
]

# --------------------------------------------------------------------------
# Healthcare Stage C value decoding examples (task 8.4)
# --------------------------------------------------------------------------

_HEALTHCARE_STAGE_C: list[dict[str, Any]] = [
    {
        "input": {
            "table_name": "sample",
            "column": "msi_type",
            "values": ["Instable (45%)", "Stable (55%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Instable",
                 "label": "microsatellite instability high (MSI-H)"},
                {"raw": "Stable",
                 "label": "microsatellite stable (MSS)"},
            ],
            "uncertainty": 0.1,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "treatment",
            "column": "treatment_subtype",
            "values": [
                "Immuno (15%)", "Chemo (40%)", "Hormone (20%)",
                "Targeted (15%)", "Other (10%)",
            ],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Immuno", "label": "immunotherapy"},
                {"raw": "Chemo", "label": "chemotherapy"},
                {"raw": "Hormone", "label": "hormonal therapy"},
                {"raw": "Targeted", "label": "targeted therapy"},
                {"raw": "Other", "label": "other therapy type"},
            ],
            "uncertainty": 0.05,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "patient",
            "column": "os_status",
            "values": ["1:DECEASED (40%)", "0:LIVING (60%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "1:DECEASED", "label": "patient died"},
                {"raw": "0:LIVING", "label": "patient alive"},
            ],
            "uncertainty": 0.0,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "sample",
            "column": "sample_type",
            "values": [
                "Primary (60%)", "Metastasis (30%)",
                "Normal (5%)", "Unknown (5%)",
            ],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Primary",
                 "label": "primary tumor site"},
                {"raw": "Metastasis",
                 "label": "metastatic site"},
                {"raw": "Normal",
                 "label": "normal tissue"},
                {"raw": "Unknown",
                 "label": "unknown sample origin"},
            ],
            "uncertainty": 0.05,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "patient",
            "column": "gender",
            "values": ["Male (55%)", "Female (43%)", "Other (2%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Male", "label": "male biological sex"},
                {"raw": "Female", "label": "female biological sex"},
                {"raw": "Other",
                 "label": "other/unspecified biological sex"},
            ],
            "uncertainty": 0.1,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "patient",
            "column": "stage_highest",
            "values": [
                "IV (25%)", "III (20%)", "II (20%)",
                "I (15%)", "IA (8%)", "IIB (7%)", "IIIA (5%)",
            ],
        },
        "output": {
            "decoded_categories": [
                {"raw": "I", "label": "AJCC stage I"},
                {"raw": "IA", "label": "AJCC stage IA"},
                {"raw": "II", "label": "AJCC stage II"},
                {"raw": "IIB", "label": "AJCC stage IIB"},
                {"raw": "III", "label": "AJCC stage III"},
                {"raw": "IIIA", "label": "AJCC stage IIIA"},
                {"raw": "IV", "label": "AJCC stage IV"},
            ],
            "uncertainty": 0.15,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "mutation",
            "column": "variant_classification",
            "values": [
                "Silent (30%)", "Missense_Mutation (45%)",
                "Nonsense_Mutation (10%)", "Frame_Shift_Del (8%)",
                "Splice_Site (7%)",
            ],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Silent",
                 "label": "synonymous, no protein change"},
                {"raw": "Missense_Mutation",
                 "label": "single amino acid change"},
                {"raw": "Nonsense_Mutation",
                 "label": "premature stop codon"},
                {"raw": "Frame_Shift_Del",
                 "label": "frameshift deletion"},
                {"raw": "Splice_Site",
                 "label": "splice site disruption"},
            ],
            "uncertainty": 0.05,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "progression",
            "column": "progression",
            "values": ["Y (35%)", "N (65%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Y",
                 "label": "disease progressed"},
                {"raw": "N",
                 "label": "no disease progression"},
            ],
            "uncertainty": 0.0,
            "codebook_lookup_needed": False,
        },
    },
]

# --------------------------------------------------------------------------
# Registry: domain → stage → examples
# --------------------------------------------------------------------------

_REGISTRY: dict[str, dict[str, list[dict[str, Any]]]] = {
    "healthcare": {
        "A": _HEALTHCARE_STAGE_A,
        "B": _HEALTHCARE_STAGE_B,
        "C": _HEALTHCARE_STAGE_C,
    },
}


def get_examples(
    domain: str | None,
    stage: str,
) -> list[dict[str, Any]]:
    """Look up few-shot examples by domain and stage.

    Returns empty list for unknown/None domain (zero-shot fallback).
    """
    if domain is None:
        return []
    return _REGISTRY.get(domain, {}).get(stage, [])


def format_examples(
    domain: str | None,
    stage: str,
) -> str:
    """Format few-shot examples as a prompt block.

    Returns empty string when no examples available.
    """
    examples = get_examples(domain, stage)
    if not examples:
        return ""

    lines = ["Here are examples of correct output:"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Input: {json.dumps(ex['input'], indent=2)}")
        lines.append(f"Output: {json.dumps(ex['output'], indent=2)}")
    return "\n".join(lines)
