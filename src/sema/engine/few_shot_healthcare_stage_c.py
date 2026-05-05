"""Stage C healthcare few-shot examples (value decoding)."""
from __future__ import annotations

from typing import Any

HEALTHCARE_STAGE_C: list[dict[str, Any]] = [
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
    {
        "input": {
            "table_name": "biomarker",
            "column": "pd_l1_status",
            "values": ["POSITIVE (45%)", "NEGATIVE (50%)", "EQUIVOCAL (5%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "POSITIVE",
                 "label": "PD-L1 positive expression (TPS or CPS above threshold)"},
                {"raw": "NEGATIVE",
                 "label": "PD-L1 negative (below assay threshold)"},
                {"raw": "EQUIVOCAL",
                 "label": "PD-L1 equivocal/indeterminate result"},
            ],
            "uncertainty": 0.1,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "biomarker",
            "column": "mmr_status",
            "values": ["Proficient (80%)", "Deficient (20%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Proficient",
                 "label": "mismatch repair proficient (pMMR)"},
                {"raw": "Deficient",
                 "label": "mismatch repair deficient (dMMR / MSI-H surrogate)"},
            ],
            "uncertainty": 0.05,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "biomarker",
            "column": "gleason_score",
            "values": ["6 (10%)", "7 (40%)", "8 (25%)", "9 (15%)", "10 (10%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "6", "label": "Gleason 6 (Grade Group 1, low-grade)"},
                {"raw": "7", "label": "Gleason 7 (Grade Group 2 or 3, intermediate)"},
                {"raw": "8", "label": "Gleason 8 (Grade Group 4, high-grade)"},
                {"raw": "9", "label": "Gleason 9 (Grade Group 5, very high-grade)"},
                {"raw": "10", "label": "Gleason 10 (Grade Group 5, very high-grade)"},
            ],
            "uncertainty": 0.1,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "timeline_performance_status",
            "column": "ECOG_SCORE",
            "values": ["0 (40%)", "1 (35%)", "2 (15%)", "3 (8%)", "4 (2%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "0", "label": "ECOG 0 — fully active, no restriction"},
                {"raw": "1", "label": "ECOG 1 — restricted in strenuous activity"},
                {"raw": "2", "label": "ECOG 2 — ambulatory, self-care, <50% in bed"},
                {"raw": "3", "label": "ECOG 3 — limited self-care, >50% in bed"},
                {"raw": "4", "label": "ECOG 4 — completely disabled, no self-care"},
                {"raw": "5", "label": "ECOG 5 — dead"},
            ],
            "uncertainty": 0.05,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "timeline_performance_status",
            "column": "KARNOFSKY_SCORE",
            "values": [
                "100 (20%)", "90 (25%)", "80 (20%)", "70 (15%)",
                "60 (10%)", "50 (5%)", "40 (3%)", "30 (1%)", "20 (1%)",
            ],
        },
        "output": {
            "decoded_categories": [
                {"raw": "100", "label": "Karnofsky 100 — normal, no complaints"},
                {"raw": "90", "label": "Karnofsky 90 — normal activity, minor symptoms"},
                {"raw": "80", "label": "Karnofsky 80 — normal activity with effort"},
                {"raw": "70", "label": "Karnofsky 70 — cares for self, can't do normal activity"},
                {"raw": "60", "label": "Karnofsky 60 — needs occasional assistance"},
                {"raw": "50", "label": "Karnofsky 50 — needs considerable assistance"},
                {"raw": "40", "label": "Karnofsky 40 — disabled, requires special care"},
                {"raw": "30", "label": "Karnofsky 30 — severely disabled, hospitalization indicated"},
                {"raw": "20", "label": "Karnofsky 20 — very sick, active supportive treatment"},
                {"raw": "10", "label": "Karnofsky 10 — moribund"},
                {"raw": "0", "label": "Karnofsky 0 — dead"},
            ],
            "uncertainty": 0.1,
            "codebook_lookup_needed": False,
        },
    },
]
