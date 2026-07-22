"""Vocabulary reference knowledge consumed by the semantic engine.

R29: specific source-vocabulary names (e.g. cancer ontologies) are reference
DATA, not engine-core logic. The hierarchy-gating heuristic in
``engine/vocabulary_utils.py`` reads this set so the engine spine names no
specific ontology in its own source.
"""
from __future__ import annotations

HIERARCHICAL_VOCABULARIES = frozenset({
    "icd-10", "atc", "cpt", "snomed", "ajcc",
    "tnm", "oncotree", "loinc", "meddra",
})
