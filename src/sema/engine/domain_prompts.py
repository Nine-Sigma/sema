"""Domain-aware prompt composition layers.

Each layer is a standalone function returning a prompt block (or empty
string). Layers are independently toggleable and composed by the stage
prompt builders in stage_utils.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sema.models.domain import DomainContext

_CONFIDENCE_THRESHOLD = 0.4

# -- Domain bias headers per known domain ------------------------------------

_DOMAIN_HEADERS: dict[str, str] = {
    "healthcare": (
        "This table is likely from the healthcare domain. "
        "Prefer interpretations consistent with clinical, genomic, "
        "specimen, diagnosis, treatment, and outcome data unless "
        "evidence suggests otherwise."
    ),
    "financial": (
        "This table is likely from the financial domain. "
        "Prefer interpretations consistent with accounts, transactions, "
        "instruments, risk metrics, and regulatory data unless "
        "evidence suggests otherwise."
    ),
    "real_estate": (
        "This table is likely from the real estate domain. "
        "Prefer interpretations consistent with properties, listings, "
        "leases, assessments, and zoning data unless "
        "evidence suggests otherwise."
    ),
    "logistics": (
        "This table is likely from the logistics domain. "
        "Prefer interpretations consistent with shipments, inventory, "
        "orders, routes, and fulfillment data unless "
        "evidence suggests otherwise."
    ),
}

# -- Semantic type inventories per domain ------------------------------------

_HEALTHCARE_TYPES = (
    "patient identifier, encounter identifier, specimen/sample identifier, "
    "diagnosis/condition, biomarker/gene/variant, therapy/drug/regimen, "
    "lab measurement, outcome/survival, temporal field, demographic, "
    "administrative metadata, free text, unknown/ambiguous"
)

_FINANCIAL_TYPES = (
    "account identifier, transaction identifier, instrument identifier, "
    "monetary amount, risk metric, temporal field, counterparty reference, "
    "classification code, administrative metadata, free text, "
    "unknown/ambiguous"
)

_GENERIC_TYPES = (
    "identifier, categorical, temporal, numeric, free_text, "
    "boolean, ordinal, measurement"
)

_INVENTORY_MAP: dict[str, str] = {
    "healthcare": _HEALTHCARE_TYPES,
    "financial": _FINANCIAL_TYPES,
}

# -- Vocabulary family hints per domain --------------------------------------

_HEALTHCARE_VOCAB_HINTS = (
    "When classifying candidate_vocab_families, consider these "
    "vocabulary families common in healthcare:\n"
    "  - OMOP concept domains (condition, drug, measurement, "
    "observation, procedure)\n"
    "  - SNOMED-like condition/finding concepts\n"
    "  - RxNorm-like drug/ingredient concepts\n"
    "  - HGNC gene symbol namespaces\n"
    "  - LOINC-like measurement/lab test concepts\n"
    "  - Cancer staging systems (AJCC, TNM)\n"
    "  - Cancer classification systems (OncoTree, ICD-O)\n"
    "Remember: name semantic families, not specific ontologies, "
    "unless the column header or values explicitly identify one."
)

_VOCAB_HINTS_MAP: dict[str, str] = {
    "healthcare": _HEALTHCARE_VOCAB_HINTS,
}


def _has_conflict(ctx: DomainContext) -> bool:
    """Check if declared and detected domains disagree sharply."""
    if not ctx.declared_domain or not ctx.detected_domain:
        return False
    if ctx.declared_domain == ctx.detected_domain:
        return False
    # Check alternates for high-confidence disagreement
    for alt in ctx.alternate_domains:
        if alt.domain == ctx.detected_domain and alt.confidence >= 0.6:
            return True
    return ctx.domain_confidence >= 0.6


def build_domain_bias_header(
    ctx: DomainContext | None,
) -> str:
    """Build a domain bias header for prompt injection.

    Returns empty string when domain is absent or confidence too low.
    User-declared domain always applies regardless of confidence.
    """
    if ctx is None:
        return ""

    domain = ctx.effective_domain
    if not domain:
        return ""

    # User-declared always applies; profiler needs confidence
    if ctx.domain_source == "profiler":
        if ctx.domain_confidence < _CONFIDENCE_THRESHOLD:
            return ""

    # Conflict: declared vs detected disagree sharply
    if _has_conflict(ctx):
        return (
            f"User-declared domain: {ctx.declared_domain}. "
            f"Profiler signal suggests {ctx.detected_domain} "
            f"({ctx.domain_confidence:.2f}). "
            f"Interpret with awareness of both domains; prefer "
            f"{ctx.declared_domain} framing unless column evidence "
            f"clearly contradicts."
        )

    return _DOMAIN_HEADERS.get(domain, _generic_header(domain))


def _generic_header(domain: str) -> str:
    return (
        f"This table is likely from the {domain} domain. "
        f"Prefer interpretations consistent with {domain} data "
        f"unless evidence suggests otherwise."
    )


def get_semantic_type_inventory(
    ctx: DomainContext | None,
) -> str:
    """Return the semantic type inventory string for the domain."""
    if ctx is None:
        return _GENERIC_TYPES

    domain = ctx.effective_domain
    if not domain:
        return _GENERIC_TYPES

    return _INVENTORY_MAP.get(domain, _GENERIC_TYPES)


def build_vocab_family_hints(
    ctx: DomainContext | None,
) -> str:
    """Return vocabulary family hints for the domain.

    Returns empty string when no domain or domain has no hints.
    """
    if ctx is None:
        return ""

    domain = ctx.effective_domain
    if not domain:
        return ""

    return _VOCAB_HINTS_MAP.get(domain, "")
