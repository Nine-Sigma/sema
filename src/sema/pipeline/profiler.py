"""WarehouseProfiler: domain detection from extraction DTOs.

Runs heuristic keyword matching against table/column names,
with optional LLM confirmation for ambiguous results.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Final

from sema.models.extraction import ExtractedColumn, ExtractedTable
from sema.models.warehouse_profile import WarehouseProfile

DOMAIN_KEYWORDS: Final[dict[str, frozenset[str]]] = {
    "healthcare": frozenset({
        "patient", "diagnosis", "encounter", "provider", "claim",
        "procedure", "medication", "prescription", "clinical",
        "icd", "cpt", "ndc", "loinc", "snomed", "vitals",
        "lab", "observation", "condition", "allergy", "immunization",
        "practitioner", "hospital", "admission", "discharge",
    }),
    "financial": frozenset({
        "account", "transaction", "ledger", "portfolio", "trade",
        "ticker", "cusip", "isin", "naics", "sic", "balance",
        "payment", "invoice", "revenue", "expense", "asset",
        "liability", "equity", "dividend", "interest", "loan",
        "credit", "debit", "settlement", "clearing",
    }),
    "real_estate": frozenset({
        "property", "listing", "parcel", "lease", "tenant",
        "unit", "building", "address", "zoning", "mls",
        "appraisal", "mortgage", "deed", "escrow", "closing",
        "sqft", "bedroom", "bathroom", "lot", "assessment",
        "landlord", "rental", "vacancy", "occupancy",
    }),
    "logistics": frozenset({
        "shipment", "carrier", "tracking", "warehouse", "inventory",
        "order", "delivery", "route", "fleet", "freight",
        "container", "manifest", "customs", "dispatch", "dock",
        "pallet", "sku", "fulfillment", "transit",
    }),
    "general": frozenset({
        "user", "customer", "product", "order", "category",
        "session", "event", "log", "metric", "config",
    }),
}


class WarehouseProfiler:
    """Profile warehouse domain from extraction DTOs."""

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def profile(
        self,
        tables: list[ExtractedTable],
        columns: list[ExtractedColumn],
        datasource_id: str,
        run_id: str,
    ) -> WarehouseProfile:
        """Run heuristic domain detection, optionally confirmed by LLM."""
        domains, evidence = self._heuristic_pass(tables, columns)
        confidence = self._compute_confidence(domains)

        if self._llm and confidence < 0.5:
            domains, evidence = self._llm_confirm(
                domains, evidence, tables, columns,
            )
            confidence = self._compute_confidence(domains)

        return WarehouseProfile(
            profile_id=str(uuid.uuid4()),
            run_id=run_id,
            datasource_id=datasource_id,
            domains=domains,
            evidence=evidence,
            confidence=confidence,
            profiled_at=datetime.now(timezone.utc),
        )

    def _heuristic_pass(
        self,
        tables: list[ExtractedTable],
        columns: list[ExtractedColumn],
    ) -> tuple[dict[str, float], list[str]]:
        """Count keyword matches per domain from table/column names."""
        all_names: list[str] = []
        for t in tables:
            all_names.append(t.name.lower())
        for c in columns:
            all_names.append(c.name.lower())
            all_names.append(c.table_name.lower())

        # Tokenize names by splitting on underscores and common separators
        tokens: list[str] = []
        for name in all_names:
            tokens.extend(
                part for part in name.replace("-", "_").split("_") if part
            )

        domain_counts: Counter[str] = Counter()
        evidence: list[str] = []
        total_tokens = len(tokens)

        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = [t for t in tokens if t in keywords]
            count = len(matches)
            if count > 0:
                domain_counts[domain] = count
                unique = set(matches)
                evidence.append(
                    f"{domain}: {count} matches from "
                    f"{len(unique)} keywords "
                    f"({', '.join(sorted(unique)[:5])})"
                )

        # Normalize to weights
        total_matches = sum(domain_counts.values()) or 1
        domains = {
            domain: round(count / total_matches, 2)
            for domain, count in domain_counts.items()
        }

        evidence.insert(
            0, f"Scanned {len(tables)} tables, "
            f"{len(columns)} columns, {total_tokens} tokens"
        )

        return domains, evidence

    def _compute_confidence(self, domains: dict[str, float]) -> float:
        """Confidence based on how dominant the primary domain is."""
        if not domains:
            return 0.2
        top = max(domains.values())
        if top >= 0.6:
            return 0.85
        if top >= 0.4:
            return 0.6
        return 0.4

    def _llm_confirm(
        self,
        domains: dict[str, float],
        evidence: list[str],
        tables: list[ExtractedTable],
        columns: list[ExtractedColumn],
    ) -> tuple[dict[str, float], list[str]]:
        """Use LLM to refine ambiguous heuristic results.

        Stub — returns heuristic results unchanged if LLM call fails.
        """
        # TODO: implement LLM confirmation prompt
        evidence.append("LLM confirmation: skipped (not implemented)")
        return domains, evidence
