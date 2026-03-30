"""WarehouseProfile: first-class pipeline artifact for domain detection.

Not an assertion — stored alongside assertions in a separate collection.
Has its own lifecycle (AUTO/ACCEPTED/PINNED/REJECTED/SUPERSEDED).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from sema.models.lifecycle import AssertionStatusValue


class WarehouseProfile(BaseModel):
    """Domain profile for a warehouse/datasource."""

    profile_id: str
    run_id: str
    datasource_id: str
    domains: dict[str, float] = Field(
        default_factory=dict,
        description="Domain -> weight (0.0-1.0). E.g. {'healthcare': 0.7}",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Human-readable evidence strings",
    )
    profiler_version: str = "v1"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    status: AssertionStatusValue = AssertionStatusValue.AUTO
    profiled_at: datetime

    @property
    def primary_domain(self) -> str | None:
        """Return the highest-weighted domain, or None if empty."""
        if not self.domains:
            return None
        return max(self.domains, key=self.domains.get)  # type: ignore[arg-type]

    def domain_weight(self, domain: str) -> float:
        """Return weight for a domain, 0.0 if not present."""
        return self.domains.get(domain, 0.0)
