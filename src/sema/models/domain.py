"""DomainContext: first-class model for domain signal flowing through the pipeline.

Carries declared (user/config) and detected (profiler) domain with confidence,
alternates, and source provenance. Travels independently from BuildConfig.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sema.models.warehouse_profile import WarehouseProfile


class DomainCandidate(BaseModel):
    """An alternative domain hypothesis with confidence."""

    domain: str
    confidence: float = Field(ge=0.0, le=1.0)


class DomainContext(BaseModel):
    """Domain signal that flows through the entire build pipeline."""

    declared_domain: str | None = None
    detected_domain: str | None = None
    domain_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    alternate_domains: list[DomainCandidate] = Field(default_factory=list)
    domain_source: Literal["user", "config", "profiler", "default"] = "default"

    @property
    def effective_domain(self) -> str | None:
        """Resolved domain: declared takes precedence over detected."""
        return self.declared_domain or self.detected_domain


def _domain_context_from_profile(profile: WarehouseProfile) -> DomainContext:
    """Convert a WarehouseProfile into a DomainContext."""
    primary = profile.primary_domain
    if not primary:
        return DomainContext()

    alternates = [
        DomainCandidate(domain=d, confidence=w)
        for d, w in profile.domains.items()
        if d != primary
    ]
    return DomainContext(
        detected_domain=primary,
        domain_confidence=profile.confidence,
        alternate_domains=alternates,
        domain_source="profiler",
    )


def resolve_domain_context(
    *,
    cli_domain: str | None,
    config_domain: str | None,
    profile: WarehouseProfile | None,
) -> DomainContext:
    """Resolve domain with precedence: CLI > config > profiler > default.

    When CLI or config provides a domain, profiler evidence is still
    preserved in detected_domain/alternate_domains so conflict
    handling can compare declared vs detected signals.
    """
    profiler_ctx = _domain_context_from_profile(profile) if profile else None

    if cli_domain:
        return DomainContext(
            declared_domain=cli_domain,
            detected_domain=(
                profiler_ctx.detected_domain if profiler_ctx else None
            ),
            domain_confidence=(
                profiler_ctx.domain_confidence if profiler_ctx else 1.0
            ),
            alternate_domains=(
                profiler_ctx.alternate_domains if profiler_ctx else []
            ),
            domain_source="user",
        )
    if config_domain:
        return DomainContext(
            declared_domain=config_domain,
            detected_domain=(
                profiler_ctx.detected_domain if profiler_ctx else None
            ),
            domain_confidence=(
                profiler_ctx.domain_confidence if profiler_ctx else 1.0
            ),
            alternate_domains=(
                profiler_ctx.alternate_domains if profiler_ctx else []
            ),
            domain_source="config",
        )
    if profiler_ctx:
        return profiler_ctx
    return DomainContext()
