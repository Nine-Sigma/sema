"""Errors raised by the target adapter registry and loader pipeline."""

from __future__ import annotations


class AdapterRegistryError(Exception):
    """Base error for the target adapter registry."""


class UnknownAdapterError(AdapterRegistryError):
    """Raised when registry.get cannot resolve adapter_id or target_model_id."""


class AmbiguousAdapterError(AdapterRegistryError):
    """Raised when registry.get matches more than one registration."""


class NoMatchingAdapterError(AdapterRegistryError):
    """Raised when registry.get finds no adapter for a requested version."""


class OverlappingVersionRangeError(AdapterRegistryError):
    """Raised when a registration overlaps an existing supported_versions range."""


class AdapterContractError(Exception):
    """Raised when an adapter violates its declarative contract."""


class DanglingRefError(AdapterContractError):
    """Raised by the normalizer when a DTO reference cannot be resolved."""


class LoaderStageOrderError(Exception):
    """Raised when the loader pipeline stages run out of order."""


class CardContentDriftError(Exception):
    """Raised when a card's content hash drifts under an unchanged card_version."""


class EnrichmentStatusDivergenceError(Exception):
    """Raised when compact and structured enrichment statuses disagree."""
