"""Target ontology adapters.

Declarative protocol for loading target-side ontology declarations into
the planner-contract graph. Mirrors `src/sema/connectors/` structurally
but emits DTOs instead of observed metadata.
"""

from __future__ import annotations

from sema.targets.base import (
    REQUIRED_METHODS,
    TargetOntologyAdapter,
    TargetOntologyAdapterMixin,
)
from sema.targets.exceptions import (
    AdapterContractError,
    AdapterRegistryError,
    AmbiguousAdapterError,
    CardContentDriftError,
    DanglingRefError,
    EnrichmentStatusDivergenceError,
    LoaderStageOrderError,
    NoMatchingAdapterError,
    OverlappingVersionRangeError,
    UnknownAdapterError,
)
from sema.targets.registry import (
    discover_entry_points,
    get,
    list_registered,
    register_target_adapter,
)

__all__ = [
    "TargetOntologyAdapter",
    "TargetOntologyAdapterMixin",
    "REQUIRED_METHODS",
    "register_target_adapter",
    "get",
    "list_registered",
    "discover_entry_points",
    "AdapterContractError",
    "AdapterRegistryError",
    "AmbiguousAdapterError",
    "CardContentDriftError",
    "DanglingRefError",
    "EnrichmentStatusDivergenceError",
    "LoaderStageOrderError",
    "NoMatchingAdapterError",
    "OverlappingVersionRangeError",
    "UnknownAdapterError",
]
