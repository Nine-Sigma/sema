"""Per-vocabulary resolver policy registry (allowlisted domain-literal home).

``resolve_policy`` dispatches a loaded
:class:`~sema.models.target.vocab_binding.VocabularyBindingDecl` to its
concrete :class:`~sema.resolve.policy.ResolverPolicy` via the binding's
``resolver_policy_ref``. New vocabularies register a factory in ``_FACTORIES``.
"""

from __future__ import annotations

import importlib.util
from typing import Callable

from sema._showcase_path import ensure_showcase_importable
from sema.log import logger
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.resolve.policy import ResolverPolicy

PolicyFactory = Callable[[VocabularyBindingDecl], ResolverPolicy]

# Concrete per-vocabulary factories are supplied by showcase packages, not by
# core: the OMOP/OncoTree binding lives in the cBioPortal→OMOP showcase. They are
# registered lazily so core never hard-depends on a showcase and a wheel install
# without one still loads (the registry is simply empty).
_FACTORIES: dict[str, PolicyFactory] = {}


def _load_showcase_factories() -> None:
    if _FACTORIES:
        return
    ensure_showcase_importable()
    if importlib.util.find_spec("showcase") is None:
        return
    try:
        from showcase.cbioportal_to_omop.omop_policy import (
            OMOP_ONCOTREE_CONDITION_REF,
            make_omop_oncotree_condition_policy,
        )
    except ImportError as exc:
        logger.warning(f"showcase present but omop_policy failed to import: {exc}")
        return
    _FACTORIES[OMOP_ONCOTREE_CONDITION_REF] = make_omop_oncotree_condition_policy


class UnknownResolverPolicyError(KeyError):
    """Raised when a binding's ``resolver_policy_ref`` resolves to no policy."""


def resolve_policy(binding: VocabularyBindingDecl) -> ResolverPolicy:
    """Resolve ``binding.resolver_policy_ref`` to its concrete policy."""
    _load_showcase_factories()
    ref = binding.resolver_policy_ref
    factory = _FACTORIES.get(ref) if ref is not None else None
    if factory is None:
        raise UnknownResolverPolicyError(
            f"no resolver policy registered for ref={ref!r}"
        )
    return factory(binding)


__all__ = [
    "UnknownResolverPolicyError",
    "resolve_policy",
]
