"""Protocol surface tests for TargetOntologyAdapter."""

from __future__ import annotations

import pytest

from sema.targets import (
    TargetOntologyAdapter,
    TargetOntologyAdapterMixin,
    register_target_adapter,
)
from sema.targets.base import REQUIRED_METHODS

pytestmark = pytest.mark.unit


def test_fake_adapter_satisfies_runtime_protocol(fake_adapter_cls: type) -> None:
    adapter = fake_adapter_cls()
    assert isinstance(adapter, TargetOntologyAdapter)


def test_register_rejects_class_missing_required_method() -> None:
    class IncompleteAdapter:
        def describe(self) -> None:
            return None

        def discover_entities(self) -> list[None]:
            return []

        def load_entity(self, ref: object) -> None:
            return None

        def load_vocabulary_bindings(self, ref: object) -> list[None]:
            return []

        def load_context_card(self, ref: object) -> None:
            return None

    with pytest.raises(TypeError) as excinfo:
        register_target_adapter(
            adapter_id="incomplete",
            target_model_id="bad",
            supported_versions="",
        )(IncompleteAdapter)
    assert "load_obligation" in str(excinfo.value)


def test_required_methods_constant_matches_protocol() -> None:
    expected = {
        "describe",
        "discover_entities",
        "load_entity",
        "load_obligation",
        "load_vocabulary_bindings",
        "load_context_card",
    }
    assert set(REQUIRED_METHODS) == expected


def test_default_iter_terms_raises_not_implemented() -> None:
    class BareMixinAdapter(TargetOntologyAdapterMixin):
        pass

    with pytest.raises(NotImplementedError, match="EXTERNAL"):
        next(BareMixinAdapter().iter_terms(vocabulary_ref=None))  # type: ignore[arg-type]
