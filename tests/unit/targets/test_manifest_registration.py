"""ManifestTargetAdapter registry registration (task 6.17)."""

from __future__ import annotations

import pytest

from sema.targets.adapters.manifest import (
    MANIFEST_ADAPTER_ID,
    MANIFEST_REGISTRY_TARGET_MODEL_ID,
    ManifestTargetAdapter,
    register_manifest_adapter,
)
from sema.targets.registry import get, list_registered

pytestmark = pytest.mark.unit


def test_register_manifest_adapter_registers_under_sentinel() -> None:
    register_manifest_adapter()
    cls = get(MANIFEST_ADAPTER_ID, MANIFEST_REGISTRY_TARGET_MODEL_ID)
    assert cls is ManifestTargetAdapter


def test_manifest_registration_uses_wildcard_supported_versions() -> None:
    register_manifest_adapter()
    rows = list_registered()
    assert (MANIFEST_ADAPTER_ID, MANIFEST_REGISTRY_TARGET_MODEL_ID, "") in rows


def test_register_manifest_adapter_is_idempotent_after_clear() -> None:
    register_manifest_adapter()
    register_manifest_adapter()
    cls = get(MANIFEST_ADAPTER_ID, MANIFEST_REGISTRY_TARGET_MODEL_ID)
    assert cls is ManifestTargetAdapter


def test_manifest_lookup_resolves_for_arbitrary_target_model_id() -> None:
    register_manifest_adapter()
    assert get("manifest", "acris-nyc") is ManifestTargetAdapter
    assert get("manifest", "omop-cdm", "5.4") is ManifestTargetAdapter


def test_manifest_wildcard_does_not_shadow_specific_registration() -> None:
    """A specific (adapter_id, target_model_id) registration overrides
    the wildcard manifest registration for that exact key."""
    from sema.targets.registry import register_target_adapter

    register_manifest_adapter()
    specific = type("SpecificAdapter", (ManifestTargetAdapter,), {})
    register_target_adapter(
        adapter_id="manifest",
        target_model_id="omop-cdm",
        supported_versions="",
    )(specific)
    assert get("manifest", "omop-cdm") is specific
    assert get("manifest", "acris-nyc") is ManifestTargetAdapter


def test_unknown_target_for_non_wildcard_adapter_still_raises(
    fake_adapter_cls: type,
) -> None:
    from sema.targets.registry import register_target_adapter as reg
    from sema.targets import UnknownAdapterError

    reg(adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions="")(
        fake_adapter_cls
    )
    with pytest.raises(UnknownAdapterError):
        get("omop_cdm", "no-such-model")
