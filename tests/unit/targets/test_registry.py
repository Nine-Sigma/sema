"""Version-aware adapter registry tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from sema.targets import (
    AdapterRegistryError,
    AmbiguousAdapterError,
    NoMatchingAdapterError,
    OverlappingVersionRangeError,
    UnknownAdapterError,
    discover_entry_points,
    get,
    list_registered,
    register_target_adapter,
)


def _make_subclass(base: type, name: str) -> type:
    return type(name, (base,), {})


def test_single_match_resolves(fake_adapter_cls: type) -> None:
    register_target_adapter(
        adapter_id="manifest", target_model_id="acris-nyc", supported_versions=""
    )(fake_adapter_cls)
    assert get("manifest", "acris-nyc") is fake_adapter_cls
    assert get("manifest", "acris-nyc", "9.9.9") is fake_adapter_cls


def test_two_non_overlapping_ranges_resolve_per_range(fake_adapter_cls: type) -> None:
    a = _make_subclass(fake_adapter_cls, "OmopV5_0")
    b = _make_subclass(fake_adapter_cls, "OmopV5_4")
    register_target_adapter(
        adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions=">=5.0,<5.4"
    )(a)
    register_target_adapter(
        adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions=">=5.4,<6.0"
    )(b)
    assert get("omop_cdm", "omop-cdm", "5.3.1") is a
    assert get("omop_cdm", "omop-cdm", "5.4") is b
    with pytest.raises(AmbiguousAdapterError):
        get("omop_cdm", "omop-cdm")


def test_overlap_detected_when_ranges_share_interior_only(
    fake_adapter_cls: type,
) -> None:
    """Two non-boundary-aligned ranges that share an interior version
    must trip overlap rejection. e.g. (>=5.0,<5.4) vs (>=5.3,<6.0)
    overlap on [5.3, 5.4)."""
    a = _make_subclass(fake_adapter_cls, "A")
    b = _make_subclass(fake_adapter_cls, "B")
    register_target_adapter(
        adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions=">=5.0,<5.4"
    )(a)
    with pytest.raises(OverlappingVersionRangeError):
        register_target_adapter(
            adapter_id="omop_cdm",
            target_model_id="omop-cdm",
            supported_versions=">=5.3,<6.0",
        )(b)


def test_overlap_detected_for_strict_inequality_ranges(
    fake_adapter_cls: type,
) -> None:
    """`>1,<3` and `>2,<4` overlap on (2, 3); both endpoints exclusive."""
    a = _make_subclass(fake_adapter_cls, "A")
    b = _make_subclass(fake_adapter_cls, "B")
    register_target_adapter(
        adapter_id="x", target_model_id="m", supported_versions=">1,<3"
    )(a)
    with pytest.raises(OverlappingVersionRangeError):
        register_target_adapter(
            adapter_id="x", target_model_id="m", supported_versions=">2,<4"
        )(b)


def test_overlap_with_open_lower_bound(fake_adapter_cls: type) -> None:
    """`<5.0` and `>=4.0,<6.0` overlap on [4.0, 5.0)."""
    a = _make_subclass(fake_adapter_cls, "A")
    b = _make_subclass(fake_adapter_cls, "B")
    register_target_adapter(
        adapter_id="x", target_model_id="m", supported_versions="<5.0"
    )(a)
    with pytest.raises(OverlappingVersionRangeError):
        register_target_adapter(
            adapter_id="x", target_model_id="m", supported_versions=">=4.0,<6.0"
        )(b)


def test_overlapping_ranges_rejected_at_registration(fake_adapter_cls: type) -> None:
    a = _make_subclass(fake_adapter_cls, "A")
    b = _make_subclass(fake_adapter_cls, "B")
    register_target_adapter(
        adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions=">=5.0,<6.0"
    )(a)
    with pytest.raises(OverlappingVersionRangeError):
        register_target_adapter(
            adapter_id="omop_cdm",
            target_model_id="omop-cdm",
            supported_versions=">=5.4,<7.0",
        )(b)


def test_no_matching_version_raises(fake_adapter_cls: type) -> None:
    register_target_adapter(
        adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions=">=5.4,<6.0"
    )(fake_adapter_cls)
    with pytest.raises(NoMatchingAdapterError):
        get("omop_cdm", "omop-cdm", "5.3")


def test_unknown_adapter_id_raises(fake_adapter_cls: type) -> None:
    register_target_adapter(
        adapter_id="manifest", target_model_id="acris-nyc", supported_versions=""
    )(fake_adapter_cls)
    with pytest.raises(UnknownAdapterError) as excinfo:
        get("nonexistent", "anything")
    assert "manifest" in str(excinfo.value)


def test_unknown_target_model_id_raises(fake_adapter_cls: type) -> None:
    register_target_adapter(
        adapter_id="manifest", target_model_id="acris-nyc", supported_versions=""
    )(fake_adapter_cls)
    with pytest.raises(UnknownAdapterError) as excinfo:
        get("manifest", "no-such-model")
    assert "acris-nyc" in str(excinfo.value)


def test_literal_star_wildcard_rejected(fake_adapter_cls: type) -> None:
    with pytest.raises(AdapterRegistryError, match="empty string"):
        register_target_adapter(
            adapter_id="manifest",
            target_model_id="any",
            supported_versions="*",
        )(fake_adapter_cls)


def test_list_returns_sorted_tuples(fake_adapter_cls: type) -> None:
    a = _make_subclass(fake_adapter_cls, "A")
    b = _make_subclass(fake_adapter_cls, "B")
    register_target_adapter(adapter_id="z_adapter", target_model_id="z", supported_versions="")(a)
    register_target_adapter(adapter_id="a_adapter", target_model_id="a", supported_versions="")(b)
    rows = list_registered()
    assert rows == sorted(rows)
    assert ("a_adapter", "a", "") in rows
    assert ("z_adapter", "z", "") in rows


def test_omitted_version_with_multiple_registrations_raises_ambiguous(
    fake_adapter_cls: type,
) -> None:
    a = _make_subclass(fake_adapter_cls, "A")
    b = _make_subclass(fake_adapter_cls, "B")
    register_target_adapter(
        adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions=">=5.0,<5.4"
    )(a)
    register_target_adapter(
        adapter_id="omop_cdm", target_model_id="omop-cdm", supported_versions=">=5.4,<6.0"
    )(b)
    with pytest.raises(AmbiguousAdapterError):
        get("omop_cdm", "omop-cdm")


def test_discover_entry_points_returns_classes_without_registering() -> None:
    found = discover_entry_points("sema.target_adapters")
    assert isinstance(found, list)
    assert list_registered() == []
