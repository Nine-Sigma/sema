"""Version-aware target ontology adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from sema.targets.base import REQUIRED_METHODS, TargetOntologyAdapter
from sema.targets.exceptions import (
    AmbiguousAdapterError,
    NoMatchingAdapterError,
    UnknownAdapterError,
)
from sema.targets.registry_utils import check_no_overlap, parse_supported_versions

_Key = tuple[str, str]


class _Registration:
    __slots__ = (
        "adapter_id",
        "target_model_id",
        "supported_versions",
        "specifier_set",
        "cls",
        "wildcard_target_model_id",
    )

    def __init__(
        self,
        adapter_id: str,
        target_model_id: str,
        supported_versions: str,
        specifier_set: SpecifierSet,
        cls: type,
        wildcard_target_model_id: bool = False,
    ) -> None:
        self.adapter_id = adapter_id
        self.target_model_id = target_model_id
        self.supported_versions = supported_versions
        self.specifier_set = specifier_set
        self.cls = cls
        self.wildcard_target_model_id = wildcard_target_model_id


class _Registry:
    def __init__(self) -> None:
        self._by_key: dict[_Key, list[_Registration]] = {}

    def clear(self) -> None:
        self._by_key.clear()

    def register(
        self,
        adapter_id: str,
        target_model_id: str,
        supported_versions: str,
        cls: type,
        wildcard_target_model_id: bool = False,
    ) -> None:
        _ensure_protocol_methods(cls)
        spec = parse_supported_versions(supported_versions)
        key = (adapter_id, target_model_id)
        existing = self._by_key.get(key, [])
        check_no_overlap(
            supported_versions,
            spec,
            ((r.supported_versions, r.specifier_set) for r in existing),
            adapter_id,
            target_model_id,
        )
        registration = _Registration(
            adapter_id,
            target_model_id,
            supported_versions,
            spec,
            cls,
            wildcard_target_model_id=wildcard_target_model_id,
        )
        self._by_key.setdefault(key, []).append(registration)

    def get(
        self,
        adapter_id: str,
        target_model_id: str,
        target_model_version: str | None = None,
    ) -> type:
        registrations = self._lookup_or_raise(adapter_id, target_model_id)
        if target_model_version is None:
            return _resolve_versionless(adapter_id, target_model_id, registrations)
        return _resolve_with_version(
            adapter_id, target_model_id, target_model_version, registrations
        )

    def list_all(self) -> list[tuple[str, str, str]]:
        rows = [
            (r.adapter_id, r.target_model_id, r.supported_versions)
            for regs in self._by_key.values()
            for r in regs
        ]
        return sorted(rows)

    def discover_entry_points(self, group: str) -> list[type]:
        eps = entry_points(group=group)
        return [ep.load() for ep in eps]

    def _lookup_or_raise(
        self, adapter_id: str, target_model_id: str
    ) -> list[_Registration]:
        registrations = self._by_key.get((adapter_id, target_model_id))
        if registrations:
            return registrations
        wildcard = self._wildcard_registrations(adapter_id)
        if wildcard:
            return wildcard
        if not any(k[0] == adapter_id for k in self._by_key):
            ids = sorted({k[0] for k in self._by_key})
            raise UnknownAdapterError(
                f"unknown adapter_id={adapter_id!r}; registered ids={ids}"
            )
        models = sorted({k[1] for k in self._by_key if k[0] == adapter_id})
        raise UnknownAdapterError(
            f"unknown target_model_id={target_model_id!r} for adapter_id={adapter_id!r}; "
            f"registered models={models}"
        )

    def _wildcard_registrations(self, adapter_id: str) -> list[_Registration]:
        return [
            r
            for regs in self._by_key.values()
            for r in regs
            if r.adapter_id == adapter_id and r.wildcard_target_model_id
        ]


def _ensure_protocol_methods(cls: type) -> None:
    missing = [m for m in REQUIRED_METHODS if not callable(getattr(cls, m, None))]
    if missing:
        raise TypeError(
            f"{cls.__qualname__} is missing required TargetOntologyAdapter methods: {missing}"
        )


def _resolve_versionless(
    adapter_id: str, target_model_id: str, registrations: list[_Registration]
) -> type:
    if len(registrations) == 1:
        return registrations[0].cls
    candidates = [r.supported_versions for r in registrations]
    raise AmbiguousAdapterError(
        f"({adapter_id!r}, {target_model_id!r}) has multiple registrations "
        f"{candidates}; specify target_model_version to disambiguate"
    )


def _resolve_with_version(
    adapter_id: str,
    target_model_id: str,
    target_model_version: str,
    registrations: list[_Registration],
) -> type:
    version = Version(target_model_version)
    matches = [r for r in registrations if r.specifier_set.contains(version, prereleases=True)]
    if len(matches) == 1:
        return matches[0].cls
    if len(matches) > 1:
        candidates = [m.supported_versions for m in matches]
        raise AmbiguousAdapterError(
            f"version {target_model_version!r} matches multiple registrations "
            f"{candidates} for ({adapter_id!r}, {target_model_id!r})"
        )
    registered = [r.supported_versions for r in registrations]
    raise NoMatchingAdapterError(
        f"no registration matches version {target_model_version!r} for "
        f"({adapter_id!r}, {target_model_id!r}); registered ranges={registered}"
    )


_REGISTRY = _Registry()


def register_target_adapter(
    *,
    adapter_id: str,
    target_model_id: str,
    supported_versions: str = "",
    wildcard_target_model_id: bool = False,
) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        _REGISTRY.register(
            adapter_id,
            target_model_id,
            supported_versions,
            cls,
            wildcard_target_model_id=wildcard_target_model_id,
        )
        return cls

    return decorator


def get(
    adapter_id: str,
    target_model_id: str,
    target_model_version: str | None = None,
) -> type:
    return _REGISTRY.get(adapter_id, target_model_id, target_model_version)


def list_registered() -> list[tuple[str, str, str]]:
    return _REGISTRY.list_all()


def discover_entry_points(group: str = "sema.target_adapters") -> list[type]:
    return _REGISTRY.discover_entry_points(group)


def _clear_for_tests() -> None:
    _REGISTRY.clear()


__all__ = [
    "TargetOntologyAdapter",
    "register_target_adapter",
    "get",
    "list_registered",
    "discover_entry_points",
]
