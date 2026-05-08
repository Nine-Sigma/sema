"""Meta-test: every target DTO is `extra='forbid'` + `frozen=True`."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

import sema.models.target as target_pkg


pytestmark = pytest.mark.unit


def _iter_dto_classes() -> list[type[BaseModel]]:
    classes: list[type[BaseModel]] = []
    for mod_info in pkgutil.iter_modules(target_pkg.__path__):
        mod = importlib.import_module(f"{target_pkg.__name__}.{mod_info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != mod.__name__:
                continue
            if not issubclass(obj, BaseModel):
                continue
            classes.append(obj)
    return classes


def test_every_target_dto_has_extra_forbid_and_frozen() -> None:
    classes = _iter_dto_classes()
    assert classes, "expected at least one DTO under sema.models.target"
    failures: list[str] = []
    for cls in classes:
        config = cls.model_config
        if config.get("extra") != "forbid":
            failures.append(f"{cls.__module__}.{cls.__name__}: extra != forbid")
        if not config.get("frozen"):
            failures.append(f"{cls.__module__}.{cls.__name__}: frozen != True")
    assert not failures, "\n".join(failures)
