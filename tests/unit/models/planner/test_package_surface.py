import pytest

pytestmark = pytest.mark.unit


def test_planner_package_importable() -> None:
    import sema.models.planner as planner

    assert planner is not None


def test_models_package_reexports_planner() -> None:
    from sema.models import planner

    assert planner is not None


def test_shared_enums_module_exists() -> None:
    from sema.models.planner import _enums

    assert _enums is not None


def test_shared_refs_module_exists() -> None:
    from sema.models.planner import _refs

    assert _refs is not None
