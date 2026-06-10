"""TargetObligationDecl is the same class as planner.TargetObligation."""

from __future__ import annotations

import pytest

from sema.models.planner.target_model import TargetObligation
from sema.models.target.obligation import TargetObligationDecl


pytestmark = pytest.mark.unit


def test_target_obligation_decl_is_same_class() -> None:
    assert TargetObligationDecl is TargetObligation
