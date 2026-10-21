"""The closed set of views a layout proposal may name (design Decision 4).

Adding a view is adding a module with ``ID``, ``verify``, and ``apply`` and listing it here.
"""

from __future__ import annotations

from sema.intake.views import fill_down
from sema.intake.views.base import View, ViewRefusedError, ViewSpec

__all__ = ["VIEWS", "View", "ViewRefusedError", "ViewSpec"]

VIEWS: tuple[View, ...] = (fill_down,)
