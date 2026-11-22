from __future__ import annotations

from pydantic import BaseModel, Field


class HierarchyAnchorConfig(BaseModel):
    """Bounds for walking down from a matched anchor term to governed columns."""

    max_depth: int = Field(default=3, ge=1)
    member_cap: int = Field(default=200, ge=1)
