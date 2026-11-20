"""One schema's objects, columns, and view text as a warehouse reports them."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ColumnMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    data_type: str
    comment: str | None


class ObjectMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    catalog: str
    schema_name: str
    name: str
    object_type: str
    columns: tuple[ColumnMetadata, ...]
    view_definition: str | None

    @property
    def is_view(self) -> bool:
        return bool(self.view_definition)


class SchemaMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    catalog: str
    schema_name: str
    objects: tuple[ObjectMetadata, ...]

    def get(self, name: str) -> ObjectMetadata | None:
        wanted = name.lower()
        return next((o for o in self.objects if o.name.lower() == wanted), None)


class Unreadable(BaseModel):
    """The schema exists or may exist but could not be read (permission, visibility)."""

    model_config = ConfigDict(frozen=True)

    catalog: str
    schema_name: str
    reason: str
