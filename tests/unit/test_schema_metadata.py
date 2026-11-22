"""Schema metadata models shared by connectors and the lineage resolver (design 6.1.1)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

IMMUTABLE_ERRORS = (AttributeError, TypeError, ValueError)


def _column(name: str, data_type: str = "STRING", comment: str | None = None) -> object:
    from sema.models.schema_metadata import ColumnMetadata

    return ColumnMetadata(name=name, data_type=data_type, comment=comment)


def _object(name: str, view_definition: str | None = None, object_type: str = "TABLE") -> object:
    from sema.models.schema_metadata import ObjectMetadata

    return ObjectMetadata(
        catalog="cat_a",
        schema_name="sch_a",
        name=name,
        object_type=object_type,
        columns=(_column("id", "BIGINT", "row id"), _column("label")),
        view_definition=view_definition,
    )


def test_column_metadata_carries_name_type_and_optional_comment() -> None:
    from sema.models.schema_metadata import ColumnMetadata

    col = ColumnMetadata(name="amount", data_type="DECIMAL(10,2)", comment=None)

    assert (col.name, col.data_type, col.comment) == ("amount", "DECIMAL(10,2)", None)


def test_object_metadata_keeps_columns_in_given_order_as_a_tuple() -> None:
    obj = _object("orders")

    assert isinstance(obj.columns, tuple)  # type: ignore[attr-defined]
    assert [c.name for c in obj.columns] == ["id", "label"]  # type: ignore[attr-defined]
    assert obj.object_type == "TABLE"  # type: ignore[attr-defined]
    assert (obj.catalog, obj.schema_name) == ("cat_a", "sch_a")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("view_definition", "expected"),
    [
        (None, False),
        ("", False),
        ("SELECT id, label FROM cat_a.sch_a.orders", True),
    ],
)
def test_is_view_is_true_exactly_when_view_text_is_non_empty(
    view_definition: str | None, expected: bool
) -> None:
    obj = _object("orders_v", view_definition=view_definition, object_type="VIEW")

    assert obj.is_view is expected  # type: ignore[attr-defined]


def test_is_view_follows_view_text_not_the_reported_object_type() -> None:
    obj = _object("mv", view_definition="SELECT 1 AS id", object_type="MATERIALIZED_VIEW")

    assert obj.is_view is True  # type: ignore[attr-defined]


def test_schema_lookup_by_object_name_ignores_case() -> None:
    from sema.models.schema_metadata import SchemaMetadata

    orders = _object("Orders")
    view = _object("orders_v", view_definition="SELECT 1 AS id")
    meta = SchemaMetadata(catalog="cat_a", schema_name="sch_a", objects=(orders, view))

    assert meta.get("orders") == orders
    assert meta.get("ORDERS") == orders
    assert meta.get("Orders_V") == view
    assert meta.get("missing") is None


def test_unreadable_carries_catalog_schema_and_reason() -> None:
    from sema.models.schema_metadata import Unreadable

    result = Unreadable(catalog="cat_a", schema_name="sch_a", reason="PermissionError: denied")

    assert (result.catalog, result.schema_name, result.reason) == (
        "cat_a",
        "sch_a",
        "PermissionError: denied",
    )


def test_models_are_immutable() -> None:
    from sema.models.schema_metadata import SchemaMetadata, Unreadable

    col = _column("id")
    obj = _object("orders")
    meta = SchemaMetadata(catalog="cat_a", schema_name="sch_a", objects=(obj,))
    unreadable = Unreadable(catalog="cat_a", schema_name="sch_a", reason="x")

    for instance, field in (
        (col, "name"),
        (obj, "name"),
        (meta, "catalog"),
        (unreadable, "reason"),
    ):
        with pytest.raises(IMMUTABLE_ERRORS):
            setattr(instance, field, "changed")
