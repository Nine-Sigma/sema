"""TargetEntityDecl's optional identity block: key, null-key policy, attribute rules."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from sema.models.target.entity import TargetEntityDecl

pytestmark = pytest.mark.unit

REF = {"target_model_id": "m", "qualified_name": "ns.visit", "kind": "TABLE_ROW"}
IDENTITY = {
    "key_columns": ["visit_id", "site"],
    "exclude_null_keys": True,
    "attributes": [
        {"column": "admitted_on", "rule": "min"},
        {"column": "ward", "rule": "min_by", "by": "admitted_on",
         "prefer": {"column": "ward_kind", "value": "primary"}},
        {"column": "admitted_on", "rule": "max", "name": "last_admitted_on"},
    ],
}  # fmt: skip


def _decl(identity: dict[str, Any] | None = None) -> TargetEntityDecl:
    data: dict[str, Any] = {"ref": REF}
    if identity is not None:
        data["identity"] = identity
    return TargetEntityDecl.model_validate(data)


def test_an_absent_identity_leaves_the_serialized_entity_unchanged() -> None:
    decl = _decl()
    assert decl.identity is None
    assert set(decl.model_dump(mode="json")) == {
        "ref", "properties", "completeness", "endpoints", "declared_joins", "declared_domains",
    }
    assert "identity" not in decl.model_dump_json()


def test_an_identity_block_is_parsed_and_serialized() -> None:
    identity = _decl(IDENTITY).identity
    assert identity is not None
    assert identity.key_columns == ("visit_id", "site")
    assert identity.exclude_null_keys is True
    ward = identity.attributes[1]
    assert (ward.rule, ward.by, ward.output_name) == ("min_by", "admitted_on", "ward")
    assert ward.prefer is not None and ward.prefer.value == "primary"
    assert identity.attributes[2].output_name == "last_admitted_on"
    assert _decl(IDENTITY).model_dump(mode="json")["identity"]["key_columns"] == [
        "visit_id", "site",
    ]


def test_null_keys_group_by_default() -> None:
    identity = _decl({"key_columns": ["visit_id"]}).identity
    assert identity is not None
    assert (identity.exclude_null_keys, identity.null_keys_group) == (False, True)


def _attr(**rule: Any) -> dict[str, Any]:
    return {"key_columns": ["visit_id"], "attributes": [{"column": "ward", **rule}]}


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ({"key_columns": []}, "at least 1"),
        ({"key_columns": ["visit_id", "visit_id"]}, "repeat"),
        (_attr(rule="min_by"), "requires by"),
        (_attr(rule="max_by"), "requires by"),
        (_attr(rule="min", by="admitted_on"), "takes no by"),
        (_attr(rule="median"), "rule"),
        ({"key_columns": ["ward"], "attributes": [{"column": "ward", "rule": "min"}]},
         "key column"),
        ({"key_columns": ["visit_id"], "attributes": [
            {"column": "ward", "rule": "min"}, {"column": "ward", "rule": "max"}]}, "repeat"),
        ({"key_columns": ["visit_id"], "exclude_null_keys": True, "null_keys_group": True},
         "null_keys_group"),
        ({"key_columns": ["visit_id"], "grain": "one row per visit"}, "Extra inputs"),
        ({"key_columns": ["visit_id"], "exclude_null_keys": ["ward"]}, "not a key column"),
        ({"key_columns": ["visit_id"], "exclude_null_keys": []}, "at least one key"),
        ({"key_columns": ["visit_id", "site"], "exclude_null_keys": ["site", "site"]},
         "repeat"),
        ({"key_columns": ["visit_id", "site"], "exclude_null_keys": ["site", "visit_id"],
          "null_keys_group": False}, "null_keys_group"),
        (_attr(rule="min", prefer={"column": "ward_kind", "value": []}), "at least one value"),
        (_attr(rule="min", prefer={"column": "ward_kind", "value": ["a", "a"]}), "repeat"),
    ],
)  # fmt: skip
def test_an_invalid_identity_is_refused(identity: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _decl(identity)


def test_null_keys_are_excluded_by_named_key_columns() -> None:
    identity = _decl({"key_columns": ["visit_id", "site"], "exclude_null_keys": ["visit_id"],
                      "null_keys_group": False}).identity  # fmt: skip
    assert identity is not None
    assert (identity.non_null_keys, identity.nullable_keys) == (("visit_id",), ("site",))
    every_key = _decl(IDENTITY).identity
    assert every_key is not None
    assert (every_key.non_null_keys, every_key.nullable_keys) == (("visit_id", "site"), ())


def test_a_preference_may_rank_several_values() -> None:
    identity = _decl(_attr(rule="min", prefer={"column": "ward_kind",
                                               "value": ["primary", "secondary"]})).identity
    assert identity is not None
    prefer = identity.attributes[0].prefer
    assert prefer is not None and prefer.ranked == ("primary", "secondary")
    single = _decl(IDENTITY).identity
    assert single is not None and single.attributes[1].prefer is not None
    assert single.attributes[1].prefer.ranked == ("primary",)
