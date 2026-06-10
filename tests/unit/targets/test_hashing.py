"""SnapshotHasher determinism, schema-only projection, snapshot_hash rejection."""

from __future__ import annotations

import pytest

from sema.models.target.context_card import TargetContextCard
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.properties import TargetPropertyDecl
from sema.targets.exceptions import AdapterContractError
from sema.targets.hashing import SnapshotHasher, compute_card_hash
from sema.targets.normalizer import TargetModelNormalizer

from tests.unit.targets.conftest import (
    ScriptedAdapter,
    make_descriptor,
    make_entity_ref,
    make_obligation,
    make_table_row_entity,
)

pytestmark = pytest.mark.unit


def _build_simple_normalized() -> NormalizedTargetModel:
    entity = make_table_row_entity()
    obligation = make_obligation()
    adapter = ScriptedAdapter(make_descriptor(), [entity], [obligation])
    return TargetModelNormalizer.normalize(adapter)


def test_hash_is_64_char_hex() -> None:
    digest = SnapshotHasher.hash(_build_simple_normalized())
    assert len(digest) == 64
    int(digest, 16)


def test_hasher_determinism_same_input_same_digest() -> None:
    digest1 = SnapshotHasher.hash(_build_simple_normalized())
    digest2 = SnapshotHasher.hash(_build_simple_normalized())
    assert digest1 == digest2


def test_hasher_mutated_property_type_changes_digest() -> None:
    base = _build_simple_normalized()
    base_digest = SnapshotHasher.hash(base)
    mutated_entity = base.entities[0].model_copy(
        update={
            "properties": [
                TargetPropertyDecl(name="person_id", type="integer", nullable=False)
            ]
        }
    )
    mutated = base.model_copy(update={"entities": [mutated_entity]})
    assert SnapshotHasher.hash(mutated) != base_digest


def test_hasher_descriptor_display_name_and_owner_excluded() -> None:
    base = _build_simple_normalized()
    mutated_descriptor = base.descriptor.model_copy(
        update={"display_name": "Different", "owner": "someone-else"}
    )
    mutated = base.model_copy(update={"descriptor": mutated_descriptor})
    assert SnapshotHasher.hash(mutated) == SnapshotHasher.hash(base)


def test_hasher_context_card_changes_excluded_from_schema_hash() -> None:
    base = _build_simple_normalized()
    card = TargetContextCard(
        entity_ref=base.entities[0].ref,
        card_version="1.0.0",
        description="Original",
    )
    with_card = base.model_copy(update={"context_cards": [card]})
    different_card = card.model_copy(update={"description": "Different"})
    with_diff_card = base.model_copy(update={"context_cards": [different_card]})
    assert SnapshotHasher.hash(with_card) == SnapshotHasher.hash(with_diff_card)


def test_hasher_rejects_snapshot_hash_field_anywhere() -> None:
    base = _build_simple_normalized()
    descriptor_dict = base.descriptor.model_dump(mode="json")
    descriptor_dict["snapshot_hash"] = "fake"

    class _ToyModel:
        descriptor = base.descriptor
        entities: list[TargetEntityDecl] = base.entities
        obligations: list[TargetObligationDecl] = base.obligations
        vocabularies = base.vocabularies
        vocabulary_bindings = base.vocabulary_bindings
        terms = base.terms
        context_cards = base.context_cards

        def model_dump(self, mode: str = "json") -> dict[str, object]:  # noqa: ARG002
            return descriptor_dict

    from sema.targets.hashing import _scan_for_snapshot_hash_field

    with pytest.raises(AdapterContractError, match="snapshot_hash"):
        _scan_for_snapshot_hash_field({"some_dto": {"snapshot_hash": "x"}})


def test_card_hash_determinism() -> None:
    card = TargetContextCard(
        entity_ref=make_entity_ref(),
        card_version="1.0.0",
        description="Hello",
        examples=["a", "b"],
        curated_synonyms=["alias"],
    )
    h1 = compute_card_hash(card)
    h2 = compute_card_hash(card)
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)


def test_card_hash_changes_on_content_mutation() -> None:
    card_a = TargetContextCard(
        entity_ref=make_entity_ref(), card_version="1.0.0", description="A"
    )
    card_b = card_a.model_copy(update={"description": "B"})
    assert compute_card_hash(card_a) != compute_card_hash(card_b)


def test_card_hash_changes_when_synonyms_mutate() -> None:
    base = TargetContextCard(
        entity_ref=make_entity_ref(),
        card_version="1.0.0",
        description="Hi",
        curated_synonyms=["x"],
    )
    mutated = base.model_copy(update={"curated_synonyms": ["x", "y"]})
    assert compute_card_hash(base) != compute_card_hash(mutated)
