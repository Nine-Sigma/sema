"""S1-02 — deterministic identity resolver unit tests (hermetic, temp DuckDB).

The resolver partitions source rows by whether they carry a usable entity key:
- rows WITH a key resolve through the registry to a canonical ``entity_id``;
- rows WITHOUT one route to a typed review disposition (D5) — never a synthetic
  identity, never a silent drop.

The resolver is DOMAIN-GENERIC (D6/R29): the missing-key reason ("person"-flavored
in OMOP) is injected by the policy layer, not named here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.resolve.identity_registry import open_duckdb_identity_registry
from sema.resolve.identity_resolver import (
    DeterministicIdentityResolver,
    IdentitySourceRow,
)

pytestmark = pytest.mark.unit

_REASON = "MISSING_ENTITY_KEY"


@pytest.fixture()
def resolver(tmp_path: Path) -> DeterministicIdentityResolver:
    registry = open_duckdb_identity_registry(str(tmp_path / "identity.duckdb"))
    return DeterministicIdentityResolver(registry, missing_key_reason=_REASON)


def _row(ref: str, key: str | None, ns: str = "study_x") -> IdentitySourceRow:
    return IdentitySourceRow(
        source_namespace=ns, source_entity_key=key, source_row_ref=ref
    )


class TestResolve:
    def test_rows_with_keys_resolve_to_entity_ids(
        self, resolver: DeterministicIdentityResolver
    ) -> None:
        out = resolver.resolve(
            [_row("s1", "P-1"), _row("s2", "P-2")], run_id="run-1"
        )
        assert out.review == []
        ids = {r.source_row_ref: r.entity_id for r in out.resolved}
        assert set(ids) == {"s1", "s2"}
        assert ids["s1"] != ids["s2"]

    def test_many_rows_one_patient_share_entity_id(
        self, resolver: DeterministicIdentityResolver
    ) -> None:
        # multiple conditions for one person -> one registry row, one entity_id.
        out = resolver.resolve(
            [_row("s1", "P-1"), _row("s2", "P-1"), _row("s3", "P-1")], run_id="run-1"
        )
        ids = {r.entity_id for r in out.resolved}
        assert len(ids) == 1
        assert resolver.registry.count() == 1

    def test_registry_is_written(
        self, resolver: DeterministicIdentityResolver
    ) -> None:
        resolver.resolve([_row("s1", "P-1"), _row("s2", "P-2")], run_id="run-1")
        assert resolver.registry.count() == 2

    def test_rerun_is_idempotent(
        self, resolver: DeterministicIdentityResolver
    ) -> None:
        first = resolver.resolve([_row("s1", "P-1")], run_id="run-1")
        second = resolver.resolve([_row("s1", "P-1")], run_id="run-2")
        assert first.resolved[0].entity_id == second.resolved[0].entity_id
        assert resolver.registry.count() == 1

    def test_key_whitespace_normalized(
        self, resolver: DeterministicIdentityResolver
    ) -> None:
        out = resolver.resolve(
            [_row("s1", "P-1"), _row("s2", " P-1 ")], run_id="run-1"
        )
        assert out.resolved[0].entity_id == out.resolved[1].entity_id
        assert out.resolved[1].source_entity_key == "P-1"


class TestMissingKeyDisposition:
    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_missing_key_routes_to_review(
        self, resolver: DeterministicIdentityResolver, blank: str | None
    ) -> None:
        out = resolver.resolve([_row("s1", blank)], run_id="run-1")
        assert out.resolved == []
        assert len(out.review) == 1
        assert out.review[0].source_row_ref == "s1"
        assert out.review[0].reason == _REASON

    def test_missing_key_never_minted(
        self, resolver: DeterministicIdentityResolver
    ) -> None:
        resolver.resolve([_row("s1", None), _row("s2", "")], run_id="run-1")
        # D5: no synthetic person is ever created for a blank key.
        assert resolver.registry.count() == 0

    def test_mixed_batch_partitions_correctly(
        self, resolver: DeterministicIdentityResolver
    ) -> None:
        out = resolver.resolve(
            [_row("s1", "P-1"), _row("s2", None), _row("s3", "P-2")],
            run_id="run-1",
        )
        assert {r.source_row_ref for r in out.resolved} == {"s1", "s3"}
        assert {d.source_row_ref for d in out.review} == {"s2"}
        assert out.resolved_count == 2
        assert out.review_count == 1

    def test_reason_is_policy_supplied_not_hardcoded(self, tmp_path: Path) -> None:
        registry = open_duckdb_identity_registry(str(tmp_path / "i.duckdb"))
        r = DeterministicIdentityResolver(registry, missing_key_reason="CUSTOM")
        out = r.resolve([_row("s1", None)], run_id="run-1")
        assert out.review[0].reason == "CUSTOM"


class TestConstruction:
    def test_blank_reason_rejected(self, tmp_path: Path) -> None:
        registry = open_duckdb_identity_registry(str(tmp_path / "i.duckdb"))
        with pytest.raises(ValueError, match="missing_key_reason"):
            DeterministicIdentityResolver(registry, missing_key_reason="")
