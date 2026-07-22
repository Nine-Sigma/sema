"""S1-01 — generic identity-registry unit tests (hermetic, temp DuckDB).

Freezes the two-level identity schema (D1/D7):
``(source_namespace, source_entity_key) -> source_entity_uid -> entity_id``.

The registry is DOMAIN-GENERIC (D6/R29): it mints a canonical ``entity_id`` for
each distinct source entity and never names ``person``/OMOP anywhere. Stage A
writes an identity map (one entity_id per source entity); Stage B (later) revises
identity by remapping the uid->entity_id level, which is why entity_id is NOT in
the grain. Allocation is atomic get-or-create on the transactional unique key
(D7): a re-run reads the same assignment and never re-mints an existing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.resolve.identity_registry import (
    IdentityRegistry,
    open_duckdb_identity_registry,
)
from sema.resolve.identity_registry_utils import (
    FROZEN_COLUMNS,
    GRAIN_KEY,
    IdentityAssignment,
    source_entity_uid,
)

pytestmark = pytest.mark.unit


# Two-level frozen schema — restated literally ONLY here as the drift anchor.
_SPEC_COLUMNS = (
    "source_namespace",
    "source_entity_key",
    "source_entity_uid",
    "entity_id",
    "run_id",
)

_SPEC_GRAIN = (
    "source_namespace",
    "source_entity_key",
)


@pytest.fixture()
def registry(tmp_path: Path) -> IdentityRegistry:
    return open_duckdb_identity_registry(str(tmp_path / "identity.duckdb"))


class TestFrozenSchema:
    def test_frozen_columns_constant_matches_spec(self) -> None:
        assert FROZEN_COLUMNS == _SPEC_COLUMNS

    def test_grain_key_constant_matches_spec(self) -> None:
        assert GRAIN_KEY == _SPEC_GRAIN

    def test_entity_id_is_not_in_grain(self) -> None:
        # Stage B collapses uids onto a shared entity_id; if entity_id were the
        # grain, a collapse would be an illegal PK change instead of an UPDATE.
        assert "entity_id" not in GRAIN_KEY

    def test_created_table_columns_equal_frozen_list_exactly(
        self, registry: IdentityRegistry
    ) -> None:
        assert tuple(registry.column_names()) == FROZEN_COLUMNS

    def test_column_names_scoped_to_current_catalog(self, tmp_path: Path) -> None:
        import duckdb

        other = tmp_path / "other.duckdb"
        oc = duckdb.connect(str(other))
        IdentityRegistry(oc)
        oc.close()
        conn = duckdb.connect(str(tmp_path / "work.duckdb"))
        reg = IdentityRegistry(conn)
        conn.execute(f"ATTACH '{other}' AS poc")
        assert tuple(reg.column_names()) == FROZEN_COLUMNS

    def test_schema_is_domain_generic(self) -> None:
        # R29: no OMOP identity literal leaks into the frozen contract.
        for banned in ("person", "patient", "condition"):
            assert not any(banned in col for col in FROZEN_COLUMNS)


class TestUid:
    def test_uid_is_deterministic(self) -> None:
        assert source_entity_uid("s", "P-1") == source_entity_uid("s", "P-1")

    def test_uid_differs_by_key(self) -> None:
        assert source_entity_uid("s", "P-1") != source_entity_uid("s", "P-2")

    def test_uid_differs_by_namespace(self) -> None:
        # Same institutional key text in two namespaces is NOT the same uid.
        assert source_entity_uid("a", "P-1") != source_entity_uid("b", "P-1")


class TestGetOrCreate:
    def test_assigns_monotonic_entity_ids(self, registry: IdentityRegistry) -> None:
        out = registry.get_or_create(
            [("study_x", "P-1"), ("study_x", "P-2"), ("study_x", "P-3")],
            run_id="run-1",
        )
        ids = sorted(a.entity_id for a in out.values())
        assert ids == [1, 2, 3]

    def test_stage_a_is_an_identity_map(self, registry: IdentityRegistry) -> None:
        out = registry.get_or_create(
            [("study_x", "P-1"), ("study_x", "P-2")], run_id="run-1"
        )
        # distinct source entities -> distinct uid AND distinct entity_id.
        assert len({a.source_entity_uid for a in out.values()}) == 2
        assert len({a.entity_id for a in out.values()}) == 2

    def test_uid_matches_helper(self, registry: IdentityRegistry) -> None:
        out = registry.get_or_create([("study_x", "P-1")], run_id="run-1")
        got = out[("study_x", "P-1")]
        assert got.source_entity_uid == source_entity_uid("study_x", "P-1")

    def test_rerun_is_idempotent_same_assignment(
        self, registry: IdentityRegistry
    ) -> None:
        first = registry.get_or_create([("study_x", "P-1")], run_id="run-1")
        # A later run (new run_id) must NOT re-mint or change the assignment (D7).
        second = registry.get_or_create([("study_x", "P-1")], run_id="run-2")
        assert first[("study_x", "P-1")].entity_id == second[("study_x", "P-1")].entity_id
        assert registry.count() == 1

    def test_new_keys_continue_from_max(self, registry: IdentityRegistry) -> None:
        registry.get_or_create([("s", "P-1"), ("s", "P-2")], run_id="run-1")
        out = registry.get_or_create([("s", "P-2"), ("s", "P-3")], run_id="run-2")
        assert out[("s", "P-2")].entity_id == 2  # unchanged
        assert out[("s", "P-3")].entity_id == 3  # continues from MAX

    def test_cross_namespace_same_key_does_not_collapse(
        self, registry: IdentityRegistry
    ) -> None:
        out = registry.get_or_create(
            [("gbm", "P-1"), ("msk", "P-1")], run_id="run-1"
        )
        assert out[("gbm", "P-1")].entity_id != out[("msk", "P-1")].entity_id
        assert out[("gbm", "P-1")].source_entity_uid != out[("msk", "P-1")].source_entity_uid

    def test_run_id_is_creating_run_and_immutable(
        self, registry: IdentityRegistry
    ) -> None:
        registry.get_or_create([("s", "P-1")], run_id="run-1")
        registry.get_or_create([("s", "P-1")], run_id="run-2")
        assert registry.get("s", "P-1").run_id == "run-1"

    def test_duplicate_input_keys_deduped(self, registry: IdentityRegistry) -> None:
        out = registry.get_or_create(
            [("s", "P-1"), ("s", "P-1")], run_id="run-1"
        )
        assert registry.count() == 1
        assert out[("s", "P-1")].entity_id == 1

    def test_blank_key_is_rejected_never_minted(
        self, registry: IdentityRegistry
    ) -> None:
        # D5: a missing/blank source key must route to NO_MAP upstream, never
        # receive a synthetic identity. The registry refuses at the boundary.
        with pytest.raises(ValueError, match="source_entity_key"):
            registry.get_or_create([("s", "")], run_id="run-1")


class TestReadBack:
    def test_get_missing_returns_none(self, registry: IdentityRegistry) -> None:
        assert registry.get("s", "NOPE") is None

    def test_read_all_round_trips(self, registry: IdentityRegistry) -> None:
        registry.get_or_create(
            [("s", "P-1"), ("s", "P-2")], run_id="run-1"
        )
        rows = {a.source_entity_key: a for a in registry.read_all()}
        assert set(rows) == {"P-1", "P-2"}
        assert rows["P-1"] == registry.get("s", "P-1")


class TestAssignmentValidation:
    def test_blank_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_entity_key"):
            IdentityAssignment("s", "", "uid", 1, "run-1")

    def test_nonpositive_entity_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            IdentityAssignment("s", "P-1", "uid", 0, "run-1")
