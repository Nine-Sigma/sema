"""S1-03 — person obligation + assertions materialize omop.person.

Proves the person table is produced through the EXISTING assembler (US-008) and
``compile_projection`` (US-014) path — no new compiler machinery. The identity
registry's canonical ``entity_id`` (S1-01) is DIRECT_COPY'd into ``person_id``;
the assembler folds the ``omop.person`` obligation like any other and the plan
passes; the projection materializes a person table with exactly a person_id
column, one distinct row per canonical entity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from sema.compile.compiler import TransformCompiler
from sema.models.planner.lifecycle import PlanVerdict
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.resolve.assembler import Slice0PlanAssembler
from sema.resolve.identity_registry import (
    DEFAULT_SCHEMA,
    DEFAULT_TABLE,
    IdentityRegistry,
)
from sema.resolve.policies.omop import (
    OMOP_PERSON_ID_REF,
    make_person_id_assertion,
    make_person_obligation,
    make_person_row_identity,
)

pytestmark = pytest.mark.unit

# The registry's canonical-id column, addressed as a source ref (the projection
# copies its last dotted segment, ``entity_id``, into ``person_id``).
_ENTITY_ID_REF = f"source.{DEFAULT_TABLE}.entity_id"
_PERSON_SCHEMA = "omop_staging"
_PERSON_TABLE = "person"


def _provenance() -> Provenance:
    return Provenance(
        run=RunProvenance(
            run_id="s1-03",
            target_model_version="omop-condition-slice0-0.2.0",
            target_schema_snapshot_hash="t",
            context_card_version="cc",
            prompt_template_version="pt",
            few_shot_set_version="fs",
            constraint_version="cv",
            llm_model="none",
        ),
        source=SourceScope(
            source_id="msk", source_schema_hash="s", source_profile_hash="p"
        ),
        timestamp=datetime(2026, 7, 22, tzinfo=timezone.utc),
    )


@pytest.fixture()
def conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(tmp_path / "poc.duckdb"))


def _seed_registry(conn: duckdb.DuckDBPyConnection, n: int) -> IdentityRegistry:
    registry = IdentityRegistry(conn)
    registry.get_or_create(
        [("study_x", f"P-{i}") for i in range(1, n + 1)], run_id="s1-03"
    )
    return registry


def _person_plan() -> object:
    assertion = make_person_id_assertion(
        source_entity_id_ref=_ENTITY_ID_REF, provenance=_provenance()
    )
    return Slice0PlanAssembler().assemble(
        [assertion],
        make_person_obligation(),
        make_person_row_identity(_ENTITY_ID_REF),
    )


class TestPersonPlan:
    def test_plan_passes_and_covers_person_id(self) -> None:
        plan = _person_plan()
        assert plan.derive_verdict() is PlanVerdict.compilable
        assert plan.covered_required_fields() == {OMOP_PERSON_ID_REF}

    def test_projection_target_is_person_id_only(self) -> None:
        compiled = TransformCompiler().compile_projection(
            _person_plan(), source_schema=DEFAULT_SCHEMA, source_table=DEFAULT_TABLE
        )
        assert compiled.target_columns == ("person_id",)


class TestPersonMaterialization:
    def test_person_table_has_one_row_per_canonical_entity(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_registry(conn, 3)
        compiled = TransformCompiler().compile_projection(
            _person_plan(), source_schema=DEFAULT_SCHEMA, source_table=DEFAULT_TABLE
        )
        written = TransformCompiler().execute_projection(
            conn, compiled, staging_schema=_PERSON_SCHEMA, staging_table=_PERSON_TABLE
        )
        assert written == 3
        cols = [
            r[1]
            for r in conn.execute(
                f'PRAGMA table_info("{_PERSON_SCHEMA}"."{_PERSON_TABLE}")'
            ).fetchall()
        ]
        assert cols == ["person_id"]
        ids = [
            r[0]
            for r in conn.execute(
                f'SELECT person_id FROM "{_PERSON_SCHEMA}"."{_PERSON_TABLE}" '
                "ORDER BY person_id"
            ).fetchall()
        ]
        assert ids == [1, 2, 3]

    def test_person_ids_are_the_registry_entity_ids(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        registry = _seed_registry(conn, 2)
        entity_ids = {a.entity_id for a in registry.read_all()}
        compiled = TransformCompiler().compile_projection(
            _person_plan(), source_schema=DEFAULT_SCHEMA, source_table=DEFAULT_TABLE
        )
        TransformCompiler().execute_projection(
            conn, compiled, staging_schema=_PERSON_SCHEMA, staging_table=_PERSON_TABLE
        )
        person_ids = {
            r[0]
            for r in conn.execute(
                f'SELECT person_id FROM "{_PERSON_SCHEMA}"."{_PERSON_TABLE}"'
            ).fetchall()
        }
        assert person_ids == entity_ids

    def test_materialization_is_idempotent(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        _seed_registry(conn, 3)
        compiler = TransformCompiler()
        compiled = compiler.compile_projection(
            _person_plan(), source_schema=DEFAULT_SCHEMA, source_table=DEFAULT_TABLE
        )
        first = compiler.execute_projection(
            conn, compiled, staging_schema=_PERSON_SCHEMA, staging_table=_PERSON_TABLE
        )
        second = compiler.execute_projection(
            conn, compiled, staging_schema=_PERSON_SCHEMA, staging_table=_PERSON_TABLE
        )
        assert first == second == 3
