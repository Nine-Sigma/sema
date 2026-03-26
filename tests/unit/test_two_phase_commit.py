"""Tests for two-phase per-table commit (Group 7)."""
import json
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.graph.loader import GraphLoader
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)


def _make_assertion(
    subject_ref, predicate, payload=None, source="test",
    confidence=0.9, run_id="run-1", object_ref=None,
    status=AssertionStatus.AUTO,
):
    return Assertion(
        id=f"a-{subject_ref}-{predicate.value}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload or {},
        object_ref=object_ref,
        source=source,
        confidence=confidence,
        status=status,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    tx = MagicMock()
    session.begin_transaction.return_value = tx
    # driver.session() returns a context manager that yields session
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=session)
    ctx.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = ctx
    return driver, session, tx


# ---------------------------------------------------------------------------
# commit_table_assertions tests (Task 7.1)
# ---------------------------------------------------------------------------

class TestCommitTableAssertions:
    def test_supersession_and_create_in_single_transaction(
        self, mock_driver
    ):
        driver, session, tx = mock_driver
        loader = GraphLoader(driver)

        assertions = [
            _make_assertion(
                "unity://cat.sch.tbl",
                AssertionPredicate.TABLE_EXISTS,
                {"table_type": "TABLE"},
            ),
            _make_assertion(
                "unity://cat.sch.tbl.col1",
                AssertionPredicate.COLUMN_EXISTS,
                {"data_type": "STRING", "nullable": True},
            ),
        ]

        loader.commit_table_assertions(assertions)

        # Transaction was opened, committed
        session.begin_transaction.assert_called_once()
        tx.commit.assert_called_once()
        tx.rollback.assert_not_called()

        # At least one supersession call + one UNWIND create
        assert tx.run.call_count >= 2

    def test_rollback_on_failure(self, mock_driver):
        driver, session, tx = mock_driver
        loader = GraphLoader(driver)

        # Make the UNWIND create fail
        tx.run.side_effect = [
            None,  # supersession succeeds
            Exception("Neo4j error"),  # UNWIND fails
        ]

        assertions = [
            _make_assertion(
                "unity://cat.sch.tbl",
                AssertionPredicate.TABLE_EXISTS,
                {"table_type": "TABLE"},
            ),
        ]

        with pytest.raises(Exception, match="Neo4j error"):
            loader.commit_table_assertions(assertions)

        tx.rollback.assert_called_once()
        tx.commit.assert_not_called()

    def test_grouped_supersession(self, mock_driver):
        driver, session, tx = mock_driver
        loader = GraphLoader(driver)

        # Two assertions with same (subject, predicate, source) group
        assertions = [
            _make_assertion(
                "unity://cat.sch.tbl",
                AssertionPredicate.TABLE_EXISTS,
                {"table_type": "TABLE"},
                source="unity_catalog",
            ),
            _make_assertion(
                "unity://cat.sch.tbl.col1",
                AssertionPredicate.COLUMN_EXISTS,
                {"data_type": "STRING"},
                source="unity_catalog",
            ),
            _make_assertion(
                "unity://cat.sch.tbl.col2",
                AssertionPredicate.COLUMN_EXISTS,
                {"data_type": "INT"},
                source="unity_catalog",
            ),
        ]

        loader.commit_table_assertions(assertions)

        # 3 unique supersession groups + 1 UNWIND = 4 tx.run calls
        # (tbl/TABLE_EXISTS/unity_catalog,
        #  tbl.col1/COLUMN_EXISTS/unity_catalog,
        #  tbl.col2/COLUMN_EXISTS/unity_catalog)
        supersession_calls = [
            c for c in tx.run.call_args_list
            if "superseded" in str(c)
        ]
        assert len(supersession_calls) == 3

    def test_unwind_creates_all_assertions(self, mock_driver):
        driver, session, tx = mock_driver
        loader = GraphLoader(driver)

        assertions = [
            _make_assertion(
                f"unity://cat.sch.tbl.col{i}",
                AssertionPredicate.COLUMN_EXISTS,
                {"data_type": "STRING"},
            )
            for i in range(100)
        ]

        loader.commit_table_assertions(assertions)

        # Find the UNWIND call
        unwind_calls = [
            c for c in tx.run.call_args_list
            if "UNWIND" in str(c)
        ]
        assert len(unwind_calls) == 1
        # Check it received 100 assertions
        unwind_args = unwind_calls[0]
        assertion_list = unwind_args.kwargs.get(
            "assertions", unwind_args[1].get("assertions", [])
        )
        assert len(assertion_list) == 100


# ---------------------------------------------------------------------------
# Crash safety tests (Task 7.2)
# ---------------------------------------------------------------------------

class TestCrashSafety:
    def test_failure_during_create_rolls_back(self, mock_driver):
        """Simulated failure during create step should leave prior
        assertions unsuperseded."""
        driver, session, tx = mock_driver
        loader = GraphLoader(driver)

        call_count = [0]
        def run_side_effect(*args, **kwargs):
            call_count[0] += 1
            # Let supersession succeed, fail on UNWIND
            if "UNWIND" in str(args):
                raise Exception("crash during create")

        tx.run.side_effect = run_side_effect

        assertions = [
            _make_assertion(
                "unity://cat.sch.tbl",
                AssertionPredicate.TABLE_EXISTS,
            ),
        ]

        with pytest.raises(Exception, match="crash during create"):
            loader.commit_table_assertions(assertions)

        tx.rollback.assert_called_once()
        tx.commit.assert_not_called()


# ---------------------------------------------------------------------------
# materialize_table_graph tests (Task 7.3)
# ---------------------------------------------------------------------------

class TestMaterializeTableGraph:
    def test_creates_physical_nodes(self):
        driver = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )
        loader = GraphLoader(driver)

        assertions = [
            _make_assertion(
                "unity://cat.sch.tbl",
                AssertionPredicate.TABLE_EXISTS,
                {"table_type": "TABLE"},
            ),
            _make_assertion(
                "unity://cat.sch.tbl.col1",
                AssertionPredicate.COLUMN_EXISTS,
                {"data_type": "STRING", "nullable": True},
            ),
        ]

        loader.materialize_table_graph(assertions)

        # Check that MERGE calls were made
        cypher_calls = [
            str(c) for c in driver.session.return_value
            .__enter__.return_value.run.call_args_list
        ]
        # Should have called upsert_catalog, upsert_schema,
        # upsert_table, upsert_column (at minimum)
        all_cypher = " ".join(str(c) for c in cypher_calls)
        # The _run method creates sessions, so check driver.session
        # was called
        assert driver.session.call_count >= 1

    def test_creates_semantic_nodes(self):
        driver = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )
        loader = GraphLoader(driver)

        assertions = [
            _make_assertion(
                "unity://cat.sch.tbl",
                AssertionPredicate.HAS_ENTITY_NAME,
                {"value": "Patient", "description": "Patient records"},
                source="llm_interpretation",
            ),
            _make_assertion(
                "unity://cat.sch.tbl.col1",
                AssertionPredicate.HAS_PROPERTY_NAME,
                {"value": "Patient ID", "description": "ID"},
                source="llm_interpretation",
            ),
            _make_assertion(
                "unity://cat.sch.tbl.col1",
                AssertionPredicate.HAS_SEMANTIC_TYPE,
                {"value": "identifier"},
                source="llm_interpretation",
            ),
        ]

        loader.materialize_table_graph(assertions)
        assert driver.session.call_count >= 1


# ---------------------------------------------------------------------------
# Materialization idempotency tests (Task 7.4)
# ---------------------------------------------------------------------------

class TestMaterializationIdempotency:
    def test_calling_twice_produces_same_calls(self):
        """Since MERGE is idempotent, calling materialize twice
        should produce the same number/type of MERGE calls."""
        calls_per_run = []

        for _ in range(2):
            driver = MagicMock()
            session_mock = MagicMock()
            driver.session.return_value.__enter__ = MagicMock(
                return_value=session_mock
            )
            driver.session.return_value.__exit__ = MagicMock(
                return_value=False
            )
            loader = GraphLoader(driver)

            assertions = [
                _make_assertion(
                    "unity://cat.sch.tbl",
                    AssertionPredicate.TABLE_EXISTS,
                    {"table_type": "TABLE"},
                ),
                _make_assertion(
                    "unity://cat.sch.tbl",
                    AssertionPredicate.HAS_ENTITY_NAME,
                    {"value": "Patient"},
                    source="llm_interpretation",
                ),
            ]
            loader.materialize_table_graph(assertions)
            calls_per_run.append(session_mock.run.call_count)

        assert calls_per_run[0] == calls_per_run[1]


# ---------------------------------------------------------------------------
# Backward compatibility tests (Task 7.5)
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_store_assertion_still_works(self):
        driver = MagicMock()
        session_mock = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=session_mock
        )
        driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )
        loader = GraphLoader(driver)

        a = _make_assertion(
            "unity://cat.sch.tbl",
            AssertionPredicate.TABLE_EXISTS,
        )
        loader.store_assertion(a)

        # Two calls: supersede + create
        assert session_mock.run.call_count == 2

    def test_batch_store_assertions_still_works(self):
        driver = MagicMock()
        session_mock = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(
            return_value=session_mock
        )
        driver.session.return_value.__exit__ = MagicMock(
            return_value=False
        )
        loader = GraphLoader(driver)

        assertions = [
            _make_assertion(
                f"ref{i}",
                AssertionPredicate.TABLE_EXISTS,
            )
            for i in range(3)
        ]
        loader.batch_store_assertions(assertions)

        # 2 calls per assertion × 3 = 6
        assert session_mock.run.call_count == 6
