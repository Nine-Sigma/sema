"""US-006: provenance SUBJECT/OBJECT edges resolve to physical nodes.

Before the fix (bug-239) ``materialize_provenance_edges`` matched on
``Assertion.subject_id`` — always None — so zero edges were ever written.
The fix resolves ``subject_ref``/``object_ref`` to the physical
:Table/:Column merge key.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from sema.graph.loader import GraphLoader
from sema.graph.provenance_utils import materialize_provenance_edges
from sema.models.assertions import Assertion, AssertionPredicate


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def loader(mock_driver):
    driver, _ = mock_driver
    return GraphLoader(driver)


def _assertion(predicate, subject_ref, object_ref=None, a_id="a-1"):
    return Assertion(
        id=a_id,
        subject_ref=subject_ref,
        predicate=predicate,
        payload={"value": "X"},
        object_ref=object_ref,
        source="llm_interpretation",
        confidence=0.9,
        run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _calls(session):
    return session.run.call_args_list


class TestSubjectEdges:
    def test_entity_name_issues_table_subject_edge(self, loader, mock_driver):
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.HAS_ENTITY_NAME,
            "databricks://ws/cdm/clinical/tbl",
        )
        materialize_provenance_edges(loader, [a])

        assert session.run.call_count == 1
        cypher, kwargs = _calls(session)[0][0][0], _calls(session)[0][1]
        assert "SUBJECT" in cypher
        assert ":Table" in cypher
        assert kwargs["rows"] == [{
            "a_id": "a-1", "name": "tbl",
            "schema_name": "clinical", "catalog": "cdm",
        }]

    def test_property_name_issues_column_subject_edge(
        self, loader, mock_driver,
    ):
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.HAS_PROPERTY_NAME,
            "databricks://ws/cdm/clinical/tbl/col",
        )
        materialize_provenance_edges(loader, [a])

        cypher, kwargs = _calls(session)[0][0][0], _calls(session)[0][1]
        assert "SUBJECT" in cypher
        assert ":Column" in cypher
        assert kwargs["rows"] == [{
            "a_id": "a-1", "name": "col", "table_name": "tbl",
            "schema_name": "clinical", "catalog": "cdm",
        }]

    def test_dotted_column_ref_resolves_same_key(self, loader, mock_driver):
        """L2 emits dotted col refs; they must resolve to the column node."""
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.HAS_SEMANTIC_TYPE,
            "databricks://ws/cdm/clinical/tbl.col",
        )
        materialize_provenance_edges(loader, [a])

        kwargs = _calls(session)[0][1]
        assert kwargs["rows"][0]["name"] == "col"
        assert kwargs["rows"][0]["table_name"] == "tbl"

    def test_no_subject_id_match_remains(self, loader, mock_driver):
        """The old null-MATCH-on-subject_id path is gone."""
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.HAS_ENTITY_NAME,
            "databricks://ws/cdm/clinical/tbl",
        )
        materialize_provenance_edges(loader, [a])
        for c in _calls(session):
            assert "subject_id" not in c[0][0]


class TestObjectEdges:
    def test_join_evidence_issues_object_edge(self, loader, mock_driver):
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.HAS_JOIN_EVIDENCE,
            "databricks://ws/cdm/clinical/orders/customer_id",
            object_ref="databricks://ws/cdm/clinical/customers/id",
        )
        materialize_provenance_edges(loader, [a])

        joined = " ".join(c[0][0] for c in _calls(session))
        assert "SUBJECT" in joined
        assert "OBJECT" in joined
        object_call = next(
            c for c in _calls(session) if "OBJECT" in c[0][0]
        )
        assert object_call[1]["rows"] == [{
            "a_id": "a-1", "name": "id", "table_name": "customers",
            "schema_name": "clinical", "catalog": "cdm",
        }]

    def test_no_object_ref_issues_no_object_edge(self, loader, mock_driver):
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.HAS_ENTITY_NAME,
            "databricks://ws/cdm/clinical/tbl",
        )
        materialize_provenance_edges(loader, [a])
        for c in _calls(session):
            assert "OBJECT" not in c[0][0]


class TestGuards:
    def test_structural_predicate_skipped(self, loader, mock_driver):
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.TABLE_EXISTS,
            "databricks://ws/cdm/clinical/tbl",
        )
        materialize_provenance_edges(loader, [a])
        session.run.assert_not_called()

    def test_unparseable_ref_skipped(self, loader, mock_driver):
        _, session = mock_driver
        a = _assertion(
            AssertionPredicate.HAS_ENTITY_NAME, "not-a-valid-ref",
        )
        materialize_provenance_edges(loader, [a])
        session.run.assert_not_called()

    def test_batches_many_assertions_into_few_statements(
        self, loader, mock_driver,
    ):
        """Many provenance assertions -> one batched UNWIND per kind, not N."""
        _, session = mock_driver
        assertions = [
            _assertion(
                AssertionPredicate.HAS_PROPERTY_NAME,
                f"databricks://ws/cdm/clinical/tbl/col{i}",
                a_id=f"a-{i}",
            )
            for i in range(10)
        ]
        materialize_provenance_edges(loader, assertions)

        assert session.run.call_count == 1
        assert len(_calls(session)[0][1]["rows"]) == 10
