"""Unit tests for sema.graph.materializer_utils."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from tests.conftest import make_assertion
from sema.models.assertions import AssertionPredicate, AssertionStatus

pytestmark = pytest.mark.unit

from sema.graph.materializer_utils import (
    _collect_alias_batch,
    _resolve_property_details,
    apply_resolution_edges,
    parse_ref_any,
    pick_winner,
    run_lifecycle_phase,
    upsert_column_nodes,
    upsert_decoded_values,
    upsert_entities,
    upsert_physical_nodes,
    upsert_properties,
)

REF_TABLE = "databricks://ws/cdm/clinical/patients"
REF_COL = "databricks://ws/cdm/clinical/patients/patient_id"


def _assertion(
    ref: str = REF_TABLE,
    predicate: str = AssertionPredicate.HAS_ENTITY_NAME.value,
    value: str | dict | None = "Patient",
    source: str = "llm_interpretation",
    confidence: float = 0.9,
    status: AssertionStatus = AssertionStatus.AUTO,
) -> MagicMock:
    return make_assertion(
        subject_ref=ref,
        predicate=predicate,
        value=value,
        source=source,
        confidence=confidence,
        status=status,
    )


class TestPickWinner:
    def test_returns_none_for_empty(self):
        assert pick_winner([]) is None

    def test_returns_none_when_all_rejected(self):
        a = _assertion(status=AssertionStatus.REJECTED)
        assert pick_winner([a]) is None

    def test_returns_none_when_all_superseded(self):
        a = _assertion(status=AssertionStatus.SUPERSEDED)
        assert pick_winner([a]) is None

    def test_pinned_wins_over_accepted(self):
        pinned = _assertion(status=AssertionStatus.PINNED, value="Pinned")
        accepted = _assertion(status=AssertionStatus.ACCEPTED, value="Accepted")
        result = pick_winner([accepted, pinned])
        assert result is not None
        assert result.payload["value"] == "Pinned"

    def test_accepted_wins_over_auto(self):
        accepted = _assertion(status=AssertionStatus.ACCEPTED, value="Accepted")
        auto = _assertion(confidence=0.99, value="Auto")
        result = pick_winner([auto, accepted])
        assert result is not None
        assert result.payload["value"] == "Accepted"

    def test_highest_confidence_wins_among_auto(self):
        low = _assertion(confidence=0.5, value="Low")
        high = _assertion(confidence=0.95, value="High")
        result = pick_winner([low, high])
        assert result is not None
        assert result.payload["value"] == "High"

    def test_source_precedence_breaks_confidence_tie(self):
        llm = _assertion(
            confidence=0.9, source="llm_interpretation", value="LLM",
        )
        human = _assertion(confidence=0.9, source="human", value="Human")
        result = pick_winner([llm, human])
        assert result is not None
        assert result.payload["value"] == "Human"

    def test_rejected_filtered_out(self):
        rejected = _assertion(status=AssertionStatus.REJECTED, value="Bad")
        good = _assertion(value="Good")
        result = pick_winner([rejected, good])
        assert result is not None
        assert result.payload["value"] == "Good"


class TestParseRefAny:
    def test_table_ref(self):
        catalog, schema, table, column = parse_ref_any(REF_TABLE)
        assert catalog == "cdm"
        assert schema == "clinical"
        assert table == "patients"
        assert column is None

    def test_column_ref(self):
        catalog, schema, table, column = parse_ref_any(REF_COL)
        assert catalog == "cdm"
        assert schema == "clinical"
        assert table == "patients"
        assert column == "patient_id"


class TestUpsertPhysicalNodes:
    def test_creates_catalog_schema_table(self):
        loader = MagicMock()
        by_subject = {
            REF_TABLE: [
                _assertion(
                    ref=REF_TABLE,
                    predicate=AssertionPredicate.TABLE_EXISTS.value,
                    value={"table_type": "TABLE"},
                ),
            ],
        }
        upsert_physical_nodes(loader, by_subject)
        loader.upsert_catalog.assert_called_with("cdm")
        loader.upsert_schema.assert_called_with("clinical", "cdm")
        loader.upsert_table.assert_called_once()

    def test_skips_non_table_exists(self):
        loader = MagicMock()
        by_subject = {
            REF_TABLE: [
                _assertion(
                    ref=REF_TABLE,
                    predicate=AssertionPredicate.HAS_ENTITY_NAME.value,
                ),
            ],
        }
        upsert_physical_nodes(loader, by_subject)
        loader.upsert_table.assert_not_called()

    def test_table_with_comment(self):
        loader = MagicMock()
        by_subject = {
            REF_TABLE: [
                _assertion(
                    ref=REF_TABLE,
                    predicate=AssertionPredicate.TABLE_EXISTS.value,
                    value={"table_type": "VIEW"},
                ),
                _assertion(
                    ref=REF_TABLE,
                    predicate=AssertionPredicate.HAS_COMMENT.value,
                    value="Patient records",
                ),
            ],
        }
        upsert_physical_nodes(loader, by_subject)
        call_args = loader.upsert_table.call_args
        assert call_args.kwargs.get("comment") == "Patient records"
        assert call_args.kwargs.get("table_type") == "VIEW"


class TestUpsertColumnNodes:
    def test_creates_column_node(self):
        loader = MagicMock()
        by_subject = {
            REF_COL: [
                _assertion(
                    ref=REF_COL,
                    predicate=AssertionPredicate.COLUMN_EXISTS.value,
                    value={"data_type": "STRING", "nullable": False},
                ),
            ],
        }
        upsert_column_nodes(loader, by_subject)
        loader.upsert_column.assert_called_once()
        call_args = loader.upsert_column.call_args
        assert call_args[0][0] == "patient_id"
        assert call_args.kwargs.get("data_type") == "STRING"

    def test_skips_table_ref(self):
        loader = MagicMock()
        by_subject = {
            REF_TABLE: [
                _assertion(
                    ref=REF_TABLE,
                    predicate=AssertionPredicate.COLUMN_EXISTS.value,
                    value={"data_type": "INT"},
                ),
            ],
        }
        upsert_column_nodes(loader, by_subject)
        loader.upsert_column.assert_not_called()


class TestUpsertEntities:
    @patch("sema.graph.materializer_utils.batch_upsert_entities")
    def test_creates_entity_batch(self, mock_batch):
        loader = MagicMock()
        groups = {
            (REF_TABLE, AssertionPredicate.HAS_ENTITY_NAME.value): [
                _assertion(value="Patient"),
            ],
        }
        upsert_entities(loader, groups)
        mock_batch.assert_called_once()
        batch = mock_batch.call_args[0][1]
        assert len(batch) == 1
        assert batch[0]["name"] == "Patient"
        assert batch[0]["table_name"] == "patients"

    @patch("sema.graph.materializer_utils.batch_upsert_entities")
    def test_skips_non_entity_predicates(self, mock_batch):
        loader = MagicMock()
        groups = {
            (REF_TABLE, AssertionPredicate.HAS_PROPERTY_NAME.value): [
                _assertion(predicate=AssertionPredicate.HAS_PROPERTY_NAME.value),
            ],
        }
        upsert_entities(loader, groups)
        batch = mock_batch.call_args[0][1]
        assert len(batch) == 0


class TestResolvePropertyDetails:
    def test_returns_details_for_valid_column(self):
        group = [_assertion(
            ref=REF_COL,
            predicate=AssertionPredicate.HAS_PROPERTY_NAME.value,
            value="Patient ID",
        )]
        groups = {
            (REF_COL, AssertionPredicate.HAS_PROPERTY_NAME.value): group,
            (REF_COL, AssertionPredicate.HAS_SEMANTIC_TYPE.value): [
                _assertion(
                    ref=REF_COL,
                    predicate=AssertionPredicate.HAS_SEMANTIC_TYPE.value,
                    value="identifier",
                ),
            ],
            (REF_TABLE, AssertionPredicate.HAS_ENTITY_NAME.value): [
                _assertion(value="Patient"),
            ],
        }
        result = _resolve_property_details(REF_COL, group, groups)
        assert result is not None
        assert result["name"] == "Patient ID"
        assert result["semantic_type"] == "identifier"
        assert result["entity_name"] == "Patient"
        assert result["column_name"] == "patient_id"
        assert result["table_name"] == "patients"

    def test_returns_none_for_no_winner(self):
        group = [_assertion(status=AssertionStatus.REJECTED)]
        result = _resolve_property_details(REF_COL, group, {})
        assert result is None

    def test_returns_none_for_invalid_ref(self):
        group = [_assertion(value="test")]
        result = _resolve_property_details("not-a-valid-ref", group, {})
        assert result is None

    def test_returns_none_for_table_ref(self):
        group = [_assertion(
            ref=REF_TABLE,
            predicate=AssertionPredicate.HAS_PROPERTY_NAME.value,
            value="SomeProperty",
        )]
        result = _resolve_property_details(REF_TABLE, group, {})
        assert result is None

    def test_defaults_semantic_type_to_free_text(self):
        group = [_assertion(
            ref=REF_COL,
            predicate=AssertionPredicate.HAS_PROPERTY_NAME.value,
            value="Name",
        )]
        groups = {
            (REF_COL, AssertionPredicate.HAS_PROPERTY_NAME.value): group,
        }
        result = _resolve_property_details(REF_COL, group, groups)
        assert result is not None
        assert result["semantic_type"] == "free_text"

    def test_falls_back_to_table_name_for_entity(self):
        group = [_assertion(
            ref=REF_COL,
            predicate=AssertionPredicate.HAS_PROPERTY_NAME.value,
            value="Name",
        )]
        result = _resolve_property_details(REF_COL, group, {})
        assert result is not None
        assert result["entity_name"] == "patients"


class TestUpsertDecodedValues:
    @patch("sema.graph.materializer_utils.batch_upsert_terms")
    @patch("sema.graph.materializer_utils.batch_upsert_value_sets")
    def test_creates_value_sets_and_terms(self, mock_vs, mock_terms):
        loader = MagicMock()
        groups = {
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                _assertion(
                    ref=REF_COL,
                    predicate=AssertionPredicate.HAS_DECODED_VALUE.value,
                    value={"raw": "M", "label": "Male"},
                ),
                _assertion(
                    ref=REF_COL,
                    predicate=AssertionPredicate.HAS_DECODED_VALUE.value,
                    value={"raw": "F", "label": "Female"},
                ),
            ],
        }
        upsert_decoded_values(loader, groups)

        mock_vs.assert_called_once()
        vs_batch = mock_vs.call_args[0][1]
        assert len(vs_batch) == 1
        assert vs_batch[0]["name"] == "patients_patient_id_values"

        mock_terms.assert_called_once()
        term_batch = mock_terms.call_args[0][1]
        assert len(term_batch) == 2
        assert term_batch[0]["code"] == "M"
        assert term_batch[0]["label"] == "Male"

    @patch("sema.graph.materializer_utils.batch_upsert_terms")
    @patch("sema.graph.materializer_utils.batch_upsert_value_sets")
    def test_skips_rejected_decoded_values(self, mock_vs, mock_terms):
        loader = MagicMock()
        groups = {
            (REF_COL, AssertionPredicate.HAS_DECODED_VALUE.value): [
                _assertion(
                    ref=REF_COL,
                    predicate=AssertionPredicate.HAS_DECODED_VALUE.value,
                    value={"raw": "X", "label": "Unknown"},
                    status=AssertionStatus.REJECTED,
                ),
            ],
        }
        upsert_decoded_values(loader, groups)
        term_batch = mock_terms.call_args[0][1]
        assert len(term_batch) == 0

    @patch("sema.graph.materializer_utils.batch_upsert_terms")
    @patch("sema.graph.materializer_utils.batch_upsert_value_sets")
    def test_skips_invalid_ref(self, mock_vs, mock_terms):
        loader = MagicMock()
        groups = {
            ("bad-ref", AssertionPredicate.HAS_DECODED_VALUE.value): [
                _assertion(
                    ref="bad-ref",
                    predicate=AssertionPredicate.HAS_DECODED_VALUE.value,
                    value={"raw": "X"},
                ),
            ],
        }
        upsert_decoded_values(loader, groups)
        vs_batch = mock_vs.call_args[0][1]
        assert len(vs_batch) == 0


class TestCollectAliasBatch:
    def test_column_alias(self):
        group = [
            _assertion(
                ref=REF_COL,
                predicate=AssertionPredicate.HAS_ALIAS.value,
                value="pat_id",
            ),
        ]
        groups = {
            (REF_COL, AssertionPredicate.HAS_PROPERTY_NAME.value): [
                _assertion(
                    ref=REF_COL,
                    predicate=AssertionPredicate.HAS_PROPERTY_NAME.value,
                    value="Patient ID",
                ),
            ],
        }
        batch, label = _collect_alias_batch(REF_COL, group, groups)
        assert label == ":Property"
        assert len(batch) == 1
        assert batch[0]["text"] == "pat_id"
        assert batch[0]["parent_name"] == "Patient ID"

    def test_table_alias(self):
        group = [
            _assertion(
                ref=REF_TABLE,
                predicate=AssertionPredicate.HAS_ALIAS.value,
                value="pts",
            ),
        ]
        groups = {
            (REF_TABLE, AssertionPredicate.HAS_ENTITY_NAME.value): [
                _assertion(value="Patient"),
            ],
        }
        batch, label = _collect_alias_batch(REF_TABLE, group, groups)
        assert label == ":Entity"
        assert len(batch) == 1
        assert batch[0]["text"] == "pts"
        assert batch[0]["parent_name"] == "Patient"

    def test_invalid_ref_returns_empty(self):
        group = [_assertion(ref="bad-ref", value="alias")]
        batch, label = _collect_alias_batch("bad-ref", group, {})
        assert batch == []
        assert label == ":Entity"

    def test_skips_rejected_aliases(self):
        group = [
            _assertion(
                ref=REF_COL,
                predicate=AssertionPredicate.HAS_ALIAS.value,
                value="bad_alias",
                status=AssertionStatus.REJECTED,
            ),
        ]
        batch, _ = _collect_alias_batch(REF_COL, group, {})
        assert len(batch) == 0


class TestApplyResolutionEdges:
    def test_creates_hierarchy_edges(self):
        loader = MagicMock()
        groups = {
            ("ref", AssertionPredicate.PARENT_OF.value): [
                _assertion(
                    predicate=AssertionPredicate.PARENT_OF.value,
                    value={"parent": "NEOPLASM", "child": "CARCINOMA"},
                ),
            ],
        }
        apply_resolution_edges(loader, groups)
        loader.add_term_hierarchy.assert_called_once_with(
            parent_code="NEOPLASM", child_code="CARCINOMA",
        )

    def test_skips_rejected_hierarchy(self):
        loader = MagicMock()
        groups = {
            ("ref", AssertionPredicate.PARENT_OF.value): [
                _assertion(
                    predicate=AssertionPredicate.PARENT_OF.value,
                    value={"parent": "A", "child": "B"},
                    status=AssertionStatus.REJECTED,
                ),
            ],
        }
        apply_resolution_edges(loader, groups)
        loader.add_term_hierarchy.assert_not_called()


class TestRunLifecyclePhase:
    def test_deprecates_inactive_vocabularies(self):
        loader = MagicMock()
        assertions = [
            _assertion(
                predicate=AssertionPredicate.VOCABULARY_MATCH.value,
                value="ICD-10",
            ),
        ]
        run_lifecycle_phase(loader, assertions)
        loader._run.assert_called_once()
        call_kwargs = loader._run.call_args
        assert "ICD-10" in call_kwargs.kwargs.get(
            "active_names", call_kwargs[1].get("active_names", []),
        )

    def test_no_op_when_no_active_vocabs(self):
        loader = MagicMock()
        assertions = [
            _assertion(
                predicate=AssertionPredicate.VOCABULARY_MATCH.value,
                value="stale",
                status=AssertionStatus.REJECTED,
            ),
        ]
        run_lifecycle_phase(loader, assertions)
        loader._run.assert_not_called()
