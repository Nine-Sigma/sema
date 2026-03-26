import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.models.graph_nodes import (
    Catalog,
    Schema,
    Table,
    Column,
    Entity,
    Property,
    Metric,
    Term,
    ValueSet,
    Synonym,
    Transformation,
    SemanticType,
)


class TestPhysicalNodes:
    def test_catalog(self):
        c = Catalog(name="cdm")
        assert c.name == "cdm"

    def test_schema(self):
        s = Schema(name="clinical", catalog="cdm")
        assert s.name == "clinical"
        assert s.catalog == "cdm"

    def test_table(self):
        t = Table(name="cancer_diagnosis", schema_name="clinical", catalog="cdm")
        assert t.name == "cancer_diagnosis"
        assert t.table_type == "TABLE"

    def test_table_view_type(self):
        t = Table(name="v_patients", schema_name="clinical", catalog="cdm", table_type="VIEW")
        assert t.table_type == "VIEW"

    def test_column(self):
        c = Column(
            name="dx_type_cd",
            table_name="cancer_diagnosis",
            data_type="string",
            nullable=True,
        )
        assert c.name == "dx_type_cd"
        assert c.data_type == "string"
        assert c.nullable is True
        assert c.comment is None


class TestSemanticNodes:
    def test_entity_with_provenance(self):
        e = Entity(
            name="Cancer Diagnosis",
            description="Primary cancer diagnosis record",
            source="llm_interpretation",
            confidence=0.75,
            resolved_at=datetime.now(timezone.utc),
        )
        assert e.name == "Cancer Diagnosis"
        assert e.confidence == 0.75
        assert e.source == "llm_interpretation"
        assert e.resolved_at is not None

    def test_property_with_semantic_type(self):
        p = Property(
            name="Diagnosis Type",
            semantic_type=SemanticType.CATEGORICAL,
            source="llm_interpretation",
            confidence=0.8,
            resolved_at=datetime.now(timezone.utc),
        )
        assert p.semantic_type == SemanticType.CATEGORICAL

    def test_all_semantic_types(self):
        expected = ["identifier", "categorical", "temporal", "numeric", "free_text"]
        for st in expected:
            assert SemanticType(st) is not None

    def test_metric(self):
        m = Metric(
            name="Average Days to Surgery",
            description="Mean days between diagnosis and first surgery",
            formula="AVG(DATEDIFF(surgery_date, diagnosis_date))",
            source="llm_interpretation",
            confidence=0.7,
            resolved_at=datetime.now(timezone.utc),
        )
        assert m.formula is not None

    def test_term(self):
        t = Term(
            code="CRC",
            label="Colorectal Cancer",
            source="llm_interpretation",
            confidence=0.85,
            resolved_at=datetime.now(timezone.utc),
        )
        assert t.code == "CRC"
        assert t.label == "Colorectal Cancer"

    def test_value_set(self):
        vs = ValueSet(name="oncotree_dx_types")
        assert vs.name == "oncotree_dx_types"

    def test_synonym(self):
        s = Synonym(
            text="colon cancer",
            source="llm_interpretation",
            confidence=0.8,
            resolved_at=datetime.now(timezone.utc),
        )
        assert s.text == "colon cancer"

    def test_transformation(self):
        t = Transformation(
            name="stg_cancer_diagnosis",
            transform_type="dbt_model",
            source="dbt",
            confidence=1.0,
            resolved_at=datetime.now(timezone.utc),
        )
        assert t.transform_type == "dbt_model"


class TestJsonSerialization:
    def test_entity_roundtrip(self):
        e = Entity(
            name="Test Entity",
            description="desc",
            source="test",
            confidence=0.9,
            resolved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        data = e.model_dump(mode="json")
        roundtrip = Entity.model_validate(data)
        assert roundtrip.name == e.name
        assert roundtrip.confidence == e.confidence

    def test_column_roundtrip(self):
        c = Column(
            name="col1",
            table_name="tbl1",
            data_type="int",
            nullable=False,
            comment="test comment",
        )
        data = c.model_dump(mode="json")
        roundtrip = Column.model_validate(data)
        assert roundtrip.comment == "test comment"
