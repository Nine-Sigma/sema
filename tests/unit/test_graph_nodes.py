import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.models.graph_nodes import (
    Alias,
    Catalog,
    Column,
    DataSource,
    Entity,
    JoinPath,
    JoinPredicate,
    Metric,
    Property,
    Schema,
    SemanticType,
    Table,
    Term,
    Transformation,
    ValueSet,
)


class TestPhysicalNodes:
    def test_catalog_has_id_and_ref(self):
        c = Catalog(id="cat-1", ref="databricks://ws/cdm", name="cdm")
        assert c.id == "cat-1"
        assert c.ref == "databricks://ws/cdm"
        assert c.name == "cdm"

    def test_schema_has_id_and_ref(self):
        s = Schema(
            id="sch-1",
            ref="databricks://ws/cdm/clinical",
            name="clinical",
            catalog="cdm",
        )
        assert s.id == "sch-1"
        assert s.ref == "databricks://ws/cdm/clinical"

    def test_table_has_id_and_ref(self):
        t = Table(
            id="tbl-1",
            ref="databricks://ws/cdm/clinical/cancer_diagnosis",
            name="cancer_diagnosis",
            schema_name="clinical",
            catalog="cdm",
        )
        assert t.id == "tbl-1"
        assert t.ref == "databricks://ws/cdm/clinical/cancer_diagnosis"
        assert t.table_type == "TABLE"

    def test_table_view_type(self):
        t = Table(
            id="tbl-2",
            ref="databricks://ws/cdm/clinical/v_patients",
            name="v_patients",
            schema_name="clinical",
            catalog="cdm",
            table_type="VIEW",
        )
        assert t.table_type == "VIEW"

    def test_column_has_id_and_ref(self):
        c = Column(
            id="col-1",
            ref="databricks://ws/cdm/clinical/cancer_diagnosis/dx_type_cd",
            name="dx_type_cd",
            table_name="cancer_diagnosis",
            data_type="string",
            nullable=True,
        )
        assert c.id == "col-1"
        assert c.ref == "databricks://ws/cdm/clinical/cancer_diagnosis/dx_type_cd"
        assert c.comment is None


class TestDataSource:
    def test_create(self):
        ds = DataSource(
            id="ds-1",
            ref="databricks://my-workspace",
            platform="databricks",
            workspace="my-workspace",
        )
        assert ds.id == "ds-1"
        assert ds.ref == "databricks://my-workspace"
        assert ds.platform == "databricks"
        assert ds.workspace == "my-workspace"

    def test_roundtrip(self):
        ds = DataSource(
            id="ds-1",
            ref="databricks://ws",
            platform="databricks",
            workspace="ws",
        )
        data = ds.model_dump(mode="json")
        roundtrip = DataSource.model_validate(data)
        assert roundtrip.id == ds.id
        assert roundtrip.ref == ds.ref


class TestSemanticNodes:
    def test_entity_has_id(self):
        e = Entity(
            id="ent-1",
            name="Cancer Diagnosis",
            description="Primary cancer diagnosis record",
            source="llm_interpretation",
            confidence=0.75,
            resolved_at=datetime.now(timezone.utc),
        )
        assert e.id == "ent-1"
        assert e.name == "Cancer Diagnosis"
        assert e.confidence == 0.75

    def test_entity_embedding_updated_at(self):
        now = datetime.now(timezone.utc)
        e = Entity(
            id="ent-2",
            name="Test",
            source="test",
            confidence=0.9,
            embedding_updated_at=now,
        )
        assert e.embedding_updated_at == now

    def test_entity_embedding_updated_at_default_none(self):
        e = Entity(id="ent-3", name="Test", source="test", confidence=0.9)
        assert e.embedding_updated_at is None

    def test_property_has_id(self):
        p = Property(
            id="prop-1",
            name="Diagnosis Type",
            semantic_type=SemanticType.CATEGORICAL,
            source="llm_interpretation",
            confidence=0.8,
        )
        assert p.id == "prop-1"
        assert p.semantic_type == SemanticType.CATEGORICAL

    def test_property_embedding_updated_at(self):
        p = Property(
            id="prop-2",
            name="Test",
            semantic_type=SemanticType.NUMERIC,
            source="test",
            confidence=0.9,
            embedding_updated_at=datetime.now(timezone.utc),
        )
        assert p.embedding_updated_at is not None

    def test_all_semantic_types(self):
        expected = ["identifier", "categorical", "temporal", "numeric", "free_text"]
        for st in expected:
            assert SemanticType(st) is not None

    def test_metric_has_id_and_grain(self):
        m = Metric(
            id="met-1",
            name="Average Days to Surgery",
            description="Mean days between diagnosis and first surgery",
            formula="AVG(DATEDIFF(surgery_date, diagnosis_date))",
            grain="patient",
            source="llm_interpretation",
            confidence=0.7,
        )
        assert m.id == "met-1"
        assert m.grain == "patient"
        assert m.formula is not None

    def test_metric_grain_default_none(self):
        m = Metric(
            id="met-2",
            name="Count",
            source="test",
            confidence=0.9,
        )
        assert m.grain is None

    def test_metric_embedding_updated_at(self):
        m = Metric(
            id="met-3",
            name="Count",
            source="test",
            confidence=0.9,
            embedding_updated_at=datetime.now(timezone.utc),
        )
        assert m.embedding_updated_at is not None

    def test_term_has_id(self):
        t = Term(
            id="term-1",
            code="CRC",
            label="Colorectal Cancer",
            source="llm_interpretation",
            confidence=0.85,
        )
        assert t.id == "term-1"
        assert t.code == "CRC"

    def test_term_embedding_updated_at(self):
        t = Term(
            id="term-2",
            code="X",
            label="X",
            source="test",
            confidence=0.9,
            embedding_updated_at=datetime.now(timezone.utc),
        )
        assert t.embedding_updated_at is not None

    def test_value_set_has_id(self):
        vs = ValueSet(id="vs-1", name="oncotree_dx_types")
        assert vs.id == "vs-1"
        assert vs.name == "oncotree_dx_types"

    def test_transformation_has_id_and_ref(self):
        t = Transformation(
            id="tr-1",
            ref="dbt://models/marts/fct_revenue",
            name="stg_cancer_diagnosis",
            transform_type="dbt_model",
            source="dbt",
            confidence=1.0,
        )
        assert t.id == "tr-1"
        assert t.ref == "dbt://models/marts/fct_revenue"


class TestAlias:
    def test_create(self):
        a = Alias(
            id="alias-1",
            text="colon cancer",
            description="Common name for colorectal cancer",
            is_preferred=False,
            source="llm_interpretation",
            confidence=0.8,
        )
        assert a.id == "alias-1"
        assert a.text == "colon cancer"
        assert a.description == "Common name for colorectal cancer"
        assert a.is_preferred is False

    def test_preferred_alias(self):
        a = Alias(
            id="alias-2",
            text="Colorectal Cancer",
            is_preferred=True,
            source="llm_interpretation",
            confidence=0.9,
        )
        assert a.is_preferred is True

    def test_description_optional(self):
        a = Alias(
            id="alias-3",
            text="CRC",
            is_preferred=False,
            source="test",
            confidence=0.8,
        )
        assert a.description is None

    def test_embedding_updated_at(self):
        a = Alias(
            id="alias-4",
            text="test",
            is_preferred=False,
            source="test",
            confidence=0.8,
            embedding_updated_at=datetime.now(timezone.utc),
        )
        assert a.embedding_updated_at is not None

    def test_roundtrip(self):
        a = Alias(
            id="alias-5",
            text="bowel cancer",
            description="UK term",
            is_preferred=False,
            source="llm_interpretation",
            confidence=0.85,
            resolved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        data = a.model_dump(mode="json")
        roundtrip = Alias.model_validate(data)
        assert roundtrip.text == a.text
        assert roundtrip.is_preferred == a.is_preferred


class TestJoinPredicate:
    def test_create(self):
        jp = JoinPredicate(
            left_table="databricks://ws/cdm/clinical/diagnosis",
            left_column="patient_id",
            right_table="databricks://ws/cdm/clinical/patient",
            right_column="patient_id",
        )
        assert jp.left_table == "databricks://ws/cdm/clinical/diagnosis"
        assert jp.operator == "="

    def test_custom_operator(self):
        jp = JoinPredicate(
            left_table="t1",
            left_column="c1",
            right_table="t2",
            right_column="c2",
            operator=">=",
        )
        assert jp.operator == ">="


class TestJoinPathNode:
    def test_create(self):
        pred = JoinPredicate(
            left_table="databricks://ws/cdm/clinical/diagnosis",
            left_column="patient_id",
            right_table="databricks://ws/cdm/clinical/patient",
            right_column="patient_id",
        )
        jp = JoinPath(
            id="jp-1",
            name="databricks://ws/cdm/clinical/diagnosis/patient_id=databricks://ws/cdm/clinical/patient/patient_id",
            join_predicates=[pred],
            hop_count=1,
            source="heuristic",
            confidence=0.85,
        )
        assert jp.id == "jp-1"
        assert jp.hop_count == 1
        assert len(jp.join_predicates) == 1
        assert jp.sql_snippet is None
        assert jp.cardinality_hint is None

    def test_with_sql_snippet(self):
        jp = JoinPath(
            id="jp-2",
            name="test-join",
            join_predicates=[],
            hop_count=0,
            sql_snippet="ON a.id = b.id",
            cardinality_hint="one-to-many",
            source="test",
            confidence=0.9,
        )
        assert jp.sql_snippet == "ON a.id = b.id"
        assert jp.cardinality_hint == "one-to-many"

    def test_multi_hop(self):
        preds = [
            JoinPredicate(
                left_table="t1", left_column="id",
                right_table="t2", right_column="t1_id",
            ),
            JoinPredicate(
                left_table="t2", left_column="id",
                right_table="t3", right_column="t2_id",
            ),
        ]
        jp = JoinPath(
            id="jp-3",
            name="t1/id=t2/t1_id|t2/id=t3/t2_id",
            join_predicates=preds,
            hop_count=2,
            source="test",
            confidence=0.8,
        )
        assert jp.hop_count == 2
        assert len(jp.join_predicates) == 2

    def test_roundtrip(self):
        pred = JoinPredicate(
            left_table="t1", left_column="c1",
            right_table="t2", right_column="c2",
        )
        jp = JoinPath(
            id="jp-4",
            name="t1/c1=t2/c2",
            join_predicates=[pred],
            hop_count=1,
            source="test",
            confidence=0.9,
            resolved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        data = jp.model_dump(mode="json")
        roundtrip = JoinPath.model_validate(data)
        assert roundtrip.name == jp.name
        assert len(roundtrip.join_predicates) == 1


class TestJsonSerialization:
    def test_entity_roundtrip(self):
        e = Entity(
            id="ent-rt",
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
            id="col-rt",
            ref="databricks://ws/cdm/clinical/tbl1/col1",
            name="col1",
            table_name="tbl1",
            data_type="int",
            nullable=False,
            comment="test comment",
        )
        data = c.model_dump(mode="json")
        roundtrip = Column.model_validate(data)
        assert roundtrip.comment == "test comment"
        assert roundtrip.id == "col-rt"
