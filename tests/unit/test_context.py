import json

import pytest

pytestmark = pytest.mark.unit

from sema.models.context import (
    SemanticCandidateSet,
    SemanticContextObject,
    ResolvedEntity,
    ResolvedProperty,
    PhysicalAsset,
    JoinPath,
    GovernedValue,
    Provenance,
)


@pytest.fixture
def sample_sco():
    return SemanticContextObject(
        entities=[
            ResolvedEntity(
                name="Cancer Diagnosis",
                description="Primary cancer diagnosis",
                provenance=Provenance(source="llm_interpretation", confidence=0.85, assertion_ids=["a1"]),
                properties=[
                    ResolvedProperty(
                        name="Diagnosis Type",
                        semantic_type="categorical",
                        physical_column="dx_type_cd",
                        physical_table="cdm.clinical.cancer_diagnosis",
                        provenance=Provenance(source="llm_interpretation", confidence=0.8, assertion_ids=["a2"]),
                    ),
                ],
                physical_table="cdm.clinical.cancer_diagnosis",
            ),
        ],
        physical_assets=[
            PhysicalAsset(
                catalog="cdm",
                schema="clinical",
                table="cancer_diagnosis",
                columns=["dx_type_cd", "tnm_stage", "date_of_diagnosis", "patient_id"],
            ),
        ],
        join_paths=[
            JoinPath(
                from_table="cdm.clinical.cancer_diagnosis",
                to_table="cdm.clinical.cancer_surgery",
                on_column="patient_id",
                cardinality="one-to-many",
                confidence=0.8,
            ),
        ],
        governed_values=[
            GovernedValue(
                property_name="Diagnosis Type",
                column="dx_type_cd",
                table="cdm.clinical.cancer_diagnosis",
                values=[
                    {"code": "CRC", "label": "Colorectal Cancer"},
                    {"code": "BRCA", "label": "Breast Cancer"},
                ],
            ),
        ],
        consumer_hint="nl2sql",
    )


class TestSemanticContextObject:
    def test_sco_creation(self, sample_sco):
        assert len(sample_sco.entities) == 1
        assert sample_sco.entities[0].name == "Cancer Diagnosis"
        assert len(sample_sco.physical_assets) == 1
        assert len(sample_sco.join_paths) == 1
        assert len(sample_sco.governed_values) == 1

    def test_sco_json_serialization(self, sample_sco):
        data = sample_sco.model_dump(mode="json")
        json_str = json.dumps(data)
        assert "Cancer Diagnosis" in json_str

    def test_sco_roundtrip(self, sample_sco):
        data = sample_sco.model_dump(mode="json")
        roundtrip = SemanticContextObject.model_validate(data)
        assert roundtrip.entities[0].name == "Cancer Diagnosis"
        assert roundtrip.consumer_hint == "nl2sql"

    def test_sco_consumer_hint(self):
        sco = SemanticContextObject(
            entities=[],
            physical_assets=[],
            join_paths=[],
            governed_values=[],
            consumer_hint="discovery",
        )
        assert sco.consumer_hint == "discovery"

    def test_sco_provenance_on_entities(self, sample_sco):
        entity = sample_sco.entities[0]
        assert entity.provenance.source == "llm_interpretation"
        assert entity.provenance.confidence == 0.85
        assert "a1" in entity.provenance.assertion_ids

    def test_sco_provenance_on_properties(self, sample_sco):
        prop = sample_sco.entities[0].properties[0]
        assert prop.provenance.source == "llm_interpretation"
        assert prop.provenance.confidence == 0.8


class TestSemanticCandidateSet:
    def test_candidate_set_creation(self):
        cs = SemanticCandidateSet(
            query="stage 3 colorectal patients",
            candidates=[
                {"node_type": "Entity", "name": "Cancer Diagnosis", "score": 0.94},
                {"node_type": "Term", "label": "Stage III", "score": 0.91},
                {"node_type": "Term", "label": "Colorectal Cancer", "score": 0.88},
            ],
        )
        assert cs.query == "stage 3 colorectal patients"
        assert len(cs.candidates) == 3

    def test_candidate_set_empty(self):
        cs = SemanticCandidateSet(query="unknown question", candidates=[])
        assert len(cs.candidates) == 0


class TestGovernedValue:
    def test_governed_value_with_codes(self):
        gv = GovernedValue(
            property_name="TNM Stage",
            column="tnm_stage",
            table="cancer_diagnosis",
            values=[
                {"code": "Stage III", "label": "Stage III"},
                {"code": "Stage IIIA", "label": "Stage IIIA"},
                {"code": "Stage IIIB", "label": "Stage IIIB"},
                {"code": "Stage IIIC", "label": "Stage IIIC"},
            ],
        )
        assert len(gv.values) == 4
