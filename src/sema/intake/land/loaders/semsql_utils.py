"""Fixed SQL over the semsql views: the derived relations a spec binds directly.

``semsql_terms.synonyms`` is a JSON array of exact synonyms, so a value set
binds them as one column. View names follow the semsql DDL copied in ``tests/fixtures/intake/semsql/
semsql_views.sql`` (upstream commit recorded there). Mappings read the real
SKOS predicates from ``statements``: upstream ``has_*_match_statement`` views
filter on ``skos:has*Match`` and would miss them. An ontology asserts a mapping
without saying how it was made, so the justification is
``semapv:UnspecifiedMatching``.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from sema.binding.spec_sections import (
    ConceptRoles,
    ConceptsSpec,
    DetailRoles,
    DetailsSpec,
    MappingRoles,
    MappingsSpec,
    ParentColumnHierarchy,
    RelationRef,
    StatusRole,
    ValueSetRoles,
    ValueSetsSpec,
)
from sema.intake.roles.lexicon import KEY_ROLE

MAPPING_JUSTIFICATION = "semapv:UnspecifiedMatching"
MAPPING_PREDICATES = (
    "skos:exactMatch",
    "skos:closeMatch",
    "skos:broadMatch",
    "skos:narrowMatch",
    "skos:relatedMatch",
    "skos:hasExactMatch",
    "skos:hasBroadMatch",
    "skos:hasNarrowMatch",
    "skos:hasRelatedMatch",
    "oio:hasDbXref",
)
_PREDICATE_LIST = ", ".join(f"'{p}'" for p in MAPPING_PREDICATES)
_ONTOLOGY = (
    "SELECT min(o.subject) FROM statements o"
    " WHERE o.predicate = 'rdf:type' AND o.object = 'owl:Ontology'"
)

TERMS_RELATION = "semsql_terms"
EDGES_RELATION = "semsql_edges"
MAPPINGS_RELATION = "semsql_mappings"
TERM_LABEL_COLUMN = "label"
SUBCLASS_PREDICATE = "rdfs:subClassOf"

DERIVED_RELATIONS = {
    TERMS_RELATION: f"""
        SELECT c.id AS subject,
               ({_ONTOLOGY}) AS ontology,
               (SELECT min(l.value) FROM rdfs_label_statement l
                 WHERE l.subject = c.id) AS {TERM_LABEL_COLUMN},
               (SELECT min(d.value) FROM has_text_definition_statement d
                 WHERE d.subject = c.id) AS definition,
               CASE WHEN c.id IN (SELECT id FROM deprecated_node)
                    THEN 'true' ELSE 'false' END AS deprecated,
               (SELECT min(r.object) FROM statements r
                 WHERE r.subject = c.id AND r.predicate = 'IAO:0100001') AS replaced_by,
               (SELECT json_group_array(y.value) FROM
                  (SELECT value FROM has_exact_synonym_statement
                    WHERE subject = c.id ORDER BY value) y) AS synonyms
          FROM class_node c
         ORDER BY c.id
    """,
    EDGES_RELATION: f"""
        SELECT subject, predicate, object FROM edge
         WHERE predicate = '{SUBCLASS_PREDICATE}'
         ORDER BY subject, object
    """,
    "semsql_synonyms": """
        SELECT subject, value, predicate FROM has_exact_synonym_statement
         ORDER BY subject, value
    """,
    MAPPINGS_RELATION: f"""
        SELECT subject AS subject_id,
               predicate AS predicate_id,
               coalesce(object, value) AS object_id,
               '{MAPPING_JUSTIFICATION}' AS mapping_justification
          FROM statements
         WHERE predicate IN ({_PREDICATE_LIST})
         ORDER BY subject, predicate, object_id
    """,
}

ONTOLOGY_SQL = (
    "SELECT subject FROM statements"
    " WHERE predicate = 'rdf:type' AND object = 'owl:Ontology'"
    " ORDER BY subject LIMIT 1"
)
VERSION_SQL = (
    "SELECT object FROM statements"
    " WHERE subject = ? AND predicate = 'owl:versionIRI'"
    " ORDER BY object LIMIT 1"
)


def fixed_sections(
    relations: Mapping[str, RelationRef], identity_system: str
) -> dict[str, BaseModel]:
    """The sections a spec binds over the derived relations; none names an identity system."""
    terms, ontology = relations[TERMS_RELATION], "ontology"
    concept = {KEY_ROLE: ontology}
    return {
        "concepts": ConceptsSpec(
            relation=terms,
            group_by=ontology,
            roles=ConceptRoles.model_validate(concept | {"name": ontology}),
        ),
        "value_sets": ValueSetsSpec(
            relation=terms,
            roles=ValueSetRoles.model_validate(concept | {
                "code": "subject",
                "display": TERM_LABEL_COLUMN,
                "synonyms": "synonyms",
                "status": StatusRole(
                    column="deprecated", active_values=["false"], retired_values=["true"]
                ),
                "replaced_by": "replaced_by",
            }),
        ),
        "term_details": DetailsSpec(
            relation=terms, on=["subject", "subject"], roles=DetailRoles(definition="definition")
        ),
        "hierarchy": ParentColumnHierarchy(
            relation=relations[EDGES_RELATION],
            child_column="subject",
            parent_column="object",
            relationship_type=SUBCLASS_PREDICATE,
        ),
        "mappings": MappingsSpec(
            relation=relations[MAPPINGS_RELATION],
            roles=MappingRoles(
                subject_id="subject_id",
                predicate_id="predicate_id",
                object_id="object_id",
                mapping_justification="mapping_justification",
            ),
        ),
    }
