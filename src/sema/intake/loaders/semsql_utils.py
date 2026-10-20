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

DERIVED_RELATIONS = {
    "semsql_terms": f"""
        SELECT c.id AS subject,
               ({_ONTOLOGY}) AS ontology,
               (SELECT min(l.value) FROM rdfs_label_statement l
                 WHERE l.subject = c.id) AS label,
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
    "semsql_edges": """
        SELECT subject, predicate, object FROM edge
         WHERE predicate = 'rdfs:subClassOf'
         ORDER BY subject, object
    """,
    "semsql_synonyms": """
        SELECT subject, value, predicate FROM has_exact_synonym_statement
         ORDER BY subject, value
    """,
    "semsql_mappings": f"""
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
