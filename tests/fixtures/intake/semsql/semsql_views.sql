-- Views copied verbatim from INCATools/semantic-sql
-- commit 110aa8a40689ce8c48bc3efa9d9ea8d3b407d5cf (2026-09-20),
-- file src/semsql/builder/sql_schema/semsql.sql, fetched 2026-10-20.
-- Only the views the semsql loader exposes and their dependencies, in dependency order.
-- Upstream quirk kept as is: has_*_match_statement filters on skos:has*Match, not skos:*Match.

CREATE VIEW rdf_type_statement AS SELECT * FROM statements WHERE predicate='rdf:type';

CREATE VIEW rdfs_subclass_of_statement AS SELECT * FROM statements WHERE predicate='rdfs:subClassOf';

CREATE VIEW rdfs_subclass_of_named_statement AS SELECT * FROM rdfs_subclass_of_statement WHERE object NOT LIKE '_:%';

CREATE VIEW rdfs_subproperty_of_statement AS SELECT * FROM statements WHERE predicate='rdfs:subPropertyOf';

CREATE VIEW rdfs_label_statement AS SELECT * FROM statements WHERE predicate='rdfs:label';

CREATE VIEW class_node AS SELECT distinct subject AS id from rdf_type_statement WHERE object = 'owl:Class';

CREATE VIEW deprecated_node AS SELECT DISTINCT subject AS id FROM statements WHERE predicate='owl:deprecated' AND value='true';

CREATE VIEW has_text_definition_statement AS SELECT * FROM statements WHERE predicate='IAO:0000115';

CREATE VIEW has_exact_synonym_statement AS SELECT * FROM statements WHERE predicate='oio:hasExactSynonym';

CREATE VIEW has_exact_match_statement AS SELECT * FROM statements WHERE predicate='skos:hasExactMatch';

CREATE VIEW has_broad_match_statement AS SELECT * FROM statements WHERE predicate='skos:hasBroadMatch';

CREATE VIEW has_narrow_match_statement AS SELECT * FROM statements WHERE predicate='skos:hasNarrowMatch';

CREATE VIEW has_related_match_statement AS SELECT * FROM statements WHERE predicate='skos:hasRelatedMatch';

CREATE VIEW has_match_statement AS SELECT * FROM has_exact_match_statement UNION SELECT * FROM has_broad_match_statement UNION SELECT * FROM has_narrow_match_statement UNION SELECT * FROM has_related_match_statement;

CREATE VIEW has_dbxref_statement AS SELECT * FROM statements WHERE predicate='oio:hasDbXref';

CREATE VIEW has_mapping_statement AS SELECT * FROM has_match_statement UNION SELECT * FROM has_dbxref_statement;

CREATE VIEW owl_some_values_from AS SELECT onProperty.subject AS id,
            onProperty.object AS on_property,
            f.object AS filler
            FROM
       statements AS onProperty,
       statements AS f
     WHERE
       onProperty.predicate = 'owl:onProperty' AND
       onProperty.subject=f.subject AND
       f.predicate='owl:someValuesFrom';

CREATE VIEW owl_subclass_of_some_values_from AS SELECT subClassOf.stanza,
         subClassOf.subject,
         svf.on_property AS predicate,
         svf.filler AS object
         FROM
    statements AS subClassOf,
    owl_some_values_from AS svf
  WHERE
    subClassOf.predicate = 'rdfs:subClassOf' AND
    svf.id=subClassOf.object;

CREATE VIEW edge AS SELECT subject, predicate, object
  FROM owl_subclass_of_some_values_from
   UNION
  SELECT subject, predicate, object
  FROM rdfs_subclass_of_named_statement
   UNION
  SELECT subject, predicate, object
   FROM rdfs_subproperty_of_statement
   UNION
  SELECT subject, predicate, object
  FROM rdf_type_statement WHERE object IN (SELECT id FROM class_node);
