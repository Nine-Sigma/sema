// Migration: backfill source_schema on null-scoped study relationships.
//
// Why: before scope was resolved inside materialize_unified, the public
// default path (source_schema=None) and the legacy materialize_table_graph
// delegate could write study relationships with source_schema=NULL. Those
// edges are invisible to delete_study_scoped (it matches {source_schema: X})
// and to stale-vocabulary deprecation, so a study rebuild can neither remove
// nor refresh them. This migration re-attributes such edges to the study
// schema of the physical node they hang off of.
//
// Scope: edges whose study schema is UNAMBIGUOUSLY derivable from a physical
// anchor are backfilled — edges on a :Column / :Table with a known
// schema_name, plus MEMBER_OF (a :ValueSet is column-scoped, so its single
// owning Column gives the schema). Edges hanging off nodes that may be shared
// across studies (:Property CLASSIFIED_AS, :Term IN_VOCABULARY/PARENT_OF,
// :Alias REFERS_TO) and null-scoped :JoinPath nodes are NOT auto-rewritten —
// re-attributing a shared node to one study would let another study's rebuild
// delete it. Those are surfaced by the residual diagnostic at the end for
// manual review; do not consider the migration complete while it returns rows.
//
// REQUIRED GATE: after running this migration, run the executable validator
//   uv run python scripts/validate_study_scoping.py
// which exits non-zero if any study-derived null-scoped state remains
// (including MEMBER_OF on ownerless/name-keyed ValueSets this migration
// cannot attribute). Do not treat the migration as complete until it passes.
//
// Idempotent: every statement only touches edges where source_schema IS NULL,
// so re-running is a no-op. On a graph built after the root-cause fix there
// are no null-scoped study edges and every statement matches nothing.
//
// Ontology-preloaded edges legitimately carry source_schema=NULL and connect
// to nodes with no schema_name; the schema_name guards leave them untouched.

// --- Diagnostic: count null-scoped study relationships by type ----------
// Run first to see whether any backfill is needed:
// MATCH ()-[r]-() WHERE r.source_schema IS NULL
// RETURN type(r) AS rel, count(r) AS n ORDER BY n DESC;

// --- Backfill: edges anchored to a Column with a known schema -----------
MATCH (c:Column)-[r:HAS_VALUE_SET]->(:ValueSet)
WHERE r.source_schema IS NULL AND c.schema_name IS NOT NULL
SET r.source_schema = c.schema_name;

MATCH (:Property)-[r:PROPERTY_ON_COLUMN]->(c:Column)
WHERE r.source_schema IS NULL AND c.schema_name IS NOT NULL
SET r.source_schema = c.schema_name;

// HAS_PROPERTY (Entity->Property) is written alongside PROPERTY_ON_COLUMN by
// the same path. Derive its schema from the property's owning column(s), but
// only when they agree on one schema (a Property may map across studies).
MATCH (:Entity)-[r:HAS_PROPERTY]->(p:Property)-[:PROPERTY_ON_COLUMN]->(c:Column)
WHERE r.source_schema IS NULL
WITH r, collect(DISTINCT c.schema_name) AS schemas
WHERE size(schemas) = 1 AND schemas[0] IS NOT NULL
SET r.source_schema = schemas[0];

// CLASSIFIED_AS (Property->Vocabulary) is study-derived (Property is built
// from a study column). Derive from the property's owning column when unique.
MATCH (p:Property)-[r:CLASSIFIED_AS]->(:Vocabulary)
WHERE r.source_schema IS NULL
MATCH (p)-[:PROPERTY_ON_COLUMN]->(c:Column)
WITH r, collect(DISTINCT c.schema_name) AS schemas
WHERE size(schemas) = 1 AND schemas[0] IS NOT NULL
SET r.source_schema = schemas[0];

// REFERS_TO into a study Property — derive from the property's column.
MATCH (:Alias)-[r:REFERS_TO]->(p:Property)-[:PROPERTY_ON_COLUMN]->(c:Column)
WHERE r.source_schema IS NULL
WITH r, collect(DISTINCT c.schema_name) AS schemas
WHERE size(schemas) = 1 AND schemas[0] IS NOT NULL
SET r.source_schema = schemas[0];

// REFERS_TO into a study Entity — derive from the entity's table.
MATCH (:Alias)-[r:REFERS_TO]->(e:Entity)-[:ENTITY_ON_TABLE]->(t:Table)
WHERE r.source_schema IS NULL
WITH r, collect(DISTINCT t.schema_name) AS schemas
WHERE size(schemas) = 1 AND schemas[0] IS NOT NULL
SET r.source_schema = schemas[0];

// --- Backfill: edges anchored to a Table with a known schema ------------
MATCH (:Entity)-[r:ENTITY_ON_TABLE]->(t:Table)
WHERE r.source_schema IS NULL AND t.schema_name IS NOT NULL
SET r.source_schema = t.schema_name;

// --- Backfill: MEMBER_OF via the value set's owning column --------------
// A ValueSet is keyed on column_ref, so it normally has one owning Column.
// Legacy/corrupt graphs (the migration's target) could link a ValueSet from
// columns in two schemas, so prove the owning schema is unique before SET;
// ambiguous ones are left for the diagnostic below rather than attributed
// row-order-dependently.
MATCH (c:Column)-[:HAS_VALUE_SET]->(vs:ValueSet)<-[m:MEMBER_OF]-(:Term)
WHERE m.source_schema IS NULL
WITH m, collect(DISTINCT c.schema_name) AS schemas
WHERE size(schemas) = 1 AND schemas[0] IS NOT NULL
SET m.source_schema = schemas[0];

// --- Backfill: JoinPath edges from their (scoped) JoinPath node ---------
// USES / FROM_ENTITY / TO_ENTITY all originate at the JoinPath; when the node
// itself is scoped, its edges inherit that scope. Edges off a null-scoped
// JoinPath node remain residual (the node check above flags the node).
MATCH (jp:JoinPath)-[r:USES]->()
WHERE r.source_schema IS NULL AND jp.source_schema IS NOT NULL
SET r.source_schema = jp.source_schema;

MATCH (jp:JoinPath)-[r:FROM_ENTITY]->()
WHERE r.source_schema IS NULL AND jp.source_schema IS NOT NULL
SET r.source_schema = jp.source_schema;

MATCH (jp:JoinPath)-[r:TO_ENTITY]->()
WHERE r.source_schema IS NULL AND jp.source_schema IS NOT NULL
SET r.source_schema = jp.source_schema;

// --- Residual diagnostic: MUST return zero rows before considering done -
// Any remaining null-scoped study relationship cannot be removed by
// delete_study_scoped. Shared-node edges (CLASSIFIED_AS, IN_VOCABULARY,
// PARENT_OF, REFERS_TO) and null-scoped JoinPath nodes show up here for
// manual attribution; ontology-preloaded edges connect to nodes with no
// schema_name and are expected to be absent from study-derived counts.
// MATCH ()-[r]-() WHERE r.source_schema IS NULL
// RETURN type(r) AS rel, count(r) AS n ORDER BY n DESC;
// MATCH (jp:JoinPath) WHERE jp.source_schema IS NULL
// RETURN count(jp) AS null_scoped_join_paths;
// Ambiguous MEMBER_OF (ValueSet owned by columns in >1 schema) — review:
// MATCH (c:Column)-[:HAS_VALUE_SET]->(vs:ValueSet)<-[m:MEMBER_OF]-(:Term)
// WHERE m.source_schema IS NULL
// WITH vs, collect(DISTINCT c.schema_name) AS schemas
// WHERE size(schemas) > 1
// RETURN vs.column_ref AS value_set, schemas;
