// Migration: Rename STORED_IN to HAS_VALUE_SET with direction flip
// Before: (vs:ValueSet)-[:STORED_IN]->(c:Column)
// After:  (c:Column)-[:HAS_VALUE_SET]->(vs:ValueSet)

MATCH (vs:ValueSet)-[r:STORED_IN]->(c:Column)
CREATE (c)-[:HAS_VALUE_SET]->(vs)
DELETE r;
