"""A shared entity's ordering rule reaches arm S's context from a real graph (amendment A9).

The graph is written directly: an entity owns one relation, and two other relations share it
through an `owner = false` edge, one carrying an identity with an ordering rule and one with
none. The context path arm S takes renders the clause on the first line only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sema.consumers.nl2sql.prompting import render_sections
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine

pytestmark = pytest.mark.integration

CATALOG, SCHEMA = "lake", "books"
QUESTION = "Which note and tag go with each ledger?"
IDENTITY = json.dumps({"key_columns": ["ledger_no"], "attributes": [
    {"column": "body", "rule": "max_by", "by": "note_rev"}]})
GRAPH = """
CREATE (owner:Table {name: 'ledger_rev', schema_name: $schema, catalog: $catalog}),
       (noted:Table {name: 'ledger_note', schema_name: $schema, catalog: $catalog}),
       (tagged:Table {name: 'ledger_tag', schema_name: $schema, catalog: $catalog}),
       (ledger:Entity {name: 'Ledger'}), (note:Entity {name: 'Note'}), (tag:Entity {name: 'Tag'}),
       (ledger)-[:ENTITY_ON_TABLE {owner: true, key_columns: ['ledger_no'],
                                   verification: 'proven'}]->(owner),
       (note)-[:ENTITY_ON_TABLE]->(noted),
       (tag)-[:ENTITY_ON_TABLE]->(tagged),
       (ledger)-[:ENTITY_ON_TABLE {owner: false, key_columns: ['ledger_no'],
                                   verification: 'supported', identity: $identity}]->(noted),
       (ledger)-[:ENTITY_ON_TABLE {owner: false, key_columns: ['ledger_no'],
                                   verification: 'supported'}]->(tagged)
WITH *
UNWIND [[ledger, owner, 'ledger_no'], [note, noted, 'body'], [tag, tagged, 'tag_text']] AS pair
WITH pair[0] AS e, pair[1] AS t, pair[2] AS column
CREATE (e)-[:HAS_PROPERTY]->(:Property {name: column, semantic_type: 'free_text'})
       -[:PROPERTY_ON_COLUMN]->(:Column {name: column, data_type: 'STRING'})-[:IN_TABLE]->(t)
"""


def context_lines(driver: Any) -> list[str]:
    with driver.session() as session:
        session.run(GRAPH, catalog=CATALOG, schema=SCHEMA, identity=IDENTITY).consume()
    sco = prune_to_sco(RetrievalEngine(driver=driver).retrieve(QUESTION))
    return [line.strip() for part in render_sections(sco, "duckdb") for line in part.splitlines()]


def test_a_shared_entity_with_an_ordering_rule_states_it_and_one_without_does_not(
    clean_neo4j: Any,
) -> None:
    lines = context_lines(clean_neo4j)
    shown = "\n".join(lines)
    owner = f"{CATALOG}.{SCHEMA}.ledger_rev"

    assert (f"{CATALOG}.{SCHEMA}.ledger_note: rows belong to Ledger of {owner} through "
            "ledger_no; each Ledger has exactly one row per note_rev, so the greatest note_rev "
            "picks one row of each") in lines, shown
    assert (f"{CATALOG}.{SCHEMA}.ledger_tag: rows belong to Ledger of {owner} through "
            "ledger_no") in lines, shown
