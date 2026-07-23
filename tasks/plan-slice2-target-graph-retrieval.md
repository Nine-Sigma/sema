# Plan — Slice 2: materialize the **target semantic layer** into the graph so GraphRAG NL2SQL works over materialized target tables

> Date: 2026-07-22 · Revised: 2026-07-23 (post-adversarial-review) · Revised: 2026-07-23b
> (second adversarial pass — bridge retrievability, agnosticism boundary, identity + determinism) ·
> Status: **PLANNED (revised)**
> Branch (proposed): `ralph/feat/target-graph-retrieval`
> Follows: `tasks/plan-slice1-omop-shape-identity.md` (SLICE-1 COMPLETE — OMOP tables live in
> Databricks `workspace.omop_stage_a`). Precedes the non-deterministic warehouse→ontology matcher
> (Slice 3).
> Durable record (docs/ and .wolf/ are gitignored). Mirrors the slice-0/slice-1 story style.

## What changed in this revision (2026-07-23)

The first draft framed this slice as **"wire-and-run + fill two gaps."** An adversarial review
against the live code (Claude + Codex, findings reconciled file-by-file) found that framing is
**wrong and dangerous**: the target loader and the retrieval engine speak **different graph
contracts**, so target nodes written today are unreachable *even after* physical edges are added.
The real work is **reconciling ~6 concrete contract mismatches** (below), and the safe way to do that
is a **spike-first** sequence: prove ONE vertical (one entity, one concept field, one join, one query)
end-to-end, *then* generalize. The generic manifest/schema API and multi-strategy value-set population
come **after** a working vertical, not before it — building those abstractions on top of a retrieval
path that does not yet resolve was the over-engineering the first draft walked into.

**Second pass (2026-07-23b) fixed six defects a code-verified review found:**
1. **The bridge was built but never exercised.** No retrieval query traverses `MAPS_TO`/
   `DERIVED_FROM`/`MAPS_TO_CONCEPT` (verified: retrieval reads only `ENTITY_ON_TABLE`, `HAS_PROPERTY`,
   `PROPERTY_ON_COLUMN`, `HAS_VALUE_SET`, `MEMBER_OF`, `USES`, `IN_VOCABULARY`). The old S2-10 answered
   its question purely from the target layer, so the warehouse→ontology join — the capability this
   slice exists to add — was shipped untested. **Fix:** S2-06 now also adds a retrieval query that
   *consumes* the value bridge, and S2-10 gains a **second acceptance question phrased in a source term
   (an OncoTree code)** that only resolves by traversing `MAPS_TO_CONCEPT`.
2. **"R29 green" was standing in for "agnostic," but R29 is a literal-string grep.** It cannot catch
   OMOP *semantics* entering core as method names/assumptions. S2-04 now splits a **core `ConceptSource`
   interface** from its **showcase OMOP implementation**, plus a structural check that core imports no
   `showcase.` symbol.
3. **Target `:Entity` identity was under-specified.** S2-03 now decides the MERGE key explicitly
   (does target keep the hash-composite key and *add* `name`, or key on `name`?) — this is load-bearing
   for M6.
4. **D1 (live catalog) collided with S2-10 determinism.** The catalog read now has its own
   mock/fixture + a drift test, separate from the deterministic E2E.
5. **"OMOP concepts resolve NL→concept_id" oversold a mapped-subset index.** Honesty note added:
   only already-mapped concepts (+ancestors) are in the graph; unmapped-but-askable concepts are
   Slice-3.
6. **No gate forced the spike's genericization.** An explicit lift-to-generic checkpoint (S2-01b) now
   sits between the throwaway spike and the reusable loaders.

## Framing — this is a SEMA-core capability; cBio→OMOP is the example

SEMA's headline is **GraphRAG NL2SQL over a semantic graph**: ask a question in natural language,
the retrieval engine walks the graph to entities → physical tables/columns → joins → values, and a
consumer emits constrained SQL. Slice-1 materialized real **OMOP tables** (`person`,
`condition_occurrence`) for two MSK studies. But those tables are **invisible to the graph**, so no
consumer can reach them. This slice makes a **materialized target** retrievable.

Everything here is written **target-agnostic**. The loader takes a target *manifest* + a physical
*schema binding* and projects them into the graph; OMOP is one manifest, `omop_stage_a` is one
physical schema. cBio→OMOP is the **acceptance example**, not the abstraction. R29 (no OMOP/cBio
literal in `src/sema/`) holds — verified green today (`scripts/check_engine_coupling.py`).

## The gap (verified against current code, 2026-07-22 → re-verified 2026-07-23)

NL2SQL retrieval is **100% graph-driven**. `src/sema/pipeline/retrieval.py` opens a Neo4j session,
matches `Entity` nodes, and `_expand_physical` (`retrieval_utils.py:40`) resolves them to physical
bindings via `CypherQueries.resolve_physical_mapping()` (`graph/queries.py:40`) — that binding is the
*only* way the engine learns a table exists. Today:

1. **The graph is empty** — `MATCH (n) RETURN count(n)` = **0** (verified live against the running
   `graphrag-semantic-ontology-neo4j-1` container).
2. **Even a populated target graph would be the wrong shape.** `sema build` produces the **source**
   (cBioPortal) semantic layer. Nothing materializes `omop_stage_a` into a graph the retrieval
   queries can read (see contract mismatches below).
3. The OMOP work produced the **physical target** (Databricks tables) + a **DuckDB value-mapping
   store**, but the **target semantic layer over that target was never materialized into the graph**.

### Load-bearing finding: the machinery half-exists — and speaks a *different contract* than retrieval

The target-model materializer path exists and can write an abstract semantic model live. But the
retrieval engine was built for the **source** shape (`sema build` / `graph/loader_utils.py`), and the
target writer (`targets/neo4j_writer_utils.py`) emits **different node properties, different value-set
anchoring, different term identity, and no physical/join nodes**. The table below is the real work
surface — each row is a concrete, code-verified mismatch between *what the target loader writes* and
*what retrieval reads*.

| # | Retrieval reads (source contract) | Target loader writes today | Consequence |
|---|---|---|---|
| M1 | `MATCH (e:Entity {name})` (`queries.py:42`); source writes `Entity {name}` (`loader_utils.py:118`) | `Entity {qualified_name}` — **no `name`** (`neo4j_writer_utils.py:31`) | Target entities never match; retrieval can't find them |
| M2 | `RETURN p.semantic_type`; source `Property {entity_name, name, semantic_type}` (`loader_utils.py:148`) | `Property {…, name}` — **no `entity_name`, no `semantic_type`** (`neo4j_writer_utils.py:87`) | `_expand_values` skips non-categorical → target columns dropped |
| M3 | value lookup is **Column**-anchored: `(c:Column)-[:HAS_VALUE_SET]->(vs)` (`queries.py:183,228`) | plan proposed `(:Property)-[:HAS_VALUE_SET]` | Property-anchored value set is invisible; must anchor on Column |
| M4 | lexical term search reads `t.label`; canonical `Term {vocabulary_name, code}` (`loader_utils.py:191`) | `Term` sets `display`, id is **hash-versioned**, not canonical (`neo4j_writer_utils.py:149`) | Concept terms invisible to lexical search; don't share the concept spine |
| M5 | joins come only from `(jp:JoinPath)-[:USES]->(:Table)` (`queries.py:52`; source `loader_utils.py:286`) | loader writes `:TargetObligation`, **no `:JoinPath`** (`materializer_utils.py:206`) | person↔condition FK join can't be produced deterministically |
| M6 | `_expand_physical` matches on `entity_name` only — **no schema/model_role/is_current filter** (`queries.py:40`) | source + target coexist in one graph (D3) | Same-named entities return **mixed** source/target bindings → wrong table |

Two further gaps beyond the node-shape mismatches:

- **G-phys — physical binding to the materialized target.** The target `GraphWriter` Protocol has **no
  physical method**: it emits no `:Table`/`:Column`, no `ENTITY_ON_TABLE`/`PROPERTY_ON_COLUMN`
  (`targets/neo4j_writer.py` — write ops are entity/property/term/constraint/obligation/relationship/
  vocab-binding/context-card only). Reuse `graph/materializer.upsert_physical_nodes`
  (`materializer_utils.py:87`) fed from the **live Databricks catalog** (D1), then add the two edges.
- **G-concept — concept vocabulary indexed for retrieval.** "patients with glioblastoma" must resolve
  to `condition_concept_id = <id>`. The manifest declares its vocabulary `source: EXTERNAL` with **no
  inline terms** (`omop_condition_slice0.yaml:35`), so `normalizer._collect_terms` returns `[]`
  (`normalizer.py:76`). The concept ids live in the DuckDB value-mapping store, but that store holds
  **ids only — no names/synonyms/ancestors** (`value_mapping_store_utils.py:25`), and `VocabStore` has
  **no hierarchy/ancestor op** (`vocab_store.py:66`). A **concept-extraction contract** (fetch
  name+synonyms by id from `VocabStore`; ancestors from the OMOP `concept_ancestor` source) is a real
  prerequisite, not wiring.

### Property-id reconciliation seam (US-009 bridge) — larger than "reconcile the target id"

Both the target materializer and the US-009 bridge write/reference `:Property`, but by **different id
schemes**, and the bridge endpoints don't line up:

- Target `property_merge` MERGEs on composite `{target_model_id, version, snapshot_hash,
  parent_entity_qualified_name, name}` and *sets* `n.id = {model_id}|{version}|{hash}|{entity}.{name}`
  (`neo4j_writer_utils.py:87,131`).
- Source `:Property` gets a **UUID** id (`graph/loader_utils.py:134` path).
- The US-009 producer MATCHes `(:Property {id})` on **both** endpoints and the showcase passes
  **logical field refs** as those ids (`resolve/producer.py`; `showcase/…/slice0_fit_utils.py:170`).

So reconciling only the *target* id (first draft's S2-02d) leaves the **source endpoint** and the
**logical-ref mismatch** unfixed — `MAPS_TO`/`DERIVED_FROM` would still dangle. Both endpoints must be
reconciled, and the **value-level** `source :Term → concept :Term` edge (the actual join for
"ask in cBio terms") is produced by **neither** path today and must be added.

## Graph data model — corrected to the retrieval contract (no schema redesign)

The fix is a **two-layer graph with a thin bridge**, reusing the **source-side shapes exactly** so the
existing retrieval queries work unchanged. We populate a *second instance* of the source shapes for
the target and connect the layers. **The target loader must write the same node properties, the same
Column-anchored value sets, and the same canonical term identity the source loader writes** — that is
the whole point of M1–M4.

**Layer 1 — source (already built by `sema build`, LLM L2/L3):**
```
(:DataSource)→(:Catalog)→(:Schema)→(:Table)→(:Column)
(:Entity {name})-[:ENTITY_ON_TABLE {source_schema}]->(:Table)
(:Entity)-[:HAS_PROPERTY]->(:Property {entity_name, name, semantic_type, model_role:SOURCE})
        -[:PROPERTY_ON_COLUMN]->(:Column)
(:Column)-[:HAS_VALUE_SET]->(:ValueSet)     (:Term)-[:MEMBER_OF]->(:ValueSet)   ← value set is COLUMN-anchored
(:Term {vocabulary_name, code, label})-[:IN_VOCABULARY]->(:Vocabulary)  (:Term)-[:PARENT_OF]->(:Term)
(:JoinPath {name, join_predicates})-[:USES]->(:Table)   ← joins live on JoinPath nodes
```

**Layer 2 — target (OMOP, NEW, deterministic load — OMOP is authored, never LLM-inferred). Written to
the SAME contract as Layer 1:**
```
(:Entity {name:"condition_occurrence", model_role:TARGET})-[:ENTITY_ON_TABLE]->(:Table {schema_name:"omop_stage_a"})
(:Entity)-[:HAS_PROPERTY]->(:Property {entity_name, name, semantic_type, model_role:TARGET})
        -[:PROPERTY_ON_COLUMN]->(:Column)
(:Column {condition_concept_id})-[:HAS_VALUE_SET]->(:ValueSet)     ← COLUMN-anchored, matches M3
(:Term {vocabulary_name:"OMOP", code:<concept_id>, label:<concept_name>})-[:MEMBER_OF]->(:ValueSet)
(:Term)-[:IN_VOCABULARY]->(:Vocabulary {name:"OMOP"})   (:Term)-[:PARENT_OF]->(:Term)   ← referenced-subset context
(:JoinPath {name:"person__condition_occurrence"})-[:USES]->(:Table person), -[:USES]->(:Table condition_occurrence)
```
Concept `:Term`s use **canonical `{vocabulary_name, code}` identity and `label`** (not the target
writer's hash-id/`display`) so they share the spine with source terms and are found by lexical search.

**Bridge — only mapped things get an edge (never every node):**
```
schema level:  (:FieldMap)-[:MAPS_TO]->(target :Property)  +  (:FieldMap)-[:DERIVED_FROM]->(source :Property)
value  level:  (source :Term ONCOTREE:GBM)  -[:MAPS_TO_CONCEPT]->  (OMOP concept :Term)   from the value-mapping store
```

The OMOP ontology is loaded **once as its own subgraph and reused across every study/source**.
Inlining OMOP per-column would duplicate the ontology per source and couple the layers; that is the
anti-pattern this design avoids.

**End-to-end stages (LLM vs deterministic):** (1) ingest → source schema [det]; (2) `sema build`
→ Layer 1 [**LLM**]; (3) target load → Layer 2 [**det**, this slice]; (4) mapping → bridge edges —
**deterministic where coded** (slice-0 resolver, done) / **LLM where not** (Slice-3); (5) NL2SQL walks
Layer 2 → `omop_stage_a`, OMOP concept Terms resolve NL → `concept_id`, bridge edges enable
ask-in-source-terms. **The LLM builds source understanding; OMOP concepts come from a deterministic
codebook load, not the LLM.**

**Resolved design decisions (2026-07-22, user — unchanged):**
- **Concept load = referenced subset + hierarchy.** Load into Neo4j only OMOP concepts that source
  values actually map to, plus their `PARENT_OF` ancestor context. Full OMOP vocabulary stays
  authoritative in the DuckDB vocab store (per `codebook_ingestion_strategy`).
  **Honest consequence (do not oversell):** the graph's concept vocabulary therefore equals the
  *already-mapped* endpoints (+ancestors), **not** a queryable ontology. NL resolution of a concept
  works only because that value was mapped; an unmapped-but-askable concept silently fails retrieval.
  That is acceptable and by design here — full-ontology NL resolution and re-mapping are Slice-3.
- **One graph, schema-scoped.** Source Layer 1 and target Layer 2 live in the same Neo4j,
  disambiguated by `schema_name` / `model_role`, joined by bridge edges. **This is exactly why M6 must
  be fixed** — with both layers present, `resolve_physical_mapping` must filter by schema/model_role or
  it returns mixed bindings.
- **A target field's permissible values are a first-class `:ValueSet` — GENERIC, not OMOP-specific**,
  and **Column-anchored** (matching the source shape, M3). Core sees one primitive: "a `:Column` may
  bind a `:ValueSet` of `:Term`s." What is target-specific is only HOW the set is populated, and that
  lives in the adapter/policy (`showcase/`), never core.

## Multi-study join, value-level mapping, and unmapped values (verified 2026-07-22)

**Multi-study join happens at the shared concept spine — but only if concept terms use canonical
identity (M4).** `:Term` identity in the source loader is `{vocabulary_name, code}`, **not
study-scoped** (`loader_utils.py:191`), so both studies' `ONCOTREE:GBM` MERGE to the **same** node.
The target loader's default hash-versioned term identity would **break this** — concept terms would be
per-generation, not shared. **S2-05 must write concept terms with canonical identity** so the spine
actually joins. Data-side identity is already collapsed (Slice-1: 19,567 cross-study patients merged).

**Mapping is two distinct levels — both needed, both must reach the graph:**

| Level | Question | Home | Graph shape |
|---|---|---|---|
| Field/column | which source column → which OMOP field | authored manifest | `(:FieldMap)-[:MAPS_TO]->(target :Property)` + `-[:DERIVED_FROM]->(source :Property)` |
| Value | which source value → which OMOP `concept_id` | DuckDB value-mapping store | `(source :Term)-[:MAPS_TO_CONCEPT]->(OMOP concept :Term)`; concept `MEMBER_OF` the field's target `:ValueSet` (Column-anchored) |

**"Is a value in the OMOP value set for a concept?" = the domain-gate rule** (`omop_policy.py`:
valid + `standard_concept='S'` + `domain_id='Condition'`). **Done for exactly ONE field**
(OncoTree→Condition — one `ResolverPolicy`, one manifest). NOT generalized: no Drug/Measurement/
Procedure/Gender/Race policies; non-coded enums (`SEX`/`RACE`) are inherently Slice-3.

**Unmapped values are first-class, never dropped or fabricated** (verified):
- Recorded `NO_MAP` in the store (`concept_id=NULL` + required `no_map_reason`).
- Sentinel in the materialized table: `COALESCE(target_value, no_map_default)` → OMOP `concept_id 0`
  (`fk_closed_compiler_utils.py`, "D8 NO_MAP sentinel").
- Accounted in Gate-D-lite; routed to producer #2 (Slice-3). Absent from the value index (not a null
  placeholder).

## Goal

After this slice: **a natural-language clinical question returns valid SQL over `workspace.omop_stage_a`
(person + condition_occurrence)**, produced by the existing `NL2SQLConsumer` fed by the existing
retrieval engine — with **no OMOP/cBio literal added to `src/sema/`**. The generic path is
"materialize target `M` + bind physical schema `S` → graph → retrievable"; OMOP/`omop_stage_a` is the
instance that proves it.

## Non-negotiable principles

- **Target-generic (R29).** All new core code takes a manifest + schema binding as input. OMOP
  specifics stay in `showcase/cbioportal_to_omop/`. R29 guard must stay green.
- **Deterministic loader.** This slice is a *projection* of already-resolved artifacts into the graph.
  No LLM. The non-deterministic warehouse→ontology matcher is **Slice 3**, out of scope here.
- **Retrieval-contract fidelity.** The target loader writes the **same node properties, Column-anchored
  value sets, canonical term identity, and JoinPath nodes** the source loader writes (M1–M5). "Reuse
  the existing shape" means *the actual shape retrieval reads*, verified against `graph/queries.py`.
- **Idempotent, explicitly-scoped — lifecycle decided, not assumed.** The existing target loader keeps
  hash-versioned generations (`is_current` flip; `test_two_generations_coexist…` asserts coexistence).
  This slice **must make an explicit lifecycle decision** (S2-09) for semantic + physical + bridge +
  value nodes — not assert naive count-stability.
- **Don't fabricate retrievability.** A field with no concept binding (NO_MAP) is absent from the
  index, not a null placeholder.

## Work breakdown (TDD — failing test first, per project convention)

The sequence is **spike → migrations → contract reconciliation (per mismatch) → bridge → scoping →
lifecycle → showcase → E2E**. Genericization follows the proven vertical.

**S2-00 — Probe (DONE, 2026-07-22, corrected 2026-07-23).** Confirmed: a real `Neo4jGraphWriter` exists
(live driver, hash-versioned MERGEs, stage-guarded) and writes the abstract semantic model. But it
emits **no physical binding** AND writes a **different contract** than retrieval reads (M1–M6 above),
and the manifest carries **no concept terms** (G-concept). D1 resolved: physical metadata from live
catalog.

**S2-01 — SPIKE: one vertical, end-to-end, throwaway-grade.** Before any generic API, hand-wire the
smallest possible path in a single integration test: load `condition_occurrence` as a target
`:Entity {name}` with `ENTITY_ON_TABLE` → a physical `:Table/:Column` for `omop_stage_a`, one
Column-anchored `:ValueSet` with a couple of OMOP concept `:Term {vocabulary_name, code, label}`, one
`:JoinPath` to `person`, and assert **`resolve_physical_mapping()` returns the `omop_stage_a` binding
and `_expand_values` surfaces the concept**. This proves M1–M5 are correctly understood and the
retrieval contract is satisfiable **before** we build reusable loaders. Output: a validated list of the
exact node properties each reconciliation task must write. May be deleted or promoted once S2-03/04
land.

**S2-01b — GATE: lift-to-generic checkpoint (do not skip under time pressure).** The spike is allowed
to hardcode OMOP. The reconciliation tasks (S2-03/04/05) are **not**. Before any S2-03 code lands,
this gate is explicit: every reusable loader takes `(manifest, schema_binding)` as input and contains
**zero OMOP/OncoTree/cBio identifiers** — not just no literal strings (R29 grep) but no OMOP-shaped
method names, table names, or domain assumptions. OMOP specifics are supplied by the showcase adapter.
The gate is enforced two ways: (i) R29 literal grep stays green, AND (ii) a **structural import check**
asserts `src/sema/` imports no `showcase.` symbol (added in S2-04, applied here). Rationale: the
first-draft failure mode is "spike goes green → becomes the implementation → OMOP semantics ossify in
core," which R29 alone cannot catch.

**S2-02 — Apply target-loader migrations + fix baseline test wiring.** Run
`graph/target_loader_migrations.cypher_up()`; smoke-test that constraints/indexes exist (reversible via
`cypher_down()`). **Also fix M-stale:** the baseline round-trip test points at
`src/sema/targets/manifests/omop_condition_slice0.yaml`, which does not exist (only `dim_customer.yaml`
is there; the real manifest is under `showcase/`) — decide the supported manifest location and make the
baseline test runnable (`tests/integration/targets/test_omop_condition_slice0_round_trip.py:23`).

**S2-03 — Contract reconciliation A: target entity/property retrieval compatibility + physical binding
(M1, M2, G-phys).** Extend the target load so target `:Entity` carries `name` (M1) and target
`:Property` carries `entity_name` + `semantic_type` (M2) — matching the source shape retrieval reads.
Upsert physical `Schema/Table/Column` for `omop_stage_a` (reuse `graph/materializer.upsert_physical_nodes`),
and link target `:Entity`→`:Table` (`ENTITY_ON_TABLE`) and `:Property`→`:Column` (`PROPERTY_ON_COLUMN`).
Names no domain literal. TDD: assert `resolve_physical_mapping('condition_occurrence')` returns the
`omop_stage_a` binding with non-null `semantic_type` on categorical columns.

**S2-03 — MERGE-key decision (D5, must be made here, not discovered in S2-07).** The source loader keys
`MERGE (e:Entity {name})`; the target writer keys the hash-composite `{target_model_id, version,
snapshot_hash, qualified_name}`. Adding `name` as a *set* property on top of the composite key means
two studies (or two generations) loading OMOP `condition_occurrence` produce **multiple** `:Entity`
nodes that all match `{name}` — so `resolve_physical_mapping({name})` returns several, and correctness
then depends entirely on the S2-07 `is_current`/`model_role` filter. Decide explicitly: **(a)** key the
target `:Entity`/`:Property` on `{name}`/`{entity_name,name}` scoped by `schema_name`+`model_role`
(one node per physical target, simplest for retrieval), or **(b)** keep the hash-composite key and rely
on S2-07 scoping to collapse to `is_current`. Recommend **(a)** — it matches the source contract M1/M2
claims to reuse and removes a moving part from M6. Document the choice inline; TDD asserts the chosen
cardinality (loading the same target twice yields exactly one current binding).

**S2-03 — D1 determinism seam.** The physical metadata comes from the **live Databricks catalog** (D1 —
`information_schema`, because the manifest can drift from the write). To keep tests deterministic, the
catalog read sits behind a `PhysicalCatalogSource` port with (i) a **fixture/mock** implementation used
by every graph test including S2-10, and (ii) a **drift test** that asserts the loader fails loudly
when catalog columns diverge from the manifest. The live read is exercised only by the showcase command
(S2-09) and a marked integration test, never by the deterministic E2E.

**S2-04 — Contract reconciliation B: Column-anchored `:ValueSet` + concept terms + concept-extraction
contract (M3, M4, G-concept).** For each mapped target field, materialize a **Column-anchored**
`:ValueSet` (`(c:Column)-[:HAS_VALUE_SET]->(vs)`, M3), populate it with OMOP concept `:Term`s written
with **canonical `{vocabulary_name, code}` identity + `label`** (M4) so lexical search finds them and
they share the concept spine.

**Concept-extraction contract — agnostic boundary made explicit (Finding 2).** The value-mapping store
holds concept *ids* only (`value_mapping_store_utils.py:25`); names/synonyms/ancestors must be fetched.
Do **not** put OMOP-shaped calls (`concept_ancestor`, `VocabStore`) in core. Instead:
- **Core** defines a domain-neutral port `ConceptSource` with `name_synonyms(concept_id)` and
  `ancestors(concept_id)`. The generic concept-term loader depends only on this port.
- **Showcase** (`showcase/cbioportal_to_omop/`) implements it against `VocabStore` +
  the OMOP `concept_ancestor` source.
- **Structural guard (new, reused by S2-01b):** a test asserts `src/sema/` imports no `showcase.`
  symbol and that no OMOP identifier appears in the `ConceptSource` signature. R29's literal grep
  cannot catch semantic leakage; this import check does.

The `:ValueSet` primitive + loader are **generic** (input = "resolved value bindings for target field
X"); the population *rule* (OMOP domain gate) stays in the showcase policy. TDD: `_expand_values`
surfaces the concept, `_lexical_search` finds the concept by name; core-imports-no-showcase test green.

**S2-05 — Materialize the person↔condition_occurrence `:JoinPath` from the manifest FK (M5).** The
loader records the manifest FK as a `:TargetObligation`; retrieval reads joins only from `:JoinPath`
(`queries.py:52`). Project the authored FK obligation into a deterministic `:JoinPath {name,
join_predicates}` with `USES` edges to both target tables. This is a **deterministic manifest
projection** (not heuristic FK detection — that stays Slice-3). TDD: `find_join_paths(['condition_occurrence'])`
returns the person join predicate.

**S2-06 — Cross-layer bridge, both endpoints reconciled AND made retrievable (US-009 + value bridge +
retrieval consumption).** Run the US-009 producer live for both studies, **reconciling BOTH
`:Property {id}` endpoints** (target hash-id AND source UUID/logical-ref — `producer.py`,
`slice0_fit_utils.py:170`, `loader_utils.py:134`) so `MAPS_TO`/`DERIVED_FROM` resolve, AND add the
**value-level** `(source :Term)-[:MAPS_TO_CONCEPT]->(concept :Term)` edge from the value-mapping store
(produced by neither path today).

**Retrieval must consume the value bridge (Finding 1 — the reason S2-10 gets a second question).**
Verified: no existing retrieval query traverses `MAPS_TO_CONCEPT`, so writing the edge alone leaves the
warehouse→ontology join **unreachable** by NL2SQL. Add a generic query
`resolve_concept_for_source_term` —
`(src:Term {code})-[:MAPS_TO_CONCEPT]->(concept:Term)-[:MEMBER_OF]->(vs:ValueSet)<-[:HAS_VALUE_SET]-(c:Column)`
— returning the concept `code`/`label` + governed `Column`/`Table`, and wire it into the retrieval
expansion (mirroring `_expand_term_hit`) so a query seeded on a **source** term surfaces the target
`concept_id` + the column to filter. The query names no domain literal (it matches on `MAPS_TO_CONCEPT`,
not OMOP). TDD: assert no dangling `MAPS_TO`; a source `:Term` traverses to its OMOP concept; and
`resolve_concept_for_source_term(<OncoTree code>)` returns the target `condition_concept_id` + governed
column. This is the retrieval capability the S2-10 second question exercises end-to-end.

**S2-07 — Retrieval scoping: filter `resolve_physical_mapping` by schema/model_role/is_current (M6 —
correctness fix, not a config check).** The query matches on `entity_name` only; with source + target
coexisting (D3), a shared entity name returns **mixed bindings**. Add a `schema_name` / `model_role` /
`is_current` predicate (and a retrieval-scope config so a query targets the target layer). TDD in
`tests/unit` mirroring existing retrieval tests: a source and target entity of the same name resolve to
their own physical bindings, not each other's.

**S2-08 — Lifecycle decision: idempotent re-load reconciled with hash-versioned generations.** Decide
and implement re-run semantics across semantic + physical + value + bridge + JoinPath nodes. The
existing target loader **keeps** generations (`is_current` flip; `test_two_generations_coexist…`), so
"fully replaces / count stable" is not automatically true. Options: (a) scope-delete the current
generation before reload, or (b) flip-and-supersede like the existing loader. Pick one, document it,
and test it — do not assert naive count-stability. TDD.

**S2-09 — Showcase CLI + wiring (thin, in `showcase/`).** A `sema materialize-target-graph` showcase
command that calls the generic S2-03/04/05/06 path with the OMOP manifest + `omop_stage_a`. Lazy-import
convention (`_register_showcase_commands`) so a wheel without `showcase/` still loads.

**S2-10 — End-to-end acceptance (the demo), deterministic — TWO questions.** An integration test: given
the loaded target graph and a **deterministic fake LLM** (`NL2SQLConsumer` delegates SQL to the LLM —
`consumer.py:101`; its validator only permits referenced tables/columns — `validation.py:18`, so a live
LLM can't prove correctness). The fake emits SQL as a function of the retrieved context, so asserting on
the SQL AST is an assertion about **retrieval**. Both questions assert on the retrieved governed value +
SQL AST, not just that SQL parsed.

**Q1 — target-term question (target layer only).** "how many patients have a glioblastoma condition?"
→ SQL that (i) references `omop_stage_a.condition_occurrence`, (ii) filters on the **correct**
`condition_concept_id` (assert the concept literal from the retrieved SCO), (iii) FK-joins
`omop_stage_a.person` via the S2-05 JoinPath (assert the join predicate). This proves M1–M5 + the target
value index.

**Q2 — source-term question (forces the warehouse→ontology join, Finding 1).** The **same** question
phrased with a **source OncoTree code** (e.g. "how many patients have `<ONCOTREE_CODE>`?") → SQL that
filters on the **same** `condition_concept_id` as Q1, reached by traversing
`(:Term)-[:MAPS_TO_CONCEPT]->(concept :Term)` via `resolve_concept_for_source_term` (S2-06). Assert the
retrieval trace actually traversed `MAPS_TO_CONCEPT` (not lexical-matched the concept name), and that
Q1 and Q2 resolve to the identical `condition_concept_id`. **This is the only test that proves the
bridge works end-to-end** — without it, S2-06 ships untested.

Together, Q1 + Q2 are the slice's definition of done.

## Verification / definition of done

- [ ] Neo4j non-empty: target `Entity {name}` / `Property {entity_name, semantic_type}` + physical
      `Table/Column` for `omop_stage_a` (both studies), `:JoinPath`, `VOCAB_LOOKUP` + `MAPS_TO_CONCEPT`
      edges, concept `Term`s with canonical identity + `label`.
- [ ] `resolve_physical_mapping('condition_occurrence')` returns the `omop_stage_a` binding **and only
      that** (not a source binding of the same name) — M1/M2/M6 closed.
- [ ] Both studies join at the **shared concept spine** — `ONCOTREE:GBM` resolves to ONE canonical
      `:Term` referenced by both studies' bridge edges (M4 closed).
- [ ] Each mapped target field binds a **Column-anchored** `:ValueSet`; members enumerable via
      `_expand_values`; concept found via `_lexical_search` (M3 closed); source→concept value bridge
      traversable.
- [ ] `find_join_paths(['condition_occurrence'])` returns the person join (M5 closed).
- [ ] **Bridge is retrievable:** `resolve_concept_for_source_term(<OncoTree code>)` returns the target
      `condition_concept_id` + governed column; `MAPS_TO`/`DERIVED_FROM` have no dangling endpoints.
- [ ] `NL2SQLConsumer` (unchanged) + deterministic fake LLM emits correct SQL for **both** S2-10
      questions: Q1 (target term) and Q2 (source OncoTree code), resolving to the **same**
      `condition_concept_id`, with Q2's trace traversing `MAPS_TO_CONCEPT` (Finding 1 closed).
- [ ] **Target `:Entity` cardinality decided (D5) and tested:** loading the same target twice yields
      exactly one current physical binding (S2-03).
- [ ] **Determinism seam:** graph tests (incl. S2-10) use the `PhysicalCatalogSource` fixture; a drift
      test fails loudly on catalog↔manifest divergence; the live catalog read is integration-marked.
- [ ] Lifecycle explicit and tested (S2-08): re-run behaves as documented, no dangling/duplicate nodes.
- [ ] R29 guard green; `src/sema/` names no OMOP/cBio literal; **structural import check green
      (`src/sema/` imports no `showcase.` symbol)**; all OMOP specifics in `showcase/`.
- [ ] mypy strict + full unit suite green; coverage ≥ 85% before commit (per project gate).

## Open decisions

- **D1 — Physical metadata source. RESOLVED (2026-07-22, user): live catalog.** Read `omop_stage_a`
  metadata from Databricks `information_schema` at load time (the manifest can drift from the write).
- **D2 — Concept index home. RESOLVED:** referenced-subset OMOP concepts land in the graph as `:Term`
  (canonical identity + `label` + `PARENT_OF` context) and reuse the existing graph lexical/vector
  index; the full vocabulary stays in the DuckDB vocab store. **New in this revision:** requires the
  concept-extraction contract (S2-04) because the store holds ids only.
- **D3 — Source + target coexistence. RESOLVED:** one graph, schema-scoped — **which is why M6/S2-07
  (scope the physical-mapping query) is mandatory, not optional.**
- **D4 — Re-load lifecycle. OPEN → decide in S2-08:** scope-delete-current vs flip-and-supersede.
  Must be reconciled with the existing hash-versioned generation model.
- **D5 — Target `:Entity`/`:Property` MERGE key. RESOLVED (2026-07-23b, user): option (a).** Key on
  `{name}`/`{entity_name,name}` scoped by `schema_name`+`model_role` (one node per physical target,
  updated in place) — **not** the hash-composite key. Load-bearing for M6; decided in S2-03, not
  discovered late in S2-07.
  **Rationale + future-proofing (versioning/branching):** (a) makes the target graph look exactly like
  the source contract retrieval reads (the M1–M5 premise), and it does **not** foreclose future
  versioning/branching of the graph and tables. Versioning is an **orthogonal `{branch, version}`
  dimension** added later to node identity (`MERGE (n:Entity {name, schema_name, model_role, branch,
  version})`) plus a retrieval scope predicate — which is the **same seam S2-07 introduces** (default
  `branch=main, version=latest`). The hash-composite key of option (b) is **not** real versioning: a
  content hash gives coexistence/dedup but no branch lineage, named branches, or valid-time, so real
  branching would have to be built regardless — and (b) would then be a half-baked scheme to migrate
  away from, whereas (a) has nothing to unwind. **History is authoritative in the artifact layer**
  (Databricks Delta time-travel/clones + versioned manifests + DuckDB value-mapping store); the graph
  is a **rebuildable projection** tagged with `{branch, version}` and provenance pinning the Delta
  table version it projected. Caveat: this assumes the graph stays a projection, not the authoritative
  history store — consistent with the current design (Delta + DuckDB are the sources of truth).

## Out of scope (explicitly deferred)

- **Non-deterministic warehouse→ontology matching (Slice 3 / producer #2).** LLM/GraphRAG semantic
  fitting of un-crosswalked / free-text / uncoded fields. This slice is deterministic projection only.
- **Generalizing value mapping beyond the one done field.** Only `ONCOTREE_CODE→condition_concept_id`
  is mapped today. Other OMOP fields/domains and non-coded enums (`SEX`/`RACE`) are Slice-3. `NO_MAP`
  values are surfaced (sentinel `concept_id 0`, absent from the value index) — re-mapping is Slice-3.
- **Generic multi-strategy `:ValueSet` population** (enumeration / observed-distinct-values for
  ontology-free targets). The generic primitive ships here; the additional population strategies land
  once the OMOP vertical is proven.
- New OMOP target tables beyond `person` + `condition_occurrence`.
- Query-time NL2SQL quality tuning beyond the single S2-10 acceptance question.
