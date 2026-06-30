# PRD: Data Unification Correctness & Wiring Fixes

> Source: `docs/architecture/data-unification-analysis.md` (2026-06-10, amended).
> Scope decision: deterministic correctness + wiring fixes only. Embedding-based
> entity resolution (finding A) and cross-study join inference (finding F / #31)
> are deferred to a follow-up PRD. Existing graphs are rebuildable — no
> migration scripts.

## Introduction

Sema's Neo4j semantic layer unifies metadata across studies via shared concept
nodes (`:Entity`, `:Property`, `:Term`, `:Vocabulary`) with per-study edges. A
line-by-line audit found correctness bugs and silent wiring gaps that corrupt
or erode this layer: a lifecycle phase that globally deprecates other tables'
vocabularies, last-writer-wins overwrites on shared nodes, provenance edges
that have never been written, study deletion that leaves artifacts behind, no
deadlock handling under the default `table_workers=4`, and raw un-normalized
strings as MERGE keys. This PRD fixes all of them.

## Goals

- Shared-node writes become order-independent (no last-writer-wins races).
- The lifecycle phase never deprecates vocabularies introduced by other tables
  in the same build.
- Provenance SUBJECT/OBJECT edges actually exist in the graph after a build.
- `delete_study_scoped` removes everything a study's build created (edges,
  aliases, vocabulary associations) and leaves no orphans.
- `:Term` identity is namespaced by vocabulary so `M`(Male) never merges with
  `M`(Mississippi).
- Materialization survives concurrent Neo4j transient errors (deadlocks) under
  `table_workers=4`.
- Embeddings reflect the current description of every embedded node.
- The pipeline programs against the existing `Connector` ABC, not
  `DatabricksConnector` directly.

## Project Standards (apply to EVERY story)

These are hard acceptance criteria for all stories, in addition to each
story's own list:

- **Strict TDD**: write the failing test first, watch it fail
  (`uv run pytest <test_file> -v`), then implement until green.
- All commands via `uv run` (`uv run pytest`, `uv run mypy src/sema/`).
- Functions <= 60 lines; new files <= 400 lines. If an edit would push a file
  past 400 lines, extract helpers into the module's `*_utils.py` file.
  `loader.py` (614 lines) and `materializer_utils.py` (464) already exceed the
  limit: do NOT grow them, and do NOT refactor them wholesale (out of scope) —
  put new logic in `*_utils.py` modules.
- No globals — module-level constants, class attributes, or function scope.
- DRY, self-documenting names, no excessive comments (comment *why*, never
  *what*), max 3 nesting levels (guard clauses / early returns).
- Type hints everywhere; `uv run mypy src/sema/` passes (strict mode).
- Logging via `from sema.log import logger` (loguru, not stdlib).
- Catch specific exceptions; fail fast on programming errors.
- Mock LLMs with `MagicMock()`; unit tests must not require Neo4j (mock the
  loader/driver as the existing unit tests do).
- Conventional commit per story: `fix:` / `feat:` / `refactor:` / `test:`.

## User Stories

### US-001: Normalize names and alias text before assertion creation
**Description:** As a build pipeline, I want entity/property/alias strings
normalized once at the assertion boundary so that whitespace and unicode
variants of the same name cannot fragment into separate graph nodes (finding N).

**Acceptance Criteria:**
- [ ] A `normalize_name(text: str) -> str` helper exists in a `*_utils.py`
      module (e.g. `src/sema/engine/normalize_utils.py`): `strip()`, collapse
      internal whitespace runs to single spaces, apply unicode NFC. Case is
      preserved (case-folding belongs to future entity resolution).
- [ ] Failing tests written first covering: leading/trailing whitespace,
      internal double spaces, NFC vs NFD unicode forms, idempotency, empty
      string.
- [ ] Normalization is applied at assertion-creation time (L2/L3 payload →
      `Assertion`) for entity names, property names, and alias/synonym text —
      one choke point, not scattered at each MERGE site. `Assertion()` is
      constructed at ~9 sites across 6 engine files (`semantic.py`,
      `vocabulary.py`, `joins.py`, `join_detector.py`, `stage_utils.py`,
      `corrections.py`), so the choke point must be a Pydantic validator on
      `Assertion` or a single shared builder — not edits at every call site.
- [ ] Two assertions differing only by whitespace/unicode form produce one
      `:Entity` row in the upsert payload (unit test against
      `materializer` grouping, loader mocked).
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-002: Key `:Term` on `{vocabulary_name, code}`
**Description:** As a multi-domain platform, I want term identity namespaced by
vocabulary so semantically unrelated codes never collapse into one node
(finding B). Existing graphs will be rebuilt; no migration.

**Acceptance Criteria:**
- [ ] Failing test first: two term assertions with the same `code` but
      different vocabularies produce two distinct MERGE rows / nodes; same
      code + same vocabulary still produces one.
- [ ] `batch_upsert_terms` (`loader_utils.py:142`) MERGEs on
      `{vocabulary_name, code}`; terms with no detected vocabulary use a
      documented sentinel namespace (e.g. `_unscoped`) so the key is never
      null.
- [ ] Every Cypher statement that MATCHes `:Term` by `code` (edge creation
      such as `MEMBER_OF`/`IN_VOCABULARY`, retrieval queries in
      `graph/queries.py`, deletion paths) is updated to the composite key —
      grep for `:Term` to enumerate them; list each updated site in the
      story notes.
- [ ] Any Neo4j uniqueness constraint/index on `:Term(code)` is updated to the
      composite key (as of writing none exists — `planner_migrations.py`
      creates only `*_id_unique` constraints; verify and skip if still absent).
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-003: Order-independent shared-node writes (cross-study winner)
**Description:** As a graph with nodes shared across studies, I want
`description`/`source`/`confidence` on `:Entity`/`:Property`/`:Term` updated
only when the incoming assertion has confidence strictly greater than the
stored value (with a deterministic tiebreak), so build order and worker
scheduling can never decide what a shared node says (finding C).

**Acceptance Criteria:**
- [ ] Failing tests first: (a) low-confidence write after high-confidence
      write does NOT overwrite; (b) high after low DOES; (c) equal confidence
      keeps the existing value (first-writer wins as tiebreak); (d) result is
      identical regardless of write order.
- [ ] `batch_upsert_entities`, `batch_upsert_properties`,
      `batch_upsert_terms` (`loader_utils.py:78-171`) replace unconditional
      `SET` with a confidence-guarded `CASE`/`WHERE` update; `source_id`
      coalesce behavior is preserved.
- [ ] `ON CREATE` still sets all fields for new nodes.
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-004: Run the vocabulary lifecycle phase once per build, not per table
**Description:** As a multi-table build, I want vocabulary deprecation computed
over the union of all tables' active vocabularies after materialization
completes, so table 2 can never deprecate vocabularies table 1 introduced
(finding D).

**Acceptance Criteria:**
- [ ] Failing test first reproducing the bug: materialize table A (vocab X),
      then table B (vocab Y) → X must remain ACTIVE.
- [ ] `run_lifecycle_phase` is removed from `materialize_unified`'s per-table
      steps (`materializer.py:66`) and invoked once by the orchestrator after
      all work items finish, with the union of active vocabulary names
      collected across tables.
- [ ] The union is accumulated thread-safely (no module-level mutable global —
      pass an accumulator or collect from worker return values).
- [ ] Deprecation query remains scoped to vocabularies absent from the union;
      behavior under `resume` is covered by a test.
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-005: Retry transient Neo4j errors during materialization
**Description:** As a build running `table_workers=4` (the default), I want
MERGE deadlocks on shared nodes retried with backoff instead of failing the
table (finding E).

**Acceptance Criteria:**
- [ ] Failing tests first using a mocked driver/session: `TransientError`
      raised twice then success → call succeeds with 3 attempts; persistent
      `TransientError` → raises after max attempts; non-transient errors
      (e.g. `ClientError`) are NOT retried and propagate immediately.
- [ ] A retry helper (in `graph/loader_utils.py` or a dedicated
      `*_utils.py`) wraps `GraphLoader._run` / batch execution: retries
      `neo4j.exceptions.TransientError` with exponential backoff,
      max attempts and base delay as constants, logs each retry via
      `sema.log.logger` with context (statement summary, attempt).
- [ ] No sleep in unit tests (inject/patch the sleep function).
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-006: Actually create provenance SUBJECT/OBJECT edges
**Description:** As a curator, I want every provenance assertion linked to the
node it describes, so the graph can answer "which assertion backs this?" —
today `materialize_provenance_edges` matches on `subject_id` which is always
`None`, so zero edges have ever been written (finding I, bug-239).

**Acceptance Criteria:**
- [ ] Failing test first proving the current no-op: materialize a
      `has_entity_name` assertion → assert a SUBJECT edge Cypher call is
      issued that can match the subject node (it currently cannot).
- [ ] Edge matching keys on what nodes actually have: resolve the subject from
      `assertion.subject_ref` (catalog.schema.table[.column]) to the physical
      `:Table`/`:Column` node key, or populate `subject_id`/`object_id` at
      assertion-creation time — pick ONE mechanism, document it in the
      module docstring, and delete the dead alternative.
- [ ] Guard clauses skip (with a debug log) assertions whose subject cannot be
      resolved — no unguarded null MATCH round-trips remain.
- [ ] Edges are created in batch (UNWIND), not one Cypher round-trip per
      assertion.
- [ ] Test asserts OBJECT edges for predicates that carry an object.
- [ ] `.wolf/buglog.json` bug-239 updated with the actual fix.
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-007: `source_schema` on `IN_VOCABULARY` and `CLASSIFIED_AS` edges
**Description:** As a study owner, I want my vocabulary-association edges
deleted when my study is deleted; today these two edge types carry no
`source_schema` and survive `delete_study_scoped` (finding K).

**Acceptance Criteria:**
- [ ] Failing tests first: the MERGE payloads for `batch_create_in_vocabulary`
      (`loader_utils.py:307`) and `batch_create_classified_as`
      (`loader_utils.py:291`) include `source_schema` in the relationship
      MERGE key, matching the pattern of `HAS_VALUE_SET`/`MEMBER_OF`.
- [ ] Two studies asserting the same Term→Vocabulary association produce two
      edges (one per study), each independently deletable — consistent with
      the §3 design in the analysis doc.
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-008: Complete study deletion — aliases and orphaned concept nodes
**Description:** As an operator deleting a study, I want no stranded `:Alias`
nodes and no edge-less concept nodes left behind (findings J and H).

**Acceptance Criteria:**
- [ ] Failing tests first (mocked loader, asserting issued Cypher): after
      `delete_study_scoped` removes edges, a cleanup statement deletes
      `:Alias` nodes with no remaining `REFERS_TO` edges, and another removes
      `:Entity`/`:Property`/`:Term`/`:Vocabulary`/`:ValueSet` nodes with no
      remaining relationships.
- [ ] Cleanup lives in `delete_study_scoped` (or a helper it calls in
      `loader_utils.py`) — one entry point, not a separate manual command.
- [ ] Shared nodes still referenced by surviving studies are untouched
      (covered by a test with two studies).
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-009: Resume runs study-scoped cleanup before re-materializing
**Description:** As a resumed build, I want the study's prior graph writes
cleaned up before re-materialization, so edges from tables that vanished since
the failed run cannot persist (finding L). Resume reloads the full assertion
set, so a scoped delete-then-rewrite is safe.

**Acceptance Criteria:**
- [ ] Failing test first: with `config.resume=True`, `delete_study_scoped`
      is invoked for the study schema before any `materialize_unified` call
      (today it is skipped — `orchestrate.py:64`).
- [ ] Resume still skips re-extraction/LLM work (that is its purpose); only
      graph cleanup behavior changes.
- [ ] A test covers the no-prior-state case (fresh resume) — cleanup of an
      absent study is a no-op, not an error.
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-010: Re-embed nodes whose description changed
**Description:** As the retrieval engine, I want embeddings to match current
node descriptions; today embeddings are computed once post-build with no
change detection, so a later study's overwrite leaves a stale vector
(finding M).

**Acceptance Criteria:**
- [ ] The embedding phase is `_compute_embeddings`
      (`pipeline/orchestrate_utils.py:224`) + `engine/embeddings.py`; the
      write path is `GraphLoader.set_embedding` / `set_property_embedding`.
- [ ] Failing tests first: the embedding phase selects nodes where the
      embedding is missing OR a stored `description_hash` differs from the
      hash of the current description; unchanged nodes are NOT re-embedded
      (no wasted embedding calls — assert call counts on a mocked embedder).
- [ ] `description_hash` (deterministic, e.g. sha256 of the embedded text) is
      written alongside the embedding in the same statement.
- [ ] Hashing/selection helpers live in the relevant `*_utils.py`.
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-011: Pipeline programs against the `Connector` ABC
**Description:** As a future non-Databricks source, I want orchestration typed
against the existing `Connector` interface (`connectors/base.py`) instead of
the concrete `DatabricksConnector`, and no pipeline code reaching into private
connector internals (finding G).

**Acceptance Criteria:**
- [ ] Failing tests first: orchestration accepts any `Connector`
      implementation (test with a stub subclass, no Databricks imports in the
      test).
- [ ] `orchestrate.py` / `build.py` type hints and factories use `Connector`;
      `DatabricksConnector` is constructed in exactly one place (composition
      root / factory).
- [ ] The FK sampler's use of `connector._execute`
      (`orchestrate_utils.py:340-343`) is replaced with a public method on the
      `Connector` ABC (add the abstract method; implement in
      `DatabricksConnector` by delegating to the existing logic).
- [ ] No new connector implementations (out of scope) — the stub in tests is
      sufficient proof.
- [ ] `uv run pytest` and `uv run mypy src/sema/` pass.

### US-012: Update the analysis doc and project docs to reflect the fixes
**Description:** As the next reader of
`docs/architecture/data-unification-analysis.md`, I want findings B, C, D, E,
I, J, K, L, M, N, G marked as fixed with references to the implementing
change, so the doc stays a trustworthy map of remaining gaps (A and F stay
open).

**Acceptance Criteria:**
- [ ] Each fixed finding in §5 gets a one-line `**Fixed:**` note naming the
      mechanism (no rewriting of history; the analysis text stays).
- [ ] §6 recommendations 2-5 and 8-13 marked done; 1 and 6 remain open.
- [ ] §7 verdict paragraph updated to the post-fix state.
- [ ] Coverage >= 85% confirmed
      (`uv run pytest --cov=sema --cov-report=term-missing`).

## Functional Requirements

- FR-1: Name/alias normalization (strip, collapse whitespace, NFC) applied at
  the assertion-creation boundary; case preserved.
- FR-2: `:Term` MERGE identity is `{vocabulary_name, code}` with a sentinel
  namespace for vocabulary-less terms; all `:Term` MATCH sites use the
  composite key.
- FR-3: Shared-node scalar updates are confidence-guarded and
  order-independent; ties keep the existing value.
- FR-4: Vocabulary lifecycle/deprecation runs exactly once per build over the
  union of all tables' active vocabularies.
- FR-5: All Neo4j writes retry on `TransientError` with exponential backoff
  (bounded attempts); non-transient errors propagate unchanged.
- FR-6: Every provenance-predicate assertion materializes a SUBJECT edge (and
  OBJECT edge when applicable) to a resolvable node, in batched Cypher.
- FR-7: `IN_VOCABULARY` and `CLASSIFIED_AS` relationship MERGE keys include
  `source_schema`.
- FR-8: `delete_study_scoped` additionally removes orphaned `:Alias` nodes and
  edge-less concept nodes, never touching nodes still referenced by other
  studies.
- FR-9: Resume performs study-scoped graph cleanup before re-materializing.
- FR-10: Embedding phase re-embeds only missing-or-changed descriptions,
  tracked via a stored `description_hash`.
- FR-11: Orchestration depends on `Connector` (ABC) with the concrete
  Databricks connector bound at a single composition root; no `_private`
  access across module boundaries.

## Non-Goals (Out of Scope)

- Embedding/LLM-based entity resolution behind the MERGE (finding A) — needs
  its own design; US-001's normalization is deliberately the only step taken.
- Cross-study/cross-schema join inference (finding F, issue #31).
- Migration scripts for existing Neo4j graphs — rebuild from the warehouse.
- New connector implementations (CSV/REST/other warehouses).
- Domain namespacing of `:Entity` identity (related to A; deferred with it).
- Any retrieval/query-pipeline feature work beyond updating `:Term` lookups.

## Technical Considerations

- Implementation order matters: US-001 and US-002 change MERGE payload shapes
  that later stories' tests will assert against — do them first. US-003
  before US-005 (retry wraps the new guarded writes). US-007 before US-008
  (deletion completeness assumes the edges are deletable). US-012 last.
- `materialize_unified`'s phase list (`materializer.py`) and the per-table
  worker loop (`orchestrate_utils.py:138`) are the integration points for
  US-004; worker return values are the cleanest accumulator (no shared
  mutable state).
- Unit tests must not require Neo4j: assert on the Cypher text + parameters
  passed to a mocked driver/session, following existing patterns in
  `tests/unit/`.
- File sizes today: `loader_utils.py` 320 lines (room to grow),
  `materializer_utils.py` 464 and `loader.py` 614 (already over the 400
  ceiling). Never grow the oversize files — extract new helpers into new
  `*_utils.py` modules. Wholesale refactor of the oversize files is out of
  scope for every story.
- Coverage must remain >= 85% (`uv run pytest --cov=sema
  --cov-report=term-missing`) before any commit/push.

## Success Metrics

- A two-table build (vocab X on table 1, vocab Y on table 2) ends with both
  vocabularies ACTIVE.
- Materializing the same two studies in either order yields byte-identical
  shared-node properties.
- After a build, `MATCH (:Assertion)-[:SUBJECT]->() RETURN count(*)` > 0.
- After `delete_study_scoped`, zero `:Alias` orphans, zero edge-less concept
  nodes, zero `IN_VOCABULARY`/`CLASSIFIED_AS` edges from the deleted study.
- A simulated deadlock (mocked `TransientError`) no longer fails a table.
- Full suite green via `uv run pytest`; `uv run mypy src/sema/` clean;
  coverage >= 85%.

## Open Questions

- Sentinel vocabulary namespace for unscoped terms: `_unscoped` vs the source
  schema name — `_unscoped` keeps cross-study collapse for enum-style terms
  (`0:LIVING`), schema-name would silo them per study. Default to `_unscoped`
  unless decided otherwise.
- Should the orphan-GC in US-008 also run as a standalone CLI maintenance
  command? Not required by this PRD.
