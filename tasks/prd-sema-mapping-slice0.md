# PRD — Sema Mapping Spine, Slice 0 (Ralph Loop)

> Date: 2026-06-30 (rev 4 — re-grounded after PR #109 target framework merged to main)
> Status: planning / ready for Ralph.
> Branch: `ralph/feat/mapping-slice0` (branch from `main`).
> Source of truth: `docs/architecture/sema-mapping-architecture.md` (the index),
> `sema-mapping-risk-register.md` (R1–R29), `sema-mapping-build-vs-buy.md`, and
> `docs/architecture/cbioportal-omop-concept-fitting.md` (the §4 resolver algorithm).
> This PRD implements the **single end-to-end gap** named in the architecture doc §3
> and acceptance §5: take `ONCOTREE_CODE → condition_concept_id` from a hardened
> source semantic graph all the way to a compiled, QA'd staging artifact in
> Databricks, measured against a hand-labelled gold set.

> **Rev-2 changelog (why this differs from rev 1).** A two-pass adversarial review
> (Claude + Codex) found the rev-1 stories were structurally sound but the *contracts*
> contradicted each other. Rev 2: (a) adds **§1.5 Slice-0 Contract Spec** — one frozen
> definition of the value-mapping store, the staging table, the Zone model, the status
> flow, and the precision/recall confusion matrix — referenced by every story so the
> shapes cannot drift; (b) adds **US-000** (R29 baseline migration) so US-001's guard can
> go green; (c) adds **US-012A** (DuckDB end-to-end smoke) so the spine is wired before
> the live Databricks gate; (d) reorders policy (US-004) **before** the vocab store
> (US-003); (e) cuts `MappingProposal` and the `RENAME`/`TYPE_CAST` patterns (neither
> exists; both are over-build for Slice 0); (f) records that PR #109's manifest adapter
> (`ManifestVocabularyBinding` et al.) is prior art US-004/US-007 must evaluate; (g)
> corrects stale code references. **Priorities — not ID suffixes — are the build order.**

> **Rev-3 changelog (loop-determinism).** Every code reference in rev 2 was
> ground-truthed against `main` and confirmed accurate. Rev 3 removes the three
> places a Ralph loop could behave non-deterministically or self-certify: (a) US-002
> adds an explicit **human-label checkpoint** — `gold_concept_id` is an external
> oracle, never produced by Sema's own resolver (else US-012 grades its own
> homework), and US-012 "accepted" now depends on it; (b) **single store-writer
> ownership** — US-006 (resolver) is the *sole* writer of value-mapping-store rows;
> US-009 (producer) and all downstream stories *read* the store, never write it;
> (c) **PR #109 fork removed** — `ea2aed7` is now **reference-only** (mirror its DTO
> field names, do not recover the unmerged branch), and US-007's source of truth is
> fixed to **CSV-parse**, so US-004/US-007 carry no recover-vs-reimplement decision.

> **Rev-4 changelog (PR #109 merged — supersedes rev-3 item (c)).** Between rev 3 and
> rev 4, **PR #109 (`ea2aed7`) was merged to `main`**, landing a full target-ontology
> subsystem: `src/sema/targets/` (a source-agnostic three-stage `load_target` —
> normalize→hash→materialize), a `TargetOntologyAdapter` protocol (`targets/base.py`),
> a production `ManifestTargetAdapter` reading a YAML/JSON manifest, `targets/materializer.py`
> + `targets/neo4j_writer.py` writing **`model_role=TARGET`** nodes, a real
> `:VocabularyBinding` node + **`HAS_VOCABULARY_BINDING`** edge (note: *not* `HAS_VOCAB_BINDING`),
> `ManifestVocabularyBinding` / `VocabularyBindingDecl` (fields `vocabulary, domain,
> require_standard, allow_zero_default, effective_date_ref, resolver_policy_ref`), and a
> registered CLI `sema target load --manifest <path> --writer in-memory|neo4j`. The whole
> tree is **R29-clean** (zero OMOP literals; all specifics arrive as manifest data) and
> hermetically tested. Consequences: **(1)** rev-3's "reference-only / do not recover /
> CSV-parse" is **void** — the framework is real and on `main`. **(2) US-007 is now
> "author an OMOP `condition_occurrence`-slice manifest (YAML) and load it via the
> existing framework" — no new loader/materializer code** (decided by the user; the
> CSV-parse and CSV→manifest alternatives were rejected for Slice 0). **(3)** the manifest
> file is the allowlisted home for OMOP literals (R29). **(4)** `models/target/obligation.py`
> just re-exports the planner `TargetObligation`, and `targets/` materialization is
> **complementary** to `planner_loader.py` (targets builds the TARGET graph; planner wires
> MAPS_TO/DERIVED_FROM onto it) — so US-008/US-009 are unchanged. **(5)** US-014 becomes a
> second manifest (no `vocabulary_binding`) through the same loader — even less code.

---

## 0. Framing — what this PRD is and is not

**Vision check (CLAUDE.md):** Sema is a *general* semantic-ETL engine — any source →
any target ontology, under constraints, materialized. **cBioPortal → OMOP on
Databricks is the MVP showcase, not the product.** Every story below is written so the
OMOP/OncoTree specifics live behind a **per-ontology policy boundary (R29)**, never in
`engine/` or the generic spine. US-014 proves that boundary with a non-bio target.

**Scope = Slice 0 only.** The deliverable (architecture §5) is a **resolved
value-mapping store + a compiled write to a quarantined *staging* artifact** — *not* a
production-valid OMOP `condition_occurrence` table. Identity (`person_id`), event-date
semantics, the LLM council, free-text/vector recall, and full Gate D are **out of scope
by construction** (see §"Out of scope" and `Slice 1+` notes). Building any of them here
is over-engineering and drifts from the vision.

**Why these stories, in this order.** Architecture §3 names the critical path in build
order; steps 0 and 1 parallelize, only 3 (needs 1+2) and 4 (needs the frozen store) are
truly gated. The dependency graph below encodes that — **arrow `A ─▶ B` means "A must be
built before B"** (B depends on A). Priorities (the per-story `priority:` field) realize
this order:

```
US-000 (R29 baseline migration) ─▶ US-001 (R29 CI guard) ─── protects every later story
US-002 (gold set + eval harness, TDD) ── defines "done"; written FIRST (§3 step 0)
        │
US-004 (policy obj) ─▶ US-003 (vocab store) ─▶ US-005 (store) ─▶ US-006 (resolver §4)
                                                          │
US-007 (OMOP target: authored manifest → merged loader) ─┐       │
                                       ▼                  ▼
                       US-008 (staging obligation + PlanAssembler)
                                       │
                       US-009 (VOCAB_LOOKUP assertion producer)
                                       │
                       US-010 (SQLGlot compiler → staging write)
                                       │
                       US-011 (Gate D-lite staging QA)
                                       │
                       US-012 (mapping eval: P/R vs gold set)
                                       │
                       US-012A (DuckDB end-to-end smoke: full spine, no Databricks)
                                       │
                       US-013 (LIVE end-to-end run: Databricks + cBio + OMOP)
                       US-014 (generality counter-example: no-vocab manifest target — R29 proof)
```

The §1.5 Contract Spec is a prerequisite *reference* for US-002, US-005, US-008, US-009,
US-010, and US-012; it is frozen in code by the schema **drift test** in US-005 and the
metric module in US-002, not by a separate documentation-only iteration.

---

## 1. Global Standards — apply to EVERY story (non-negotiable)

These are acceptance criteria for **every** user story. A story is not `passes: true`
until all of them hold. The converter to `prd.json` must inline this block into each
story's `acceptanceCriteria`.

- **Strict TDD.** Write the failing test(s) **first**, watch them fail for the right
  reason, then implement. The test names the behavior, not the implementation.
- **No globals.** Class attributes, module-level constants, or function scope only. No
  module-level mutable state.
- **Functions ≤ 60 lines. Files ≤ 400 lines.** Helpers go in `*_utils.py` modules — keep
  engine/resolver/compiler files thin. `src/sema/graph/loader.py` (**723 lines**) and
  `src/sema/graph/materializer_utils.py` (**429 lines**) already exceed 400 — **do not
  grow them, do not refactor them wholesale** (out of scope); put new logic in new
  `*_utils.py` modules.
- **DRY** — extract shared logic; do not abstract prematurely (three similar lines beat
  one premature abstraction).
- **Self-documenting code, no excessive comments.** Comment *why*, never *what*. Clear
  names over comments.
- **Type hints everywhere; mypy strict.** `uv run mypy src/sema/` passes.
- **Logging via `from sema.log import logger`** (loguru), never stdlib `logging`.
- **All commands via `uv run`** (`uv run pytest`, `uv run mypy`, `uv run sema`).
- **Quality gate before every commit:** `uv run pytest` green, `uv run mypy src/sema/`
  clean, and `uv run pytest --cov=sema --cov-report=term-missing` with **coverage ≥ 85%**.
- **R29 boundary (architecture-load-bearing):** no `'Maps to'`, `standard_concept`,
  `concept_id`, OMOP table/domain literal, or `OncoTree` literal may appear in
  `src/sema/engine/` or in any generic spine module (`resolve/engine.py`,
  `compile/`, `src/sema/targets/` (the *merged* generic target framework — already
  R29-clean, verified zero hits), the assembler). OMOP/OncoTree specifics live **only**
  in policy objects (US-004) and **per-ontology manifest data** (US-007's authored
  OMOP manifest). **US-000** makes the current tree pass this; **US-001** makes it
  executable in CI. The denylist's **scope of grepped "core paths" explicitly excludes**
  `eval/`, `tests/`, `resolve/policies/`, `targets/adapters/`, and **`*.yaml`/`*.yml`
  manifest files** (the allowlist), so legitimate policy/gold-set/eval/manifest references
  to OncoTree/SNOMED/Condition do not trip the guard.
- **Conventional commits** (`feat:`/`fix:`/`test:`/`refactor:`/`docs:`), simple subject
  line naming the story ID, **no `Co-Authored-By` lines**.
- **Update `.wolf/` per OpenWolf:** append to `.wolf/memory.md`; update `.wolf/anatomy.md`
  on new/renamed files; log bugs to `.wolf/buglog.json`; record reusable patterns in
  `.wolf/cerebrum.md`.

**Code-reference note (rev 2).** The mapping-planner models live under
`src/sema/models/planner/` (not `src/sema/models/`): `mapping_plan.py`
(`MappingAssertion`, `MappingPlan.derive_verdict` @62, `select_winner` @112,
`PlanAssembler` Protocol @123), `field_map.py` (`FieldMap` @30), `target_model.py`
(`TargetObligation` @84, `ForeignKeyObligation` @28, `nullable_fields`), `patterns.py`
(`MappingPattern` enum @19, **no** `RENAME`/`TYPE_CAST`), `lifecycle.py` (`Status`).
Graph edge writes: `src/sema/graph/planner_loader.py` (`MAPS_TO` @101, `DERIVED_FROM`
@112 — both originate at the `:FieldMap` node, never at a source Property). There is a
second, unrelated `select_winner` at `src/sema/models/winner_selection.py:24` operating
on core `Assertion`s — **US-008 uses the planner one (`mapping_plan.py:112`)**.

**Target-framework reference (rev 4 — merged via PR #109, all verified present).** The
generic target subsystem US-007/US-014 build on already exists on `main`:
`src/sema/targets/loader.py:36` (`load_target(adapter, *, writer, selected_refs=…)`),
`targets/base.py:17` (`TargetOntologyAdapter` protocol; six required methods incl.
`load_obligation`, `load_vocabulary_bindings`), `targets/adapters/manifest.py:39`
(`ManifestTargetAdapter`, reads a `manifest_version: 1` YAML/JSON), `targets/materializer.py`
+ `targets/neo4j_writer_utils.py:265` (writes `:VocabularyBinding` with `model_role=TARGET`),
`targets/materializer_binding_card_utils.py:64` (the `HAS_VOCABULARY_BINDING` edge —
`(Property)-[:HAS_VOCABULARY_BINDING]->(VocabularyBinding)`). DTOs:
`models/target/vocab_binding.py:10` (`VocabularyBindingDecl` — carries `entity_ref`,
`property_name` (the obligation field, e.g. `condition_concept_id`), `vocabulary`,
`domain`, `require_standard`, `allow_zero_default`, `effective_date_ref`,
`resolver_policy_ref`), `manifest_models.py:33` (`ManifestVocabularyBinding`).
`models/target/obligation.py:5` **re-exports** the planner `TargetObligation`
(`models/planner/target_model.py:84`) — one canonical obligation type, not two. CLI:
`sema target load` (`cli_target.py:33`, registered at `cli.py:348`). The `targets/` unit
tests use an `InMemoryGraphWriter` double — hermetic, no Neo4j.

**Live-data doctrine (the user's "make sure everything works").** Unit tests are mocked
and gate every story for fast TDD. In addition, components that touch real data carry an
**`integration`-marked live test** that runs against the real **local DuckDB mirror**
`~/.sema/poc.duckdb` (present, 2.5 GB: `vocabulary_omop` 10M concepts, `concept_relationship`
54.9M, `cbioportal_*` studies) — no mocks, no Neo4j required for the resolver arm. The
**bulk write** stories additionally verify against **Databricks** (`workspace.vocabulary_omop`,
`workspace.cbioportal_*`, `workspace.ontology_omop`). Live tests are skipped (not failed)
when credentials/files are absent, via a `pytest.mark.integration` + skip-guard, so the
unit suite stays hermetic. US-012A is the full local end-to-end gate; US-013 is the full
live Databricks gate.

---

## 1.5 Slice-0 Contract Spec (frozen — referenced by US-002/005/008/009/010/012)

This block resolves the entangled findings (NO_MAP vs required field, eval denominators,
status persistence, store↔staging projection, property↔field bridge) with **one**
definition. Stories reference this section; they do not re-list columns. The frozen
schemas are enforced by a **drift test** (US-005) that fails if the column set changes.

### (a) Value-mapping store — *location-independent decision cache*

Grain: **one row per distinct source code per target binding** — deliberately **not**
scoped by source schema/table, so a `OncoTree:LUAD → concept` decision is reused across
every study (cross-study transfer cache; aligns with the vision and `valentine_unification`
note). `source_schema`/`source_table` belong to **staging**, never to the store.

Frozen columns:
`source_vocabulary, normalized_source_value, target_property_ref, target_field,
vocab_binding, concept_id, vocab_release, valid_start, valid_end, resolution_status,
no_map_reason, confidence, status, resolver_policy_ref, run_id`

- **Unique grain key:** `(source_vocabulary, normalized_source_value,
  target_property_ref, resolver_policy_ref, vocab_release)`. A second write for the same
  key **upserts**, never duplicates.
- `resolution_status ∈ {RESOLVED, NO_MAP}`. `concept_id` is **NULL iff
  resolution_status = NO_MAP**; `no_map_reason` is non-null iff NO_MAP.
- `status` is the lifecycle `Status` (`auto_accepted`, `human_pinned`, `candidate`,
  `review_pending`, `rejected`) carried from the winning `MappingAssertion`.
- `run_id` is a **provenance column, not part of the grain.**
- `target_property_ref` is the OMOP *Property* node ref; `target_field` is the obligation
  *field name* (e.g. `condition_concept_id`). Both are stored so the property↔field
  bridge (below) is explicit data, not inferred.

### (b) Staging table — *per source row*

Generic, **policy-named** columns. `src/sema/compile/` must never contain the showcase
literals; the policy/adapter supplies the names (R29). Showcase resolution shown in `()`:

`source_schema, source_table, source_row_ref (nullable), source_patient_key,
<source_value_column> (→ source_oncotree_code), <target_concept_column> (→
condition_concept_id), resolver_policy_ref, vocab_release, resolution_status,
no_map_reason, status, run_id` + provenance/run fields.

- **Projection rule:** staging ⋈ store on `normalized_source_value` (after the same
  normalization). `<target_concept_column>` = `store.concept_id` → **NULL when the code's
  store row is NO_MAP**. `<source_value_column>` = the raw source code.
- **No `person_id`, no synthetic `person` row, no FK closure** (cut Slice-0 non-goal).
- **Idempotency = atomic replace** via temp-table build + swap, **scoped on
  `(source_schema, source_table)`** (not bare `source_schema` — a schema may hold sibling
  studies). Re-running reproduces byte-identical rows. No general transaction coordinator
  (over-build); the minimal temp-build-then-swap contract is sufficient.

### (c) Zone model — *derived, not a stored column*

Zones are computed from `status` + `resolution_status`:

- **Zone-1 `AUTO_ACCEPTED_RESOLVED`:** `status ∈ {auto_accepted, human_pinned}` ∧
  `resolution_status = RESOLVED` ∧ `concept_id` non-null. → the precision /
  auto-resolution population.
- **Zone-2 `REVIEW_OR_UNRESOLVED`:** `status ∈ {candidate, review_pending, rejected}`.
  Excluded from the precision denominator; for a gold-`RESOLVED` code it is a **recall
  miss**. (Slice 0 only reaches Zone-2 on a genuine >1-survivor tie → single-model
  disambiguate; reported, not gated.)
- **Zone-3 `NO_MAP_ACCEPTED`:** `resolution_status = NO_MAP` ∧ `concept_id` NULL ∧
  `no_map_reason` non-null. → counted **only** in `no_map_accuracy`.

### (d) Status flow (fixes the hard-coded verdict)

`MappingAssertion.status` → **`FieldMap.status`** (new field; default
`Status.candidate`) → `MappingPlan.derive_verdict()` → `store.status` → `staging.status`
→ eval Zone assignment. `derive_verdict()` must compute `any_review_pending` **from the
field-map statuses**, replacing the hard-coded `False` at `mapping_plan.py:76`.
Serialization of the new field updates `planner_loader.py` (write + read).

### (e) Coverage rule (fixes B1 + the R19 `CONSTANT(NULL)` gap together)

A required obligation field is **covered** only by an **accepted, value-producing**
`FieldMap`. `MappingPattern.NO_MAP` and `MappingPattern.CONSTANT` with a NULL literal
**never** cover a required field. `covered_required_fields()` (`mapping_plan.py:59`) must
filter on pattern + status accordingly.

**Level separation (key):** the required *field* `condition_concept_id` is covered at the
**plan** level by the `VOCAB_LOOKUP` FieldMap (the mapping *exists* → plan can PASS); an
individual *code* resolving to NO_MAP is a **per-row store/staging outcome** (`concept_id
= NULL`), orthogonal to field coverage. Therefore `condition_concept_id` may remain a
**required** obligation field; we do **not** need to mark it `nullable`. NO_MAP rows are
data, not an un-covered field.

### (f) Confusion matrix — *one definition for US-002 and US-012*

Unit of evaluation = **distinct source code**. Gold label ∈ {`RESOLVED` (with
`gold_concept_id`), `NO_MAP`}. Predicted population partitioned by Zone (c).

|                                   | gold = RESOLVED | gold = NO_MAP |
|-----------------------------------|-----------------|---------------|
| pred Zone-1, `concept_id == gold` | **TP**          | —             |
| pred Zone-1, `concept_id != gold` | **WRONG**       | **FP_map**    |
| pred Zone-3 (NO_MAP)              | **FN**          | **TN**        |
| pred Zone-2 (review/unresolved)   | recall miss     | (n/a)         |

Metrics (all reported at **distinct-code**, **row-weighted**, and
**per-frequency-bucket** granularity):

- **`mapped_precision` = TP / (TP + WRONG + FP_map)** — of all Zone-1 auto-accepts,
  fraction correct. **Gate ≥ 95%.**
- **`mapped_recall` = TP / (TP + WRONG + FN)** — denominator = codes with a gold mapping;
  a Zone-2/Zone-3 prediction for a gold-`RESOLVED` code is a recall miss.
- **`auto_resolution_rate` = (TP + WRONG + FP_map) / all labelled distinct codes** — the
  Zone-1 share. **Gate ≥ 70%.** Assertable as "accepted" **only when labelled coverage =
  100% of observed distinct codes** (else: provisional).
- **`no_map_accuracy` = TN / (TN + FP_map)** — reported **separately**; NO_MAP is never
  folded into `mapped_precision`, `mapped_recall`, or `auto_resolution_rate`.

---

## 2. User Stories

> Each story is **one Ralph iteration**. `passes` starts `false`. The `priority:` field
> is the build order (IDs are stable labels, not the order). "Live verification" criteria
> use the `integration` marker and the skip-guard above.

### US-000 — R29 baseline migration: make the guard green on the current tree · priority 1
**As** a multi-domain platform, **I want** the OMOP/OncoTree literals that **already**
live in `src/sema/engine/` relocated or grandfathered **before** the R29 guard exists,
**so that** US-001 can fail-on-leak without failing on pre-existing code.

Acceptance criteria (+ all Global Standards):
- Failing test first: a test that enumerates engine-core literals and asserts the
  US-001 denylist would return **empty** against `src/sema/engine/` after migration.
- Inventory and neutralize the known leaks (verified present): `engine/domain_prompts.py`
  (`OncoTree` in the classification-systems prompt), `engine/vocabulary_utils.py`
  (`"oncotree"` in a vocab list), `engine/few_shot_healthcare_stage_a.py`
  (`oncotree_code` few-shot), `engine/few_shot_healthcare.py` (`cBioPortal` docstring).
  Each is either (a) moved into an **allowlisted** demo/policy/config location, or (b)
  generalized so the literal is data, not code — document which, per file.
- **PR #109 is now MERGED to `main` (rev 4 — supersedes the rev-2/3 "unmerged" note):**
  commit `ea2aed7` landed the real `src/sema/targets/` subsystem (loader, manifest adapter,
  materializer, `:VocabularyBinding` + `HAS_VOCABULARY_BINDING`, `ManifestVocabularyBinding`)
  and it is **R29-clean** (zero OMOP literals — confirm with the US-001 guard once it
  exists; this is a verification, not a migration). US-000 does **not** touch `targets/`;
  US-004/US-007 **build on it** (US-007 authors a manifest, US-004 consumes the binding) —
  no recovery, no reimplementation. Record this in `progress.txt`.

---

### US-001 — Executable R29 OMOP-coupling guard (CI fails on a leak) · priority 2
**As** a multi-domain platform, **I want** a build-failing check that no OMOP/OncoTree
literal leaks into the engine core, **so that** the "any ontology" promise is enforced
mechanically, not in review (R29).

Acceptance criteria (+ all Global Standards):
- Failing test first: a fixture engine-core file containing `'Maps to'` /
  `standard_concept` / `condition_occurrence` / `OncoTree` makes the guard return a
  non-empty violation list; a clean file returns empty.
- `scripts/check_engine_coupling.py` (or a `tests/` guard) greps a **configured set of
  core paths** (`src/sema/engine/`, and the spine modules created by later stories:
  `src/sema/resolve/engine.py`, `src/sema/compile/`, and the **merged** generic target
  framework `src/sema/targets/` (already R29-clean — keep it in the grep so it stays that
  way), assembler) for a **policy-owned denylist** of literals; **allowlisted** paths are
  the policy modules (US-004), `targets/adapters/`, **authored manifests (`*.yaml`/`*.yml`,
  US-007/US-014)**, `eval/`, and `tests/`.
- The denylist is **case-explicit** (covers `OncoTree`, `oncotree`, `cBioPortal`,
  `cbio`, …) so US-000's migration is verified, not bypassed by casing.
- Wire it into the test suite (a `unit`-marked test that imports and runs the checker)
  so `uv run pytest` fails on a leak — no separate CI config required.
- The denylist, allowlist, and core-path set live in one self-documenting module-level
  constant, not scattered.
- Depends on **US-000** (the guard must be green on the current tree the first time it
  runs).

---

### US-002 — Slice-0 OncoTree→SNOMED gold set + mapping eval harness (the test) · priority 3
**As** the team, **I want** every distinct `ONCOTREE_CODE` in the loaded studies
hand-labelled to its correct standard SNOMED `concept_id` (or `NO_MAP`), with an eval
harness that scores the **§1.5 confusion matrix**, **so that** "done" is defined before
the resolver exists (architecture §3 step 0; R20).

Acceptance criteria (+ all Global Standards):
- Failing test first for the harness: given a tiny synthetic gold set + a synthetic
  decision set with known Zone assignments, `mapped_precision`, `mapped_recall`,
  `auto_resolution_rate`, and `no_map_accuracy` compute to the **§1.5 values**; a
  predicted `NO_MAP` is **TN** when gold is `NO_MAP` and **FN** (recall miss) when gold is
  a real concept (never silently dropped); `no_map_accuracy` is reported **separately**.
- Gold set artifact `tests/data/gold/oncotree_condition_slice0.jsonl` (or `.csv`): one
  row per **distinct** observed `ONCOTREE_CODE`, columns
  `oncotree_code, gold_concept_id, gold_label (RESOLVED|NO_MAP), row_count, notes`.
  Distinct codes enumerated from `~/.sema/poc.duckdb` `cbioportal_*` (document the query).
  Hand-labelling **may** start with a documented subset (≥ the top codes by `row_count`
  covering ≥80% of rows) — but per §1.5(f), subset metrics are **provisional only**;
  acceptance thresholds (US-012) require 100% distinct-code coverage. Record the subset
  and its coverage.
- **Human-label checkpoint (acceptance-blocking; not Ralph-automatable).**
  `gold_concept_id` / `gold_label` are an **external oracle** — they must be
  hand-labelled by a human or imported from a trusted crosswalk, **never** generated
  by Sema's own resolver (US-006). Auto-labelling makes US-012 circular (the resolver
  grading its own homework). Ralph **may** scaffold the file, enumerate distinct codes
  from `~/.sema/poc.duckdb`, and fill `row_count`/`notes`, and may mark this story
  `passes:true` on the **harness + scaffold + documented subset**. Completing labels to
  100% distinct-code coverage is a **human gate** that US-012 acceptance (not just
  "running") depends on; Ralph must surface the unlabelled remainder, not invent labels.
- Eval harness `src/sema/eval/mapping_goldset.py` (+ `*_utils.py` for the math)
  implements the **§1.5(f) confusion matrix** exactly; unit of evaluation is the
  **distinct source code**; reports **distinct-code**, **row-weighted**, and
  **per-frequency-bucket** numbers. This module is the frozen home of the metric
  definitions referenced by US-012.
- `integration` live test: enumerate distinct `ONCOTREE_CODE` from `~/.sema/poc.duckdb`
  and assert the gold set covers every code observed (or records it as a known-unlabelled
  gap), so the gold set cannot silently drift from the data.
- This story ships **no resolver** — only the gold set + harness.

---

### US-004 — Per-vocabulary resolver policy object (OMOP/OncoTree behind the boundary) · priority 4
**As** the engine, **I want** OMOP specifics (`'Maps to'`, `standard_concept='S'`,
`invalid_reason IS NULL`, the OncoTree source vocabulary name, the Condition domain)
captured in a **policy object**, **so that** the resolver stays vocabulary-agnostic and
R9 (source-vs-target vocab conflation) is structurally impossible. **Built before the
vocab store (US-003)** so the store consumes policy literals rather than hardcoding them.

Acceptance criteria (+ all Global Standards):
- **Consume the merged `VocabularyBindingDecl` (rev 4 — PR #109 is on `main`):** the
  loaded TARGET binding (`models/target/vocab_binding.py:10`) already carries `domain`,
  `require_standard`, `allow_zero_default`, `effective_date_ref`, and `resolver_policy_ref`.
  The `ResolverPolicy` is the object that `resolver_policy_ref` **points to** — it adds
  the **source-side** specifics the binding does not hold (source vocabulary name,
  `'Maps to'`, standard flag, validity predicate). Do **not** duplicate the binding's
  fields; read `domain`/`require_standard`/`allow_zero_default` from the loaded binding
  (US-007) and resolve `resolver_policy_ref` → this policy. Cite
  `models/target/vocab_binding.py` in `progress.txt`.
- Failing tests first: a `ResolverPolicy` instance exposes `source_vocabulary`
  (`OncoTree`), `maps_to_relationship` (`'Maps to'`), `standard_flag` (`'S'`),
  `target_domain` (`Condition`), `allow_zero_default` (per-obligation, default off),
  and a validity predicate; the **engine code never names these literals**.
- `src/sema/resolve/policy.py` holds the policy dataclass; the **OMOP/OncoTree instance**
  lives in an allowlisted location (`src/sema/resolve/policies/omop.py`) — the only place
  these literals may appear (US-001 allowlist).
- Explicitly carries the R9 distinction: `resolver_policy_ref` names the **source**
  vocabulary (OncoTree); the contract's singular `vocabulary_ref` is the **target**
  (SNOMED). A test asserts the resolver, given a policy, matches the source code in the
  **source** vocabulary (not the target) at candidate-generation time.

---

### US-003 — Vocabulary store query layer over `vocabulary_omop` · priority 5
**As** the resolver, **I want** a thin SQL query layer over the OMOP vocabulary tables
that runs identically on DuckDB (dev) and Databricks (prod), **so that** resolution is
"the graph fits meanings; the warehouse moves rows" — 10M concepts stay in
DuckDB/Databricks, never Neo4j. **Consumes the US-004 policy** for relationship/flag
literals.

Acceptance criteria (+ all Global Standards):
- Failing tests first (mocked connection): `concept_by_code(vocab, code)`,
  `maps_to_targets(concept_id)`, `concept_domain(concept_id)`, and a validity/standard
  filter each emit the expected SQL and shape the expected rows.
- `src/sema/resolve/vocab_store.py`: a `VocabStore` reading `concept`,
  `concept_relationship`, `concept_synonym` (+ `domain`) with a backend selected by
  config (DuckDB path `~/.sema/poc.duckdb` or Databricks `workspace.vocabulary_omop`).
- SQL is built as a **SQLGlot AST rendered per dialect** (`sqlglot>=25.0.0`, already a
  dep, `pyproject.toml:22`), so one query authored once runs on both backends. No
  hand-concatenated SQL strings.
- **No OMOP literals in this module** beyond table/column names passed in as config — the
  relationship name (`'Maps to'`) and standard flag come from the **US-004 policy**, not
  hardcoded here.
- `integration` live test against `~/.sema/poc.duckdb`: `concept_by_code('OncoTree',
  'LUAD')` returns a real row; `maps_to_targets(...)` returns ≥1 standard SNOMED concept;
  query latency on the 54.9M `concept_relationship` table is bounded (record it).

---

### US-005 — Resolved value-mapping store (DuckDB table, §1.5 frozen columns) · priority 6
**As** the compiler, **I want** a concrete executable store of `source_code → concept_id`
decisions with the **§1.5(a) frozen columns and location-independent grain**, **so that**
there is somewhere to read resolved decisions from (the compiler's hard prerequisite).

Acceptance criteria (+ all Global Standards):
- Failing tests first: write N distinct-code decisions, read them back; grain is enforced
  as **one row per §1.5(a) grain key** (a second write for the same key upserts, never
  duplicates); `NO_MAP` rows persist with `concept_id = NULL` and a `no_map_reason`;
  `status` round-trips.
- `src/sema/resolve/value_mapping_store.py`: DuckDB-backed table with the **exact §1.5(a)
  column list** (the store is **location-independent** — no `source_schema`/`source_table`).
- **Schema drift test (freezes the contract):** an `integration` test asserts the live
  table's column set equals the §1.5(a) frozen list **exactly** — adding/removing/renaming
  a column fails the test. This is the mechanism that keeps every downstream story aligned.
- Canonical home is **DuckDB** (small store); no Databricks seed sync (the compiler
  inlines decisions, US-010). Document this in the module docstring.
- **No SSSOM dependency** (`sssom-py`) — SSSOM is a later export projection only; the
  store is defined on its own §1.5(a) columns first.
- `integration` live test: round-trip a handful of real resolved codes against a temp
  DuckDB file; assert the schema matches the frozen list.

---

### US-006 — Vocabulary resolver (the §4 algorithm) · priority 7
**As** Sema, **I want** the deterministic resolver that turns a distinct source code
into a standard target `concept_id` (or `NO_MAP`), **so that** `VOCAB_LOOKUP` values
become real decisions (architecture §3 step 2; fitting doc §4; R8/R9/R10/R13).

Acceptance criteria (+ all Global Standards):
- Failing tests first (mocked `VocabStore` + policy), one per stage:
  (1a) exact code match in the **source** vocabulary; (2) `'Maps to'` standardization to
  `standard_concept='S'`, keeping only `invalid_reason IS NULL` and validity-window
  concepts; (3) **domain gate** rejects any candidate whose `domain_id != target_domain`;
  emit `MappingAssertion(pattern=VOCAB_LOOKUP, status=auto_accepted, …)` on one survivor;
  **first-class `NO_MAP`** (resolution_status=NO_MAP, Zone-3) when zero standard
  candidates survive.
- `src/sema/resolve/` modules per the fitting doc §5 layout, each thin and ≤400 lines:
  `candidates.py`, `standardize.py`, `domain_gate.py`, `engine.py` (orchestration →
  emits assertions), with the ambiguous-tail `disambiguate.py` present but **single-model
  only** (no council; council is Slice 1+). The Slice-0 hot path (`ONCOTREE_CODE`) is
  **pure SQL — invokes no LLM at all**; the LLM fires only on a genuine >1-survivor tie
  (→ Zone-2).
- **Code-bearing short-circuit:** for code-bearing inputs the engine takes the SQL path
  and never touches embeddings/vector recall (step 1d excluded from Slice 0).
- Decisions are persisted to the **value-mapping store** (US-005), one row per §1.5(a)
  grain key, including `NO_MAP` rows, carrying `status` and `resolution_status`.
  **US-006 is the *sole writer* of value-mapping-store rows** — every downstream story
  (US-009 producer, US-010 compiler, US-011 QA, US-012 eval) **reads** the store and
  never writes it. This removes the rev-2 ambiguity where US-006 and US-009 both claimed
  the per-code write.
- `integration` live test against `~/.sema/poc.duckdb`: resolve a real sample of OncoTree
  codes; assert `LUAD` (and other gold-labelled codes) resolve to their gold `concept_id`;
  assert at least one known dead-end code yields `NO_MAP`; **measure and record
  OncoTree→SNOMED path coverage** (R10) — coverage is reported, not asserted here
  (US-012 asserts the threshold).

---

### US-007 — OMOP target via authored manifest + the merged loader (`model_role=TARGET`) · priority 8
**As** the matcher, **I want** the OMOP `condition_occurrence` slice loaded as
`model_role=TARGET` entities, properties, obligations, and a vocabulary binding, **so
that** there is a target to map *to* (architecture §3 step 1). **Rev-4: the target
framework is already merged (PR #109).** This story therefore **authors an OMOP manifest
(data) and loads it via the existing `load_target` / `ManifestTargetAdapter`** — it writes
**no** new loader, materializer, or graph-writer code. Independent of the resolver —
parallelizable.

Acceptance criteria (+ all Global Standards):
- **Build on the merged framework — do not reimplement it.** Reuse `targets/loader.py`
  `load_target(...)` (`:36`), `ManifestTargetAdapter` (`adapters/manifest.py:39`), the
  `targets/materializer.py` → `Neo4jGraphWriter`/`InMemoryGraphWriter` path, and the
  `sema target load --manifest <path>` CLI (`cli_target.py:33`). The only new artifact is
  the **manifest file** (+ a tiny amount of glue/config if needed). If any framework gap
  is found, prefer a minimal fix in `targets/` over a parallel implementation; record it.
- **Authored manifest** `targets/manifests/omop_condition_slice0.yaml` (allowlisted
  OMOP-literal location, R29): `manifest_version: 1`, one entity `condition_occurrence`
  with its obligation (`required_fields` incl. `condition_concept_id`; PK; the FK
  obligations the slice needs) and a `condition_concept_id` property carrying a
  `vocabulary_binding` with `vocabulary` = the SNOMED target, `domain: Condition`,
  `require_standard: true`, and `resolver_policy_ref` → the US-004 OMOP/OncoTree policy.
  (This is the *target* model only — no `person_id`/dates beyond what `condition_occurrence`
  declares; Slice-0 staging drops identity per §1.5(b).)
- Failing tests first (hermetic, `InMemoryGraphWriter`): loading the manifest via
  `load_target` materializes `model_role=TARGET` `:Entity`/`:Property`/`:TargetObligation`,
  **a `:VocabularyBinding` node and a `HAS_VOCABULARY_BINDING` edge**
  (`(Property)-[:HAS_VOCABULARY_BINDING]->(VocabularyBinding)`, per
  `materializer_binding_card_utils.py:64` — **note the edge is `HAS_VOCABULARY_BINDING`,
  not `HAS_VOCAB_BINDING`**), and the binding's `property_name` is the obligation field
  `condition_concept_id` with `domain='Condition'` (the §1.5 property↔field bridge, which
  `VocabularyBindingDecl` already carries as data).
- **No OMOP literal in `src/sema/targets/` code** (R29 — already true on `main`); all
  OMOP specifics live in the authored manifest, which US-001 allowlists.
- `integration` live test: `sema target load --manifest targets/manifests/
  omop_condition_slice0.yaml --writer neo4j` against Neo4j **(skip-guarded)** asserts the
  `condition_concept_id` binding exists with `domain='Condition'`; the **hermetic**
  `--writer in-memory` assertion above is the always-on gate.
- **Neo4j visibility assertion (what you'll see in the graph).** The live test runs this
  Cypher and asserts exactly one row (`condition_occurrence | condition_concept_id |
  Condition | true`), proving the TARGET schema is materialized and traversable:
  ```cypher
  MATCH (e:Entity {model_role:'TARGET'})-[:HAS_PROPERTY]->(p:Property)
        -[:HAS_VOCABULARY_BINDING]->(vb:VocabularyBinding)
  WHERE vb.property_name = 'condition_concept_id'
  RETURN e.name, vb.property_name, vb.domain, vb.require_standard
  ```
  It also asserts the obligation is present:
  `MATCH (e:Entity {model_role:'TARGET'})-[:HAS_OBLIGATION]->(o:TargetObligation)
  RETURN o` returns the `condition_occurrence` obligation.

---

### US-008 — Slice-0 staging obligation + real `PlanAssembler` · priority 9
**As** the spine, **I want** a working `PlanAssembler` that composes assertions against
the **Slice-0 staging obligation** into a `MappingPlan` with a verdict, **so that** plans
assemble and fail as a unit (architecture §1 L4, §5; replaces the Protocol stub; R15;
closes the L4 `CONSTANT(NULL)` caveat R19).

Acceptance criteria (+ all Global Standards):
- Failing tests first: assembling a resolved `condition_concept_id` assertion against the
  **staging obligation** `required = {condition_concept_id, resolver_policy_ref,
  vocab_release}` yields a `PASS` verdict; a missing required field yields `FAIL`; a
  `CONSTANT(NULL)`-covered required field yields `FAIL` and a `NO_MAP`-pattern FieldMap
  does **not** cover a required field (the **§1.5(e) coverage rule** — closes R19
  structurally). Per §1.5(e), `condition_concept_id` stays **required** (not nullable);
  per-code NO_MAP is a row outcome, not an un-covered field.
- Add **`FieldMap.status: Status`** (default `candidate`) per §1.5(d); the assembler sets
  it from the winning assertion's status; `derive_verdict()` computes `any_review_pending`
  from field-map statuses (replacing the hard-coded `False` at `mapping_plan.py:76`);
  update `planner_loader.py` serialization (write + read) for the new field.
- `src/sema/resolve/assembler.py` (or `models/planner/` adjacency): a concrete
  `PlanAssembler` implementation; reuse the **planner** `select_winner`
  (`mapping_plan.py:112` — *not* `winner_selection.py:24`) and `derive_verdict`
  (`mapping_plan.py:62`); extend `covered_required_fields` per §1.5(e). Update the
  `PlanAssembler` Protocol docstring (which currently defers implementation to the
  "matching-engine change" — that engine is Slice 1+; this story supplies the concrete
  Slice-0 assembler).
- The staging obligation is **distinct from** the production `condition_occurrence`
  obligation (no `person_id`, no dates) and is defined as Slice-0 policy/config, not in
  engine core.

---

### US-009 — Deterministic `VOCAB_LOOKUP` assertion producer + graph edges · priority 10
**As** Sema, **I want** the narrow deterministic producer that turns
(target binding + source `:ValueSet` + resolver) into `MappingAssertion`s and writes the
`:FieldMap`/`MAPS_TO`/`DERIVED_FROM` edges, **so that** the mapping exists in the graph
(architecture §3 step 3; R8). **Needs none of the generic matching engine** — candidate
generation / pattern classification / graph-expansion / LLM judgment are Slice 1+.

Acceptance criteria (+ all Global Standards):
- Failing tests first: given a source `:ValueSet` of OncoTree codes, the target
  `condition_concept_id` binding (the `:VocabularyBinding` loaded by US-007 via
  `HAS_VOCABULARY_BINDING`, carrying `property_name=condition_concept_id`), and a
  (mocked) resolver, the producer emits one
  `MappingAssertion(pattern=VOCAB_LOOKUP)` carrying `source_field_ref` **and**
  `target_property_ref`, and materializes a `:FieldMap` whose **`target_field_ref` is the
  obligation field name** (`condition_concept_id`, per the §1.5 bridge) with `MAPS_TO →`
  target Property and `DERIVED_FROM →` source Property (real edge shape,
  `planner_loader.py:101/112` — `MAPS_TO` never originates at a source Property).
- `src/sema/resolve/producer.py` (thin); reuses `planner_loader.py` for the edge writes.
- **No `MappingProposal`.** (Cut — it does not exist in code and is unnecessary for one
  deterministic pattern.) The producer emits `MappingAssertion`s and
  writes the graph edges; it **reads** per-code decisions from the value-mapping store
  (written solely by US-006 — **the producer never writes the store**) to populate the
  `:FieldMap`. Eval reads the **store**, not a proposal shape. The chain is
  `MappingAssertion → FieldMap → MappingPlan` only.
- `integration` live test: run the producer over a real study's `ONCOTREE_CODE`
  `:ValueSet` (built source graph) + the US-007 target binding; assert the `:FieldMap`
  edges exist with the correct direction.
- **Neo4j visibility assertion (the source→target bridge).** The live test runs this
  Cypher and asserts exactly one row (`ONCOTREE_CODE | VOCAB_LOOKUP | condition_concept_id`),
  proving the SOURCE and TARGET halves are stitched together and the mapping is traversable
  in the graph:
  ```cypher
  MATCH (src:Property {model_role:'SOURCE'})<-[:DERIVED_FROM]-(fm:FieldMap)
        -[:MAPS_TO]->(tgt:Property {model_role:'TARGET'})
  RETURN src.name, fm.pattern, tgt.name
  ```
  The test also asserts **direction** explicitly: `MAPS_TO` originates at the `:FieldMap`
  (never at a source `:Property`), and `DERIVED_FROM` points from the `:FieldMap` to the
  SOURCE `:Property` (per `planner_loader.py:101/112`). The per-value crosswalk
  (`LUAD→concept_id`) is **not** asserted here — it lives in the value-mapping store
  (US-005), not as graph edges.

---

### US-010 — Transform compiler (`MappingPlan` → SQLGlot → staging write) · priority 11
**As** Sema, **I want** the compiler that renders a `MappingPlan` to SQL via SQLGlot and
writes the **§1.5(b) staging artifact** on both DuckDB and Databricks, **so that** the
pipeline actually moves rows (architecture §5, L5 backends note; R16). Hard-blocked on the
frozen store columns (US-005).

Acceptance criteria (+ all Global Standards):
- Failing tests first: compiling a `MappingPlan` for `VOCAB_LOOKUP` builds a SQLGlot AST
  that **inlines** resolved decisions as `JOIN (VALUES ('LUAD', 4314337), …) AS m(code,
  concept_id)` and renders to **both** DuckDB and Databricks dialects; the AST is built
  once and dialect-rendered (no per-dialect string templates).
- `src/sema/compile/` (`compiler.py` + `*_utils.py`): dispatch on `(MappingPattern,
  TargetArtifactKind)`; Slice 0 implements `VOCAB_LOOKUP → TABLE_ROW`.
- Writes the **§1.5(b) staging table** with **policy-owned generic column names**
  (`<source_value_column>`, `<target_concept_column>`, `status_column`) — `src/sema/compile/`
  **must not contain** the literals `source_oncotree_code` or `condition_concept_id`
  (R29; verified by US-001). The showcase names come from the US-004 policy / US-007
  adapter. **No `person_id`, no synthetic `person` row, no FK closure.**
- `<target_concept_column>` is **NULL for NO_MAP-status rows** (§1.5(b) projection).
- **Idempotency = atomic temp-table build + swap, scoped on `(source_schema,
  source_table)`** (§1.5(b)) — re-running a study reproduces byte-identical staging rows;
  no general transaction coordinator.
- `integration` live test: compile + execute against `~/.sema/poc.duckdb` (writes a temp
  staging table; assert row count = source `ONCOTREE_CODE` row count, the join populated
  `<target_concept_column>`, and NO_MAP rows are NULL there); **and** render the Databricks
  dialect SQL and assert it parses (full Databricks execution is US-013).

---

### US-011 — Gate D-lite staging QA · priority 12
**As** the team, **I want** a staging QA check (row count, null-rate on the resolved
concept column, `NO_MAP` accounting vs the gold set), **so that** a bad staging table is
caught before it is called "done" (architecture §3 step 5, §5; R18). **Not** full Gate D
/ OHDSI DQD (Slice 1+).

Acceptance criteria (+ all Global Standards):
- Failing tests first: a staging table with the wrong row count, an unexpected null-rate
  on `<target_concept_column>` (accounting for legitimate NO_MAP NULLs), or `NO_MAP`
  counts that disagree with the gold set each **fail** the check with a structured reason;
  a clean table **passes**.
- `src/sema/compile/staging_qa.py` (or `eval/`): computes the three checks; null-rate is
  reconciled against `resolution_status` (NO_MAP rows are *expected* NULLs, not defects);
  `NO_MAP` accounting is reconciled against the US-002 gold set (a code the gold set says
  is `RESOLVED` but whose staging row is `NO_MAP` is reported, not silently dropped).
- `integration` live test against the DuckDB staging table written in US-010: QA passes on
  a correct run and fails on a deliberately corrupted copy.

---

### US-012 — Mapping eval report (confusion matrix vs gold set, thresholds) · priority 13
**As** the team, **I want** the eval harness (US-002) run end-to-end over the real
resolved decisions to produce the **§1.5(f)** metrics with acceptance thresholds, **so
that** "reports precision/recall" is not hand-wavy (architecture §5 "measured bar"; R20).

Acceptance criteria (+ all Global Standards):
- Failing tests first: given a synthetic decision set + gold set, the report computes
  `mapped_precision`, `mapped_recall`, `auto_resolution_rate`, and `no_map_accuracy` per
  **§1.5(f)**, with `no_map_accuracy` reported **separately** (not folded into the other
  three), at distinct-code, row-weighted, and per-frequency-bucket granularity.
- `src/sema/eval/mapping_report.py` (+ `*_utils.py`): emits a structured report
  (JSON + human summary) reading the value-mapping store (US-005) and the gold set;
  reuses the US-002 metric module (no re-derived math).
- **Thresholds asserted** in the `integration` live run (against real resolved decisions):
  **≥95% `mapped_precision`** and **≥70% `auto_resolution_rate`** — **only when labelled
  gold coverage = 100% of observed distinct codes** (§1.5(f)); on a subset the report says
  **"provisional — not accepted."** Below either threshold (at full coverage): **"running,
  not accepted."**
- The report **explicitly states** that Slice 0's ~100% precision is *structural* (a
  deterministic exact-code walk) and does **not** validate the product precision approach
  on the ambiguous tail (architecture §5 "read precision honestly"; R21).
- **Depends on the US-002 human-label checkpoint:** "accepted" (vs "provisional")
  requires a 100%-coverage gold set whose `gold_concept_id`s were set by a human/trusted
  crosswalk, never by Sema's resolver. **Ralph cannot self-certify acceptance** — at less
  than human-labelled 100% coverage the report emits "provisional — not accepted" and
  surfaces the coverage gap; it must not flip US-013's acceptance on self-generated labels.

---

### US-012A — DuckDB end-to-end smoke: full spine, no Databricks · priority 14
**As** the team, **I want** the whole resolve → produce → assemble → compile →
staging-write → QA → eval chain exercised **once, end-to-end, on local DuckDB**, **so
that** interface drift between the independently-built units is caught **before** the live
Databricks gate (closes the "first integration is US-013" risk).

Acceptance criteria (+ all Global Standards):
- A `sema` CLI surface for the slice exists: extend `cli.py` with `sema fit` (or `sema
  resolve` + `sema materialize`) running the full chain for a given study + target binding
  against the **DuckDB** backend. (Today `cli.py` has `build/context/review/query/ingest/
  push/eval/**target**` — the `target` group was added by PR #109 — and `orchestrate.py`
  stops after embeddings; this is the first wiring of the *resolve→compile* spine into a
  single command.) The TARGET load is a **prerequisite step**, performed by the existing
  `sema target load` (US-007) — `sema fit` consumes the already-materialized TARGET graph /
  binding rather than re-loading it.
- `integration` live test (skip-guarded on `~/.sema/poc.duckdb`): run the full chain on a
  **small fixture study**; assert a staging table is written, Gate D-lite passes, and the
  eval report is produced. All shapes are `MappingAssertion → FieldMap → MappingPlan`
  (no `MappingProposal`); the value-mapping store and staging table match §1.5(a)/(b).
- **No Databricks objects are written.** This story is the local end-to-end gate; US-013
  is then *only* live Databricks execution + idempotency.

---

### US-013 — LIVE end-to-end Slice 0 run (Databricks + cBioPortal + OMOP) · priority 15
**As** the user, **I want** the whole Slice 0 pipeline run live against real Databricks
cBioPortal source + OMOP vocabulary/CDM, **so that** I can confirm *everything actually
works* — the explicit ask of this PRD.

Acceptance criteria (+ all Global Standards):
- Reuses the US-012A CLI surface with the **Databricks** backend (no new chain logic —
  US-012A already wired and proved it locally).
- `e2e`/`integration` live test (skip-guarded on credentials): against `~/.sema/poc.duckdb`
  **and** `workspace.*` on Databricks:
  - resolve distinct `ONCOTREE_CODE` for a real study (e.g. `msk_chord_2024`) → store;
  - compile + **execute on Databricks**, writing the §1.5(b) staging table in `workspace`
    (atomic replace scoped on `(source_schema, source_table)`); assert row count = source
    condition-row count, no `person_id` column present;
  - Gate D-lite passes; the eval report meets the US-012 thresholds at full gold coverage.
- Re-running the command is **idempotent** (second run yields identical staging rows).
- Document the exact `uv run sema …` invocation, the Databricks objects touched, and the
  measured metrics in `progress.txt` and a short runbook under `docs/runbooks/`.

---

### US-014 — Generality counter-example: no-vocab target (R29 boundary proof) · priority 16
**As** a *general* engine, **I want** one non-OMOP target — a plain analytics table with
**no vocabulary at all** — to flow through the same target-loader → assembler → compiler →
staging-write with **zero** concept resolution, **so that** the engine/policy boundary is
proven real, not asserted (architecture §3.2; R29).

Acceptance criteria (+ all Global Standards):
- Failing tests first: a **second authored manifest** `targets/manifests/dim_customer.yaml`
  for a curated `dim_customer` target declaring obligations (required columns, PK, types)
  but with **no `vocabulary_binding`** on any property. Loaded via the **same**
  `load_target` / `ManifestTargetAdapter` (US-007) — proving the framework is target-agnostic.
  A CSV source (`customers.csv`) flows through assembler → compiler; the resolver step is
  **skipped entirely**; every field map uses an **existing** `MappingPattern` —
  `DIRECT_COPY` (copy/rename) and `DERIVED` (a cast expression via `DerivedExpression`)
  and/or `CONSTANT` — with **no `concept_id`, no `'Maps to'`, no domain gate**. (Cut: do
  **not** add `RENAME`/`TYPE_CAST` enum members — they do not exist and are unnecessary;
  existing patterns cover the proof.)
- Loading the no-vocab manifest writes **no `:VocabularyBinding` node** and **no
  `HAS_VOCABULARY_BINDING` edge** — assert their absence (the positive complement of US-007).
- The same target loader (US-007), assembler (US-008), and compiler (US-010) handle this
  target **without OMOP-shaped branches** — if any needs an OMOP-specific code path, that
  is the bug this story exists to catch.
- `integration` live test: load `dim_customer.yaml` (hermetic `in-memory` writer) and
  compile + execute the `dim_customer` write against DuckDB from a fixture CSV; assert the
  rows materialize with no vocabulary columns and the R29 guard (US-001) stays green.

---

## 3. Definition of Done (the slice is *accepted*, not just *running*)

Architecture §5 acceptance, made executable here:
- The R29 guard is green on the current tree and fails on a leak (US-000/US-001).
- OMOP is in the graph as `model_role=TARGET` with the `condition_occurrence` obligation
  + Condition-domain binding (US-007).
- The resolver turns distinct ONCOTREE codes into standard SNOMED `concept_id`s with
  measured coverage and first-class `NO_MAP`, persisted to the §1.5(a) location-independent
  value-mapping store (US-005/US-006).
- The deterministic `VOCAB_LOOKUP` producer + real assembler emit a `MappingPlan` with a
  verdict against the **staging** obligation, with status carried through (US-008/US-009).
- The SQLGlot compiler writes the §1.5(b) staging table in Databricks, idempotently, with
  **no `person_id`** and policy-owned column names (US-010).
- Gate D-lite passes; the eval report meets **≥95% `mapped_precision`** and **≥70%
  `auto_resolution_rate`** at **100% human-labelled gold coverage** (US-002 oracle, not
  resolver-generated), with `no_map_accuracy` reported separately (US-011/US-012).
- The full spine runs locally end-to-end (US-012A), then **live on Databricks**
  (US-013), in the **single-model (no-council)** configuration.
- A non-bio target proves the R29 boundary (US-014).

Below either threshold, or below full gold coverage → the slice is **"running"/"provisional"
but not "accepted."**

---

## 4. Out of scope (Slice 1+ — building any of these here is drift)

- **Synthetic identity** (`person_id`, synthetic `person` rows, FK closure) and
  event-date semantics (R12/R14) — cut by construction; the half-measure placeholder is
  the most dangerous state. Slice 1.
- **LLM council / jury** (multi-model agreement) — single-model is the Slice-0 config;
  the council slots in at the gate later without re-architecture (R21).
- **`MappingProposal` intermediate shape** and **generic matching engine** (source↔target
  candidate generation, pattern classification, graph-expansion). Slice 1+/2.
- **`RENAME` / `TYPE_CAST` pattern types** — not needed; existing `DIRECT_COPY`/`DERIVED`/
  `CONSTANT` cover Slice 0 and US-014. Add only if a later slice needs them.
- **Free-text / vector recall arm** (CANCER_TYPE → SNOMED, analytes → LOINC, drugs →
  RxNorm), the **target-side vector index** (R11). Slice 2.
- **Full Gate D / OHDSI DQD**, **SQLMesh**, **Splink** (identity), **SSSOM export**,
  **OAK** — all un-spiked adoptions; spike before adopting.
- **Entity-half namespacing** (R2) and **cross-study join inference** (R5) — source-graph
  hardening tracked in `data-unification-analysis.md`, not this slice.

---

## 5. Notes for the Ralph converter

- Branch from `main` as `ralph/feat/mapping-slice0`.
- Convert each US to a `prd.json` story with `id`, `title`, `description`,
  `acceptanceCriteria` (inline §1 Global Standards **and** the relevant §1.5 Contract Spec
  subsections into each story), `priority` (the build order — **US-000=1 … US-014=16**;
  IDs are stable labels, priority dictates order), `passes: false`, `notes: ""`.
- One story per iteration, highest-priority `passes:false` first; commit only on a green
  quality gate (pytest + mypy + coverage ≥ 85%); set `passes:true` and append to
  `progress.txt` with a Learnings section.
- §1.5 is the single source of truth for schemas/metrics; a story that touches the store,
  staging, verdict, or eval must reference §1.5, not restate columns — the US-005 drift
  test enforces it.
- `integration`/`e2e` live tests are skip-guarded; the unit suite must stay hermetic and
  green without Databricks/Neo4j.
- **Two human gates the loop must not auto-satisfy:** (1) US-002 `gold_concept_id`
  labelling (external oracle — Ralph scaffolds, a human labels to 100% coverage); (2)
  US-012/US-013 *acceptance* depends on that human-labelled coverage. Ralph may reach
  `passes:true` for the harness, resolver, spine, and a **provisional** eval autonomously,
  but "accepted" requires the human gold gate. Surface the gap; never invent gold labels.
- **Store-writer ownership:** US-006 is the sole writer of the value-mapping store;
  US-009/US-010/US-011/US-012 read it.
- **Target framework is merged (rev 4):** PR #109 (`ea2aed7`) is **on `main`** — US-007
  and US-014 **author manifests and load via the existing `targets/` framework** (`sema
  target load`), writing no new loader code. The edge is `HAS_VOCABULARY_BINDING` (not
  `HAS_VOCAB_BINDING`). Do **not** parse `OMOP_CDMv5.4_Field_Level.csv`; do **not**
  reimplement `targets/`. Branch `ralph/feat/mapping-slice0` from `main` **after** this PR
  (the data-unification branch) merges, so both the target framework *and* the source-graph
  hardening are present.
