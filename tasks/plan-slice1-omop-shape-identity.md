# Plan — Slice 1: materialize cBioPortal → OMOP **table shape** (condition_occurrence + person), full identity resolution

> Date: 2026-07-22 · Status: **planning** · Branch: TBD (`{author}/feat/omop-shape-identity`, from `main` after slice-0 merges)
> Follows: `tasks/prd-sema-mapping-slice0.md` (§4 "Out of scope" parks exactly this),
> `tasks/decision-slice0-omop-target.md`, `tasks/plan-slice0-full-repoint.md` (executed).
> Durable record (docs/ and .wolf/ are gitignored). Mirrors the slice-0 story style.

## Goal

Take the resolved cBio→OMOP concept mapping (slice-0, **done**) and materialize a
**production-shaped OMOP `condition_occurrence` staging table** in Databricks — one that is
FK-closed against a materialized `omop.person` table — with **full identity resolution**
(deterministic map **plus** cross-source person dedup), per the user's 2026-07-22 decision.

Slice-0 deliberately wrote a *narrow* artifact (one resolved column + provenance). Slice-1
crosses the line slice-0 drew: it adds the three things slice-0 cut **by construction**.

## The gap (verified against current code, 2026-07-22)

`sema fit` today writes a per-source-row staging table whose only OMOP column is the resolved
`condition_concept_id`. The `Slice0PlanAssembler` (`src/sema/resolve/assembler.py`) maps against
a reduced **staging obligation** `{condition_concept_id, resolver_policy_ref, vocab_release}`.
The manifest (`src/sema/targets/manifests/omop_condition_slice0.yaml`) **already declares** the
full target model — `omop.person` (PK `person_id`) and `omop.condition_occurrence` with
`required_fields = [condition_occurrence_id, person_id, condition_concept_id,
condition_start_date]` and a `foreign_keys` closure to `omop.person`. So the *target schema* is
present; what's missing is the machinery to satisfy it:

| OMOP-shape requirement | State today | Slice-1 work |
|---|---|---|
| `condition_concept_id` | ✅ resolved + staged (slice-0) | reuse as-is |
| `person_id` | ❌ omitted from contract (never placeholdered) | **identity resolution** (this plan's core) |
| `omop.person` table | ❌ declared in manifest, never materialized | materialize + FK closure |
| `condition_start_date` | ❌ omitted (manifest req. non-null) | S1-00 measured **no absolute date exists** → D4(b): manifest → `nullable:true`, no fabrication |
| `condition_occurrence_id` | ❌ omitted | deterministic surrogate PK from `SAMPLE_ID` (S1-00-confirmed clean key), not `person_id` |
| multi-table FK-closed write | ❌ compiler writes one table | ordered person→condition write |

## Non-negotiable principle (why this is delicate)

The slice-0 PRD names the trap directly: *"the half-measure placeholder is the most dangerous
state."* Identity must be a **real mapping**, never a fabricated `person_id`. And per
`data_unification_analysis` (memory), sema has **no instance-level entity resolution today** and
its bare-key MERGE identity **over-collapses** (`M`(Male) ≡ `M`(Mississippi)). Full identity
resolution therefore inherits a documented over-collapse risk that this plan must actively gate,
not assume away.

## Architecture — apply the proven two-producer pattern to identity

Slice-0's spine is *deterministic-first, gated-fuzzy-second* (Producer #1 vs the deferred
Producer #2). Reuse that exact shape for person identity so we ship a correct OMOP table before
taking on the hard dedup:

### Stage A — deterministic identity spine (the OMOP shape unlock)
- **Person registry / store — TWO-LEVEL, so Stage B can revise identity without rewriting PKs.**
  Mirror `value_mapping_store.py` (DuckDB-canonical, one writer, downstream reads only), but do
  **not** make `person_id` a hash of the source key. Store two mappings:
  (1) `(source_schema, source_patient_key)` → `source_patient_uid` (a stable per-source-patient
  surrogate), and (2) `source_patient_uid` → `person_id` (the canonical OMOP PK, assigned via a
  monotonic registry). Stage A writes each source patient as its own `person_id` (identity map).
  **What Stage B remapping does and does NOT do (corrected):** remapping `source_patient_uid →
  person_id` in the registry does **not** retroactively edit rows already written to
  `condition_occurrence` — a persisted `person_id=2` stays 2 until the table is rebuilt. Dedup
  therefore materializes only by **rebuilding `condition_occurrence` from source through the
  *current* registry** and replacing `omop.person` with the surviving canonical persons. The
  stability guarantee is narrower and precise: **`condition_occurrence_id` (the row PK) is stable
  across dedup; the row's `person_id` FK is recomputed on rebuild.** That is exactly why the PK
  must derive from source-row identity (S1-05), not from `person_id`. This still reverses D1's hash
  recommendation — a hash `person_id` is 1:1 by construction and can never collapse — but the win
  is "PK-stable rebuild," not "no rewrite." Stage B rebuild scope + orphan-person retirement are
  specified in S1-10.
- **Materialize `omop.person`** from distinct canonical `person_id`s; **close the FK** so every
  `condition_occurrence.person_id` references a real person row. Reuse the existing US-014
  projection path (`compiler.compile_projection` / `execute_projection`) — person rows are a
  plain `CONSTANT`/`DIRECT_COPY` projection of distinct source patients, no vocabulary, no resolver.
- **Event date — gated, not assumed.** Map the source clinical date column → `condition_start_date`
  via an existing `MappingPattern` (`DIRECT_COPY`/`DERIVED`), no new pattern type — **but only if
  the source actually carries a usable absolute date.** cBio clinical data is frequently
  de-identified to relative day-offsets from diagnosis, and the manifest marks
  `condition_start_date nullable:false`. A required non-null date with no real source date forces
  fabrication — the same trap identity must avoid. See S1-04's feasibility probe and D4.
- **Missing-key disposition (explicit).** FK closure is non-negotiable, so a condition row whose
  `source_patient_key` is blank/missing **cannot** be written FK-valid. It is NOT dropped silently
  and NOT given a synthetic person: it routes to the slice-0 NO_MAP / review artifact. Define the
  disposition in code (S1-02), not as a runtime surprise.
- **Multi-table FK-closed compile — ordered write, not cross-table atomicity.** Extend
  `src/sema/compile/` to emit person **then** condition_occurrence, idempotent per
  `(source_schema, source_table)` via the existing temp-build + scoped-swap. Databricks has **no
  multi-table transaction** (S1-08 is a live Databricks run), so state the guarantee as an
  **ordering** one — person is swapped in before condition references it — not "atomic." Each
  table's swap stays individually atomic; the cross-table guarantee is write-order + FK-valid at
  rest, asserted after both swaps.
- Result: a **valid OMOP-shape** `condition_occurrence` staging table, no synthetic identity,
  no cross-source claims yet. This alone satisfies "materialize tables in OMOP shape."

### Stage B — cross-source person dedup (the "full" in full identity), **gated → NOW IN SCOPE (2nd MSK study incoming)**
**S1-09 measured in two passes, 2026-07-22:**
- *Pass 1 (existing corpus):* GBM-TCGA (585 `TCGA-*`) vs MSK-CHORD (24,950 `P-*`) → **0** overlap,
  incompatible namespaces; plus a legacy `cbioportal` schema that is a 100% GBM duplicate (drop it).
  On that 2-study corpus, Stage B had nothing to collapse.
- *Pass 2 (decisive — the incoming study):* probed `msk_impact_50k_2026`'s clinical file directly
  from the cBio datahub. **48,179 `P-*` patients; `msk_impact ∩ msk_chord = 19,567 shared patients`
  = 78.4% of msk_chord, 40.6% of msk_impact.** Cross-namespace sanity: `msk_impact ∩ gbm = 0`.

**Conclusion (reversed by Pass 2): once `msk_impact_50k_2026` is ingested, deterministic
same-namespace collapse is REQUIRED, not optional.** ~19.5k patients are literally the same person
across the two MSK studies on the identical MSK-DMP key; without collapse they would receive two
different `person_id`s — a 78% duplication of the msk_chord person set. This is the safe,
high-precision, no-probability case (Stage B first bullet), and it now has real work. Probabilistic
dedup remains unjustified (still no cross-namespace signal); **Stage B for this corpus = the
deterministic shared-key collapse only.**

Two consequences (the collapse mechanism, now backed by a real 19,567-patient overlap):
- **Deterministic shared-key collapse — the primary (likely only) warranted mechanism — is
  namespace-scoped.** Collapse two `(study, PATIENT_ID)` to one `person_id` ONLY when they share a
  real institutional key *within the same ID namespace* (e.g. the same MSK-DMP `P-000…` ID recurring
  across multiple MSK studies — a patient sequenced in msk_impact and again in msk_chord). Safe,
  high-precision, no guess. **NEVER collapse across namespaces** (`TCGA-*` vs `P-*`): a match there
  is definitionally impossible and any "match" is over-collapse. This case fires **0 times** on the
  two local studies; it fires only if Databricks holds ≥2 same-institution studies (pending — see
  S1-09 below, blocked on a live Databricks token).
- **Probabilistic/embedding-judge dedup looks UNJUSTIFIED for de-identified cBio and is deferred
  past this branch.** Across namespaces there is no real signal to judge on — no shared key, no
  dates, no names (all de-identified) — so a probabilistic merge of `TCGA-06-2566` ≡ `P-0000036`
  would be pure fabrication, exactly the over-collapse trap. If ever built, it stays the deferred
  Producer-#2 shape (candidate-gen → gated judge → default "distinct", review-surfaced, never
  force-merge) and only within a namespace where deterministic keys are *absent but plausible* —
  a situation the data has not yet shown to exist.
- Ties into `valentine_unification_decision` (Goal A). The measured result pushes that work toward
  **deterministic same-namespace collapse**, not embedding/judge dedup, for the cBio showcase.

## Execution order (TDD, one story per iteration — slice-0 US-style)

0. **S1-00 — Data-reality probe (cheap, blocks the manifest shape).** Before building anything,
   measure `msk_chord_2024` in `~/.sema/poc.duckdb`: (a) does a usable **absolute**
   `condition_start_date` source column exist, or only relative day-offsets (and if so, is there a
   per-patient anchor/index date to derive from)? (b) what is the null/blank rate of the source
   patient key on condition rows? (c) **is there a natural, immutable source-row key** for the
   condition rows (S1-05 depends on it), or must we define a fingerprint + duplicate policy? (d)
   confirm the patient-key **namespace**: cBio schemas are per-study (`<prefix>_<study_id>`), so
   `source_schema` ≈ study — verify PATIENT_ID is unique within the schema and not shared across
   studies in a way that would over-collapse. Feeds D4, S1-02, S1-05. If (a) or (c) fails, do not
   build S1-04/S1-05 on an assumed column — lock D4 / define the fingerprint first.

   **S1-00 RESULTS — RUN 2026-07-22 against `~/.sema/poc.duckdb`, study `cbioportal_msk_chord_2024`
   (condition source = `sample`, 25,040 rows):**
   - **(a) date — NO absolute date exists anywhere.** `sample` has no date column at all. Every
     timeline table (incl. `timeline_diagnosis`) stores `START_DATE`/`STOP_DATE` as VARCHAR
     **relative day-offsets** (diagnosis range −17,285…2,703; negatives = days before an
     undisclosed reference), 100% integer-castable, 0% date-castable. No birth/index/anchor date
     column is present (only `CURRENT_AGE_DEID`, `OS_MONTHS` — durations, not dates). **→ D4 option
     (a) is INFEASIBLE (no anchor to derive from); D4 resolves to (b).**
   - **(b) patient key — 0 blank / 25,040 rows**, 24,950 distinct patients. **→ MISSING_PERSON_KEY
     fires 0 times on this study**, so S1-08's plain source=written equality holds here; the D5
     machinery is still required generically.
   - **(c) natural row key — `sample.SAMPLE_ID` is a clean PK: 25,040/25,040 distinct, 0 blank.**
     **→ H2 resolved by data; `condition_occurrence_id` derives from `SAMPLE_ID`.** ⚠️ But the
     existing `sema_staging.condition_staging.source_row_ref` and `.source_patient_key` columns are
     **100% NULL** — slice-0 declared them and never populated them. Slice-1 must actually thread
     `SAMPLE_ID`→row_ref and `PATIENT_ID`→patient_key through the compile path; they are not free.
   - **(d) namespace — `PATIENT_ID` = MSK `P-00000NN` form**, 24,950 distinct, 1–2 samples/patient
     (avg 1.004), `patient` table PK clean (24,950 distinct, 0 blank). Multiple conditions per
     person confirmed (25,040 conditions → 24,950 persons). Only one study loaded → cross-study
     sharing (S1-09) cannot be measured here yet.
   - **FK closure — 0 gap:** every `sample.PATIENT_ID` exists in `patient`. **→ `omop.person`
     materializes directly from `patient` with full FK closure.**
   - **NO_MAP — 7 rows carry `condition_concept_id = NULL`** (crosswalk gaps), which violates the
     manifest's non-null `condition_concept_id`. **→ new decision D8.**
1. **S1-01 — Generic identity registry** (`resolve/identity_registry.py`, NOT `person_store.py` —
   D6/R29; §-frozen columns, drift test). Mirror `value_mapping_store.py`; DuckDB canonical;
   single-writer with atomic get-or-create on the transactional unique key (D7). **Two-level schema**
   (`(source_namespace, patient_key)`→`source_patient_uid`→`person_id`) so Stage B remaps level-2.
   `source_namespace` = the per-study source schema (confirmed in S1-00). *No resolver yet.*
2. **S1-02 — Deterministic identity resolver** — `(source_namespace, patient_key) → source_patient_uid
   → person_id` (registry-assigned, NOT a hash); writes the registry; missing/blank patient keys
   route to the NO_MAP/review artifact (defined outcome, never a synthetic person, never a silent
   drop). The `person`/`MISSING_PERSON_KEY` binding is supplied by the policy layer, not the resolver.
3. **S1-03 — Person obligation + assertions** — satisfy the manifest `omop.person` obligation.
   `Slice0PlanAssembler` is already obligation-agnostic (it groups by target property and folds any
   `TargetObligation`), so this story is *feeding* it the person/date/PK assertions + the expanded
   obligation, NOT rewriting the assembler. Reuse `compile_projection` (US-014) for the person table.
4. **S1-04 — Event-date field (D4 RESOLVED → contract change, not a copy).** S1-00 proved no
   absolute date exists, so this story is the manifest `0.2.0` revision (`condition_start_date`
   → `nullable:true`, dropped from `required_fields`) + optionally staging the raw relative
   offset in a provenance column. There is no source column to `DIRECT_COPY`.
5. **S1-05 — Surrogate PK** — deterministic `condition_occurrence_id` derived from **stable
   source-row identity** (`source_schema, source_table, SAMPLE_ID` — S1-00 confirmed `SAMPLE_ID`
   is a clean 25,040/25,040 key), NEVER from the resolved `person_id` (which Stage B may remap).
   Also thread `PATIENT_ID`→`source_patient_key` and `SAMPLE_ID`→`source_row_ref` into staging;
   S1-00 found both columns exist but are currently 100% NULL. Idempotent across re-runs and dedup.
6. **S1-06 — Multi-table FK-closed compiler** — person→condition **ordered** write (person swapped
   in first; no claim of cross-table atomicity on Databricks); idempotent; assert FK validity
   (no orphan `person_id`) at rest. **Swap scope is per-study (per source schema)** to match
   slice-0's scoped-swap grain — the person snapshot for a study must contain every person its
   condition rows reference, so a multi-study warehouse never orphans another study's conditions.
   Integration tests cover both the happy path **and** a mid-sequence failure (person swapped,
   condition not) to prove no invalid cross-table state survives a retry.
7. **S1-07 — Gate-D-lite extension** — FK-closure + required-field null-rate checks for the
   full obligation (not just the concept column); include the missing-key disposition count.
8. **S1-08 — LIVE Databricks run** — full OMOP-shape write for a real study (`msk_chord_2024`);
   **row-count identity accounts for the missing-key disposition:** `written_condition_rows +
   MISSING_PERSON_KEY_review_rows = source_condition_rows` (plain equality with source only holds
   when the S1-00 blank-key rate is zero); FK-valid at rest; idempotent re-run. Pre-live gates from
   S1-00 (source-row key proven, namespace confirmed) and S1-01 (registry deployment + single-writer
   locking) must be green before this runs.
9. **S1-09 — Cross-source shared-key feasibility probe** (Stage B gate) — **DONE 2026-07-22.**
   Live Databricks enum (2 distinct studies, 0 overlap) THEN a datahub probe of the incoming
   `msk_impact_50k_2026` clinical file: **48,179 `P-*` patients, 19,567 shared with msk_chord (78.4%
   of chord)**, cross-namespace overlap 0. **Verdict: with msk_impact ingested, Stage B deterministic
   same-namespace collapse is IN scope and REQUIRED.** Side-finding: the legacy `cbioportal` schema
   duplicates GBM and should be dropped (hygiene), NOT run through person dedup.
10. **S1-10+ — Deterministic same-namespace collapse** — **IN SCOPE** once `msk_impact_50k_2026`
    is ingested (S1-09 measured 19,567 shared MSK patients). Build the deterministic collapse
    (remap the uid→person_id registry level for identical `P-*` keys across MSK studies); the
    probabilistic judge stays deferred (no cross-namespace signal). Spec: deterministic collapse
    (remap the uid→person_id registry level), then gated judge; over-collapse guardrails; review
    surfacing. **Materialization semantics (from the H1 correction):** a registry remap does not
    edit persisted rows, so dedup **rebuilds `condition_occurrence`** for the affected source scope
    through the current registry (preserving each `condition_occurrence_id`, recomputing its
    `person_id`) and **replaces `omop.person`** with surviving canonical persons, retiring orphaned
    person rows. FK-validity re-asserted after rebuild. **Precondition:** a dedup gold/adjudication
    set exists before this ships — Stage B precision needs labels the way Producer #2 does; Stage A
    identity does not (self-consistent, like slice-0).

## Open decisions to lock before executing

- **D1 — surrogate `person_id` scheme (RESOLVED — registry, not hash):** a hash of
  `(source_schema, patient_key)` is 1:1 by construction, which makes it **incompatible with Stage B
  dedup** — collapsing two source patients would require changing a `person_id`, breaking
  idempotency and orphaning already-written FKs. Use the two-level registry instead: a stable
  `source_patient_uid` per source patient (may be a hash of the source key — fine, it never
  collapses), mapped to a **registry-assigned** canonical `person_id`. Idempotency comes from the
  registry being a durable single-writer store (re-run reads the same assignment), not from
  statelessness. Stage B revises identity by remapping the uid→person_id level only.
- **D2 — is `omop.person` a full second target manifest entity flow (like US-014's second
  manifest) or an extension of the current one?** The manifest already declares `omop.person`, so
  lean **extension**, not a new manifest.
- **D3 — Stage B scope (REVERSED by the msk_impact probe → deterministic collapse IN):** the
  incoming `msk_impact_50k_2026` shares 19,567 patients with msk_chord on the identical MSK-DMP key,
  so **Stage B deterministic same-namespace collapse is now required** (Stage A alone would create
  ~19.5k duplicate persons). Build order unchanged — Stage A first (correct single-study OMOP shape),
  then S1-10 deterministic collapse once both MSK studies are in the corpus. Probabilistic dedup
  stays out (no cross-namespace signal). Legacy `cbioportal` schema (100% GBM dup) → **drop**, not dedup.
- **D4 — `condition_start_date` (RESOLVED by S1-00 → option (b)):** the probe confirmed cBio
  msk_chord carries **no absolute date and no anchor** to derive one from (all timeline dates are
  relative day-offsets against an undisclosed reference). Option (a) is therefore infeasible.
  **Decision: version the target contract** — manifest `0.1.0`→`0.2.0` declaring
  `condition_start_date nullable:true`, drop it from `required_fields`, and let Gate-D-lite report
  the 100% null rate honestly. This is an explicit contract change, NOT a silent runtime relaxation;
  the code must match whatever the manifest says. *Optional follow-up:* stage the raw relative
  offset in a separate `condition_start_offset_days` provenance column so the signal isn't lost —
  but never cast it to a fabricated absolute `condition_start_date`. (Rejected (c) out-of-scope:
  the OMOP shape is still valuable without a date; dropping the whole study is disproportionate.)
- **D5 — disposition of condition rows with a missing/blank patient key:** route to the slice-0
  NO_MAP/review artifact with a typed reason (`MISSING_PERSON_KEY`); count surfaced in Gate-D-lite
  and in the S1-08 row-count identity. Never synthesize a person, never silently drop. Confirm the
  artifact shape matches slice-0's. *(S1-00 measured 0 such rows in msk_chord, so this fires only
  on other studies — build the path, expect it idle here.)*
- **D6 — keep OMOP out of the engine core (R29):** identity is generic; only the *target* is OMOP.
  The person store must be a **generic entity-identity registry** keyed by an entity-identity spec
  from the manifest/plan, not a hardcoded `person`; the write order (`person` before
  `condition_occurrence`), the FK-closure targets, the required-field set, and the
  `MISSING_PERSON_KEY` disposition all come from **manifest/plan metadata**, not literals baked into
  `resolve/` or `compile/`. `resolve/person_store.py` naming is a smell — prefer
  `resolve/identity_registry.py` (generic) with the OMOP `person` binding supplied by the
  allowlisted policy layer, same boundary slice-0 held. The R29 CI grep must stay green.
- **D7 — registry allocation protocol (single-writer, replay-safe):** `source_patient_uid` and the
  canonical `person_id` are assigned by an atomic **get-or-create** on a transactional unique key
  `(source_namespace, source_patient_key)` in the DuckDB canonical store — lookup-then-insert races
  and run-ordering must not change an existing assignment. Define which durable registry instance
  every writer uses and how writers are serialized before S1-08's live run.
- **D8 — NO_MAP condition rows vs the non-null `condition_concept_id` (surfaced by S1-00):** the
  probe found 7 msk_chord rows staged `NO_MAP` with `condition_concept_id = NULL`, which cannot
  satisfy the manifest's required non-null concept. OMOP's own convention is
  `condition_concept_id = 0` ("No matching concept") for unmapped rows — a real, standard sentinel,
  not a fabrication. **Recommend: map NO_MAP → concept_id `0`** (writes a valid, FK-clean,
  honestly-unmapped row) and surface the NO_MAP count in Gate-D-lite, rather than dropping the 7
  rows or nulling a required field. Confirm the resolver's `allow_zero_default` already models this;
  if so D8 is just wiring it into the concept field for NO_MAP.

## Out of scope (later slices / drift if built here)

- Full OHDSI DQD / Gate D (slice-0 kept Gate-D-lite; extend, don't replace).
- Visit/provider/observation-period OMOP tables — condition_occurrence + person only.
- LLM council/jury for dedup (single-model gated judge is enough to start).
- Rewiring the source `sema build` Neo4j graph — this plan is store/compile/staging + identity.
