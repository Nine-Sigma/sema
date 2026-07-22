# Handoff — Slice-1 OMOP shape + identity (session 2026-07-22)

> Read this with `tasks/plan-slice1-omop-shape-identity.md` (the plan) open. This handoff records
> what the 2026-07-22 session did to that plan, the live data probes that resolved its open
> decisions, and the msk_impact ingestion. Durable auto-memories: `cbio_no_absolute_dates`,
> `cbio_cross_study_patient_identity`, `cbio_lfs_pointer_ingest_bug`. Bug: `.wolf/buglog.json` bug-401.

## What we're building (goal)

Slice-1 takes the resolved cBio→OMOP concept mapping from Slice-0 (done: `ONCOTREE_CODE →
condition_concept_id`, one narrow staging table) and materializes a **production-shaped OMOP
`condition_occurrence` staging table**, FK-closed against a materialized `omop.person`, with **full
identity resolution** (deterministic person map + cross-source person dedup). The manifest
(`src/sema/targets/manifests/omop_condition_slice0.yaml`) already declares the full target; what's
missing is the machinery. Non-negotiable principle: identity must be a real mapping, never a
fabricated `person_id`; "the half-measure placeholder is the most dangerous state."

## What this session did

1. **Reviewed the plan (2 rounds).** My review found 6 issues; then a **Codex adversarial review**
   found 4 High + 5 Medium. All folded into the plan. The load-bearing corrections:
   - `person_id` must be a **two-level registry** (`(source_namespace, patient_key)` →
     `source_patient_uid` → registry-assigned `person_id`), NOT a hash — a hash PK is 1:1 and can't
     be collapsed by dedup without breaking FKs/idempotency (D1).
   - Dedup does **not** rewrite persisted FKs by remapping the registry; it **rebuilds
     `condition_occurrence`** from source through the current registry (row PK `condition_occurrence_id`
     stable, `person_id` FK recomputed). (Codex H1 — a real bug in my first fix.)
   - `condition_occurrence_id` derives from **source-row identity** (`SAMPLE_ID`), never `person_id`.
   - Keep OMOP out of the engine core (R29): generic `identity_registry.py`, `person`/write-order/
     required-fields come from manifest metadata (D6).
   - Multi-table write is **ordered, not atomic** on Databricks; missing patient keys route to
     NO_MAP/review, never a synthetic person (D5).

2. **Ran the S1-00 data-reality probe** (live, `~/.sema/poc.duckdb`, msk_chord):
   - **No absolute date exists** — `sample` has none; all timeline dates are relative day-offsets
     (−17,285…2,703), no anchor. → **D4 = version the manifest to `condition_start_date nullable:true`**
     (`0.1.0`→`0.2.0`); never fabricate a date. (Memory: `cbio_no_absolute_dates`.)
   - `SAMPLE_ID` is a clean row key (25,040/25,040). `patient` FK-closed (0 gap). 7 NO_MAP rows carry
     null concept_id → **D8: map NO_MAP → OMOP concept_id 0** ("No matching concept"), not null/drop.
   - ⚠️ The existing `sema_staging.condition_staging.source_row_ref` / `source_patient_key` columns
     are **100% NULL** — Slice-0 declared but never populated them; Slice-1 must thread them. (Note:
     that local staging table is also **stale** — still shows the pre-repoint `omop.oncotree_to_snomed_condition`
     policy ref, not the current `omop.oncotree_condition`.)

3. **Ran the S1-09 cross-study identity probe** (the Stage B gate):
   - Existing corpus: GBM-TCGA (585 `TCGA-*`) vs MSK-CHORD (24,950 `P-*`) → **0** overlap
     (incompatible namespaces). Legacy `workspace.cbioportal` schema = 100% GBM duplicate → drop it.
   - **Then ingested `msk_impact_50k_2026`** (a 2nd MSK study): **48,179 `P-*` patients; `msk_impact
     ∩ msk_chord = 19,567` shared patients (78.4% of msk_chord)** on the identical MSK-DMP key;
     cross-namespace `∩ gbm = 0`. → **Stage B deterministic same-namespace collapse is now REQUIRED**
     (Stage A alone would mint ~19.5k duplicate persons). Probabilistic dedup stays out (no
     cross-namespace signal). (Memory: `cbio_cross_study_patient_identity`.)

4. **Ingested `msk_impact_50k_2026` into DuckDB + Databricks** (user chose full study, both stores).
   - `sema ingest cbioportal msk_impact_50k_2026` → DuckDB. **GitHub LFS bandwidth is exhausted** on
     cBioPortal/datahub: `data_clinical_sample.txt` and `data_sv.txt` fetched as LFS **pointers**
     (media CDN 404; raw/api/jsDelivr/Statically all return the pointer; S3 & LFS batch 403).
     `patient`+genomics resolved via the media CDN.
   - **Fixed the ingestion bug (bug-401):** `cbioportal_fetch_utils.py::fetch_lfs_or_raw` silently
     accepted the 133-byte pointer as data → a 2-row garbage `sample` table, exit 0. Now detects the
     LFS-pointer magic, tries authenticated `api.github.com` raw (`GITHUB_TOKEN`/`GH_TOKEN`), and
     **raises** instead of ingesting a pointer. Dropped the garbage tables, cleared the cache marker.
   - **Recovered `sample` from the cBioPortal portal clinical-export TSV** (user downloaded it):
     54,331 rows, 48,179 patients, 502 ONCOTREE codes, 0 blank, **0 FK gap** to `patient`. Loaded
     into DuckDB as `cbioportal_msk_impact_50k_2026.sample` with datahub UPPER_SNAKE column names.
     (Memory: `cbio_lfs_pointer_ingest_bug`.)

## Current corpus state

**DuckDB `~/.sema/poc.duckdb`:**
| schema | key tables (rows) |
|---|---|
| `cbioportal_gbm_tcga_pan_can_atlas_2018` | patient 585, sample 585 |
| `cbioportal_msk_chord_2024` | patient 24,950, sample 25,040 (+ timelines, genomics) |
| `cbioportal_msk_impact_50k_2026` | patient 48,179, **sample 54,331** (portal export), cna 29.4M, cna_segmented 3.0M, mutation 479k, gene_panel_matrix 163k. **`structural_variant` absent** (LFS-blocked, not needed) |
| `sema_resolve.value_mapping` | 63 rows (msk_chord slice-0 decisions) |
| `sema_staging.condition_staging` | 25,040 (STALE — pre-repoint policy ref; row_ref/patient_key NULL) |
| `ontology_omop.*` | empty CDM skeleton; `vocabulary_omop.*` full (concept 10M, etc.) |

**Databricks `workspace`:** `cbioportal` (legacy GBM dup — DROP), `cbioportal_gbm_tcga_pan_can_atlas_2018`,
`cbioportal_msk_chord_2024`, and `cbioportal_msk_impact_50k_2026` (**pushed — all 6 tables verified, see below**).

## Databricks push status (msk_impact_50k_2026)

**COMPLETE** (finished 2026-07-22 12:08, exit 0). Command: `sema push --schemas cbioportal_msk_impact_50k_2026`.
All six tables landed with row counts matching target exactly (the push verifies pushed == target):

| table | rows | via |
|---|---|---|
| cna | 29,393,071 | copy_into |
| cna_segmented | 3,038,461 | copy_into |
| gene_panel_matrix | 162,993 | copy_into |
| mutation | 479,147 | copy_into |
| patient | 48,179 | insert |
| sample | 54,331 | insert |

`cbioportal_msk_impact_50k_2026` is now fully materialized in both DuckDB and Databricks `workspace`.

## Stage A status (2026-07-22 session 2) — S1-01…S1-07 DONE + real-data gate PASSED

Branch `ralph/feat/omop-shape-identity` (from slice-0 HEAD; slice-0 not yet on main). All TDD,
committed per story, full unit suite green (2216 passed), R29 + mypy clean, new-module coverage 95%.

| story | delivered | file(s) |
|---|---|---|
| S1-01 | generic 2-level identity registry (get-or-create, D1/D7) | `resolve/identity_registry{,_utils}.py` |
| S1-02 | deterministic identity resolver; blank key→review (D5) | `resolve/identity_resolver.py`, `policies/omop.py` |
| S1-03 | `omop.person` via existing assembler + `compile_projection` | `policies/omop.py` (person builders) |
| S1-04 | manifest 0.1.0→0.2.0, `condition_start_date` nullable (D4) | `targets/manifests/omop_condition_slice0.yaml` |
| S1-05 | source-row surrogate PK (md5 content hash, DuckDB/Spark-portable, person_id-independent) | `compile/row_surrogate.py` |
| S1-06 | FK-closed compiler: ordered person→condition write, registry FK join, NO_MAP→0 (D8), NULL date, per-study scope, FK assert, mid-failure recovery | `compile/fk_closed_compiler{,_utils}.py` |
| S1-07 | Gate-D-lite extension: FK-closure + required-not-null + missing-key accounting | `eval/staging_qa{,_utils}.py` |

**Real-data pre-live gate (the whole chain end-to-end on `~/.sema/poc.duckdb`, msk_chord):**
25,040 sample rows → identity resolve → 24,950 persons (0 review) → FK-closed materialize into
`omop_stage_a.person` (24,950) + `omop_stage_a.condition_occurrence` (25,040). Row-count identity
`25,040 + 0 = 25,040` ✓; FK closure 0 orphans ✓; surrogate PK 25,040/25,040 distinct ✓; **7 rows →
concept_id 0** (D8; exactly the 7 NO_MAP rows S1-00 found) ✓; Gate-D-lite PASS on all three checks.
The identity registry now lives in `poc.duckdb` `sema_identity.entity_identity` (DuckDB-canonical, correct).

## S1-08 status (2026-07-22 session 3) — LIVE Databricks run COMPLETE ✅

Stage A is fully done (S1-01…S1-08). The FK-closed OMOP shape now materializes live in Databricks.

**What shipped:** (a) `FkBackend` strategy (`src/sema/compile/fk_backend.py` + `fk_backend_utils.py`),
mirroring `StagingBackend` — DuckDB temp-swap vs Databricks atomic Delta `INSERT … REPLACE WHERE` (no
`BEGIN/COMMIT`, no client temp table); `FkClosedCompiler(backend=…)` and `run_fk_closed_qa(backend=…)`
are now dialect-agnostic. (b) Registry→Databricks bridge (`src/sema/resolve/identity_bridge.py`): mirrors
the DuckDB-canonical registry into a Delta table (`CREATE OR REPLACE TABLE` + batched `INSERT VALUES`)
before the write, so the parent `SELECT DISTINCT entity_id` and child FK join can read it in-warehouse.
(c) Live-run harness `src/sema/pipeline/fit_omop_shape.py` + CLI `sema fit-omop-shape`
(`src/sema/cli_fit_omop.py`, `cli_fit_omop_utils.py`); OMOP physical specs via `make_omop_fk_specs()` in
the R29-allowlisted policy.

**Live result** (`sema fit-omop-shape --backend databricks --study-schema cbioportal_msk_chord_2024
--omop-schema omop_stage_a --strict`, exit 0): `workspace.omop_stage_a.person` = 24,950 rows (24,950
distinct); `condition_occurrence` = 25,040 rows, 25,040/25,040 distinct surrogate PKs; **exactly 7 rows →
concept_id 0** (the 1 NO_MAP OncoTree code × 7 source rows), 57 distinct real concepts; all
`condition_start_date` NULL (D4); 0 FK orphans; Gate-D-lite PASS (fk_closure, required_not_null,
missing_key_disposition `25040 + 0 = 25040`). Bridged 24,950 registry rows. **Re-ran twice → identical
counts** (idempotent scoped `REPLACE WHERE`). Matches the DuckDB pre-live gate exactly.

**Gotchas confirmed this run:** the SQL warehouse cold-starts — first connect threw `RequestError`;
retry with ~20s backoff connected clean. Registry namespace MUST equal the source schema
(`cbioportal_msk_chord_2024`) so the live resolve reuses the gate-populated entity_ids and mints nothing.

## Open items / next steps (ordered)

1. ~~Verify the msk_impact Databricks push~~ — **DONE** (2026-07-22 12:08, all 6 tables verified).
2. ~~**S1-08 — LIVE Databricks run**~~ — **DONE** (2026-07-22 session 3; see the S1-08 status section above).
3. **Execute Stage B (S1-10) — now in scope:** deterministic same-namespace person collapse across
   msk_chord + msk_impact (~19.5k shared `P-*` patients). Remap the uid→person_id registry level;
   rebuild `condition_occurrence`. Probabilistic dedup stays deferred.
4. **Data hygiene:** drop the legacy `workspace.cbioportal` schema (100% GBM duplicate).
5. **Later:** recover msk_impact `data_sv.txt` when GitHub LFS frees up (not needed for Slice-1).

## Gotchas for next session

- **Stale Databricks token:** the shell env holds an expired `DATABRICKS_TOKEN` (`dapi3c21…`); `.env`
  has the fresh one (`dapid1fc…`). The CLI's `load_dotenv()` does NOT override an existing env var, so
  force it: `export DATABRICKS_TOKEN="$(grep '^DATABRICKS_TOKEN=' .env | cut -d= -f2-)"` before any
  `sema push`/Databricks call, or `unset DATABRICKS_TOKEN` so `.env` wins.
- **cBio LFS block:** new/large datahub studies fail to fetch `sample`/`sv` (pointers). Use the
  cBioPortal **portal clinical-export TSV** and load as `sample` with UPPER_SNAKE headers. Details in
  `cbio_lfs_pointer_ingest_bug` memory / bug-401.
- **Ingest needs `PYTHONPATH=<repo-root>`** so `showcase/` is importable.
- **Never `sema build --table-pattern <t>` without `--resume`** (wipes the study's graph; bug-343).
- The local `sema_staging.condition_staging` is stale (pre-repoint) — don't trust its rows; a fresh
  `sema fit` regenerates it.

## Artifacts

- Plan: `tasks/plan-slice1-omop-shape-identity.md` (all decisions D1–D8 locked/measured).
- Memories: `cbio_no_absolute_dates`, `cbio_cross_study_patient_identity`, `cbio_lfs_pointer_ingest_bug`.
- Bug: `.wolf/buglog.json` bug-401. Code change: `showcase/cbioportal_to_omop/cbioportal_fetch_utils.py`.
