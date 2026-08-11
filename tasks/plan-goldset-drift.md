# Plan — gold-set drift: restore the OncoTree mapping oracle

Status: proposed, 2026-08-11. Blocks a green integration suite; D4 blocks real eval power.

## What's actually broken

Two failing tests in `tests/integration/test_mapping_goldset_coverage.py` look like one problem
("the gold set drifted") but are two, and the more serious one is invisible from the failure message.

**P1 — the oracle has never been labelled.** All **64/64** rows in
`tests/data/gold/oncotree_condition_slice0.jsonl` carry `gold_label: "UNLABELLED"` and
`gold_concept_id: null`. `score()` excludes `UNLABELLED` rows by design, so the harness would today
score **zero codes**. The scoring code (`src/sema/eval/mapping_goldset.py`) is sound and unit-tested;
the artifact it grades against is an empty worksheet. US-012 cannot grade anything until this is
fixed, and no amount of re-scaffolding fixes it — it needs an external oracle (D4).

**P2 — a frozen artifact is asserted against live mutable data.** Both tests read
`~/.sema/poc.duckdb` at run time. `discover_oncotree_schemas()` auto-discovers *every*
`cbioportal_*` schema with a `sample.ONCOTREE_CODE` column, so ingesting a study silently changes
the denominator and reds the suite. That is what happened: the gold set was scaffolded when
`gbm_tcga_pan_can_atlas_2018` + `msk_chord_2024` were loaded (every row's `notes` says so), and
`msk_impact_50k_2026` landed afterwards.

### Measured, 2026-08-11

| | |
|---|---|
| distinct observed codes | **510** |
| total sample rows | **79,963** |
| gold-set rows | **64** (12.5% of codes) |
| codes missing from gold | **446** |
| gold rows with stale `row_count` | **56/64** (LUAD 5,957 → 12,211) |
| labelled gold rows | **0** |

The 64 scaffolded codes are not arbitrary — they are the head of the frequency distribution and
cover **86.9%** of all rows. Row-share by tier:

| top N codes | rows | share |
|---|---|---|
| 10 | 48,259 | 60.4% |
| 25 | 59,978 | 75.0% |
| 50 | 67,111 | 83.9% |
| 64 | 69,507 | 86.9% |
| 100 | 73,485 | 91.9% |
| 150 | 76,550 | 95.7% |
| 300 | 79,257 | 99.1% |

This is the load-bearing fact for the whole plan: a bounded labelling budget buys most of the
row-weighted signal. Demanding 100% code coverage (what the test asserts today) costs 510 hand
labels to move row-weighted coverage from 95.7% to 100%.

### Secondary finding — enumeration and materialization disagree on "loaded"

`enumerate_distinct_codes()` sees **three** studies (it reads `cbioportal_*.sample` tables), but
`sema_staging.condition_staging` holds only **two** (`msk_impact`, `msk_chord`; 79,371 rows).
`gbm_tcga_pan_can_atlas_2018` is present as raw sample data but was never staged/resolved, so the
gold set is being scored against a code universe wider than anything the pipeline actually maps.
Whatever D3 decides, these two notions of scope must be reconciled, not left implicit.

## Goal

The integration suite is green and stays green across future ingests, **and** the gold set has real
evaluative power: a bounded, explicitly-scoped, human-gated set of labels that `score()` can grade a
resolver against without grading its own homework.

Note these are separable. G-01…G-04 + G-07 deliver the green suite with **zero** labelling work.
G-05/G-06 deliver the oracle. Do not let the second block the first.

## Principles

- **Never let Sema label its own gold set.** The docstring's rule stands: `gold_concept_id` comes
  from an external oracle, never from US-006's resolver. A green suite achieved by autogenerating
  labels is worse than a red one.
- **The artifact is frozen; the data is not.** A gold set is a snapshot with provenance. Tests
  compare against the snapshot's declared scope, not against whatever happens to be in DuckDB.
- **Coverage is a budget, not an absolute.** State the target as row-weighted share with an explicit
  tail policy, so the contract survives the long tail of rare codes.
- **Unlabelled is a first-class state**, already modelled (`GoldSet.unlabelled_codes()`). Uncovered
  codes should be *accounted for*, not silently absent.

## Open decisions

- **D1 — Coverage contract. RECOMMEND: frequency-tiered.** Replace "every observed code" with
  "every code in tier T, where T is the smallest prefix reaching ≥95% row-weighted share" (today:
  ~150 codes). Codes outside T are recorded as `OUT_OF_TIER`, not missing. Alternative (100%
  coverage) is defensible only if labelling is cheap — it is not, at 510 codes.
- **D2 — `row_count` semantics. RECOMMEND: frozen snapshot + explicit refresh.** `row_count` is a
  scoring *weight*, not truth. Freeze it in the artifact with a snapshot date; provide
  `sema eval refresh-goldset` to re-stamp counts; demote exact-equality to a reported drift
  percentage. Keeping the current live-equality assertion guarantees a red suite after every ingest.
- **D3 — Study scope. RECOMMEND: declared, not discovered.** The artifact carries a header
  (`study_schemas`, `snapshot_date`, `source_of_truth`), and enumeration takes an explicit scope
  argument. Auto-discovery stays available for the refresh command only. Also decide whether scope
  is `cbioportal_*.sample` (raw) or `sema_staging.condition_staging` (staged) — the two disagree by
  one whole study. **Recommend staging**, since it is what the pipeline actually maps.
- **D4 — Labelling source. OPEN — needs your call; blocks G-05.** Candidates:
  - **(a) OncoTree's own published crosswalk.** `tests/data/gold/oncotree_reference_64.csv`
    already carries `ncit` and `umls` per code. OncoTree→NCIt/UMLS→OMOP concept is external to Sema.
    **Risk:** if US-006's resolver reaches OMOP through the same relationship rows, this is
    self-grading with extra steps. Must be verified against the resolver's actual path before use.
  - **(b) Human curation** against the reference CSV. Slowest, unimpeachable, and the original
    intent ("awaiting human label").
  - **(c) A third-party published OncoTree→SNOMED/OMOP mapping**, if one exists with a usable
    licence.
  My recommendation: verify (a)'s independence first; if the paths overlap, fall back to (b) for
  tier-1 (top 25 = 75% of rows) and leave the rest `UNLABELLED` rather than fabricate.
- **D5 — Artifact location.** `tests/data/gold/` implies test fixture; a curated oracle with a human
  gate is closer to a project asset. Low stakes — decide when touching the file.

## Work breakdown (TDD — failing test first)

**G-01 — Declare the scope (D3).** Add a header record (or sidecar `*.meta.json`) with
`study_schemas`, `snapshot_date`, `source_of_truth`. Give `enumerate_distinct_codes()` an explicit
`study_schemas` parameter; keep discovery for refresh only. Test: enumeration over a declared scope
ignores an unlisted study present in the DB.

**G-02 — Split the coverage contract (D1).** `test_gold_set_covers_every_observed_code` becomes
"every code in the declared tier is present in the gold set, and row-weighted coverage ≥ target".
Add `GoldSet.out_of_tier_codes()` alongside the existing unlabelled accounting. Test: a code below
the tier threshold does not fail coverage; a missing in-tier code does.

**G-03 — Decouple `row_count` (D2).** Delete the live-equality assertion; add
`goldset_drift_report()` returning per-code delta + aggregate drift %. Add
`sema eval refresh-goldset` to re-stamp counts and scope while preserving labels and notes. Test:
refresh updates `row_count` and never mutates `gold_concept_id`/`gold_label`.

**G-04 — Re-scaffold to current data.** Regenerate the artifact for the declared scope: preserve
all existing rows and labels, re-stamp counts, add the in-tier codes now missing, mark them
`UNLABELLED`, and fix the stale `notes` (they still claim `gbm_tcga…, msk_chord_2024`). After this,
G-07's tests pass with zero labelling. Test: re-scaffold is idempotent and label-preserving.

**G-05 — Label tier 1 (D4-blocked; human gate).** Populate `gold_concept_id`/`gold_label` for the
chosen tier from the chosen oracle. Deliverable I can prepare without D4: a labelling worksheet
(reference CSV regenerated for the current tier with OncoTree metadata + candidate OMOP concepts
from `vocabulary_omop`, clearly marked as *candidates for human review*, never written to
`gold_concept_id`).

**G-06 — Wire `score()` into a runnable eval.** A command that reads resolver decisions from
`sema_resolve.value_mapping`, scores against the gold set, and writes a report to `eval-runs/`.
This is what makes the whole artifact worth maintaining. Test: a fixture resolver + fixture gold set
produce the documented §1.5(f) matrices.

**G-07 — Re-green the integration tests** and confirm no ingest-sensitivity remains: add a test that
introducing an unlisted study schema does not change any assertion outcome.

## Definition of done

- [ ] `tests/integration/test_mapping_goldset_coverage.py` passes against current `poc.duckdb`.
- [ ] Ingesting a new `cbioportal_*` study does not red the suite (proved by G-07's test, not by
      hope).
- [ ] Gold-set artifact declares its scope, snapshot date, and source of truth.
- [ ] `row_count` drift is *reported* with a number, not asserted to zero.
- [ ] Coverage stated as row-weighted share against a declared tier; out-of-tier codes accounted for.
- [ ] Re-scaffold and refresh are idempotent and provably label-preserving.
- [ ] **No `gold_concept_id` is ever written by Sema's resolver** — enforced by a test, not a
      convention.
- [ ] D4 resolved and recorded here; if (a) is chosen, the independence check from the resolver's
      path is documented.
- [ ] mypy strict + unit suite green; coverage ≥ 85%.

## Out of scope

- Labelling the full 510-code tail. Tier policy is the whole point of D1.
- Building or tuning the resolver (US-006). This plan restores the *measuring instrument*.
- The other two pre-existing integration failures
  (`test_different_sources_coexist` — MERGE-key over-collapse; and the e2e `SemanticEngine(llm=...)`
  signature rot). Tracked separately.
- Re-ingesting `gbm_tcga_pan_can_atlas_2018` into staging. If D3 picks staging as the source of
  truth, that study simply leaves the gold-set scope until it is properly staged.
