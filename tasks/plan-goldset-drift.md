# Plan — gold-set drift: restore the OncoTree mapping oracle

## Implementation status — 2026-08-11

**G-01 … G-04, G-06, G-07 are implemented and merged. G-05 (labelling) is open — by
design: it is a human gate, and this work deliberately did not touch a label.**

Decisions taken: **D1(x)** (PRD amended — acceptance coverage is 100% of the frozen tier;
see `tasks/prd-sema-mapping-slice0.md`), **D3(ii)** (staging scope), **D4(b)** (human
curation), **D5** (artifact stays under `tests/data/gold/`, now as versioned snapshots).

What shipped:

- `tests/data/gold/snapshots/2026-08-11-raw2study/` — the prior 64-code artifact, sealed
  byte-identical with the header and manifest that make its real contract explicit. It met
  that contract exactly (64/64 codes, 64/64 row counts); the scope declaration was what
  broke, not coverage.
- `tests/data/gold/snapshots/2026-08-11-staging/` — current. 509-code universe, frozen
  137-code head at **95.02%** of staged rows, 11 challenge-only codes, 32 out-of-tier,
  `GBM` retired. **181 rows, 0 labelled.**
- `sema eval goldset drift | observe | re-tier | re-scope | worksheet | apply-labels`
  and `sema eval mapping-report`.
- `tests/data/gold/worksheets/tier1-2026-08-11-staging.csv` — the 62-code tier-1
  worksheet (top-50 head + 12 challenge), blank, interleaved, no candidate targets.

**One measured correction to this plan.** Out-of-tier is **32**, not 34, and the artifact
is **181 rows**, not 172: `MASC` and `PANEC` are existing gold codes that are *also*
challenge codes, so under G-03's disjoint states they take `CHALLENGE`, not
`OUT_OF_TIER`. States: 137 in-tier + 11 challenge + 32 out-of-tier + 1 retired.

The live figures the plan predicted were otherwise exact: 509 codes / 79,371 rows, tier
137 @ 95.02%, 12 declared-uncertain decisions, `GBM` the sole retirement.

Current honest state: `provisional — not accepted [unadjudicated]`, coverage 0/137. The
instrument is repaired and the needle is at zero, which is what G-05 is for.

---

Status: proposed, 2026-08-11. Revised three times on 2026-08-11 after three adversarial reviews,
each re-measuring every claim against `~/.sema/poc.duckdb` and the source (see the three
"Corrections" sections at the end for what changed and why). Pass 2 reversed the labelling-cost
arithmetic and found a permanent acceptance-gate ceiling; pass 3 split the benchmark into two frozen
populations and withdrew its own sentinel-union proposal. **Read G-02a, D1, and "Deferred" before
costing this work.**
Blocks a green integration suite; D4 blocks real eval power.

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

### Measured, 2026-08-11 (re-verified against `~/.sema/poc.duckdb`)

Raw scope — the union of `cbioportal_*.sample`, which is what the test enumerates today:

| | |
|---|---|
| distinct observed codes | **510** |
| total sample rows | **79,963** |
| gold-set rows | **64** (12.5% of codes) |
| codes missing from gold | **446** |
| gold rows with stale `row_count` | **56/64** (LUAD 5,957 → 12,211) |
| labelled gold rows | **0** |
| **row-share covered by the gold set's codes** | **64.7%** (51,748 / 79,963) |

**What the 64 codes actually are.** Not a frequency head — the *complete* code universe of the two
originally-loaded studies (`gbm_tcga_pan_can_atlas_2018` + `msk_chord_2024`): 64/64 codes, and every
stored `row_count` matches that 2-study scope **exactly** (64/64). The "56/64 stale" figure is stale
only against the 3-study union. So the original contract was "100% of codes in a declared scope",
which is coherent; what broke is the *scope declaration* (absent) and the live-equality assertion —
not coverage absolutism.

Row-share by frequency tier. These are top-N-by-frequency sets, which are **not** the gold set —
only **20** of the gold set's 64 codes appear in today's top 64.

> **Read the column headers literally.** `new rows` is how many tier codes are *absent from the
> scaffold*. It is **not** a labelling cost. Because **0/64 existing rows carry a label**, the
> labelling cost of tier T is **T itself** — every code in the tier needs a human label, including
> the ones already scaffolded. The two columns differ by 2.3× at top-25 and by 27% at the ≥95% tier.

Raw scope — union of `cbioportal_*.sample`, 3 studies, 510 codes / 79,963 rows:

| top N codes | rows | share | already in gold | new rows | **labels needed** |
|---|---|---|---|---|---|
| 10 | 48,259 | 60.4% | 8 | 2 | 10 |
| 25 | 59,978 | 75.0% | 14 | 11 | **25** |
| 50 | 67,111 | 83.9% | 18 | 32 | 50 |
| 64 | 69,507 | 86.9% | 20 | 44 | 64 |
| 100 | 73,485 | 91.9% | 25 | 75 | 100 |
| **138** | **76,013** | **95.1%** (first prefix ≥95%) | 30 | 108 | **138** |
| 150 | 76,550 | 95.7% | 32 | 118 | 150 |
| 300 | 79,257 | 99.1% | 41 | 259 | 300 |

Staging scope — `sema_staging.condition_staging`, 2 studies, 509 codes / 79,371 rows. **This is the
scope D3 recommends, so this is the table G-04/G-05 must be costed against**; the raw table above is
retained only to show the two scopes are near-identical in shape:

| top N codes | rows | share | already in gold | new rows | **labels needed** | gold codes out of tier |
|---|---|---|---|---|---|---|
| 10 | 48,259 | 60.8% | 8 | 2 | 10 | 56 |
| 25 | 59,809 | 75.4% | 13 | 12 | **25** | 51 |
| **50** | **66,709** | **84.1%** | 18 | 32 | **50** | 46 |
| 64 | 69,063 | 87.0% | 19 | 45 | 64 | 45 |
| 100 | 72,978 | 92.0% | 24 | 76 | 100 | 40 |
| **137** | **75,421** | **95.0%** (first prefix ≥95%) | 29 | 108 | **137** | **35** |
| 150 | 75,998 | 95.8% | 32 | 118 | 150 | 32 |
| 300 | 78,673 | 99.1% | 41 | 259 | 300 | 23 |

The load-bearing fact, corrected twice: a bounded labelling budget still buys most of the
row-weighted signal, **but it is not free relative to what exists, and what exists is not a
head start.** Under staging, a ≥95% tier is 137 codes = **137 labels** (not 108), and 35 current
gold codes fall *out* of tier — which under G-03's disjoint states is **34 out-of-tier + 1 retired**
(`GBM`, absent from staging entirely). Demanding 100% code coverage costs a further **372** labels
(509 − 137) to move 95.0% → 100%.

The last column above is *gold codes not in the head*; it is not the acceptance denominator. Per
G-02a the benchmark is **two** frozen populations — head-137 (gates the verdict) and the 12-code
challenge stratum (scored and reported separately, never merged into the primary matrix).

### Secondary finding — enumeration and materialization disagree on "loaded", but not the way it looks

`enumerate_distinct_codes()` sees **three** studies (it reads `cbioportal_*.sample`), while
`sema_staging.condition_staging` holds **two**. It is tempting to read this as "staging is the
narrower, safer scope". Measured, it is not:

| scope | studies | distinct codes | rows | covered by gold |
|---|---|---|---|---|
| raw `cbioportal_*.sample` | 3 | 510 | 79,963 | 64 codes / 64.7% of rows |
| `sema_staging.condition_staging` | 2 (`msk_impact_50k_2026` 502 codes / 54,331 rows; `msk_chord_2024` 63 codes / 25,040 rows) | **509** | 79,371 | **63 codes** |

Choosing staging drops `gbm_tcga_pan_can_atlas_2018` (~1 code unique to it) and **keeps
`msk_impact_50k_2026`, which is the entire source of the drift.** Staging is not a smaller problem —
it is the same problem minus one code. D3 must be decided on evaluative power, not on scope size.

### Third finding — the resolver already ran

`sema_resolve.value_mapping` holds **509 decisions** (one per staged code); `condition_staging` shows
79,298 `RESOLVED` / 73 `NO_MAP` rows. The decisions are **not** uniform — re-measured:

| `status` | `resolution_status` | `confidence` | count |
|---|---|---|---|
| `auto_accepted` | `RESOLVED` | 1.0 | 497 |
| `auto_accepted` | `NO_MAP` | 1.0 | **7** (`PANEC`, `BTOV`, `CCPRCC`, `ESS`, `HGESS`, `MPC`, `UNKNOWN`) |
| `review_pending` | `RESOLVED` | 0.5 | **5** (`IMMC`, `MASC`, `FHRCC`, `ICEMU`, `UESL`) |

The measuring instrument has a subject **today**. This raises the value of G-05 (labels) relative to
G-01…G-04 (machinery): the machinery grades an existing, unexamined set of 509 automated decisions
the moment a single label exists.

**The 12 non-`auto_accepted`/`RESOLVED` decisions are the highest-information labels in the whole
set** and are not a frequency-tier concern — they are where the resolver has *declared* uncertainty
(the 5 `review_pending` @ 0.5) or declared refusal (the 7 `NO_MAP`, which `score()` can only grade as
TN-or-FN with a gold label present). Label these 12 **regardless of which tier D1 freezes**; they
cost 12 labels and are the only rows that can currently distinguish "the resolver is right" from
"the resolver is confidently right and quietly wrong elsewhere".

Decision-key join **verified sound**: `Decision.source_code` is the store's `normalized_source_value`,
and 0 of 509 fail to match a raw staged `source_oncotree_code`, so `score()` will not silently
no-op on a normalization mismatch. `GBM` is the only gold code with no live decision (63/64 gold
codes are staged) — it is the sole retirement case under D3(ii).

## Goal

The integration suite is green and stays green across future ingests, **and** the gold set has real
evaluative power: a bounded, explicitly-scoped, human-gated set of labels that `score()` can grade a
resolver against without grading its own homework.

Note these are separable. G-01…G-04 + G-07 deliver the green suite with **zero** labelling work.
G-05 delivers the oracle (G-06 is now near-complete — see the work breakdown). Do not let the second
block the first.

But do not mistake the first for progress on the second. At 0 labels `score()` grades nothing, and
`test_mapping_report_live.py` already self-reports "provisional — not accepted" — a green suite is a
green light on a meter with no needle. Machinery-first is defensible on dependency order (D4 has no
cheap answer), not on value. The only work that changes what anyone knows is **G-05**.

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

- **D1 — Coverage contract. RECOMMEND: frequency-tiered, with the tier frozen in the artifact.**
  Replace "every observed code" with "every code in tier T", where T is the smallest prefix reaching
  ≥95% row-weighted share **computed once at snapshot time and written into the artifact header** —
  under the recommended staging scope (D3(ii)), **137 codes / 95.02%**. T must **not** be recomputed
  at test time: a frequency-defined tier is
  itself ingest-sensitive, so a recomputed T re-introduces exactly the drift D3 removes. Codes
  outside T are recorded as out-of-tier, not missing (see the `OUT_OF_TIER` hazard in G-02).
  Alternative (100% coverage) is defensible only if labelling is cheap — it is not, at 509 codes.
  Note the corrected cost: T is not "the existing 64 plus a tail" — under staging it is **137
  labels** (108 codes not yet scaffolded, plus the 29 scaffolded-but-unlabelled), and 35 of the
  current 64 fall out of tier.

  > **D1 contradicts the governing PRD and requires an explicit amendment — this is a blocker on
  > D1, not a footnote.** `tasks/prd-sema-mapping-slice0.md:383-385` says hand-labelling "may start
  > with a documented subset (≥ the top codes by `row_count` covering **≥80% of rows**)" and that
  > "acceptance thresholds (US-012) require **100% distinct-code coverage**." A frozen 95% tier
  > violates the second clause permanently: `evaluate_acceptance` gates `ACCEPTED` on
  > `coverage_fraction >= 1.0`, so under a tier contract US-012 can **never** reach `accepted`
  > however well the tier is labelled. Choose one and write it down:
  > - **(x) Amend the PRD** so 100%-coverage acceptance is redefined as *100% of the frozen tier*,
  >   with out-of-tier codes excluded from the denominator. **RECOMMENDED** — it is what D1 actually
  >   means, and it is the only option under which the gate can ever go green.
  > - **(y) Keep the PRD's 100%-of-observed rule** and treat D1's tier purely as a *labelling
  >   order*, accepting that every report reads "provisional — not accepted" until all 509 are
  >   labelled.
  >
  > Whichever wins, the ≥80% subset floor binds the *starting* tier: see G-05.
- **D2 — `row_count` semantics. RECOMMEND: frozen snapshot + explicit refresh.** `row_count` is a
  scoring *weight*, not truth. Freeze it in the artifact with a snapshot date; provide
  `sema eval refresh-goldset` to re-stamp counts; demote exact-equality to a reported drift
  percentage. Keeping the current live-equality assertion guarantees a red suite after every ingest.
- **D3 — Study scope. Declared, not discovered — the mechanism is settled; the *content* of the
  declaration is the real decision.** Mechanically: the artifact carries a header (`study_schemas`,
  `snapshot_date`, `source_of_truth`), enumeration takes an explicit scope argument, and
  auto-discovery survives only inside the refresh command. Uncontroversial.

  What the declaration *says* is the actual choice, and it is not raw-vs-staged (measured above:
  510 vs 509 codes — the same problem minus one code). It is **old-scope vs current-scope**:

  - **(i) Freeze at the original 2-study scope** (`gbm_tcga…` + `msk_chord_2024`). Both integration
    tests pass **today, unchanged, with no re-scaffold** — verified: 64/64 codes present, 64/64
    `row_count`s match. Cost: the oracle grades 25,040 of the 79,371 staged rows and ignores
    `msk_impact_50k_2026` entirely, i.e. it certifies the study the pipeline mostly *isn't* mapping.
  - **(ii) Re-scope to `sema_staging.condition_staging`** (`msk_impact_50k_2026` + `msk_chord_2024`,
    509 codes). This is what the pipeline actually maps and what the 509 live resolver decisions
    cover. Cost, **staging-scoped** (not the raw figures): the ≥95% tier is **137 codes = 137
    labels**, of which 108 are not yet scaffolded; **35** existing gold codes fall out of tier; and
    `GBM` leaves the scope entirely (retired, per G-03).

  **RECOMMEND (ii), staging** — but adopt it with eyes open: it is a re-scope, not a re-stamp, and it
  raises the in-tier labelling surface from 0 actual labels to 137. (i) is a legitimate *interim*
  move to unblock the suite, but it must be recorded as a knowingly-narrow oracle, not as "the drift
  is fixed".

  Note the scope choice barely moves the arithmetic (raw ≥95% tier = 138 codes / 95.06%; staging =
  137 / 95.02%), which is exactly why D3 must be argued on evaluative power. What it *does* change is
  which numbers G-04 and G-05 are costed against — use the staging table, not the raw one.
- **D4 — Labelling source. Narrowed by measurement; still needs your call; blocks G-05.**

  The independence check the earlier draft deferred has been run. **There is no oracle route inside
  `poc.duckdb` that is independent of the resolver.** Two measured facts:

  1. The resolver's path is OncoTree concept → `concept_relationship 'Maps to'` → valid standard
     Condition concept (871 of 885 OncoTree concepts carry it; `LUAD` 777926 → 45768916 SNOMED
     "Primary adenocarcinoma of lung"). Anything that reaches OMOP through `concept_relationship`
     **is** the resolver's path — self-grading with extra steps, by construction.
  2. Candidate (a) is not merely risky, it is **unimplementable here**: the `ncit` column in
     `oncotree_reference_64.csv` (`C3512`, …) does not resolve. `vocabulary_omop.concept` under
     `vocabulary_id='NCIt'` uses ICD-O-shaped codes; `C3512` is absent, and **0 of 64** gold codes
     have an NCIt→`Maps to` route. (a) would require importing a real NCIt (or UMLS) crosswalk from
     outside this build before it is even testable for independence.

  So the live candidates are:
  - **(b) Human curation** against the reference CSV — slowest, unimpeachable, the original intent
    ("awaiting human label"). **RECOMMENDED** for tier-1, whose size and composition are set by
    G-05 (top 50 head = 84.1% of staged rows, plus the 12 challenge codes = **62 labels**), leaving
    the rest `UNLABELLED` rather than fabricating.
  - **(c) A third-party published OncoTree→SNOMED/OMOP mapping** with a usable licence, shipped as a
    file in the repo. Genuinely external and re-checkable; the only route that scales past hand
    labelling. Worth a timeboxed search before committing to (b) at volume.

    > **(c) requires a derivation-lineage check before adoption — "externally hosted" is not
    > "independently derived".** (a) was rejected because every `concept_relationship 'Maps to'`
    > route *is* the resolver's own path; a published OncoTree→SNOMED crosswalk generated from those
    > same OMOP relationships is circular in exactly the same way, merely laundered through a third
    > party. Before adopting any file, establish how it was produced and reject it if the lineage
    > passes through OMOP `Maps to`. Spot-check: if its answers agree with the resolver on ~100% of
    > the head, treat that as evidence of shared derivation, not of correctness.
  - ~~(a) OncoTree→NCIt/UMLS via `vocabulary_omop`~~ — **rejected**, no route (see above).

  **Anti-anchoring constraint on whichever wins.** Because the resolver's answers *are* the
  `vocabulary_omop` answers, a worksheet that pre-fills "candidate OMOP concepts" hands the reviewer
  the resolver's output to rubber-stamp — self-grading with a human signature. The worksheet must
  present OncoTree `name`/`mainType`/`tissue` and require an independent lookup, or blind the
  reviewer to the resolver's choice.
- **D5 — Artifact location.** `tests/data/gold/` implies test fixture; a curated oracle with a human
  gate is closer to a project asset. Low stakes — decide when touching the file.

## Work breakdown (TDD — failing test first)

**G-01 — Declare the scope (D3).** Add a header record (or sidecar `*.meta.json`) with
`study_schemas`, `snapshot_date`, `source_of_truth`, **and the target-side pins** (below). Give
`enumerate_distinct_codes()` an explicit scope argument; keep discovery for refresh only. Test:
enumeration over a declared scope ignores an unlisted study present in the DB.

> **A `study_schemas` list alone does not implement D3(ii).** `distinct_oncotree_sql()` hardcodes
> `SELECT ONCOTREE_CODE FROM {schema}.sample` (`src/sema/eval/mapping_goldset.py:192`) — one schema
> per study, column `ONCOTREE_CODE`, table `sample`. Staging is a *different shape*: one table,
> `sema_staging.condition_staging`, column `source_oncotree_code`, studies distinguished by the
> `source_schema` **column value**, not by schema name. Passing `['cbioportal_msk_chord_2024', …]`
> to the existing builder generates SQL against tables that are not the declared source of truth.
> So `source_of_truth` must be an **executable source specification** — minimally
> `{kind: raw_samples | staging, table, code_column, scope_column, scope_values}` — with one
> enumerator per `kind`, and the artifact header naming which one produced it. Test: the same
> declared scope enumerated via both kinds yields the same code set minus the known `GBM` delta.

> **The header declares the source scope but not the target — pin `vocab_release` at minimum.**
> `gold_concept_id` is a bare OMOP integer, which is meaningless without the vocabulary release that
> minted it. `sema_resolve.value_mapping` **already carries a `vocab_release` column**; the gold set
> does not. That asymmetry means an OMOP vocabulary refresh presents as resolver drift — the graded
> subject is versioned and the grading key is not, and the G-03 projection hash would faithfully
> preserve an integer nobody can re-verify. Add to the header: `target_vocabulary`, `target_domain`,
> `vocab_release`, and `oracle_source` + `oracle_version` (the file or human process from D4).
> Per-row, store the durable SNOMED **concept code** alongside `gold_concept_id` so a label survives
> a concept_id change. Curator identity, review date, and adjudication notes are worth having on a
> long-lived curated asset but are **not** Slice-0 blockers — the existing `notes` field carries
> them until the artifact graduates out of `tests/data/` (D5).

**G-02 — Split the coverage contract (D1).** `test_gold_set_covers_every_observed_code` becomes
"every code in the declared, header-frozen tier is present in the gold set, and row-weighted
coverage ≥ target". Add `GoldSet.out_of_tier_codes()` alongside the existing unlabelled accounting.
Test: a code below the tier threshold does not fail coverage; a missing in-tier code does.

> **Filtering `score()` is not sufficient — the acceptance gate reads a different function.**
> `evaluate_acceptance` gates on `coverage_fraction < 1.0`
> (`src/sema/eval/mapping_report_utils.py:75`), and `GoldSet.coverage_fraction()` is
> `labelled_count / len(self.rows)` over **every artifact row**
> (`src/sema/eval/mapping_goldset.py:95-98`) — it never consults tier or retirement. So under
> D3(ii), preserving the 34 out-of-tier rows + retired `GBM` as `UNLABELLED` (as G-04 requires) caps
> coverage at **137/172 = 79.7% permanently**: fully labelling all 137 in-tier codes still reports
> "provisional — not accepted", and `unlabelled_codes()` + the report's `total_codes` surface the
> excluded rows as unfinished work forever.
>
> Eligibility must therefore flow through **one** predicate consumed by all five call sites, not
> just `score()`:
> 1. `score()` — skip ineligible rows (already skips `UNLABELLED`).
> 2. `GoldSet.coverage_fraction()` — denominator = eligible rows only.
> 3. `GoldSet.unlabelled_codes()` — report in-tier gaps as work, out-of-tier as accounted-for.
> 4. `MappingReport.total_codes` / `labelled_count` — same denominator as the gate.
> 5. Gate D-lite staging QA — must not count retired codes as coverage misses.
>
> Test the gate directly, not just the scorer: an artifact with every in-tier code labelled and
> every out-of-tier code `UNLABELLED` must reach `ACCEPTED` (given passing precision/auto rates).
> This is the single change that decides whether the eval can ever go green.

### G-02a — Two populations, not one union (supersedes the earlier "eligibility = head ∪ sentinels")

An earlier revision folded the 12 declared-uncertain codes (G-05) into the acceptance denominator,
making it 148. **Do not implement that.** Measured, staging scope:

| population | codes | Zone-1 | distinct `auto_resolution_rate` | rows | row-weighted |
|---|---|---|---|---|---|
| frozen head (top-137) | 137 | 136 | **99.27%** | 75,421 (95.02%) | 99.93% |
| union (head ∪ 12) | 148 | 136 | **91.89%** | 75,521 (95.15%) | 99.80% |
| challenge (the 12) | 12 | 0 | 0.00% | 154 (0.19%) | 0.00% |

The only non-Zone-1 code inside the head is `IMMC` (rank 130); the other 11 sit at ranks 161–488.

Two separate reasons to split, and the **second** is the load-bearing one:

1. *Distortion (real but bounded).* The union costs 7.38pp of distinct-code `auto_resolution_rate`
   and 0.13pp row-weighted. It would **not** flip acceptance today — 91.89% still clears the 70%
   gate — so this is a correctness point, not a crisis. Note the head is *already* deliberately
   unrepresentative of distinct-code difficulty (137/509 codes carrying 95% of rows); adding 11
   codes perturbs a skewed population, it does not corrupt a clean one.
2. *Comparability (decisive).* The 12 were selected **by the current resolver's output**. Folding
   them into the acceptance population makes benchmark membership depend on the system under test:
   a future resolver flags different codes, the population moves, and scores stop being comparable
   across versions — which is the entire purpose of freezing a benchmark.

So define **two frozen populations and two eligibility predicates**:

- **`acceptance_eligible`** — the frozen top-137 head. This alone feeds `coverage_fraction()`, the
  acceptance gate, and the headline verdict.
- **`score_eligible`** — head ∪ challenge. Both populations are scored and reported; the challenge
  stratum gets its **own** confusion matrix, never merged into the primary one.

The challenge stratum is **not optional garnish — it is the only population that can grade NO_MAP
at all.** `no_map_accuracy`'s denominator is `tn + fp_map`, and all 7 resolver `NO_MAP` codes rank
161–488, i.e. **none is in the head**. Head-only, `no_map_accuracy` is `None` forever. The challenge
population is what makes that metric computable; the split is what keeps it from contaminating the
gate.

Both populations are frozen in the header by explicit code list, never recomputed (D1's rule applies
to both). Test: a challenge-stratum row never changes any value in the primary matrix.

> **Hazard — do not add `OUT_OF_TIER` to `GoldLabel`.** `score()` skips only `UNLABELLED`, and
> `classify_cell` treats *every* non-`RESOLVED` gold label as gold-NO_MAP: an out-of-tier code
> predicted Zone-1 would score **`fp_map`**, so every out-of-tier auto-accepted mapping becomes a
> false positive and `mapped_precision` collapses. Tier membership must be a separate row attribute
> (or a header exclusion list) that `score()` filters on — never a fourth `GoldLabel` member. The
> §1.5(f) matrix module stays frozen.

**G-03 — Decouple `row_count` (D2).** Delete the live-equality assertion; add
`goldset_drift_report()` returning per-code delta + aggregate drift %. Add
`sema eval refresh-goldset` (the `eval` group already exists in `src/sema/cli_eval.py` with
`run`/`diff`/`report`) to re-stamp counts and scope while preserving labels and notes. Test: refresh
updates `row_count` and never mutates `gold_concept_id`/`gold_label` — assert a hash over the
`(oncotree_code → gold_concept_id, gold_label)` projection is byte-identical across refresh.

> **`refresh` as specified contradicts "the artifact is frozen".** `row_count` is the row-weighted
> scoring *weight*, so re-stamping it in place silently changes historical row-weighted metrics
> with labels and resolver decisions unchanged — two reports become incomparable with nothing in
> either to say why. **Refresh must emit a new snapshot version, never overwrite one**
> (`snapshot_version` in the header + a new filename; the prior snapshot stays on disk). Keep the
> three operations separate and separately named — they have different blast radii:
> - **observe** — record current counts into a *new* snapshot; labels carried forward verbatim.
> - **re-tier** — recompute the frozen tier (a new benchmark; scores before and after are **not**
>   comparable and the report must say so).
> - **re-scope** — change `study_schemas` / `source_of_truth` (retires and admits codes; strictly a
>   new benchmark).
>
> None of the three may mutate a published snapshot in place. Test: running refresh leaves the
> prior snapshot file byte-identical.
`refresh` is the likelier self-labelling leak path than the resolver, so it gets the stronger
invariant. Decide and encode: a code that *disappears* from the data is **retired, never deleted** —
deleting discards human labour. Under D3(ii) `GBM` is the one live retirement case, so it is the
fixture for this test rather than a hypothetical.

> **The projection hash cannot detect the corruption it is meant to catch.** `by_code()` and
> `score()`'s `gold_by_code` are both dict comprehensions keyed on `oncotree_code`
> (`src/sema/eval/mapping_goldset.py:79`, `:134`), and `load_gold_set()` never checks uniqueness —
> so a duplicated code silently last-wins, and a hash taken over the *mapping*
> `(oncotree_code → gold_concept_id, gold_label)` is by construction blind to it. Add, as artifact
> invariants asserted on load: **unique** `oncotree_code`; tier states **disjoint and exhaustive**
> (every row is exactly one of in-tier / out-of-tier / retired); canonical row ordering; and
> label↔concept consistency (`RESOLVED` ⇒ non-null `gold_concept_id`, `NO_MAP`/`UNLABELLED` ⇒ null).
> Take the hash over the **ordered row list**, not the dict.

> **Specify the drift formula so it cannot cancel.** "Aggregate drift %" over signed per-code deltas
> lets a doubled LUAD offset a halved GBM and report ~0. Use
> `sum(|new − old|) / sum(old)` over codes present in both snapshots, and report **separately**:
> codes added, codes disappeared (→ retired), and whether the declared scope itself changed — a
> scope change makes per-code drift meaningless and must be flagged, not averaged. **Label the
> scope of every drift number** — an aggregate over 509 codes of which 372 are never labelled or
> scored is a statistic with no decision attached to it. Report drift for the eligible populations
> and for the full universe as two clearly-named figures, never one blended one.

> **Freeze the whole measured universe, not just the eligible rows.** The artifact carries 172 rows
> against a 509-code scope; a code list plus a hash is **not** enough, because per-code drift for
> the 338 codes outside the artifact then has no frozen `old` value — their deltas are
> incomputable, they drop out of the aggregate denominator, and a large change among them biases
> the aggregate toward zero. Ship a sidecar **universe manifest**: `{code, frozen_row_count}` for
> **all 509** codes, plus scope + snapshot date + hash.
>
> This is storage, **not** eligibility — the two are independent, and an earlier draft wrongly
> rejected the manifest as "coverage absolutism". A frozen 509-code manifest introduces no ingest
> sensitivity and no labelling obligation *provided* nothing puts those codes into the label-coverage
> denominator; that is guaranteed by G-02a's `acceptance_eligible` predicate, not by omitting them
> from disk. It also makes membership and omissions directly inspectable, which a hash cannot.
> Test: every code in the manifest is classifiable into exactly one of in-tier / challenge /
> out-of-tier / retired, and the four sets partition the manifest exactly.

**G-04 — Re-scaffold to the declared scope.** Regenerate the artifact for whatever D3 declares:
preserve all existing rows and labels, re-stamp counts, add the in-tier codes now missing, mark them
`UNLABELLED`, and fix the stale `notes` (they still claim `gbm_tcga…, msk_chord_2024`). Under D3(ii)
this adds **108** in-tier codes, marks **34** existing ones out-of-tier, and retires `GBM` — a
re-scope, not a re-stamp. After this, G-07's tests pass with zero labelling. Test: re-scaffold is
idempotent and label-preserving.

> **The "35 out-of-tier" figure violated G-03's own disjointness invariant.** `GBM` is not in
> `condition_staging` at all, so it is one of the 64 − 29 = 35 gold codes outside the top-137 — but
> it is also the retirement case. Under mutually exclusive states the split is **34 out-of-tier +
> 1 retired**, not 35 + 1. Corrected above and in D3(ii). The four states
> (in-tier / challenge / out-of-tier / retired) must partition the universe manifest exactly, which
> is the test G-03 already requires.

**G-05 — Label tier 1 (D4; human gate).** Populate `gold_concept_id`/`gold_label` for the frozen
tier from the chosen oracle.

**Tier 1 = the 12 declared-uncertain decisions + top 50 by row_count = 62 labels**, covering
**84.1%** of staged rows. Rationale for each half:
- **Top 50, not top 25.** The PRD's documented-subset floor is **≥80% of rows**
  (`tasks/prd-sema-mapping-slice0.md:383-385`); top-25 is **75.4%** and fails it, top-50 is
  **84.1%** and clears it. Starting at 25 would need a PRD amendment for no gain.
- **All 50 are new labels, not 32.** 18 of the 50 are already scaffolded — but scaffolded rows are
  `UNLABELLED`, so they carry no oracle value. The cost is 50 human labels.
- **Plus the 12 non-`auto_accepted`/`RESOLVED` codes** (5 `review_pending` @ 0.5, 7 `NO_MAP`),
  wherever they sit in the frequency distribution. These are the only rows that can currently grade
  the resolver's *self-assessment*; without them `no_map_accuracy` has an empty denominator and the
  0.5-confidence decisions are ungraded. Cheapest disproportionate signal in the plan.
  **They are labelled into the challenge population (G-02a), not into the acceptance denominator.**
  Only `IMMC` (rank 130) is also in the head; the other 11 are challenge-only.

Deliverable preparable before D4 lands: a labelling worksheet — reference CSV regenerated for the
declared tier with OncoTree `name`/`mainType`/`tissue`. Per D4's anti-anchoring constraint, it must
**not** pre-fill candidate concepts drawn from `vocabulary_omop`: those are the resolver's own
answers. For the 12 uncertain codes specifically, the worksheet must also **not** reveal that the
resolver flagged them — a reviewer told "the machine was unsure here" labels differently, and the
worksheet must interleave them with head codes so the stratum is not visually identifiable.

**Minimum annotation floor (per label).** Curator, review date, the evidence consulted, and — for
every `NO_MAP` — an explicit justification of *absence* ("no acceptable target exists in
`{vocabulary}@{release}` because …"). A `NO_MAP` is a positive claim, not a shrug.

**Minimum independent review (acceptance-gating).** Not a full adjudication programme, but not
nothing either, because `classify_cell` makes gold `NO_MAP` labels asymmetrically dangerous: a code
the resolver maps **correctly** scores `fp_map` if the gold label is wrongly `NO_MAP`, so one bad
`NO_MAP` directly damages `mapped_precision`. Second-review:
- all **12** challenge codes (the 7 `NO_MAP` **are** a subset of the 12 — one list, not two), and
- a random sample of ~10 head `RESOLVED` labels.

That is ~22 second-reviews — an hour or two, not a programme. **If no second reviewer is available,
that is acceptable, but the verdict must then carry `unadjudicated`** (G-06) rather than silently
implying the oracle was checked.

**G-06 — Add the eval CLI surface (mostly already built).** `score()` is already wired end-to-end:
`src/sema/eval/mapping_report.py` (US-012) reads `ValueMappingStore`, calls `score()`, applies the
acceptance gate via `mapping_report_utils`, and `tests/integration/test_mapping_report_live.py`
already runs the real resolver against the gold set and asserts "provisional — not accepted" while
unlabelled. The remaining gap is narrow: a `sema eval mapping-report` subcommand that writes to
`eval-runs/`. Note the store's column is `normalized_source_value` (plus `source_vocabulary`,
`resolution_status`, `status`, `confidence`, `policy_ref`, `run_id`) — there is no `source_code`.

> **The report must name its subject; today it silently averages over whatever is in the store.**
> The store's `GRAIN_KEY` is a 5-tuple — `source_vocabulary, normalized_source_value,
> target_property_ref, resolver_policy_ref, vocab_release`
> (`src/sema/resolve/value_mapping_store_utils.py:44-50`) — but `report_from_store()` defaults every
> filter to `None` (`src/sema/eval/mapping_report.py:78-84`) and `score()` collapses on
> `d.source_code` alone with a last-wins dict (`mapping_goldset.py:135`). Measured today: 1 distinct
> value each for policy / release / property / vocabulary and 0 duplicate codes, so this is
> **latent, not live** — but the first re-run under a second policy or vocabulary release silently
> blends two resolvers into one matrix, with the winner decided by DuckDB row order.
>
> Require an explicit **evaluation-subject key** (all five fields, no defaults), derive the store
> filter from it, and **raise** on more than one decision per eligible code rather than last-wins.
> Test: a store holding two releases yields a hard error without a subject key, and identical
> results under either ordering with one.

> **Make each report an immutable, self-describing artifact.** `sema eval mapping-report` writes to
> `eval-runs/<id>/` : the subject key, the `run_id`, the gold-set `snapshot_version` + hash, the
> universe-manifest hash, **and the decision set actually graded** (plus its digest). This is the
> eval-side answer to the store's mutability (see "Deferred"); it makes any past report
> re-verifiable without the store having to preserve history.

> **The verdict must be scope-qualified.** A generic `ACCEPTED` over a frequency-selected head
> overclaims: the head is 137/509 codes chosen *by row frequency*, so it says nothing about the
> distinct-code tail, and the existing `per_bucket` matrices only bucket rows that were labelled and
> scored — they are not evidence about an unlabelled tail. `STRUCTURAL_PRECISION_CAVEAT` covers
> determinism, not population coverage. Emit
> `accepted_for_frozen_frequency_head` (with the snapshot version), never bare `accepted`, plus
> `unadjudicated` when G-05's independent review was skipped. Report **three strata separately**:
> head, challenge, and an informational random-tail sample (~15–20 codes, never gating) — the
> random tail is the only thing that speaks to "confidently wrong in the tail", and it must not be
> presented as supporting a ≥95% precision claim over the full universe.

**G-07 — Re-green the integration tests** and confirm no ingest-sensitivity remains: add a test that
introducing an unlisted study schema does not change any assertion outcome. **This needs a fixture
DuckDB (or a fake cursor), not `~/.sema/poc.duckdb`** — both current tests skip when that file is
absent, so the whole contract is unverified anywhere but one developer's machine, and "an unlisted
study is ignored" cannot be staged against a live personal DB at all.

> **Two contracts, deliberately separated — the plan's goal was self-contradictory without this.**
> "Ingesting a new study does not red the suite" and "fail when the frozen tier's current row-share
> falls below a floor" cannot both hold unconditionally: a large enough ingest must trip the second.
> Left implicit, implementation encodes whichever reading makes the tests pass. Split them:
>
> - **Snapshot integrity** (`test_mapping_goldset_coverage.py`) — the artifact is internally
>   consistent and matches its *declared* scope. **Never reds on ingest.** This is the green-suite
>   goal.
> - **Benchmark freshness** (a separately named test) — the frozen tier still covers ≥ a declared
>   floor of *current* rows, and the unseen-code share is below a declared ceiling. **May red**, and
>   its failure message says "benchmark stale — re-snapshot via G-03 `re-tier`", never "gold set
>   drifted".
>
> The freshness numbers (current frozen-tier row-share, unseen-code share, scope-change flag) are
> two fields on `goldset_drift_report()`, not a new subsystem — but they are logically a
> representativeness monitor and are named as one, so a stale benchmark is never mistaken for a
> corrupt one. Declare both thresholds in the artifact header, so "stale" is a stated policy rather
> than a number someone picked at review time.

## Definition of done

- [x] `tests/integration/test_mapping_goldset_coverage.py` passes against current `poc.duckdb`.
- [x] The same contract is verified **without** `poc.duckdb` — fixture-backed, so it holds off this
      machine.
- [x] Ingesting a new `cbioportal_*` study does not red the **snapshot-integrity** suite (proved by
      G-07's test, not by hope) — while the separately-named **benchmark-freshness** check is free
      to red, with both thresholds declared in the artifact header.
- [x] **Two frozen populations, never merged**: `acceptance_eligible` (head-137) gates the verdict;
      `score_eligible` (head ∪ 12 challenge) is scored, with the challenge stratum reported in its
      own matrix. Proved by a test that a challenge row cannot move any primary-matrix value.
- [x] A **universe manifest** freezes `{code, frozen_row_count}` for all 509 codes, and the four
      states (in-tier / challenge / out-of-tier / retired) partition it exactly.
- [x] The report names an explicit **evaluation-subject key** (all 5 grain fields), **raises** on
      duplicate decisions per eligible code, and writes an immutable self-describing run to
      `eval-runs/` including the graded decision set.
- [x] Refresh **emits a new snapshot version**; the prior snapshot file is byte-identical after.
      `observe` / `re-tier` / `re-scope` are separate named operations.
- [x] The verdict is **scope-qualified** (`accepted_for_frozen_frequency_head` + snapshot version),
      carries `unadjudicated` when independent review was skipped, and reports head / challenge /
      random-tail strata separately.
- [x] Gold-set artifact declares its scope, snapshot date, source of truth, **and its frozen tier
      membership** (tier is never recomputed at test time).
- [x] `row_count` drift is *reported* with a number, not asserted to zero.
- [x] Coverage stated as row-weighted share against a declared tier; out-of-tier codes accounted for.
- [x] **Tier/retirement eligibility flows through `coverage_fraction()`, `unlabelled_codes()`, the
      report totals, and the acceptance gate — not `score()` alone** (G-02). Proved by a test that
      an all-in-tier-labelled artifact reaches `ACCEPTED`, not `provisional`.
- [ ] **(G-05, open — human gate)** Every label carries curator, date, and evidence; every `NO_MAP` carries an explicit
      justification of absence. All 12 challenge codes + ~10 sampled head labels independently
      second-reviewed — **or** the verdict says `unadjudicated`.
- [ ] **(G-05, open — human gate) The oracle actually grades something.** Not satisfiable at zero labels:
      - `labelled_count > 0` and **≥80% of in-scope rows** covered by labelled codes (the PRD's
        documented-subset floor), i.e. G-05's top-50 tier complete;
      - all **12** declared-uncertain decisions (5 `review_pending`, 7 `NO_MAP`) labelled;
      - at least one scored decision appears in the report (`scored_codes > 0`).
      The eventual **acceptance** target (100% of the frozen tier) is tracked separately — this
      checkbox is the floor below which G-01…G-04 are machinery with no needle.
- [x] Artifact invariants asserted on load: unique codes, disjoint/exhaustive tier states, canonical
      ordering, label↔concept consistency; projection hash taken over the ordered rows, not a dict.
- [x] D1's conflict with the PRD resolved in writing — either the PRD is amended (option x) or the
      tier is demoted to a labelling order (option y). Not left implicit.
- [x] Re-scaffold and refresh are idempotent and provably label-preserving.
- [x] **No `gold_concept_id` is ever written by Sema's resolver, nor by `refresh-goldset`** —
      enforced by a projection-hash test, not a convention.
- [x] D4 resolved and recorded here. The independence check is **done and negative**: no route
      inside `poc.duckdb` is independent of the resolver, so the chosen oracle must be human
      curation or a genuinely external file.
- [x] mypy strict + unit suite green; coverage ≥ 85%.

## Out of scope

- Labelling the full 510-code tail. Tier policy is the whole point of D1.
- Building or tuning the resolver (US-006). This plan restores the *measuring instrument*.
- The other two pre-existing integration failures
  (`test_different_sources_coexist` — MERGE-key over-collapse; and the e2e `SemanticEngine(llm=...)`
  signature rot). Tracked separately.
- Re-ingesting `gbm_tcga_pan_can_atlas_2018` into staging. If D3 picks staging as the source of
  truth, that study simply leaves the gold-set scope until it is properly staged — and this costs
  almost nothing: ~1 distinct code is unique to it. Dropping gbm is **not** what shrinks or fixes
  the scope problem.
- Calibration of the resolver's confidence. The live decisions carry only **two** confidence values
  (504 @ 1.0, 5 @ 0.5), so the gold set cannot fit a calibration curve no matter how well it is
  labelled. Note this is weaker than the earlier claim that *all* 509 are 1.0: the 5 low-confidence
  decisions **are** gradeable, and G-05 labels them — that is a spot-check of whether
  `review_pending` means anything, not calibration. Curve-fitting stays out of scope.

## Corrections (2026-08-11 adversarial review)

Every number below was re-measured against `~/.sema/poc.duckdb`; the earlier figures were not wrong
arithmetic, they were computed over the wrong set.

1. **"The 64 scaffolded codes are the head of the frequency distribution and cover 86.9% of all
   rows" — false.** 86.9% is the top-64-*by-current-frequency*, a different set overlapping the gold
   set in only 20 codes. The gold set is the complete 64-code universe of the original 2 studies and
   covers **64.7%** of current rows. This inverted the plan's central argument: the old contract was
   a coherent "100% of a declared scope", broken by scope drift — not coverage absolutism.
2. **≥95% tier is 138 codes, not ~150**, and costs **108 new labels** while pushing 34 existing gold
   codes out of tier. The 100%-coverage alternative costs **372** further labels, not 510.
3. **D3 raw-vs-staged is a non-choice** (510 vs 509 codes). The real axis is old-scope vs
   current-scope; staging keeps `msk_impact_50k_2026`, the entire source of the drift.
4. **D4(a) rejected on measurement, not on risk** — the reference CSV's NCIt codes have no route in
   this OMOP build (0/64), and every `concept_relationship` route is the resolver's own path.
5. **G-06 was already built** (US-012 `mapping_report.py` + live integration test); only a CLI
   subcommand remains. `value_mapping` has no `source_code` column.
6. **`OUT_OF_TIER` as a `GoldLabel` would silently corrupt precision** via `classify_cell`.
7. **A recomputed frequency tier re-introduces the drift it was meant to fix** — freeze it in the
   header.
8. **Neither integration test runs without `~/.sema/poc.duckdb`**, so the contract is currently
   unverifiable off this machine.

## Deferred, with reasons (not dropped — scoped out of this slice)

Three reviews have converged on these as genuinely right *in direction* and genuinely outside this
plan's job, which is to restore a **measuring instrument** for a 62-label POC. Recorded here so
round four re-litigates nothing.

- **Immutable resolver-execution identity.** `GRAIN_KEY` excludes `run_id`, and `upsert_sql` is
  `ON CONFLICT (GRAIN_KEY) DO UPDATE`
  (`src/sema/resolve/value_mapping_store_utils.py:44-50`, `:179-182`) — so re-running changed
  resolver code under the same policy + release overwrites both the decision **and** the `run_id`
  that produced it. True cross-version resolver comparison therefore needs store-side snapshots
  (build SHA, config digest, append-only decision history). **Out of scope**: this plan's Out of
  Scope already excludes "building or tuning the resolver", and the store is US-005/US-006's
  contract, not the eval's. **Mitigated here** by G-06 writing the graded decision set + digests
  into `eval-runs/`, which makes every *report* immutable and self-describing without changing the
  store's grain. Track the store-side work against US-005/US-006.
- **A full adjudication programme** — dual independent labelling, curator qualification criteria,
  formal QA sampling rates, inter-annotator agreement. Right for a long-lived curated multi-domain
  layer; disproportionate for 62 labels. **Reduced here** to the G-05 floor: provenance per label,
  NO_MAP absence justification, 12 + ~10 second-reviews, and an `unadjudicated` verdict when even
  that is skipped. Revisit when the gold set outgrows `tests/data/` (D5).
- **Benchmark/monitor as an architectural split.** Adopted in substance (G-07's two contracts) but
  implemented as two fields on `goldset_drift_report()` plus a separately-named test — not a new
  subsystem. Revisit if a second target model or domain arrives, which is when one monitor stops
  fitting.

## Corrections (2026-08-11, second adversarial pass)

A second review re-measured the *first* review's corrections. Seven findings held; two were
overstated. Everything below was re-run against `~/.sema/poc.duckdb` and the source.

9. **"New rows" was being read as "new labels" throughout.** With 0/64 rows labelled, a tier of T
    codes costs **T** labels, not "T minus what's scaffolded". Top-25 is 25 labels (not 11); the
    ≥95% tier is 137 (not 108). This contradicted the plan's own P1 diagnosis and understated G-05
    by 2.3×. Fixed in the tier tables, D1, D3(ii), G-04, and G-05.
10. **Tier arithmetic was raw-scoped while the recommendation was staging-scoped.** Both tables are
    now present and labelled. Staging: ≥95% at **137** codes / 95.02%, overlap **29**, out-of-tier
    **35**, top-25 = 13 existing + 12 new / 75.35%. The raw table was *not* arithmetically wrong
    (raw ≥95% genuinely is 138 / 95.06%) — it was applied to the wrong scope. No decision in the
    plan flips on the difference; this is a consistency fix, not a reversal.
11. **Out-of-tier rows would have blocked acceptance permanently.** `evaluate_acceptance` gates on
    `coverage_fraction >= 1.0` and `coverage_fraction()` counts every artifact row, so filtering in
    `score()` alone leaves a hard ceiling of 137/172 = 79.7%. This is the load-bearing finding of
    the second pass — see the G-02 block for the five call sites.
12. **D1 conflicts with the PRD in two places, not one.** Beyond the ≥80% subset floor (which
    top-25 fails and top-50 clears), the PRD requires **100% distinct-code coverage** for US-012
    acceptance — so a frozen 95% tier makes `ACCEPTED` unreachable by construction. D1 now carries
    an explicit amend-or-demote decision.
13. **The resolver state was mis-stated.** Not "all `auto_accepted`/`RESOLVED`, confidence 1.0":
    497 / **7 `NO_MAP` @ 1.0** / **5 `review_pending` @ 0.5**. The 12 non-uniform decisions are
    promoted into G-05's tier 1 as the highest-information labels available.
14. **The definition of done was satisfiable at zero labels**, contradicting the stated Goal. An
    objective labelled-coverage floor is now a checkbox.
15. **G-01 did not implement D3(ii).** The enumerator is hardcoded to `{schema}.sample` /
    `ONCOTREE_CODE`; staging is one table keyed by a `source_schema` **column**. `source_of_truth`
    must be an executable source spec, not a schema list.
16. **The oracle had no target-side provenance.** `value_mapping` carries `vocab_release`; the gold
    set does not, so an OMOP refresh reads as resolver drift. Header now pins vocabulary/domain/
    release/oracle-source; per-row durable SNOMED code added. Curator/licence/adjudication metadata
    judged worth having but not a Slice-0 blocker.
17. **Duplicate codes are undetectable by the proposed hash.** `by_code()` and `score()` last-win on
    duplicates and `load_gold_set()` never checks uniqueness, so a hash over the code→label
    *mapping* is blind to exactly the corruption it guards. Hash the ordered rows; assert artifact
    invariants on load. Drift % also needs an absolute-delta formula so increases cannot cancel
    decreases.
18. **Verified sound, recorded so it is not re-litigated:** `Decision.source_code` is the store's
    `normalized_source_value` and 0 of 509 fail to join a raw staged code, so `score()` will not
    silently no-op on normalization. `GBM` is the only gold code with no live decision — the one
    real retirement fixture for G-03.

## Corrections (2026-08-11, third adversarial pass)

The third pass reviewed the *second* pass's fixes and reversed one of its own recommendations. All
figures below re-measured against `~/.sema/poc.duckdb` and the source.

19. **The sentinel union was wrong and is withdrawn** (proposed in pass 2, retracted in pass 3;
    G-02a supersedes it). Folding the 12 declared-uncertain codes into the acceptance denominator
    costs 7.38pp of distinct-code `auto_resolution_rate` (99.27% → 91.89%) and 0.13pp row-weighted,
    which would **not** flip acceptance at the 70% gate — so the distortion is real but bounded.
    The decisive objection is comparability, not bias magnitude: the 12 were selected **by the
    system under test**, so a union makes benchmark membership move whenever the resolver changes.
    Two frozen populations instead. Note the head is *already* deliberately unrepresentative of
    distinct-code difficulty (137/509 codes carrying 95% of rows) — adding 11 codes perturbs a
    skewed population, it does not corrupt a clean one; that framing does not carry the argument.
20. **The challenge stratum is necessary, not merely tolerated** — a point no review made until
    now. All 7 resolver `NO_MAP` codes rank 161–488, so **none is in the head**; head-only,
    `no_map_accuracy` (`tn + fp_map` denominator) is `None` permanently. The challenge population is
    what makes that metric computable at all; the split is what stops it contaminating the gate.
21. **A code list + hash cannot support the promised drift report — conceded.** With no frozen
    `old` count for the 338 codes outside the artifact, their per-code deltas are incomputable and
    a large change among them biases the aggregate toward zero. Ship a 509-row
    `{code, frozen_row_count}` universe manifest. Pass 2's objection that this "re-creates coverage
    absolutism" **conflated storage with eligibility** and was wrong: the label-coverage denominator
    is controlled by `acceptance_eligible`, not by what is on disk.
22. **`refresh` contradicted "the artifact is frozen".** Re-stamping `row_count` in place mutates a
    scoring weight, silently changing historical row-weighted metrics with labels and decisions
    unchanged. Refresh now emits a **new snapshot version**; `observe` / `re-tier` / `re-scope` are
    separate named operations, none mutating a published snapshot.
23. **The report never named its subject.** `GRAIN_KEY` is a 5-tuple; `report_from_store()` defaults
    every filter to `None` and `score()` last-wins on `source_code`. Latent today (1 distinct value
    each for policy/release/property/vocabulary, 0 duplicate codes) but silently blends resolvers on
    the first second-release run. Explicit subject key + raise-on-duplicate.
24. **"35 out-of-tier" violated the disjointness invariant the same plan requires.** `GBM` is both
    outside the top-137 and the retirement case; the split is **34 + 1**.
25. **The goal was self-contradictory on ingest.** "Does not red the suite" vs a tier-share floor
    that must red on a large enough ingest. Split into snapshot-integrity (never reds) and
    benchmark-freshness (may red, distinct failure message).
26. **The verdict overclaimed.** `per_bucket` only buckets rows that were labelled and scored, so it
    is not evidence about the unlabelled tail, and `STRUCTURAL_PRECISION_CAVEAT` is about
    determinism, not population coverage. Verdict is now scope-qualified, with head / challenge /
    random-tail reported separately and the tail sample explicitly non-gating.
27. **`NO_MAP` gold labels are asymmetrically dangerous** — `classify_cell` scores `fp_map` when a
    correctly-mapped code carries a wrong gold `NO_MAP`, so one bad label damages
    `mapped_precision` directly. Hence the annotation floor (evidence of *absence*) and the 12 + ~10
    second-review, with `unadjudicated` as the honest fallback rather than silence.
28. **Third-party crosswalks need a lineage check.** "Externally hosted" ≠ "independently derived":
    a published OncoTree→SNOMED file generated from OMOP `Maps to` is circular in exactly the way
    D4(a) was rejected for. ~100% agreement with the resolver on the head is evidence of shared
    derivation, not of correctness.
29. **Residual raw-scope figures corrected** in D1 (138 → 137 / 95.02%) and D4(b) (top-25 framing →
    G-05's top-50 + 12 = 62 labels).
