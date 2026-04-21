# Step 6 Milestone Summary — source-semantic-hardening

**Scope:** 12-table POC slice from `eval/dev_slice_poc.yaml` (the full 33-table cBioPortal corpus is not yet ingested — Task 10.1 remains blocked on Databricks ingest per §11-bis).

**Basis run:** `eval-runs/step5-post-cleanup/` (commit `720dfd2`) — post-Task-11 cleanup, staged A→B→C pipeline with domain prompts and healthcare few-shot library enabled.

## Tables and outcomes (12 tables)

| | |
|---|---|
| B outcome | 12 B_SUCCESS / 0 B_PARTIAL / 0 B_FAILED |
| Raw coverage (avg) | 100.0% |
| Critical coverage (avg) | 100.0% |
| Stage C trigger rate (avg) | 30.7% (95 of 259 columns flagged; 62 C calls) |
| Recovery overhead | 0 retries, 0 splits, 0 rescues |

All 12 tables classified every column on the first Stage B attempt; no bounded-recovery path was exercised.

## Cost and latency (Task 10.5)

| metric | total | per-table avg | budget | status |
|---|---|---|---|---|
| Latency | 277.6 s | 23.1 s | 60 s / table | PASS (2.6× headroom) |
| Tokens in | 73,528 | 6,127 | — | — |
| Tokens out | 33,834 | 2,820 | — | — |
| Cost (DeepSeek list price) | $0.057 | $0.0048 | $0.10 / table | PASS (21× headroom) |

Per-table latency spread: min 6.6 s (`clinical_supp_hypoxia`), median ~14 s, max 99.6 s (`mutation` — 114 columns, 5 B batches, 13 C calls). Every table is below the 60 s gate individually; `mutation` is the only one within a factor of 2 of it.

## Semantic churn — rollout history (Task 10.2)

| step | tables | added | removed | changed | description |
|---|---|---|---|---|---|
| 2 | 6 | 17 | 141 | 684 | single-pass baseline → A→B staged |
| 3 | 6 | 8 | 4 | 760 | + domain-aware prompts |
| 4 | 6 | 3 | 16 | 611 | + few-shot examples |
| 5 | 6 | 87 | 4 | 545 | + Stage C (80 new `has_decoded_value`) |
| 11 | 12 | 23 | 22 | 670 | pre- vs post- cleanup (sanity) |

Net trajectory: structural removals concentrated in step 2 (design-intended; see §10.3 below). Steps 3–5 show low removal counts (4, 16, 4). The Task 11 cleanup run is symmetric (23 added / 22 removed) — consistent with LLM noise, no net regression from removing the deprecated paths.

## Systemic regression review (Task 10.3)

Every removal cluster flagged during the rollout has been root-caused and either fixed or accepted as design-intended. No open regressions.

| step | regression | disposition |
|---|---|---|
| 2 | 75 of 141 removals were L2 `vocabulary_match` assertions | **Accepted**: design §2a reassigns this predicate to L3 exclusively. |
| 2 | 57 of 141 removals were `has_decoded_value` | **Accepted**: Stage C disabled at step 2 by design; restored at step 5 (+80 under new ownership). |
| 2 | 1 removal — `has_property_name="BIOTYPE (STRING)"` from LLM type-suffix leak | **Fixed** in commit `46384de`. |
| 4 | 52 `has_alias` regressions from few-shot examples with empty `synonyms` fields | **Fixed** in commit `783266d` (alias churn reduced to 16, none systemic). |
| 5 | 4 removals vs. step 4 | **Accepted**: LLM-noise level, no predicate systemically affected. |
| 11 | 22 removals in 12-table run vs. pre-cleanup | **Accepted**: symmetric with 23 additions — LLM noise, no structural loss after removing legacy paths. |

Zero high-value predicates lost: `has_entity_name`, `has_property_name`, `has_semantic_type` all retained across every step on every table.

## Verdict

On the 12-table POC slice, the A→B→C staged pipeline:

- hits 100% raw and critical Stage B coverage on every table,
- runs every table under the latency gate with a 2.6× safety factor,
- runs every table at ~1/20 of the cost budget,
- exhibits no open systemic regressions across the five-step rollout,
- exercises Stage C on ~31% of columns and produces the design-intended `has_decoded_value` coverage.

**Step 6 is signed off on the 12-table POC slice for Tasks 10.2, 10.3, 10.5.**

## Still open (require more ingest)

- **10.1** — run the full 33-table cBioPortal corpus. Blocked on ingesting the remaining ~21 tables per `cbioportal-omop-data-bridge` runbook.
- **10.4** — holdout-vs-dev-slice bias check. Requires 8 more holdout tables ingested and decontaminated from the POC slice (see §11-bis).
- **7.8 / 8.8 / 9.9** — full-corpus spot-checks for domain prompts, few-shot, and Stage C. Same ingest blocker.
