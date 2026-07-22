# Decision — Slice-0 target is OMOP (vocabulary-agnostic), not SNOMED

> Date: 2026-07-02 · Status: decided (human) · Branch: `ralph/feat/mapping-slice0`
> Supersedes the "ONCOTREE_CODE → standard **SNOMED** condition_concept_id" and
> "hand-labelled gold set gates Slice-0" framing in `tasks/prd-sema-mapping-slice0.md`.
> Tracked here because `docs/` and `.wolf/` are gitignored (this is the durable record).

## Decision

The user-provided semantics for Slice-0 are **OMOP**, not a vocabulary. The target
contract is:

> cBioPortal `ONCOTREE_CODE` → OMOP `condition_concept_id` where the concept is
> **valid + standard (`standard_concept='S'`) + `domain_id='Condition'`**.

OMOP standardizes the Condition *domain* across SNOMED **and** — via the oncology
extension — ICDO3 histology. So the resolver is **vocabulary-agnostic**: it gates on
valid + standard + domain and does NOT filter on a target `vocabulary_id`. A code
resolving to a standard ICDO3 Condition concept is a CORRECT OMOP mapping, **not a
miss**. A strict-SNOMED filter was rejected as anti-vision (it would drop ~1/3 of
clinically-correct mappings and re-introduce domain coupling).

## Why gold labels do not gate the deterministic path

Gold labels grade a *semantic* producer against independent ground truth. For an
OMOP-owned deterministic crosswalk, the authority IS OMOP's own `Maps to`, so
hand-labelling to "check" it is largely circular. The Slice-0 deterministic gate is
therefore **contract conformance**, not an LLM eval. Gold / adjudication is for
**producer #2** (LLM / fuzzy names / free text / ambiguous multi-survivor) and for any
*independent* benchmark claim — the cases where the machine can't decide by itself.

## Consequences (implemented on this branch)

- **F1** — `sema fit --strict` = **Gate D-lite passes AND contract conformance passes
  AND provided gold labels do not contradict**. It no longer requires the gold-based
  ACCEPTED verdict. Conformance re-verifies every resolved `concept_id` against the
  vocab (valid + standard + domain). A labelled contradiction (`wrong`/`fn`/`fp_map`)
  fails strict at any `labelled_count > 0`; full coverage is needed only to *grant* the
  (informational) ACCEPTED verdict. Code: `eval/conformance.py`,
  `VocabStore.concepts_by_ids`, `MappingReport.has_labelled_contradiction`,
  `cli_fit._enforce_strict`.
- **F2** — a machine-readable `standard_domain_governed: true` binding flag declares that
  acceptance is governed by standardness + domain, not by the target vocabulary id. The
  manifest binding vocabulary and the policy ref are domain-governed identity/provenance
  strings, not resolution filters. (The SNOMED-named anchors were retired by the full
  repoint, 2026-07-02 — see below.)
- **F4** — NO_MAP reasons are bucketed: source-code absent · no curated crosswalk
  (PANEC) · domain-gated. PANEC (OncoTree 905261, zero `Maps to`) surfaces as NO_MAP and
  routes to producer #2 / future user-review UI — an OMOP crosswalk gap, not a sema bug.

## Done (was deferred)

- **bug-374/383/384/386** — the unscoped `store.read_all()` reads over the persistent
  DuckDB cache (staging compile, strict report, VOCAB_LOOKUP producer) are all scoped to
  the current run, with run-grain dedup. These were the prerequisite for renaming any
  `resolver_policy_ref`.
- **Full repoint** — DONE 2026-07-02 (`tasks/plan-slice0-full-repoint.md`). Renamed the
  policy ref `omop.oncotree_to_snomed_condition` → `omop.oncotree_condition` and the
  manifest binding vocabulary `SNOMED` → `OMOP-Condition` (D1-A: kept the field required,
  renamed both the top-level `vocabularies:` declaration and the binding slot together to
  avoid `DanglingRefError`). Chose D1-A over making `VocabularyBindingDecl.vocabulary`
  optional (D1-B) to avoid re-keying the `:VocabularyBinding`/`:Term` graph identity.
  The `poc.duckdb` value-mapping cache was wiped of old-ref rows (D2, backed up to
  `~/.sema/value_mapping_oldref_backup.csv`) and repopulated under the new ref by a live
  Databricks `sema fit` (rows_staged 25040 == source count, Gate D-lite PASS, conformance
  0 violations — the store-key change did not fork staging).
