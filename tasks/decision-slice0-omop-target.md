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
  manifest's `vocabulary: SNOMED` and the `oncotree_to_snomed_condition` policy id are
  **legacy/governance anchors**, not resolution filters.
- **F4** — NO_MAP reasons are bucketed: source-code absent · no curated crosswalk
  (PANEC) · domain-gated. PANEC (OncoTree 905261, zero `Maps to`) surfaces as NO_MAP and
  routes to producer #2 / future user-review UI — an OMOP crosswalk gap, not a sema bug.

## Deferred (tracked follow-ups)

- **bug-374** — staging compiles from an unscoped `store.read_all()` over a persistent
  DuckDB cache; benign today, but a prerequisite before renaming any `resolver_policy_ref`.
- **Full repoint** — making `VocabularyBindingDecl.vocabulary` optional/domain-governed and
  renaming `oncotree_to_snomed_condition` → `oncotree_condition` (touches the graph
  `:VocabularyBinding` keys + provenance ids) is gated on bug-374.
