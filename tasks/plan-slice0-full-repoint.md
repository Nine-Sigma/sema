# Plan — Full Slice-0 repoint (retire the SNOMED legacy anchors)

> Date: 2026-07-02 · Status: planned (not executed) · Branch: `ralph/feat/mapping-slice0`
> Follows: `tasks/decision-slice0-omop-target.md` (the OMOP-not-SNOMED decision).
> Unblocked by: **bug-383** (report scoped to run) + **bug-374** (staging scoped to run),
> both fixed on this branch. The two scoping fixes were the stated prerequisite for
> renaming any `resolver_policy_ref`.

## Goal

Make the code say what the decision record already declares: Slice-0's target is
**OMOP standard-Condition (vocabulary-agnostic)**, not SNOMED. Retire the two legacy
anchors that still read as SNOMED-specific:

1. The manifest binding field `vocabulary: SNOMED`.
2. The policy ref value `omop.oncotree_to_snomed_condition`.

Neither is a *resolution filter* today (the resolver gates on valid + standard +
domain via `standard_domain_governed: true`). They are identity/provenance strings —
which is exactly why the repoint is delicate: both are **keys**, not free labels.

## Why this is delicate (the two keys)

- `resolver_policy_ref` is a **value-mapping store grain key** (persistent DuckDB
  cache; `GRAIN_KEY` includes it) AND a `:VocabularyBinding` graph property. Renaming
  it makes every existing store row's grain change — old rows become inert siblings of
  the new ones. bug-374/bug-383 now keep those inert rows out of staging and the strict
  report, so the rename is *safe*, but the cache still accumulates dead rows.
- `vocabulary_name` is part of the **`:VocabularyBinding` MERGE key** and is embedded in
  its node `id` (`neo4j_writer_utils.py:265-305`), and `:Term` supersede/flip matches on
  `n.vocabulary_name = key[0]` (`neo4j_writer_flip_utils.py:114`). Changing the binding's
  vocabulary value re-keys those nodes. Per the 2026-06-11 Do-Not-Repeat lesson, any
  identity/MERGE-key change must grep every read-side Cypher — done below.

## Decisions to lock before executing

**D1 — what does `vocabulary: SNOMED` become?** (recommended: **A**)
- **A. Keep the field required; rename the value to a domain-governed sentinel**
  (e.g. `OMOP-Condition`). Minimal model change; the graph key stays a non-null string;
  reads that group by `vocabulary_name` keep working; the string now honestly names the
  governance scope, not a source vocabulary. Provenance id changes (expected).
- B. Make `ManifestVocabularyBinding.vocabulary` optional and drop it from the graph key,
  re-keying `:VocabularyBinding`/`:Term` on `(domain, resolver_policy_ref)`. Most correct
  long-term, but touches every read-side Cypher and the flip/supersede path — larger blast
  radius, defer to a dedicated graph-identity change.

**D2 — persistent store hygiene on rename.** (recommended: **wipe-and-note**)
- The `poc.duckdb` value-mapping store holds rows under the OLD policy ref. After the
  rename they are inert (bug-374/383), but dead. Recommend a one-line note + a
  `DELETE FROM value_mapping WHERE resolver_policy_ref = '<old>'` maintenance step run
  once against `~/.sema/poc.duckdb`, rather than adding migration code for a POC cache.

**D3 — golden-manifest hash: NO change needed.** Verified: `golden_manifest.yaml` is an
independent fixture (its own `SNOMED`/`GENDER_CV` bindings, no `omop.oncotree_to_snomed_condition`
ref). The repoint edits only `omop_condition_slice0.yaml`, so `golden_manifest_hash.txt` stays
valid. (The slice-0 manifest's own snapshot hash is computed, not pinned to that fixture.)

## Touch points (grepped)

Policy ref value `omop.oncotree_to_snomed_condition` → `omop.oncotree_condition`:
- `src/sema/resolve/policies/omop.py:18` — `OMOP_ONCOTREE_CONDITION_REF` (single source of truth).
- `src/sema/targets/manifests/omop_condition_slice0.yaml:71` + legacy-anchor comments (14-15, 62-67).
- 8 test files hardcode the literal (unit: test_value_mapping_store, test_mapping_report,
  test_transform_compiler, test_plan_assembler; integration: test_value_mapping_store_schema,
  test_staging_qa_live, test_mapping_report_live, test_transform_compiler_live). Prefer importing
  `OMOP_ONCOTREE_CONDITION_REF` over re-hardcoding, so a future rename is one-line. (Note:
  test_vocab_lookup_producer already imports the constant — the model to follow.)

Binding vocabulary value `SNOMED` → sentinel (D1-A):
- `src/sema/targets/manifests/omop_condition_slice0.yaml:67` (`vocabulary: SNOMED`) + comments.
- Graph key/id/provenance: `neo4j_writer_utils.py:265-305` (MERGE key + id string) — no code change,
  but the emitted id/keys change; assert new shape in `tests/unit/targets/test_neo4j_writer.py`.
- Read-side: `neo4j_writer_flip_utils.py:114` matches `:Term` on `vocabulary_name` — that is the
  Term's SOURCE vocabulary (`OncoTree`), NOT the binding's target, so it is unaffected by D1. Confirm.
- `tests/unit/targets/test_omop_condition_slice0_manifest.py` (binding assertions).

## Execution order (TDD, one atomic commit per step)

1. **D1/D2/D3 locked** (this doc; recommendations above stand unless overridden).
2. Rename the constant `OMOP_ONCOTREE_CONDITION_REF` value → `omop.oncotree_condition`; update
   the manifest ref. Update tests to import the constant instead of the literal where feasible.
   Run unit suite (expect only the pinned-literal tests to move).
3. Repoint the manifest `vocabulary:` value to the D1 sentinel; drop the "legacy anchor" comments
   (the field is now honest). Update binding/graph-writer shape tests. Regenerate any pinned hash.
4. Full `uv run pytest` + `uv run mypy src/sema/` + coverage ≥ 85%.
5. Live re-run: `uv run sema fit --backend databricks --study-schema cbioportal_msk_chord_2024
   --manifest ... --gold ... --strict` — expect exit 0, Gate D-lite PASS, conformance 0 violations,
   rows_staged == source count (proves the store-key change didn't fork staging).
6. One-off: `DELETE FROM value_mapping WHERE resolver_policy_ref='omop.oncotree_to_snomed_condition'`
   on `~/.sema/poc.duckdb` (D2).
7. Update `tasks/decision-slice0-omop-target.md` "Deferred" → done; log to `.wolf/`.

## Out of scope

- The full source `sema build` into Neo4j and the Gemini-3.5-Flash extraction-LLM switch
  (separate tracked next-session task) — the repoint is store/manifest/graph-key only.
- Producer #2 (LLM/fuzzy) and the human gold-labelling — the deterministic path stays
  gated by contract conformance, not gold.
