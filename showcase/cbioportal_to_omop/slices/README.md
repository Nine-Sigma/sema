# cBioPortal Eval Slices

YAML slice configs that select tables from a study's namespaced staging
schema for the eval runner.

## Slice files

| File | Schema | Role |
|---|---|---|
| `dev_slice.yaml` | `cbioportal_brca_tcga_pan_can_atlas_2018` (post-Phase 2) | Original BRCA dev slice |
| `dev_slice_poc.yaml` | `cbioportal_brca_tcga_pan_can_atlas_2018` (post-Phase 2) | BRCA POC regression-guard slice — small, cheap, run on every change |
| `holdout.yaml` | `cbioportal_brca_tcga_pan_can_atlas_2018` (post-Phase 2) | Original BRCA holdout |
| `msk_chord_dev.yaml` | `cbioportal_msk_chord_2024` | MSK CHORD 2024 dev slice (12 tables) — overlaps with few-shots intentionally |
| `msk_chord_holdout.yaml` | `cbioportal_msk_chord_2024` | MSK CHORD 2024 holdout (9 tables) — disjoint from few-shot sources + dev |
| `contamination_map.yaml` | n/a | Lists every table referenced by a few-shot example. Holdouts MUST NOT include any of these. |

## Contamination policy

A table that appears in any few-shot prompt example becomes
"contaminated" — the LLM has been shown its expected output, so its
performance on that table is not unbiased. Holdout slices are designed to
measure pipeline generalization and therefore MUST exclude every
contaminated table.

`scripts/check_slice_contamination.py` enforces this:

```bash
uv run python scripts/check_slice_contamination.py \
  --holdout showcase/cbioportal_to_omop/slices/msk_chord_holdout.yaml \
  --against showcase/cbioportal_to_omop/slices/contamination_map.yaml \
  --against showcase/cbioportal_to_omop/slices/msk_chord_dev.yaml
```

The check runs in CI; PRs that contaminate a holdout fail.

When adding a new few-shot example to
`src/sema/engine/few_shot_healthcare_stage_{a,b,c}.py`, add the example's
`table_name` to `contamination_map.yaml` if it isn't already there.

## BRCA POC regression guard

`dev_slice_poc.yaml` is intentionally retained as a small, cheap slice
that catches regressions on the patterns the original
`source-semantic-hardening` change established. Run it before/after any
change touching prompts, materializer, or graph loader.
