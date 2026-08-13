# Labelling worksheet — 2026-08-11-staging

- **snapshot_version**: `2026-08-11-staging` — pre-filled in every
  row. Do not edit it: it is what binds these labels to this snapshot.
- **target vocabulary**: `SNOMED`
- **target domain**: `Condition`
- **vocabulary release**: `omop-vocab-2024` — a `gold_concept_id` is
  meaningless without it, so a concept id from another release is not a label
  for this gold set.
- **rows to label**: 62

## Filling a row

1. `gold_label` is one of: RESOLVED, NO_MAP. Leave it blank for a row you have not
   answered — `UNLABELLED` is a scaffold state, not a curator's answer.
2. `RESOLVED` needs BOTH `gold_concept_id` and `target_concept_code` (the
   durable SNOMED code, which survives a release change),
   and the concept must be standard and in the Condition domain.
3. `NO_MAP` is a positive claim, not a blank: it carries NO target concept and
   still requires `evidence` saying what you looked for and why nothing fits.
   A wrong NO_MAP scores directly against a correctly-mapped code.
4. `curator`, `review_date` and `evidence` are the annotation floor — every
   label needs all three.
5. `second_reviewer` must be someone OTHER than the curator; your own name
   there adjudicates nothing.

No candidate targets are pre-filled and the rows are interleaved: both
deliberate, so the oracle stays independent of the resolver being graded.
