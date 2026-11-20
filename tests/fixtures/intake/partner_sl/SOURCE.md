# Standard library fixture (A1, offline)

Every row in `library.duckdb` is synthesized by `build_fixture.py`. The fixture
holds no partner data values: no patient data, no rows copied from any partner system,
no credentials, and no hostnames.

Rebuild: `uv run python tests/fixtures/intake/partner_sl/build_fixture.py`.

## What comes from the design, and from where

Names only, all recorded as public metadata in
`docs/architecture/concept-binding-slice-design.md`:

- Relation names and column names: section 5.2 (the spec block), with the
  corrections from the section 10 live log: `domain_prt_id` (the 5.2 filter),
  `defines_ancestry` and `unified_dataset_location` (section 10, real relation
  and column names), and `mapped_permissible_value_prt_ids` as `ARRAY<string>`
  (section 10 live import; 5.2 names a scalar that does not exist).
- The twelve data element ids and labels of the Race domain: section 10,
  sibling set.
- The domain id `PRTD2140001103` and Race's crosswalk codes (Athena 4013886,
  SNOMED 103579009): section 10.

## What is synthesized

Permissible values (ids `SYNPV*`, labels, titles, synonyms), definitions, data
types, the PHI flag, crosswalk ids other than Race's, relationship rows,
column bindings (dataset labels, ids, column lists), and value mappings. The
mapping statuses are taken from the closed list section 5.2 records.

## Files

- `library.duckdb`: five relations in schema `standard_library`.
- `standard_library.spec.yaml`: the hand-written concept source spec, read from
  staging; the oracle a no-hint run must import equal to.
- `standard_library.run.yaml`: the run descriptor (values no predicate derives).
- `dataset_map.yaml`: synthesized dataset map; one label resolves, one does not.
