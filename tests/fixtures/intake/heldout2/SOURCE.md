# Second held-out fixture

Source: American Community Survey (ACS) Public Use Microdata Sample (PUMS) Data
Dictionary 2024, U.S. Census Bureau.
URL: https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.csv
License: U.S. Government work, public domain (17 U.S.C. 105).
Copy date: 2026-10-21. File: `PUMS_Data_Dictionary_2024.csv` (329,514 bytes,
6,092 lines), copied unchanged.

Commit: the commit that adds this file (test(intake): seal a second held-out
source and record run metrics). It precedes every file under
`src/sema/intake/prompts/` and `src/sema/intake/views/`.

Chosen by Dean on 2026-10-21 (design section 14). Axes on which it differs
from every other fixture, read from the publisher's description only:

- Producer: a federal statistical agency's survey microdata dictionary. A2 is a
  synthetic code library, A3 a semsql ontology, A7 the OMOP clinical CDM, A10
  NYC property records, the first held-out source a higher-education workbook.
- File layout: one headerless CSV. Every other tabular fixture has a header
  row (A10 on rows 2 and 5, the first held-out source on row 1).
- Record-type column: the first field says what each row is, `NAME` rows for
  527 variables and `VAL` rows for 5,565 code values, so variables and their
  values interleave in one relation with different row widths.
- Value ranges: `VAL` rows can state a range of codes, not one code per row.

S2 baseline (no model, `metadata_only`, no hint), recorded in
`tests/unit/intake/test_heldout_gate.py`: the tabular loader lands 1 relation
with 6,091 rows and one data column whose name is the whole first line; it does
not mark the relation headerless. Structure binds it as one `concept_list`
(partition and Structure verdict pass). The Land verdict fails
`header_row_valid`: DuckDB `sniff_csv` returns delimiter `|` for this ragged
CSV and the tabular loader's `_csv_dialect` accepts it. Roles lists 1 open
choice and assembles no spec.

Post-fix (task 1.8, measured once after the tabular loader stopped accepting a
sniffed delimiter that splits no row), recorded beside the S2 baseline in
`tests/unit/intake/test_heldout_gate.py`: the loader splits the CSV on a
delimiter that splits its rows, lands 1 relation with positional columns, and
marks it headerless. The Land verdict passes. Structure ignores the relation as
`no_header` (partition and Structure verdict pass), so 0 relations are bound
(bound fraction 0.0), Roles lists 0 open choices, and no spec is assembled.
Run metrics: relations 1, ignored `{no_header: 1}`, bound 0, open structure 0,
open roles 0, proposals 0/0, model calls 0, tokens 0, cost 0.0, value asks 0.
