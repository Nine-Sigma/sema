# Third held-out fixture

Source: 2018 Standard Occupational Classification (SOC), U.S. Bureau of Labor
Statistics (Department of Labor).
URLs (linked from https://www.bls.gov/soc/2018/):
- https://www.bls.gov/soc/2018/soc_structure_2018.xlsx
- https://www.bls.gov/soc/2018/soc_2018_definitions.xlsx

License: U.S. Government work, public domain (17 U.S.C. 105).
Copy date: 2026-10-22. Downloaded by Dean in a browser on 2026-10-22 because
www.bls.gov returns HTTP 403 to scripted clients, copied unchanged. Files:

- `soc_structure_2018.xlsx` (51,367 bytes), sha256
  `ade08af40923266f3a854842e888ca3e93c15b26a147c20a2b12a61f4c4f4077`
- `soc_2018_definitions.xlsx` (143,843 bytes), sha256
  `f8b9ca7cccea78e30abfc9b96ab8a9dd2632cc8c8b57ee9b38a9b871e8fe2a89`

Commit: the commit that adds this file (test(intake): seal two fresh held-out
sources). It precedes every S4 commit that edits a rule, floor, lexicon entry,
predicate, prompt, or view.

Chosen by Dean on 2026-10-22 (S4 task 3.1,
`docs/architecture/s4/heldout-candidates.md`). Axes on which it differs from
every other fixture, read from the publisher's description only:

- Hierarchy encoded by code shape and column position (`11-0000` major,
  `11-1000` minor, `11-1010` broad, `11-1011` detailed; one of four level
  columns filled per row), not by a parent column (A2) or an edge list (A3,
  Athena).
- Header position: a multi-row title preamble above the header row in both
  workbooks (A2 and College Scorecard have the header on row 1; A10 on rows 2
  and 5).
- File layout: one code list split across two files, structure in one and
  definitions in the other, joined on the code.
- Domain: labor and occupations (no fixture is occupational).
- Definitions carry cross-references in prose ("Excludes ... (11-3021)",
  "Illustrative examples: ...") that name other codes inside text.

Gate thresholds (Dean, 2026-10-22, committed before any model-backed run):
`min_bound_fraction` 1.0, `max_open_choices` 2.

Seal baseline: recorded in task 3.4 (S3 code, one recording round).
