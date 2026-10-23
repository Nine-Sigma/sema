# Fifth held-out fixture

Source: FDIC BankFind Suite field definitions, Federal Deposit Insurance
Corporation (FDIC).
URLs (catalogued in https://www.fdic.gov/data.json; the double slash is the
catalogued form):
- https://api.fdic.gov/banks/docs//institutions_definitions.csv
- https://api.fdic.gov/banks/docs//locations_definitions.csv

License: US federal government public data; licence not stated on the download
page (the FDIC `data.json` catalog carries no `license` field for these
datasets, checked 2026-10-23).
Copy date: 2026-10-23. Fetched with `curl` on 2026-10-23: HTTP 200,
`content-type: text/csv; charset=UTF-8`, `last-modified: Thu, 08 Oct 2026
16:36:44 GMT` for both. Files:

- `institutions_definitions.csv` (35,237 bytes), sha256
  `c857ed5b0a09bffebb4a140f0f2f225bb8a6eaa449aa9c9b3875615c27ad28ac`;
  about 152 rows (approximate, from the pre-download fetch)
- `locations_definitions.csv` (5,995 bytes), sha256
  `9e902fa965e8bc2fc79173ac40418a0ad2eaeb36ad9837c5721ac159e6013f73`;
  about 52 rows, about 22 of them continuation rows (approximate, from the
  pre-download fetch)

Summary of Deposits definitions: looked for in the FDIC `data.json` catalog on
2026-10-23 and not found (no distribution names a Summary of Deposits
definitions CSV), so the fixture is the two files above.

Fetched blind: no agent has opened these files (rule 13).

added at commit: <filled at task 1.5 from git log>

Chosen on 2026-10-23 (S5 design decision 24,
`docs/architecture/s5/heldout-candidates.md`). Axes on which it differs from
every other fixture, read from the publisher's catalog and the candidates
document only:

- Producer: FDIC, not used.
- Domain: bank supervision, not used.
- Layout: one file per entity with different header vocabularies across files
  (`Variable Name` against `NAME`), where A7 is one file for many entities and
  A10's per-file dictionaries share one header.
- Inline code list separator: codes embedded in definition prose as
  `code='label'` (institutions) and as continuation rows under an empty key
  (locations); no fixture carries the first form, and College Scorecard carries
  the second from another producer.
- Key-value header sheet: none.
- Size: about 200 rows in total, the smallest dictionary fixture.

Coverage-matrix rows covered: "A data dictionary (A7, A10, College Scorecard)"
and "A codebook with continuation rows (College Scorecard)".

Gate thresholds (supervisor, 2026-10-23, committed before any model-backed
run): `min_bound_fraction` 1.0, `max_open_choices` 4.

Seal baseline: recorded in S5 task 1.6 (S3 code, one source, cap 20 calls,
$0.03).
