# Held-out fixture

Source: College Scorecard Data Dictionary, U.S. Department of Education.
URL: https://collegescorecard.ed.gov/files/CollegeScorecardDataDictionary.xlsx
(linked from https://collegescorecard.ed.gov/data/documentation/).
License: U.S. Government work, public domain (17 U.S.C. 105).
Copy date: 2026-10-20. File: `CollegeScorecardDataDictionary.xlsx` (770 KB),
copied unchanged.

Commit: 5d7afa8 (chore: add acris and held-out fixtures and the source-literal
guard). It precedes every commit that sets a floor, lexicon token, ignore
profile, or extractor pattern.

Axes on which it differs, read from the publisher's documentation and the
sheet list and first rows only:

- File layout: one workbook holding the dictionaries of several data files, one
  sheet per data file, beside README, ChangeLog, Glossary, and cohort-map
  sheets. A2 is one workbook with one sheet per section, A7 one CSV, A10 one
  workbook per entity plus code-table CSVs.
- Header position: dictionary sheets have their header on row 1 and the README
  sheet has no header at all. A10 headers are on rows 2 and 5.
- Inline code list separator: code lists are continuation rows in `VALUE` and
  `LABEL` columns of the dictionary sheet itself, not `code = label` text in a
  description (A10), not a separate values sheet (A2), not concept tables (A7).
- Key-value header sheet: none; README is free prose. A10 has one per workbook.
- Domain: U.S. higher education institutions and fields of study. A2 is a
  synthetic code library, A7 the OMOP clinical CDM, A10 NYC property records.

Landing: after the group 2 and group 3 fixes, the tabular loader lands the
workbook with 8 relations, one per sheet
(`tests/unit/intake/test_heldout_landing.py`). Before those fixes it failed
first at sheet ChangeLog (no row passed the header tests, the trailing empty
column case), then with bug-738 (a column name holding a double quote broke
the staging INSERT).
