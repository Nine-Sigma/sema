# Fourth held-out fixture

Source: Standard Geographical Classification (SGC) 2021, classification
structure, Statistics Canada.
URLs (linked from https://www.statcan.gc.ca/en/subjects/standard/sgc/2021/index
and its French counterpart):
- https://www.statcan.gc.ca/en/statistical-programs/document/sgc-cgt-2021-structure-eng.csv
- https://www.statcan.gc.ca/fr/programmes-statistiques/document/sgc-cgt-2021-structure-fra.csv

License: Statistics Canada Open Licence,
https://www.statcan.gc.ca/en/reference/licence.
Copy date: 2026-10-22. Files, copied unchanged:

- `sgc-cgt-2021-structure-eng.csv` (239,897 bytes, 5,474 lines), sha256
  `55f6eb915415b4aa911c6fce3cbaced641c4ed9af023bebc414b2929721a62d2`
- `sgc-cgt-2021-structure-fra.csv` (283,562 bytes, 5,474 lines), sha256
  `6e310e740c67770d3cc060d3b41079a827adf2edfb229a84745fc8fa5e67085a`

Commit: the commit that adds this file (test(intake): seal two fresh held-out
sources). It precedes every S4 commit that edits a rule, floor, lexicon entry,
predicate, prompt, or view.

Chosen by Dean on 2026-10-22 (S4 task 3.1,
`docs/architecture/s4/heldout-candidates.md`); sealed as the English and
French pair by Dean's decision of the same day. Axes on which it differs from
every other fixture, read from the publisher's description only:

- Two relations with one schema in two languages, a cross-file language axis
  no fixture has. The publisher ships each language as a separate file, so the
  bilingual axis runs across relations, not across columns.
- Non-ASCII text: French titles; every other fixture is ASCII English.
- Hierarchy by an explicit level column plus code-prefix nesting (2, 4, 7
  digits), a third hierarchy shape beside A2's parent column and A3's edges.
- Domain: administrative geography (no fixture is geographic).
- Producer and jurisdiction: a Canadian federal agency.

Gate thresholds (Dean, 2026-10-22, committed before any model-backed run):
`min_bound_fraction` 1.0, `max_open_choices` 3.

Seal baseline: recorded in task 3.4 (S3 code, one recording round).
