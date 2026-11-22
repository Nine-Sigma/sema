# ACRIS fixture (A10)

Source: NYC Open Data (`data.cityofnewyork.us`), Automated City Register
Information System (ACRIS) datasets published by the NYC Department of Finance.
Terms: NYC Open Data Terms of Use (public data, free to use and redistribute).

Copy date: 2026-10-20. Copied flat from `docs/poc/acris/` with file names
preserved: 15 data dictionary workbooks and 5 code-table CSVs.

Catalog query (Socrata Discovery API):

    https://api.us.socrata.com/api/catalog/v1?domains=data.cityofnewyork.us&q=ACRIS&limit=50

The 15 results named "ACRIS - ..." are the datasets below; the other results of
that query are not ACRIS and were not copied.

Record tables (data dictionary attachment `*_Data_Dictionary.xlsx`):

| Dataset | Id |
|---|---|
| ACRIS - Real Property Master | bnx9-e6tj |
| ACRIS - Real Property Legals | 8h5j-fqxa |
| ACRIS - Real Property Parties | 636b-3b5g |
| ACRIS - Real Property References | pwkr-dpni |
| ACRIS - Real Property Remarks | 9p4w-7npp |
| ACRIS - Personal Property Master | sv7x-dduq |
| ACRIS - Personal Property Legals | uqqa-hym2 |
| ACRIS - Personal Property Parties | nbbg-wtuz |
| ACRIS - Personal Property References | 6y3e-jcrc |
| ACRIS - Personal Property Remarks | fuzi-5ks9 |

Code tables (data dictionary attachment and a CSV export of the rows):

| Dataset | Id |
|---|---|
| ACRIS - Document Control Codes | 7isb-wh4c |
| ACRIS - Property Types Codes | 94g4-w6xz |
| ACRIS - Country Codes | j2iz-mwzu |
| ACRIS - State Codes | 5c9e-33xj |
| ACRIS - UCC Collateral Codes | q9kp-jvxv |
