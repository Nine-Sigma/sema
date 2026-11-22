# OMOP CDM v5.4 fixtures

Fetched once on 2026-10-20 from the OHDSI CommonDataModel repository, tag `v5.4`, at the URLs in `showcase/cbioportal_to_omop/omop_ingest_utils.py`:

- `OMOPCDM_postgresql_5.4_ddl.sql`: https://raw.githubusercontent.com/OHDSI/CommonDataModel/v5.4/inst/ddl/5.4/postgresql/OMOPCDM_postgresql_5.4_ddl.sql
- `OMOP_CDMv5.4_Field_Level.csv`: https://raw.githubusercontent.com/OHDSI/CommonDataModel/v5.4/inst/csv/OMOP_CDMv5.4_Field_Level.csv

Committed so tests run offline. The DDL tests the dictionary loader alone; acceptance A7 uses only the field-level CSV.
