# cBioPortal → OMOP showcase

End-to-end demo of the sema pipeline:

1. Ingest a cBioPortal study (TCGA) into a local DuckDB staging file
2. Push staged tables to a Databricks workspace
3. Run the staged A→B→C L2 pipeline (with healthcare domain + few-shot) against the Databricks catalog
4. Compare the produced assertions against a dev-slice or holdout baseline

## Layout

- `parsers.py` — cBioPortal source parsers (clinical, MAF, SV, CNA, gene panel matrix, resources, timelines)
- `cbioportal_utils.py` — download/type/IO helpers for `parsers.py`
- `slices/dev_slice.yaml` — 13-table dev slice for prompt tuning
- `slices/dev_slice_poc.yaml` — 12-table subset matching the current Databricks POC ingest
- `slices/holdout.yaml` — 10-table held-out slice for bias checks

## Run from a source checkout

```bash
# Step 1 — stage cBioPortal study into DuckDB
uv run sema ingest cbioportal gbm_tcga_pan_can_atlas_2018 \
    --cache-dir ~/.cache/sema/cbioportal \
    --duckdb-path ./poc.duckdb

# Step 2 — push to Databricks (requires DATABRICKS_* env)
uv run sema push --target databricks --duckdb-path ./poc.duckdb

# Step 3 — run the staged L2 pipeline against the catalog
uv run sema build \
    --catalog workspace --schemas cbioportal_omop \
    --domain healthcare \
    --table-workers 1 --skip-embeddings --verbose

# Step 4 — evaluate against a slice
uv run sema eval run \
    --slice showcase/cbioportal_to_omop/slices/dev_slice_poc.yaml \
    --label post-showcase-refactor \
    --output-dir eval-runs/post-showcase-refactor
```

## Packaging note

The `showcase/` directory is not part of the installable `sema` package. It's
importable from a source checkout (e.g. `uv run pytest`, `uv run sema ingest
cbioportal`) because the project root is on `sys.path`. If you `pip install
sema` without the source checkout, `sema ingest cbioportal` will fail with a
helpful message — use the generic `sema ingest` primitives or write your own
adapter against the `SourceParser` protocol instead.
