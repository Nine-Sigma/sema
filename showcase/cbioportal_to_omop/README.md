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

## Multi-study workflow (post-namespacing, 2026-04-24)

Each cBioPortal `study_id` produces its own DuckDB schema and own Databricks
schema, named `cbioportal_<sanitized_study_id>`. The ingest path sanitizes,
records the schema in `_sema_study_registry` (DuckDB), and is idempotent on
re-ingest. Push (default) discovers schemas from the registry plus the
known shared schemas (`ontology_omop`, `vocabulary_omop`); scratch schemas
in DuckDB are NOT published unless `--discover-all-schemas` is set.

```bash
# Step 1a — stage two cBioPortal studies into DuckDB (each lands in its own schema)
PYTHONPATH=. uv run sema ingest cbioportal gbm_tcga_pan_can_atlas_2018 \
    --cache-dir ~/.sema/cache/cbioportal \
    --duckdb-path ~/.sema/poc.duckdb
PYTHONPATH=. uv run sema ingest cbioportal msk_chord_2024 \
    --cache-dir ~/.sema/cache/cbioportal \
    --duckdb-path ~/.sema/poc.duckdb

# Step 1b — OMOP CDM + vocabulary
uv run sema ingest omop --vocab-path ~/data/omop/athena_2026_04

# Step 2 — push to Databricks. Default mode discovers from
# `_sema_study_registry` ∪ {ontology_omop, vocabulary_omop}; both study
# schemas land alongside the shared ontology/vocab schemas.
uv run sema push --target databricks --duckdb-path ~/.sema/poc.duckdb

# Step 3 — run the staged L2 pipeline against ONE study's catalog
uv run sema build \
    --catalog workspace --schemas cbioportal_msk_chord_2024 \
    --domain healthcare --table-workers 1 --skip-embeddings --verbose

# Step 4 — evaluate against a slice (slice file references the namespaced schema)
uv run sema eval run \
    --slice showcase/cbioportal_to_omop/slices/msk_chord_dev.yaml \
    --label baseline-A \
    --output-dir eval-runs/msk-chord-baseline-A
```

### Scoped re-build per study

Each `sema build --schemas X` run begins with a scoped-delete that removes
every relationship stamped with `source_schema = X` plus `:Assertion` and
`:JoinPath` nodes whose `source_schema = X`. Other studies' assertions and
provenance edges are untouched; shared concept and physical nodes are
never deleted. Re-running BRCA(GBM) does not affect MSK CHORD's slice of
the graph, and vice versa.

### Legacy schema deprecation

The flat `workspace.cbioportal` schema (pre-2026-04-24) is **deprecated**
and tagged with a `COMMENT ON SCHEMA` describing the migration. Its
contents (GBM TCGA Pan-Can Atlas 2018) live at
`workspace.cbioportal_gbm_tcga_pan_can_atlas_2018` post-migration. The
flat schema will be dropped in a follow-up change after a one-milestone
deprecation window. To migrate a local DuckDB staging file, run:

```bash
uv run python scripts/migrate_cbioportal_to_namespaced.py \
    --duckdb-path ~/.sema/poc.duckdb \
    --study-id gbm_tcga_pan_can_atlas_2018
```

## Packaging note

The `showcase/` directory is not part of the installable `sema` package. It's
importable from a source checkout (e.g. `uv run pytest`, `uv run sema ingest
cbioportal`) because the project root is on `sys.path`. If you `pip install
sema` without the source checkout, `sema ingest cbioportal` will fail with a
helpful message — use the generic `sema ingest` primitives or write your own
adapter against the `SourceParser` protocol instead.
