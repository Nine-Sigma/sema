# Standard Library intake input

The benchmark's automatic path runs `sema target discover` over a DuckDB file that holds only the Standard Library rows the Timbr POC workload needs. Two inputs feed it, and they are reported apart:

- **Supplied source projection.** `showcase/ontology_query_benchmark/intake_extract.py` selects rows by the 45 data element ids declared in the two CPD v2 product specs and writes `workload_projection.duckdb` plus `extract_receipt.json`. The selection uses the declared workload, so the receipt labels it `assistance.kind: supplied`. Report it as supplied assistance, never as automatic discovery.
- **Role-neutral descriptor.** `intake_standard_library.run.yaml` states locations only: the descriptor id, identity system, release, and the relation names. It holds no role map, binding, rule, authored spec or gold. Supplied CPD bindings and authored specs enter through the separately attributed assisted path.

Everything intake does with the file (Land, Structure, Roles, the dry run and any model call) is automatic.

## The projection

The extract reads `partner_library_test.standard_library` on TEST warehouse `feedface00000000`. It writes both relations under their source names, as `standard_library.<relation>`, with the source column names, the source column order, and the DuckDB form of each source type (`string` and `varchar(n)` to `VARCHAR`, `boolean` to `BOOLEAN`, `timestamp` to `TIMESTAMPTZ`, `array<string>` to `VARCHAR[]`; any other type stops the extract). Table comments carry over; the source columns have none.

| Relation | Predicate | Order | Rows (TEST, 2026-11-09) |
| --- | --- | --- | --- |
| `standard_library_data_element` | `data_element_prt_id` in the declared ids | `data_element_prt_id`, `permissible_value_prt_id`, `data_element_enriched_surrogate_id` | 2,788 (33 elements, 2,785 element and value pairs, 3 rows with no value) |
| `standard_library_concept` | `concept_prt_id` is a permissible value of a selected element, or a declared id | `concept_prt_id` | 2,750 (2,714 permissible values, 36 declared ids) |

The full relations hold 84,860 and 53,446 rows. The projection does not read `standard_library_concept_relationship` (15.1M rows); hierarchy facts stay supplied or missing for this recording. Bindings come from the supplied CPD governed input, so `data_element_mappings` and the permissible value mapping relations are not read either.

The extract issues 7 read-only statements: one column catalog read, one table catalog read, a count and an ordered `SELECT ... LIMIT row_limit + 1` per relation, and a second table catalog read. The run is complete only when each count is at most `--row-limit` (default 10,000), each select returns exactly the counted rows, and neither relation's `last_altered` moved between the two catalog reads. Otherwise the receipt lists each finding under `incomplete`, `complete` is false, no DuckDB file is written (a stale one is removed), and the command exits 1. Never pass an incomplete projection to discover.

The receipt records each statement and its returned rows, the counted and returned rows per relation, each relation's `updated_at`, comment and column types, the spec files and their sha256, the declared ids split into `found_as_data_element`, `found_as_concept_only` and `missing`, and the DuckDB file's sha256. On 2026-11-09 the split was 33, 3 (`PRTL2140007961`, `PRTL2140007963`, `PRTL2140008130`, typed `Permissible Value`) and 9 (`PRT4335813`, `PRTL2140007288`, `PRTL2140007289`, `PRT4181605`, `PRT4216292`, `PRT35918704`, `PRT35918637`, `PRT430549`, `PRTL2140009899`). Report the missing ids as missing coverage, not as discovery failures.

`release` in the descriptor is the refresh date both relations report (`updated_at` 2026-11-09 06:27 UTC). The views refresh in place; if the receipt's `updated_at` moved, update `release` before recording.

The descriptor's `relations` list is the warehouse Land allowlist. A file run ignores it; the projection file holds exactly those two relations.

## Evidence policy

Run with `--tier metadata_only`. The model sees relation and column names, types and aggregate counts, and no values. A run that needs values stops at `value_ask` and names each `(relation, column)`. Approve only Standard Library definition and code-label columns, with `sema target approve RUN_ID --values --approver NAME`. Approval raises only those columns to `bounded_values`, each capped at `max_distinct_values_read`, and the values come from the projection file. Neither relation holds patient rows, and no approval here authorizes demographic or cancer-registry patient data.

## Extract command

The extract connects with `WorkspaceClient(profile="TEST")`, refuses any host other than `https://partner-workspace.cloud.databricks.com`, and opens `databricks-sql` with that host and the profile's token. It never loads `.env`. Run it from the repository root:

```bash
RUN_ID=<run_id>
RUN=docs/poc/partner-timbr-benchmark/runs/$RUN_ID
HARNESS=/path/to/partner-harness
env -i HOME="$HOME" PATH="$PATH" PYTHON_DOTENV_DISABLED=1 \
  uv run python -m showcase.ontology_query_benchmark.intake_extract \
    --spec "$HARNESS/config/demographics_product_spec_cpd_v2.json" \
    --spec "$HARNESS/config/cancer_registry_product_spec_cpd_v2.json" \
    --out "$RUN/source"
SOURCE="$RUN/source/workload_projection.duckdb"
```

Check `"complete": true` and the row counts above in the printed summary before recording.

## Recording command

The `sema` root group calls `load_dotenv()`, which fills any variable the shell leaves unset from the nearest `.env`. Sema has no switch of its own. python-dotenv 1.2.2 (the locked version) skips loading when `PYTHON_DOTENV_DISABLED=1`; `tests/unit/ontology_query_benchmark/test_intake_config.py` pins that behavior. The token comes from the `TEST` profile in `~/.databrickscfg` and is never echoed. The command also starts from `env -i`, so no inherited `OPENROUTER_*`, `LLM_*`, `INTAKE_MODEL_*` or `EMBEDDING_*` value applies. `DATABRICKS_HOST` and `DATABRICKS_TOKEN` serve the model and embedding endpoints only; the run reads no warehouse.

```bash
TEST_TOKEN="$(python3 -c 'import configparser,os;c=configparser.ConfigParser();c.read(os.path.expanduser("~/.databrickscfg"));print(c["TEST"]["token"])')"
env -i HOME="$HOME" PATH="$PATH" \
  PYTHON_DOTENV_DISABLED=1 \
  DATABRICKS_HOST=https://partner-workspace.cloud.databricks.com \
  DATABRICKS_TOKEN="$TEST_TOKEN" \
  INTAKE_MODEL_PROVIDER=databricks \
  INTAKE_MODEL_MODEL=databricks-claude-sonnet-5 \
  INTAKE_MODEL_MAX_CALLS=40 \
  INTAKE_MODEL_MAX_COST=1 \
  INTAKE_TRACING_BACKEND=none \
  EMBEDDING_PROVIDER=databricks \
  EMBEDDING_MODEL=databricks-bge-large-en \
  uv run sema target discover "$SOURCE" \
    --descriptor showcase/ontology_query_benchmark/config/intake_standard_library.run.yaml \
    --tier metadata_only --run-id "$RUN_ID" \
    --root "$RUN/intake" --out "$RUN/specs"
```

Leave `INTAKE_MODEL_RECORDINGS_DIR`, `INTAKE_MODEL_BASE_URL` and `INTAKE_MODEL_API_KEY` unset. An empty `INTAKE_MODEL_RECORDINGS_DIR` parses as the current directory and switches agreement off. No price is configured for the serving endpoint, so each call counts at cost 0 with a warning, and the 40-call cap is the binding limit. Keep `INTAKE_MODEL_MAX_COST` above 0: a call proceeds only while the recorded cost is below the cap, so a cap of 0 refuses every live call.

## Zero-call replay

The live run writes its calls to `$RUN/intake/$RUN_ID/calls/`. Replay reads a recordings directory whose `recording.json` states the agreement flag the recording used; without it, replay asks once per call and misses every agreement key. Prepare the directory once:

```bash
mkdir -p "$RUN/recording"
cp -R "$RUN/intake/$RUN_ID/calls" "$RUN/recording/calls"
printf '{"source": "partner_sl_timbr_bench", "agreement": true}\n' > "$RUN/recording/recording.json"
```

Then replay over the same projection file, changing only the model caps and the embedder:

```bash
env -i HOME="$HOME" PATH="$PATH" \
  PYTHON_DOTENV_DISABLED=1 \
  DATABRICKS_HOST=https://partner-workspace.cloud.databricks.com \
  DATABRICKS_TOKEN="$TEST_TOKEN" \
  INTAKE_MODEL_PROVIDER=databricks \
  INTAKE_MODEL_MODEL=databricks-claude-sonnet-5 \
  INTAKE_MODEL_MAX_CALLS=0 \
  INTAKE_MODEL_MAX_COST=0 \
  INTAKE_MODEL_RECORDINGS_DIR="$RUN/recording" \
  INTAKE_TRACING_BACKEND=none \
  EMBEDDING_PROVIDER= \
  uv run sema target discover "$SOURCE" \
    --descriptor showcase/ontology_query_benchmark/config/intake_standard_library.run.yaml \
    --tier metadata_only --run-id "$RUN_ID-replay" \
    --root "$RUN/replay" --out "$RUN/replay-specs"
```

Replay makes no model, embedding or warehouse call. Prompts carry the projection's counts, so replay over a different projection file moves call keys; a stale key fails the replay at the call cap rather than calling the provider. Compare the DuckDB sha256 in `extract_receipt.json` with the file you replay over.
