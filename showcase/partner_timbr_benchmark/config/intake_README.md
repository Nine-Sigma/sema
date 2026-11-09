# Standard Library intake input

`intake_standard_library.run.yaml` is the role-neutral discovery input for the benchmark's automatic path. It states locations only: the descriptor id, identity system, release, and the relation allowlist. It holds no role map, binding, rule, authored spec or gold. Supplied CPD bindings and authored specs enter through the separately attributed assisted path.

## Selected relations

The allowlist selects 2 of the 95 relations in `partner_library_test.standard_library`. Land reads the catalog once, keeps only these two, and records the rest as excluded in the run's `land_scope` artifact before any row count.

| Relation | Rows (TEST, 2026-11-09) | Why it is selected |
| --- | --- | --- |
| `standard_library_data_element` | 84,860 | Data elements with titles, definitions, data types and their permissible values, one row per element and value. Holds 33 of the 45 data element ids in the two CPD v2 product specs. |
| `standard_library_concept` | 53,446 | Concepts with titles, definitions, vocabulary codes and Athena ids. All 2,714 permissible values of those 33 elements resolve here. |

These relations are excluded on purpose:

- `standard_library_concept_relationship` (15,148,770 rows, 34 relationship types, 13.3M of them OMOP `Maps to`/`Mapped from`). The pre-Review dry run imports every selected relation in full, so selecting it would pull all 15.1M rows at discover time. Intake has no row predicate. Hierarchy facts stay supplied or missing for this recording.
- `data_element_mappings` and the permissible value mapping relations. Bindings come from the supplied CPD governed input, not from automatic discovery.

12 workload data element ids are not data elements in this release. `PRTL2140007961`, `PRTL2140007963` and `PRTL2140008130` exist only as concepts typed `Permissible Value`. `PRT4335813`, `PRTL2140007288`, `PRTL2140007289`, `PRT4181605`, `PRT4216292`, `PRT35918704`, `PRT35918637`, `PRT430549` and `PRTL2140009899` are absent from both selected relations. Report them as missing coverage, not as discovery failures.

`release` is the refresh date that both selected materialized views report in TEST catalog metadata (`updated_at` 2026-11-09 06:27 UTC). The views refresh in place, so re-read `updated_at` before recording and update `release` if it moved.

## Evidence policy

Run with `--tier metadata_only`. The model sees relation and column names, types and aggregate counts, and no values. A run that needs values stops at `value_ask` and names each `(relation, column)`. Approve only Standard Library definition and code-label columns of the two selected relations, with `sema target approve RUN_ID --values --approver NAME`. Approval raises only those columns to `bounded_values`, each capped at `max_distinct_values_read`. Neither relation holds patient rows, and no approval here authorizes demographic or cancer-registry patient data.

## Recording command

The `sema` root group calls `load_dotenv()`, which fills any variable the shell leaves unset from the nearest `.env`. Sema has no switch of its own. python-dotenv 1.2.2 (the locked version) skips loading when `PYTHON_DOTENV_DISABLED=1`; `tests/unit/partner_timbr_benchmark/test_intake_config.py` pins that behavior. The token comes from the `TEST` profile in `~/.databrickscfg` and is never echoed. The command also starts from `env -i`, so no inherited `OPENROUTER_*`, `LLM_*`, `INTAKE_MODEL_*` or `EMBEDDING_*` value applies. Run it from the repository root:

```bash
RUN_ID=<run_id>
RUN=docs/poc/partner-timbr-benchmark/runs/$RUN_ID
TEST_TOKEN="$(python3 -c 'import configparser,os;c=configparser.ConfigParser();c.read(os.path.expanduser("~/.databrickscfg"));print(c["TEST"]["token"])')"
env -i HOME="$HOME" PATH="$PATH" \
  PYTHON_DOTENV_DISABLED=1 \
  DATABRICKS_HOST=https://partner-workspace.cloud.databricks.com \
  DATABRICKS_TOKEN="$TEST_TOKEN" \
  DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/feedface00000000 \
  INTAKE_MODEL_PROVIDER=databricks \
  INTAKE_MODEL_MODEL=databricks-claude-sonnet-5 \
  INTAKE_MODEL_MAX_CALLS=40 \
  INTAKE_MODEL_MAX_COST=1 \
  INTAKE_TRACING_BACKEND=none \
  EMBEDDING_PROVIDER=databricks \
  EMBEDDING_MODEL=databricks-bge-large-en \
  uv run sema target discover \
    --warehouse partner_library_test.standard_library \
    --descriptor showcase/partner_timbr_benchmark/config/intake_standard_library.run.yaml \
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

Then replay with the same env form, changing only the model caps and the embedder:

```bash
env -i HOME="$HOME" PATH="$PATH" \
  PYTHON_DOTENV_DISABLED=1 \
  DATABRICKS_HOST=https://partner-workspace.cloud.databricks.com \
  DATABRICKS_TOKEN="$TEST_TOKEN" \
  DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/feedface00000000 \
  INTAKE_MODEL_PROVIDER=databricks \
  INTAKE_MODEL_MODEL=databricks-claude-sonnet-5 \
  INTAKE_MODEL_MAX_CALLS=0 \
  INTAKE_MODEL_MAX_COST=0 \
  INTAKE_MODEL_RECORDINGS_DIR="$RUN/recording" \
  INTAKE_TRACING_BACKEND=none \
  EMBEDDING_PROVIDER= \
  uv run sema target discover \
    --warehouse partner_library_test.standard_library \
    --descriptor showcase/partner_timbr_benchmark/config/intake_standard_library.run.yaml \
    --tier metadata_only --run-id "$RUN_ID-replay" \
    --root "$RUN/replay" --out "$RUN/replay-specs"
```

Replay makes no model or embedding call. It still reads the TEST warehouse: Land re-reads the catalog and counts rows, profiling issues aggregates, and the dry run imports the two relations. Prompts carry those counts, so a refresh that changes them moves call keys. A stale key fails the replay at the call cap rather than calling the provider. Compare the two runs' `land_scope` artifacts and `sql.jsonl` counts to tell a moved snapshot from a code change.
