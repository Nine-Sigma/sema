# Intake: landing any ontology medium as relations

Intake reduces every medium to relations in DuckDB staging. A version 2 concept source spec then names those relations and their column roles, and the deterministic importer (`sema.binding.importer`) turns them into a target manifest plus an authored-evidence sidecar. Nothing in this package calls a model.

## The landing contract

Every loader writes relations through `Staging.write_table` into a schema the caller names, creates that schema when it is absent, and returns one `LandResult` (`sema.intake.landing`):

- `medium`: `tabular`, `semsql`, `dictionary`, or `athena`.
- `source_identity`: what was landed, for example file names with SHA-256 digests, or an ontology IRI and version IRI.
- `relations`: one `LandedRelation` per relation, with its columns and types, row count, header row (tabular only), all-null columns, and a `Provenance` record: `source_uri`, `loader`, `loader_version`, `landed_at` (ISO 8601 UTC), `policy_class`, `locator_kind`, and `locator`.
- `schema_snapshot`: relation names and columns only, so two landings of one source compare equal.

Row-kind loaders (`locator_kind: row`) append a `_sema_locator` column to every row. The Athena loader keeps its tables byte-identical to the original showcase loader, so it adds no column and declares `locator_kind: relation` with the source file name as `locator`.

Landed relation names replace non-word characters with `_`, because the importer builds unquoted SQL from spec identifiers. Column names keep the header text, so a header with a space or an SQL keyword (`table`, `column`) cannot be bound as a role yet.

## Loaders

| Loader | Function | Relations | Locator |
|---|---|---|---|
| Tabular (CSV, XLSX, Parquet) | `loaders.tabular.land_tabular(source, staging, schema=...)` | one per sheet (`<file stem>__<sheet>`) or per file (`<file stem>`); `source` is a file, a directory, or a glob | `<sheet>!<row>`: the physical row for XLSX and CSV (a CSV record that spans lines is located by its first line), the record number for Parquet |
| semsql (OWL, OBO, SKOS via a prebuilt semantic-sql SQLite file) | `loaders.semsql.land_semsql(path, staging, schema=...)` | `statements`, `entailed_edge`, `prefix`, the label, definition, synonym, mapping, deprecation, class, and edge views, plus `semsql_terms`, `semsql_edges`, `semsql_synonyms`, `semsql_mappings` | subject IRI (prefix for `prefix`) |
| Dictionary (PostgreSQL DDL) | `loaders.dictionary.land_ddl(path, staging, schema=...)` | `<file stem>__columns`: `entity`, `property`, `raw_type`, `nullable`, `ordinal` | `<file name>!<entity>.<property>` |
| Athena bundle | `loaders.athena.land_athena_bundle(bundle_dir, staging)` | `vocabulary_omop.concept`, `concept_relationship`, and the rest of the bundle | none; relation-level |

The tabular loader picks the header as the first of the first 50 rows where at least 80 percent of cells are non-empty, non-numeric strings, those strings are unique ignoring case, and none recurs in its column. Merged cells are not expanded. A column is typed as a number only when every text cell survives the round trip, so `1.10`, `007`, and integers beyond 64 bits stay text. A header named `_sema_locator` fails the landing.

## Naming a staging relation in a spec

A relation is a string, which resolves to the spec's `default_provider` (`warehouse` unless set), or an object naming the provider:

```yaml
concept_source:
  spec_version: 2
  default_provider: staging
  concepts:
    relation: land_a2.codes_a2__Concepts          # staging, from default_provider
  mappings:
    relation: {provider: warehouse, name: cat.schema.sssom}
```

Sections filtered to the concept scope (`value_sets`, the two details sections, `relationships`, `authored_bindings`, `value_mappings`) must use the same provider as `concepts`. `mappings` is joined in memory and may use either.

## Running an import over landed relations

Land first, then import with `--staging` pointing at the same DuckDB file. The A2 fixture, from the repository root:

```bash
uv run python -c "
from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.tabular import land_tabular
with Staging('/tmp/a2.duckdb', schemas=()) as staging:
    land_tabular('tests/fixtures/intake/tabular/codes_a2.xlsx', staging, schema='land_a2')
"
uv run sema target import \
  --spec tests/fixtures/intake/tabular/codes_a2.spec.yaml \
  --staging /tmp/a2.duckdb \
  --out /tmp/a2.manifest.yaml \
  --evidence-out /tmp/a2.evidence.json
```

`--staging` serves staging relations and receives term staging: a version 2 value set above `term_storage.inline_cap` (default 5000) is written to the spec's external namespace (default `vocabulary_<spec id>`) in the vocabulary store shape and declared `EXTERNAL` in the manifest. A spec that reads only staging relations never constructs a warehouse connector.
