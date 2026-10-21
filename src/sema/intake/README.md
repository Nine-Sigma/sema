# Intake: landing any ontology medium as relations

Intake reduces every medium to relations in DuckDB staging, works out from aggregates what shape those relations have and which column plays which role, and assembles a version 2 concept source spec. The deterministic importer (`sema.binding.importer`) turns that spec into a target manifest plus an authored-evidence sidecar. Nothing in this package calls a model: where code cannot decide, it lists an ambiguity for a human (or, from S3, a model) to resolve.

The stages run in this order, each with its own verifier and each writing to one run ledger:

| Stage | Entry point | Verifier |
|---|---|---|
| Land | a loader, below | `verify.land.verify_land(land, tools)` |
| Structure | `structure.run_structure(land, tools, provider=..., hint=None)` | `verify.structure.verify_structure(result, land, tools)` |
| Roles | `eligibility.run_roles(structure, profiles, tools, hint=None, embedder=None)`, then `eligibility.assemble_spec(descriptor, roles)` | `verify.roles.verify_roles(roles, tools, spec=spec)` |
| Dry run | `binding.importer.import_concept_source`, then `report.dry_run(spec, provider, staging, ledger)` | `verify.dry_run.verify_dry_run(spec, imported, structure=..., tools=...)` |

## The generality rule

The general code must handle a source nobody has seen. No module under `src/sema/intake/` outside `loaders/` may contain a relation name, sheet name, column name, header key, or phrase from any fixture. `tests/unit/intake/test_no_source_literals.py` enforces this by reading every workbook and CSV under `tests/fixtures/intake/` and scanning the general modules: a single-word literal is a hit when it is a whole string constant, a multi-word literal is a hit anywhere. Python keywords, role names read from the spec section models, single-word lexicon tokens read from `lexicon.py`, and Table Schema vocabulary read from the vendored schema are exempt.

A count in a test scenario is evidence that a general rule holds, never the rule. When a fixture and a rule disagree, the disagreement is recorded in the design document and the test stays a strict `xfail`; the rule changes only when a second source shows the same shape. `tests/fixtures/intake/heldout/` holds a public source that set no rule, and `tests/unit/intake/test_heldout_gate.py` runs it through every stage with no code change.

## The landing contract

Every loader writes relations through `Staging.write_table` into a schema the caller names, creates that schema when it is absent, and returns one `LandResult` (`sema.intake.landing`):

- `medium`: `tabular`, `semsql`, `dictionary`, or `athena`.
- `source_identity`: what was landed, for example file names with SHA-256 digests, or an ontology IRI and version IRI.
- `relations`: one `LandedRelation` per relation, with its columns and types, row count, header row (tabular only), all-null columns, and a `Provenance` record: `source_uri`, `loader`, `loader_version`, `landed_at` (ISO 8601 UTC), `policy_class`, `locator_kind`, and `locator`.
- `headerless`: the tabular relations that landed with positional column names because no row passed the header tests.
- `schema_snapshot`: relation names and columns only, so two landings of one source compare equal.

Row-kind loaders (`locator_kind: row`) append a `_sema_locator` column to every row. The Athena loader keeps its tables byte-identical to the original showcase loader, so it adds no column and declares `locator_kind: relation` with the source file name as `locator`.

Landed relation names replace non-word characters with `_`. Column names keep the header text, with surrounding whitespace stripped and each inner whitespace run (line breaks included) collapsed to one space, so a spec names the column a human sees.

## Loaders

| Loader | Function | Relations | Locator |
|---|---|---|---|
| Tabular (CSV, XLSX, Parquet) | `loaders.tabular.land_tabular(source, staging, schema=...)` | one per sheet (`<file stem>__<sheet>`) or per file (`<file stem>`); `source` is a file, a directory (not recursive), or a glob | `<sheet>!<row>`: the physical row for XLSX and CSV (a CSV record that spans lines is located by its first line), the record number for Parquet |
| semsql (OWL, OBO, SKOS via a prebuilt semantic-sql SQLite file) | `loaders.semsql.land_semsql(path, staging, schema=...)` | `statements`, `entailed_edge`, `prefix`, the label, definition, synonym, mapping, deprecation, class, and edge views, plus `semsql_terms`, `semsql_edges`, `semsql_synonyms`, `semsql_mappings` | subject IRI (prefix for `prefix`) |
| Dictionary (PostgreSQL DDL) | `loaders.dictionary.land_ddl(path, staging, schema=...)` | `<file stem>__columns`: `entity`, `property`, `raw_type`, `nullable`, `ordinal` | `<file name>!<entity>.<property>` |
| Athena bundle | `loaders.athena.land_athena_bundle(bundle_dir, staging)` | `vocabulary_omop.concept`, `concept_relationship`, and the rest of the bundle | none; relation-level |

The tabular loader picks the header as the first of the first 50 rows where at least 80 percent of cells are non-empty, non-numeric strings, those strings are unique ignoring case, and none recurs in its column. The width of that test is the highest column index holding a non-empty cell anywhere in the window, so trailing empty columns do not lower the score. A sheet where no row passes lands with columns `_col_1` to `_col_n`, a null header row, and its name in `LandResult.headerless`; the other sheets still land. Merged cells are not expanded. A column is typed as a number only when every text cell survives the round trip, so `1.10`, `007`, and integers beyond 64 bits stay text. A header named `_sema_locator` fails the landing.

## Identifier quoting

`RelationProvider.quote(ref, identifier)` wraps an identifier in double quotes on `staging`, doubling an embedded double quote, and in backticks on `warehouse`, where an identifier containing a backtick is rejected. `RelationProvider.relation_sql(ref)` splits a dotted relation name on `.` and quotes each part. Every importer statement and every profiling statement builds its relation and column names through these two methods, so a column such as `Column Name`, `table`, or one holding a comma can be bound as a role. A spec's `filter` fragment is passed through verbatim, because the operator writes it as SQL.

## Profiling and evidence policy

`profile.ProfileTools(provider, EvidencePolicy(tier=...), ledger, stage=..., structure=None)` is the only way the Structure, Roles, and dry-run stages read landed data. `tier` is `metadata_only` (the intake default), `bounded_values`, or `full`. Every statement is written to the ledger's `sql.jsonl` before it runs, with its stage, provider, `value_bearing`, and `schema_relation` flags.

- `describe_relation` issues one catalog read and one aggregate statement per relation (row count, non-null count, distinct count, and mean text length per column) and never reads a value.
- `check_join`, `check_hierarchy`, `count_distinct`, `count_where`, `check_levels`, and `check_paths` return aggregates only. A join check across providers is refused before any statement.
- `sample_values` returns values only when the policy allows them for that column.
- `read_schema_text` is the schema-relation rule: once Structure binds a relation as an `entity_model` dictionary or header, its rows describe another relation's schema, so its text may be read under any tier. The statement is logged with `value_bearing: true` and `schema_relation: true`. Any other relation, and any column in `policy.sensitive`, is refused. The audit rule under `metadata_only` is that every value-bearing statement in the Structure and Roles stages carries `schema_relation: true`.

## Structure

`run_structure` computes, in the fixed order of `structure.STEPS`:

1. **Stem groups.** Relations that share a `<file stem>` before `__` form one group. Stem groups are final; only step 5 can merge them.
2. **Profiles and signatures.** One `RelationProfile` per relation from `describe_relation`; a relation's signature is its sorted `(column, type)` pairs.
3. **Ignore.** `shapes.ignore` assigns one structural reason, in this order: `no_header` (the loader landed it header-less), `no_rows` (zero data rows), `boilerplate` (in a multi-relation group, its signature and row count equal those of a relation in at least `BOILERPLATE_GROUP_SHARE_FLOOR` of the other multi-relation groups, and every column resolves at 1.0 in both directions against each). The ignore binding's `would_match` slot lists the templates the relation would otherwise satisfy, so a reviewer can rebind it. Ignored relations leave the later steps.
4. **Candidates and precedence.** Each template in `shapes.TEMPLATES` proposes slot bindings from profiles and aggregate checks. Precedence (`shapes.precedence`) is structural and never reads a name: relations claimed by a loader-fixed binding (`semsql`, `athena_bundle`) leave every other template's candidate set first; a multi-relation binding (`concept_and_value_set`, `edge_list_hierarchy`, per-file `entity_model`, and the two loader-fixed templates) then claims its relations, and single-relation candidates over a claimed relation are dropped.
5. **Join graph.** Edges between id-signature columns (unique, no nulls, same type, names equal or one a suffix of the other) of unclaimed relations on one provider, resolving at or above `JOIN_RESOLVE_FLOOR`, merge groups by union-find, and the merged groups' candidates are recomputed. Edges within a group, and edges that touch a claimed relation, are kept as slot evidence and never merge.
6. **Binding.** A relation with exactly one candidate and one slot binding is bound with `source: code` and confidence 1.0. Everything else goes on the ambiguity list: two claims over one relation, more than one single-relation binding, or `no_template` when nothing matched.

A hint (a hand-written spec) pins the bindings it names as `source: human` before step 3. Each pinned binding still passes its template's `verify`; a failing or overlapping hint raises `HintRejectedError` and binds nothing.

`StructureResult` holds `order`, `groups` (every relation, ignored ones included, in landing order), `signatures`, `bound`, `ignored`, `ambiguities`, `edges`, and `floors` (every floor the stage used, so a moved floor is visible). `schema_relations` lists the dictionary and header relations of bound entity models. `to_shape()` renders the spec's `shape` list once every bound section is resolved, and raises before that.

## Roles

`run_roles` computes one `RoleEligibility` per role of each bound binding: the `members` that satisfy the role's predicate, the hints consulted, `excluded_by`, the `choice`, its confidence, and its `source`. Roles are settled in the fixed order of `lexicon.ROLES`. For each role:

1. One member settles it with `source: code`.
2. Else the role map already settled in this run for the same template and relation signature settles it with `source: memory`, so identical sheets are decided once.
3. Else the lexicon settles it when exactly one member's name matches the role's tokens whole-token, after normalization (case fold, case changes split, punctuation removed, whitespace collapsed, trailing plural `s` stripped; English only).
4. Else the embedding rank settles it when an embedder and a margin floor are given and the margin reaches the floor. `EMBEDDING_MARGIN_FLOOR` is unset in S2, so this step runs only when a caller passes a floor.
5. Else the role stays open.

A settled column leaves later roles' sets, recorded as `excluded_by`, unless the pair is in `lexicon.COINCIDING_ROLES`. Every role that had more than one member before narrowing is listed as an ambiguity, settled or not, so the review sees what the hint decided.

For bound dictionaries, Roles reads the dictionary and header text through `read_schema_text` and computes three things from landed names: domains (a property whose text mentions a code list's relation stem), declared joins (each other entity the header text mentions yields a join on the properties both entities hold that the same sentence names; a mention with no such property is an ambiguity), and the section of each `concept_list` binding (`code_lists` when a dictionary names the relation as a domain, otherwise an ambiguity between `concepts` and `code_lists` with hint `no_reference`). A phrase matching `lexicon.REFERENCE_PHRASES` that names nothing landed is an `unresolved_reference` ambiguity.

`assemble_spec(descriptor, roles)` returns a spec only when every required role and every section is settled, and `None` otherwise. The `RunDescriptor` (`eligibility_utils`) supplies what no predicate can derive: `id`, `identity_system`, `release`, and optionally `scheme`, `term_storage`, and `entity_case`. Fixed rules supply the rest: a code list's `vocabulary_id` is its relation's file stem, a per-file entity is named by its header `name` value or else its file stem, `roles.notes` is every dictionary text column bound to no other role, and declared joins and domains go into each entry's `foreign_keys` and `domains`. The A2 and ACRIS descriptors are `tests/fixtures/intake/tabular/codes_a2.run.yaml` and `tests/fixtures/intake/acris/acris.run.yaml`.

## Verifiers

Every verifier returns a `Verdict` (`verify.base`): `status` (`pass` or `fail`), `failures` (each a check id, relation, optional column, and evidence), and the floors it used. A verifier never retries; the S3 orchestrator owns the retry budget.

| Verifier | Checks |
|---|---|
| Land | `all_landed`, `row_counts_match`, `header_row_valid`, `types_parse`, `locator_present`, `policy_class_permitted` |
| Structure | `partition_complete` (every landed relation exactly once across bound, ignored, and ambiguous), `unknown_template`, `in_eligibility_set`, `template_verifies`, `precedence_applied`, `ignore_reason_known`, `ignore_profile_holds`, `ambiguity_list_complete` |
| Roles | `required_roles_bound`, `column_exists`, `in_eligibility_set`, `one_role_per_column`, `section_resolved`, `ambiguity_list_complete`, `external_refs_resolve`, plus per-role value checks (key and code uniqueness, display null rate, synonyms shape, status values, valid dates, `replaced_by` resolves, language tags) |
| Dry run | `import_completed`, `counts_in_range`, `unique_active_code`, `manifest_parses`, `inline_under_cap`, `replaced_by_active` |

Under `metadata_only`, a check that needs values it may not read (for example external mappings with no loaded vocabulary) is listed as an `unverifiable` ambiguity rather than passed or failed.

## Dry-run report

`report.dry_run(spec, provider, staging, ledger)` copies the spec with its term staging redirected to a scratch namespace (`<namespace>__dryrun_<first 8 characters of the run id>`), imports, and drops the scratch schema in a `finally` block. The `DryRunReport` holds concept and vocabulary counts, per-vocabulary term counts with rendered sample terms (`random.Random(REPORT_SEED)` over sorted codes, plus the longest labels), root count, maximum depth, orphan parents, active and retired counts, external reference resolve rates, per-entity property, foreign key, domain, and inline value counts, and the entities that have no obligation. It is written to the ledger as the `dry_run_report` artifact. Retired inline terms are counted but not rendered in the sample lists.

## Run ledger and registry

`ledger.Ledger(root).run(run_id)` returns a `RunLedger` over `<root>/<run_id>/`. The run id is a UUID4 string, usable as the S3 graph thread id.

- `event(stage, kind, payload)` appends to `events.jsonl`; `provenance(...)` is an event of kind `provenance`.
- `sql(stage, provider, statement, value_bearing, schema_relation)` appends to `sql.jsonl`.
- `artifact(name, model)` writes `artifacts/<name>.json` (or `.yaml`).
- `record_call(model, prompt, schema_name, output, tokens)` writes `calls/<sha256>.json`, keyed by SHA-256 over `model`, `prompt`, and `schema_name` joined by line breaks; `replay(model, prompt, schema_name)` returns the recorded output and logs a `replay` event on a hit, or returns `None`.

`registry.Registry(path)` is one SQLite table, `runs`, kept beside the ledger and written through the standard library `sqlite3` module so it stays readable while a run holds the staging file. `register(run_id, source=..., stage=...)` starts a row as `running`; `advance(run_id, stage, status)` moves it (`running`, `waiting_review`, `approved`, `rejected`, `failed`, `complete`); `count_call(run_id, tokens, cost)` adds to its totals; `list(status=None, stage=None, source=None)` filters. The ledger does not reference the registry, and neither imports LangGraph or LangChain.

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

## Per-file dictionaries and code lists

A version 2 spec needs at least one of `concepts`, `entity_model`, or `code_lists`.

`entity_model.layout: per_file` describes one dictionary relation per entity, each with an optional key-value header relation. The entity is named by the header's `name` value, so `roles.entity` is rejected under `per_file` (and required under the default `single_relation`). `roles.type` is optional; a field with no type is `any`. `roles.notes` lists further text columns whose prose feeds the Table Schema. Each entry may carry `foreign_keys` and `domains`, which the roles stage derives from prose and a human may edit:

```yaml
  entity_model:
    layout: per_file
    roles:
      property: Column Name
      description: Column Description
      notes: ["Term, Acronym, or Code Definitions"]
    header_keys: {name: Dataset Name, description: Dataset Description, text: []}   # optional
    entities:
      - relation: land_acris.ACRIS___Real_Property_Legals_Data_Dictionary__Column_Info
        header: land_acris.ACRIS___Real_Property_Legals_Data_Dictionary__Dataset_Info
        foreign_keys:
          - {columns: [Document ID, Good Through Date], entity: 'ACRIS - Real Property Master',
             reference_columns: [Document ID, Good Through Date]}
        domains:
          Property Type: Property_Types_Codes
```

The header relation is read as key-value pairs, with its two column names as the first pair, because the tabular loader lands the first passing row as the header. Keys are matched after normalization; a key with no value is kept as a null. `header_keys.text` names prose keys explicitly. Without `header_keys`, `name` and `description` are found by the lexicon's header tokens, the first match in reading order wins, a key bound to `name` is not also `description`, and every other key is prose.

`code_lists` declares a vocabulary that has no concept relation. Each entry names a `vocabulary_id`, a `relation`, `roles` (the value set roles without `concept_id`), an optional `concept`, and an optional `filter`; duplicate vocabulary ids are rejected:

```yaml
  code_lists:
    - vocabulary_id: Country_Codes
      relation: land_acris.Country_Codes
      roles: {code: COUNTRY CODE, display: DESCRIPTION}
```

`code = label` lists written inside a description or note are extracted by the versioned patterns in `lexicon.PATTERNS` (at least two pairs from one text; texts are never merged). They land in the Table Schema as `x-sema-values`, with the extractor version as `x-sema-values-extractor`, and in the manifest as the property's `decoded_values`.

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

The ACRIS directory (15 dictionary workbooks and 5 code-table CSVs, 65 relations) with its hand-written per-file spec:

```bash
uv run python -c "
from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.tabular import land_tabular
with Staging('/tmp/acris.duckdb', schemas=()) as staging:
    land_tabular('tests/fixtures/intake/acris', staging, schema='land_acris')
"
uv run sema target import \
  --spec tests/fixtures/intake/acris/acris.spec.yaml \
  --staging /tmp/acris.duckdb \
  --out /tmp/acris.manifest.yaml \
  --evidence-out /tmp/acris.evidence.json
```

It imports 487 terms in five vocabularies and 15 entities with 6 inline value lists. ACRIS states no primary key or required field, so no entity has an obligation and the declared foreign keys and domains reach the Table Schemas and the dry-run report but not the manifest (design Decision 15, open).

`--staging` serves staging relations and receives term staging: a version 2 value set above `term_storage.inline_cap` (default 5000) is written to the spec's external namespace (default `vocabulary_<spec id>`) in the vocabulary store shape and declared `EXTERNAL` in the manifest. A spec that reads only staging relations never constructs a warehouse connector.
