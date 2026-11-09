# Intake: landing any ontology medium as relations

Intake reduces every medium to relations in DuckDB staging, works out from aggregates what shape those relations have and which column plays which role, and assembles a version 2 concept source spec. The deterministic importer (`sema.binding.importer`) turns that spec into a target manifest plus an authored-evidence sidecar. Land, Structure, Roles, and Dry run call no model: where code cannot decide, they list an ambiguity. The ingestion graph (see [The ingestion graph](#the-ingestion-graph)) hands those entries to a model within the members code proved, and then to a human at Review. Roles ranks members with the configured embedder, and Enrich calls the model through the same call path.

The stages run in this order, each with its own verifier and each writing to one run ledger:

| Stage | Entry point | Verifier |
|---|---|---|
| Land | a loader, below | `verify.land.verify_land(land, tools)` |
| Structure | `structure.run_structure(land, tools, provider=..., hint=None)` | `verify.structure.verify_structure(result, land, tools)` |
| Roles | `eligibility.run_roles(structure, profiles, tools, hint=None, embedder=None)`, then `eligibility.assemble_spec(descriptor, roles)` | `verify.roles.verify_roles(roles, tools, spec=spec)` |
| Enrich (optional) | `enrichment.enrich(spec, roles, tools, calls, ledger, enabled=False)`: engine stages B and C with domain layers off, written to the `enrichment` artifact; Import applies it with `enrichment.apply_enrichment(manifest, enrichment)`, filling only empty fields | none; it changes no structural decision |
| Dry run | `binding.importer.import_concept_source`, then `report.dry_run(spec, provider, staging, ledger)` | `verify.dry_run.verify_dry_run(spec, imported, structure=..., tools=...)` |

## The generality rule

The general code must handle a source nobody has seen. No module under `src/sema/intake/` outside `loaders/` may contain a relation name, sheet name, column name, header key, or phrase from any fixture. `tests/unit/intake/test_no_source_literals.py` enforces this by reading every workbook and CSV under `tests/fixtures/intake/` and scanning the general modules: a single-word literal is a hit when it is a whole string constant, a multi-word literal is a hit anywhere. Python keywords, role names read from the spec section models, single-word lexicon tokens read from `lexicon.py`, and Table Schema vocabulary read from the vendored schema are exempt.

A count in a test scenario is evidence that a general rule holds, never the rule. When a fixture and a rule disagree, the disagreement is recorded in the design document and the test stays a strict `xfail`; the rule changes only when a second source shows the same shape. `tests/fixtures/intake/heldout/` holds a public source that set no rule, and `tests/unit/intake/test_heldout_gate.py` runs it through every stage with no code change.

## The landing contract

Every loader writes relations through `Staging.write_table` into a schema the caller names, creates that schema when it is absent, and returns one `LandResult` (`sema.intake.landing`):

- `medium`: `tabular`, `semsql`, `dictionary`, `athena`, or `warehouse`.
- `source_identity`: what was landed, for example file names with SHA-256 digests, or an ontology IRI and version IRI.
- `relations`: one `LandedRelation` per relation, with its columns and types, row count, header row (tabular only), all-null columns, and a `Provenance` record: `source_uri`, `loader`, `loader_version`, `landed_at` (ISO 8601 UTC), `policy_class`, `locator_kind`, and `locator`.
- `headerless`: the tabular relations that landed with positional column names because no row passed the header tests.
- `schema_snapshot`: relation names and columns only, so two landings of one source compare equal.

Row-kind loaders (`locator_kind: row`) append a `_sema_locator` column to every row. The Athena loader keeps its tables byte-identical to the original showcase loader, so it adds no column and declares `locator_kind: relation` with the source file name as `locator`.

The warehouse loader calls `Connector.read_schema_metadata(catalog, schema)` once and fills each relation's columns, types, and comments from the returned objects; it writes nothing to staging, declares `locator_kind: relation` with `<catalog>.<schema>` as `locator`, and, because the catalog read carries no counts, counts each relation with one `SELECT COUNT(*)` through `ProfileTools.count_rows` at stage `land`, so the count is logged to the run ledger like every profiling statement. `land_warehouse` takes the run's `ledger` for that. An `Unreadable` result raises `UnreadableSchemaError` naming the connector's reason before any statement runs. A run descriptor's `relations` list is an allowlist: Land keeps only those objects from the catalog read before counting any row, writes the `land_scope` artifact (`selected`, `excluded`, `missing`), and raises `MissingRelationsError` before any statement when a listed relation is absent.

Landed relation names replace non-word characters with `_`. Column names keep the header text, with surrounding whitespace stripped and each inner whitespace run (line breaks included) collapsed to one space, so a spec names the column a human sees.

## Loaders

| Loader | Function | Relations | Locator |
|---|---|---|---|
| Tabular (CSV, XLSX, Parquet) | `loaders.tabular.land_tabular(source, staging, schema=...)` | one per sheet (`<file stem>__<sheet>`) or per file (`<file stem>`); `source` is a file, a directory (not recursive), or a glob | `<sheet>!<row>`: the physical row for XLSX and CSV (a CSV record that spans lines is located by its first line), the record number for Parquet |
| semsql (OWL, OBO, SKOS via a prebuilt semantic-sql SQLite file) | `loaders.semsql.land_semsql(path, staging, schema=...)` | `statements`, `entailed_edge`, `prefix`, the label, definition, synonym, mapping, deprecation, class, and edge views, plus `semsql_terms`, `semsql_edges`, `semsql_synonyms`, `semsql_mappings` | subject IRI (prefix for `prefix`) |
| Dictionary (PostgreSQL DDL) | `loaders.dictionary.land_ddl(path, staging, schema=...)` | `<file stem>__columns`: `entity`, `property`, `raw_type`, `nullable`, `ordinal` | `<file name>!<entity>.<property>` |
| Athena bundle | `loaders.athena.land_athena_bundle(bundle_dir, staging)` | `vocabulary_omop.concept`, `concept_relationship`, and the rest of the bundle | none; relation-level |
| Database file (DuckDB) | `loaders.database.land_database(path, staging, schema=None)` | every base table of every schema except `information_schema` and `pg_catalog`, as `<schema>.<table>` with the source names, types, rows, and comments unchanged; views do not land and `schema` renames nothing | none; relation-level, `<schema>.<table>` |
| Warehouse schema | `loaders.warehouse.land_warehouse(connector, catalog=..., schema=..., ledger=..., relations=...)` | every object the connector reports, or only the `relations` allowlist, as `<catalog>.<schema>.<object>`; nothing is copied | none; relation-level |

The graph's Land node picks the loader by medium through `loaders.detect.land_source(source, staging, schema=...)`: a SQLite header lands through semsql, a DuckDB header (`DUCK` at byte 8, under any file name) through the database loader, which attaches the file read-only, a directory holding every required Athena file through the Athena loader, a `.sql` file through the DDL dictionary loader, and any file the tabular loader reads through it. A source no loader reads raises `NoLoaderError` (a `ValueError`) naming it; the registry row is left at `land`, `failed`, `no_loader`. A warehouse is not a path and is never detected.

The tabular loader picks the header as the first of the first 50 rows where at least 80 percent of cells are non-empty, non-numeric strings, those strings are unique ignoring case, and none recurs in its column. The width of that test is the highest column index holding a non-empty cell anywhere in the window, so trailing empty columns do not lower the score. A sheet where no row passes lands with columns `_col_1` to `_col_n`, a null header row, and its name in `LandResult.headerless`; the other sheets still land. Merged cells are not expanded. A column is typed as a number only when every text cell survives the round trip, so `1.10`, `007`, and integers beyond 64 bits stay text. A header named `_sema_locator` fails the landing.

## Identifier quoting

`RelationProvider.quote(ref, identifier)` wraps an identifier in double quotes on `staging`, doubling an embedded double quote, and in backticks on `warehouse`, where an identifier containing a backtick is rejected. `RelationProvider.relation_sql(ref)` splits a dotted relation name on `.` and quotes each part. Every importer statement and every profiling statement builds its relation and column names through these two methods, so a column such as `Column Name`, `table`, or one holding a comma can be bound as a role. A spec's `filter` fragment is passed through verbatim, because the operator writes it as SQL.

## Profiling and evidence policy

`profile.ProfileTools(provider, EvidencePolicy(tier=...), ledger, stage=..., structure=None, land=None)` is the only way the Structure, Roles, and dry-run stages read landed data. `tier` is `metadata_only` (the intake default), `bounded_values`, or `full`; the type is `sema.models.config.EvidenceTier`, shared with `BuildConfig.evidence_policy` (build default `full`), and `EvidencePolicy.from_build_config` copies a build's tier and `max_distinct_values_read`. Every statement is written to the ledger's `sql.jsonl` before it runs, with its stage, provider, `value_bearing`, and `schema_relation` flags.

- `catalog` reads a staging relation through `duckdb_columns()` and a warehouse relation from `land`, issuing no statement; a warehouse relation missing from `land` is refused. `list_relations` lists a warehouse namespace from `land` the same way.
- `describe_relation` issues the catalog read above and one aggregate statement per relation (row count, non-null count, distinct count, and mean text length per column) and never reads a value.
- `check_join`, `check_hierarchy`, `count_distinct`, `count_where`, `check_levels`, and `check_paths` return aggregates only. A join check across providers is refused before any statement.
- `sample_values` returns values only when the policy allows them for that column.
- `read_schema_text` is the schema-relation rule: once Structure binds a relation as an `entity_model` dictionary or header, its rows describe another relation's schema, so its text may be read under any tier. The statement is logged with `value_bearing: true` and `schema_relation: true`. Any other relation, and any column in `policy.sensitive`, is refused. The audit rule under `metadata_only` is that every value-bearing statement in the Structure and Roles stages carries `schema_relation: true`.

## Structure

`run_structure` computes, in the fixed order of `structure.STEPS`:

1. **Stem groups.** Relations that share a `<file stem>` before `__` form one group. Stem groups are final; only step 5 can merge them.
2. **Profiles and signatures.** One `RelationProfile` per relation from `describe_relation`; a relation's signature is its sorted `(column, type)` pairs.
3. **Ignore.** `shapes.ignore` assigns one structural reason, in this order: `no_header` (the loader landed it header-less), `no_rows` (zero data rows), `boilerplate` (in a multi-relation group, its signature and row count equal those of a relation in at least `BOILERPLATE_GROUP_SHARE_FLOOR` of the other multi-relation groups, and every column resolves at 1.0 in both directions against each). The ignore binding's `would_match` slot lists the templates the relation would otherwise satisfy, so a reviewer can rebind it. Ignored relations leave the later steps.
4. **Candidates and precedence.** Each template in `shapes.TEMPLATES` proposes slot bindings from profiles and aggregate checks. Precedence (`shapes.precedence`) is structural and never reads a name: relations claimed by a loader-fixed binding (`semsql`, `athena_bundle`) leave every other template's candidate set first; a multi-relation binding (`concept_and_value_set`, `edge_list_hierarchy`, per-file `entity_model` with a header, and the two loader-fixed templates) then claims its relations, and single-relation candidates over a claimed relation are dropped. A dictionary with no header relation in its group yields a header-less per-file `entity_model` binding over that one relation; it claims nothing, so a relation that is also a `concept_list` stays ambiguous, because a lone dictionary and a code list have the same structure.
5. **Join graph.** Edges between id-signature columns (unique, no nulls, same type, names equal or one a suffix of the other) of unclaimed relations on one provider, resolving at or above `JOIN_RESOLVE_FLOOR`, merge groups by union-find, and the merged groups' candidates are recomputed. Edges within a group, and edges that touch a claimed relation, are kept as slot evidence and never merge.
6. **Binding.** A relation with exactly one candidate and one slot binding is bound with `source: code` and confidence 1.0. Everything else goes on the ambiguity list: two claims over one relation, more than one single-relation binding, or `no_template` when nothing matched.

**Described data is not a dictionary.** Before step 6 binds, a relation whose column names equal, after `lexicon.normalize` and with loader-added `_sema_` columns skipped, the values of a key column (unique, non-null text) of a per-file dictionary is the data that dictionary describes, so every `entity_model` candidate that uses it as a dictionary, header, or single relation is dropped. Any dictionary in the namespace counts, but only one that ends up bound: when a describing dictionary is left unbound, what it described returns to the candidates and binding runs again. The comparison reads no text: for each key column and candidate relation whose column count equals the key's distinct count, `ProfileTools.count_listed_names` issues one aggregate statement that normalizes the key values inside SQL (`name_sql_utils`, exact for printable ASCII and counting any other value apart) and returns only the distinct normalized values, how many are among the relation's normalized column names, and the values not compared; the relation is described when nothing is set apart and both counts equal its name count, so Structure still issues no value-bearing statement before binding. The Structure verifier applies the same rule over the bound dictionaries: a code-bound binding that uses described data is outside the eligibility set, and a pinned or proposed one fails `in_eligibility_set` with `described_by`.

A hint (a hand-written spec) pins the bindings it names as `source: human` before step 3. Each pinned binding still passes its template's `verify`; a failing or overlapping hint raises `HintRejectedError` and binds nothing.

`StructureResult` holds `order`, `groups` (every relation, ignored ones included, in landing order), `signatures`, `bound`, `ignored`, `ambiguities` (open entries only), `edges`, `floors` (every floor the stage used, so a moved floor is visible), and `proposal_targets` (the `no_template` relations and the relations ignored as `no_header`, in landing order; a `no_header` relation stays in the ignore partition while it is listed). `settled` keeps each entry a model or a reviewer settled, with its `choice`, `confidence`, `source` (`model` or `human`), and `call_key`; spec assembly checks the open list only, and Review presents both. `schema_relations` lists the dictionary and header relations of bound entity models. `to_shape()` renders the spec's `shape` list once every bound section is resolved, and raises before that. A loader-fixed binding (`semsql`, `athena_bundle`) is resolved when bound: its sections are the template module's `SECTIONS`, the sections its loader fills, read through `shapes.filled_sections` by both `to_shape()` and the Roles verifier.

## Roles

`run_roles` computes one `RoleEligibility` per role of each bound binding: the `members`, the hints consulted, `excluded_by`, the `choice`, its confidence, and its `source`. Members are every column of the relation, unless the role has a data fact (`eligibility_utils.FACTS`: `key_unique_non_null` for `concept_id` and `code`, `date_shape` for `valid_from` and `valid_to`, `array_shape` for `synonyms`), in which case they are the columns that meet it. Readings that fail a fixture-measured floor (`constant`, `display_null_rate`, `definition_mean_length`, `status_cardinality`) are findings on the column, never removals. Containment (`fk_entity`, `fk_property`, `replaced_by`) and value checks run in the verifier. Roles are settled in the fixed order of `lexicon.ROLES`. A role that no role model of its target requires is optional, and its members end with `none` (`lexicon.NO_REFERENCE`). A `none` choice leaves the role unbound: it never reaches the spec, a value read, or a column check. No role settles by member count. For each role:

1. A stated choice (hint, decision, or verified model answer) settles it.
2. Else the role map already settled in this run for the same template and relation signature settles it with `source: memory`, so identical sheets are decided once.
3. Else the lexicon claims it when exactly one member's name matches the role's tokens whole-token, after normalization (case fold, case changes split, punctuation removed, whitespace collapsed, trailing plural `s` stripped; English only). The claim is pre-filled in the role-map prompt; the model may contest it, and a contested claim opens the role.
4. Else an optional role whose every member another role holds settles `none` with `source: code`.
5. Else the role stays open and is asked in the role map. With an embedder, the rank-1 column per role and its margin are stated as a fact; the rank settles nothing.

The identity roles `title`, `vocabulary_id`, and `concept_code` are role-map fields like any other; a name match is a lexicon fact. A settled column leaves later roles' sets, recorded as `excluded_by`, unless the pair is in `lexicon.COINCIDING_ROLES`. A column that code or the lexicon settles for one role is also listed under `exclusions` of every other open role on the same side of the relation, whatever the order, so it is never offered; it does not narrow that role. A stated choice of such a column leaves the claim in force, so the verifier fails `one_role_per_column`. Every role that had more than one member before narrowing is listed as an ambiguity, settled or not, so the review sees what the hint decided.

For bound dictionaries, Roles reads the dictionary and header text through `read_schema_text` and computes three things from landed names: domains (a property whose text mentions a code list's relation stem), declared joins (each other entity the header text mentions yields a join on the properties both entities hold that the same sentence names; a mention with no such property is an ambiguity), and the section of each `concept_list` binding (`code_lists` when a dictionary names the relation as a domain, otherwise an ambiguity between `concepts` and `code_lists` with hint `no_reference`, which a stated `<relation>.section` choice settles and a later concept list with the same signature inherits with `source: memory`). A phrase matching `lexicon.REFERENCE_PHRASES` that names nothing landed is an `unresolved_reference` ambiguity.

`assemble_spec(descriptor, roles)` returns a spec only when every required role and every section is settled, and `None` otherwise. The `RunDescriptor` (`eligibility_utils`) supplies what no predicate can derive: `id`, `identity_system`, `release`, and optionally `scheme`, `term_storage`, `entity_case`, and `relations` (a warehouse Land allowlist). Fixed rules supply the rest: a code list's `vocabulary_id` is its relation's file stem, a per-file entity is named by its header `name` value or else its file stem, `roles.notes` is every dictionary text column bound to no other role, and declared joins and domains go into each entry's `foreign_keys` and `domains`. The A2 and ACRIS descriptors are `tests/fixtures/intake/tabular/codes_a2.run.yaml` and `tests/fixtures/intake/acris/acris.run.yaml`. The spec names entities by their source names; the importer (`sema.binding.identifier_utils`) keeps a name that is already a valid identifier and converts any other (NFKD to ASCII, lowercase, each run of other characters to `_`), keeps the source name in the entity's context-card `curated_synonyms`, and fails when two names convert to one identifier. A `YYYY-MM-DD` release becomes card version `YYYY.MM.DD`; any other release must be PEP 440.

## Verifiers

Every verifier returns a `Verdict` (`verify.base`): `status` (`pass` or `fail`), `failures` (each a check id, relation, optional column, and evidence), `findings` (each a `Finding` with a reading's name, subject, and value, which never fails a choice), and the floors it used. A verifier never retries; the S3 orchestrator owns the retry budget.

| Verifier | Checks |
|---|---|
| Land | `all_landed`, `row_counts_match`, `header_row_valid`, `types_parse`, `locator_present`, `policy_class_permitted` |
| Structure | `partition_complete` (every landed relation exactly once across bound, ignored, and ambiguous), `unknown_template`, `in_eligibility_set`, `template_verifies`, `precedence_applied`, `ignore_reason_known`, `ignore_profile_holds`, `ambiguity_list_complete` |
| Roles | `required_roles_bound`, `column_exists`, `in_eligibility_set` (the role's data fact, named in its evidence), `contained` (`fk_entity` and `fk_property` values in their target column, count-only, trimmed and case-folded, `verify.roles_contained`), `one_role_per_column` (a concept key and `name` may share a column when it is the `group_by` key and the counts show one value and no nulls), `section_resolved`, `ambiguity_list_complete`, `external_refs_resolve`, plus per-role value checks (key and code uniqueness, synonyms shape, status values, valid dates, `replaced_by` resolves, language tags); the display null rate is a finding |
| Dry run | `import_completed`, `counts_in_range`, `unique_active_code`, `manifest_parses`, `inline_under_cap`, `replaced_by_active` |

Under `metadata_only`, a check that needs values it may not read (for example external mappings with no loaded vocabulary) is listed as an `unverifiable` ambiguity rather than passed or failed. A `status` role whose values cannot be read under the policy stays bound with empty value lists and an `unverifiable` entry.

## Dry-run report

`report.dry_run(spec, provider, staging, ledger)` copies the spec with its term staging redirected to a scratch namespace (`<namespace>__dryrun_<first 8 characters of the run id>`), imports, and drops the scratch schema in a `finally` block. The `DryRunReport` holds concept and vocabulary counts, per-vocabulary term counts with rendered sample terms (`random.Random(REPORT_SEED)` over sorted codes, plus the longest labels), root count, maximum depth, orphan parents, active and retired counts, external reference resolve rates, per-entity property, foreign key, domain, and inline value counts, and the entities that have no obligation. It is written to the ledger as the `dry_run_report` artifact. Retired inline terms are counted but not rendered in the sample lists.

## Run ledger and registry

`ledger.Ledger(root).run(run_id)` returns a `RunLedger` over `<root>/<run_id>/`. The run id is a UUID4 string, usable as the S3 graph thread id.

- `event(stage, kind, payload)` appends to `events.jsonl`; `provenance(...)` is an event of kind `provenance`.
- `sql(stage, provider, statement, value_bearing, schema_relation)` appends to `sql.jsonl`.
- `artifact(name, model)` writes `artifacts/<name>.json` (or `.yaml`).
- `record_call(model, prompt, schema_name, output, tokens)` writes `calls/<sha256>.json`, keyed by SHA-256 over `model`, `prompt`, and `schema_name` joined by line breaks; `replay(model, prompt, schema_name)` returns the recorded output and logs a `replay` event on a hit, or returns `None`.

A prompt that misses both replay sources must fit `model_calls.MAX_PROMPT_TOKENS`, counted as the client's estimate over the prompt plus its output schema. The role-map prompt (`resolve_prompt_utils`) shows the first `SCHEMA_TEXT_SAMPLE_ROWS` (20) rows of each schema relation's text with a line counting the rows not shown, and compacts only an over-budget prompt, in a fixed order, each omission stated under "Omitted to fit the budget": comments are truncated to `COMMENT_CHARS`, then the schema sample shrinks to `SHRUNK_SCHEMA_ROWS` rows, then every column keeps only its name and type. The Structure prompt keeps its own ladder (`resolve_budget_utils`). A prompt still over budget raises `PromptBudgetError` before the call cap is checked: no provider call, no count, a `call_refused` event with the key and `prompt_tokens`, and the entries stay open with the failure as their reason.

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

It imports 487 terms in five vocabularies and 15 entities with 6 inline value lists. ACRIS states no primary key or required field, so no entity has an obligation. The declared foreign keys and domains reach the Table Schemas, the dry-run report, and the manifest, as each entity's `declared_joins` and `domains` with `evidence: declared`; `sema target project` turns the joins into `:JoinPath` edges.

`--staging` serves staging relations and receives term staging: a version 2 value set above `term_storage.inline_cap` (default 5000) is written to the spec's external namespace (default `vocabulary_<spec id>`) in the vocabulary store shape and declared `EXTERNAL` in the manifest. A spec that reads only staging relations never constructs a warehouse connector.

## The ingestion graph

`graph.py` runs one ingestion as a LangGraph graph with a SQLite checkpointer. The design fixes six stages: Land, Structure, Roles, Dry run, Review, and Import. The graph runs them as these nodes, in this order:

1. **Land** (`land`): lands the source through `loaders.detect.land_source`, or a warehouse schema through `land_warehouse`, then runs the Land verifier.
2. **Structure** (`structure`, `resolve_structure`, `propose_layouts`): `run_structure` binds what code can, the resolver hands open entries to the model, and layout proposals cover relations no template reads. The Structure verifier runs after the resolver and after the proposals.
3. **Roles** (`roles`, `resolve_roles`): `run_roles`, then the role resolver, then spec assembly and the Roles verifier. When no spec assembles, the run goes to Review.
4. **Dry run** (`dry_run`): `report.dry_run` and the dry-run verifier. When `--enrich` is set, the `enrich` node runs first.
5. **Review** (`review`): an interrupt. It writes the ambiguity lists and `run_metrics`, then presents the open structure entries, the settled structure entries with their source, the role entries, the layout proposals, the spec, and the dry-run report.
6. **Import** (`import`): imports the reviewed spec and writes `import/manifest.yaml` and `import/authored_evidence.json` under the run's ledger directory. It does not load the graph; `sema target load` and `sema target project` do that.

The graph has two interrupts:

- `review`: approve imports, reject ends the run, and a decisions file or an edited spec routes to the `decide` node. `decide` writes the `decisions` artifact when a decisions file was given, re-runs Structure and Roles with the decisions applied, and verifies every decided binding. A passing run with a spec dry-runs and returns to Review; any other result returns to Review directly.
- `value_ask`: a layout proposal or a Roles tool asked for values the evidence policy withholds. The payload lists each relation with its columns. Approval adds those `(relation, column)` pairs to the run's allowlist, raises a `metadata_only` run to `bounded_values` for those columns only (each still bounded by `max_distinct_values_read`), and re-enters the node that asked.

A failing verdict re-enters the node it followed. The third consecutive failure sets the reason `escalate` and routes to Review. When the call budget refuses a call inside a node, the run stops at Review with the reason `budget_exceeded`, whatever the verdict; the node's fixed successor (`resolve_structure`, `resolve_roles`, or `dry_run`) still runs first, with no further call.

Graph state holds scalars, names, and small records only (attempt counts, approved value pairs, decisions). A node never reads a stage object back from the ledger or a checkpoint: `graph_stages.Stages` recomputes every input from the landed staging file, and model calls inside the stages replay by key, so a recomputation reaches the provider only for a prompt the run has not recorded. The checkpointer only names the node to re-enter.

A ledger root holds `registry.sqlite`, `checkpoints.sqlite` (thread id is the run id), and one `<run_id>/` directory per run with `staging.duckdb`, `events.jsonl`, `sql.jsonl`, `calls/`, and `artifacts/`. The Python entry points are `discover`, `approve`, `reject`, `approve_values`, and `resume`. Each returns `RunOutcome(run_id, node, payload)`, where `node` is the interrupt the run stopped at, or `None` once the run ended. `resume` re-enters a run after a process exit at its pending node and logs a `resume` event. Pass the same `embedder` and `warehouse` to every call on one run; the state holds neither.

## Model calls

Every model call goes through `model_calls.ModelCalls.call(stage, prompt, output_model)`:

1. The key is `prompt_hash(model, prompt, output_model.__name__)`. Prompt versions ride inside the key: each prompt starts with a version line such as `# prompt: structure v1`, and each output model name carries a version suffix such as `StructureBatch_v1`. Prompts carry no run id, timestamp, path, or provider.
2. The run ledger's `calls/` is checked first, then `<recordings_dir>/calls/<key>.json`. The recordings directory is read-only. A recording hit is copied into the run ledger and logged as a `replay` event, so the run stays self-contained.
3. On a miss in both, the prompt must fit `MAX_PROMPT_TOKENS` (see [Run ledger and registry](#run-ledger-and-registry)).
4. The registry row must hold fewer calls than `max_calls` and a cost below `max_cost`. Otherwise the call is refused with `BudgetExceededError`: the registry row moves to `review`, `waiting_review`, `budget_exceeded`, and the caller leaves its entries open. Cost is known only after a call, so one call can overshoot the cost cap by its own cost.
5. One `invoke` is one call. Its tokens come from the client's `last_stats` and its cost from `prices[model]`, per 1,000 tokens. A model with no price counts at cost 0 and logs one warning. A failed invoke is counted once and logged as `call_failed` with its step errors; a successful one is recorded under `calls/` and logged as `call` with the client's fallback path.

The Structure resolver sends open entries in batches of at most 50 (`output_models.MAX_STRUCTURE_BATCH`). The Roles resolver asks one role map per relation signature (`prompts/roles_v2.md`, output model `RoleMap…_v1`): one free-text field per role of every role model the binding's sections use, section-qualified when a role repeats, answered with a column name or `none`. Relations with the same signature inherit a verified map with `source: memory`. Reference entries take one call per relation, in batches of at most `MAX_REFERENCE_BATCH` (62) entries, over the members code proved. The map is verified after the answer (`column_exists`, `one_role_per_column`, the data facts, containment); a failing answer is re-asked once with every failure listed (`resolve.DEFAULT_MAX_RETRIES`, shared with Structure and layout proposals), then the failing roles are left open with the model's choice ranked first and the failure as their reason. A finding never fails a choice; it is attached to the entry.

`IntakeModelConfig` reads the environment with the prefix `INTAKE_MODEL_`:

- `INTAKE_MODEL_PROVIDER`: required by `build_intake_client`, which the CLI runs at start, so a replay needs it too. Intake never falls back to `LLM_PROVIDER`.
- `INTAKE_MODEL_MODEL`: required, and part of every call key, so a replay must name the recorded model.
- `INTAKE_MODEL_MAX_CALLS` and `INTAKE_MODEL_MAX_COST`: required caps. `0` refuses every provider call.
- `INTAKE_MODEL_PRICES`: JSON, model id to `{"input_per_1k": ..., "output_per_1k": ...}`.
- `INTAKE_MODEL_RECORDINGS_DIR`: the read-only recordings directory.
- `INTAKE_MODEL_API_KEY`, `INTAKE_MODEL_BASE_URL`, `INTAKE_MODEL_USE_STRUCTURED_OUTPUT` (default `auto`), and `INTAKE_MODEL_REQUEST_TIMEOUT` (default 120 seconds).

The CLI also builds SEMA's embedder whenever `EMBEDDING_PROVIDER` is non-empty, and the CLI loads `.env` first. The embedding rank changes the Roles prompt, so a run with an embedder misses recordings made without one, and it calls the embedding provider. Under `INTAKE_MODEL_MAX_CALLS=0` with `INTAKE_MODEL_RECORDINGS_DIR` set, the CLI builds no embedder whatever `EMBEDDING_PROVIDER` says and logs that once, unless the recording's `recording.json` holds `"embeddings": true`. A replay follows its recording: a recording made with an embedder states that flag, and its replay needs the same `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` and calls the embedding provider, though never the model. With an embedder, the Roles prompt states each role's rank-1 column and margin; the rank settles no role. To replay with no provider call of any kind, set `EMBEDDING_PROVIDER=` to empty.

`scripts/record_intake_run.py` records one live run per committed source to `tests/fixtures/intake/recordings/<source>/`: the run's `calls/` beside `recording.json`, which holds the provider, model, prompt versions, calls, tokens, cost, and value asks. The script then replays the source with a cap of 0 and fails on any miss.

`metrics.run_metrics` writes the `run_metrics` artifact before every Review and after every Decide: `relations`, `ignored` by reason, `bound`, `bound_fraction`, `open_structure`, `open_roles`, `proposals_made`, `proposals_bound`, `model_calls`, `tokens`, `cost`, and `value_asks`. `bound_fraction` divides bound relations by landed relations minus ignored ones, except that a `no_header` relation stays in the denominator, so ignoring a real table cannot raise the fraction.

## Review decisions

A decisions file settles open entries with no model call. It is a YAML list of `{stage, subject, choice}`:

```yaml
- stage: structure
  subject: standard_library.data_element_mappings
  choice: ignore
```

Each decision is checked against the ambiguity lists Review wrote (`structure_ambiguities` and `roles_ambiguities`), before the graph runs, so a refused file changes nothing. A refusal raises `DecisionRejectedError` naming the subject, the choice, and the members. The rules:

- `choice` is a member of its entry, or `ignore`.
- A structure `ignore` may name any landed relation, listed or not. The relation is ignored with reason `human`. Ignoring some but not all relations of one binding is refused.
- A structure member binds that member's binding with `source: human` and confidence 1.0, and the entry moves from the open list to `settled`.
- `no_template` is not a choice. Review presents a `no_template` entry with `ignore` as its member.
- A roles decision enters Roles as a stated choice with `source: human`. `ignore` states `none`, which leaves the role unbound; a required role then fails the Roles verifier and the run stays at Review.

A decision never bypasses a verifier. A model's settled role choices are kept when decisions re-run Roles. `approve --spec` passes an edited spec instead or as well: alone it replaces the assembled spec, and beside decisions it overrides only the values it sets.

## Layout proposals and views

`proposal.propose_layouts` makes one model call per proposal target: a `no_template` relation, or one ignored as `no_header` (`StructureResult.proposal_targets`). A proposal names one template, its relations and slots, and optionally one view. It may also abstain, which is recorded as a `layout_finding` event, or answer `needs_values` with the columns it needs, which routes to the `value_ask` interrupt. Code applies the view, runs the template's `verify`, then the Structure verifier over the relations the binding spans. A passing proposal is bound with `source: model`; a failing one is re-asked with every earlier failure at most once, then escalated for Review.

A view turns a landed relation a template cannot read into a derived staging relation it can. `views.VIEWS` is the closed set a proposal may name; adding a view is adding a module with `ID`, `verify`, and `apply` and listing it there. `fill_down` is the one view: each null in the named columns takes the last non-null value above it in landed row order. It applies to staging relations with a `_sema_locator` column only, and it refuses a column whose first landed row is null (`fill_down_anchor`). The view lands as `<schema>.<table>_fill_down_<12 hex digits of the view spec's SHA-256>`. The bound entry records the view in `BoundEntry.view`; the binding names the landed relation, never the view.

## Enrichment

`--enrich` adds the `enrich` node between Roles and Dry run. `enrichment.enrich` runs SEMA's semantic engine stages B and C through the intake call path, with every domain layer and the inline examples off; stage A is never called, because the bound entity and its key role supply it. It writes the `enrichment` artifact and changes no structural decision. Import recomputes it and applies it with `apply_enrichment`, which fills a property's `definition`, `synonyms`, and `decoded_values` only where the source left them empty. A stated value always wins, and a semantic type is recorded but never applied. Enrichment prompts carry their own version line (`# prompt: enrich_stage_b v1`), so they replay like any other call.

## Tracing

`TracingConfig` reads `INTAKE_TRACING_BACKEND`: `none` (default) or `mlflow`. `langfuse` fails validation. With `mlflow`, every entry point (`discover`, `approve`, `reject`, `approve_values`, `resume`) opens one root span named after itself, `mlflow.langchain.autolog()` traces the LangChain runnables the client invokes, and every span carries `sema.run_id` and `sema.stage`. `mlflow` is imported only when that backend is configured; a missing package raises `TracingUnavailableError` before Land, naming the `tracing-mlflow` extra.

## Running an ingestion from the command line

The commands print JSON. `--root` names the ledger root and defaults to `.sema/intake`; keep it outside the repository.

`sema target discover SOURCE` starts a run and runs it to its first interrupt. `--descriptor` defaults to the single `*.run.yaml` beside the source. `--schema` defaults to `land_` plus the source stem as a relation name (non-word characters become `_`), lowercased; relation names appear in prompts, so a replay must use the schema its recording used. `--warehouse <catalog>.<schema>` replaces `SOURCE` and needs `--descriptor`. `--tier` (default `metadata_only`), `--hint`, `--enrich`, and `--run-id` are optional. The A2 fixture, replayed from its recording with no provider call:

```bash
export INTAKE_MODEL_PROVIDER=openrouter INTAKE_MODEL_MODEL=openai/gpt-6-luna \
  INTAKE_MODEL_MAX_CALLS=0 INTAKE_MODEL_MAX_COST=0 EMBEDDING_PROVIDER= \
  INTAKE_MODEL_RECORDINGS_DIR=tests/fixtures/intake/recordings/a2
uv run sema target discover tests/fixtures/intake/tabular/codes_a2.xlsx \
  --schema land_a2 --root /tmp/intake
```

It stops at `value_ask`, requesting `status` of `land_a2.codes_a2__Values`. The printout names the run id.

`sema target approve RUN_ID --approver NAME` answers the interrupt the run waits at, and refuses a run that waits elsewhere:

- `--values` approves the requested value reads. A run waiting at `value_ask` accepts nothing else.
- With no other option, it imports. A run with no assembled spec is refused.
- `--decisions FILE` settles open entries, and `--spec FILE` re-enters with an edited spec.
- `--reject` ends the run; it cannot be combined with `--spec`, `--decisions`, or `--values`.

Both commands write the run's spec as YAML to `<out>/<descriptor id>/spec.yaml`: `discover`, or an `approve` that returns the run to Review, writes the spec it assembled, and an `approve` that imports writes the spec it imported. `--out DIR` names the folder; it defaults to `specs` beside the ledger root, so `.sema/specs` for the default root `.sema/intake`. The file opens with `#` comment lines naming the run id, the run's status at the write (`waiting_review` or `imported`), and the source; the YAML below them loads with `load_concept_source_spec`, so you can edit it and pass it back with `--spec`. A later run of the same descriptor replaces the file, and the run ledger keeps its `spec` artifact. The printed JSON carries `spec_path`, the written file, or null when the run holds no assembled spec, in which case nothing is written. `.sema/intake/` is ignored by git; `.sema/specs/` is not, so a reviewed spec can be committed.

The A1 fixture reaches Review with open structure entries, and its committed decisions file settles them, with the same variables pointing at `tests/fixtures/intake/recordings/a1`:

```bash
uv run sema target discover tests/fixtures/intake/partner_sl/library.duckdb --root /tmp/intake
uv run sema target approve <run id> --approver <name> --root /tmp/intake \
  --decisions tests/fixtures/intake/partner_sl/decisions.yaml
uv run sema target approve <run id> --approver <name> --root /tmp/intake
```

The second command returns the run to Review with `open_entries: 0`; the third imports it. After the final approve, `<ledger root>/../specs/<descriptor id>/spec.yaml` holds the imported spec, so `/tmp/specs/partner-standard-library/spec.yaml` here.

`sema target runs` lists the registry newest first, filtered by `--status`, `--stage`, or `--source`. Each row holds the run's stage, status, reason, approver, `model_calls`, `tokens`, and `cost_estimate`; the cost is an estimate from the price table. `--metrics RUN_ID` prints the run's `run_metrics`:

```bash
uv run sema target runs --status waiting_review --root /tmp/intake
```

`sema target project --manifest FILE --binding FILE` projects a manifest's entities onto physical tables and writes the retrieval projection plus one `:JoinPath` per foreign key and declared join. Declared joins carry `source: manifest_declared` and confidence `DECLARED_JOIN_CONFIDENCE` (0.8). The binding is YAML naming `catalog`, `schema`, and `entity_tables`, a map from entity qualified name to table. `--writer` is `in-memory` (default, records statements only) or `neo4j`. The command reads the physical catalog through a Databricks cursor with either writer, so it needs a workspace.
