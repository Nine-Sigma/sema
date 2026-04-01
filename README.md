
<h1 align="center">Sema</h1>

<p align="center">
  <strong>Ontology extraction for knowledge graphs — semantic layer for GraphRAG</strong>
</p>

<p align="center">
  <em>From Greek σῆμα — "sign" or "meaning"</em>
</p>

<p align="center">
  <img src="assets/sema-banner.png" alt="Sema" width="100%" />
</p>


Sema automatically extracts semantic ontology from any data warehouse, builds a knowledge graph, and serves it as a structured context layer for downstream consumers — NL2SQL engines, AI agents, data catalogs, lineage tools, and more.

```
Warehouse Metadata → LLM Interpretation → Knowledge Graph → Hybrid Search (GraphRAG) → Consumer
                           ↑
                   External Enrichment
                   (Atlan, etc.) [optional]
```

---

## What It Does

Sema reads your warehouse catalog and produces a **Semantic Context Object (SCO)** — a query-relevant slice of the knowledge graph that any consumer can use without knowing about graph internals, embeddings, or LLM details.

**Build** (one-time per catalog):

```
Databricks Catalog
    → L1 Structural Extraction  — deterministic schema parsing
    → L2 Semantic Interpretation — LLM-powered entity and property detection
    → L3 Vocabulary Detection    — pattern matching + LLM synonym expansion
    → Commit to Neo4j
    → Embed nodes for vector search
```

**Retrieve** (per question):

```
Natural language question
    → Hybrid search (vector + lexical)
    → Normalize + dedup seed hits
    → Type-aware graph expansion
    → Dedup expanded artifacts
    → Visibility policy pruning
    → SCO
```

The SCO contains entities, physical assets, join paths, governed values, metrics, ancestry, and semantic type annotations — everything a consumer needs to understand the data.

---

## Graph Data Model (v1)

Sema builds a multi-layer knowledge graph in Neo4j. Every node has a stable `id` (UUID); source-backed nodes also carry a `ref` that anchors them to the external system.

### Node Types

| Layer | Nodes | Purpose |
|-------|-------|---------|
| **Physical** | `DataSource`, `Catalog`, `Schema`, `Table`, `Column` | Mirrors your warehouse structure. Each carries a platform-scoped `ref` (e.g. `databricks://workspace/catalog/schema/table`) |
| **Semantic** | `Entity`, `Property`, `Alias`, `Metric` | LLM-inferred business concepts. Entities map to tables via `ENTITY_ON_TABLE`, properties map to columns via `PROPERTY_ON_COLUMN`. Aliases replace synonyms with `is_preferred` and `description` fields |
| **Vocabulary** | `Vocabulary`, `ValueSet`, `Term` | Named vocabularies, coded value sets, term hierarchies (ICD-10, AJCC, etc.), and aliases for search expansion |
| **Joins** | `JoinPath` | First-class join artifacts with ordered `join_predicates`, `hop_count`, `cardinality_hint`, and optional `sql_snippet`. Linked to tables via `USES` and to entities via `FROM_ENTITY`/`TO_ENTITY` |
| **Provenance** | `Assertion` | Every fact is backed by an assertion with `source`, `confidence`, and `status` (`auto`, `accepted`, `rejected`, `pinned`, `superseded`). Selective `SUBJECT`/`OBJECT` edges link assertions to their resolved nodes |

### Relationships

| Relationship | From | To | Purpose |
|---|---|---|---|
| `IN_SOURCE` | Catalog | DataSource | Catalog belongs to data source |
| `IN_CATALOG` | Schema | Catalog | Schema belongs to catalog |
| `IN_SCHEMA` | Table | Schema | Table belongs to schema |
| `IN_TABLE` | Column | Table | Column belongs to table |
| `ENTITY_ON_TABLE` | Entity | Table | Entity is implemented by table |
| `PROPERTY_ON_COLUMN` | Property | Column | Property is implemented by column |
| `HAS_PROPERTY` | Entity | Property | Entity has semantic property |
| `REFERS_TO` | Alias | Entity/Property/Term | Alias refers to canonical node |
| `HAS_VALUE_SET` | Column | ValueSet | Column has a set of permissible values |
| `MEMBER_OF` | Term | ValueSet | Term belongs to value set |
| `PARENT_OF` | Term | Term | Hierarchical term relationship |
| `CLASSIFIED_AS` | Property | Vocabulary | Property classified under vocabulary |
| `IN_VOCABULARY` | Term | Vocabulary | Term belongs to vocabulary |
| `MEASURES` | Metric | Entity | Metric measures entity |
| `AGGREGATES` | Metric | Property | Metric aggregates property |
| `FILTERS_BY` | Metric | Property/Term | Metric filters by property or term |
| `AT_GRAIN` | Metric | Property/Term | Metric operates at this grain |
| `FROM_ENTITY` | JoinPath | Entity | Join starts from entity |
| `TO_ENTITY` | JoinPath | Entity | Join ends at entity |
| `USES` | JoinPath | Table/Column | Join uses physical asset |
| `SUBJECT` | Assertion | Node | Assertion is about this node |
| `OBJECT` | Assertion | Node | Assertion references this node |

### Assertion-Driven Architecture

All extracted and inferred facts are stored as first-class `Assertion` records before being resolved into the graph. This enables:

- **Conflict resolution** — multiple sources can assert different facts; winner selection uses `pinned > accepted > source_precedence > confidence`
- **Human overrides** — pin or reject assertions without losing the original extraction
- **Auditability** — every node traces back to the assertions that created it
- **Safe rebuilds** — wipe and rebuild from source; human overrides are exported and re-imported via `translate_ref()`

### SCO Visibility Policy

The Semantic Context Object filters candidates by assertion status and confidence policy before serving them to consumers:

| Status | Included in SCO |
|--------|----------------|
| `pinned` | Always |
| `accepted` | Always |
| `auto` | If confidence >= threshold |
| `rejected` | Never |
| `superseded` | Never |

Confidence thresholds are determined by `confidence_policy` on each candidate: `structural` uses 0.5, `semantic` uses 0.7. When `confidence_policy` is absent, the threshold is inferred from the `source` field for backward compatibility.

---

## Works With Any Domain

Sema is domain-agnostic. If your warehouse has tables and columns, Sema can extract the ontology.

| Domain | Example Vocabularies |
|--------|---------------------|
| **Healthcare** | ICD-10 codes, AJCC staging, TNM classification, patient registries |
| **Finance** | NAICS codes, currency codes, transaction types, risk categories |
| **Retail** | SKU hierarchies, product categories, UPC codes, brand taxonomies |
| **Manufacturing** | BOM hierarchies, part classifications, process codes |
| **PropTech** | Zoning codes, property types, building classifications, land use codes |
| **Any warehouse** | Tables, columns, coded values, entity relationships |

---

## Connector & Model Support

Sema currently ships with a **Databricks** connector. Additional warehouse connectors (Snowflake, BigQuery, etc.) are on the roadmap — the connector interface is pluggable.

For LLM and embedding providers, **bring your own model**. Sema works with any provider through a unified interface:

| | Supported Providers |
|---|---|
| **LLM** | OpenRouter, Anthropic, OpenAI, Databricks Model Serving, any OpenAI-compatible endpoint |
| **Embeddings** | OpenRouter, OpenAI, sentence-transformers (local), Databricks, any OpenAI-compatible endpoint |

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager (recommended)
- **Neo4j 5.x** — local via Docker or remote
- **Databricks SQL Warehouse** — data source (currently the only supported connector)
- **LLM API key** — any supported provider above
- **Embedding API key** — any supported provider above, or use local sentence-transformers

### 1. Clone and install

```bash
git clone git@github.com:Nine-Sigma/sema.git
cd sema
uv sync            # installs all dependencies into a virtual environment
```

<details>
<summary>Using pip instead of uv</summary>

```bash
pip install -e .
```
</details>

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials — see `.env.example` for the full list. Settings can also be passed via CLI flags or a YAML config file (`--config path/to/config.yaml`). CLI flags override env vars, and env vars override config file values.

### 3. Start Neo4j

```bash
docker compose up -d
```

This starts Neo4j 5.26 with APOC on `bolt://localhost:7687`. Browser UI at `http://localhost:7474`.

### 4. Build the knowledge graph

```bash
uv run sema build --catalog my_catalog --schemas schema1,schema2
```

This runs the full pipeline: structural extraction, semantic interpretation, vocabulary detection, graph materialization, and embedding computation. On a typical catalog with ~50 tables, expect 5-15 minutes depending on your LLM provider.

### 5. Query

```bash
# Get a Semantic Context Object (JSON)
uv run sema context --question "How many patients have stage III breast cancer?"

# Generate SQL from natural language
uv run sema query --question "Average age of patients by cancer type"
```

---

## CLI Reference

All commands are run with `uv run sema` (or just `sema` if you installed with pip).

### `sema build`

Build the knowledge graph from your warehouse catalog.

```bash
uv run sema build --catalog my_catalog --schemas schema1,schema2
```

| Flag | Description | Default |
|------|-------------|---------|
| `--catalog` | Catalog name to extract from | — |
| `--schemas` | Comma-separated schema names | all schemas |
| `--table-pattern` | Glob pattern to filter tables | `*` |
| `--table-workers` | Parallel table workers | `4` |
| `--llm-provider` | LLM provider | `openrouter` |
| `--llm-model` | LLM model name | `anthropic/claude-sonnet-4` |
| `--llm-timeout` | LLM request timeout in seconds | `120` |
| `--skip-embeddings` | Create indexes only, skip embeddings | `false` |
| `--resume` | Skip tables already in the graph | `false` |
| `--config` | Path to YAML config file | — |
| `--verbose` | Enable verbose output | `false` |

### `sema context`

Retrieve a Semantic Context Object — a query-relevant slice of the knowledge graph.

```bash
uv run sema context --question "How many patients have stage III breast cancer?"
```

| Flag | Description | Default |
|------|-------------|---------|
| `--question` | Natural language question | required |
| `--consumer` | Consumer type for pruning: `nl2sql`, `discovery` | `nl2sql` |

### `sema query`

Generate and optionally execute SQL from natural language. Uses the NL2SQL consumer with plan/explain/execute operations.

```bash
# Plan — generate SQL without executing
uv run sema query --question "Average age of patients by cancer type" --operation plan

# Explain — generate SQL and show the execution plan
uv run sema query --question "Average age of patients by cancer type" --operation explain

# Execute — generate and run SQL against Databricks
uv run sema query --question "Average age of patients by cancer type" --operation execute
```

| Flag | Description | Default |
|------|-------------|---------|
| `--question` | Natural language question | required |
| `--operation` | `plan`, `explain`, or `execute` | `plan` |
| `--consumer` | Consumer type | `nl2sql` |
| `--llm-provider` | LLM provider | `openrouter` |
| `--llm-model` | LLM model name | `anthropic/claude-sonnet-4` |
| `--llm-timeout` | LLM request timeout in seconds | `120` |
| `--embedding-provider` | Embedding provider | `openrouter` |
| `--embedding-model` | Embedding model name | `google/gemini-embedding-001` |
| `--verbose` | Return full response JSON | `false` |

### `sema review`

Export low-confidence assertions for human review. Useful for identifying extraction results that may need manual correction.

```bash
# Print to stdout
uv run sema review --threshold 0.85

# Save to file
uv run sema review --threshold 0.7 --output review.json
```

| Flag | Description | Default |
|------|-------------|---------|
| `--threshold` | Confidence threshold — assertions below this are exported | `0.85` |
| `--output` | Output file path | stdout |

---

## Consumer Architecture

Sema uses a pluggable consumer protocol. Consumers receive a pruned SCO and produce task-specific outputs.

### NL2SQL Consumer

The built-in NL2SQL consumer generates constrained SQL from natural language questions:

- **Plan** — generates SQL with closed-world validation against the SCO
- **Explain** — generates SQL and shows the Databricks execution plan
- **Execute** — generates, validates, executes SQL, and synthesizes results

The consumer receives the SQL dialect explicitly and the prompt includes:
- Entity context (name + description)
- Table/column listing with semantic type annotations (e.g., `tnm_stage (categorical)`)
- Join paths with predicates
- Governed filter values (exact values for WHERE clauses)
- Metric definitions (name, formula, aggregates, filters, grains)
- Term hierarchy context
- Dialect-specific guidance (Databricks SQL rules, ANSI fallback)

Prompt truncation follows a strict deterministic cut order when the budget is exceeded.

### Writing a Custom Consumer

Implement the `Consumer` protocol in `src/sema/consumers/base.py`:

```python
class MyConsumer:
    name: str = "my_consumer"
    capabilities: set[str] = {"analyze"}

    def context_profile(self) -> ContextProfile: ...
    def run(self, request, sco, deps) -> ConsumerResult: ...
```

Register it in `src/sema/consumers/__init__.py` and it becomes available via `--consumer my_consumer`.

---

## Running Tests

```bash
# Unit tests (no external services needed)
uv run pytest

# Integration tests (requires Neo4j running)
uv run pytest -m integration

# All tests with coverage
uv run pytest --cov=sema --cov-report=term-missing

# Type checking
uv run mypy src/sema/
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
