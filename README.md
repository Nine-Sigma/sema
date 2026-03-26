
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
Warehouse Metadata → LLM Interpretation → Knowledge Graph → Semantic Search (GraphRAG) → Consumer
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
Natural language question → Embed → Vector search → Graph expansion → Prune → SCO
```

The SCO contains entities, physical assets, join paths, governed values, and metrics — everything a consumer needs to understand the data.

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

## Prerequisites

- **Python 3.12+**
- **Neo4j 5.x** — local via Docker or remote
- **Databricks SQL Warehouse** — data source (currently the only supported connector)
- **LLM API key** — any supported provider above
- **Embedding API key** — any supported provider above, or use local sentence-transformers

---

## Installation

```bash
git clone git@github.com:Nine-Sigma/sema.git
cd sema
```

Install with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

---

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

All settings are read from environment variables. See `.env.example` for the full list.

Configuration can also be passed via CLI flags or a YAML config file (`--config path/to/config.yaml`).

---

## Start Neo4j

```bash
docker compose up -d
```

Neo4j 5.26 with APOC on `bolt://localhost:7687`. Browser UI at `http://localhost:7474`.

---

## Usage

### Build the knowledge graph

```bash
sema build --catalog my_catalog --schemas schema1,schema2
```

| Flag | Description | Default |
|------|-------------|---------|
| `--catalog` | Catalog name to extract from | — |
| `--schemas` | Comma-separated schema names | all schemas |
| `--table-pattern` | Glob pattern to filter tables | `*` |
| `--table-workers` | Parallel table workers | `4` |
| `--llm-provider` | LLM provider | `openrouter` |
| `--llm-model` | LLM model name | `anthropic/claude-sonnet-4` |
| `--skip-embeddings` | Create indexes only, skip embeddings | `false` |
| `--resume` | Skip tables already in the graph | `false` |
| `--config` | Path to YAML config file | — |

### Get a Semantic Context Object

```bash
sema context --question "How many patients have stage III breast cancer?"
```

| Flag | Description | Default |
|------|-------------|---------|
| `--question` | Natural language question | required |
| `--consumer-hint` | Pruning strategy: `nl2sql`, `catalog`, `lineage` | `nl2sql` |

### Query with NL2SQL

```bash
# Plan — generate SQL without executing
sema query --question "Average age of patients by cancer type" --mode plan

# Explain — generate SQL with explanation
sema query --question "Average age of patients by cancer type" --mode explain

# Execute — run the SQL against Databricks
sema query --question "Average age of patients by cancer type" --mode execute
```

---

## Running Tests

```bash
# Unit tests (no external services)
uv run pytest

# Integration tests (requires Neo4j)
uv run pytest -m integration

# All tests
uv run pytest -m "unit or integration or e2e"
```

---

## License

Proprietary. Copyright Nine Sigma.
