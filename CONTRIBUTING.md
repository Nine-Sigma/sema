# Contributing to Sema

Thank you for your interest in contributing to Sema.

## Developer Certificate of Origin (DCO)

All contributions must include a `Signed-off-by` line in the commit message, certifying that you wrote the code or have the right to submit it. This is enforced by CI.

Add it automatically with:

```bash
git commit -s -m "feat: your change"
```

This adds a line like:

```
Signed-off-by: Your Name <your.email@example.com>
```

See the [DCO](DCO) file for the full text.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Neo4j 5.x (local via Docker or remote)

### Setup

```bash
git clone https://github.com/Nine-Sigma/sema.git
cd sema
uv sync --dev
```

Start Neo4j locally:

```bash
docker compose up -d
```

Copy the example environment file:

```bash
cp .env.example .env
# Fill in your Databricks and LLM credentials
```

### Running Tests

```bash
# Unit tests (no external services needed)
uv run pytest tests/unit/

# Unit tests with coverage
uv run pytest tests/unit/ --cov=sema --cov-fail-under=85

# Integration tests (requires Neo4j)
uv run pytest tests/integration/

# All tests
uv run pytest
```

## Submitting Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests — coverage must not drop below 85%
4. Run `uv run pytest tests/unit/` and ensure all tests pass
5. Commit with DCO sign-off: `git commit -s`
6. Open a pull request against `main`

## Code Standards

- Functions should be 60 lines or fewer
- Files should be 400 lines or fewer
- Use type hints
- Helpers belong in `*_utils.py` files
- No globals — pass dependencies explicitly
- Tests use `pytest`; unit tests must not require external services

## Reporting Issues

Open an issue on GitHub. For security vulnerabilities, see [SECURITY.md](SECURITY.md).
