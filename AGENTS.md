## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update src` to keep the graph current (AST-only, no API cost).
- graphify-out/ is local only and gitignored. Never commit graph.json or any other graphify output.

## Test cost rule (always)

The suite is large. While developing, run only the tests relevant to the change: the test file(s) for the modules touched, or `uv run pytest -k <name>`. Never run the full suite after each edit. Run the full unit suite exactly once at the end of the task, in parallel: `uv run pytest -n auto`. Run integration or e2e tests once at the end only when the task needs them. A Claude Code hook (`.claude/hooks/pytest-scope.py`) refuses a full serial `pytest` call; Codex and other agents must follow the rule themselves.

## Coding standard: no globals

No globals, including module-level constants. Keep a value in the function that uses it. When several methods share a value, return it from a private static method on the class that owns it (example: `CypherQueries._physical_mapping_tail()`). Prefer pure functions that take inputs and return values. Applies to new and changed code; do not sweep existing constants unless asked.
