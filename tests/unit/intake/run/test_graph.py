"""The ingestion graph: fixed node order, verifiers between nodes, Review, and Import.

Pinned API of ``sema.intake.run.graph`` (task 11.2), imported lazily through ``graph_env``:

- ``discover(source, *, root, descriptor, config, client=None, hint=None, enrich=False,
  tier="metadata_only", run_id=None, schema=None) -> RunOutcome``. ``hint`` is a spec path;
  ``schema`` is the landing schema (a hint names its relations in one), derived from the
  source name when None. ``root`` is the ledger root: the
  registry is ``root/registry.sqlite``, the LangGraph checkpointer is
  ``root/checkpoints.sqlite`` (SqliteSaver), the run ledger is ``Ledger(root).run(run_id)``.
  ``descriptor`` is a run descriptor YAML path; ``client`` is the ``LLMClient`` the run's
  ``ModelCalls`` wraps (built from ``config`` when None); ``run_id`` defaults to a uuid4.
- ``RunOutcome(run_id, node, payload)``: ``node`` is the interrupt the run stopped at,
  ``"review"`` or ``"value_ask"``, or None once the run ended (approved or rejected).
  A Review payload is a dict with the keys, in order, ``ambiguities``, ``proposals``,
  ``spec``, ``dry_run``, ``run_metrics``, ``ledger``; ``spec`` is None when none assembled.
  A value-ask payload is ``{"requests": [{"relation": <qualified name>, "columns": [...]}]}``.
- ``approve(run_id, *, root, config, approver, client=None, spec=None, decisions=None)``,
  ``reject(run_id, *, root, config, approver, client=None)``,
  ``approve_values(run_id, *, root, config, client=None)`` (approves every requested
  column), and ``resume(run_id, *, root, config, client=None)``, each -> ``RunOutcome``.
  Approve with neither file imports: ``write_import`` to ``<run dir>/import/manifest.yaml``
  and ``authored_evidence.json``, then registry ``approved`` with the approver at stage
  ``import``. With enrichment on, Import writes ``apply_enrichment(manifest, enrichment)``
  where ``enrichment`` is this run's Enrich output, recomputed by re-execution (calls
  replay), never read back from the ``enrichment`` artifact; off, it writes the plain
  import. Reject sets ``rejected`` with the approver and imports nothing.
- Ledger events written by the graph: ``kind="node"`` with ``stage`` = node name on every
  node entry; ``kind="verdict"`` with ``stage`` in ``land|structure|roles|dry_run`` and
  payload ``{"status", "attempt", ...}`` after each verifier; ``kind="resume"`` with
  ``stage`` = the node the run continues at, once per ``resume`` call.
- Artifacts written by the graph beside the S2 ones: ``structure``, ``spec`` (the edited
  spec once one is submitted), ``structure_ambiguities`` and ``roles_ambiguities`` in the
  ``pipeline_env.AmbiguityList`` shape, ``run_metrics`` (before every Review, so after
  ``decide`` too), and ``decisions`` as ``{"decisions": [{stage, subject, choice}, ...]}``.
- A decisions file is validated before anything re-runs (``DecisionRejectedError``
  propagates from ``approve``); ``decide`` applies it with zero model calls and presents
  Review again, without the resolvers. Given beside ``spec``, both apply (design
  Decision 11) and the spec wins where both speak: the edited spec is what is imported.
- Verifier placement: Land after ``land``; Structure after ``resolve_structure`` and after
  ``propose_layouts``; Roles after ``resolve_roles``; Dry run after ``dry_run``. The
  code-only ``structure`` and ``roles`` nodes hand their open entries to their resolver
  unverified (S2's Roles verifier fails ``section_resolved`` while sections are open).
  A failing verdict re-enters the node it followed; the third failure is written with status
  ``escalate`` and routes to Review with registry reason ``escalate``.
- ``graph.py`` imports ``verify_roles`` and ``enrich`` by name, so tests patch
  ``sema.intake.run.graph.verify_roles`` and ``sema.intake.run.graph.enrich``. Enrich runs only
  when enabled and passes ``enabled=True`` through.
- State holds scalars only (run id, source and descriptor paths, stage, attempts per
  stage, decisions and value approvals received, artifact names); stages recompute.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import load_concept_source_spec
from sema.intake.verify.base import Failure, Verdict
from sema.intake.verify.roles import REQUIRED_ROLES_BOUND
from tests.unit.intake.graph_env import (
    A2_DESCRIPTOR,
    ACRIS_DESCRIPTOR,
    APPROVER,
    CHECKPOINT_FILE,
    IMPORT_DIR,
    REGISTRY_FILE,
    REVIEW_KEYS,
    STRUCTURE_BATCH,
    GraphRun,
    approve,
    code_tables,
    discover,
    graph,
    model_names,
    routed_client,
    settings,
    subsequence,
    through_values,
)
from tests.unit.intake.decisions_env import write_decisions
from tests.unit.intake.eligibility_env import descriptor
from tests.unit.intake.enrichment_env import manifest_property, synonym_for
from tests.unit.intake.pipeline_env import landed_run, run_stages
from tests.unit.intake.structure_env import A2_WORKBOOK, ACRIS, ACRIS_SPEC

pytestmark = pytest.mark.unit

INTAKE = Path(__file__).parents[4] / "src" / "sema" / "intake"
RUNTIME_MODULES = {"run/graph.py", "calls/tracing.py"}
RUNTIME_PREFIXES = ("langgraph", "langchain_core.callbacks", "langchain.callbacks")
GRAPH_MODULE = "sema.intake.run.graph"
ACRIS_SCHEMA = "land_acris"
ACRIS_MASTER = "acris.acris_real_property_master"
BOROUGH = "Recorded Borough"
A2_ORDER = [
    ("node", "land"), ("verdict", "land"),
    ("node", "structure"),
    ("node", "resolve_structure"), ("verdict", "structure"),
    ("node", "propose_layouts"), ("verdict", "structure"),
    ("node", "roles"),
    ("node", "value_ask"), ("node", "value_ask"),
    ("node", "roles"),
    ("node", "resolve_roles"), ("verdict", "roles"),
    ("node", "dry_run"), ("verdict", "dry_run"),
    ("node", "review"),
]


def _a2(tmp_path: Path, **options: object) -> GraphRun:
    """A2 at Review, after the one value ask its settled status column makes."""
    run = discover(tmp_path, A2_WORKBOOK, A2_DESCRIPTOR, routed_client(), **options)
    through_values(run)
    return run


def _failing_roles(*_args: object, **_kwargs: object) -> Verdict:
    failure = Failure(check=REQUIRED_ROLES_BOUND, relation="any", evidence={})
    return Verdict(status="fail", failures=(failure,))


def _imported(run: GraphRun) -> Path:
    return run.ledger_dir / IMPORT_DIR / "manifest.yaml"


def test_nodes_run_in_order_with_verifiers_between_them(tmp_path: Path) -> None:
    run = _a2(tmp_path)

    assert run.outcome.node == "review"
    assert run.steps() == A2_ORDER
    assert all(e["payload"]["status"] == "pass" for e in run.events("verdict"))
    assert (tmp_path / CHECKPOINT_FILE).is_file()
    assert (tmp_path / REGISTRY_FILE).is_file()
    row = run.row()
    assert (row.stage, row.status, row.reason) == ("review", "waiting_review", None)


def test_review_presents_open_entries_before_the_spec(tmp_path: Path) -> None:
    run = _a2(tmp_path)

    assert tuple(run.outcome.payload) == REVIEW_KEYS
    assert run.outcome.payload["spec"] is not None
    assert run.has_artifact("spec")
    assert run.has_artifact("structure_ambiguities")
    assert run.has_artifact("roles_ambiguities")


def test_three_roles_failures_reach_review_with_escalate(tmp_path: Path) -> None:
    with patch(f"{GRAPH_MODULE}.verify_roles", side_effect=_failing_roles):
        run = _a2(tmp_path)

    verdicts = run.verdicts("roles")
    assert [v["status"] for v in verdicts] == ["fail", "fail", "escalate"]
    assert [v["attempt"] for v in verdicts] == [1, 2, 3]
    assert run.nodes().count("resolve_roles") == 3
    assert run.nodes()[-1] == "review"
    assert "dry_run" not in run.nodes()
    row = run.row()
    assert (row.stage, row.status, row.reason) == ("review", "waiting_review", "escalate")
    assert run.outcome.node == "review"


def test_enrich_runs_between_roles_and_dry_run_only_when_enabled(tmp_path: Path) -> None:
    enrich = MagicMock(return_value=None)
    with patch(f"{GRAPH_MODULE}.enrich", enrich):
        off = _a2(tmp_path / "off")
        assert enrich.call_count == 0
        on = _a2(tmp_path / "on", enrich=True)

    assert "enrich" not in off.nodes()
    assert subsequence(
        on.steps(), [("verdict", "roles"), ("node", "enrich"), ("node", "dry_run")]
    )
    assert enrich.call_count == 1
    assert enrich.call_args.kwargs["enabled"] is True


def test_approve_imports_and_sets_approved_with_the_approver(tmp_path: Path) -> None:
    run = _a2(tmp_path)

    outcome = approve(run)

    assert outcome.node is None
    row = run.row()
    assert (row.stage, row.status, row.approver) == ("import", "approved", APPROVER)
    assert run.nodes()[-1] == "import"
    manifest = yaml.safe_load(_imported(run).read_text())
    assert manifest["descriptor"]["target_model_id"] == "codes-a2"
    assert (_imported(run).parent / "authored_evidence.json").is_file()


def _acris(root: Path, *, enrich: bool) -> GraphRun:
    return discover(
        root, ACRIS, ACRIS_DESCRIPTOR, routed_client(),
        hint=ACRIS_SPEC, schema=ACRIS_SCHEMA, enrich=enrich,
    )


def _manifest(run: GraphRun) -> dict[str, object]:
    loaded: dict[str, object] = yaml.safe_load(_imported(run).read_text())
    return loaded


def _plain_import(root: Path) -> dict[str, object]:
    """The S2 harness's import of the hinted ACRIS spec, as ``write_import`` writes it."""
    hint = load_concept_source_spec(ACRIS_SPEC)
    with landed_run(ACRIS, root, ACRIS_SCHEMA) as landed:
        stages = run_stages(landed, descriptor(ACRIS_DESCRIPTOR), hint=hint)
        assert stages.spec is not None
        manifest = import_concept_source(stages.spec, landed.provider, dataset_map={}).manifest
    loaded: dict[str, object] = yaml.safe_load(yaml.safe_dump(manifest, sort_keys=False))
    return loaded


def test_import_applies_the_enrichment_this_run_produced(tmp_path: Path) -> None:
    from sema.intake.report.enrichment import Enrichment, apply_enrichment

    on = _acris(tmp_path / "on", enrich=True)
    assert on.outcome.node == "review"
    produced = Enrichment.model_validate(on.artifact("enrichment"))
    (on.ledger_dir / "artifacts" / "enrichment.json").unlink()
    approve(on)
    off = _acris(tmp_path / "off", enrich=False)
    approve(off)

    plain, enriched = _manifest(off), _manifest(on)
    assert not off.has_artifact("enrichment")
    assert plain == _plain_import(tmp_path / "oracle")
    assert {entry.source for entry in produced.entries} == {"model"}
    assert enriched == apply_enrichment(plain, produced)
    assert manifest_property(plain, ACRIS_MASTER, BOROUGH).get("synonyms", []) == []
    assert manifest_property(enriched, ACRIS_MASTER, BOROUGH)["synonyms"] == [
        synonym_for(BOROUGH)
    ]


def test_reject_imports_nothing(tmp_path: Path) -> None:
    run = _a2(tmp_path)

    outcome = graph().reject(
        run.run_id, approver=APPROVER, client=run.client, **settings(tmp_path)
    )

    assert outcome.node is None
    row = run.row()
    assert (row.status, row.approver) == ("rejected", APPROVER)
    assert "import" not in run.nodes()
    assert not (run.ledger_dir / IMPORT_DIR).exists()


def test_edited_spec_reenters_at_the_roles_verifier_with_zero_calls(tmp_path: Path) -> None:
    run = _a2(tmp_path)
    edited = run.artifact("spec")
    assert edited["value_sets"]["roles"]["replaced_by"] is None
    edited["value_sets"]["roles"]["replaced_by"] = "successor"
    path = tmp_path / "edited.spec.yaml"
    path.write_text(yaml.safe_dump({"concept_source": edited}, sort_keys=False))
    before, mark, client = run.counters(), run.mark(), routed_client()

    outcome = approve(run, client, spec=path)

    assert outcome.node == "review"
    assert client.invoke.call_count == 0
    assert run.counters() == before
    later = run.steps(mark)
    assert subsequence(
        later,
        [("verdict", "roles"), ("node", "dry_run"), ("verdict", "dry_run"), ("node", "review")],
    )
    assert not {"land", "structure", "roles", "resolve_roles"} & set(run.nodes(mark))
    assert run.artifact("spec")["value_sets"]["roles"]["replaced_by"] == "successor"
    assert run.row().status == "waiting_review"
    assert not (run.ledger_dir / IMPORT_DIR).exists()


A2_STATUS = {"column": "status", "active_values": ["active"], "retired_values": ["deprecated"]}


def _review_inputs(root: Path, run: GraphRun) -> tuple[Path, Path, list[str]]:
    """An edited spec stating the status role and a successor, and decisions closing both
    status entries as ``none``: on the values relation both speak (the spec keeps the status
    role the decision drops), on the concepts relation only the decision does."""
    edited = run.artifact("spec")
    roles = edited["value_sets"]["roles"]
    assert roles["status"] == A2_STATUS and roles["replaced_by"] is None
    assert edited["concepts"]["roles"]["status"] is None
    roles["status"], roles["replaced_by"] = A2_STATUS, "successor"
    spec = root / "edited.spec.yaml"
    spec.write_text(yaml.safe_dump({"concept_source": edited}, sort_keys=False))
    subjects = [f"{edited[s]['relation']['name']}.status" for s in ("value_sets", "concepts")]
    rows = [("roles", subject, "none") for subject in subjects]
    return spec, write_decisions(root / "decisions.yaml", rows), subjects


def _decided(run: GraphRun) -> dict[str, tuple[object, object]]:
    listed = run.artifact("roles_ambiguities")["ambiguities"]
    return {entry["subject"]: (entry["choice"], entry["source"]) for entry in listed}


def test_spec_beside_decisions_applies_both_and_the_spec_wins(tmp_path: Path) -> None:
    run = _a2(tmp_path / "both")
    spec, decisions, subjects = _review_inputs(tmp_path, run)
    client = routed_client()

    outcome = approve(run, client, spec=spec, decisions=decisions)

    assert outcome.node == "review"
    assert client.invoke.call_count == 0
    assert len(run.artifact("decisions")["decisions"]) == 2
    assert all(_decided(run)[subject] == ("none", "human") for subject in subjects)
    roles = run.artifact("spec")["value_sets"]["roles"]
    assert (roles["status"], roles["replaced_by"]) == (A2_STATUS, "successor")
    approve(run)
    only_spec = _a2(tmp_path / "spec")
    approve(only_spec, spec=spec)
    approve(only_spec)
    assert _manifest(run) == _manifest(only_spec)


@pytest.mark.parametrize("given", ["spec", "decisions"])
def test_a_spec_or_decisions_alone_behave_as_before(tmp_path: Path, given: str) -> None:
    run = _a2(tmp_path / "run")
    spec, decisions, subjects = _review_inputs(tmp_path, run)

    outcome = approve(run, **({"spec": spec} if given == "spec" else {"decisions": decisions}))

    assert outcome.node == "review"
    roles = run.artifact("spec")["value_sets"]["roles"]
    if given == "spec":
        assert (roles["status"], roles["replaced_by"]) == (A2_STATUS, "successor")
        assert not run.has_artifact("decisions")
    else:
        assert (roles["status"], roles["replaced_by"]) == (None, None)
        assert all(_decided(run)[subject] == ("none", "human") for subject in subjects)


def test_run_metrics_at_review_equal_the_registry_row(tmp_path: Path) -> None:
    client = routed_client()
    run = discover(tmp_path, code_tables(tmp_path), ACRIS_DESCRIPTOR, client)

    assert run.outcome.node == "review"
    assert run.outcome.payload["spec"] is None
    assert STRUCTURE_BATCH in model_names(client)
    metrics = run.artifact("run_metrics")
    row = run.row()
    assert row.model_calls >= 1
    assert (metrics["model_calls"], metrics["tokens"], metrics["cost"]) == (
        row.model_calls, row.tokens, pytest.approx(row.cost),
    )
    assert metrics["open_structure"] > 0


def _runtime_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return {
        name for name in found if name.startswith(RUNTIME_PREFIXES) or name == GRAPH_MODULE
    }


def test_only_graph_and_tracing_import_langgraph() -> None:
    importing = {
        str(path.relative_to(INTAKE)): _runtime_imports(path)
        for path in sorted(INTAKE.rglob("*.py"))
    }
    offenders = {
        name: found for name, found in importing.items() if found and name not in RUNTIME_MODULES
    }

    assert offenders == {}
    assert any(name.startswith("langgraph") for name in importing.get("run/graph.py", set()))
