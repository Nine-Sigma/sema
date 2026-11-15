"""Sources for the graph follow-up tests: one per landed medium, a capped run, a ranked CSV.

``discover_with`` calls ``sema.intake.run.graph.discover`` with an explicit ``IntakeModelConfig``,
so a test can set the call or cost cap that ``graph_env.settings`` fixes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml

from tests.support.athena_bundle import make_minimal_bundle
from tests.unit.intake.eligibility_env import KeywordEmbedder
from tests.unit.intake.graph_env import GraphRun, graph
from tests.unit.intake.model_calls_env import intake_config
from tests.unit.intake.structure_env import A3_DB

A3_DESCRIPTOR = A3_DB.parent / "ontology.run.yaml"
A3_RAW_ONTOLOGY = A3_DB.parent / "ontology.ttl"
OMOP = A3_DB.parents[1] / "omop"
DDL = OMOP / "OMOPCDM_postgresql_5.4_ddl.sql"
ATHENA_RUN = {
    "id": "athena-minimal",
    "identity_system": "ATH",
    "release": "1",
    "scheme": {"qualified_name": "ath.vocabularies", "description": "A minimal vocabulary bundle"},
}
RANKED_HEADER = ("id", "label", "zorbit_text", "quaxle_body")
KEYED_HEADER = ("concept_id", "code", "name", "zorbit_text", "quaxle_body")
DEFINITION_DESCRIPTION = "explaining what something means"
RANKED_ROWS = 20
RANKED_REPEAT = 15
LONG_TAIL = " written out at length for the embedding rank case"


@dataclass(frozen=True)
class Source:
    path: Path
    descriptor: Path


def athena_bundle(root: Path) -> Source:
    """The minimal bundle the Athena loader tests land, with its run descriptor."""
    bundle = root / "inputs" / "bundle"
    make_minimal_bundle(bundle)
    descriptor = root / "inputs" / "athena.run.yaml"
    descriptor.write_text(yaml.safe_dump(ATHENA_RUN), encoding="utf-8")
    return Source(bundle, descriptor)


def ranked_source(root: Path, header: tuple[str, ...] = RANKED_HEADER) -> Path:
    """One relation whose ``definition`` role has two eligible text columns.

    Both text columns are long and repeat, so no lexicon hit or code rule settles
    ``definition``; only an embedding rank can tell them apart. ``KEYED_HEADER`` names
    every other role by a lexicon word, so only ``definition`` and the section stay open.
    """
    directory = root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "ranked.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(1, RANKED_ROWS + 1):
            cycle = index % RANKED_REPEAT
            values = {
                "id": index, "concept_id": index, "code": f"C{index}",
                "label": f"Lbl {index}", "name": f"Lbl {index}",
                "zorbit_text": f"Zorbit text {cycle}{LONG_TAIL}",
                "quaxle_body": f"Quaxle body {cycle}{LONG_TAIL}",
            }
            writer.writerow([values[column] for column in header])
    return directory


def definition_embedder() -> KeywordEmbedder:
    """Only the ``definition`` role description lies near one column; the rest tie."""
    near, far, other = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    return KeywordEmbedder(
        {DEFINITION_DESCRIPTION: near, "zorbit": near, "quaxle": far}, other
    )


def definitions(manifest: Any) -> list[Any]:
    """Every ``definition`` value anywhere in an imported manifest."""
    if isinstance(manifest, dict):
        own = [manifest["definition"]] if "definition" in manifest else []
        return own + [found for value in manifest.values() for found in definitions(value)]
    if isinstance(manifest, list):
        return [found for item in manifest for found in definitions(item)]
    return []


def discover_with(
    root: Path,
    source: Path,
    descriptor: Path,
    client: MagicMock,
    *,
    config: dict[str, Any] | None = None,
    **options: Any,
) -> GraphRun:
    """``discover`` under an intake config with ``config`` overrides (caps, prices)."""
    outcome = graph().discover(
        source,
        root=root,
        descriptor=descriptor,
        # Mocked answers are scripted one per ask; only the agreement tests turn the pair on.
        config=intake_config(**{"agreement": False, **(config or {})}),
        client=client,
        **options,
    )
    return GraphRun(root, outcome.run_id, client, outcome)


def role_entry(run: GraphRun, role: str) -> dict[str, Any]:
    """The run's Roles ambiguity entry for ``role`` (subjects end in ``.<role>``)."""
    entries = run.artifact("roles_ambiguities")["ambiguities"]
    (entry,) = [e for e in entries if e["subject"].rpartition(".")[2] == role]
    found: dict[str, Any] = entry
    return found


def embedding_hints(entry: dict[str, Any]) -> list[str]:
    return [text for text in entry["hints"] if text.startswith("embedding")]
