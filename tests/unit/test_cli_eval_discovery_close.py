"""Done-when 7 (N6): `sema eval discovery` closes the connector it opens.

The connector is closed once, after the command's last read of it, when the command succeeds
and when a read or the scoring raises. Every graph, warehouse and scorecard call is a stub that
records, in order, each call that reads the connector; the connector and the driver are spies.
No Neo4j, no warehouse, no oracle file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner, Result

import sema.cli_factories as factories
import sema.eval.discovery_scorecard as scorecard
import sema.eval.discovery_scorecard_utils as scorecard_utils
import sema.pipeline.build as build
import sema.pipeline.coded_columns as coded_columns
from sema.cli_eval import eval_group
from sema.models.config import BuildConfig

pytestmark = pytest.mark.unit

STUBS = {"facts_from_graph": {}, "joins_from_graph": [], "read_coded_bindings": [],
         "read_identity": ([], []),
         "loaded_code_lists": {}, "coded_facts": {}, "identity_facts": {},
         "declared_constraints": {}, "recheck": {}, "recheck_coded": {},
         "score_discovery": {}, "render": "stub scorecard\n"}  # fmt: skip


@dataclass
class Discovery:
    result: Result
    events: list[str] = field(default_factory=list)


def _stub(name: str, value: Any, connector: Any, events: list[str], fails: str | None) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        if any(arg is connector for arg in (*args, *kwargs.values())):
            events.append(f"read {name}")
        if name == fails:
            raise RuntimeError(f"{name} failed")
        return value

    return call


def run_discovery(tmp_path: Path, mp: pytest.MonkeyPatch, fails: str | None = None) -> Discovery:
    events: list[str] = []
    connector, driver = MagicMock(name="connector"), MagicMock(name="driver")
    connector.close.side_effect = lambda: events.append("close connector")
    driver.close.side_effect = lambda: events.append("close driver")
    config = SimpleNamespace(catalog="cat", schemas=["sch"], neo4j=None,
                             concept_spec=None, concept_staging=None)  # fmt: skip
    path = tmp_path / "build.yaml"
    path.write_text("stub: true\n")
    mp.setattr(BuildConfig, "from_file", classmethod(lambda cls, _: config))
    mp.setattr(build, "resolve_connector_factory",
               lambda _: SimpleNamespace(create=lambda: connector))  # fmt: skip
    mp.setattr(factories, "_get_neo4j_driver", lambda _: driver)
    mp.setattr(coded_columns, "read_dictionary", lambda *_: None)
    mp.setattr(scorecard_utils, "read_oracle", lambda *_: {})
    for name, value in STUBS.items():
        mp.setattr(scorecard, name, _stub(name, value, connector, events, fails))
    result = CliRunner().invoke(
        eval_group, ["discovery", "--config", str(path), "--out", str(tmp_path / "card.md")])
    return Discovery(result, events)


CASES = {"succeeds": None, "scoring_raises": "score_discovery", "a_read_raises": "recheck"}


@pytest.mark.parametrize("case", list(CASES))
def test_the_connector_is_closed_once_after_its_last_read(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = run_discovery(tmp_path, monkeypatch, CASES[case])

    assert (run.result.exit_code == 0) == (CASES[case] is None), run.result.output
    reads = [i for i, event in enumerate(run.events) if event.startswith("read ")]
    assert reads, run.events
    assert run.events.count("close connector") == 1, run.events
    assert run.events.index("close connector") > reads[-1], run.events


@pytest.mark.parametrize("case", list(CASES))
def test_decoy_the_driver_is_still_closed_once(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = run_discovery(tmp_path, monkeypatch, CASES[case])

    assert run.events.count("close driver") == 1, run.events
