"""The code-domains report: one row per proposed binding, its link state and data coverage."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec
from showcase.open_data_load import code_domains_report as report_module
from showcase.open_data_load.build import BuildTarget
from showcase.open_data_load.code_domains_report import GraphLink, Row, report

pytestmark = pytest.mark.unit

CATALOG = "shop"
SCHEMA = "main"
WAREHOUSE = (
    "CREATE TABLE shade_codes (shade_code VARCHAR PRIMARY KEY, description VARCHAR)",
    "INSERT INTO shade_codes VALUES ('r', 'red'), ('g', 'green'), ('b', 'blue')",
    "CREATE TABLE size_codes (size_kind INTEGER PRIMARY KEY, label VARCHAR)",
    "INSERT INTO size_codes VALUES (1, 'small'), (2, 'large')",
    "CREATE TABLE item (item_id INTEGER, shade VARCHAR,"
    " size_kind INTEGER REFERENCES size_codes (size_kind), finish VARCHAR)",
    "INSERT INTO item VALUES (1, 'r', 1, 'm'), (2, 'g', 1, 'm'), (3, 'g', 2, 's'),"
    " (4, 'x', NULL, 's'), (5, NULL, 2, 'm')",
)
SPEC = {
    "spec_version": 2,
    "id": "neutral",
    "identity_system": "neutral",
    "release": "1",
    "default_provider": "staging",
    "entity_model": {
        "layout": "per_file",
        "roles": {"property": "Column Name"},
        "entities": [
            {
                "relation": "land_n.Item_Dictionary_Columns",
                "domains": {
                    "Shade": "Shade_Codes",
                    "Misspelt Fieldname": "Shade_Codes",
                    "Finish": "Finish_Codes",
                },
            },
            {"relation": "land_n.Shade_Codes_Dictionary", "domains": {"SHADE CODE": "Shade_Codes"}},
        ],
    },
    "code_lists": [
        {
            "vocabulary_id": "Shade_Codes",
            "relation": "land_n.Shade_Codes",
            "roles": {"code": "SHADE CODE", "display": "DESCRIPTION"},
        },
        {
            "vocabulary_id": "Finish_Codes",
            "relation": "land_n.Finish_Codes",
            "roles": {"code": "FINISH", "display": "DESCRIPTION"},
        },
    ],
}


@pytest.fixture()
def target(tmp_path: Path) -> BuildTarget:
    database = tmp_path / f"{CATALOG}.duckdb"
    with duckdb.connect(str(database)) as con:
        for statement in WAREHOUSE:
            con.execute(statement)
    return BuildTarget(database=database, catalog=CATALOG, schema=SCHEMA)


def spec() -> ConceptSourceSpec:
    return ConceptSourceSpec.model_validate(SPEC)


def by_column(rows: list[Row]) -> dict[str | None, Row]:
    return {row.column: row for row in rows if row.column is not None}


def ref(target: BuildTarget, table: str, column: str) -> str:
    (item,) = [t for t in report_module.warehouse_tables(target) if t.name == table]
    return f"{item.ref}/{column}"


def test_stated_domain_without_a_graph_link_is_not_linked_and_carries_data_coverage(
    target: BuildTarget,
) -> None:
    shade = by_column(report(target, spec(), links=[]))["item.shade"]
    assert (shade.value_set, shade.evidence, shade.linked) == (
        "Shade_Codes", ("stated_domain",), False,
    )
    assert shade.listed_value_share == pytest.approx(2 / 3)
    assert shade.listed_row_share == pytest.approx(3 / 4)
    assert shade.null_rows == 1
    assert "no HAS_VALUE_SET" in shade.reason


def test_a_declared_key_proposes_the_referenced_relation(target: BuildTarget) -> None:
    size = by_column(report(target, spec(), links=[]))["item.size_kind"]
    assert size.evidence == ("declared_key",)
    assert size.value_set == "size_codes"
    assert (size.listed_value_share, size.listed_row_share, size.null_rows) == (1.0, 1.0, 1)


def test_a_stated_domain_the_warehouse_cannot_place_is_listed_with_the_reason(
    target: BuildTarget,
) -> None:
    rows = report(target, spec(), links=[])
    (unplaced,) = [row for row in rows if row.column is None]
    assert unplaced.value_set == "Shade_Codes"
    assert "Misspelt Fieldname" in unplaced.reason
    finish = by_column(rows)["item.finish"]
    assert finish.listed_value_share is None
    assert "Finish_Codes" in finish.reason


def test_graph_links_decide_linked_and_labels(target: BuildTarget) -> None:
    links = [
        GraphLink(ref(target, "item", "shade"), "shade set", frozenset({"r", "g", "b"}), 3),
        GraphLink(ref(target, "item", "finish"), "finish set", frozenset(), 0),
        GraphLink(ref(target, "item", "item_id"), "id set", frozenset({"1", "2"}), 0),
    ]
    rows = by_column(report(target, spec(), links=links))
    assert (rows["item.shade"].linked, rows["item.shade"].labels) == (True, True)
    assert rows["item.finish"].linked is False
    assert "no members" in rows["item.finish"].reason
    graph_only = rows["item.item_id"]
    assert (graph_only.evidence, graph_only.linked, graph_only.labels) == (
        ("build_graph",), True, False,
    )
    assert graph_only.listed_value_share == pytest.approx(2 / 5)
    assert graph_only.null_rows == 0


def test_main_writes_one_markdown_row_per_proposed_binding(
    target: BuildTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "profile.yaml"
    build = {"database": str(target.database), "catalog": CATALOG, "schema": SCHEMA}
    profile.write_text(yaml.safe_dump({"build": build}), encoding="utf-8")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({"concept_source": SPEC}), encoding="utf-8")
    out = tmp_path / "report.md"
    monkeypatch.setattr(report_module, "read_links", lambda *_: [])
    argv = ["--profile", str(profile), "--spec", str(spec_path), "--out", str(out)]
    assert report_module.main(argv) == 0
    table = [line for line in out.read_text().splitlines() if line.startswith("| ")]
    assert len(table) == 1 + len(report(target, spec(), links=[]))
    assert any("item.shade" in line and "Shade_Codes" in line for line in table)
