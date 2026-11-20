"""The code-domains report reads what the build wrote: one row per value-set link and per
binding the evidence proposes, with the link's verification and coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from showcase.open_data_load import code_domains_report as report_module
from showcase.open_data_load.code_domains_report import (
    Evidence,
    GraphLink,
    Member,
    Row,
    report,
)

pytestmark = pytest.mark.unit

SHADES = "main.shade_codes.shade_code"
SIZES = "main.size_codes.size_kind"
LINKED_CONTAINMENT = {"evidence": "containment", "linked": True, "verification": "verified"}


def by_column(rows: list[Row]) -> dict[str | None, list[Row]]:
    found: dict[str | None, list[Row]] = {}
    for row in rows:
        found.setdefault(row.column, []).append(row)
    return found


def test_label_and_share_counts_honor_listed() -> None:
    members = (
        Member("r", "red", 3, True), Member("g", "green", 1, True),
        Member("b", "blue", 0, True), Member("x", None, 1, False),
    )
    unlabeled = (Member("s", None, 2, True), Member("l", "large", 2, True))
    links = [GraphLink("item.shade", "shade set", {}, members),
             GraphLink("item.size_kind", "size set", {}, unlabeled)]
    rows = by_column(report([], links))
    (shade,) = rows["item.shade"]
    assert shade.labels is True
    assert shade.listed_value_share == pytest.approx(2 / 3)
    assert shade.listed_row_share == pytest.approx(4 / 5)
    (size,) = rows["item.size_kind"]
    assert size.labels is False


def test_every_value_set_link_is_its_own_row() -> None:
    links = [GraphLink("item.shade", "shade set", {}, (Member("r", "red", 1, True),)),
             GraphLink("item.shade", "tint set", {}, (Member("r", None, 1, True),))]
    rows = by_column(report([], links))["item.shade"]
    assert sorted(row.value_set for row in rows) == ["shade set", "tint set"]
    assert all(row.linked for row in rows)


def test_a_link_shows_its_verification_and_coverage() -> None:
    props = {"verification": "unverified", "listed_value_share": 1.0,
             "listed_row_share": 0.9, "observed_distinct": 3, "null_rows": 4}
    evidence = [Evidence("item.shade", SHADES, "containment", {**LINKED_CONTAINMENT,
                                                               "verification": "unverified"})]
    links = [GraphLink("item.shade", "item_shade_values", props, (Member("r", "red", 9, True),))]
    (row,) = by_column(report(evidence, links))["item.shade"]
    assert (row.value_set, row.evidence, row.linked) == (SHADES, ("containment",), True)
    assert row.verification == "unverified"
    assert (row.listed_value_share, row.listed_row_share, row.null_rows) == (1.0, 0.9, 4)


def test_every_proposed_binding_is_a_row_with_its_evidence_types() -> None:
    refuted = {"evidence": "containment", "linked": False, "verification": "unverified",
               "listed_value_share": 0.5, "listed_row_share": 0.25, "null_rows": 1,
               "reason": "containment below the floor"}
    tie = {"evidence": "containment", "linked": False, "verification": "unverified",
           "listed_value_share": 1.0, "listed_row_share": 1.0, "null_rows": 0,
           "alternatives": [SHADES, SIZES], "reason": "tie"}
    evidence = [
        Evidence("item.shade", SHADES, "name_match", {"evidence": "name_match"}),
        Evidence("item.shade", SHADES, "containment", refuted),
        Evidence("item.finish", SHADES, "containment", tie),
        Evidence("item.finish", SIZES, "containment", tie),
    ]
    rows = by_column(report(evidence, []))
    (shade,) = rows["item.shade"]
    assert (shade.evidence, shade.linked, shade.verification) == (
        ("containment", "name_match"), False, "unverified",
    )
    assert (shade.listed_value_share, shade.listed_row_share, shade.null_rows) == (0.5, 0.25, 1)
    assert "below the floor" in shade.reason
    finish = rows["item.finish"]
    assert sorted(row.value_set for row in finish) == [SHADES, SIZES]
    assert all(row.evidence == ("containment",) and not row.linked for row in finish)
    assert all(SIZES in row.reason and SHADES in row.reason for row in finish)


def test_an_unplaced_stated_domain_is_listed_with_its_reason() -> None:
    payload = {"evidence": "stated_domain", "property": "Misspelt Fieldname",
               "reason": "no column's normalized name equals the property's"}
    rows = report([Evidence(None, "Shade_Codes", "stated_domain", payload)], [])
    (unplaced,) = by_column(rows)[None]
    assert (unplaced.value_set, unplaced.evidence, unplaced.linked) == (
        "Shade_Codes", ("stated_domain",), False,
    )
    assert "Misspelt Fieldname" in unplaced.reason and "normalized name" in unplaced.reason


def test_main_writes_one_markdown_row_per_reported_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "profile.yaml"
    build = {"database": str(tmp_path / "shop.duckdb"), "catalog": "shop", "schema": "main"}
    profile.write_text(yaml.safe_dump({"build": build}), encoding="utf-8")
    evidence = [Evidence("item.shade", SHADES, "containment", LINKED_CONTAINMENT)]
    links = [GraphLink("item.shade", "item_shade_values", {"verification": "verified"},
                       (Member("r", "red", 1, True),))]
    monkeypatch.setattr(report_module, "read_evidence", lambda *_: evidence)
    monkeypatch.setattr(report_module, "read_links", lambda *_: links)
    out = tmp_path / "report.md"
    assert report_module.main(["--profile", str(profile), "--out", str(out)]) == 0
    table = [line for line in out.read_text().splitlines() if line.startswith("| ")]
    assert len(table) == 1 + len(report(evidence, links))
    assert any("item.shade" in line and SHADES in line and "verified" in line for line in table)
