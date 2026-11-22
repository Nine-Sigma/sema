"""A neutral two-relation source: one published dataset served over file://, one local file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

HEADER = "ORDER NO,SHIP DATE,AMOUNT,QTY,Unused Flag,region_code\n"
PAGE_SIZE = 2
ORDERS_CSV = (
    HEADER + "A1,02.01.2020,10.50,3,,007\n"
    "A2,31.12.2021,,4,,010\n"
    "A3,,2.25,,,\n"
)
COLORS_CSV = "COLOR CODE,description\nR,Red\nG,Green\n"

SCHEMA = {
    "tables": [
        {
            "name": "orders",
            "dataset": "Neutral Orders",
            "columns": [
                {"dictionary_name": "Order No", "name": "order_no", "type": "VARCHAR"},
                {"dictionary_name": "Shipped On", "name": "ship_date", "type": "DATE"},
                {"dictionary_name": "Amount", "name": "amount", "type": "DECIMAL(18,2)"},
                {"dictionary_name": "Quantity", "name": "qty", "type": "INTEGER"},
                {"dictionary_name": "Region  Code", "name": "region", "type": "VARCHAR"},
            ],
        },
        {
            "name": "colors",
            "dataset": "Neutral Colors",
            "columns": [
                {"dictionary_name": "Color Code", "name": "color_code", "type": "VARCHAR"},
                {"dictionary_name": "Description", "name": "description", "type": "VARCHAR"},
            ],
        },
    ]
}


@dataclass
class Source:
    root: Path
    published: Path
    profile: Path
    schema: Path
    db: Path

    def publish(self, name: str, text: str, count: int) -> None:
        """Serve ``text`` as pages of PAGE_SIZE rows, each with the header, then a short page."""
        header, *rows = text.splitlines(keepends=True)
        for offset in range(0, len(rows) + 1, PAGE_SIZE):
            page = header + "".join(rows[offset : offset + PAGE_SIZE])
            (self.published / f"{name}-{offset}-{PAGE_SIZE}.csv").write_text(page)
        (self.published / f"{name}.count.json").write_text(json.dumps([{"n": str(count)}]))

    def write_profile(self, profile: dict[str, object]) -> None:
        self.profile.write_text(yaml.safe_dump(profile))

    def args(self, command: str) -> list[str]:
        return [
            command, "--profile", str(self.profile), "--schema", str(self.schema),
            "--db", str(self.db), "--source-root", str(self.root),
        ]  # fmt: skip


def neutral_profile(published: Path) -> dict[str, object]:
    return {
        "date_format": "%d.%m.%Y",
        "page_url": f"file://{published}/{{dataset}}-{{offset}}-{{limit}}.csv",
        "page_size": PAGE_SIZE,
        "count_url": f"file://{published}/{{dataset}}.count.json",
        "relations": {
            "orders": {
                "dataset": "ord-0001",
                "headers": {"ship_date": "SHIP DATE", "qty": "QTY"},
                "ignore_headers": ["Unused Flag"],
            },
            "colors": {"file": "local/colors.csv"},
        },
    }


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("showcase.open_data_load.fetch_utils.RETRY_SECONDS", 0)


@pytest.fixture
def source(tmp_path: Path) -> Source:
    published = tmp_path / "published"
    published.mkdir()
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "colors.csv").write_text(COLORS_CSV)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA))
    built = Source(tmp_path, published, tmp_path / "profile.yaml", schema, tmp_path / "out.duckdb")
    built.publish("ord-0001", ORDERS_CSV, 3)
    built.write_profile(neutral_profile(published))
    return built
