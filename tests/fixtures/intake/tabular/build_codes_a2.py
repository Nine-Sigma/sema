"""Write ``codes_a2.xlsx``, the A2 tabular fixture, byte for byte reproducibly.

Run: ``uv run python tests/fixtures/intake/tabular/build_codes_a2.py``.

Sheets:
- ``Concepts``: a title in row 1, a blank row 2, the header on row 3, all-text
  columns, and one deliberately empty column (no header, no values).
- ``Values``: header on row 1, one value set row per code, an all-integer
  ``rank`` column, and lifecycle columns (one deprecated code with a successor).
- ``Hierarchy``: level columns ``l1``..``l3`` over the concept codes.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import openpyxl

OUT = Path(__file__).with_name("codes_a2.xlsx")
FIXED_TIME = datetime(2026, 10, 20, 0, 0, 0)

CONCEPTS = (
    ("concept_code", "concept_name", None, "definition", "data_type"),
    ("ATTR", "Attribute", None, "Any attribute of an item", "string"),
    ("COLOR", "Color", None, "Colour of an item", "string"),
    ("DIM", "Dimension", None, "A measurable extent", "string"),
    ("SIZE", "Size", None, "Size of an item", "string"),
    ("SHAPE", "Shape", None, "Outline of an item", "string"),
)

VALUES = (
    ("concept_code", "code", "label", "rank", "status", "successor"),
    ("COLOR", "RED", "Red", 1, "active", None),
    ("COLOR", "GREEN", "Green", 2, "active", None),
    ("COLOR", "BLUE", "Blue", 3, "active", None),
    ("COLOR", "TEAL", "Teal (old)", 4, "deprecated", "BLUE"),
    ("SIZE", "S", "Small", 1, "active", None),
    ("SIZE", "M", "Medium", 2, "active", None),
    ("SIZE", "L", "Large", 3, "active", None),
    ("SHAPE", "ROUND", "Round", 1, "active", None),
    ("SHAPE", "SQUARE", "Square", 2, "active", None),
)

HIERARCHY = (
    ("l1", "l2", "l3"),
    ("ATTR", "COLOR", None),
    ("ATTR", "DIM", "SIZE"),
    ("ATTR", "DIM", "SHAPE"),
)


def build() -> bytes:
    workbook = openpyxl.Workbook()
    concepts = workbook.active
    assert concepts is not None
    concepts.title = "Concepts"
    concepts.append(["Synthetic code library, release 2026-10"])
    concepts.append([])
    for row in CONCEPTS:
        concepts.append(list(row))
    for title, rows in (("Values", VALUES), ("Hierarchy", HIERARCHY)):
        sheet = workbook.create_sheet(title)
        for row in rows:
            sheet.append(list(row))
    workbook.properties.created = FIXED_TIME
    workbook.properties.modified = FIXED_TIME
    buffer = io.BytesIO()
    workbook.save(buffer)
    return _fixed_zip_times(buffer.getvalue())


def _fixed_zip_times(data: bytes) -> bytes:
    """openpyxl stamps zip entries with the current time; pin them."""
    source = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            pinned = zipfile.ZipInfo(info.filename, date_time=FIXED_TIME.timetuple()[:6])
            pinned.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(pinned, source.read(info.filename))
    return out.getvalue()


if __name__ == "__main__":
    OUT.write_bytes(build())
    print(f"wrote {OUT}")
