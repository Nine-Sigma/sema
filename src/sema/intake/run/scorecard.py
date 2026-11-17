"""Accuracy of one import against another, cell by cell over the import's own shape.

An import is its manifest mapping, its term staging export text (typed header lines under
``## <table>`` marks), and optionally its authored evidence mapping. A cell is one value the
import writes: an entity, a property's field, an entity's obligation field or declared block,
a manifest relationship, a manifest term's field, an authored record, or one term staging row;
records sharing a key are numbered in sorted order, so each is its own cell. A cell counts only
when at least one side writes it, and scores ``equal``, ``wrong`` (both sides, different
values), ``missing`` (expected side only), or ``extra`` (assembled side only).
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from sema.binding.term_staging import TERM_STAGING_SCHEMA as STORE

ENTITY_KIND = "entities"
REF_FIELDS = ("system", "id")
PROPERTY_VALUES: dict[str, Callable[[Any], Any]] = {
    "type": lambda value: value,
    "nullable": lambda value: value,
    "definition": lambda value: value,
    "decoded_values": lambda value: value,
    "external_refs": lambda value: _records(value, REF_FIELDS),
}
OBLIGATION_KINDS = ("required_fields", "primary_key", "foreign_keys", "domain_constraints")
DECLARED_KIND = "declared"
RELATIONSHIP_KIND = "relationships"
TERM_KEY_FIELDS = ("vocabulary", "code")
TERM_VALUES: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "term_definition": lambda term: term.get("definition"),
    "term_external_refs": lambda term: _records(term.get("external_refs"), REF_FIELDS),
    "terms": lambda term: (term.get("display"),),
}
AUTHORED_KEY_FIELDS: dict[str, Key] = {
    "column_bindings": ("concept_ref", "dataset_label"),
    "value_mappings": ("concept_ref", "source_value"),
    "mappings": ("subject_id", "predicate_id", "object_id"),
    "retirements": ("kind", "vocabulary", "code"),
}
FIXED_KINDS = (
    ENTITY_KIND, *PROPERTY_VALUES, *OBLIGATION_KINDS, DECLARED_KIND, RELATIONSHIP_KIND,
    *TERM_VALUES, *AUTHORED_KEY_FIELDS,
)
TERM_KIND_PREFIX = "terms:"
TABLE_MARK = "## "
TYPE_SEPARATOR = ":"
TERM_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    STORE.concept_table: (STORE.vocab_col, STORE.code_col),
    STORE.relationship_table: (STORE.rel_from_col, STORE.rel_to_col, STORE.rel_id_col),
}
JOIN_FIELDS = ("referenced_entity", "join_keys")
DOMAIN_FIELDS = ("property_name", "domain_id")
OBLIGATION_VALUES: dict[str, Callable[[Any], Any]] = {
    "required_fields": lambda value: frozenset(value or ()),
    "primary_key": lambda value: value,
    "foreign_keys": lambda value: _records(value, JOIN_FIELDS),
    "domain_constraints": lambda value: _records(value, DOMAIN_FIELDS),
}
VERDICTS = ("equal", "missing", "wrong", "extra")
COLUMNS = ("precision", "recall", *VERDICTS)
OVERALL = "overall"
DIGITS = 4

Key = tuple[str, ...]
Cells = dict[Key, Any]


class KindScore(BaseModel):
    """Precision, recall, and verdict counts over every cell of one kind."""

    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    equal: int
    missing: int
    wrong: int
    extra: int


class Scorecard(BaseModel):
    """Per cell kind scores and the overall precision and recall over every cell."""

    model_config = ConfigDict(frozen=True)

    kinds: dict[str, KindScore]
    precision: float
    recall: float


def score(
    expected_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_staged: str,
    staged: str,
    *,
    expected_authored: Mapping[str, Any] | None = None,
    authored: Mapping[str, Any] | None = None,
) -> Scorecard:
    """Score the assembled import (``manifest``, ``staged``, ``authored``) against the
    expected one; authored evidence left out on a side writes no authored cells there."""
    expected = _cells(expected_manifest, expected_staged, expected_authored)
    actual = _cells(manifest, staged, authored)
    terms = sorted((expected.keys() | actual.keys()) - set(FIXED_KINDS))
    kinds = {
        kind: kind_score(verdicts(expected.get(kind, {}), actual.get(kind, {})))
        for kind in (*FIXED_KINDS, *terms)
    }
    overall = kind_score(_totals(kinds.values()))
    return Scorecard(kinds=kinds, precision=overall.precision, recall=overall.recall)


def render(card: Scorecard) -> str:
    """One aligned text row per cell kind and an overall row, under a column header."""
    totals = _totals(card.kinds.values())
    rows: list[tuple[str, ...]] = [("kind", *COLUMNS)]
    rows += [(name, *_row_cells(kind)) for name, kind in card.kinds.items()]
    rows.append((OVERALL, *_numbers(card.precision, card.recall, totals)))
    widths = [max(len(row[at]) for row in rows) for at in range(len(rows[0]))]
    return "\n".join(
        "  ".join(
            cell.ljust(width) if at == 0 else cell.rjust(width)
            for at, (cell, width) in enumerate(zip(row, widths, strict=True))
        ).rstrip()
        for row in rows
    )


def _row_cells(kind: KindScore) -> tuple[str, ...]:
    counts = Counter({verdict: getattr(kind, verdict) for verdict in VERDICTS})
    return _numbers(kind.precision, kind.recall, counts)


def _numbers(precision: float, recall: float, counts: Counter[str]) -> tuple[str, ...]:
    return (
        f"{precision:.{DIGITS}f}",
        f"{recall:.{DIGITS}f}",
        *(str(counts[verdict]) for verdict in VERDICTS),
    )


def _totals(kinds: Iterable[KindScore]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for kind in kinds:
        totals.update({verdict: getattr(kind, verdict) for verdict in VERDICTS})
    return totals


def kind_score(counts: Counter[str]) -> KindScore:
    equal, missing, wrong, extra = (counts[verdict] for verdict in VERDICTS)
    return KindScore(
        precision=_ratio(equal, equal + wrong + extra),
        recall=_ratio(equal, equal + wrong + missing),
        equal=equal,
        missing=missing,
        wrong=wrong,
        extra=extra,
    )


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 1.0


def verdicts(expected: Cells, actual: Cells) -> Counter[str]:
    verdicts: Counter[str] = Counter()
    for key in expected.keys() | actual.keys():
        verdict = _verdict(expected.get(key), actual.get(key))
        if verdict is not None:
            verdicts[verdict] += 1
    return verdicts


def _verdict(expected: Any, actual: Any) -> str | None:
    has_expected, has_actual = _written(expected), _written(actual)
    if has_expected and has_actual:
        return "equal" if expected == actual else "wrong"
    if has_expected:
        return "missing"
    return "extra" if has_actual else None


def _written(value: Any) -> bool:
    if isinstance(value, Collection):
        return bool(value)
    return value is not None


def _frozen(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(name), _frozen(item)) for name, item in value.items()))
    return value


def _records(items: Iterable[Mapping[str, Any]] | None, fields: Key) -> frozenset[Any]:
    return frozenset(tuple(_frozen(item[name]) for name in fields) for item in items or ())


def _numbered(records: Iterable[tuple[Key, Any]]) -> Cells:
    occurrences: Counter[Key] = Counter()
    cells: Cells = {}
    for key, value in sorted(records, key=lambda record: repr(_canonical(record))):
        cells[(*key, str(occurrences[key]))] = value
        occurrences[key] += 1
    return cells


def _canonical(value: Any) -> Any:
    if isinstance(value, frozenset):
        return tuple(sorted((_canonical(item) for item in value), key=repr))
    if isinstance(value, tuple):
        return tuple(_canonical(item) for item in value)
    return value


def _key(record: Mapping[str, Any], fields: Key) -> Key:
    return tuple("" if record.get(name) is None else str(record[name]) for name in fields)


def _cells(
    manifest: Mapping[str, Any], staged: str, authored: Mapping[str, Any] | None
) -> dict[str, Cells]:
    return (
        _manifest_cells(manifest)
        | _record_cells(manifest)
        | _authored_cells(authored or {})
        | _staged_cells(staged)
    )


def _declared(entity: Mapping[str, Any]) -> tuple[frozenset[Any], frozenset[Any]] | None:
    joins = _records(entity.get("joins"), JOIN_FIELDS)
    domains = _records(entity.get("domains"), DOMAIN_FIELDS)
    return (joins, domains) if joins or domains else None


def _manifest_cells(manifest: Mapping[str, Any]) -> dict[str, Cells]:
    cells: dict[str, Cells] = {kind: {} for kind in FIXED_KINDS}
    for entity in manifest.get("entities") or ():
        name = str(entity["qualified_name"])
        cells[ENTITY_KIND][(name,)] = True
        for prop in entity.get("properties") or ():
            for kind, value_of in PROPERTY_VALUES.items():
                cells[kind][(name, str(prop["name"]))] = value_of(prop.get(kind))
        obligation = entity.get("obligation") or {}
        for kind, value_of in OBLIGATION_VALUES.items():
            cells[kind][(name,)] = value_of(obligation.get(kind))
        cells[DECLARED_KIND][(name,)] = _declared(entity)
    return cells


def _record_cells(manifest: Mapping[str, Any]) -> dict[str, Cells]:
    relationships = manifest.get(RELATIONSHIP_KIND) or ()
    terms = manifest.get("terms") or ()
    cells = {
        kind: _numbered((_key(term, TERM_KEY_FIELDS), value_of(term)) for term in terms)
        for kind, value_of in TERM_VALUES.items()
    }
    cells[RELATIONSHIP_KIND] = _numbered(
        (_relationship_key(record), _frozen(record)) for record in relationships
    )
    return cells


def _relationship_key(record: Mapping[str, Any]) -> Key:
    ends = (record.get("from") or {}, record.get("to") or {})
    return (*(part for end in ends for part in _key(end, REF_FIELDS)), str(record["type"]))


def _authored_cells(authored: Mapping[str, Any]) -> dict[str, Cells]:
    return {
        kind: _numbered(
            (_key(record, fields), _frozen(record)) for record in authored.get(kind) or ()
        )
        for kind, fields in AUTHORED_KEY_FIELDS.items()
    }


def _staged_cells(text: str) -> dict[str, Cells]:
    return {
        f"{TERM_KIND_PREFIX}{table}": _term_rows(table, rows)
        for table, rows in _term_tables(text).items()
    }


def _term_tables(text: str) -> dict[str, list[list[str]]]:
    tables: dict[str, list[list[str]]] = {}
    current: list[list[str]] = []
    for row in csv.reader(io.StringIO(text)):
        if row and row[0].startswith(TABLE_MARK):
            current = tables.setdefault(row[0].removeprefix(TABLE_MARK), [])
        elif row:
            current.append(row)
    return tables


def _term_rows(table: str, rows: list[list[str]]) -> Cells:
    if not rows:
        return {}
    header = [cell.rsplit(TYPE_SEPARATOR, 1)[0] for cell in rows[0]]
    keys = TERM_KEY_COLUMNS.get(table, ())
    if keys and set(keys) <= set(header):
        indexes = [header.index(column) for column in keys]
    else:
        indexes = list(range(len(header)))
    return _numbered((tuple(row[at] for at in indexes), row) for row in rows[1:])
