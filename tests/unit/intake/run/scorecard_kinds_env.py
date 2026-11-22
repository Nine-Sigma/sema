"""Synthetic imports for the ``terms``, ``mappings``, and ``retirements`` scorer kinds.

Every value is invented: inline terms of one vocabulary with no definition and no external
refs (so no older kind writes a cell for them), SSSOM mapping records, and retirement records
in the authored evidence's shapes. Both sides stage nothing, so no ``terms:<table>`` kind
appears.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sema.intake.run.scorecard import KindScore, Scorecard, score

TERMS_KIND, MAPPINGS_KIND, RETIREMENTS_KIND = "terms", "mappings", "retirements"
NEW_KINDS = (TERMS_KIND, MAPPINGS_KIND, RETIREMENTS_KIND)
TIDES = "tide_kinds"
EXACT, CLOSE = "skos:exactMatch", "skos:closeMatch"
CURATED, LEXICAL = "semapv:ManualMappingCuration", "semapv:LexicalMatching"
NO_STAGING = ""

Record = dict[str, Any]
Change = Callable[[list[Record]], Any]


def term(code: str, display: str, definition: str | None = None) -> Record:
    return {
        "vocabulary": TIDES, "code": code, "display": display, "definition": definition,
        "synonyms": [], "external_refs": [],
    }


def oracle_terms() -> list[Record]:
    return [term("EBB", "Ebb"), term("FLD", "Flood"), term("SLK", "Slack")]


def manifest(terms: Sequence[Record]) -> Record:
    return {"manifest_version": "1", "terms": list(terms), "entities": [], "relationships": []}


def mapping(
    subject: str, predicate: str, object_: str, justification: str = CURATED
) -> Record:
    return {
        "subject_id": subject, "predicate_id": predicate, "object_id": object_,
        "mapping_justification": justification, "subject_label": None, "object_label": None,
        "confidence": None,
    }


def oracle_mappings() -> list[Record]:
    return [
        mapping("tide:EBB", EXACT, "ocean:0001"),
        mapping("tide:FLD", EXACT, "ocean:0002"),
        mapping("tide:SLK", CLOSE, "ocean:0003", LEXICAL),
    ]


def retirement(
    code: str, replaced_by: str | None, kind: str = "term", vocabulary: str | None = TIDES
) -> Record:
    return {
        "kind": kind, "vocabulary": vocabulary, "code": code, "replaced_by": replaced_by,
        "valid_from": None, "valid_to": None,
    }


def oracle_retirements() -> list[Record]:
    return [
        retirement("NEAP", "SLK"),
        retirement("SPRG", None),
        retirement("TIDE", None, kind="concept", vocabulary=None),
    ]


def authored(
    mappings: Sequence[Record] = (), retirements: Sequence[Record] = ()
) -> Record:
    return {
        "column_bindings": [], "value_mappings": [],
        "mappings": list(mappings), "retirements": list(retirements),
    }


def oracle_authored() -> Record:
    return authored(oracle_mappings(), oracle_retirements())


def changed(records: list[Record], change: Change | None) -> list[Record]:
    if change is not None:
        change(records)
    return records


def edit(at: int, **fields: Any) -> Change:
    return lambda records: records[at].update(fields)


def drop(at: int) -> Change:
    return lambda records: records.pop(at)


def add(record: Record) -> Change:
    return lambda records: records.append(record)


def reverse(records: list[Record]) -> None:
    records.reverse()


def score_terms(terms: Sequence[Record]) -> Scorecard:
    return score(manifest(oracle_terms()), manifest(terms), NO_STAGING, NO_STAGING)


def score_authored(evidence: Mapping[str, Any]) -> Scorecard:
    return score(
        manifest(()), manifest(()), NO_STAGING, NO_STAGING,
        expected_authored=oracle_authored(), authored=evidence,
    )


def counts(kind: KindScore) -> tuple[int, int, int, int]:
    """``(equal, missing, wrong, extra)``."""
    return kind.equal, kind.missing, kind.wrong, kind.extra


def moved(card: Scorecard) -> dict[str, tuple[int, int, int, int]]:
    """Every kind with a cell, by its counts."""
    return {name: counts(kind) for name, kind in card.kinds.items() if any(counts(kind))}
