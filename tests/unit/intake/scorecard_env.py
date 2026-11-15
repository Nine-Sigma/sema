"""Scorecard floors per known-source leg, and synthetic imports for the scorer's tests.

``BASELINE`` is keyed ``(source, cap_id)``, cap ids from ``test_no_hint.CAP_VALUES``; a floor holds
overall precision and recall at or above, and ignored relations at or below, its values (design 2).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from sema.intake.scorecard import Scorecard
from tests.unit.intake.oracle_review_env import ORACLES as ORACLES, UNREVIEWED as UNREVIEWED
from tests.unit.intake.oracle_review_env import oracle_review as oracle_review
from tests.unit.intake.oracle_review_env import unreviewed_floors as unreviewed_floors

DECLARED_EVIDENCE = "declared"


@dataclass(frozen=True)
class Floor:
    precision: float
    recall: float
    ignored: int


BASELINE: dict[tuple[str, str], Floor] = {
    # honest floor: the four relations the decisions file ignores carry concept_details,
    # term_details, relationships, value_mappings, authored_bindings; measured at 20a704b.
    # stage_all writes no term definitions or term external refs (term_staging.staged_tables writes
    # id, display, vocabulary, code, lifecycle, replaced_by, synonyms only), so those two
    # kinds have zero cells and the stage_all floor leaves them out by construction (carried gap).
    ("a1", "spec_cap"): Floor(precision=77 / 97, recall=77 / 129, ignored=4),
    ("a1", "stage_all"): Floor(precision=1.0, recall=81 / 101, ignored=4),
    # 4.4 leftover, declared 13/14: PROPERTY TYPE domain reference settled none (IMPORT_LEFTOVERS)
    ("a10", "spec_cap"): Floor(precision=1.0, recall=896 / 897, ignored=30),
    ("a10", "stage_all"): Floor(precision=1.0, recall=896 / 897, ignored=30),
    ("a2", "spec_cap"): Floor(precision=1.0, recall=1.0, ignored=0),
    ("a2", "stage_all"): Floor(precision=1.0, recall=1.0, ignored=0),
    ("a3", "spec_cap"): Floor(precision=1.0, recall=1.0, ignored=0),
    ("a3", "stage_all"): Floor(precision=1.0, recall=1.0, ignored=0),
    ("a7_csv", "spec_cap"): Floor(precision=1.0, recall=1.0, ignored=0),
    ("a7_csv", "stage_all"): Floor(precision=1.0, recall=1.0, ignored=0),
    # honest floor against the correct oracle (codes are the concepts; 2026-10-23): the decided
    # run models the four levels as concepts and the codes as one value set, so one cell matches.
    # The code-prefix hierarchy is not expressible in spec_version 2 and is unscored until S6.
    ("heldout4", "spec_cap"): Floor(precision=1 / 5478, recall=1 / 5110, ignored=1),
    ("heldout4", "stage_all"): Floor(precision=1 / 5478, recall=1 / 5110, ignored=1),
}


def floor_misses(card: Scorecard, ignored: int, floor: Floor) -> tuple[str, ...]:
    """Every deviation from the floor either way, so a fix fails until its floor is raised."""
    misses: list[str] = []
    for name, value, held in (
        ("precision", card.precision, floor.precision),
        ("recall", card.recall, floor.recall),
    ):
        if value < held:
            misses.append(f"{name} {value} is below its floor {held}")
        elif value > held:
            misses.append(f"{name} {value} is above its floor {held}; raise the floor")
    if ignored > floor.ignored:
        misses.append(f"ignored {ignored} is above its floor {floor.ignored}")
    elif ignored < floor.ignored:
        misses.append(f"ignored {ignored} is below its floor {floor.ignored}; lower the floor")
    return tuple(misses)


def prop(
    name: str,
    type_: str = "string",
    *,
    nullable: bool = True,
    definition: str | None = None,
    decoded_values: Mapping[str, str] | None = None,
    external_refs: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    built: dict[str, Any] = {"name": name, "type": type_, "nullable": nullable}
    if definition is not None:
        built["definition"] = definition
    if decoded_values is not None:
        built["decoded_values"] = dict(decoded_values)
    if external_refs:
        built["external_refs"] = list(external_refs)
    return built


def ref(system: str, id_: str) -> dict[str, str]:
    return {"system": system, "id": id_}


def term(
    code: str, display: str, definition: str | None = None, *refs: dict[str, str]
) -> dict[str, Any]:
    return {
        "vocabulary": RIG_KINDS, "code": code, "display": display, "definition": definition,
        "synonyms": [], "external_refs": list(refs),
    }


def relationship(start: str, end: str, type_: str, hierarchical: bool | None) -> dict[str, Any]:
    return {
        "from": ref(RIG_KINDS, start), "to": ref(RIG_KINDS, end), "type": type_,
        "hierarchical": hierarchical, "far_end_loaded": True,
    }


def binding(concept_ref: str, *columns: str) -> dict[str, Any]:
    return {
        "concept_ref": concept_ref, "dataset_label": "fleet", "dataset_id": None,
        "physical": None, "columns": list(columns), "resolved": False, "project": None,
    }


def value_mapping(concept_ref: str, source_value: str, *codes: str) -> dict[str, Any]:
    return {
        "concept_ref": concept_ref, "source_value": source_value, "codes": list(codes),
        "status": None, "approved": False,
    }


def oracle_authored() -> dict[str, Any]:
    bindings = [binding("rig", "rig"), binding("hull", "hull_code")]
    return {"column_bindings": bindings, "value_mappings": [value_mapping("rig", "S", "S")]}


def fk(referenced_entity: str, *pairs: tuple[str, str]) -> dict[str, Any]:
    return {"referenced_entity": referenced_entity, "join_keys": [list(p) for p in pairs]}


def domain(property_name: str, domain_id: str) -> dict[str, Any]:
    return {"property_name": property_name, "domain_id": domain_id}


def obligation(
    required_fields: Sequence[str],
    primary_key: str = "NATURAL_KEY",
    *,
    foreign_keys: Sequence[dict[str, Any]] = (),
    domain_constraints: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    built: dict[str, Any] = {
        "required_fields": list(required_fields),
        "primary_key": primary_key,
    }
    if foreign_keys:
        built["foreign_keys"] = list(foreign_keys)
    if domain_constraints:
        built["domain_constraints"] = list(domain_constraints)
    return built


def entity(
    qualified_name: str,
    *properties: dict[str, Any],
    obligation: dict[str, Any] | None = None,
    joins: Sequence[dict[str, Any]] = (),
    domains: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """An entity as the import writes it: an obligation, or top-level declared joins and
    domains (each tagged with the declared evidence) when no key is stated."""
    built: dict[str, Any] = {
        "qualified_name": qualified_name,
        "kind": "table_row",
        "properties": list(properties),
    }
    if obligation is not None:
        built["obligation"] = obligation
    evidence = {"evidence": DECLARED_EVIDENCE}
    if joins:
        built["joins"] = [join | evidence for join in joins]
    if domains:
        built["domains"] = [item | evidence for item in domains]
    return built


def manifest(
    *entities: dict[str, Any],
    terms: Sequence[dict[str, Any]] = (),
    relationships: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    ordered = sorted(entities, key=lambda e: str(e["qualified_name"]))
    return {
        "manifest_version": "1", "terms": list(terms), "entities": ordered,
        "relationships": list(relationships),
    }


@dataclass(frozen=True)
class Table:
    """One term staging table: typed header cells (``name:TYPE``) and its rows."""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...] = ()


def export(tables: Mapping[str, Table]) -> str:
    """Term staging export text in ``export_schema``'s format: tables in name order, each a
    ``## <table>`` line, the typed header, then the sorted rows."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for name in sorted(tables):
        writer.writerow([f"## {name}"])
        writer.writerow(tables[name].header)
        writer.writerows(sorted(tables[name].rows))
    return out.getvalue()


VESSEL, VOYAGE, CREW, HARBOR = "fleet.vessel", "fleet.voyage", "fleet.crew", "fleet.harbor"
RIG_KINDS, REGISTRY = "rig_kinds", "sail_registry"
CONCEPT, RELATIONSHIP, SYNONYM = "concept", "concept_relationship", "concept_synonym"
CONCEPT_HEADER = (
    "concept_id:VARCHAR", "concept_name:VARCHAR", "domain_id:VARCHAR",
    "vocabulary_id:VARCHAR", "standard_concept:VARCHAR", "concept_code:VARCHAR",
    "valid_start_date:DATE", "valid_end_date:DATE", "invalid_reason:VARCHAR",
)
RELATIONSHIP_HEADER = ("concept_id_1:VARCHAR", "concept_id_2:VARCHAR", "relationship_id:VARCHAR")
SYNONYM_HEADER = ("concept_id:VARCHAR", "concept_synonym_name:VARCHAR", "language:VARCHAR")


def oracle_manifest() -> dict[str, Any]:
    """Three invented entities: a stated single key with a domain, a compound key with a
    foreign key, and no stated key with declared joins and domains; three terms and two
    relationships between them."""
    vessel = entity(
        VESSEL,
        prop("hull_code", nullable=False, definition="Registry code of the hull"),
        prop("tonnage", "integer", definition="Gross tonnage"),
        prop(
            "rig", decoded_values={"S": "Sloop", "K": "Ketch"},
            external_refs=[ref(REGISTRY, "R-1")],
        ),
        obligation=obligation(["hull_code"], domain_constraints=[domain("rig", "rig_kinds")]),
    )
    voyage = entity(
        VOYAGE,
        prop("voyage_id", nullable=False),
        prop("hull_code", nullable=False),
        prop("port", definition="Port of departure"),
        obligation=obligation(
            ["voyage_id", "hull_code"],
            "COMPOUND",
            foreign_keys=[fk(VESSEL, ("hull_code", "hull_code"))],
        ),
    )
    crew = entity(
        CREW,
        prop("member"),
        prop("hull_code"),
        joins=[fk(VESSEL, ("hull_code", "hull_code"))],
        domains=[domain("member", "crew_roles")],
    )
    terms = [
        term("S", "Sloop", "Single-masted", ref(REGISTRY, "S-1")),
        term("K", "Ketch", "Two-masted"),
        term("Y", "Yawl"),
    ]
    links = [relationship("Y", "K", "succeeded_by", None), relationship("S", "K", "is_a", True)]
    return manifest(vessel, voyage, crew, terms=terms, relationships=links)


def concept_row(code: str, name: str) -> tuple[str, ...]:
    return (
        f"rig_kinds:{code}", name, "Rig", "rig_kinds", "", code, "1970-01-01", "2099-12-31", ""
    )


def oracle_tables() -> dict[str, Table]:
    concepts = (concept_row("S", "Sloop"), concept_row("K", "Ketch"), concept_row("Y", "Yawl"))
    return {
        CONCEPT: Table(CONCEPT_HEADER, concepts),
        RELATIONSHIP: Table(RELATIONSHIP_HEADER, (("rig_kinds:Y", "rig_kinds:K", "succeeded_by"),)),
        SYNONYM: Table(SYNONYM_HEADER, (("rig_kinds:S", "Single mast", "en"),)),
    }


def entity_of(built: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = [
        e for e in built["entities"] if e["qualified_name"] == name
    ]
    (found,) = matches
    return found


def prop_of(built: Mapping[str, Any], entity_name: str, name: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = [
        p for p in entity_of(built, entity_name)["properties"] if p["name"] == name
    ]
    (found,) = matches
    return found


Manifest = dict[str, Any]


def drop_prop(entity_name: str, name: str) -> Callable[[Manifest], None]:
    def change(built: Manifest) -> None:
        found = entity_of(built, entity_name)
        found["properties"] = [p for p in found["properties"] if p["name"] != name]

    return change


def change_definitions(built: Manifest) -> None:
    del prop_of(built, VESSEL, "tonnage")["definition"]
    prop_of(built, VESSEL, "hull_code")["definition"] = "Paint colour of the hull"
    prop_of(built, VESSEL, "rig")["definition"] = "Sail plan"


def change_decoded(built: Manifest) -> None:
    prop_of(built, VESSEL, "rig")["decoded_values"] = {"S": "Sloop"}
    prop_of(built, VOYAGE, "port")["decoded_values"] = {"N": "North quay"}


PROPERTY_CASES: list[Any] = [
    pytest.param(
        lambda m: prop_of(m, VESSEL, "tonnage").update(type="string"), "type", (7, 0, 1, 0),
        id="type-wrong",
    ),
    pytest.param(
        lambda m: prop_of(m, VOYAGE, "voyage_id").update(nullable=True), "nullable",
        (7, 0, 1, 0), id="nullable-wrong",
    ),
    pytest.param(change_definitions, "definition", (1, 1, 1, 1), id="definition-every-verdict"),
    pytest.param(change_decoded, "decoded_values", (0, 0, 1, 1), id="decoded-values-wrong-extra"),
    pytest.param(
        lambda m: prop_of(m, VESSEL, "rig").pop("decoded_values"), "decoded_values",
        (0, 1, 0, 0), id="decoded-values-missing",
    ),
    pytest.param(drop_prop(VESSEL, "tonnage"), "type", (7, 1, 0, 0), id="property-absent"),
    pytest.param(
        drop_prop(VESSEL, "tonnage"), "definition", (2, 1, 0, 0), id="property-absent-def"
    ),
]


def undeclared_key(built: Manifest) -> None:
    voyage = entity_of(built, VOYAGE)
    voyage.pop("obligation")
    voyage["joins"] = [fk(VESSEL, ("hull_code", "hull_code")) | {"evidence": "declared"}]


ENTITY_CASES: list[Any] = [
    pytest.param(
        lambda m: entity_of(m, VOYAGE)["obligation"]["required_fields"].reverse(),
        {"required_fields": (2, 0, 0, 0)}, id="required-fields-are-a-set",
    ),
    pytest.param(
        lambda m: entity_of(m, VOYAGE)["obligation"].update(required_fields=["voyage_id"]),
        {"required_fields": (1, 0, 1, 0)}, id="required-fields-wrong",
    ),
    pytest.param(
        lambda m: entity_of(m, VOYAGE)["obligation"].update(primary_key="NATURAL_KEY"),
        {"primary_key": (1, 0, 1, 0)}, id="primary-key-wrong",
    ),
    pytest.param(
        lambda m: entity_of(m, VOYAGE)["obligation"].update(
            foreign_keys=[fk(VESSEL, ("voyage_id", "hull_code"))]
        ),
        {"foreign_keys": (0, 0, 1, 0)}, id="foreign-keys-wrong",
    ),
    pytest.param(
        lambda m: entity_of(m, VESSEL)["obligation"].update(
            foreign_keys=[fk(VOYAGE, ("hull_code", "hull_code"))]
        ),
        {"foreign_keys": (1, 0, 0, 1)}, id="foreign-keys-extra",
    ),
    pytest.param(
        lambda m: entity_of(m, VESSEL)["obligation"].update(
            domain_constraints=[domain("rig", "hull_kinds")]
        ),
        {"domain_constraints": (0, 0, 1, 0)}, id="domain-constraints-wrong",
    ),
    pytest.param(
        lambda m: entity_of(m, VESSEL)["obligation"].pop("domain_constraints"),
        {"domain_constraints": (0, 1, 0, 0)}, id="domain-constraints-missing",
    ),
    pytest.param(
        lambda m: entity_of(m, CREW).pop("joins"), {"declared": (0, 0, 1, 0)},
        id="declared-wrong",
    ),
    pytest.param(
        lambda m: [entity_of(m, CREW).pop(key) for key in ("joins", "domains")],
        {"declared": (0, 1, 0, 0)}, id="declared-missing",
    ),
    pytest.param(
        undeclared_key,
        {
            "required_fields": (1, 1, 0, 0), "primary_key": (1, 1, 0, 0),
            "foreign_keys": (0, 1, 0, 0), "declared": (1, 0, 0, 1),
        },
        id="key-not-stated",
    ),
]
