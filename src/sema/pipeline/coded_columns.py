"""Build step: coded columns, their value sets and labels, verified on the data.

Runs once every table of the run is built and the joins are detected. Nominates code sources
for each low-cardinality column (declared keys, equal names, the data shape, and the domains
the concept spec states), decides each column by containment, and writes the value sets,
their members, the label edges and the evidence. A verified binding to a code relation is a
join: it is returned for the join materializer.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sema.binding.importer import import_concept_source
from sema.binding.spec import load_concept_source_spec
from sema.connectors.base import Connector
from sema.engine.coded_column_utils import (
    DECLARED_KEY,
    NAME_MATCH,
    STATED_DOMAIN,
    VERIFIED,
    Binding,
    Candidate,
    CodeSource,
    StatedDomain,
    binding_assertions,
    decide,
    determines,
    majority_labels,
    place,
    unplaced_assertion,
)
from sema.engine.join_detector import TIER_1, FKAssertion, to_fk_assertion
from sema.engine.join_detector_utils import (
    Profiles,
    column_key,
    containment,
    fk_candidate,
    is_code_key,
    is_text,
    nominate_by_data_shape,
    nominate_by_name,
)
from sema.engine.warehouse_lookup import WarehouseProfileLookup, WarehouseValues
from sema.graph.concept_value_set import (
    CodedMember,
    ValueSetColumn,
    link_label_column,
    materialize_coded_value_set,
)
from sema.graph.join_materializer_utils import (
    fetch_join_assertions,
    join_keys,
    join_path_name,
    normalize_fk_to_assertion,
)
from sema.graph.loader import GraphLoader
from sema.graph.loader_utils import fetch_columns_by_schema
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.relations import RelationProvider
from sema.log import logger
from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.config import BuildConfig
from sema.models.extraction import ExtractedColumn

_VERIFIED_TERM_CONFIDENCE = 0.95
_UNVERIFIED_TERM_CONFIDENCE = 0.5


@dataclass(frozen=True)
class Dictionary:
    """What intake's concept spec states: its domains, and each code list's codes and labels."""

    spec_id: str
    domains: tuple[StatedDomain, ...]
    code_lists: dict[str, dict[str, str | None]]


@dataclass(frozen=True)
class LabelEdge:
    label: ExtractedColumn
    violation_rate: float
    via: tuple[str, ...]


@dataclass
class _Run:
    loader: GraphLoader
    values: WarehouseValues
    dictionary: Dictionary | None
    columns: list[ExtractedColumn]
    run_id: str
    schema: str
    joins: list[Assertion] = field(default_factory=list)


def read_dictionary(spec_path: str | None, staging_path: str | None) -> Dictionary | None:
    if spec_path is None:
        return None
    if staging_path is None:
        raise ValueError("concept_spec needs concept_staging, the file its relations live in")
    spec = load_concept_source_spec(Path(spec_path))
    with Staging(staging_path, schemas=()) as staging:
        provider = RelationProvider(warehouse=None, staging=staging)
        terms = import_concept_source(spec, provider, dataset_map={}).manifest["terms"]
    code_lists: dict[str, dict[str, str | None]] = {}
    for term in terms:
        code_lists.setdefault(term["vocabulary"], {})[str(term["code"])] = term.get("display")
    # shortcut: domains are read from per-file entries' `domains` maps only; read a role
    # column's domain too once a source states its domains that way.
    entries = (spec.entity_model.entities or []) if spec.entity_model else []
    domains = tuple(
        StatedDomain(entry.entity, prop, vocabulary)
        for entry in entries for prop, vocabulary in entry.domains.items()
    )
    return Dictionary(spec.id, domains, code_lists)


def run_coded_columns(
    loader: GraphLoader, connector: Connector, config: BuildConfig, schemas: list[str],
    run_id: str,
) -> dict[str, list[Assertion]]:
    """Write every schema's coded columns; return the joins their verified bindings prove."""
    if not config.enable_fk_detection:
        return {}
    dictionary = read_dictionary(config.concept_spec, config.concept_staging)
    profile_lookup = WarehouseProfileLookup(connector.execute_query, config.catalog)
    values = WarehouseValues(connector.execute_query, config.catalog)
    bound = config.profiling.categorical_threshold
    joins: dict[str, list[Assertion]] = {}
    for schema in schemas:
        columns = fetch_columns_by_schema(loader, schema)
        profiles = {
            column_key(c): stats for c in columns
            if (stats := profile_lookup(column_key(c))) is not None
        }
        run = _Run(loader, values, dictionary, columns, run_id, schema)
        for column, candidates in _candidates(run, profiles, bound).items():
            _bind(run, column, candidates)
        joins[schema] = run.joins
    return joins


def _within(profiles: Profiles, column: ExtractedColumn, bound: int) -> bool:
    stats = profiles.get(column_key(column))
    return stats is not None and 0 < stats[0] <= bound


def _relation(key: ExtractedColumn) -> CodeSource:
    return CodeSource(key.ref or "", f"{key.schema}.{key.table_name}.{key.name}", key)


def _candidates(
    run: _Run, profiles: Profiles, bound: int,
) -> dict[str, list[Candidate]]:
    """Each coded column's candidates, keyed by column ref; unplaced domains are recorded."""
    by_ref = {c.ref: c for c in run.columns if c.ref}
    found: dict[tuple[str, str], Candidate] = {}

    def add(column: ExtractedColumn, source: CodeSource, kind: str | None, **detail: Any) -> None:
        candidate = found.setdefault((column.ref or "", source.ref), Candidate(column, source))
        if kind is not None:
            candidate.evidence[kind] = detail

    for fk in nominate_by_data_shape(run.columns, profiles, bound):
        add(by_ref[fk.fk_ref], _relation(by_ref[fk.pk_ref]), None)
    for fk in nominate_by_name(run.columns, profiles, bound):
        add(by_ref[fk.fk_ref], _relation(by_ref[fk.pk_ref]), NAME_MATCH)
    for column, key, constraint in _declared(run, profiles, bound):
        add(column, _relation(key), DECLARED_KEY, constraint=constraint)
    dictionary = run.dictionary
    for domain in dictionary.domains if dictionary else ():
        named = place(domain, run.columns)
        if dictionary and not named:
            run.loader.store_assertion(
                unplaced_assertion(dictionary.spec_id, domain, run.run_id, run.schema),
                source_schema=run.schema,
            )
        for column in (c for c in named if _within(profiles, c, bound)):
            source = CodeSource(domain.vocabulary, domain.vocabulary)
            add(column, source, STATED_DOMAIN, property=domain.property, entity=domain.entity)
    by_column: dict[str, list[Candidate]] = {}
    for (column_ref, _), candidate in sorted(found.items()):
        by_column.setdefault(column_ref, []).append(candidate)
    return by_column


def _declared(
    run: _Run, profiles: Profiles, bound: int,
) -> Iterable[tuple[ExtractedColumn, ExtractedColumn, str]]:
    """Single-column declared keys to a code relation within the bound."""
    by_name = {(c.table_name, c.name): c for c in run.columns}
    for a in fetch_join_assertions(run.loader, run.schema):
        predicates = a.payload.get("join_predicates") or []
        constraint = a.payload.get("constraint_name")
        if a.predicate != AssertionPredicate.JOINS_TO or not constraint or len(predicates) != 1:
            continue
        (p,) = predicates
        key = by_name.get((p["left_table"], p["left_column"]))
        column = by_name.get((p["right_table"], p["right_column"]))
        if key is None or column is None or key.table_name == column.table_name:
            continue
        if is_code_key(profiles.get(column_key(key)), bound) and _within(profiles, column, bound):
            yield column, key, str(constraint)


def _listed(run: _Run, source: CodeSource) -> dict[str, str | None] | None:
    """The source's codes, each with its dictionary label (None for a relation's codes)."""
    if source.key is None:
        return run.dictionary.code_lists.get(source.ref, {}) if run.dictionary else {}
    counts = run.values.counts(column_key(source.key))
    return None if counts is None else {code: None for code in counts if code is not None}


def _bind(run: _Run, column_ref: str, candidates: list[Candidate]) -> None:
    column = candidates[0].column
    counts = run.values.counts(column_key(column))
    if counts is None:
        logger.warning(f"No value counts for {column_ref}; its value set is not decided")
        return
    listed = {c.source.ref: _listed(run, c.source) for c in candidates}
    scored = [
        (c, containment(counts, codes)) for c in candidates
        if (codes := listed[c.source.ref]) is not None
    ]
    for binding in decide(scored):
        for assertion in binding_assertions(binding, run.run_id, run.schema):
            run.loader.store_assertion(assertion, source_schema=run.schema)
        if binding.linked:
            _link(run, binding, counts, listed[binding.candidate.source.ref] or {})


def _link(
    run: _Run, binding: Binding, counts: dict[str | None, int], listed: dict[str, str | None],
) -> None:
    column, source = binding.candidate.column, binding.candidate.source
    verified = binding.verification == VERIFIED
    labels, label_edge = _labels(run, binding, listed) if verified else ({}, None)
    observed = sorted(code for code in counts if code is not None and code not in listed)
    members = [
        CodedMember(code, labels.get(code), counts.get(code, 0), code in listed)
        for code in [*listed, *observed]
    ]
    measured = binding.containment
    edge = {
        "verification": binding.verification,
        "listed_value_share": measured.listed_value_share,
        "listed_row_share": measured.listed_row_share,
        "observed_distinct": measured.observed_distinct, "null_rows": measured.null_rows,
    }
    materialize_coded_value_set(
        run.loader,
        column=ValueSetColumn(column.catalog, column.schema, column.table_name, column.name),
        vocabulary=source.vocabulary, members=members, edge=edge, source_schema=run.schema,
        confidence=_VERIFIED_TERM_CONFIDENCE if verified else _UNVERIFIED_TERM_CONFIDENCE,
    )
    if label_edge is not None:
        link_label_column(
            run.loader, column_ref=column.ref or "", label_ref=label_edge.label.ref or "",
            edge={"verification": VERIFIED, "violation_rate": label_edge.violation_rate,
                  "via": list(label_edge.via)},
            source_schema=run.schema,
        )


def _labels(
    run: _Run, binding: Binding, listed: dict[str, str | None],
) -> tuple[dict[str, str | None], LabelEdge | None]:
    """A dictionary's labels, else the code relation's one text column, else one sibling."""
    column, key = binding.candidate.column, binding.candidate.source.key
    if key is None and any(listed.values()):
        return listed, None
    if key is not None:
        join = to_fk_assertion(FKAssertion(fk_candidate(key, column), TIER_1, 1, run.schema),
                               run.run_id)
        run.joins.append(join)
        texts = _texts(run, key)
        if len(texts) > 1:
            return {}, None
        if texts:
            rate, labels = majority_labels(_pairs(run, key, texts[0]))
            return labels, LabelEdge(texts[0], rate or 0.0, (_join_name(join),))
    determined = [
        (text, rate, labels) for text in _texts(run, column)
        for rate, labels in [majority_labels(_pairs(run, column, text))]
        if rate is not None and determines(rate)
    ]
    if len(determined) != 1:
        return {}, None
    ((text, rate, labels),) = determined
    return labels, LabelEdge(text, rate, ())


def _texts(run: _Run, beside: ExtractedColumn) -> list[ExtractedColumn]:
    return [
        c for c in run.columns
        if c.table_name == beside.table_name and c.name != beside.name and is_text(c)
    ]


def _pairs(
    run: _Run, code: ExtractedColumn, text: ExtractedColumn,
) -> dict[tuple[str | None, str | None], int]:
    return run.values.pairs(run.schema, code.table_name, code.name, text.name) or {}


def _join_name(join: Assertion) -> str:
    normalized = normalize_fk_to_assertion(join)
    keys = join_keys(normalized)
    if keys is None:
        return ""
    return join_path_name(normalized.payload["join_predicates"], *keys)
