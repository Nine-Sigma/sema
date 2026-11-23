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
from typing import Any

from sema.connectors.base import Connector
from sema.engine.coded_column_utils import (
    DECLARED_KEY,
    NAME_MATCH,
    NO_PROFILE,
    NO_VALUES,
    OWN_KEY,
    PAST_BOUND,
    STATED_DOMAIN,
    Binding,
    Candidate,
    CodeSource,
    StatedDomain,
    binding_assertions,
    decide,
    label_column,
    label_reading,
    loaded_relation,
    place,
    unplaced_assertion,
    unread_key_assertion,
)
from sema.engine.declared_join_utils import bind_declared
from sema.engine.identity_utils import Counts, Relation
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
from sema.engine.warehouse_lookup import (
    RelationEvidence,
    WarehouseProfileLookup,
    WarehouseValues,
)
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
from sema.log import logger
from sema.models.assertions import VERIFIED, Assertion, AssertionPredicate
from sema.models.config import BuildConfig
from sema.models.extraction import ExtractedColumn
from sema.models.physical_key import CanonicalRef
from sema.pipeline.dictionary_import_utils import Dictionary, read_dictionary

_VERIFIED_TERM_CONFIDENCE = 0.95


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
    by_ref: dict[str, ExtractedColumn]
    run_id: str
    schema: str
    joins: list[Assertion] = field(default_factory=list)


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
    by_schema = {schema: fetch_columns_by_schema(loader, schema) for schema in schemas}
    by_ref = {c.ref: c for columns in by_schema.values() for c in columns if c.ref}
    profiles = {
        column_key(c): stats for c in by_ref.values()
        if (stats := profile_lookup(column_key(c))) is not None
    }
    placed = _place(loader, dictionary, by_schema, run_id, _bound(dictionary, by_schema, Counts(
        RelationEvidence(connector, config.catalog))))
    joins: dict[str, list[Assertion]] = {}
    for schema, columns in by_schema.items():
        run = _Run(loader, values, dictionary, columns, by_ref, run_id, schema)
        stated = [(domain, c) for domain, c in placed if c.schema == schema]
        for column, candidates in _candidates(run, profiles, bound, stated).items():
            _bind(run, column, candidates)
        joins[schema] = run.joins
    return joins


def _bound(
    dictionary: Dictionary | None, by_schema: dict[str, list[ExtractedColumn]], counts: Counts,
) -> dict[str, Relation]:
    """The loaded relation each entity a declared join binds is, by the entity's name."""
    if dictionary is None:
        return {}
    columns: dict[Relation, list[str]] = {}
    for column in (c for schema_columns in by_schema.values() for c in schema_columns):
        columns.setdefault((column.schema, column.table_name), []).append(column.name)
    return bind_declared(dictionary.entities, dictionary.declared_joins, columns, counts).bound


def _place(
    loader: GraphLoader, dictionary: Dictionary | None,
    by_schema: dict[str, list[ExtractedColumn]], run_id: str, bound: dict[str, Relation],
) -> list[tuple[StatedDomain, ExtractedColumn]]:
    """Each stated domain on the run's columns, on its entity's bound relation when a declared
    join binds one; one placed nowhere is recorded in each schema."""
    if dictionary is None:
        return []
    every = [c for columns in by_schema.values() for c in columns]
    placed = []
    for domain in dictionary.domains:
        named = place(domain, every, bound.get(domain.entity) if domain.entity else None)
        placed += [(domain, column) for column in named]
        if named:
            continue
        for schema in by_schema:
            loader.store_assertion(
                unplaced_assertion(dictionary.spec_id, domain, run_id, schema),
                source_schema=schema,
            )
    return placed


def _uncoded(run: _Run, profiles: Profiles, column: ExtractedColumn, bound: int) -> str | None:
    """Why a column is no coded column, or None when it is one; codes repeat within the bound."""
    stats = profiles.get(column_key(column))
    if stats is None:
        return NO_PROFILE
    if stats[0] == 0:
        return NO_VALUES
    if stats[0] > bound:
        return PAST_BOUND
    counts = run.values.counts(column_key(column))
    if counts and all(rows == 1 for value, rows in counts.items() if value is not None):
        return OWN_KEY
    return None


def _relation(key: ExtractedColumn) -> CodeSource:
    return CodeSource(key.ref or "", f"{key.schema}.{key.table_name}.{key.name}", key)


def _candidates(
    run: _Run, profiles: Profiles, bound: int,
    stated: list[tuple[StatedDomain, ExtractedColumn]],
) -> dict[str, list[Candidate]]:
    """Each coded column's candidates, keyed by column ref; a domain no column takes is recorded."""
    by_ref = {c.ref: c for c in run.columns if c.ref}
    found: dict[tuple[str, str], Candidate] = {}

    def add(column: ExtractedColumn, source: CodeSource, kind: str | None, **detail: Any) -> None:
        if _uncoded(run, profiles, column, bound) is not None:
            return
        candidate = found.setdefault((column.ref or "", source.ref), Candidate(column, source))
        if kind is not None:
            candidate.evidence[kind] = detail

    for fk in nominate_by_data_shape(run.columns, profiles, bound):
        add(by_ref[fk.fk_ref], _relation(by_ref[fk.pk_ref]), None)
    for fk in nominate_by_name(run.columns, profiles, bound):
        add(by_ref[fk.fk_ref], _relation(by_ref[fk.pk_ref]), NAME_MATCH)
    for column, key, constraint in _declared(run, profiles, bound):
        add(column, _relation(key), DECLARED_KEY, constraint=constraint)
    for domain, column in stated:
        reason = _uncoded(run, profiles, column, bound)
        if reason is not None:
            run.loader.store_assertion(
                unplaced_assertion(column.ref or "", domain, run.run_id, run.schema, reason),
                source_schema=run.schema,
            )
            continue
        source = CodeSource(domain.vocabulary, domain.vocabulary)
        add(column, source, STATED_DOMAIN, property=domain.property, entity=domain.entity)
    by_column: dict[str, list[Candidate]] = {}
    for (column_ref, _), candidate in sorted(found.items()):
        by_column.setdefault(column_ref, []).append(candidate)
    return by_column


def _declared(
    run: _Run, profiles: Profiles, bound: int,
) -> Iterable[tuple[ExtractedColumn, ExtractedColumn, str]]:
    """Single-column declared keys to a code relation within the bound, found by reference.

    A key to a relation the run did not read is recorded for a coded column, never matched.
    """
    for a in fetch_join_assertions(run.loader, run.schema):
        predicates = a.payload.get("join_predicates") or []
        constraint, refs = a.payload.get("constraint_name"), (a.object_ref, a.subject_ref)
        if (a.predicate != AssertionPredicate.JOINS_TO or not constraint
                or len(predicates) != 1 or not all(refs)):
            continue
        (p,) = predicates
        key_ref = CanonicalRef.with_column(a.object_ref or "", p["left_column"])
        column = run.by_ref.get(CanonicalRef.with_column(a.subject_ref, p["right_column"]))
        key = run.by_ref.get(key_ref)
        same_relation = key is not None and column is not None and (
            (key.schema, key.table_name) == (column.schema, column.table_name))
        if column is None or same_relation:
            continue
        if key is None:
            if _uncoded(run, profiles, column, bound) is None:
                run.loader.store_assertion(unread_key_assertion(
                    column.ref or "", key_ref, str(constraint), run.run_id, run.schema,
                ), source_schema=run.schema)
        elif is_code_key(profiles.get(column_key(key)), bound):
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
    relations = [
        (c.source.key, codes) for c, measured in scored
        if c.source.key is not None and measured.holds and (codes := listed[c.source.ref])
    ]
    for binding in decide(scored):
        for assertion in binding_assertions(binding, run.run_id, run.schema):
            run.loader.store_assertion(assertion, source_schema=run.schema)
        if binding.linked:
            codes = listed[binding.candidate.source.ref] or {}
            _link(run, binding, counts, codes, loaded_relation(codes, relations))


def _link(
    run: _Run, binding: Binding, counts: dict[str | None, int], listed: dict[str, str | None],
    loaded: ExtractedColumn | None,
) -> None:
    column, source = binding.candidate.column, binding.candidate.source
    verified = binding.verification == VERIFIED
    labels, label_edge = _labels(run, binding, listed, loaded) if verified else ({}, None)
    observed = sorted(code for code in counts if code is not None and code not in listed)
    # An unverified binding is no governed value set: its numbers only, never its codes.
    members = [
        CodedMember(code, labels.get(code), counts.get(code, 0), code in listed)
        for code in [*listed, *observed]
    ] if verified else []
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
        confidence=_VERIFIED_TERM_CONFIDENCE,
    )
    if label_edge is not None:
        link_label_column(
            run.loader, column_ref=column.ref or "", label_ref=label_edge.label.ref or "",
            edge={"verification": VERIFIED, "violation_rate": label_edge.violation_rate,
                  "via": list(label_edge.via)},
            source_schema=run.schema,
        )


def _labels(
    run: _Run, binding: Binding, listed: dict[str, str | None], loaded: ExtractedColumn | None,
) -> tuple[dict[str, str | None], LabelEdge | None]:
    """A code relation's label column, a dictionary list's `loaded` relation's, the list's own
    labels, else one sibling; a loaded relation wins, as the column a query can join to."""
    column, source = binding.candidate.column, binding.candidate.source
    key, display = source.key, None
    if key is None and loaded is not None:
        roles = run.dictionary.roles if run.dictionary else {}
        key, display = loaded, roles.get(source.vocabulary, ("", None))[1]
    if key is not None:
        join = to_fk_assertion(FKAssertion(fk_candidate(key, column), TIER_1, 1, run.schema),
                               run.run_id)
        chosen = label_column(_readings(run, key), display)
        if source.key is not None or chosen is not None:
            run.joins.append(join)
        if chosen is not None:
            text, (rate, labels) = chosen
            return labels, LabelEdge(text, rate, (_join_name(join),))
        if source.key is not None and _texts(run, key):
            return {}, None
    if source.key is None and any(listed.values()):
        return listed, None
    determined = _readings(run, column)
    if len(determined) != 1:
        return {}, None
    ((sibling, (rate, labels)),) = determined
    return labels, LabelEdge(sibling, rate, ())


def _texts(run: _Run, beside: ExtractedColumn) -> list[ExtractedColumn]:
    return [
        c for c in run.by_ref.values()
        if (c.schema, c.table_name) == (beside.schema, beside.table_name)
        and c.name != beside.name and is_text(c)
    ]


def _readings(
    run: _Run, code: ExtractedColumn,
) -> list[tuple[ExtractedColumn, tuple[float, dict[str, str | None]]]]:
    """Each text column beside `code` that `code` determines, with its label reading."""
    return [
        (text, reading) for text in _texts(run, code)
        if (reading := label_reading(run.values.pairs(
            code.schema, code.table_name, code.name, text.name))) is not None
    ]


def _join_name(join: Assertion) -> str:
    normalized = normalize_fk_to_assertion(join)
    keys = join_keys(normalized)
    if keys is None:
        return ""
    return join_path_name(normalized.payload["join_predicates"], *keys)
