"""CLI for the gold-set snapshot lifecycle and the US-012 mapping report.

``observe`` / ``re-tier`` / ``re-scope`` are separate commands, not flags: they
have different blast radii and only ``observe`` leaves scores comparable. None
overwrites a published snapshot — each emits a new ``--version``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from sema.eval.goldset_drift import goldset_drift_report
from sema.eval.goldset_ops import SnapshotDraft, observe, publish, re_scope, re_tier
from sema.eval.goldset_snapshot import GOLD_ROOT, load_current_snapshot
from sema.eval.goldset_source import (
    SourceKind,
    SourceSpec,
    enumerate_scoped_codes,
    raw_samples_view,
    source_main_types,
)
from sema.eval.mapping_goldset import GoldSet
from sema.eval.goldset_worksheet import (
    SourceContext,
    TargetFacts,
    apply_labels,
    build_worksheet,
    challenge_codes_needing_review,
    instructions_path,
    reference_tissues,
)
from sema.log import logger
from sema.eval.mapping_run import EvaluationSubject, write_eval_run

_VERSION = click.option("--version", required=True, help="New snapshot version (never reused).")
_DATE = click.option("--date", required=True, help="Snapshot date, ISO (YYYY-MM-DD).")
_DB = click.option(
    "--db", default=str(Path.home() / ".sema" / "poc.duckdb"), type=click.Path(exists=True),
    show_default=True,
)
_SHARE = click.option("--target-row-share", default=0.95, show_default=True, type=float)


@click.group("goldset")
def goldset_group() -> None:
    """Publish and inspect frozen gold-set snapshots (US-002)."""


@goldset_group.command("drift")
@_DB
def drift_cmd(db: str) -> None:
    """Report weight drift and benchmark freshness against live data."""
    snapshot = load_current_snapshot(GOLD_ROOT)
    with _connect(db) as con:
        observed = dict(enumerate_scoped_codes(con, snapshot.header.source_of_truth))
    report = goldset_drift_report(snapshot, observed, snapshot.header.source_of_truth)
    click.echo(json.dumps(report.as_dict(), indent=2))


@goldset_group.command("observe")
@_DB
@_VERSION
@_DATE
def observe_cmd(db: str, version: str, date: str) -> None:
    """Re-stamp row counts. Frozen populations do not move."""
    snapshot = load_current_snapshot(GOLD_ROOT)
    with _connect(db) as con:
        observed = dict(enumerate_scoped_codes(con, snapshot.header.source_of_truth))
    _publish(observe(snapshot, observed, version=version, date=date))


@goldset_group.command("re-tier")
@_DB
@_VERSION
@_DATE
@_SHARE
def re_tier_cmd(db: str, version: str, date: str, target_row_share: float) -> None:
    """Recompute the frozen head. A NEW benchmark: scores are not comparable."""
    snapshot = load_current_snapshot(GOLD_ROOT)
    with _connect(db) as con:
        observed = dict(enumerate_scoped_codes(con, snapshot.header.source_of_truth))
    _publish(
        re_tier(
            snapshot, observed, version=version, date=date,
            target_row_share=target_row_share,
            challenge_codes=snapshot.header.challenge_codes,
        )
    )


@goldset_group.command("re-scope")
@_DB
@_VERSION
@_DATE
@_SHARE
@click.option("--kind", type=click.Choice([k.value for k in SourceKind]), required=True)
@click.option("--table", required=True)
@click.option("--code-column", required=True)
@click.option("--scope-column", default=None)
@click.option("--scope-value", "scope_values", multiple=True, required=True)
def re_scope_cmd(
    db: str,
    version: str,
    date: str,
    target_row_share: float,
    kind: str,
    table: str,
    code_column: str,
    scope_column: str | None,
    scope_values: tuple[str, ...],
) -> None:
    """Change the declared source of truth. Retires codes that leave it."""
    snapshot = load_current_snapshot(GOLD_ROOT)
    spec = SourceSpec(
        kind=SourceKind(kind), table=table, code_column=code_column,
        scope_column=scope_column, scope_values=tuple(scope_values),
    )
    with _connect(db) as con:
        observed = dict(enumerate_scoped_codes(con, spec))
    _publish(
        re_scope(
            snapshot, observed, spec, version=version, date=date,
            target_row_share=target_row_share,
            challenge_codes=snapshot.header.challenge_codes,
        )
    )


@goldset_group.command("worksheet")
@_DB
@click.option("--head-size", default=50, show_default=True, type=int)
@click.option("--output", "output_path", required=True, type=click.Path())
def worksheet_cmd(db: str, head_size: int, output_path: str) -> None:
    """Write a blank tier-1 labelling worksheet (source-side context only)."""
    snapshot = load_current_snapshot(GOLD_ROOT)
    with _connect(db) as con:
        names = dict(
            con.execute(
                "SELECT concept_code, concept_name FROM vocabulary_omop.concept "
                "WHERE vocabulary_id = 'OncoTree'"
            ).fetchall()
        )
        # Main type is source-side context, and only the raw sample tables carry
        # it — the staging table the scope is declared against does not.
        main_type_report = source_main_types(
            con, raw_samples_view(snapshot.header.source_of_truth)
        )
    context = SourceContext(
        names=names,
        main_types=main_type_report.main_types,
        tissues=reference_tissues(GOLD_ROOT),
    )
    for scope, reason in main_type_report.failures.items():
        logger.warning("no source main types from {}: {}", scope, reason)
    codes = build_worksheet(snapshot, output_path, head_size=head_size, context=context)
    gaps = "\n".join(
        f"  no {column} for {len(missing)}: {missing}"
        for column, missing in context.missing(codes).items()
        if missing
    )
    unreadable = (
        f"  {len(main_type_report.failures)} scope(s) could not be read for main_type: "
        f"{sorted(main_type_report.failures)} — those blanks are a FAILURE, not a gap\n"
        if main_type_report.failures
        else ""
    )
    click.echo(
        f"wrote {len(codes)} rows to {output_path}\n"
        f"{unreadable}"
        f"  pins and filling rules: {instructions_path(Path(output_path))}\n"
        f"{gaps}\n"
        "  a blank cell means NOT LOOKED UP, not 'no such context exists' — fill the\n"
        "  gaps above from the OncoTree browser before labelling those codes.\n"
        "  no candidate target concepts are pre-filled, and the challenge stratum is\n"
        "  interleaved — both deliberate, so the oracle stays independent."
    )


@goldset_group.command("apply-labels")
@click.option("--worksheet", "worksheet_path", required=True, type=click.Path(exists=True))
@_DB
@_VERSION
@_DATE
def apply_labels_cmd(worksheet_path: str, db: str, version: str, date: str) -> None:
    """Apply a curator's completed worksheet as a NEW snapshot."""
    snapshot = load_current_snapshot(GOLD_ROOT)
    with _connect(db) as con:
        targets = _target_facts(con)
    draft = apply_labels(
        snapshot, worksheet_path, version=version, date=date, targets=targets
    )
    _publish(draft)
    pending = challenge_codes_needing_review(draft.rows, draft.header.challenge_codes)
    if pending:
        click.echo(
            f"  {len(pending)} of {len(draft.header.challenge_codes)} declared challenge "
            f"codes still lack a label and a second reviewer ({', '.join(pending)}); "
            "the verdict will read `unadjudicated`."
        )


def _target_facts(con: Any) -> dict[int, TargetFacts]:
    """What the OMOP vocabulary says about every concept a label could name.

    Read once, passed in as data: the artifact modules that validate a label stay
    pure, so snapshot integrity keeps holding off-machine.
    """
    rows = con.execute(
        "SELECT concept_id, concept_code, vocabulary_id, domain_id, standard_concept "
        "FROM vocabulary_omop.concept WHERE concept_id IS NOT NULL"
    ).fetchall()
    return {
        int(concept_id): TargetFacts(
            concept_code=str(code),
            vocabulary=str(vocabulary),
            domain=str(domain),
            standard=str(standard) == "S",
        )
        for concept_id, code, vocabulary, domain, standard in rows
    }


@click.command("mapping-report")
@click.option("--store", "store_path", required=True, type=click.Path(exists=True))
@click.option("--source-vocabulary", required=True)
@click.option("--target-property-ref", required=True)
@click.option("--resolver-policy-ref", required=True)
@click.option("--vocab-release", required=True)
@click.option("--run-id", required=True, help="Identifies this run; never reused.")
@click.option("--output-dir", default="eval-runs", show_default=True, type=click.Path())
@click.option("--schema", default="sema_resolve", show_default=True)
@click.option("--table", default="value_mapping", show_default=True)
def mapping_report_cmd(
    store_path: str,
    source_vocabulary: str,
    target_property_ref: str,
    resolver_policy_ref: str,
    vocab_release: str,
    run_id: str,
    output_dir: str,
    schema: str,
    table: str,
) -> None:
    """Grade the value-mapping store against the current gold-set snapshot."""
    import duckdb

    from sema.eval.mapping_report import (
        GradingContext,
        GradingReleaseError,
        mappings_for_subject,
        report_for_snapshot,
    )
    from sema.eval.mapping_report_utils import decision_from_value_mapping
    from sema.resolve.value_mapping_store import ValueMappingStore

    subject = EvaluationSubject(
        source_vocabulary=source_vocabulary,
        target_property_ref=target_property_ref,
        resolver_policy_ref=resolver_policy_ref,
        vocab_release=vocab_release,
    )
    snapshot = load_current_snapshot(GOLD_ROOT)
    store = ValueMappingStore(duckdb.connect(store_path), schema=schema, table=table)
    try:
        mappings = mappings_for_subject(store, subject)
    finally:
        store.close()
    decisions = [decision_from_value_mapping(m) for m in mappings]
    try:
        report = report_for_snapshot(
            GradingContext.from_snapshot(snapshot),
            decisions,
            graded_release=subject.vocab_release,
        )
    except GradingReleaseError as exc:
        raise click.ClickException(str(exc)) from exc
    directory = write_eval_run(
        output_dir, run_id=run_id, report=report, subject=subject, decisions=decisions,
        header=snapshot.header,
        resolver_run_ids=sorted({m.run_id for m in mappings}),
    )
    click.echo(report.human_summary())
    click.echo(f"\nrun written to {directory}")


def _connect(db: str) -> Any:
    import duckdb

    return duckdb.connect(db, read_only=True)


def _publish(draft: SnapshotDraft) -> None:
    directory = publish(GOLD_ROOT, draft)
    gold = GoldSet(draft.rows)
    click.echo(
        f"published {draft.header.snapshot_version} to {directory}\n"
        f"  tier {len(draft.header.tier_codes)} codes "
        f"({draft.header.tier_achieved_row_share:.2%} of rows), "
        f"universe {len(draft.universe)} codes\n"
        f"  rows {len(draft.rows)}: {gold.total_eligible_codes} in-tier, "
        f"{len(gold.challenge_codes())} challenge, "
        f"{len(gold.out_of_tier_codes())} out-of-tier, "
        f"{len(gold.retired_codes())} retired\n"
        f"  labelled {gold.labelled_count} — labels are a human gate, never written here"
    )
