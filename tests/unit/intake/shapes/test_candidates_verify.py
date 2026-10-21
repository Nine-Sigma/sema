"""Property: every binding a template enumerates passes that template's verify, value-free."""

from __future__ import annotations

import pytest

from tests.unit.intake.shapes.shapes_env import IGNORE, TEMPLATE_IDS, Source, candidates, verify

pytestmark = pytest.mark.unit

STEM_GROUPED = ("a2", "acris", "omop")
NAMESPACE_GROUPED = ("a3", "athena", "syn", "v2", "cycle")


def _groups(name: str, source: Source) -> list[list[object]]:
    if name in STEM_GROUPED:
        return source.stem_groups()
    return [list(source.profiles())]


def _disagreements(source: Source, groups: list[list[object]]) -> list[str]:
    found = []
    for template_id in TEMPLATE_IDS:
        scopes = [list(source.profiles())] if template_id == IGNORE else groups
        for group in scopes:
            names = {p.ref.name for p in group}  # type: ignore[attr-defined]
            for bound in candidates(template_id, group, source):
                verdict = verify(template_id, bound, group, source)
                if bound.template != template_id or not set(bound.relations) <= names:
                    found.append(f"{template_id}: malformed {bound}")
                if verdict.status != "pass":
                    found.append(f"{template_id}: {bound.slots} -> {verdict.failures}")
    return found


@pytest.mark.parametrize("name", STEM_GROUPED + NAMESPACE_GROUPED)
def test_every_candidate_passes_verify_without_reading_a_value(
    name: str, request: pytest.FixtureRequest
) -> None:
    source: Source = request.getfixturevalue(name)

    assert _disagreements(source, _groups(name, source)) == []
    assert source.sql_lines()
    assert source.value_bearing() == []
