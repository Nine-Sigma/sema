"""A template's ``verify`` over a landed relation or over the view a binding names.

The binding names the landed relation; over a view the template reads the view's profile
under the view's name, and its failures are reported against the landed relation again.
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.intake.land.profile import ProfileTools
from sema.intake.structure.shapes.base import RelationProfile, SlotBinding, Template
from sema.intake.structure.shapes.base_utils import failure
from sema.intake.verify.base import Failure
from sema.intake.structure.views import VIEWS
from sema.intake.structure.views.base import ViewRefusedError, ViewSpec

VIEW_KNOWN = "view_known"


def verify_over(
    template: Template,
    binding: SlotBinding,
    view: ViewSpec | None,
    group: Sequence[RelationProfile],
    tools: ProfileTools,
) -> list[Failure]:
    if view is None:
        return list(template.verify(binding, list(group), tools).failures)
    registered = next((v for v in VIEWS if v.ID == view.view), None)
    if registered is None:
        return [failure(VIEW_KNOWN, view.relation, view=view.view)]
    try:
        profile = registered.apply(view, group, tools)
    except ViewRefusedError as refused:
        return list(refused.failures)
    viewed = profile.ref.name
    over = [profile if p.ref.name == view.relation else p for p in group]
    verdict = template.verify(_renamed(binding, view.relation, viewed), over, tools)
    return [_landed(found, viewed, view.relation) for found in verdict.failures]


def _renamed(binding: SlotBinding, landed: str, viewed: str) -> SlotBinding:
    slots = {slot: viewed if value == landed else value for slot, value in binding.slots.items()}
    relations = tuple(viewed if name == landed else name for name in binding.relations)
    return binding.model_copy(update={"slots": slots, "relations": relations})


def _landed(found: Failure, viewed: str, landed: str) -> Failure:
    if found.relation != viewed:
        return found
    return found.model_copy(update={"relation": landed})
