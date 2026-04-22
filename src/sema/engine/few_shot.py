"""Few-shot example registry and prompt composer.

Examples live in per-domain modules (`few_shot_generic.py`,
`few_shot_healthcare.py`, …). The registry maps each domain key to its
Stage A/B/C example arrays. `format_examples()` composes the generic base
layer with the domain-specific overlay so every prompt gets industry-agnostic
archetypes plus whatever domain-specific guidance is available.
"""
from __future__ import annotations

import json
from typing import Any

from sema.engine.few_shot_generic import (
    GENERIC_STAGE_A,
    GENERIC_STAGE_B,
    GENERIC_STAGE_C,
)
from sema.engine.few_shot_healthcare import (
    HEALTHCARE_STAGE_A,
    HEALTHCARE_STAGE_B,
    HEALTHCARE_STAGE_C,
)

_GENERIC_DOMAIN = "generic"

_REGISTRY: dict[str, dict[str, list[dict[str, Any]]]] = {
    _GENERIC_DOMAIN: {
        "A": GENERIC_STAGE_A,
        "B": GENERIC_STAGE_B,
        "C": GENERIC_STAGE_C,
    },
    "healthcare": {
        "A": HEALTHCARE_STAGE_A,
        "B": HEALTHCARE_STAGE_B,
        "C": HEALTHCARE_STAGE_C,
    },
}


def get_examples(
    domain: str | None,
    stage: str,
) -> list[dict[str, Any]]:
    """Look up few-shot examples for a specific domain key only.

    Returns the raw list registered for that domain. Does not compose with
    the generic base — use ``format_examples`` for composed prompt blocks
    or ``compose_examples`` to get the composed list.
    """
    if domain is None:
        return []
    return _REGISTRY.get(domain, {}).get(stage, [])


def compose_examples(
    domain: str | None,
    stage: str,
) -> list[dict[str, Any]]:
    """Compose the generic base layer with a domain-specific overlay.

    Generic archetypal examples come first so the LLM sees cross-industry
    patterns before the domain-specific framing. When ``domain`` is ``None``
    or equals ``"generic"``, only the generic layer is returned.
    """
    generic = _REGISTRY.get(_GENERIC_DOMAIN, {}).get(stage, [])
    if domain is None or domain == _GENERIC_DOMAIN:
        return list(generic)
    specific = _REGISTRY.get(domain, {}).get(stage, [])
    return list(generic) + list(specific)


def format_examples(
    domain: str | None,
    stage: str,
) -> str:
    """Format few-shot examples as a prompt block.

    Returns empty string when no examples are available for any layer
    (generic or domain-specific).
    """
    examples = compose_examples(domain, stage)
    if not examples:
        return ""

    lines = ["Here are examples of correct output:"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Input: {json.dumps(ex['input'], separators=(',', ':'))}")
        lines.append(
            f"Output: {json.dumps(ex['output'], separators=(',', ':'))}",
        )
    return "\n".join(lines)
