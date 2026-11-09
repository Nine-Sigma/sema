import math
from typing import Any

from toy_harness.benchmark_models import ArmContext

ARM_A_DIALECT = "ansi"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def arm_a_context(schema: Any, relations: Any, policy: Any, budget: int, values: Any) -> ArmContext:
    lines = []
    for relation in relations:
        lines.append(f"TABLE {relation}")
        lines += [f"  {name} {kind}" for name, kind in schema["relations"][relation]["columns"]]
    for relation in relations:
        profiles = values["relations"].get(relation, {}).get("values", {})
        for column, profile in profiles.items():
            if profile["values"]:
                listed = ", ".join(str(value) for value, _ in profile["values"])
                lines.append(f"VALUES {relation}.{column}: {listed}")
    text = "\n".join(lines)
    tokens = estimate_tokens(text)
    return ArmContext("A", ARM_A_DIALECT, text, tokens, budget, tokens > budget)
