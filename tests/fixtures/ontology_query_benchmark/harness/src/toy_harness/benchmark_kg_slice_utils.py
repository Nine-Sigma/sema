from typing import Any

from toy_harness.benchmark_context_utils import ARM_A_DIALECT, estimate_tokens
from toy_harness.benchmark_models import ArmContext


def kg_sections(kg: Any, policy: Any) -> tuple[set[str], list[Any]]:
    return {concept["name"] for concept in kg["concepts"]}, []


def arm_b_context(kg: Any, question: str, *, row_concepts: Any, always_concepts: Any,
                  policy: Any, budget: int, description_cap: int) -> ArmContext:  # fmt: skip
    lines: list[str] = []
    for concept in kg["concepts"]:
        line = f"CONCEPT {concept['name']}: {concept['description'][:description_cap]}"
        if estimate_tokens("\n".join([*lines, line])) > budget:
            break
        lines.append(line)
    text = "\n".join(lines)
    return ArmContext("B", ARM_A_DIALECT, text, estimate_tokens(text), budget, False)
