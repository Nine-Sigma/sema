"""Helpers for SemanticEngine: LLM-attempt buffer, prompt hashing."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class LLMAttempt:
    """One logical LLMClient.invoke() call recorded for forensics.

    Populated around each invoke (appended before, mutated after) so both
    successful and failed invocations are captured. Lives on the
    SemanticEngine instance and dies with it.
    """

    stage: str
    batch_index: int | None
    prompt_text: str
    prompt_hash: str
    raw_response: str | None
    parsed_ok: bool


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def stringify_response(response: object) -> str | None:
    if response is None:
        return None
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if hasattr(response, "model_dump_json"):
        try:
            return str(response.model_dump_json())
        except Exception:
            return None
    try:
        return str(response)
    except Exception:
        return None
