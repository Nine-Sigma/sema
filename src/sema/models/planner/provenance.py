"""provenance-and-caching capability."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, model_validator


class RunProvenance(BaseModel):
    run_id: str = Field(min_length=1)
    target_model_version: str = Field(min_length=1)
    target_schema_snapshot_hash: str = Field(min_length=1)
    vocab_release: str | None = None
    context_card_version: str = Field(min_length=1)
    prompt_template_version: str = Field(min_length=1)
    few_shot_set_version: str = Field(min_length=1)
    constraint_version: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    embedding_model: str | None = None


class SourceScope(BaseModel):
    source_id: str = Field(min_length=1)
    source_schema_hash: str = Field(min_length=1)
    source_profile_hash: str = Field(min_length=1)


class Provenance(BaseModel):
    run: RunProvenance
    source: SourceScope
    timestamp: datetime


class RunVersionLock:
    """Enforces RunProvenance immutability within a run_id."""

    def __init__(self) -> None:
        self._bound: RunProvenance | None = None

    def bind(self, rp: RunProvenance) -> None:
        if self._bound is None:
            self._bound = rp
            return
        if self._bound.model_dump() != rp.model_dump():
            raise ValueError(
                "RunProvenance fields must remain constant within run_id; "
                "increment run_id to change run-locked context"
            )


class SourceScopeLock:
    """Enforces SourceScope immutability per (run_id, source_id) pair."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._bound: dict[str, SourceScope] = {}

    def bind(self, scope: SourceScope) -> None:
        prior = self._bound.get(scope.source_id)
        if prior is None:
            self._bound[scope.source_id] = scope
            return
        if prior.model_dump() != scope.model_dump():
            raise ValueError(
                f"SourceScope drift for ({self.run_id}, {scope.source_id})"
            )


def _stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptArtifact(BaseModel):
    prefix_text: str
    prefix_hash: str
    suffix_text: str
    versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_hash(self) -> Self:
        expected = _stable_digest(self.prefix_text)
        if self.prefix_hash != expected:
            raise ValueError("prefix_hash must be sha256(prefix_text)")
        return self

    @classmethod
    def build(
        cls,
        prefix_text: str,
        suffix_text: str,
        versions: dict[str, str] | None = None,
    ) -> PromptArtifact:
        return cls(
            prefix_text=prefix_text,
            prefix_hash=_stable_digest(prefix_text),
            suffix_text=suffix_text,
            versions=versions or {},
        )

    def assert_source_isolated(self, source_field_ref: str) -> None:
        """Reject prefixes contaminated by source-similar few-shots.

        Spec 6.6: prefix_text MUST contain only target-Entity-stable few-shots;
        any reference to the per-call source field belongs in suffix_text. If
        the source ref leaks into prefix_text, the prefix_hash diverges per
        call and breaks LLM prompt caching.
        """
        if not source_field_ref:
            raise ValueError("source_field_ref must be non-empty")
        if source_field_ref in self.prefix_text:
            raise ValueError(
                f"prefix_text contains source_field_ref={source_field_ref!r}; "
                "move source-similar content to suffix_text"
            )


_CACHE_KEY_FIELDS = (
    "target_model_version",
    "context_card_version",
    "prompt_template_version",
    "few_shot_set_version",
    "llm_model",
)


def derive_cache_key(
    artifact: PromptArtifact,
    run: RunProvenance,
    source: SourceScope | None = None,  # noqa: ARG001  (intentional: scope is excluded)
) -> str:
    parts = [artifact.prefix_hash]
    parts.extend(getattr(run, name) for name in _CACHE_KEY_FIELDS)
    return _stable_digest("|".join(parts))


def compute_source_profile_hash(signature: dict[str, Any]) -> str:
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return _stable_digest(canonical)


class LLMResponse(BaseModel):
    text: str
    cache_hit: bool = False
    raw_meta: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMRuntime(Protocol):
    name: str
    dialect: str

    def call(self, artifact: PromptArtifact) -> LLMResponse: ...

    def cache_directives(self, artifact: PromptArtifact) -> dict[str, str]: ...


class _AdapterBase:
    dialect: str = ""

    def __init__(self, name: str) -> None:
        self.name = name

    def call(self, artifact: PromptArtifact) -> LLMResponse:
        return LLMResponse(
            text="",
            cache_hit=False,
            raw_meta={
                "adapter": self.dialect,
                "model": self.name,
                "prefix_hash": artifact.prefix_hash,
            },
        )


class AnthropicCachingAdapter(_AdapterBase):
    dialect = "anthropic"

    def cache_directives(self, artifact: PromptArtifact) -> dict[str, str]:
        return {"cache_control": "ephemeral", "prefix_hash": artifact.prefix_hash}


class MosaicAIAdapter(_AdapterBase):
    dialect = "mosaic"

    def cache_directives(self, artifact: PromptArtifact) -> dict[str, str]:
        return {"prefix_hash": artifact.prefix_hash}


class DeepSeekAdapter(_AdapterBase):
    dialect = "deepseek"

    def cache_directives(self, artifact: PromptArtifact) -> dict[str, str]:
        return {"prefix_hash": artifact.prefix_hash}
