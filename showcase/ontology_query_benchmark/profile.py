"""A benchmark source profile: every source-specific value, parsed once and frozen.

Code reads hosts, endpoints, harness layout, limits, knobs, extract relations, S inputs and
pinned source versions only from a loaded profile. Relative paths resolve against the profile file's directory.
"""

from __future__ import annotations

import itertools
import string
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class Attribution(StrEnum):
    """How an input file came to be: discovered by code, supplied, or authored with help."""

    AUTOMATIC = "automatic"
    SUPPLIED = "supplied"
    ASSISTED = "assisted"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _resolved(value: Path, info: ValidationInfo) -> Path:
    base = (info.context or {}).get("base")
    return value if value.is_absolute() or base is None else Path(base) / value


class Routing(_Frozen):
    databricks_profile: str
    host: str
    sql_warehouse_id: str
    chat_endpoint: str
    embedding_endpoint: str
    embedding_dimensions: int
    model_timeout_s: float

    @field_validator("host")
    @classmethod
    def _normalized(cls, host: str) -> str:
        if host != host.lower().rstrip("/") or not host.startswith("https://"):
            raise ValueError(f"host {host!r} must be lowercase https:// without a trailing /")
        return host


class EnvKeys(_Frozen):
    token: str
    ca_bundle: str
    verify_ssl: str


class Timbr(_Frozen):
    url: str
    mcp_path: str
    ontology: str
    allowed_tools: frozenset[str]
    env_keys: EnvKeys

    @property
    def mcp_url(self) -> str:
        return f"{self.url.rstrip('/')}{self.mcp_path}"


class Harness(_Frozen):
    root: Path
    package: str
    config: str
    question_dir: str
    question_files: tuple[str, ...]
    cache_dir: str
    kg_cache: str
    receipts: str
    closure: tuple[str, ...]
    retired_questions: frozenset[str] = frozenset()

    _root = field_validator("root")(_resolved)


class Limits(_Frozen):
    """Frozen per run (manifest.json); input_ceiling and repair_reserve may come from the CLI."""

    input_ceiling: int
    repair_reserve: int
    output_ceiling: int
    guard_chars_per_token: float
    conservative_chars_per_token: float
    row_limit: int
    max_repairs: int

    @property
    def first_prompt_ceiling(self) -> int:
        return self.input_ceiling - self.repair_reserve


class Knobs(_Frozen):
    """Common context knobs; None keeps the harness config value."""

    budget_tokens: int | None = None
    values_top_k: int | None = None
    description_cap_chars: int | None = None


class Sweep(_Frozen):
    budget_tokens: tuple[int, ...]
    values_top_k: tuple[int, ...]
    description_cap_chars: tuple[int, ...]
    input_ceiling: tuple[int, ...]

    def grid(self) -> list[tuple[Knobs, int]]:
        points = itertools.product(self.budget_tokens, self.values_top_k,
                                   self.description_cap_chars, self.input_ceiling)  # fmt: skip
        return [(Knobs(budget_tokens=b, values_top_k=k, description_cap_chars=c), ceiling)
                for b, k, c, ceiling in points]  # fmt: skip


class ExtractRelation(_Frozen):
    relation: str
    key: str
    order_keys: tuple[str, ...]


class ElementRelation(ExtractRelation):
    member_key: str


Column = Field(pattern=r"^\w+$")
QualifiedRelation = Field(pattern=r"^\w+\.\w+\.\w+$")


class MappingRelation(_Frozen):
    """Typed rows taking a source member id to a concept id, then to its replacement."""

    relation: str = QualifiedRelation
    source_column: str = Column
    target_column: str = Column
    type_column: str = Column
    mapping_types: tuple[str, ...] = Field(min_length=1)
    replacement_types: tuple[str, ...] = ()
    target_vocabulary: str = Field(min_length=1)
    target_type: str | None = Field(default=None, pattern=r"^\w+$")


class ClosureRelation(_Frozen):
    """Transitive closure rows: the ancestor column holds the broader concept."""

    relation: str = QualifiedRelation
    ancestor_column: str = Column
    descendant_column: str = Column
    level_column: str = Column
    self_rows: bool


class ConceptRelation(_Frozen):
    """Concept ids and names; with a standard column, only its listed values count."""

    relation: str = QualifiedRelation
    id_column: str = Column
    name_column: str = Column
    standard_column: str | None = Field(default=None, pattern=r"^\w+$")
    standard_values: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _values_need_the_column(self) -> ConceptRelation:
        if bool(self.standard_values) != bool(self.standard_column):
            raise ValueError("standard_column and standard_values go together")
        return self


class ConceptHierarchy(_Frozen):
    """Value-to-standard bridge and ancestor closure for governed coded columns.

    Without ``mapping`` the governed column holds the hierarchy's own concept ids. Every
    relation name holds the ``release`` qualifier, so joins never cross vocabulary
    releases; a relation the source owns is listed in ``release_exempt`` instead.
    """

    attribution: Attribution
    release: str | None = Field(default=None, min_length=1)
    release_exempt: tuple[str, ...] = ()
    mapping: MappingRelation | None = None
    closure: ClosureRelation
    concepts: ConceptRelation

    @property
    def relations(self) -> tuple[str, ...]:
        mapping = (self.mapping.relation,) if self.mapping else ()
        return (*mapping, self.closure.relation, self.concepts.relation)

    @model_validator(mode="after")
    def _one_release(self) -> ConceptHierarchy:
        undeclared = sorted(set(self.release_exempt) - set(self.relations))
        if undeclared:
            raise ValueError(f"release_exempt names undeclared relations: {undeclared}")
        if self.release is not None:
            outside = [r for r in self.relations
                       if self.release not in r and r not in self.release_exempt]  # fmt: skip
            if outside:
                raise ValueError(f"relations outside release {self.release}: {outside}")
        return self


class Extract(_Frozen):
    """The scoped library projection: element rows by declared id, then their members."""

    catalog: str
    schema_name: str
    id_pattern: str
    elements: ElementRelation
    members: ExtractRelation
    row_limit: int
    concept_hierarchy: ConceptHierarchy | None = None

    @property
    def relations(self) -> tuple[str, str]:
        return self.elements.relation, self.members.relation


class Sema(_Frozen):
    sidecar: Path
    concept_spec: Path
    concept_spec_attribution: Attribution

    _paths = field_validator("sidecar", "concept_spec")(_resolved)


class Source(_Frozen):
    """A fully qualified relation the arms read and the version the run is pinned to."""

    relation: str = Field(pattern=r"^\w+\.\w+\.\w+$")
    version: int = Field(ge=0)


class Sources(_Frozen):
    """The pinned relations and the warehouse's own query for a relation's current version."""

    version_sql: str
    version_column: str
    pins: tuple[Source, ...] = Field(min_length=1)

    @field_validator("version_sql")
    @classmethod
    def _templates_the_relation(cls, sql: str) -> str:
        fields = {name for _, name, _, _ in string.Formatter().parse(sql) if name is not None}
        if fields != {"relation"}:
            raise ValueError(f"must template exactly {{relation}}, found {sorted(fields)}")
        return sql


class Profile(_Frozen):
    profile_version: Literal[1]
    name: str
    routing: Routing
    timbr: Timbr
    harness: Harness
    limits: Limits
    knobs: Knobs
    sweep: Sweep
    extract: Extract
    sema: Sema
    sources: Sources

    @property
    def source_versions(self) -> dict[str, int]:
        return {pin.relation: pin.version for pin in self.sources.pins}


def load_profile(path: Path) -> Profile:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Profile.model_validate(raw, context={"base": Path(path).resolve().parent})
