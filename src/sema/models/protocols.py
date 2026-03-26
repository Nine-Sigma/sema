from __future__ import annotations

from typing import Any, Protocol, Union, runtime_checkable

JsonScalar = Union[str, int, float, bool, None]
JsonObject = dict[str, "JsonValue"]
JsonArray = list["JsonValue"]
JsonValue = Union[JsonScalar, JsonObject, JsonArray]


@runtime_checkable
class LLMProtocol(Protocol):
    model_name: str

    def invoke(self, prompt: str) -> Any: ...

    def with_structured_output(self, schema: type) -> LLMProtocol: ...
