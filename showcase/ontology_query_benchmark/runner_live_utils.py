"""Live clients for the runner: the serving model and one executor per arm.

Every host, endpoint, URL and tool is checked against the profile's pinned routing. The
Databricks and Timbr paths are not exercised by unit tests because each opens a network
client; the OpenRouter client takes its HTTP call as an argument, and the connector
executor reads through a `sema` Connector.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sema.cli_factories_utils import OPENROUTER_BASE_URL
from sema.connectors.base import Connector
from showcase.ontology_query_benchmark.profile import Limits, ModelRoute, Profile
from showcase.ontology_query_benchmark.routing import (
    RoutingError,
    require_chat_endpoint,
    require_host,
    require_timbr_mcp_url,
)
from showcase.ontology_query_benchmark.runner_utils import guard_mcp_tools, read_env_keys
from showcase.ontology_query_benchmark.sealed_set_utils import normalize
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from showcase.ontology_query_benchmark.sema_trace import open_live_provider

S_INPUTS = ("neo4j_uri", "s_manifest", "s_projection", "s_projection_receipt")
OPENROUTER = "openrouter"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
MODEL_TIMEOUT_S = 300.0
Post = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def post_json(url: str, body: dict[str, Any], headers: dict[str, str],
              timeout: float) -> dict[str, Any]:  # fmt: skip
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return dict(json.loads(response.read()))
    except urllib.error.HTTPError as exc:
        # The body may echo the prompt, so only the status leaves this function.
        raise RuntimeError(f"model HTTP {exc.code}") from None


class OpenRouterClient:
    """The harness model protocol, `complete(messages) -> Generation`, on OpenRouter."""

    def __init__(self, harness: SimpleNamespace, model: str, api_key: str, *, max_tokens: int,
                 timeout: float, post: Post = post_json) -> None:  # fmt: skip
        self.generation, self.model, self.api_key = harness.models.Generation, model, api_key
        self.max_tokens, self.timeout, self.post = max_tokens, timeout, post

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        start = time.perf_counter()
        payload = self.post(f"{OPENROUTER_BASE_URL}/chat/completions", body,
                            {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
        usage = payload.get("usage") or {}
        content = payload["choices"][0]["message"].get("content")
        return self.generation(content if isinstance(content, str) else "",
                               usage.get("prompt_tokens"), usage.get("completion_tokens"),
                               time.perf_counter() - start)  # fmt: skip


def model_client(harness: SimpleNamespace, route: ModelRoute, limits: Limits,
                 environ: Mapping[str, str]) -> OpenRouterClient:  # fmt: skip
    """The profile's model client; only OpenRouter has one outside Databricks."""
    if route.provider != OPENROUTER:
        raise RoutingError(f"no harness model client for provider {route.provider!r}")
    key = environ.get(OPENROUTER_KEY_ENV, "")
    if not key:
        raise RoutingError(f"{OPENROUTER_KEY_ENV} is not set")
    return OpenRouterClient(harness, route.name, key, max_tokens=limits.output_ceiling,
                            timeout=MODEL_TIMEOUT_S)  # fmt: skip


class ConnectorExecutor:
    """The harness executor protocol on a `sema` Connector, one connector read per statement.

    Rows leave in the expected-answer writer's normal form, so the scorer compares like with
    like. As the Databricks executor, it returns every row; the runner checks the row limit.
    """

    def __init__(self, harness: SimpleNamespace, connector: Connector) -> None:
        self.harness, self.connector = harness, connector

    def execute(self, sql: str, row_limit: int) -> Any:
        start = time.perf_counter()
        try:
            columns, rows = self.connector.execute_query_with_columns(sql)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a classified outcome
            return self.harness.clients.failed_execution(exc, time.perf_counter() - start)
        normal = tuple(tuple(normalize(cell) for cell in row) for row in rows)
        return self.harness.models.Execution(True, tuple(columns), normal,
                                             time.perf_counter() - start)  # fmt: skip


def add_s_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--neo4j-uri", help="arm S: the isolated graph")
    parser.add_argument("--s-manifest", type=Path, help="arm S: the graph build manifest")
    parser.add_argument("--s-projection", type=Path, help="arm S: the library projection")
    parser.add_argument("--s-projection-receipt", type=Path, help="arm S: its receipt")
    parser.add_argument("--s-concept-spec", type=Path, help="arm S (default: the profile's)")


def open_s_provider(
    args: argparse.Namespace, profile: Profile, limits: Limits
) -> AbstractContextManager[SContextProvider]:
    """Arm S's provider from explicit inputs; nothing is opened or called until entry."""
    missing = [f"--{name.replace('_', '-')}" for name in S_INPUTS if getattr(args, name) is None]
    if missing:
        raise ValueError(f"arm S needs {' '.join(missing)}")
    return open_live_provider(
        profile, args.neo4j_uri, args.s_manifest, args.s_projection, args.s_projection_receipt,
        concept_spec=args.s_concept_spec, limits=limits,
    )


def live_clients(harness: SimpleNamespace, arms: list[str], timbr_env: Path | None,
                 limits: Limits, profile: Profile) -> tuple[Any, dict[str, Any]]:  # fmt: skip
    """The pinned serving model and one executor per arm; any routing mismatch raises."""
    from databricks import sql
    from databricks.sdk.core import Config

    routing, timbr = profile.routing, profile.timbr
    config = Config(profile=routing.databricks_profile)
    host, token = require_host(config.host or "", routing), config.token
    if not token:
        raise RoutingError(f"profile {routing.databricks_profile} has no token")
    clients = harness.clients
    model = clients.DatabricksServingClient(
        host, token, require_chat_endpoint(routing.chat_endpoint, routing),
        max_tokens=limits.output_ceiling, temperature=None, timeout=routing.model_timeout_s,
    )  # fmt: skip

    def connect() -> Any:
        return sql.connect(server_hostname=host.removeprefix("https://"),
                           http_path=f"/sql/1.0/warehouses/{routing.sql_warehouse_id}",
                           access_token=token)  # fmt: skip

    executors: dict[str, Any] = {arm: clients.DatabricksSqlExecutor(connect)
                                 for arm in arms if arm in ("A", "S")}  # fmt: skip
    if "B" in arms:
        if timbr_env is None:
            raise ValueError("arm B needs --timbr-env (the ontology engine env file)")
        keys = timbr.env_keys
        env = read_env_keys(timbr_env, (keys.token, keys.ca_bundle, keys.verify_ssl))
        if not env.get(keys.token):
            raise ValueError(f"{keys.token} missing in {timbr_env}")
        verify: bool | str = env.get(keys.ca_bundle) or env.get(keys.verify_ssl) != "false"
        executor = clients.TimbrMcpExecutor(require_timbr_mcp_url(timbr.mcp_url, timbr),
                                            env[keys.token], verify, timbr.ontology,
                                            timeout=routing.model_timeout_s)  # fmt: skip
        executors["B"] = guard_mcp_tools(executor, timbr)
    return model, executors
