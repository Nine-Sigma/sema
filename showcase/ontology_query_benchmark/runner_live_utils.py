"""Live clients for the runner: the serving model and one executor per arm.

Every host, endpoint, URL and tool is checked against the profile's pinned routing;
nothing here is exercised by unit tests because each path opens a network client.
"""

import argparse
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from showcase.ontology_query_benchmark.profile import Limits, Profile
from showcase.ontology_query_benchmark.routing import (
    RoutingError,
    require_chat_endpoint,
    require_host,
    require_timbr_mcp_url,
)
from showcase.ontology_query_benchmark.runner_utils import guard_mcp_tools, read_env_keys
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from showcase.ontology_query_benchmark.sema_trace import open_live_provider

S_INPUTS = ("neo4j_uri", "s_manifest", "s_projection", "s_projection_receipt")


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
