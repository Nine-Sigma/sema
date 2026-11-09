"""Live TEST clients for the runner: the serving model and one executor per arm.

Every host, endpoint, URL and tool is checked against the pinned routing; nothing here is
exercised by unit tests because each path opens a network client.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from showcase.partner_timbr_benchmark.routing import (
    SQL_WAREHOUSE_ID,
    TEST_PROFILE,
    TIMBR_MCP_URL,
    TIMBR_ONTOLOGY,
    RoutingError,
    require_chat_endpoint,
    require_test_host,
    require_timbr_mcp_url,
)
from showcase.partner_timbr_benchmark.runner_utils import Limits, guard_mcp_tools, read_env_keys

CHAT_ENDPOINT = "databricks-claude-sonnet-5"
MODEL_TIMEOUT_S = 300.0


def live_clients(harness: SimpleNamespace, arms: list[str], timbr_env: Path | None,
                 limits: Limits) -> tuple[Any, dict[str, Any]]:  # fmt: skip
    """The TEST serving model and one executor per arm; any routing mismatch raises."""
    from databricks import sql
    from databricks.sdk.core import Config

    profile = Config(profile=TEST_PROFILE)
    host, token = require_test_host(profile.host or ""), profile.token
    if not token:
        raise RoutingError(f"profile {TEST_PROFILE} has no token")
    clients = harness.clients
    model = clients.DatabricksServingClient(
        host, token, require_chat_endpoint(CHAT_ENDPOINT), max_tokens=limits.output_ceiling,
        temperature=None, timeout=MODEL_TIMEOUT_S,
    )  # fmt: skip

    def connect() -> Any:
        return sql.connect(server_hostname=host.removeprefix("https://"),
                           http_path=f"/sql/1.0/warehouses/{SQL_WAREHOUSE_ID}",
                           access_token=token)  # fmt: skip

    executors: dict[str, Any] = {arm: clients.DatabricksSqlExecutor(connect)
                                 for arm in arms if arm in ("A", "S")}  # fmt: skip
    if "B" in arms:
        if timbr_env is None:
            raise ValueError("arm B needs --timbr-env (the original POC env file)")
        env = read_env_keys(timbr_env, ("TIMBR_TOKEN", "TIMBR_CA_BUNDLE", "TIMBR_VERIFY_SSL"))
        if not env.get("TIMBR_TOKEN"):
            raise ValueError(f"TIMBR_TOKEN missing in {timbr_env}")
        verify: bool | str = env.get("TIMBR_CA_BUNDLE") or env.get("TIMBR_VERIFY_SSL") != "false"
        executor = clients.TimbrMcpExecutor(require_timbr_mcp_url(TIMBR_MCP_URL),
                                            env["TIMBR_TOKEN"], verify, TIMBR_ONTOLOGY,
                                            timeout=MODEL_TIMEOUT_S)  # fmt: skip
        executors["B"] = guard_mcp_tools(executor)
    return model, executors
