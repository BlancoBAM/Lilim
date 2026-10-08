"""MCP client integration for configured local and remote tool servers.

Servers are user-configured in ``~/.config/lilim/mcp-servers.json``. Local
stdio commands are launched without a shell; remote HTTP servers must use
HTTPS unless they are on loopback. Tool calls are exposed to the agent, with
mutating/unknown tools requiring a user confirmation in the desktop UI.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


CONFIG_PATH = Path(os.environ.get(
    "LILIM_MCP_CONFIG",
    str(Path.home() / ".config" / "lilim" / "mcp-servers.json"),
))
MAX_SERVERS = 8
MAX_TOOL_OUTPUT = 12_000


class MCPConfigError(ValueError):
    """Invalid or unsafe MCP server configuration."""


def _loopback(host: str) -> bool:
    return host.lower() in {"localhost", "127.0.0.1", "::1"}


def validate_server(server: Any) -> dict[str, Any]:
    if not isinstance(server, dict):
        raise MCPConfigError("Each MCP server entry must be an object")
    name = server.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
        raise MCPConfigError("MCP server name must use 1–64 letters, numbers, dots, underscores, or hyphens")
    enabled = server.get("enabled", True)
    if not isinstance(enabled, bool):
        raise MCPConfigError(f"MCP server {name!r}: enabled must be boolean")

    if "url" in server:
        url = server["url"]
        if not isinstance(url, str) or len(url) > 2048:
            raise MCPConfigError(f"MCP server {name!r}: invalid URL")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise MCPConfigError(f"MCP server {name!r}: URL must be an HTTP(S) endpoint without embedded credentials")
        if parts.scheme != "https" and not _loopback(parts.hostname):
            raise MCPConfigError(f"MCP server {name!r}: non-loopback HTTP endpoints must use HTTPS")
        headers = server.get("headers", {})
        if not isinstance(headers, dict) or len(headers) > 64:
            raise MCPConfigError(f"MCP server {name!r}: headers must be an object with at most 64 entries")
        for key, value in headers.items():
            if (not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,80}", key)
                    or not isinstance(value, str) or len(value) > 8192 or "\r" in value or "\n" in value):
                raise MCPConfigError(f"MCP server {name!r}: invalid HTTP header")
        return {"name": name, "enabled": enabled, "url": url, "headers": headers, "transport": "http"}

    command = server.get("command")
    args = server.get("args", [])
    env = server.get("env", {})
    if not isinstance(command, str) or not command.strip() or len(command) > 1024:
        raise MCPConfigError(f"MCP server {name!r}: stdio command is required")
    if not isinstance(args, list) or len(args) > 128 or any(not isinstance(x, str) or len(x) > 4096 for x in args):
        raise MCPConfigError(f"MCP server {name!r}: args must be an array of strings")
    if not isinstance(env, dict) or len(env) > 128:
        raise MCPConfigError(f"MCP server {name!r}: env must be an object")
    for key, value in env.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", key):
            raise MCPConfigError(f"MCP server {name!r}: invalid environment variable name")
        if not isinstance(value, str) or len(value) > 8192:
            raise MCPConfigError(f"MCP server {name!r}: environment values must be strings no longer than 8192 characters")
    return {"name": name, "enabled": enabled, "command": command.strip(), "args": args,
            "env": env, "transport": "stdio"}


def load_config() -> list[dict[str, Any]]:
    if not CONFIG_PATH.exists():
        return []
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"Could not read MCP config: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("servers", []), list):
        raise MCPConfigError("MCP config must contain a servers array")
    servers = [validate_server(item) for item in raw.get("servers", [])]
    if len(servers) > MAX_SERVERS:
        raise MCPConfigError(f"At most {MAX_SERVERS} MCP servers may be configured")
    names = [server["name"] for server in servers]
    if len(names) != len(set(names)):
        raise MCPConfigError("MCP server names must be unique")
    return servers


def save_config(servers: list[dict[str, Any]]) -> None:
    if not isinstance(servers, list) or len(servers) > MAX_SERVERS:
        raise MCPConfigError(f"MCP servers must be an array with at most {MAX_SERVERS} entries")
    normalized = [validate_server(server) for server in servers]
    names = [server["name"] for server in normalized]
    if len(names) != len(set(names)):
        raise MCPConfigError("MCP server names must be unique")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"servers": normalized}, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(CONFIG_PATH)


class MCPManager:
    """Manage persistent MCP sessions and expose safe tool metadata/calls."""

    def __init__(self):
        self._connections: dict[str, tuple[str, AsyncExitStack, Any]] = {}

    async def _session(self, server: dict[str, Any]):
        name = server["name"]
        signature = json.dumps(server, sort_keys=True)
        cached = self._connections.get(name)
        if cached and cached[0] == signature:
            return cached[2]
        if cached:
            await cached[1].aclose()

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise RuntimeError("MCP support is not installed; install the Python dependency 'mcp>=1.19,<2'") from exc

        stack = AsyncExitStack()
        try:
            if server["transport"] == "http":
                transport = await stack.enter_async_context(
                    streamablehttp_client(server["url"], headers=server.get("headers") or None)
                )
            else:
                params = StdioServerParameters(
                    command=server["command"], args=server["args"], env=server.get("env") or None,
                )
                transport = await stack.enter_async_context(stdio_client(params, errlog=sys.stderr))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            await asyncio.wait_for(session.initialize(), timeout=10)
        except Exception:
            await stack.aclose()
            raise
        self._connections[name] = (signature, stack, session)
        return session

    async def list_tools(self) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        try:
            servers = load_config()
        except MCPConfigError as exc:
            return [], [{"server": "configuration", "error": str(exc)}]

        async def fetch_server(server: dict[str, Any]):
            try:
                session = await self._session(server)
                result = await asyncio.wait_for(session.list_tools(), timeout=10)
                tools = []
                for tool in result.tools:
                    annotations = getattr(tool, "annotations", None)
                    read_only = (annotations.get("readOnlyHint", False)
                                 if isinstance(annotations, dict)
                                 else bool(getattr(annotations, "readOnlyHint", False)))
                    tools.append({
                        "server": server["name"], "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": getattr(tool, "inputSchema", {}) or {},
                        "read_only": read_only,
                    })
                return tools, None
            except Exception as exc:
                return [], {"server": server["name"], "error": f"{type(exc).__name__}: {exc}"[:500]}

        async def bounded_fetch(server: dict[str, Any]):
            try:
                return await asyncio.wait_for(fetch_server(server), timeout=18)
            except asyncio.TimeoutError:
                return [], {"server": server["name"], "error": "TimeoutError: MCP discovery exceeded 18 seconds"}

        results = await asyncio.gather(*(
            bounded_fetch(server) for server in servers if server["enabled"]
        ))
        tools: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for result in results:
            found, error = result
            tools.extend(found)
            if error:
                errors.append(error)
        return tools, errors

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server = next((s for s in load_config() if s["name"] == server_name and s["enabled"]), None)
        if not server:
            raise MCPConfigError(f"MCP server {server_name!r} is not enabled")
        session = await self._session(server)
        tools = await asyncio.wait_for(session.list_tools(), timeout=10)
        spec = next((item for item in tools.tools if item.name == tool_name), None)
        if not spec:
            raise MCPConfigError(f"Tool {tool_name!r} is not exposed by MCP server {server_name!r}")
        if not isinstance(arguments, dict):
            raise MCPConfigError("MCP tool arguments must be a JSON object")
        result = await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=90)
        content = []
        for item in getattr(result, "content", []) or []:
            if getattr(item, "text", None) is not None:
                content.append(item.text)
            else:
                content.append(f"[{getattr(item, 'type', 'content')} output omitted]")
        output = "\n".join(content) or json.dumps(getattr(result, "structuredContent", None) or {})
        return {"output": output[:MAX_TOOL_OUTPUT], "is_error": bool(getattr(result, "isError", False))}

    async def close(self) -> None:
        connections, self._connections = self._connections, {}
        for _, stack, _ in connections.values():
            await stack.aclose()


_manager = MCPManager()


def manager() -> MCPManager:
    return _manager
