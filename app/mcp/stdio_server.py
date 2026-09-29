"""Official Python MCP SDK stdio server adapter.

The application intentionally has no handwritten JSON-RPC fallback.  Missing
or incompatible SDK installations fail before stdio protocol traffic starts,
so a host never mistakes a partial implementation for a standards-compliant
MCP server.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class MCPServerDependencyError(RuntimeError):
    """The pinned official MCP SDK is unavailable or incompatible."""

    error_code = "mcp_sdk_missing"


@dataclass(frozen=True)
class _OfficialServerSDK:
    Server: Any
    NotificationOptions: Any
    InitializationOptions: Any
    stdio_server: Any
    Tool: Any


def _load_official_sdk() -> _OfficialServerSDK:
    try:
        from mcp.server import NotificationOptions, Server  # type: ignore
        from mcp.server.models import InitializationOptions  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp.types import Tool  # type: ignore
    except (ImportError, AttributeError) as exc:
        raise MCPServerDependencyError(
            "缺少兼容的官方 MCP Python SDK；请安装项目锁定的 mcp 依赖后重试"
        ) from exc
    return _OfficialServerSDK(
        Server=Server,
        NotificationOptions=NotificationOptions,
        InitializationOptions=InitializationOptions,
        stdio_server=stdio_server,
        Tool=Tool,
    )


class StdioMCPServer:
    """Register existing synchronous handlers on the official MCP server."""

    def __init__(
        self,
        name: str,
        version: str,
        tools: dict[str, tuple[dict[str, Any], ToolHandler]],
    ) -> None:
        self.name = name
        self.version = version
        self.tools = tools
        # Playwright's synchronous API is thread-affine.  A dedicated worker
        # keeps every call for this MCP server on the same non-asyncio thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{name}-mcp-tool")

    async def _invoke_tool(self, handler: ToolHandler, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute blocking tool code on a worker thread."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, handler, arguments)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _run_official(self) -> None:
        sdk = _load_official_sdk()
        server = sdk.Server(self.name)

        @server.list_tools()
        async def list_tools() -> list[Any]:
            return [
                sdk.Tool(
                    name=name,
                    description=str(schema.get("description", "")),
                    inputSchema=dict(schema.get("inputSchema") or {"type": "object"}),
                )
                for name, (schema, _) in self.tools.items()
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
            entry = self.tools.get(name)
            if entry is None:
                value = {
                    "ok": False,
                    "data": None,
                    "error_code": "unknown_tool",
                    "retryable": False,
                    "requires_user": False,
                }
            else:
                try:
                    # Tool implementations include Playwright's synchronous
                    # API and other blocking local adapters.  Run them outside
                    # the MCP event-loop thread; Playwright Sync API rejects
                    # being created or used from an active asyncio loop.
                    value = await self._invoke_tool(entry[1], dict(arguments or {}))
                except Exception:
                    # Tool exceptions are represented by the project's stable
                    # envelope.  Exception strings can contain URLs or secrets
                    # and therefore never cross the protocol boundary.
                    value = {
                        "ok": False,
                        "data": None,
                        "error_code": "tool_exception",
                        "retryable": False,
                        "requires_user": False,
                    }
            # Returning a dictionary lets the official SDK populate both
            # structuredContent and the backwards-compatible JSON text block.
            return value

        options = sdk.InitializationOptions(
            server_name=self.name,
            server_version=self.version,
            capabilities=server.get_capabilities(
                notification_options=sdk.NotificationOptions(),
                experimental_capabilities={},
            ),
        )
        async with sdk.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, options)

    def run(self) -> None:
        # Dependency loading occurs before the official stdio context opens;
        # there is no point at which this can silently become another protocol.
        try:
            _load_official_sdk()
            asyncio.run(self._run_official())
        finally:
            self.close()


__all__ = ["MCPServerDependencyError", "StdioMCPServer", "ToolHandler"]
